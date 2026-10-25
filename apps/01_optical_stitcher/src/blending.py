"""
blending.py — Assembling the final panorama with multi-band blending.

MEMORY OPTIMIZATION:
- Laplacian and Gaussian pyramids are built in TILE SIZE (not canvas!)
- Each tile is processed one at a time, contributions accumulated incrementally
- Expected memory consumption: ~1.5 GB for a 24000x1900 canvas
  (instead of ~10+ GB with canvas-sized pyramids for all tiles at once)

Strategy:
1. Gain compensation — equalize average brightness of each tile to the first
2. Pre-allocate canvas arrays for each pyramid level
3. For each tile:
   a. Build tile-sized Laplacian pyramid (small!)
   b. Build tile-sized Gaussian pyramid for weights
   c. Add the contribution to the canvas at the tile's offset
4. Normalize each level (with anti-zero protection in the coverage zone)
5. Reconstruct the final image from the pyramid
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import preprocessing


@dataclass
class PlacedTile:
    """A tile with a known position on the final canvas."""
    image: np.ndarray          # BGR uint8
    offset_x: int              # position of the top-left corner on the canvas
    offset_y: int
    weight: Optional[np.ndarray] = None  # HxW float32 weights (if None — will be computed)


def compensate_gain(
    tiles_images: List[np.ndarray],
    reference_index: int = 0,
    strength: float = 0.8,
) -> List[np.ndarray]:
    """Equalize average brightness of tiles to the reference tile.

    Solves the problem of varying illumination between shots (e.g. when
    the specimen moves and lighting changes slightly).

    tiles_images: list of BGR uint8 images
    reference_index: index of the tile to equalize to
    strength: 0.0 — no correction, 1.0 — full correction

    Returns: list of BGR float32 images with equalized brightness.
    """
    if not tiles_images:
        return []

    n = len(tiles_images)
    # Average brightness per tile per channel
    means = np.array([img.reshape(-1, img.shape[-1] if img.ndim == 3 else 1).mean(axis=0)
                      for img in tiles_images], dtype=np.float32)
    ref_mean = means[reference_index]

    result = []
    for i, img in enumerate(tiles_images):
        img_f = img.astype(np.float32)
        if i == reference_index or strength <= 0:
            result.append(img_f)
            continue
        # Compute gain so that means[i] * gain ≈ ref_mean
        # gain = ref_mean / means[i], but clip and mix with 1.0 by strength
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(means[i] > 1.0, ref_mean / means[i], 1.0)
        # Clip gain to reasonable bounds
        gain = np.clip(gain, 0.5, 2.0)
        # Mix with unit gain
        gain = 1.0 + (gain - 1.0) * strength
        # Apply
        if img.ndim == 3:
            img_f = img_f * gain[None, None, :]
        else:
            img_f = img_f * float(gain[0])
        result.append(img_f)
    return result


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Laplacian pyramid: levels layers + the last is the Gaussian residual."""
    img = img.astype(np.float32)
    pyramid = []
    current = img
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = current - up
        pyramid.append(lap)
        current = down
    pyramid.append(current)
    return pyramid


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    img = img.astype(np.float32)
    pyramid = [img]
    current = img
    for _ in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from a Laplacian pyramid."""
    current = pyramid[-1]
    for lap in reversed(pyramid[:-1]):
        h, w = lap.shape[:2]
        current = cv2.pyrUp(current, dstsize=(w, h))
        current = current + lap
    return current


def _level_size(canvas_h: int, canvas_w: int, level: int) -> Tuple[int, int]:
    """Canvas size at a given pyramid level (with 2x downsampling per level).

    cv2.pyrDown uses the formula dst = (src + 1) // 2.
    We apply it iteratively to get the size of each level.
    """
    h, w = canvas_h, canvas_w
    for _ in range(level):
        h = (h + 1) // 2
        w = (w + 1) // 2
    return h, w


def _level_offset(offset: int, level: int) -> int:
    """Offset at a given pyramid level.

    With 2x downsampling per level, the offset is divided by 2^level.
    We use integer division — this gives approximate alignment
    with a maximum error of 1 px per level, which is acceptable for multi-band blending.
    """
    return offset // (2 ** level)


def blend_tiles(
    placed: List[PlacedTile],
    canvas_size: Tuple[int, int],  # (height, width)
    levels: int = 5,
    feather: int = 64,
    block_size: int = 16,
    gamma_correction: float = 1.0,
    gain_strength: float = 0.8,
    use_focus_weights: bool = False,
    focus_base_weight: float = 0.3,
    min_total_weight: float = 0.05,
) -> np.ndarray:
    """Multi-band blending of tiles on the final canvas (memory-optimized).

    placed: list of tiles with offset_x, offset_y
    canvas_size: (H, W) of the final image
    levels: number of Laplacian pyramid levels (4-6 typically)
    feather: smooth mask falloff width toward tile edges
    block_size: block size for the sharpness map (focus mode only)
    gamma_correction: if > 1, raises weight contrast (focus mode only)
    gain_strength: 0.0 — no gain compensation, 1.0 — full brightness correction
    use_focus_weights: True — focus-weighted mask (for blurred edges),
        False — simple feather mask (more stable on dark images)
    focus_base_weight: minimum weight in focus mode (anti-zero protection)
    min_total_weight: protection — if total weight in a pixel is below this,
        but the pixel is covered by at least one tile, the division is corrected.

    Returns: BGR uint8 HxWx3.
    """
    canvas_h, canvas_w = canvas_size
    n_tiles = len(placed)
    if n_tiles == 0:
        return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Adjust the number of levels to the canvas size
    min_dim = min(canvas_h, canvas_w)
    max_levels = max(1, int(np.floor(np.log2(max(min_dim, 2)))) - 2)
    levels = max(1, min(levels, max_levels))

    # === Step 0: Gain compensation (visible parts of tiles only) ===
    if gain_strength > 0 and n_tiles > 1:
        compensated = compensate_gain(
            [t.image for t in placed],
            reference_index=0,
            strength=gain_strength,
        )
        compensated_uint8 = [np.clip(img, 0, 255).astype(np.uint8) for img in compensated]
    else:
        compensated_uint8 = [t.image for t in placed]

    # === Step 1: Compute coverage (one canvas-sized array, uint8) ===
    coverage_count = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for tile in placed:
        th, tw = tile.image.shape[:2]
        x0, y0 = tile.offset_x, tile.offset_y
        x1, y1 = min(x0 + tw, canvas_w), min(y0 + th, canvas_h)
        if x1 > x0 and y1 > y0:
            coverage_count[y0:y1, x0:x1] += 1
    has_coverage = coverage_count > 0

    # === Step 2: Pre-allocate canvas arrays for each pyramid level ===
    # canvas_laps[lvl]: (H_lvl, W_lvl, 3) float32 — accumulated weighted sum of Laplacians
    # canvas_weights[lvl]: (H_lvl, W_lvl) float32 — accumulated sum of weights
    canvas_laps: List[np.ndarray] = []
    canvas_weights: List[np.ndarray] = []
    for lvl in range(levels + 1):
        h_l, w_l = _level_size(canvas_h, canvas_w, lvl)
        canvas_laps.append(np.zeros((h_l, w_l, 3), dtype=np.float32))
        canvas_weights.append(np.zeros((h_l, w_l), dtype=np.float32))

    # === Step 3: Process tiles one by one (minimum memory!) ===
    for tile, img_compensated in zip(placed, compensated_uint8):
        th, tw = img_compensated.shape[:2]
        x0, y0 = tile.offset_x, tile.offset_y
        x1, y1 = min(x0 + tw, canvas_w), min(y0 + th, canvas_h)
        actual_w = x1 - x0
        actual_h = y1 - y0
        if actual_w <= 0 or actual_h <= 0:
            continue

        # Take only the visible part of the tile (it falls within the canvas)
        tile_img = img_compensated[:actual_h, :actual_w].astype(np.float32)

        # Weight mask in tile size
        if tile.weight is not None:
            tile_weight = tile.weight[:actual_h, :actual_w].astype(np.float32)
            if use_focus_weights and gamma_correction != 1.0:
                tile_weight = tile_weight ** gamma_correction
        elif use_focus_weights:
            tile_weight = preprocessing.focus_weight_mask(
                tile_img, block_size=block_size, edge_feather=feather,
                base_weight=focus_base_weight, gamma=gamma_correction,
            )
        else:
            tile_weight = preprocessing.feather_mask(tile_img, edge_feather=feather)

        # Build tile-sized pyramids (small! max ~60 MB)
        tile_lap_pyr = _build_laplacian_pyramid(tile_img, levels)
        tile_weight_pyr = _build_gaussian_pyramid(tile_weight[..., None], levels)
        tile_weight_pyr_2d = [w[..., 0] if w.ndim == 3 else w for w in tile_weight_pyr]

        # Place each level on the canvas
        for lvl in range(levels + 1):
            lap_l = tile_lap_pyr[lvl]      # (h_l_tile, w_l_tile, 3)
            w_l = tile_weight_pyr_2d[lvl]  # (h_l_tile, w_l_tile)

            # Offset and size on the canvas for this level
            cx0 = _level_offset(x0, lvl)
            cy0 = _level_offset(y0, lvl)
            ch, cw = lap_l.shape[:2]

            # Guard against going beyond this level's canvas bounds
            canvas_h_l, canvas_w_l = canvas_laps[lvl].shape[:2]
            cy1 = min(cy0 + ch, canvas_h_l)
            cx1 = min(cx0 + cw, canvas_w_l)
            if cy1 <= cy0 or cx1 <= cx0:
                continue

            # Take the corresponding part of the tile level
            ph = cy1 - cy0
            pw = cx1 - cx0
            lap_part = lap_l[:ph, :pw]
            w_part = w_l[:ph, :pw]

            # Accumulate incrementally (+= instead of storing all pyramids)
            canvas_laps[lvl][cy0:cy1, cx0:cx1] += lap_part * w_part[..., None]
            canvas_weights[lvl][cy0:cy1, cx0:cx1] += w_part

        # Free tile pyramid memory (not strictly required, but helps GC)
        del tile_lap_pyr, tile_weight_pyr, tile_weight_pyr_2d

    # === Step 4: Build coverage pyramid and normalize each level ===
    coverage_pyr = _build_gaussian_pyramid(
        has_coverage.astype(np.float32)[..., None], levels
    )
    coverage_pyr_2d = [c[..., 0] if c.ndim == 3 else c for c in coverage_pyr]

    for lvl in range(levels + 1):
        lvl_coverage = coverage_pyr_2d[lvl] > 0.1
        lvl_total = canvas_weights[lvl]
        # Anti-zero protection: in the coverage zone, weight must not be too small
        lvl_total_safe = np.where(
            lvl_coverage & (lvl_total < min_total_weight),
            min_total_weight,
            lvl_total,
        )
        lvl_total_safe = np.where(lvl_total_safe > 1e-6, lvl_total_safe, 1.0)
        canvas_laps[lvl] /= lvl_total_safe[..., None]

    # Free coverage_pyr and canvas_weights (no longer needed)
    del coverage_pyr, coverage_pyr_2d, canvas_weights

    # === Step 5: Reconstruct the final image ===
    final = _reconstruct_from_laplacian(canvas_laps)
    final = np.clip(final, 0, 255).astype(np.uint8)
    return final
