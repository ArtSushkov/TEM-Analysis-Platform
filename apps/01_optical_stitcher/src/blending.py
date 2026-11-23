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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

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
    # Sides that keep FULL weight up to the edge (no feather falloff):
    # (left, right, top, bottom). Used for tiles bordering a "no overlap"
    # black gap so the separator band gets crisp image borders. Ignored
    # when a precomputed `weight` is supplied.
    hard_edges: Tuple[bool, bool, bool, bool] = (False, False, False, False)


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


# --------------------------------------------------------------------------
# Parallel pyramid build (optimization No4)
# --------------------------------------------------------------------------
#
# Step 3 of `blend_tiles` originally processed each tile sequentially:
# compute weight mask, build Laplacian + weight (+ optional Tenengrad)
# pyramids, then accumulate into the canvas arrays. The pyramid BUILDING
# is independent per tile (cv2.pyrDown, cv2.Sobel, cv2.GaussianBlur all
# release the GIL), so running them in parallel threads gives a real
# 2-4x speedup on multi-core CPUs.
#
# The accumulation phase writes to shared canvas arrays and stays
# sequential to avoid locking overhead.
#
# Memory cost: peak parallel allocation = max_workers * per-tile-pyramid-set.
# For a 2736x1824 tile with 5 levels + focus_priority on, one tile's set is
# ~150 MB (Laplacian ~80 MB + feather weight ~27 MB + Tenengrad weight
# ~27 MB + tile_img float32 ~30 MB). With max_workers=4 that is ~600 MB on
# top of the canvas arrays. Capped at min(4, cpu_count) to stay near the
# existing ~1.5 GB peak on the largest expected canvas (24000x1900).


@dataclass
class _TilePyramidSet:
    """Per-tile prebuilt pyramids + placement metadata for accumulation."""
    k_tile: int
    # Canvas-coordinates of the tile's top-left corner at level 0
    # (already adjusted for any left gap-pad).
    x0: int
    y0: int
    # Per-level Laplacian pyramid (levels + 1 entries; last is residual)
    tile_lap_pyr: List[np.ndarray]
    # Per-level feather weight pyramid (2D float32)
    tile_weight_pyr_2d: List[np.ndarray]
    # Per-level Tenengrad weight pyramid (None when focus_priority is off)
    ten_pyr_2d: Optional[List[np.ndarray]]


def _build_tile_pyramid_set(
    k_tile: int,
    tile: PlacedTile,
    img_compensated: np.ndarray,
    gap_pad: Tuple[int, int],
    canvas_h: int,
    canvas_w: int,
    levels: int,
    feather: int,
    block_size: int,
    gamma_correction: float,
    use_focus_weights: bool,
    focus_base_weight: float,
    focus_priority: bool,
    focus_priority_gamma: float,
) -> Optional[_TilePyramidSet]:
    """Build all per-tile pyramids (Laplacian + weight + Tenengrad).

    Returns None when the tile falls entirely outside the canvas (skipped
    by the caller). All cv2 ops here release the GIL, so this function
    is safe to call from multiple worker threads in parallel.
    """
    th, tw = img_compensated.shape[:2]
    # A left pad (gap_pad[0]) prepends columns to the array, so the
    # placement origin moves LEFT by the same amount — otherwise the
    # tile's real content would be shifted right on the canvas.
    x0 = tile.offset_x - gap_pad[0]
    y0 = tile.offset_y
    x1, y1 = min(x0 + tw, canvas_w), min(y0 + th, canvas_h)
    actual_w = x1 - x0
    actual_h = y1 - y0
    if actual_w <= 0 or actual_h <= 0:
        return None

    # Take only the visible part of the tile (it falls within the canvas)
    tile_img = img_compensated[:actual_h, :actual_w].astype(np.float32)

    # Weight mask in tile size
    hl, hr, ht, hb = tile.hard_edges
    if tile.weight is not None:
        # Align a precomputed weight with the gap-padded image: slice to
        # the tile's OWN original size first, then replicate the border
        # columns into the padded zone.
        w_src = tile.weight[:tile.image.shape[0], :tile.image.shape[1]]
        pl, pr = gap_pad
        if pl or pr:
            w_src = np.pad(w_src, ((0, 0), (pl, pr)), mode="edge")
        tile_weight = w_src[:actual_h, :actual_w].astype(np.float32)
        if use_focus_weights and gamma_correction != 1.0:
            tile_weight = tile_weight ** gamma_correction
    elif focus_priority:
        # FOCUS-PRIORITY mode: feather mask is still built (used at low-
        # frequency levels and for the hard-edge gap guard), but at the
        # top `focus_priority_levels` pyramid levels the Tenengrad focus
        # measure overrides the feather weight — see the per-level switch
        # in the level loop below.
        tile_weight = preprocessing.feather_mask(
            tile_img, edge_feather=feather,
            hard_left=hl, hard_right=hr, hard_top=ht, hard_bottom=hb,
        )
    elif use_focus_weights:
        tile_weight = preprocessing.focus_weight_mask(
            tile_img, block_size=block_size, edge_feather=feather,
            base_weight=focus_base_weight, gamma=gamma_correction,
            hard_left=hl, hard_right=hr, hard_top=ht, hard_bottom=hb,
        )
    else:
        tile_weight = preprocessing.feather_mask(
            tile_img, edge_feather=feather,
            hard_left=hl, hard_right=hr, hard_top=ht, hard_bottom=hb,
        )

    # Build tile-sized pyramids (small! max ~60 MB)
    tile_lap_pyr = _build_laplacian_pyramid(tile_img, levels)
    tile_weight_pyr = _build_gaussian_pyramid(tile_weight[..., None], levels)
    tile_weight_pyr_2d = [w[..., 0] if w.ndim == 3 else w for w in tile_weight_pyr]

    # Focus-priority: build a SECOND weight pyramid from the Tenengrad
    # focus measure. Per-level, the high-frequency levels (0 ..
    # focus_priority_levels - 1) use this pyramid; the remaining levels
    # fall back to the feather pyramid above. The Tenengrad map is
    # computed on the (already gain-compensated) tile_img so the score
    # is comparable across tiles — see `tenengrad_map` for the rationale.
    if focus_priority:
        ten_map = preprocessing.tenengrad_map(tile_img, block_size=block_size)
        # Soft winner-takes-all: w = (ten + eps) ** gamma
        # The epsilon is scaled to the tile's own max so it is always
        # relatively small — it just guarantees that a tile whose
        # Tenengrad is locally zero still contributes a tiny bit (avoids
        # div-by-zero in low-contrast zones without disturbing the
        # winner selection where scores are non-trivial).
        ten_max = float(ten_map.max())
        eps = 1e-3 * ten_max if ten_max > 0 else 1e-3
        ten_weight = (ten_map + eps) ** focus_priority_gamma
        # Multiply by the feather mask so the hard-edge gap guard
        # still applies at high-frequency levels: a tile bordering a
        # black gap keeps full weight up to its hard edge, and the
        # Tenengrad score (which is ~0 inside the edge-replicated gap
        # pad) does not let the neighbouring tile "win" the gap zone.
        # Coverage is later forced to black there anyway, so this is
        # belt-and-braces.
        ten_weight = ten_weight * tile_weight
        ten_pyr = _build_gaussian_pyramid(ten_weight[..., None], levels)
        ten_pyr_2d = [w[..., 0] if w.ndim == 3 else w for w in ten_pyr]
    else:
        ten_pyr_2d = None

    # === Canvas-border extension of the coarse levels ======================
    # The canvas level sizes are CEILs ((h+1)//2 per level) while the tile
    # pyramids are placed at FLOOR offsets with their own CEIL widths — for
    # a tile that reaches the canvas right/bottom border the two disagree
    # by 1 (sometimes 2) cells AT THE COARSEST LEVELS. Those cells get
    # ZERO weight and zero content, and the Laplacian reconstruction
    # (pyrUp) spreads the hole into a dark grey gradient ~10-20 px wide
    # along the panorama border ("grey band" artifact). Fix: when the tile
    # actually reaches the border, edge-replicate the last column/row of
    # every level array so the floor-placed content exactly reaches the
    # canvas border at ALL levels. Interior tiles are untouched.
    reach_r = (x0 + img_compensated.shape[1]) >= canvas_w
    reach_b = (y0 + img_compensated.shape[0]) >= canvas_h
    if reach_r or reach_b:
        def _pad_border(arr: np.ndarray, lvl: int) -> np.ndarray:
            """Edge-replicate arr right/bottom up to the canvas level edge."""
            cx0 = x0 // (2 ** lvl)
            cy0 = y0 // (2 ** lvl)
            h_l, w_l = _level_size(canvas_h, canvas_w, lvl)
            need_w = w_l - cx0          # columns from the offset to the edge
            need_h = h_l - cy0
            pads = [
                (0, max(0, need_h - arr.shape[0])) if reach_b else (0, 0),   # rows
                (0, max(0, need_w - arr.shape[1])) if reach_r else (0, 0),   # cols
            ]
            if pads[0][1] or pads[1][1]:
                extra = ((0, 0),) * (arr.ndim - 2)   # channel axis for 3D
                return np.pad(arr, tuple(pads) + extra, mode="edge")
            return arr

        for lvl in range(levels + 1):
            tile_lap_pyr[lvl] = _pad_border(tile_lap_pyr[lvl], lvl)
            tile_weight_pyr_2d[lvl] = _pad_border(tile_weight_pyr_2d[lvl], lvl)
        if ten_pyr_2d is not None:
            for lvl in range(levels + 1):
                ten_pyr_2d[lvl] = _pad_border(ten_pyr_2d[lvl], lvl)

    return _TilePyramidSet(
        k_tile=k_tile,
        x0=x0,
        y0=y0,
        tile_lap_pyr=tile_lap_pyr,
        tile_weight_pyr_2d=tile_weight_pyr_2d,
        ten_pyr_2d=ten_pyr_2d,
    )


def blend_tiles(
    placed: List[PlacedTile],
    canvas_size: Tuple[int, int],  # (height, width)
    levels: int = 3,
    feather: int = 64,
    block_size: int = 16,
    gamma_correction: float = 1.0,
    gain_strength: float = 0.8,
    use_focus_weights: bool = False,
    focus_base_weight: float = 0.3,
    min_total_weight: float = 0.05,
    focus_priority: bool = False,
    focus_priority_gamma: float = 6.0,
    focus_priority_levels: int = 2,
    repair_stats: Optional[dict] = None,
) -> np.ndarray:
    """Multi-band blending of tiles on the final canvas (memory-optimized).

    placed: list of tiles with offset_x, offset_y
    canvas_size: (H, W) of the final image
    levels: number of Laplacian pyramid levels (4-6 typically)
    feather: smooth mask falloff width toward tile edges
    block_size: block size for the sharpness map (focus modes only)
    gamma_correction: if > 1, raises weight contrast (legacy focus-weighted
        mode only)
    gain_strength: 0.0 — no gain compensation, 1.0 — full brightness correction
    use_focus_weights: True — legacy per-tile focus-weighted mask. Off by
        default because per-tile normalization makes the metric incomparable
        across tiles.
    focus_base_weight: minimum weight in legacy focus mode (anti-zero).
    min_total_weight: protection — if total weight in a pixel is below this,
        but the pixel is covered by at least one tile, the division is corrected.

    focus_priority: True — FOCUS-PRIORITY BLENDING (winner-takes-all at high
        frequencies). In the overlap zone, the tile whose Tenengrad focus
        measure is highest at a given pixel wins the high-frequency content
        of the panorama; low frequencies are still averaged with a feather
        mask so brightness/illumination transitions stay smooth and small
        residual alignment errors are NOT revealed as visible steps.
        Designed for the use case where the operator focuses each frame on
        a different depth and the goal is to keep the sharpest visible
        detail at every point of the cross-section, not to produce a pretty
        averaged picture.
        When True, `use_focus_weights` and `gamma_correction` are ignored.
    focus_priority_gamma: exponent applied to the Tenengrad score before
        using it as a weight. Higher = stronger winner-takes-all. The blend
        weight at a high-frequency level becomes:
            w_k = (Tenengrad_k + epsilon) ** gamma
        With gamma=6 the soft-max behavior is sharp enough that the
        highest-scoring tile contributes ~95% of the high-frequency content
        when its score is ~2x the runner-up, while still degrading
        gracefully (no hard seam) when scores are close.
    focus_priority_levels: number of TOP pyramid levels (levels 0 ..
        focus_priority_levels - 1) that use the Tenengrad winner weight.
        Lower-indexed Laplacian levels carry the highest frequencies
        (edges, fine detail); higher-indexed levels carry coarse structure
        and the residual. Default 2 = the two highest-frequency bands are
        chosen by focus, coarser bands are averaged. Increase to push the
        focus selection into coarser structure too (risk: reveals residual
        misalignment as visible steps).

    repair_stats: optional dict — when supplied it is filled with the
        post-blend artifact-repair counters ("gap_px_filled",
        "outlier_px_repaired", "wide_gap_px_left") so the caller can
        report what had to be fixed (see "broken and grey pixels" fix).

    Pixels not covered by any tile (zero coverage — e.g. the black gap of a
    "no overlap" pair or ragged canvas corners) are returned as pure black.

    Gap-bleed guard: tiles bordering an uncovered gap keep FULL weight on
    that side (hard_edges) AND are edge-extended into the gap before the
    pyramid accumulation (see Step 0b). Without the extension, the Laplacian
    pyramid reconstruction (cv2.pyrUp) pulls zeros from the uncovered gap
    into neighbouring tile pixels at every level, producing a dark gradient
    that eats ~10-20 px into the tiles on both sides of the gap.

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

    # === Step 0b: edge-extend hard-edge tiles into the adjacent gap ===
    # cv2.pyrUp interpolates each pyramid level from its neighbours. An
    # interior UNCOVERED region (the black gap of a "no overlap" pair)
    # therefore leaks zeros into adjacent tile pixels at every pyramid
    # level; the errors accumulate level by level into a visible dark
    # gradient (~10-20 px) on both sides of the gap.
    # Fix: replicate the tile's border columns into the adjacent gap on the
    # hard-edge sides, so pyramid content is continuous there. Coverage
    # (Step 1) is computed from the ORIGINAL tile rectangles, and the final
    # step forces all uncovered pixels back to pure black — the padded
    # content never reaches the output.
    MAX_GAP_PAD = 4096  # sanity cap for pathologically large manual gaps
    images_blend: List[np.ndarray] = list(compensated_uint8)
    gap_pads: List[Tuple[int, int]] = []      # (pad_left, pad_right) per tile
    for k, tile in enumerate(placed):
        hl, hr, _, _ = tile.hard_edges
        # NOTE: pad sizes are computed from the ORIGINAL tile widths
        # (compensated_uint8), so BOTH tiles of a gap extend across the full
        # gap independently — the second tile must not "lose" its pad just
        # because the first tile's pad already covers the gap.
        pad_l = pad_r = 0
        if hr and k + 1 < len(placed):
            pad_r = max(0, min(
                placed[k + 1].offset_x
                - (tile.offset_x + compensated_uint8[k].shape[1]),
                MAX_GAP_PAD))
        if hl and k >= 1:
            pad_l = max(0, min(
                tile.offset_x - (placed[k - 1].offset_x
                                 + compensated_uint8[k - 1].shape[1]),
                MAX_GAP_PAD))
        gap_pads.append((pad_l, pad_r))
        if pad_l or pad_r:
            images_blend[k] = np.pad(
                images_blend[k], ((0, 0), (pad_l, pad_r), (0, 0)), mode="edge")

    # === Step 0c: canvas-border sides keep FULL weight ======================
    # The feather mask ramps the weight to 0 at tile edges. Where two tiles
    # MEET that fade is the blend ramp — but on a side that lies ON the
    # panorama border (no tile beyond it, e.g. the top/bottom edges of a
    # thin strip panorama) the fade darkens REAL image content: after the
    # weight normalization the outermost pixels of the panorama are
    # attenuated and the border looks like a grey gradient band. Mark every
    # canvas-border side "hard" (full weight up to the edge) — nothing has
    # to be blended there.
    for k, tile in enumerate(placed):
        hl, hr, ht, hb = tile.hard_edges
        x0 = tile.offset_x - gap_pads[k][0]
        tw = compensated_uint8[k].shape[1]
        th = compensated_uint8[k].shape[0]
        hl = hl or (x0 <= 0)
        hr = hr or (tile.offset_x + tw >= canvas_w)
        ht = ht or (tile.offset_y <= 0)
        hb = hb or (tile.offset_y + th >= canvas_h)
        if (hl, hr, ht, hb) != tile.hard_edges:
            placed[k] = replace(tile, hard_edges=(hl, hr, ht, hb))

    # === Step 1: Compute coverage (one canvas-sized array, uint8) ===
    # NOTE: computed from the ORIGINAL tile rectangles (tile.image), NOT the
    # gap-padded copies above — the padded strip must never count as
    # covered, otherwise the black gap would be filled with replicated
    # image content.
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

    # === Step 3: Build per-tile pyramids in parallel, then accumulate ===
    #
    # Phase A (parallel): for each tile, compute weight mask + build the
    # Laplacian pyramid + build the feather-weight pyramid (+ optional
    # Tenengrad-weight pyramid). These cv2 operations all release the GIL,
    # so running them across multiple worker threads gives a real 2-4x
    # speedup on multi-core CPUs. Memory cost: peak parallel allocation
    # = max_workers * per-tile-pyramid-set, capped via min(4, cpu_count).
    #
    # Phase B (sequential): iterate the built pyramid sets and accumulate
    # into the canvas arrays. This writes to shared memory and stays
    # sequential to avoid locking overhead (the accumulation itself is
    # cheap compared to the pyramid build).
    max_workers = max(1, min(4, os.cpu_count() or 1))
    pyramid_sets: List[Optional[_TilePyramidSet]] = [None] * n_tiles
    if max_workers == 1 or n_tiles <= 1:
        # Single-threaded path — no thread-pool overhead, useful for tests
        # and tiny rows.
        for k_tile, (tile, img_compensated) in enumerate(zip(placed, images_blend)):
            pyramid_sets[k_tile] = _build_tile_pyramid_set(
                k_tile, tile, img_compensated, gap_pads[k_tile],
                canvas_h, canvas_w, levels, feather, block_size,
                gamma_correction, use_focus_weights, focus_base_weight,
                focus_priority, focus_priority_gamma,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _build_tile_pyramid_set,
                    k_tile, tile, img_compensated, gap_pads[k_tile],
                    canvas_h, canvas_w, levels, feather, block_size,
                    gamma_correction, use_focus_weights, focus_base_weight,
                    focus_priority, focus_priority_gamma,
                ): k_tile
                for k_tile, (tile, img_compensated)
                in enumerate(zip(placed, images_blend))
            }
            for fut in futures:
                k_tile = futures[fut]
                pyramid_sets[k_tile] = fut.result()

    # Phase B: sequential accumulation into the canvas arrays.
    for ps in pyramid_sets:
        if ps is None:
            continue
        # Place each level on the canvas
        for lvl in range(levels + 1):
            lap_l = ps.tile_lap_pyr[lvl]      # (h_l_tile, w_l_tile, 3)
            if focus_priority and lvl < focus_priority_levels:
                w_l = ps.ten_pyr_2d[lvl]      # high-freq: focus winner weight
            else:
                w_l = ps.tile_weight_pyr_2d[lvl]  # low-freq / fallback: feather

            # Offset and size on the canvas for this level
            cx0 = _level_offset(ps.x0, lvl)
            cy0 = _level_offset(ps.y0, lvl)
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

    # Free pyramid sets (not strictly required, but helps GC)
    del pyramid_sets

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
    # Pixels outside every tile are pure black: this removes the pyramid
    # upsampling bleed into the black gap of "no overlap" pairs (and into
    # ragged canvas corners), so separator bands have crisp borders.
    final[~has_coverage] = 0

    # === Step 6: artifact repair (fix of "broken and grey pixels") ========
    # Two residual artifact classes survive the blend itself:
    #   1. thin UNCOVERED slivers between adjacent tiles (integer rounding
    #      of cumulative offsets, a pair placed 1-3 px short) — rendered
    #      black/grey vertical streaks;
    #   2. isolated broken pixels (uint8 clipping of float rounding at
    #      pyramid seams).
    # Both are repaired deterministically; the counters are reported to
    # the caller (warnings in the UI), so nothing is hidden from the
    # operator.
    stats = _repair_panorama_artifacts(final, has_coverage)
    if repair_stats is not None:
        repair_stats.update(stats)
    return final


def _repair_panorama_artifacts(final: np.ndarray,
                               has_coverage: np.ndarray,
                               max_gap_px: int = 32,
                               outlier_thresh: int = 60) -> dict:
    """Repair residual blending artifacts IN PLACE; return counters.

    1. Narrow uncovered runs (<= ``max_gap_px`` wide) that have covered
       pixels on BOTH sides in the same row are filled with a horizontal
       linear interpolation between those neighbours — the classic 1-3 px
       seam slivers from integer position rounding disappear; wider runs
       (real no-overlap gaps, ragged corners) are left untouched.
    2. Isolated outlier pixels inside the covered area (value deviating
       from the 3x3 median by more than ``outlier_thresh`` in ANY channel)
       are replaced by that median — salt/pepper-style specks from the
       pyramid reconstruction vanish; real texture (grain boundaries,
       dust) is untouched because it is never an isolated 1-px spike.
    """
    H, W = has_coverage.shape[:2]
    filled = 0
    nc = ~has_coverage
    if nc.any():
        int8v = nc.astype(np.int8)
        d = np.diff(np.concatenate([np.zeros((H, 1), np.int8), int8v,
                                    np.zeros((H, 1), np.int8)], axis=1))
        for r in range(H):
            row = int8v[r]
            if not row.any():
                continue
            starts = np.where(d[r] == 1)[0]
            ends = np.where(d[r] == -1)[0]
            for s, e in zip(starts, ends):
                if s <= 0 or e >= W:          # touches the row border
                    continue
                if e - s > max_gap_px:        # a real gap, not a sliver
                    continue
                left = final[r, s - 1].astype(np.float32)
                right = final[r, e].astype(np.float32)
                n = e - s
                t = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))
                final[r, s:e] = (
                    left[None, :] * (1.0 - t[:, None])
                    + right[None, :] * t[:, None]).astype(np.uint8)
                filled += int(n)
    repaired = 0
    if has_coverage.any() and outlier_thresh > 0:
        med = cv2.medianBlur(final, 3)
        diff = np.abs(final.astype(np.int16) - med.astype(np.int16)).max(axis=2)
        bad = (diff > outlier_thresh) & has_coverage
        repaired = int(bad.sum())
        if repaired:
            final[bad] = med[bad]
    return {"gap_px_filled": filled, "outlier_px_repaired": repaired}
