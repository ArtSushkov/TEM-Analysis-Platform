"""
blending.py — Assembling the final panorama with multi-band blending.

MEMORY OPTIMIZATION:
- Laplacian and Gaussian pyramids are built in TILE SIZE (not canvas!)
- Each tile is processed one at a time, contributions accumulated incrementally
- Expected memory consumption: ~1.5 GB for a 24000x1900 canvas
  (instead of ~10+ GB with canvas-sized pyramids for all tiles at once)
- SLABBED LOW-RAM MODE (operator report: "Unable to allocate 595 MiB for an
  array with shape (1568, 99496) float32"): for wide panoramas the float32
  pyramid CANVASES alone are several GB and the reconstruction transient is
  even larger. blend_tiles() therefore AUTO-SWITCHES to a slabbed blend:
  the canvas is processed in narrow column slabs, every per-slab array is
  small, and the result is written into the final uint8 panorama slab by
  slab. The float32 canvas arrays are NEVER allocated at full canvas size.
  The per-slab math replicates the global blend exactly (slab pitch/margin
  are multiples of the coarsest pyramid cell, margin absorbs the pyramid
  reconstruction support), so the output is identical to the standard blend
  (bit-exact for the feather weight mode; the only known deviation is the
  legacy per-tile focus normalization, which becomes per-slice — that mode
  is not the default and the difference is confined to overlap zones).

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
from typing import Callable, Dict, List, Optional, Tuple

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

    Returns: list of BGR **uint8** images with equalized brightness.

    MEMORY NOTE (operator report: "Unable to allocate 22.7 MiB" during
    the blend of a large series): the previous version returned float32
    copies of EVERY tile (~36 MB per 3 MP tile) while blend_tiles held
    ANOTHER uint8 copy of the same tiles — together ~45 MB per tile of
    avoidable peak RAM. The gain is now applied per tile in a transient
    float32 buffer that is clipped back to uint8 IMMEDIATELY, so at any
    moment only ONE tile-sized float32 buffer exists. The math is
    unchanged: blend_tiles clipped the very same float32 values to uint8
    right after this call, so the output pixels are identical.
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
        if i == reference_index or strength <= 0:
            # The reference tile is returned UNCHANGED (shared object —
            # nothing downstream mutates these entries in place; they are
            # only read and padded into fresh arrays).
            result.append(img)
            continue
        # Compute gain so that means[i] * gain ≈ ref_mean
        # gain = ref_mean / means[i], but clip and mix with 1.0 by strength
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(means[i] > 1.0, ref_mean / means[i], 1.0)
        # Clip gain to reasonable bounds
        gain = np.clip(gain, 0.5, 2.0)
        # Mix with unit gain
        gain = 1.0 + (gain - 1.0) * strength
        # Apply in a transient float32 buffer, then clip back to uint8
        # immediately — the float32 copy of the tile is released on the
        # very next loop iteration instead of living for the whole blend.
        img_f = img.astype(np.float32)
        if img.ndim == 3:
            img_f = img_f * gain[None, None, :]
        else:
            img_f = img_f * float(gain[0])
        result.append(np.clip(img_f, 0, 255).astype(np.uint8))
        del img_f
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
    progress_callback: Optional[Callable[[str, float], None]] = None,
    slab_width: Optional[int] = None,
    force_slabbed: Optional[bool] = None,
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
        When the automatic LOW-RAM slabbed path runs, it also receives
        "mode" ("slabbed") and "slabs" (number of column slabs).

    progress_callback: optional (message, progress_0_to_1) — reports the
        slab-by-slab assembly progress of the low-RAM path.

    slab_width: optional explicit slab width in canvas pixels for the
        low-RAM path (None = automatic from the canvas height).

    force_slabbed: None (default) — auto-detect: the slabbed path runs
        when the estimated float32 canvas arrays exceed
        _SLAB_AUTO_THRESHOLD_BYTES; False — never slab; True — always
        slab (used by tests).

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

    # === Automatic LOW-RAM (slabbed) path ===================================
    # The standard path allocates the FULL-canvas float32 pyramid arrays
    # (3 channels + weights at level 0 alone: canvas_w * canvas_h * 16 B)
    # plus a canvas-sized float32 coverage pyramid and a reconstruction
    # transient of the same order — several GB for wide panoramas. When the
    # estimated canvas arrays exceed the threshold, assemble the panorama
    # in narrow column slabs instead (see _blend_tiles_slabbed): identical
    # output, a fraction of the RAM.
    _est_canvas_bytes = _canvas_arrays_bytes(canvas_h, canvas_w, levels)
    _use_slabs = (
        force_slabbed
        if force_slabbed is not None
        else (_est_canvas_bytes > _SLAB_AUTO_THRESHOLD_BYTES)
    )
    if _use_slabs:
        return _blend_tiles_slabbed(
            placed=placed,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            levels=levels,
            feather=feather,
            block_size=block_size,
            gamma_correction=gamma_correction,
            gain_strength=gain_strength,
            use_focus_weights=use_focus_weights,
            focus_base_weight=focus_base_weight,
            min_total_weight=min_total_weight,
            focus_priority=focus_priority,
            focus_priority_gamma=focus_priority_gamma,
            focus_priority_levels=focus_priority_levels,
            repair_stats=repair_stats,
            progress_callback=progress_callback,
            slab_width=slab_width,
        )

    # === Step 0: Gain compensation (visible parts of tiles only) ===
    # compensate_gain returns uint8 tiles now (the gain is applied in a
    # per-tile transient float32 buffer and clipped back immediately),
    # so the old "float32 list + uint8 copy" double storage — about
    # 45 MB per 3 MP tile of avoidable peak RAM — is gone.
    if gain_strength > 0 and n_tiles > 1:
        compensated_uint8 = compensate_gain(
            [t.image for t in placed],
            reference_index=0,
            strength=gain_strength,
        )
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
    # Memory guard (operator report: MemoryError while allocating the
    # panorama-size float32 weight map): ONE per-tile pyramid set of a
    # large tile is ~100-150 MB, and the parallel phase holds
    # max_workers of them at once. When the tiles are large (the ROI
    # band of a 2736x1824 tile is ~3 MP), cap the pool at 2 workers —
    # the peak parallel allocation drops by half while the build phase
    # stays ~2x faster than sequential.
    _biggest_tile_px = max(
        (int(t.image.shape[0]) * int(t.image.shape[1]) for t in placed),
        default=0)
    if _biggest_tile_px > 2_000_000:
        max_workers = min(max_workers, 2)
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


# ==========================================================================
# LOW-RAM SLABBED BLEND (operator report: "Stitching ran out of RAM:
# Unable to allocate 595 MiB for an array with shape (1568, 99496) float32")
# ==========================================================================
#
# Why the standard path dies on wide panoramas: the float32 pyramid
# canvases are allocated for the WHOLE panorama at once —
#   level 0 alone: canvas_w * canvas_h * (3 channels + 1 weight) * 4 B,
# for (1568, 99496) that is 2.4 GB at level 0, ~3.2 GB total, plus a
# canvas-sized float32 coverage pyramid (~0.8 GB) and the Laplacian
# reconstruction transient (~2x the level-0 arrays). None of these are
# needed at full size AT THE SAME TIME as everything else — and none of
# them need the full WIDTH at all, because a column slab of the panorama
# only ever receives contributions from the tiles that overlap it.
#
# This path assembles the panorama in column slabs:
#   for every slab [sx0, sx1):
#     - take the window [sx0 - margin, sx1 + margin) clipped to the canvas
#     - for each tile overlapping the window, slice ONLY the window part
#       (plus gap-pad edge replication), build its tile-sized pyramids,
#       accumulate into window-sized canvas arrays
#     - normalize, reconstruct, and paste the core columns into the final
#       uint8 panorama
# The margin (a multiple of the coarsest pyramid cell) makes the per-slab
# result EXACTLY equal to the global blend inside the core: every per-pixel
# value the core sees (pyramid content, weights, coverage, normalization)
# is identical to the global path; pyramid border effects decay within the
# margin and are discarded. Slab pitch and margin are multiples of
# 2**levels, so the level-l cell grid of a window is the global grid
# shifted — the accumulated sums per cell match the global blend cell for
# cell. Bit-exact for the feather weight mode; the legacy per-tile focus
# normalization (non-default mode) becomes per-slice, a benign difference
# confined to overlap zones near slab cuts.

_SLAB_AUTO_THRESHOLD_BYTES = 400 * 1024 * 1024
# Standard-path float32 canvas arrays above this → slabbed blend. At 400 MB
# of canvas arrays the standard path also allocates a float32 coverage
# pyramid of the same order plus reconstruction transients ~2x level 0, so
# its true peak is roughly 3-4x this number.
_SLAB_TARGET_BYTES = 48 * 1024 * 1024
# Per-slab level-0 accumulator target (laps 3ch + weights, float32): keeps
# the per-slab working set (arrays + reconstruction transient + one sliced
# tile pyramid set) around 250-350 MB regardless of canvas height.
_SLAB_MARGIN_CELLS = 24
# Window margin in COARSEST-level cells (2**levels px each). Two effects
# must decay inside the margin: (a) the pyramid border effects of the tile
# slices (a few px at L0, up to ~1 cell per pyramid level at the coarsest)
# and (b) the Laplacian RECONSTRUCTION (pyrUp chain), which diffuses the
# contaminated window-edge cells of the coarse levels ~4-6 cells inward.
# 24 cells is 3-4x the observed reach (empirically bit-exact on noise
# images, the worst case).


def _canvas_arrays_bytes(canvas_h: int, canvas_w: int, levels: int) -> int:
    """Total bytes of the standard path's float32 canvas arrays."""
    total = 0
    for lvl in range(levels + 1):
        h_l, w_l = _level_size(canvas_h, canvas_w, lvl)
        total += h_l * w_l * 4 * (3 + 1)     # laps (3 ch) + weights
    return total


def _gain_factors(
    tiles_images: List[np.ndarray],
    strength: float,
    reference_index: int = 0,
) -> Optional[List[Optional[np.ndarray]]]:
    """Per-tile per-channel gain factors (compensate_gain math, no copies).

    Replicates compensate_gain() exactly: means as float32, gain clipped to
    [0.5, 2.0], mixed with 1.0 by `strength`. The reference tile gets None
    (applied unchanged). The gain is then applied per SLICE by the slabbed
    blend — a per-pixel constant multiplier, so slice-wise application
    produces bit-identical results to tile-wise application.
    """
    n = len(tiles_images)
    if n == 0 or strength <= 0 or n == 1:
        return None
    means = np.array(
        [img.reshape(-1, img.shape[-1] if img.ndim == 3 else 1).mean(axis=0)
         for img in tiles_images], dtype=np.float32)
    ref_mean = means[reference_index]
    factors: List[Optional[np.ndarray]] = []
    for i in range(n):
        if i == reference_index:
            factors.append(None)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(means[i] > 1.0, ref_mean / means[i], 1.0)
        gain = np.clip(gain, 0.5, 2.0)
        gain = 1.0 + (gain - 1.0) * strength
        factors.append(gain.astype(np.float32))
    return factors


def _apply_gain_slice(img: np.ndarray, gain: Optional[np.ndarray]) -> np.ndarray:
    """Apply the per-channel gain to a slice, clip back to uint8."""
    if gain is None:
        return img
    img_f = img.astype(np.float32)
    if img.ndim == 3:
        img_f = img_f * gain[None, None, :]
    else:
        img_f = img_f * float(gain[0])
    return np.clip(img_f, 0, 255).astype(np.uint8)


def _slice_padded_columns(
    arr: np.ndarray,
    pad_l: int,
    pad_r: int,
    lo: int,
    hi: int,
) -> np.ndarray:
    """Columns [lo, hi) of the EDGE-PADDED array without materializing it.

    The global blend pads hard-edge tiles into the adjacent gap with
    np.pad(..., mode="edge"). For the slabbed blend the padded image is
    never built at full tile width: this returns exactly the same values
    for a window of padded columns — the interior part comes from `arr`,
    the pad zones are edge-replications of the array's first/last column.
    A window that lies ENTIRELY inside a pad zone (no real tile content)
    is built by replicating the corresponding border column.
    """
    tw = arr.shape[1]
    olo = min(max(lo - pad_l, 0), tw)
    ohi = min(max(hi - pad_l, 0), tw)
    if olo == ohi:
        # window lies entirely inside a gap-pad zone
        edge = 0 if hi <= pad_l else tw - 1
        col = arr[:, edge]
        width = max(0, hi - lo)
        if arr.ndim == 3:
            return np.tile(col[:, None, :], (1, width, 1))
        return np.tile(col[:, None], (1, width))
    sl = arr[:, olo:ohi]
    pl_need = pad_l - lo if lo < pad_l else 0
    pr_need = hi - (pad_l + tw) if hi > pad_l + tw else 0
    if pl_need > 0 or pr_need > 0:
        pads = ((0, 0), (pl_need, pr_need))
        if arr.ndim == 3:
            pads = pads + ((0, 0),)
        sl = np.pad(sl, pads, mode="edge")
    return sl


def _slice_feather_mask(
    tile_h: int,
    tile_w: int,
    pad_l: int,
    pad_r: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    feather: int,
    hard: Tuple[bool, bool, bool, bool],
) -> np.ndarray:
    """Feather weight for a tile slice, anchored at the tile's TRUE edges.

    Replicates preprocessing.feather_mask (as applied by the global path to
    the canvas-clipped PADDED image): ramps of length ef at every non-hard
    side, full weight on hard sides and inside gap-pad zones. The ramps are
    positioned in the tile's own coordinate frame, so a slice cut through
    the middle of a tile does NOT create a spurious ramp (a naive
    feather_mask on the slice would ramp the cut to zero and break the
    cross-fade in the overlap zone).
    """
    hl, hr, ht, hb = hard
    pw = col_hi - col_lo
    ph = row_hi - row_lo
    mask = np.ones((ph, pw), dtype=np.float32)
    # ef exactly as feather_mask computes it for the padded image
    ef = min(feather, tile_h // 2, (tile_w + pad_l + pad_r) // 2)
    if ef > 0:
        ramp = np.linspace(0, 1, ef, dtype=np.float32)
        if not hl:
            a = max(col_lo, 0)
            b = min(ef, col_hi)
            if b > a:
                mask[:, a - col_lo:b - col_lo] *= ramp[a:b][None, :]
        if not hr:
            a = max(tile_w + pad_l + pad_r - ef, col_lo)
            b = min(tile_w + pad_l + pad_r, col_hi)
            if b > a:
                rev = ramp[::-1][a - (tile_w + pad_l + pad_r - ef):
                                 b - (tile_w + pad_l + pad_r - ef)]
                mask[:, a - col_lo:b - col_lo] *= rev[None, :]
        if not ht:
            a = max(row_lo, 0)
            b = min(ef, row_hi)
            if b > a:
                mask[a - row_lo:b - row_lo, :] *= ramp[a:b][:, None]
        if not hb:
            a = max(tile_h - ef, row_lo)
            b = min(tile_h, row_hi)
            if b > a:
                rev = ramp[::-1][a - (tile_h - ef):b - (tile_h - ef)]
                mask[a - row_lo:b - row_lo, :] *= rev[:, None]
    return mask


def _ctx_focus_slice(
    img_full: np.ndarray,
    gain: Optional[np.ndarray],
    pad_l: int,
    pad_r: int,
    wp: int,
    r0t: int,
    r1t: int,
    lo: int,
    hi: int,
    ctx: int,
    block_size: int,
    fn,
) -> np.ndarray:
    """Focus map (Tenengrad / raw sharpness) of the window slice, computed
    with EXTRA context so the core values are exact.

    The focus maps run Sobel + a block-mean filter2D + a Gaussian blur
    (sigma = block_size/2) over the array — near an array border OpenCV
    reflects, which differs from the global path's values when the slice
    is cut through a tile's interior. Extending the slice by `ctx`
    pixels (>= kernel radii) on each side and cropping the result back to
    [lo, hi) reproduces the global per-pixel values exactly; where the
    extension hits the tile's own (padded) bounds the reflect border is
    the same one the global path uses.
    """
    ext_lo = max(0, lo - ctx)
    ext_hi = min(wp, hi + ctx)
    sl = _slice_padded_columns(img_full[r0t:r1t], pad_l, pad_r, ext_lo, ext_hi)
    sl = _apply_gain_slice(sl, gain)
    # The standard path builds its focus maps on the FLOAT32 view of the
    # uint8 tile (tile_img = img_compensated.astype(np.float32)), so
    # to_gray() runs on float32 input and yields FRACTIONAL gray values.
    # Running it on the uint8 slice would round the gray first and produce
    # a slightly different map — replicate the standard path exactly.
    sl = sl.astype(np.float32)
    m = fn(sl, block_size=block_size)
    del sl
    return m[:, lo - ext_lo:hi - ext_lo]


def _build_slice_pyramid_set(
    img_full: np.ndarray,
    weight_full: Optional[np.ndarray],
    offset_x: int,
    offset_y: int,
    pad_l: int,
    pad_r: int,
    hard: Tuple[bool, bool, bool, bool],
    canvas_h: int,
    canvas_w: int,
    wx0: int,
    wx1: int,
    levels: int,
    feather: int,
    block_size: int,
    gamma_correction: float,
    use_focus_weights: bool,
    focus_base_weight: float,
    focus_priority: bool,
    focus_priority_gamma: float,
    gain: Optional[np.ndarray],
    ten_max: Optional[float] = None,
    sharp_stats: Optional[Tuple[float, float, float]] = None,
    focus_ctx: int = 64,
) -> Optional[Tuple[int, int, List[np.ndarray], List[np.ndarray],
                    Optional[List[np.ndarray]]]]:
    """Per-tile-slice pyramid set for one slab window (see module notes).

    Returns (x0, y0, lap_pyr, weight_pyr_2d, ten_pyr_2d) with x0/y0 = the
    slice's top-left corner on the CANVAS, or None when the tile does not
    intersect the window / canvas.

    ten_max: the tile's GLOBAL Tenengrad maximum (over the full clipped
        padded compensated tile, exactly as the standard path computes it)
        — required for focus_priority so the winner-takes-all weights are
        a per-TILE function and stay continuous across slab cuts.
    sharp_stats: (smin, smax, cmax) of the whole tile for the legacy
        focus-weighted mode — required so the per-tile normalization of
        the standard path is replicated exactly.
    """
    th, tw = img_full.shape[:2]
    x0p = offset_x - pad_l
    wp = tw + pad_l + pad_r
    cx0 = max(x0p, wx0)
    cx1 = min(x0p + wp, wx1)
    cy0 = max(offset_y, 0)
    cy1 = min(offset_y + th, canvas_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    r0t = cy0 - offset_y                     # tile-row index of the slice top
    r1t = r0t + (cy1 - cy0)
    lo = cx0 - x0p                           # padded-column index of slice left
    hi = cx1 - x0p

    # --- slice of the (conceptually) gap-padded, gain-compensated tile ---
    sl = _slice_padded_columns(img_full[r0t:r1t], pad_l, pad_r, lo, hi)
    sl = _apply_gain_slice(sl, gain)

    # --- weight mask (branch order mirrors _build_tile_pyramid_set) ---
    if weight_full is not None:
        tile_weight = _slice_padded_columns(weight_full[r0t:r1t],
                                            pad_l, pad_r, lo, hi)
        if use_focus_weights and gamma_correction != 1.0:
            tile_weight = tile_weight ** gamma_correction
    elif focus_priority:
        tile_weight = _slice_feather_mask(
            th, tw, pad_l, pad_r, r0t, r1t, lo, hi, feather, hard)
    elif use_focus_weights:
        # NOTE: the per-tile normalization constants (smin/smax of the
        # whole tile + the tile's combined-weight maximum) come from the
        # upfront pass — they replicate the standard path's per-TILE
        # normalization exactly.
        raw = _ctx_focus_slice(
            img_full, gain, pad_l, pad_r, wp, r0t, r1t, lo, hi,
            focus_ctx, block_size, preprocessing.sharpness_raw_map)
        smin_t, smax_t, cmax_t = sharp_stats
        if smax_t > smin_t:
            sharp = (raw - smin_t) / (smax_t - smin_t)
        else:
            sharp = np.zeros_like(raw)
        del raw
        sharp = sharp ** gamma_correction
        feather_m = _slice_feather_mask(
            th, tw, pad_l, pad_r, r0t, r1t, lo, hi, feather, hard)
        combined = np.maximum(sharp, focus_base_weight) * feather_m
        del sharp, feather_m
        if cmax_t > 1e-6:
            combined = combined / cmax_t
        tile_weight = combined.astype(np.float32)
    else:
        tile_weight = _slice_feather_mask(
            th, tw, pad_l, pad_r, r0t, r1t, lo, hi, feather, hard)

    # --- pyramids (tile-slice sized) ---
    tile_lap_pyr = _build_laplacian_pyramid(sl, levels)
    w_pyr = _build_gaussian_pyramid(tile_weight[..., None], levels)
    tile_weight_pyr_2d = [w[..., 0] if w.ndim == 3 else w for w in w_pyr]
    del sl, w_pyr

    ten_pyr_2d: Optional[List[np.ndarray]] = None
    if focus_priority:
        ten_map = _ctx_focus_slice(
            img_full, gain, pad_l, pad_r, wp, r0t, r1t, lo, hi,
            focus_ctx, block_size, preprocessing.tenengrad_map)
        eps = (1e-3 * ten_max) if (ten_max is not None and ten_max > 0) else 1e-3
        ten_weight = (ten_map + eps) ** focus_priority_gamma
        del ten_map
        # multiply by the feather/precomputed weight (gap guard) — the SAME
        # tile_weight the standard path multiplies with
        ten_weight = ten_weight * tile_weight
        t_pyr = _build_gaussian_pyramid(ten_weight[..., None], levels)
        ten_pyr_2d = [w[..., 0] if w.ndim == 3 else w for w in t_pyr]
        del ten_weight, t_pyr
    del tile_weight

    # === Canvas-border extension of the coarse levels ======================
    # Same fix as the global path, but triggered only by the GLOBAL canvas
    # border (a slab cut is not a border). The right border is inside this
    # window only when the window reaches the canvas edge; the bottom
    # border is always inside (windows span the full canvas height).
    reach_r = (x0p + wp) >= canvas_w and wx1 >= canvas_w
    reach_b = (offset_y + th) >= canvas_h
    if reach_r or reach_b:
        def _pad_border(arr: np.ndarray, lvl: int) -> np.ndarray:
            cx0l = _level_offset(cx0, lvl) - _level_offset(wx0, lvl)
            cy0l = _level_offset(cy0, lvl)
            h_l, w_l = _level_size(canvas_h, wx1 - wx0, lvl)
            need_w = w_l - cx0l
            need_h = h_l - cy0l
            pads = [
                (0, max(0, need_h - arr.shape[0])) if reach_b else (0, 0),
                (0, max(0, need_w - arr.shape[1])) if reach_r else (0, 0),
            ]
            if pads[0][1] or pads[1][1]:
                extra = ((0, 0),) * (arr.ndim - 2)
                return np.pad(arr, tuple(pads) + extra, mode="edge")
            return arr

        for lvl in range(levels + 1):
            tile_lap_pyr[lvl] = _pad_border(tile_lap_pyr[lvl], lvl)
            tile_weight_pyr_2d[lvl] = _pad_border(tile_weight_pyr_2d[lvl], lvl)
        if ten_pyr_2d is not None:
            for lvl in range(levels + 1):
                ten_pyr_2d[lvl] = _pad_border(ten_pyr_2d[lvl], lvl)

    return (cx0, cy0, tile_lap_pyr, tile_weight_pyr_2d, ten_pyr_2d)


def _blend_tiles_slabbed(
    placed: List[PlacedTile],
    canvas_h: int,
    canvas_w: int,
    levels: int,
    feather: int,
    block_size: int,
    gamma_correction: float,
    gain_strength: float,
    use_focus_weights: bool,
    focus_base_weight: float,
    min_total_weight: float,
    focus_priority: bool,
    focus_priority_gamma: float,
    focus_priority_levels: int,
    repair_stats: Optional[dict],
    progress_callback: Optional[Callable[[str, float], None]] = None,
    slab_width: Optional[int] = None,
) -> np.ndarray:
    """Assemble the panorama in narrow column slabs (low-RAM path).

    Produces the same output as the standard blend_tiles path (see the
    module-level notes for the exactness argument) while never allocating
    canvas-sized float32 arrays. Peak RAM ≈ tiles + uint8 output canvas +
    coverage flags + ~300 MB per-slab working set.
    """

    def report(msg, p):
        if progress_callback:
            progress_callback(msg, p)

    n_tiles = len(placed)
    if n_tiles == 0:
        return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    report("Low-RAM blending: preparing tiles...", 0.0)

    # === Gain factors (compensate_gain math, applied per slice) =========
    gain_factors = _gain_factors([t.image for t in placed], gain_strength)

    # === Canvas-border sides keep FULL weight (global Step 0c) ==========
    for k, tile in enumerate(placed):
        hl, hr, ht, hb = tile.hard_edges
        tw = tile.image.shape[1]
        th = tile.image.shape[0]
        hl = hl or (tile.offset_x <= 0)
        hr = hr or (tile.offset_x + tw >= canvas_w)
        ht = ht or (tile.offset_y <= 0)
        hb = hb or (tile.offset_y + th >= canvas_h)
        if (hl, hr, ht, hb) != tile.hard_edges:
            placed[k] = replace(tile, hard_edges=(hl, hr, ht, hb))

    # === Gap pads (global Step 0b math, from ORIGINAL tile widths) ======
    MAX_GAP_PAD = 4096
    gap_pads: List[Tuple[int, int]] = []
    for k, tile in enumerate(placed):
        hl, hr, _, _ = tile.hard_edges
        pad_l = pad_r = 0
        if hr and k + 1 < n_tiles:
            pad_r = max(0, min(
                placed[k + 1].offset_x
                - (tile.offset_x + tile.image.shape[1]),
                MAX_GAP_PAD))
        if hl and k >= 1:
            pad_l = max(0, min(
                tile.offset_x
                - (placed[k - 1].offset_x + placed[k - 1].image.shape[1]),
                MAX_GAP_PAD))
        gap_pads.append((pad_l, pad_r))

    # === Upfront per-tile normalization constants (focus modes) ==========
    # The standard path computes the Tenengrad epsilon and the legacy
    # focus-weight normalization over the WHOLE (clipped, padded,
    # gain-compensated) tile. Replicate those scalars once per tile so the
    # slabbed path is identical (and the focus weights stay a per-TILE
    # function — continuous across slab cuts). Transient cost: one
    # tile-sized image + map per tile, freed immediately.
    ten_maxes: List[Optional[float]] = [None] * n_tiles
    sharp_stats: List[Optional[Tuple[float, float, float]]] = [None] * n_tiles
    if focus_priority or use_focus_weights:
        report("Low-RAM blending: per-tile focus statistics...", 0.01)
        for k, tile in enumerate(placed):
            th_t, tw_t = tile.image.shape[:2]
            pad_l_t, pad_r_t = gap_pads[k]
            x0p_t = tile.offset_x - pad_l_t
            wp_t = tw_t + pad_l_t + pad_r_t
            actual_w_t = min(wp_t, canvas_w - x0p_t)
            r0f = max(0, -tile.offset_y)
            r1f = min(th_t, canvas_h - tile.offset_y)
            img_cl = _slice_padded_columns(
                tile.image[r0f:r1f], pad_l_t, pad_r_t, 0, actual_w_t)
            img_cl = _apply_gain_slice(
                img_cl, gain_factors[k] if gain_factors else None)
            # float32 view — see the note in _ctx_focus_slice (the standard
            # path computes its focus maps on float32 input)
            img_cl = img_cl.astype(np.float32)
            hl_t, hr_t, ht_t, hb_t = tile.hard_edges
            if focus_priority:
                tm = preprocessing.tenengrad_map(img_cl, block_size=block_size)
                ten_maxes[k] = float(tm.max())
                del tm
            if use_focus_weights and not focus_priority:
                raw = preprocessing.sharpness_raw_map(
                    img_cl, block_size=block_size)
                smin_t = raw.min()
                smax_t = raw.max()
                fw_full = preprocessing.feather_mask(
                    img_cl, edge_feather=feather,
                    hard_left=hl_t, hard_right=hr_t,
                    hard_top=ht_t, hard_bottom=hb_t)
                if smax_t > smin_t:
                    sharp_full = (raw - smin_t) / (smax_t - smin_t)
                else:
                    sharp_full = np.zeros_like(raw)
                del raw
                sharp_full = sharp_full ** gamma_correction
                comb_full = np.maximum(sharp_full, focus_base_weight) * fw_full
                del sharp_full, fw_full
                cmax_t = float(comb_full.max())
                del comb_full
                sharp_stats[k] = (np.float32(smin_t), np.float32(smax_t),
                                  np.float32(cmax_t))
            del img_cl

    # === Coverage flags (global Step 1, one bool canvas) =================
    has_coverage = np.zeros((canvas_h, canvas_w), dtype=bool)
    for tile in placed:
        th, tw = tile.image.shape[:2]
        x0, y0 = tile.offset_x, tile.offset_y
        x1, y1 = min(x0 + tw, canvas_w), min(y0 + th, canvas_h)
        if x1 > x0 and y1 > y0:
            has_coverage[y0:y1, x0:x1] = True

    # === Output canvas (uint8 — this is the panorama itself) =============
    final = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # === Slab geometry ====================================================
    align = 1 << min(levels, 6)              # coarsest cell width
    margin = _SLAB_MARGIN_CELLS * align
    if slab_width is None or slab_width <= 0:
        slab_width = max(
            align * 8,
            int(_SLAB_TARGET_BYTES / (16 * max(canvas_h, 1))))
        # slab pitch must be a multiple of the coarsest cell so the level-l
        # cell grid of every window is the global grid shifted by a whole
        # cell count (see the exactness notes)
        slab_width = ((slab_width + align - 1) // align) * align
    slab_width = min(slab_width, canvas_w)
    n_slabs = max(1, (canvas_w + slab_width - 1) // slab_width)

    for s in range(n_slabs):
        sx0 = s * slab_width
        sx1 = min(canvas_w, sx0 + slab_width)          # core columns
        wx0 = max(0, sx0 - margin)                     # window (with context)
        wx1 = min(canvas_w, sx1 + margin)
        w_loc = wx1 - wx0

        # Per-slab canvas arrays (window-sized — the whole point)
        canvas_laps: List[np.ndarray] = []
        canvas_weights: List[np.ndarray] = []
        for lvl in range(levels + 1):
            h_l, w_l = _level_size(canvas_h, w_loc, lvl)
            canvas_laps.append(np.zeros((h_l, w_l, 3), dtype=np.float32))
            canvas_weights.append(np.zeros((h_l, w_l), dtype=np.float32))

        # Build + accumulate every tile slice of this window sequentially
        # (sequential keeps peak RAM at ONE slice pyramid set)
        for k, tile in enumerate(placed):
            ps = _build_slice_pyramid_set(
                img_full=tile.image,
                weight_full=tile.weight,
                offset_x=tile.offset_x,
                offset_y=tile.offset_y,
                pad_l=gap_pads[k][0],
                pad_r=gap_pads[k][1],
                hard=tile.hard_edges,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
                wx0=wx0,
                wx1=wx1,
                levels=levels,
                feather=feather,
                block_size=block_size,
                gamma_correction=gamma_correction,
                use_focus_weights=use_focus_weights,
                focus_base_weight=focus_base_weight,
                focus_priority=focus_priority,
                focus_priority_gamma=focus_priority_gamma,
                gain=(gain_factors[k] if gain_factors else None),
                ten_max=ten_maxes[k],
                sharp_stats=sharp_stats[k],
            )
            if ps is None:
                continue
            x0s, y0s, lap_pyr, w_pyr2d, ten_pyr2d = ps
            for lvl in range(levels + 1):
                lap_l = lap_pyr[lvl]
                if focus_priority and lvl < focus_priority_levels:
                    w_l = ten_pyr2d[lvl]
                else:
                    w_l = w_pyr2d[lvl]
                cx0l = _level_offset(x0s, lvl) - _level_offset(wx0, lvl)
                cy0l = _level_offset(y0s, lvl)
                ch, cw = lap_l.shape[:2]
                canvas_h_l, canvas_w_l = canvas_laps[lvl].shape[:2]
                cy1 = min(cy0l + ch, canvas_h_l)
                cx1 = min(cx0l + cw, canvas_w_l)
                if cy1 <= cy0l or cx1 <= cx0l:
                    continue
                lap_part = lap_l[:cy1 - cy0l, :cx1 - cx0l]
                w_part = w_l[:cy1 - cy0l, :cx1 - cx0l]
                canvas_laps[lvl][cy0l:cy1, cx0l:cx1] += \
                    lap_part * w_part[..., None]
                canvas_weights[lvl][cy0l:cy1, cx0l:cx1] += w_part
            del lap_pyr, w_pyr2d, ten_pyr2d, ps

        # Normalize (same math as the global path, on the window slice)
        cov_sl = has_coverage[:, wx0:wx1]
        cov_pyr = _build_gaussian_pyramid(
            cov_sl.astype(np.float32)[..., None], levels)
        cov_pyr_2d = [c[..., 0] if c.ndim == 3 else c for c in cov_pyr]
        del cov_sl
        for lvl in range(levels + 1):
            lvl_coverage = cov_pyr_2d[lvl] > 0.1
            lvl_total = canvas_weights[lvl]
            lvl_total_safe = np.where(
                lvl_coverage & (lvl_total < min_total_weight),
                min_total_weight,
                lvl_total,
            )
            lvl_total_safe = np.where(lvl_total_safe > 1e-6,
                                      lvl_total_safe, 1.0)
            canvas_laps[lvl] /= lvl_total_safe[..., None]
        del cov_pyr, cov_pyr_2d, canvas_weights

        # Reconstruct and paste the CORE columns (margins are context only)
        recon = _reconstruct_from_laplacian(canvas_laps)
        del canvas_laps
        recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)
        del recon
        recon_u8[~has_coverage[:, wx0:wx1]] = 0
        px0, px1 = sx0, sx1
        final[:, px0:px1] = recon_u8[:, px0 - wx0:px1 - wx0]
        del recon_u8

        report(f"Low-RAM blending: slab {s + 1}/{n_slabs} "
               f"(columns {sx0}..{sx1})...",
               0.02 + 0.9 * ((s + 1) / n_slabs))

    # Uncovered pixels are pure black (same as the global path)
    final[~has_coverage] = 0

    stats = _repair_panorama_artifacts(final, has_coverage)
    if repair_stats is not None:
        repair_stats.update(stats)
        repair_stats["mode"] = "slabbed"
        repair_stats["slabs"] = int(n_slabs)
    return final


def _repair_panorama_artifacts(final: np.ndarray,
                               has_coverage: np.ndarray,
                               max_gap_px: int = 32,
                               outlier_thresh: int = 60,
                               row_band: int = 128) -> dict:
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

    MEMORY: both repairs are processed in ROW BANDS. The gap fill is
    row-wise anyway, and the 3x3 median only needs one row of context, so
    banding (with a one-row overlap for the median) produces IDENTICAL
    results to the old full-image implementation while the transient
    buffers shrink from ~2 GB (full-size int16 copies of a wide panorama)
    to a few tens of MB per band.
    """
    H, W = has_coverage.shape[:2]
    filled = 0
    nc = ~has_coverage
    if nc.any():
        for r0 in range(0, H, row_band):
            r1 = min(H, r0 + row_band)
            band = nc[r0:r1].astype(np.int8)
            d = np.diff(np.concatenate(
                [np.zeros((r1 - r0, 1), np.int8), band,
                 np.zeros((r1 - r0, 1), np.int8)], axis=1), axis=1)
            for rl in range(r1 - r0):
                row = band[rl]
                if not row.any():
                    continue
                starts = np.where(d[rl] == 1)[0]
                ends = np.where(d[rl] == -1)[0]
                if len(starts) == 0:
                    continue
                r = r0 + rl
                fr = final[r]
                for s, e in zip(starts, ends):
                    if s <= 0 or e >= W:          # touches the row border
                        continue
                    if e - s > max_gap_px:        # a real gap, not a sliver
                        continue
                    left = fr[s - 1].astype(np.float32)
                    right = fr[e].astype(np.float32)
                    n = e - s
                    t = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))
                    fr[s:e] = (
                        left[None, :] * (1.0 - t[:, None])
                        + right[None, :] * t[:, None]).astype(np.uint8)
                    filled += int(n)
            del band, d
    repaired = 0
    if has_coverage.any() and outlier_thresh > 0:
        for r0 in range(0, H, row_band):
            r1 = min(H, r0 + row_band)
            lo = max(0, r0 - 1)
            hi = min(H, r1 + 1)
            med = cv2.medianBlur(final[lo:hi], 3)
            # rows r0..r1-1 sit at med rows r0-lo..r1-lo; the ±1 context
            # rows make their 3x3 medians identical to the full-image pass
            f16 = final[r0:r1].astype(np.int16)
            m16 = med[r0 - lo:r1 - lo].astype(np.int16)
            diff = np.abs(f16 - m16).max(axis=2)
            bad = (diff > outlier_thresh) & has_coverage[r0:r1]
            repaired += int(bad.sum())
            if bad.any():
                final[r0:r1][bad] = med[r0 - lo:r1 - lo][bad]
            del med, f16, m16, diff, bad
    return {"gap_px_filled": filled, "outlier_px_repaired": repaired}
