"""
pipeline.py — Orchestrating the entire tile-row stitching process.

Steps:
1. Load tiles from a folder, sort by _N index
2. Estimate pairwise shifts between adjacent tiles
   (SIFT verified against a global NCC scan; fallbacks: phase, NCC)
3. Drift-outlier guard: detect pairwise shifts that deviate sharply from
   the row's vertical drift trend and re-verify them with a constrained
   NCC scan (protects against SIFT blunders on repetitive textures)
4. Compute cumulative absolute positions on the final canvas
5. Multi-band blending
6. Return the result + diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import blending, io_utils, preprocessing, stitching


@dataclass
class PairDiagnostic:
    """Diagnostics for a single pair."""
    pair_index: int                  # 0-based, pair (i, i+1)
    method: str
    success: bool
    translation: Tuple[float, float]
    rotation_deg: float
    scale: float
    matches: int
    inliers: int
    message: str
    # NCC verification evidence (None when the NCC scan was not run)
    score: float = 0.0                                  # NCC score at the final translation
    ncc_best: Optional[Tuple[float, float]] = None      # global NCC optimum
    ncc_best_score: float = 0.0
    # Row-structure corrected translation (None when no correction happened)
    profile_translation: Optional[Tuple[float, float]] = None


@dataclass
class StitchResult:
    """Full result of stitching a row of tiles."""
    panorama: np.ndarray            # BGR uint8
    canvas_size: Tuple[int, int]    # (H, W)
    tile_positions: List[Tuple[int, int]]  # absolute (offset_x, offset_y) of each tile
    diagnostics: List[PairDiagnostic] = field(default_factory=list)
    n_tiles: int = 0
    n_pairs: int = 0
    warnings: List[str] = field(default_factory=list)
    drift_trend: Optional[Tuple[float, float]] = None  # (dy per px of dx, intercept)
    # Optional full-resolution panorama (without ROI cropping) — only set
    # when the user enabled "Also keep full panorama (debug)" AND a ROI band
    # was applied. None in all other cases.
    panorama_full: Optional[np.ndarray] = None
    # ROI bands actually used per tile (in pixels of the ORIGINAL tile
    # coordinate frame). None when no ROI crop was applied.
    roi_bands_used: Optional[List[Tuple[int, int]]] = None


def _robust_dy_trend(
    dxs: np.ndarray,
    dys: np.ndarray,
    inlier_tol: float = 60.0,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """Robust linear fit dy = slope * dx + intercept for a 1D tile row.

    Uses exhaustive line hypotheses through pairs of points (RANSAC-style),
    which survives one or more gross outliers even when the chain is short
    (plain Theil-Sen gets skewed and stops flagging the outlier).
    Returns (slope, intercept, inlier_mask) or None if no plausible trend.
    """
    x = np.asarray(dxs, dtype=np.float64)
    y = np.asarray(dys, dtype=np.float64)
    n = len(x)
    if n < 3:
        return None
    best = None  # (n_inliers, slope, intercept)
    for i in range(n):
        for j in range(i + 1, n):
            ddx = x[j] - x[i]
            if abs(ddx) < 1e-9:
                continue
            slope = (y[j] - y[i]) / ddx
            if abs(slope) > 2.0:  # >63 deg vertical per px horizontal — implausible
                continue
            intercept = y[i] - slope * x[i]
            resid = np.abs(y - (slope * x + intercept))
            ninl = int((resid < inlier_tol).sum())
            if best is None or ninl > best[0]:
                best = (ninl, slope, intercept)
    if best is None:
        return None
    ninl, slope, intercept = best
    if ninl < max(3, int(np.ceil(0.6 * n))):
        return None  # trend not trusted
    resid = np.abs(y - (slope * x + intercept))
    mask = resid < inlier_tol
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept), mask


def stitch_row(
    tiles: List[io_utils.TileInfo],
    n_features: int = 2000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
    min_inliers_sift: int = 8,
    enable_phase_fallback: bool = True,
    enable_ncc_fallback: bool = True,
    verify_with_ncc: bool = True,
    ncc_verify_gap: float = 0.07,
    suppress_fixed_pattern: bool = True,
    enable_drift_guard: bool = True,
    expected_overlap_pct: Optional[float] = None,
    blend_levels: int = 3,
    blend_feather: int = 64,
    blend_block_size: int = 16,
    blend_gamma: float = 1.5,
    blend_gain_strength: float = 0.8,
    blend_use_focus_weights: bool = False,
    blend_focus_priority: bool = False,
    blend_focus_priority_gamma: float = 6.0,
    blend_focus_priority_levels: int = 2,
    pair_translations: Optional[List[Tuple[float, float]]] = None,
    pair_methods: Optional[List[str]] = None,
    estimation_max_side: Optional[int] = 1600,
    estimation_polish_full_res: bool = True,
    roi_band: Optional[Tuple[int, int]] = None,
    roi_bands_per_tile: Optional[List[Tuple[int, int]]] = None,
    keep_full_panorama: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> StitchResult:
    """Stitch a row of tiles into a single panorama.

    tiles: list of TileInfo, already sorted by index.
    progress_callback: optional (message, progress_0_to_1).

    verify_with_ncc: cross-check every SIFT result against a global NCC scan
        and reject SIFT answers that have a clearly better NCC alignment.
    ncc_verify_gap: minimum NCC score advantage for rejecting SIFT.
    suppress_fixed_pattern: drop matches with ~zero translation (dust on optics).
    enable_drift_guard: detect and fix pairwise dy values that deviate sharply
        from the row's vertical drift trend.
    expected_overlap_pct: user's approximate capture overlap between adjacent
        frames (% of tile width), or None if unknown. Used only to
        disambiguate pairs whose horizontal shift is ambiguous (flat
        correlation ridge on layered samples).
    pair_translations: optional list of (dx, dy) shifts, one per adjacent
        pair, in full-resolution pixels. When given, pairwise estimation is
        skipped entirely and these positions are used exactly as provided
        (used by the Pair Review workflow: user-approved / manually adjusted
        positions). Must contain exactly n-1 entries.
    pair_methods: optional labels ("reviewed" / "manual" / ...) shown in the
        diagnostics for each overridden pair.

    estimation_max_side: when set (default 1600), pairwise estimation
        (SIFT + NCC + phase + ECC) runs on DOWNSCALED copies of the tiles
        whose longest side is at most this many pixels; the estimated
        translations are then scaled back to full resolution. This is the
        single biggest speedup for large images (e.g. 5472x3648 20 MP
        frames): estimation is ~9-13x faster while the blending still
        happens at FULL resolution, so the final panorama quality is
        unchanged. None or 0 disables downscaling (full-resolution
        estimation, legacy behaviour).
    estimation_polish_full_res: when True (default), after scaling the
        downscaled estimate back up, one quick full-resolution ECC polish
        (~50-300 ms per pair) refines the translation to true sub-pixel
        accuracy — this removes the quantization error introduced by the
        downscale. Disable for a small extra speedup at the cost of
        ~0.5-1 px seam accuracy.

    roi_band: optional (y0, y1) in pixels of the ORIGINAL tile height,
        the same vertical band applied to EVERY tile before stitching.
        Crops each tile to img[y0:y1, :, :] (width preserved). Saves
        roughly (H / (y1-y0))x on SIFT/NCC/blend time AND on the final
        panorama size. Pairwise translations computed on the cropped
        tiles are also valid in the full-tile coordinate frame, so the
        optional `panorama_full` debug output (see below) is exact. Use
        this when the operator kept the disk vertically aligned across
        the row.
    roi_bands_per_tile: optional list of N (y0, y1) tuples — one per
        tile — for the case where the disk shifted vertically between
        frames (e.g. hand-positioned stage). Each tile is cropped to
        its own band. Mutually exclusive with `roi_band` (if both are
        given, `roi_bands_per_tile` wins). Pairwise dy values are in
        the cropped coordinate frame; they are NOT valid in the full-
        tile frame, so `keep_full_panorama` is then a no-op and a
        warning is emitted.
    keep_full_panorama: when True AND a ROI band was applied (via
        `roi_band` only, not `roi_bands_per_tile`), the pipeline also
        blends the panorama on the ORIGINAL full tiles using the same
        pair translations, and the result is returned as
        `result.panorama_full`. The full panorama costs one EXTRA blend
        pass (roughly (H / (y1-y0))x slower than the ROI panorama) and
        one extra canvas allocation — keep it OFF for production runs.

    Returns: StitchResult with panorama, positions, and diagnostics.
    """
    n = len(tiles)
    if n < 2:
        raise ValueError(f"At least 2 tiles are required for stitching, got: {n}")

    use_overrides = pair_translations is not None
    if use_overrides and len(pair_translations) != n - 1:
        raise ValueError(
            f"pair_translations must contain exactly {n - 1} (dx, dy) entries "
            f"for {n} tiles, got {len(pair_translations)}")
    if pair_methods is not None and use_overrides and len(pair_methods) != n - 1:
        raise ValueError(
            f"pair_methods must contain exactly {n - 1} entries, "
            f"got {len(pair_methods)}")

    # Resolve ROI crop: roi_bands_per_tile takes precedence over roi_band.
    # Keep references to the ORIGINAL (uncropped) tiles so we can build
    # the optional panorama_full debug output later.
    original_tiles: List[io_utils.TileInfo] = list(tiles)
    roi_bands_used: Optional[List[Tuple[int, int]]] = None
    if roi_bands_per_tile is not None:
        if len(roi_bands_per_tile) != n:
            raise ValueError(
                f"roi_bands_per_tile must contain exactly {n} (y0, y1) "
                f"entries, got {len(roi_bands_per_tile)}")
        roi_bands_used = list(roi_bands_per_tile)
        tiles = [io_utils.crop_tile_to_roi_band(t, y0, y1)
                 for t, (y0, y1) in zip(tiles, roi_bands_used)]
    elif roi_band is not None:
        y0, y1 = roi_band
        roi_bands_used = [(y0, y1)] * n
        tiles = [io_utils.crop_tile_to_roi_band(t, y0, y1) for t in tiles]

    def report(msg, p):
        if progress_callback:
            progress_callback(msg, p)

    if roi_bands_used is not None:
        first_h_orig = int(original_tiles[0].image.shape[0])
        if roi_band is not None and roi_bands_per_tile is None:
            y0r, y1r = roi_band
            report(f"ROI band applied to all tiles: y=[{y0r},{y1r}) "
                   f"({y1r - y0r} px of {first_h_orig} px; ~"
                   f"{(y1r - y0r) / max(1, first_h_orig):.2f} of full height)",
                   0.02)
        else:
            heights = [y1 - y0 for y0, y1 in roi_bands_used]
            report(f"Per-tile ROI bands applied (heights: {heights} px "
                   f"of {first_h_orig} px). Per-tile dy is in cropped "
                   f"coordinates \xe2\x80\x94 keep_full_panorama is a no-op here.",
                   0.02)

    # === Step 1: pairwise stitching ===
    pair_results: List[PairDiagnostic] = []

    if use_overrides:
        # Positions fixed by the Pair Review workflow — no estimation.
        report("Using reviewed / manual pair positions (estimation skipped)...", 0.60)
        for i, (dx, dy) in enumerate(pair_translations):
            method = pair_methods[i] if pair_methods else "reviewed"
            if method == "manual":
                msg = "Position set manually in Pair Review"
            elif method == "no-overlap":
                msg = ("No overlap: tiles placed side by side with a black "
                       "gap (set in Pair Review)")
            else:
                msg = "Position taken from Pair Review (automatic estimate)"
            pair_results.append(PairDiagnostic(
                pair_index=i,
                method=method,
                success=True,
                translation=(float(dx), float(dy)),
                rotation_deg=0.0,
                scale=1.0,
                matches=0,
                inliers=0,
                message=msg,
            ))
    else:
        report(f"Analyzing {n-1} adjacent pairs...", 0.05)

        tile_w = int(tiles[0].image.shape[1])
        expected_dx = (tile_w * (1.0 - expected_overlap_pct / 100.0)
                       if expected_overlap_pct else None)
        dx_history: List[float] = []          # accepted dx of solved pairs

        # Optimization No1: per-tile feature cache. Each interior tile is
        # processed as the right tile of pair (i-1, i) AND the left tile of
        # pair (i, i+1) — without a cache, CLAHE and SIFT detectAndCompute
        # run twice per interior tile. The cache is keyed by id(img) (the
        # pipeline holds the SAME ndarray object for each tile throughout
        # the row, so id(img) is stable). Fresh per row, discarded at exit.
        tile_cache: Dict[int, Tuple] = {}

        # Downscale-for-estimation: when estimation_max_side is set and the
        # tiles are larger, all pairwise estimation (SIFT + NCC + phase +
        # ECC) runs on downscaled copies. The estimated translations are
        # scaled back up (est_scale divides: full_px = est_px / est_scale)
        # and optionally polished at full resolution with one quick ECC
        # run to recover sub-pixel accuracy. Blending always happens on
        # the ORIGINAL full-resolution images, so panorama quality is
        # unchanged.
        est_imgs: List[np.ndarray] = list(t.image for t in tiles)
        est_scale = 1.0
        if estimation_max_side and estimation_max_side > 0:
            h0, w0 = tiles[0].image.shape[:2]
            longest = max(h0, w0)
            if longest > estimation_max_side:
                est_scale = float(estimation_max_side) / float(longest)
                # Downscale each tile ONCE (each tile appears at exactly
                # one index in the row, so no cross-pair reuse is needed
                # for the downscaled copies themselves).
                est_imgs = [
                    cv2.resize(t.image, None, fx=est_scale, fy=est_scale,
                               interpolation=cv2.INTER_AREA)
                    for t in tiles
                ]
                report(f"Estimation resolution: {est_imgs[0].shape[1]}x"
                       f"{est_imgs[0].shape[0]} (scale {est_scale:.3f} of "
                       f"full resolution; blending still at full res)",
                       0.05)

        for i in range(n - 1):
            report(f"Pair {i+1}/{n-1}: tile {tiles[i].name} -> {tiles[i+1].name}",
                   0.03 + 0.55 * (i / max(1, n - 2)))
            # dx prior: median of already-solved pairs (>= 2), else the user's
            # expected overlap. Robust to one bad pair among several good ones.
            row_prior_dx = (float(np.median(dx_history))
                            if len(dx_history) >= 2 else None)
            # The prior and expected_dx are expressed in FULL-resolution
            # pixels; the estimation stage works on downscaled images, so
            # both are converted to estimation scale for this call.
            row_prior_est = (row_prior_dx * est_scale
                             if row_prior_dx is not None else None)
            expected_dx_est = (expected_dx * est_scale
                               if expected_dx is not None else None)
            result = stitching.stitch_pair(
                est_imgs[i], est_imgs[i + 1],
                n_features=n_features,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
                use_clahe=use_clahe,
                clahe_clip=clahe_clip,
                min_inliers_sift=min_inliers_sift,
                enable_phase_fallback=enable_phase_fallback,
                enable_ncc_fallback=enable_ncc_fallback,
                verify_with_ncc=verify_with_ncc,
                ncc_verify_gap=ncc_verify_gap,
                suppress_fixed_pattern=suppress_fixed_pattern,
                row_prior_dx=row_prior_est,
                expected_dx=expected_dx_est,
                tile_cache=tile_cache,
            )

            # Scale the estimated translation (and all NCC evidence) back
            # to FULL-resolution pixels.
            if est_scale != 1.0:
                dx_f, dy_f = result.translation
                result.translation = (dx_f / est_scale, dy_f / est_scale)
                if result.ncc_best is not None:
                    result.ncc_best = (result.ncc_best[0] / est_scale,
                                       result.ncc_best[1] / est_scale)
                if result.profile_translation is not None:
                    result.profile_translation = (
                        result.profile_translation[0] / est_scale,
                        result.profile_translation[1] / est_scale)
                # One quick full-resolution ECC polish recovers the
                # sub-pixel accuracy lost to the downscale quantization.
                if estimation_polish_full_res and result.success:
                    px, py, cc, ok = stitching.polish_translation_full_res(
                        tiles[i].image, tiles[i + 1].image,
                        result.translation[0], result.translation[1],
                        use_clahe=use_clahe, clahe_clip=clahe_clip,
                        tile_cache=tile_cache, n_features=n_features,
                    )
                    if ok:
                        result.translation = (px, py)
                        result.message += (f" | full-res ECC polish "
                                           f"({cc:.4f}) after downscale "
                                           f"estimation")

            if result.success:
                dx_history.append(result.translation[0])
            # When per-tile ROI bands differ, the estimation ran on
            # CROPPED tiles, so result.translation[1] (dy) is in the
            # CROPPED frame. Convert it to the FULL-tile frame for
            # diagnostics / drift guard / Pair Review consistency:
            #     dy_full = dy_cropped - (y0_{i+1} - y0_i)
            # (see the inverse conversion in Step 2 below). dx is
            # unaffected — the crop is vertical only.
            if roi_bands_per_tile is not None and result.success:
                dy_shift = (roi_bands_per_tile[i + 1][0]
                            - roi_bands_per_tile[i][0])
                dx_cur, dy_cur = result.translation
                result.translation = (dx_cur, dy_cur - dy_shift)
                if result.ncc_best is not None:
                    result.ncc_best = (result.ncc_best[0],
                                       result.ncc_best[1] - dy_shift)
            pair_results.append(PairDiagnostic(
                pair_index=i,
                method=result.method,
                success=result.success,
                translation=result.translation,
                rotation_deg=result.rotation_deg,
                scale=result.scale,
                matches=result.matches,
                inliers=result.inliers,
                message=result.message,
                score=result.score,
                ncc_best=result.ncc_best,
                ncc_best_score=result.ncc_best_score,
                profile_translation=result.profile_translation,
            ))

    # === Step 1b: drift-outlier guard ===
    warnings: List[str] = []
    drift_trend: Optional[Tuple[float, float]] = None
    n_pairs = n - 1

    # With user-reviewed positions the chain is intentionally kept exactly
    # as reviewed — no automatic correction is applied.
    # NOTE: the trend is computed over SUCCESSFUL pairs only. A failed pair
    # carries a (0, 0) placeholder translation; feeding it into the trend
    # both destroyed the trend fit (dxs > 50 fails -> guard disabled for
    # the whole row) and poisoned the fit itself.
    if enable_drift_guard and n_pairs >= 4 and not use_overrides:
        ok_idx = [i for i, d in enumerate(pair_results) if d.success]
        if len(ok_idx) >= 4:
            report("Checking vertical drift consistency...", 0.62)
            dxs = np.array([pair_results[i].translation[0] for i in ok_idx],
                           dtype=np.float64)
            dys = np.array([pair_results[i].translation[1] for i in ok_idx],
                           dtype=np.float64)

            if (dxs > 50).all():  # row progresses to the right — trend is meaningful
                trend = _robust_dy_trend(dxs, dys)
                if trend is not None:
                    slope, intercept, inlier_mask = trend
                    drift_trend = (slope, intercept)
                    pred = slope * dxs + intercept
                    resid = dys - pred
                    mad = float(np.median(np.abs(
                        resid[inlier_mask] - np.median(resid[inlier_mask]))))
                    sigma = 1.4826 * mad
                    thresh = max(35.0, 5.0 * sigma)

                    for local_i in np.where(np.abs(resid) > thresh)[0]:
                        local_i = int(local_i)
                        i = ok_idx[local_i]          # index in pair_results
                        d = pair_results[i]
                        dy_pred = float(pred[local_i])
                        old = d.translation
                        corrected: Optional[Tuple[float, float]] = None
                        source = ""

                        # (a) trust the pair's own NCC evidence when it matches the trend
                        if d.ncc_best is not None and abs(d.ncc_best[1] - dy_pred) <= 80.0 \
                                and d.ncc_best_score >= max(d.score, d.ncc_best_score) - 0.05:
                            if abs(d.ncc_best[0] - old[0]) > 3 or abs(d.ncc_best[1] - old[1]) > 3:
                                corrected = d.ncc_best
                                source = "NCC evidence"
                        # (b) constrained NCC re-scan around the trend prediction
                        elif d.ncc_best is not None:
                            resc = stitching.stitch_pair_ncc(
                                tiles[i].image, tiles[i + 1].image,
                                dy_range=(int(dy_pred) - 70, int(dy_pred) + 70),
                                use_clahe=use_clahe,
                                clahe_clip=clahe_clip,
                                tile_cache=tile_cache,
                                n_features=n_features,
                            )
                            if resc.success and \
                                    resc.ncc_best_score >= max(d.ncc_best_score, d.score) - 0.10:
                                corrected = resc.translation
                                source = "constrained NCC re-scan"

                        if corrected is not None:
                            d.translation = corrected
                            d.message += (f" | drift guard: dy corrected {old[1]:.1f} -> "
                                          f"{corrected[1]:.1f} (deviated {resid[local_i]:.0f} px "
                                          f"from trend dy~{dy_pred:.1f}; {source})")
                            warnings.append(
                                f"Pair {i+1}: dy corrected {old[1]:.1f} -> {corrected[1]:.1f} "
                                f"px (was {resid[local_i]:+.0f} px off the vertical drift trend; "
                                f"{source})")
                        elif d.ncc_best is None:
                            warnings.append(
                                f"Pair {i+1}: dy={old[1]:.1f} deviates {resid[local_i]:+.0f} px from "
                                f"the vertical drift trend (dy~{dy_pred:.1f}). Drift guard has "
                                f"no NCC evidence for this pair (verification disabled?) — "
                                f"kept as is.")
                        else:
                            warnings.append(
                                f"Pair {i+1}: dy={old[1]:.1f} deviates {resid[local_i]:+.0f} px from "
                                f"the vertical drift trend (dy~{dy_pred:.1f}). Kept as is — "
                                f"NCC does not confirm the trend here. Check this pair "
                                f"manually if the seam looks shifted.")

    # === Step 1c: failed-pair continuation ================================
    # A pair whose estimation FAILED carries a (0, 0) placeholder. Chaining
    # it verbatim placed the next tile EXACTLY ON TOP of the previous one
    # (double-exposed "broken" content) and shrank the canvas — the main
    # source of catastrophically corrupted 15+-tile panoramas. Instead the
    # chain continues by the row prior: dx = median of the accepted dxs
    # (falls back to one tile width), dy = the last accepted dy (drift is
    # smooth). The pair is flagged in the warnings and MUST be reviewed in
    # Pair Review.
    if not use_overrides:
        good = [d for d in pair_results if d.success]
        if len(good) < len(pair_results):
            med_dx = (float(np.median([d.translation[0] for d in good]))
                      if good else float(tiles[0].image.shape[1]))
            last_dy = (float(good[-1].translation[1]) if good else 0.0)
            for d in pair_results:
                if d.success:
                    continue
                d.translation = (med_dx, last_dy)
                d.message += (" | FAILED pair — chain continued by the row "
                              f"prior (dx={med_dx:.1f}, dy={last_dy:.1f}); "
                              "REVIEW THIS PAIR in Pair Review")
                warnings.append(
                    f"Pair {d.pair_index + 1}: FAILED ({d.message.split(' | ')[0]}) "
                    "— its position was CONTINUED by the row prior instead of "
                    "stacking the tiles on top of each other. The seam at this "
                    "pair is UNRELIABLE — verify it in Pair Review before "
                    "trusting the panorama.")

    # === Step 2: cumulative absolute positions ===
    report("Computing absolute tile positions...", 0.65)
    # Position of the first tile = (0, 0).
    # NOTE on coordinate frames: `pair_results` store translations in the
    # FULL-tile frame (consistent with Pair Review and the diagnostics
    # table). But the BLEND happens on the (possibly ROI-cropped) tiles,
    # so the cumulative positions below must be in the CROPPED frame when
    # per-tile bands differ:
    #     dy_cropped = dy_full + (y0_{i+1} - y0_i)
    # (see the forward conversion in the estimation loop above). For a
    # single roi_band (same y0 everywhere) or no ROI crop the shift is 0
    # and both frames coincide.
    abs_positions: List[Tuple[int, int]] = [(0, 0)]
    cum_x, cum_y = 0.0, 0.0
    for i, d in enumerate(pair_results):
        dy_for_blend = d.translation[1]
        if roi_bands_per_tile is not None and (i + 1) < n:
            dy_for_blend += (roi_bands_per_tile[i + 1][0]
                             - roi_bands_per_tile[i][0])
        cum_x += d.translation[0]
        cum_y += dy_for_blend
        abs_positions.append((int(round(cum_x)), int(round(cum_y))))

    # === Step 3: compute canvas size ===
    min_x = min(pos[0] for pos in abs_positions)
    min_y = min(pos[1] for pos in abs_positions)
    max_x_end = max(pos[0] + t.image.shape[1] for pos, t in zip(abs_positions, tiles))
    max_y_end = max(pos[1] + t.image.shape[0] for pos, t in zip(abs_positions, tiles))

    shift_x = -min_x
    shift_y = -min_y
    abs_positions = [(p[0] + shift_x, p[1] + shift_y) for p in abs_positions]
    canvas_w = max_x_end - min_x
    canvas_h = max_y_end - min_y

    # === Step 4: prepare PlacedTile and blend ===
    report(f"Assembling panorama {canvas_w}x{canvas_h} (multi-band blending)...", 0.7)
    # Tiles adjacent to a "no overlap" pair keep full weight on the gap-facing
    # edge (no feather falloff), so the black separator band has crisp borders.
    hard_lefts = [False] * n
    hard_rights = [False] * n
    if use_overrides:
        eff_methods = [pair_methods[i] if pair_methods else "reviewed"
                       for i in range(n_pairs)]
        for i, m in enumerate(eff_methods):
            if m == "no-overlap":
                hard_rights[i] = True
                hard_lefts[i + 1] = True
    placed = [
        blending.PlacedTile(
            image=t.image, offset_x=p[0], offset_y=p[1],
            hard_edges=(hard_lefts[k], hard_rights[k], False, False),
        )
        for k, (t, p) in enumerate(zip(tiles, abs_positions))
    ]
    repair_stats: dict = {}
    panorama = blending.blend_tiles(
        placed,
        canvas_size=(canvas_h, canvas_w),
        levels=blend_levels,
        feather=blend_feather,
        block_size=blend_block_size,
        gamma_correction=blend_gamma,
        gain_strength=blend_gain_strength,
        use_focus_weights=blend_use_focus_weights,
        focus_priority=blend_focus_priority,
        focus_priority_gamma=blend_focus_priority_gamma,
        focus_priority_levels=blend_focus_priority_levels,
        repair_stats=repair_stats,
    )
    if repair_stats.get("gap_px_filled"):
        warnings.append(
            f"Artifact repair: {repair_stats['gap_px_filled']} uncovered "
            "seam-sliver pixel(s) between tiles were filled by interpolation "
            "(thin black/grey streaks from integer position rounding).")
    if repair_stats.get("outlier_px_repaired"):
        warnings.append(
            f"Artifact repair: {repair_stats['outlier_px_repaired']} isolated "
            "broken pixel(s) inside the stitched area were replaced by their "
            "3×3 median value.")

    report("Done!", 1.0)

    # Collect warnings
    for d in pair_results:
        if not d.success:
            # Already reported (with the continuation explanation) by the
            # failed-pair continuation block above — do not duplicate.
            continue
        elif d.method == "ncc" and "rejected" not in d.message:
            warnings.append(f"Pair {d.pair_index+1}: SIFT failed, used NCC scan "
                            f"({d.message})")
        elif d.method == "phase":
            warnings.append(f"Pair {d.pair_index+1}: SIFT failed, used phase "
                            f"correlation ({d.message})")

    # Optional debug full-resolution panorama (without ROI crop). Only valid
    # in the `roi_band` (single-band) case where pairwise translations are
    # identical in the cropped and full-tile coordinate frames. In the
    # `roi_bands_per_tile` case the per-tile dy values refer to per-tile
    # cropped frames that do NOT line up across tiles, so the resulting
    # panorama would be wrong — we skip it and emit a warning instead.
    panorama_full: Optional[np.ndarray] = None
    if keep_full_panorama and roi_bands_used is not None:
        if roi_bands_per_tile is not None:
            warnings.append(
                "keep_full_panorama is ignored when roi_bands_per_tile is "
                "set: per-tile dy is in per-tile cropped frames, so the "
                "translations do not transfer to the full-tile canvas. "
                "Use roi_band (single band for all tiles) if you need a "
                "debug full panorama.")
        else:
            report("Building debug full-resolution panorama (no ROI crop)...",
                   0.92)
            try:
                panorama_full = _blend_with_translations(
                    original_tiles, pair_results, use_overrides,
                    pair_methods, n_pairs,
                    blend_levels=blend_levels,
                    blend_feather=blend_feather,
                    blend_block_size=blend_block_size,
                    blend_gamma=blend_gamma,
                    blend_gain_strength=blend_gain_strength,
                    blend_use_focus_weights=blend_use_focus_weights,
                    blend_focus_priority=blend_focus_priority,
                    blend_focus_priority_gamma=blend_focus_priority_gamma,
                    blend_focus_priority_levels=blend_focus_priority_levels,
                )
            except Exception as e:
                warnings.append(f"keep_full_panorama failed: {e}")

    return StitchResult(
        panorama=panorama,
        canvas_size=(canvas_h, canvas_w),
        tile_positions=abs_positions,
        diagnostics=pair_results,
        n_tiles=n,
        n_pairs=n_pairs,
        warnings=warnings,
        drift_trend=drift_trend,
        panorama_full=panorama_full,
        roi_bands_used=roi_bands_used,
    )


def _blend_with_translations(
    tiles: List[io_utils.TileInfo],
    pair_results: List[PairDiagnostic],
    use_overrides: bool,
    pair_methods: Optional[List[str]],
    n_pairs: int,
    *,
    blend_levels: int,
    blend_feather: int,
    blend_block_size: int,
    blend_gamma: float,
    blend_gain_strength: float,
    blend_use_focus_weights: bool,
    blend_focus_priority: bool,
    blend_focus_priority_gamma: float,
    blend_focus_priority_levels: int,
) -> np.ndarray:
    """Re-run only the blend stage on a different set of tile images using
    the SAME pairwise translations. Used by `keep_full_panorama` to build a
    full-resolution debug panorama after the ROI-cropped panorama has been
    produced. The translations must already be in the coordinate frame of
    the provided `tiles` (which is true for the `roi_band` single-band case
    where every tile is cropped by the same [y0, y1) window).
    """
    n = len(tiles)
    abs_positions: List[Tuple[int, int]] = [(0, 0)]
    cum_x, cum_y = 0.0, 0.0
    for d in pair_results:
        cum_x += d.translation[0]
        cum_y += d.translation[1]
        abs_positions.append((int(round(cum_x)), int(round(cum_y))))

    min_x = min(pos[0] for pos in abs_positions)
    min_y = min(pos[1] for pos in abs_positions)
    max_x_end = max(pos[0] + t.image.shape[1]
                    for pos, t in zip(abs_positions, tiles))
    max_y_end = max(pos[1] + t.image.shape[0]
                    for pos, t in zip(abs_positions, tiles))
    shift_x = -min_x
    shift_y = -min_y
    abs_positions = [(p[0] + shift_x, p[1] + shift_y) for p in abs_positions]
    canvas_w = max_x_end - min_x
    canvas_h = max_y_end - min_y

    hard_lefts = [False] * n
    hard_rights = [False] * n
    if use_overrides:
        eff_methods = [pair_methods[i] if pair_methods else "reviewed"
                       for i in range(n_pairs)]
        for i, m in enumerate(eff_methods):
            if m == "no-overlap":
                hard_rights[i] = True
                hard_lefts[i + 1] = True
    placed = [
        blending.PlacedTile(
            image=t.image, offset_x=p[0], offset_y=p[1],
            hard_edges=(hard_lefts[k], hard_rights[k], False, False),
        )
        for k, (t, p) in enumerate(zip(tiles, abs_positions))
    ]
    return blending.blend_tiles(
        placed,
        canvas_size=(canvas_h, canvas_w),
        levels=blend_levels,
        feather=blend_feather,
        block_size=blend_block_size,
        gamma_correction=blend_gamma,
        gain_strength=blend_gain_strength,
        use_focus_weights=blend_use_focus_weights,
        focus_priority=blend_focus_priority,
        focus_priority_gamma=blend_focus_priority_gamma,
        focus_priority_levels=blend_focus_priority_levels,
    )


def save_panorama(panorama: np.ndarray, out_path: Path, format: str = "auto") -> Path:
    """Save the panorama. format: 'png' | 'tiff' | 'jpg' | 'auto' (by extension)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower().lstrip(".")
    if ext in ("tif", "tiff"):
        # Safe exporter (io_utils): LZW only when TWO independent codecs
        # (tifffile + OpenCV/libtiff) verify the pixels; otherwise the
        # panorama is written uncompressed — "broken pixels" in some
        # viewers can no longer be produced by this path either.
        rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        data = io_utils.encode_tiff_rgb(rgb, compress=True)
        if data is None:
            data = io_utils.encode_tiff_rgb(rgb, compress=False)
        if data is None:
            raise RuntimeError("TIFF encoding failed the pixel-exact "
                               "verification — save as PNG instead.")
        out_path.write_bytes(data)
    else:
        cv2.imwrite(str(out_path), panorama)
    return out_path
