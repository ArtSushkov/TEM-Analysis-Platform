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
from typing import Callable, List, Optional, Tuple

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
    n_features: int = 5000,
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
    blend_levels: int = 5,
    blend_feather: int = 64,
    blend_block_size: int = 16,
    blend_gamma: float = 1.5,
    blend_gain_strength: float = 0.8,
    blend_use_focus_weights: bool = False,
    pair_translations: Optional[List[Tuple[float, float]]] = None,
    pair_methods: Optional[List[str]] = None,
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

    def report(msg, p):
        if progress_callback:
            progress_callback(msg, p)

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

        for i in range(n - 1):
            report(f"Pair {i+1}/{n-1}: tile {tiles[i].name} -> {tiles[i+1].name}",
                   0.03 + 0.55 * (i / max(1, n - 2)))
            # dx prior: median of already-solved pairs (>= 2), else the user's
            # expected overlap. Robust to one bad pair among several good ones.
            row_prior_dx = (float(np.median(dx_history))
                            if len(dx_history) >= 2 else None)
            result = stitching.stitch_pair(
                tiles[i].image, tiles[i + 1].image,
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
                row_prior_dx=row_prior_dx,
                expected_dx=expected_dx,
            )
            if result.success:
                dx_history.append(result.translation[0])
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
    if enable_drift_guard and n_pairs >= 4 and not use_overrides:
        report("Checking vertical drift consistency...", 0.62)
        dxs = np.array([d.translation[0] for d in pair_results], dtype=np.float64)
        dys = np.array([d.translation[1] for d in pair_results], dtype=np.float64)

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

                for i in np.where(np.abs(resid) > thresh)[0]:
                    i = int(i)
                    d = pair_results[i]
                    dy_pred = float(pred[i])
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
                        )
                        if resc.success and \
                                resc.ncc_best_score >= max(d.ncc_best_score, d.score) - 0.10:
                            corrected = resc.translation
                            source = "constrained NCC re-scan"

                    if corrected is not None:
                        d.translation = corrected
                        d.message += (f" | drift guard: dy corrected {old[1]:.1f} -> "
                                      f"{corrected[1]:.1f} (deviated {resid[i]:.0f} px "
                                      f"from trend dy~{dy_pred:.1f}; {source})")
                        warnings.append(
                            f"Pair {i+1}: dy corrected {old[1]:.1f} -> {corrected[1]:.1f} "
                            f"px (was {resid[i]:+.0f} px off the vertical drift trend; "
                            f"{source})")
                    elif d.ncc_best is None:
                        warnings.append(
                            f"Pair {i+1}: dy={old[1]:.1f} deviates {resid[i]:+.0f} px from "
                            f"the vertical drift trend (dy~{dy_pred:.1f}). Drift guard has "
                            f"no NCC evidence for this pair (verification disabled?) — "
                            f"kept as is.")
                    else:
                        warnings.append(
                            f"Pair {i+1}: dy={old[1]:.1f} deviates {resid[i]:+.0f} px from "
                            f"the vertical drift trend (dy~{dy_pred:.1f}). Kept as is — "
                            f"NCC does not confirm the trend here. Check this pair "
                            f"manually if the seam looks shifted.")

    # === Step 2: cumulative absolute positions ===
    report("Computing absolute tile positions...", 0.65)
    # Position of the first tile = (0, 0)
    abs_positions: List[Tuple[int, int]] = [(0, 0)]
    cum_x, cum_y = 0.0, 0.0
    for d in pair_results:
        cum_x += d.translation[0]
        cum_y += d.translation[1]
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
    panorama = blending.blend_tiles(
        placed,
        canvas_size=(canvas_h, canvas_w),
        levels=blend_levels,
        feather=blend_feather,
        block_size=blend_block_size,
        gamma_correction=blend_gamma,
        gain_strength=blend_gain_strength,
        use_focus_weights=blend_use_focus_weights,
    )

    report("Done!", 1.0)

    # Collect warnings
    for d in pair_results:
        if not d.success:
            warnings.append(f"Pair {d.pair_index+1}: failed ({d.message})")
        elif d.method == "ncc" and "rejected" not in d.message:
            warnings.append(f"Pair {d.pair_index+1}: SIFT failed, used NCC scan "
                            f"({d.message})")
        elif d.method == "phase":
            warnings.append(f"Pair {d.pair_index+1}: SIFT failed, used phase "
                            f"correlation ({d.message})")

    return StitchResult(
        panorama=panorama,
        canvas_size=(canvas_h, canvas_w),
        tile_positions=abs_positions,
        diagnostics=pair_results,
        n_tiles=n,
        n_pairs=n_pairs,
        warnings=warnings,
        drift_trend=drift_trend,
    )


def save_panorama(panorama: np.ndarray, out_path: Path, format: str = "auto") -> Path:
    """Save the panorama. format: 'png' | 'tiff' | 'jpg' | 'auto' (by extension)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower().lstrip(".")
    if ext in ("tif", "tiff"):
        # Save via tifffile for 16-bit and metadata
        import tifffile
        rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        tifffile.imwrite(str(out_path), rgb, compression="lzw")
    else:
        cv2.imwrite(str(out_path), panorama)
    return out_path
