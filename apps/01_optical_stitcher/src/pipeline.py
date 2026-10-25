"""
pipeline.py — Orchestrating the entire tile-row stitching process.

Steps:
1. Load tiles from a folder, sort by _N index
2. Estimate pairwise shifts between adjacent tiles
3. Compute cumulative absolute positions on the final canvas
4. Focus-weighted multi-band blending
5. Return the result + diagnostics
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
    blend_levels: int = 5,
    blend_feather: int = 64,
    blend_block_size: int = 16,
    blend_gamma: float = 1.5,
    blend_gain_strength: float = 0.8,
    blend_use_focus_weights: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> StitchResult:
    """Stitch a row of tiles into a single panorama.

    tiles: list of TileInfo, already sorted by index.
    progress_callback: optional (message, progress_0_to_1).

    Returns: StitchResult with panorama, positions, and diagnostics.
    """
    n = len(tiles)
    if n < 2:
        raise ValueError(f"At least 2 tiles are required for stitching, got: {n}")

    def report(msg, p):
        if progress_callback:
            progress_callback(msg, p)

    # === Step 1: pairwise stitching ===
    report(f"Analyzing {n-1} adjacent pairs...", 0.05)
    pair_results: List[PairDiagnostic] = []
    relative_shifts: List[Tuple[float, float]] = []  # shifts of each next tile relative to the previous

    for i in range(n - 1):
        report(f"Pair {i+1}/{n-1}: tile {tiles[i].name} -> {tiles[i+1].name}",
               0.05 + 0.45 * (i / max(1, n - 2)))
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
        )
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
        ))
        if not result.success:
            # Fallback: still use the result, but flag a warning
            pass
        # Shift of tile i+1 relative to tile i:
        # translation = (dx, dy) — the shift to apply to b to get a.
        # So b is shifted relative to a by (+dx, +dy) in a's coordinates.
        # Therefore, the absolute position of tile i+1 = position of tile i + (dx, dy).
        relative_shifts.append(result.translation)

    # === Step 2: cumulative absolute positions ===
    report("Computing absolute tile positions...", 0.55)
    # Position of the first tile = (0, 0)
    abs_positions: List[Tuple[int, int]] = [(0, 0)]
    cum_x, cum_y = 0.0, 0.0
    for (dx, dy) in relative_shifts:
        cum_x += dx
        cum_y += dy
        abs_positions.append((int(round(cum_x)), int(round(cum_y))))

    # === Step 3: compute canvas size ===
    # Account for each tile's size + its offset
    min_x = min(pos[0] for pos in abs_positions)
    min_y = min(pos[1] for pos in abs_positions)
    max_x_end = max(pos[0] + t.image.shape[1] for pos, t in zip(abs_positions, tiles))
    max_y_end = max(pos[1] + t.image.shape[0] for pos, t in zip(abs_positions, tiles))

    # Shift all positions so that min is (0, 0)
    shift_x = -min_x
    shift_y = -min_y
    abs_positions = [(p[0] + shift_x, p[1] + shift_y) for p in abs_positions]
    canvas_w = max_x_end - min_x
    canvas_h = max_y_end - min_y

    # === Step 4: prepare PlacedTile and blend ===
    report(f"Assembling panorama {canvas_w}x{canvas_h} (multi-band blending)...", 0.65)
    placed = [
        blending.PlacedTile(image=t.image, offset_x=p[0], offset_y=p[1])
        for t, p in zip(tiles, abs_positions)
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
    warnings = []
    for d in pair_results:
        if not d.success:
            warnings.append(f"Pair {d.pair_index+1}: failed ({d.message})")
        elif d.method != "sift" or d.inliers < min_inliers_sift:
            warnings.append(f"Pair {d.pair_index+1}: used {d.method}, "
                            f"inliers={d.inliers} ({d.message})")

    return StitchResult(
        panorama=panorama,
        canvas_size=(canvas_h, canvas_w),
        tile_positions=abs_positions,
        diagnostics=pair_results,
        n_tiles=n,
        n_pairs=n - 1,
        warnings=warnings,
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
