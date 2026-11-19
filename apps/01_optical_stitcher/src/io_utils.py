"""
io_utils.py — Loading and sorting microscope images.

Filename pattern: <prefix>_<N>.<ext>, where N is the sequential index in the panorama.
Example: x10_1.tif, x10_2.tif, ..., x10_5.tif
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tifffile


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class TileInfo:
    """Metadata of a single tile image."""
    path: Path
    index: int            # sequential index extracted from name (N from _N)
    name: str             # filename without extension
    image: np.ndarray     # source image in BGR format (OpenCV convention)


def _extract_index(name: str) -> Optional[int]:
    """Extract N from a name like 'x10_5' or 'sample_12.tif'.

    Returns the largest numeric suffix after the last underscore.
    """
    stem = Path(name).stem
    # Look for the final _N pattern (N is an integer)
    m = re.search(r"_(\d+)$", stem)
    if m:
        return int(m.group(1))
    # Fallback: any number at the end of the name
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def load_image(path: Path) -> np.ndarray:
    """Load an image as BGR uint8.

    Uses tifffile for TIFF (including LZW compression), OpenCV for other formats.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        img = tifffile.imread(str(path))
        # Normalize to BGR uint8
        img = _normalize_to_bgr_uint8(img)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
    return img


def _normalize_to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an array to BGR uint8.

    Supports: grayscale, RGB, RGBA, 8/16-bit.
    """
    # 16-bit -> 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = (img * 255).astype(np.uint8)

    # Channels -> BGR
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # RGB -> BGR (tifffile returns RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def discover_tiles(folder: Path) -> List[Path]:
    """Find all supported image files in a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    # Deduplicate (case-insensitive filesystems)
    seen = set()
    unique = []
    for f in files:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def load_tiles(folder: Path) -> List[TileInfo]:
    """Load all tiles from a folder, sorted by their _N index.

    Ensures that every filename has an N index and that indices are unique.
    """
    files = discover_tiles(folder)
    if not files:
        raise FileNotFoundError(f"No supported images found in {folder}")

    tiles: List[TileInfo] = []
    indices: List[int] = []
    for f in files:
        idx = _extract_index(f.name)
        if idx is None:
            raise ValueError(
                f"Failed to extract sequential index from filename: {f.name}. "
                f"Expected format: <name>_N.<ext> (e.g. sample_1.tif)"
            )
        if idx in indices:
            raise ValueError(
                f"Duplicate index {idx} in file {f.name}. "
                f"Indices _N must be unique."
            )
        img = load_image(f)
        tiles.append(TileInfo(path=f, index=idx, name=f.stem, image=img))
        indices.append(idx)

    # Sort by index
    tiles.sort(key=lambda t: t.index)
    return tiles


def load_tiles_from_paths(paths: List[Path]) -> List[TileInfo]:
    """Alternative entry point: load from an explicit list of paths."""
    tiles: List[TileInfo] = []
    indices: List[int] = []
    for p in paths:
        p = Path(p)
        idx = _extract_index(p.name)
        if idx is None:
            raise ValueError(f"No index in filename: {p.name}")
        if idx in indices:
            raise ValueError(f"Duplicate index {idx}: {p.name}")
        img = load_image(p)
        tiles.append(TileInfo(path=p, index=idx, name=p.stem, image=img))
        indices.append(idx)
    tiles.sort(key=lambda t: t.index)
    return tiles


# --------------------------------------------------------------------------
# Region of interest (ROI) — vertical band crop, applied per tile BEFORE
# stitching so the whole pipeline (SIFT, NCC, blending) runs on a smaller
# image. Saves ~Nx time and memory where N = (tile height / ROI height).
# --------------------------------------------------------------------------

def auto_detect_roi_band(
    img: np.ndarray,
    target_fraction: float = 0.30,
    min_fraction: float = 0.15,
    max_fraction: float = 0.50,
) -> Tuple[int, int]:
    """Auto-detect a horizontal ROI band that brackets the most textured
    region of the tile (the visible part of the TEM disk cross-section).

    Strategy (v2 — robust anchor + centred window):
    - Per-row Tenengrad energy (sum of squared Sobel responses), lightly
      smoothed. Rows inside the disk have high gradient energy; rows in
      the empty microscope background above/below have almost none.
    - STEP 1 (anchor): a sliding window of height `target_fraction * H`
      picks the position that CONTAINS THE MOST ENERGY — the same robust
      selection as v1: a bright dust speck outside the disk carries far
      less TOTAL energy than the disk itself, so the anchor snaps onto
      the disk instead of the speck.
    - STEP 2 (centering): the window is then placed SYMMETRICALLY around
      the energy centroid of the clearly-textured rows in a generous
      neighbourhood of the anchor (row energies above a noise floor
      derived from the typical disk-row energy, extremes clipped at
      3x that scale so a bright scratch or speck cannot pull the
      centroid). The disk band therefore ends up in the MIDDLE of the
      crop on every tile — v1 kept the window's TOP edge at the first
      fully-covering position, which pushed the disk to the bottom edge
      of the crop whenever the disk band is narrower than the window.
    - The window height is clamped to [min_fraction, max_fraction] of
      the tile height so the ROI stays meaningful (not a sliver, not the
      whole frame).
    - Falls back to a centered band of height target_fraction * H when
      the energy profile is flat (uniform noise / blank frame).

    Because the window height is FIXED by the caller, the band ALWAYS
    has the same height on every tile (important for the "reference
    tile" workflow where tile 1's band is reused for the whole row).
    Only the vertical POSITION is detected per tile.

    Returns (y0, y1) in pixels, y0 < y1, both within [0, H).
    """
    h = int(img.shape[0])
    if h < 4:
        return (0, h)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = (gx * gx + gy * gy).sum(axis=1)   # per-row total gradient energy
    # Smooth across rows so a single bright scratch cannot dominate
    k = max(11, h // 60)
    if k % 2 == 0:
        k += 1
    energy = cv2.GaussianBlur(energy[:, None], (1, k), 0)[:, 0]

    total = float(energy.sum())
    h_roi = int(round(min(max_fraction * h,
                          max(min_fraction * h, target_fraction * h))))
    h_roi = max(1, min(h, h_roi))
    if total <= 0.0 or h_roi >= h:
        # Flat / blank / band covers the whole frame — centered band
        y0 = max(0, (h - h_roi) // 2)
        return (y0, min(h, y0 + h_roi))

    e = energy.astype(np.float64)

    # STEP 1 (anchor): best window of height h_roi via the O(h) cumsum
    # trick — the same robust selection as v1 (robust to specks far
    # outside the disk, which have much lower TOTAL energy).
    cum = np.concatenate([[0.0], np.cumsum(e)])
    sums = cum[h_roi:] - cum[:h - h_roi + 1]
    if len(sums) == 0:
        y0 = max(0, (h - h_roi) // 2)
        return (y0, min(h, y0 + h_roi))
    w_start = int(np.argmax(sums))
    best_window = float(sums[w_start])

    # STEP 2 (centering): centroid of the clearly-textured rows in a
    # generous neighbourhood of the anchor. scale = typical disk-row
    # energy; rows must exceed a fraction of it to count as "textured"
    # (the median guard handles frames where the disk fills most of the
    # field of view), and extremes are clipped at 3x the scale so one
    # very bright row cannot pull the centroid away from the disk band.
    scale = best_window / max(1, h_roi)
    thr = max(scale * 0.15, float(np.median(e)) * 1.5, 1e-12)
    cap = scale * 3.0
    lo = max(0, w_start - h_roi // 2)
    hi = min(h, w_start + h_roi + h_roi // 2)
    idx = np.arange(lo, hi, dtype=np.float64)
    wgt = np.clip(np.minimum(e[lo:hi], cap) - thr, 0.0, None)
    if float(wgt.sum()) <= 0.0:
        # Nothing stands out above the noise floor — centered band
        y0 = max(0, (h - h_roi) // 2)
        return (y0, min(h, y0 + h_roi))
    y_center = float((idx * wgt).sum() / float(wgt.sum()))
    # Center the window on the centroid (do NOT stick the top edge of
    # the window to the first textured row) so the disk stays in the
    # middle of the crop.
    y0 = int(round(y_center - h_roi / 2.0))
    y0 = max(0, min(h - h_roi, y0))
    return (y0, y0 + h_roi)


def crop_tile_to_roi_band(
    tile: TileInfo,
    y0: int,
    y1: int,
) -> TileInfo:
    """Return a NEW TileInfo whose `image` is the vertical band [y0, y1)
    of the original. Path / index / name are preserved so diagnostics and
    the Pair Review UI keep referring to the same tile identity.

    Width is preserved: only rows are cropped, so horizontal overlap
    between adjacent tiles is unchanged and pairwise SIFT/NCC alignment
    is unaffected. Vertical drift / dy translations computed on cropped
    tiles apply to the cropped coordinate frame; for the `roi_band`
    case (same band for every tile) they are also valid in the full-tile
    frame, so the optional "keep full panorama" debug output is exact.
    """
    h = int(tile.image.shape[0])
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(y0 + 1, min(h, int(y1)))
    return replace(tile, image=tile.image[y0:y1, :, :].copy())


def roi_band_for_tiles(
    tiles: List[TileInfo],
    mode: str = "auto",
    target_fraction: float = 0.25,
    offset_fraction: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """Compute a per-tile ROI band (y0, y1) in pixels.

    mode:
      "auto"   — `auto_detect_roi_band` on each tile independently. Each
                 tile gets its own band; if the operator shifted the disk
                 vertically between shots, every tile still keeps the
                 disk inside its band. This is the default for the
                 Pair Review workflow where per-tile offset correction
                 is also offered.
      "center" — same centered band on every tile:
                 y0 = H * offset_fraction, y1 = H * (offset_fraction +
                 target_fraction). Used when the operator kept the disk
                 vertically aligned across the row (e.g. motorized
                 stage with locked Y).

    offset_fraction is only consulted in "center" mode (default 0.375
    gives a centered band of height 0.25 * H — the central quarter).

    Returns a list of (y0, y1) tuples, one per tile.
    """
    out: List[Tuple[int, int]] = []
    if mode == "auto":
        for t in tiles:
            out.append(auto_detect_roi_band(
                t.image,
                target_fraction=target_fraction,
                min_fraction=max(0.10, target_fraction - 0.10),
                max_fraction=min(1.0, target_fraction + 0.25),
            ))
        return out
    # center
    if offset_fraction is None:
        offset_fraction = max(0.0, 0.5 - target_fraction / 2.0)
    for t in tiles:
        h = int(t.image.shape[0])
        y0 = max(0, int(round(h * offset_fraction)))
        y1 = min(h, int(round(h * (offset_fraction + target_fraction))))
        y1 = max(y0 + 1, y1)
        out.append((y0, y1))
    return out
