"""
io_utils.py — Loading and sorting microscope images.

Filename pattern: <prefix>_<N>.<ext>, where N is the sequential index in the panorama.
Example: x10_1.tif, x10_2.tif, ..., x10_5.tif
"""
from __future__ import annotations

import re
from dataclasses import dataclass
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
