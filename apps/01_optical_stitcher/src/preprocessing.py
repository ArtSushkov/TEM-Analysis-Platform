"""
preprocessing.py — Image preprocessing before feature detection,
and sharpness map computation for focus-weighted blending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a BGR image.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) is critical
    for dark microscope images — it increases the number of detected
    keypoints by 5-10x.

    Returns: BGR image of the same size, dtype=uint8.
    """
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpness_map(img: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Local sharpness map (Laplacian variance per block).

    Used for focus-weighted blending: pixels from sharper
    blocks get higher weight during blending. Solves the problem
    of blurred tile edges in microscopy.

    Returns: float32 array of the same HxW as input.
    """
    gray = to_gray(img).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = lap * lap  # squared deviation

    h, w = gray.shape
    # Average over block_size blocks
    # For each pixel, take the average energy in its block
    kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size * block_size)
    sharp = cv2.filter2D(energy, -1, kernel, borderType=cv2.BORDER_REFLECT)

    # Additional light smoothing to avoid noisy weights
    sharp = cv2.GaussianBlur(sharp, (0, 0), sigmaX=block_size / 2.0)

    # Normalize to [0, 1]
    smin, smax = sharp.min(), sharp.max()
    if smax > smin:
        sharp = (sharp - smin) / (smax - smin)
    else:
        sharp = np.zeros_like(sharp)
    return sharp.astype(np.float32)


def feather_mask(
    img: np.ndarray,
    edge_feather: int = 64,
) -> np.ndarray:
    """Simple mask with smooth falloff toward edges.

    Used as DEFAULT for blending — works robustly
    in zones with a single tile (no overlap), does not produce artifacts
    on dark microscope images.

    Returns: float32 HxW, values in [0, 1].
    """
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    if edge_feather > 0:
        ef = min(edge_feather, h // 2, w // 2)
        if ef > 0:
            ramp = np.linspace(0, 1, ef, dtype=np.float32)
            # Falloff toward each side
            mask[:ef, :] *= ramp[:, None]
            mask[-ef:, :] *= ramp[::-1][:, None]
            mask[:, :ef] *= ramp[None, :]
            mask[:, -ef:] *= ramp[::-1][None, :]
    return mask


def focus_weight_mask(
    img: np.ndarray,
    block_size: int = 16,
    edge_feather: int = 64,
    base_weight: float = 0.3,
    gamma: float = 1.5,
) -> np.ndarray:
    """Focus-weighted mask for blending.

    Combines:
    1. Sharpness map (Laplacian variance) with gamma contrast
    2. Smooth falloff toward edges (feather)
    3. Base weight (important!): guarantees that in a single-tile zone
       without overlap, blending does not zero out pixels. Without this
       protection, dark areas at the panorama edges become fully black.

    base_weight: minimum weight in the tile center (0.3 = 30% of max).
        Increase to 0.5-0.7 for more stability.
        Set to 0.0 for pure focus-weighted blending (legacy behavior).

    Returns: float32 HxW, values in [0, 1].
    """
    sharp = sharpness_map(img, block_size=block_size)
    sharp = sharp ** gamma

    # Add base weight: even in zones with poor sharpness
    # the weight does not drop below base_weight * (feather mask)
    feather = feather_mask(img, edge_feather=edge_feather)
    combined = np.maximum(sharp, base_weight) * feather
    # Normalize to [0, 1]
    cmax = combined.max()
    if cmax > 1e-6:
        combined = combined / cmax
    return combined.astype(np.float32)
