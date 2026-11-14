"""
polygon.py — Polygon-based layer selection.

Provides:
* ``polygon_to_mask`` — rasterise a polygon into a boolean mask.
* ``mask_to_polygon`` — extract the largest contour from a mask as a polygon.
* ``crop_to_mask_bbox`` — crop an image to a mask's bounding box.
* ``apply_mask_to_image`` — set pixels outside the mask to a fill value.
* ``polygon_area_nm2`` — compute the polygon area in nm².
* ``polygon_centroid`` — compute the polygon centroid.
* ``polygon_bbox`` — compute the polygon bounding box.

Polygons are represented as Nx2 arrays of (row, col) float coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def polygon_to_mask(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon into a boolean mask.

    Parameters
    ----------
    polygon : np.ndarray (N, 2)
        Polygon vertices as (row, col) coordinates.
    shape : (height, width)
        Output mask shape.

    Returns
    -------
    np.ndarray of bool, shape (H, W)
    """
    if polygon is None or len(polygon) < 3:
        return np.zeros(shape, dtype=bool)
    # cv2.fillPoly expects (N, 1, 2) of (col, row) = (x, y).
    pts = polygon[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.005) -> np.ndarray | None:
    """Extract the largest contour from a mask as a simplified polygon.

    Parameters
    ----------
    mask : np.ndarray of bool
    epsilon_frac : float
        Simplification strength as a fraction of the contour perimeter.

    Returns
    -------
    np.ndarray (N, 2) of (row, col), or None if no contour found.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    # Convert (col, row) → (row, col).
    return approx.reshape(-1, 2)[:, [1, 0]].astype(np.float32)


def crop_to_mask_bbox(image: np.ndarray, mask: np.ndarray,
                       padding: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop an image and mask to the mask's bounding box.

    Parameters
    ----------
    image : np.ndarray
    mask : np.ndarray of bool
    padding : int
        Extra pixels to include on each side of the bounding box.

    Returns
    -------
    (cropped_image, cropped_mask, bbox)
        bbox = (min_row, min_col, max_row, max_col) in original coordinates.
    """
    if not mask.any():
        return image, mask, (0, 0, image.shape[0], image.shape[1])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Apply padding.
    rmin = max(0, rmin - padding)
    rmax = min(image.shape[0], rmax + 1 + padding)
    cmin = max(0, cmin - padding)
    cmax = min(image.shape[1], cmax + 1 + padding)
    return (image[rmin:rmax, cmin:cmax],
            mask[rmin:rmax, cmin:cmax],
            (int(rmin), int(cmin), int(rmax), int(cmax)))


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray,
                         fill_value: float = 0.0) -> np.ndarray:
    """Return a copy of the image with pixels outside the mask set to fill_value."""
    out = image.astype(np.float64).copy()
    out[~mask] = fill_value
    if image.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(image.dtype)


def polygon_area_nm2(polygon: np.ndarray, pixel_size_nm: float) -> float:
    """Compute the polygon area in nm² using the shoelace formula.

    Parameters
    ----------
    polygon : np.ndarray (N, 2) of (row, col)
    pixel_size_nm : float
        Pixel size in nm/px.

    Returns
    -------
    float
        Area in nm².
    """
    if polygon is None or len(polygon) < 3:
        return 0.0
    # Shoelace formula on (row, col) coordinates.
    rows = polygon[:, 0]
    cols = polygon[:, 1]
    # Area in pixel².
    area_px2 = 0.5 * abs(np.dot(rows, np.roll(cols, -1)) - np.dot(cols, np.roll(rows, -1)))
    return float(area_px2 * pixel_size_nm * pixel_size_nm)


def polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
    """Compute the polygon centroid (row, col) using the area-weighted formula.

    Parameters
    ----------
    polygon : np.ndarray (N, 2) of (row, col)

    Returns
    -------
    (centroid_row, centroid_col)
    """
    if polygon is None or len(polygon) < 3:
        return (0.0, 0.0)
    rows = polygon[:, 0]
    cols = polygon[:, 1]
    # Cross products.
    cross = rows * np.roll(cols, -1) - np.roll(rows, -1) * cols
    area2 = cross.sum()  # 2 * signed area
    if abs(area2) < 1e-9:
        # Degenerate polygon → return vertex mean.
        return (float(rows.mean()), float(cols.mean()))
    cr = (rows * cross).sum() / area2 / 3.0 + (np.roll(rows, -1) * cross).sum() / area2 / 3.0
    cc = (cols * cross).sum() / area2 / 3.0 + (np.roll(cols, -1) * cross).sum() / area2 / 3.0
    # Simpler: use the standard centroid formula.
    cr = (rows + np.roll(rows, -1)) * cross
    cc = (cols + np.roll(cols, -1)) * cross
    cr_sum = cr.sum() / (3.0 * area2)
    cc_sum = cc.sum() / (3.0 * area2)
    return (float(cr_sum), float(cc_sum))


def polygon_bbox(polygon: np.ndarray) -> tuple[int, int, int, int]:
    """Compute the polygon's axis-aligned bounding box.

    Returns
    -------
    (min_row, min_col, max_row, max_col)
    """
    if polygon is None or len(polygon) < 1:
        return (0, 0, 0, 0)
    rows = polygon[:, 0]
    cols = polygon[:, 1]
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


# === LayerMask: unified representation of a selected layer ===

@dataclass
class LayerMask:
    """Unified representation of a selected layer.

    A layer is defined by a boolean mask of the same shape as the source
    image. The mask can come from:
      * SAM automatic segmentation (``source = "sam"``)
      * A user-drawn polygon (``source = "polygon"``)
      * A rectangular row range (``source = "manual_rows"`` — backward compat)

    The polygon representation is also stored (for JSON serialisation and
    display), and the source-image bounding box for cropping.
    """
    mask: np.ndarray                  # bool, shape (H, W) of the source image
    polygon: np.ndarray               # (N, 2) of (row, col), float
    bbox: tuple[int, int, int, int]   # (min_row, min_col, max_row, max_col)
    source: str                       # "sam" | "polygon" | "manual_rows"
    # Metadata for display.
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    area_px: int = 0
    area_nm2: float = 0.0
    height_nm: float = 0.0           # bbox height in nm (for backward compat)
    width_nm: float = 0.0            # bbox width in nm

    def to_dict(self, pixel_size_nm: float) -> dict:
        """Serialise to a JSON-friendly dict (mask → RLE is omitted; polygon is kept)."""
        return {
            "source": self.source,
            "polygon": self.polygon.tolist() if self.polygon is not None else None,
            "bbox": list(self.bbox),
            "mean_intensity": float(self.mean_intensity),
            "std_intensity": float(self.std_intensity),
            "area_px": int(self.area_px),
            "area_nm2": float(self.area_nm2),
            "height_nm": float(self.height_nm),
            "width_nm": float(self.width_nm),
            "pixel_size_nm": float(pixel_size_nm),
        }


def layer_mask_from_polygon(polygon: np.ndarray, image_shape: tuple[int, int],
                              pixel_size_nm: float, image: np.ndarray | None = None,
                              source: str = "polygon") -> LayerMask:
    """Build a LayerMask from a polygon."""
    mask = polygon_to_mask(polygon, image_shape)
    bbox = polygon_bbox(polygon)
    area_px = int(mask.sum())
    area_nm2 = float(area_px * pixel_size_nm * pixel_size_nm)
    height_nm = float((bbox[2] - bbox[0]) * pixel_size_nm)
    width_nm = float((bbox[3] - bbox[1]) * pixel_size_nm)
    mean_i = std_i = 0.0
    if image is not None:
        pixels = image[mask]
        if pixels.size > 0:
            mean_i = float(pixels.mean())
            std_i = float(pixels.std())
    return LayerMask(
        mask=mask, polygon=polygon.astype(np.float32), bbox=bbox,
        source=source, mean_intensity=mean_i, std_intensity=std_i,
        area_px=area_px, area_nm2=area_nm2, height_nm=height_nm, width_nm=width_nm,
    )


def layer_mask_from_rows(y0: int, y1: int, image_shape: tuple[int, int],
                          pixel_size_nm: float, image: np.ndarray | None = None) -> LayerMask:
    """Build a LayerMask from a rectangular row range [y0, y1).

    This is the backward-compatible path: the mask is a full-width horizontal
    band. The polygon is a rectangle with 4 corners.
    """
    h, w = image_shape
    y0 = max(0, y0)
    y1 = min(h, y1)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, :] = True
    polygon = np.array([[y0, 0], [y0, w - 1], [y1 - 1, w - 1], [y1 - 1, 0]],
                        dtype=np.float32)
    bbox = (y0, 0, y1, w)
    area_px = int(mask.sum())
    area_nm2 = float(area_px * pixel_size_nm * pixel_size_nm)
    height_nm = float((y1 - y0) * pixel_size_nm)
    width_nm = float(w * pixel_size_nm)
    mean_i = std_i = 0.0
    if image is not None:
        pixels = image[mask]
        if pixels.size > 0:
            mean_i = float(pixels.mean())
            std_i = float(pixels.std())
    return LayerMask(
        mask=mask, polygon=polygon, bbox=bbox,
        source="manual_rows", mean_intensity=mean_i, std_intensity=std_i,
        area_px=area_px, area_nm2=area_nm2, height_nm=height_nm, width_nm=width_nm,
    )
