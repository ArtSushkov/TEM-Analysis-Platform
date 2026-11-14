"""
segmentation.py — Grain-boundary segmentation for TEM cross-section images.

Pipeline
--------
Given a 2-D grayscale image of a single horizontal layer (band) extracted
from a TEM cross-section:

1. **Bilateral filter** — denoise while preserving grain-boundary edges.
2. **CLAHE** — local contrast enhancement so weak boundaries become visible.
3. **Gradient magnitude** (Scharr operator) — boundaries = high gradient.
4. **Threshold** — Otsu (default) on the gradient magnitude to get a robust,
   image-adaptive boundary mask. A percentile fallback is also supported.
5. **Morphological close + open** — connect broken boundaries, remove noise.
6. **Fill holes** — close the grain interiors.
7. **Distance transform** + regional maxima → watershed markers.
8. **Marker-controlled watershed** on the CLAHE image.
9. **Filter regions** by area and left/right (X-axis) border-touching.

Important: by default, grains touching the **top/bottom** (Y-axis) borders of
the band are NOT excluded, because those borders are the layer boundaries —
in a thin polycrystalline layer, every grain naturally touches both layer
interfaces. Only grains touching the **left/right** (X-axis) image borders
are excluded, as those represent arbitrary crop edges.

References
----------
* Marker-controlled watershed: Beucher & Lantuéjoul (1979); see also
  Soille, "Morphological Image Analysis" (2003).
* Scharr operator: Scharr (2000) — optimally rotation-symmetric 3×3 kernel.
* Otsu threshold: Otsu (1979).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from skimage import feature, filters, measure, morphology
from skimage import segmentation as skseg


# === Parameters ===

@dataclass
class SegmentationParams:
    """All tunable parameters for the grain-segmentation pipeline.

    Sensible defaults are tuned for TEM cross-sections at 20–35 pm/px.
    """
    # Step 1: Bilateral filter (edge-preserving denoise).
    bilateral_d: int = 9              # neighbourhood diameter (px)
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0

    # Step 2: CLAHE (local contrast enhancement).
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8          # tile size = (image_size / this)

    # Step 3: Gradient.
    use_scharr: bool = True           # True=Scharr, False=Sobel

    # Step 4: Gradient threshold.
    # 'otsu' = Otsu threshold on gradient magnitude (recommended, adaptive).
    # 'percentile' = use the given percentile.
    # 'absolute' = use the given absolute value.
    gradient_threshold_mode: str = "otsu"
    gradient_threshold: float | None = None
    gradient_percentile: float = 90.0  # used when mode == 'percentile'
    # Multiplicative factor applied to the computed threshold.
    # >1 makes the boundary mask stricter (fewer boundaries, larger grains).
    # <1 makes the mask looser (more boundaries, smaller grains).
    gradient_threshold_scale: float = 1.0

    # Step 5: Morphological operations.
    close_kernel_px: int = 3
    open_kernel_px: int = 2

    # Step 7: Watershed seeds.
    # Min distance between seeds in pixels. Scaled by image height when 0.
    watershed_min_distance_px: int = 0
    watershed_min_distance_factor: float = 0.05  # = 5% of band height
    # Minimum normalised distance (fraction of max) for a peak to count.
    watershed_min_intensity_frac: float = 0.1

    # Step 9: Region filters.
    min_grain_area_nm2: float = 25.0
    max_grain_area_nm2: float = 1.0e5
    # Excludes grains touching the left/right (X-axis) image borders.
    exclude_left_right_borders: bool = True
    # Excludes grains touching the top/bottom (Y-axis) band borders.
    # Default False — these are the layer interfaces.
    exclude_top_bottom_borders: bool = False
    border_margin_px: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "bilateral_d": self.bilateral_d,
            "bilateral_sigma_color": self.bilateral_sigma_color,
            "bilateral_sigma_space": self.bilateral_sigma_space,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid": self.clahe_tile_grid,
            "use_scharr": self.use_scharr,
            "gradient_threshold_mode": self.gradient_threshold_mode,
            "gradient_threshold": self.gradient_threshold,
            "gradient_percentile": self.gradient_percentile,
            "gradient_threshold_scale": self.gradient_threshold_scale,
            "close_kernel_px": self.close_kernel_px,
            "open_kernel_px": self.open_kernel_px,
            "watershed_min_distance_px": self.watershed_min_distance_px,
            "watershed_min_distance_factor": self.watershed_min_distance_factor,
            "watershed_min_intensity_frac": self.watershed_min_intensity_frac,
            "min_grain_area_nm2": self.min_grain_area_nm2,
            "max_grain_area_nm2": self.max_grain_area_nm2,
            "exclude_left_right_borders": self.exclude_left_right_borders,
            "exclude_top_bottom_borders": self.exclude_top_bottom_borders,
            "border_margin_px": self.border_margin_px,
        }


@dataclass
class SegmentationResult:
    """Output of the segmentation pipeline."""
    labels: np.ndarray              # int32 label image (0 = background)
    boundaries: np.ndarray          # bool mask of boundary pixels
    n_grains_raw: int               # before filtering
    n_grains_filtered: int          # after area/border filters
    gradient: np.ndarray            # float32 gradient magnitude
    preprocessed: np.ndarray        # uint8 image after CLAHE
    gradient_threshold_value: float = 0.0  # the threshold actually used
    removed_labels: list[int] = field(default_factory=list)
    removal_reasons: dict[int, str] = field(default_factory=dict)
    # Bounding box of the cropped region in original-image coordinates.
    # Set by segment_grains_in_mask; (0, 0, H, W) for the full-image path.
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


# === Pipeline ===

def _bilateral_denoise(image: np.ndarray, p: SegmentationParams) -> np.ndarray:
    """Edge-preserving denoise."""
    return cv2.bilateralFilter(
        image,
        d=p.bilateral_d,
        sigmaColor=p.bilateral_sigma_color,
        sigmaSpace=p.bilateral_sigma_space,
    )


def _clahe(image: np.ndarray, p: SegmentationParams) -> np.ndarray:
    """Local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip_limit,
                            tileGridSize=(p.clahe_tile_grid, p.clahe_tile_grid))
    return clahe.apply(image)


def _gradient_magnitude(image: np.ndarray, p: SegmentationParams) -> np.ndarray:
    """Compute gradient magnitude using Scharr or Sobel."""
    if p.use_scharr:
        gx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _compute_gradient_threshold(gradient: np.ndarray, p: SegmentationParams) -> float:
    """Determine the gradient threshold according to the selected mode."""
    if p.gradient_threshold_mode == "absolute" and p.gradient_threshold is not None:
        thr = float(p.gradient_threshold)
    elif p.gradient_threshold_mode == "percentile":
        thr = float(np.percentile(gradient, p.gradient_percentile))
    else:  # "otsu" (default)
        # Otsu requires uint8. Rescale gradient to 0-255.
        g = gradient.astype(np.float64)
        g_min, g_max = float(g.min()), float(g.max())
        if g_max - g_min < 1e-6:
            return float(g_max)
        g_u8 = np.clip((g - g_min) * 255.0 / (g_max - g_min), 0, 255).astype(np.uint8)
        thr = float(filters.threshold_otsu(g_u8) / 255.0 * (g_max - g_min) + g_min)
    return thr * p.gradient_threshold_scale


def _boundary_mask(gradient: np.ndarray, p: SegmentationParams,
                   threshold_value: float) -> np.ndarray:
    """Threshold the gradient magnitude to get a boundary mask."""
    mask = (gradient > threshold_value).astype(np.uint8)
    # Morphological close → connect broken boundary segments.
    k_close = p.close_kernel_px
    if k_close > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # Morphological open → remove small noise.
    k_open = p.open_kernel_px
    if k_open > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask


def _watershed(image: np.ndarray, boundary_mask: np.ndarray,
               p: SegmentationParams) -> np.ndarray:
    """Marker-controlled watershed.

    Steps:
      - Invert the boundary mask → "grain interior" mask.
      - Fill holes in the interior mask.
      - Compute the Euclidean distance transform inside the interiors.
      - Find regional maxima of the distance transform (with min-distance
        constraint) → these are the watershed seeds.
      - Run watershed on the original image, seeded by these markers.
    """
    h, w = image.shape

    # Interiors = NOT boundary.
    interiors = (boundary_mask == 0)
    # Fill holes (closed grain interiors).
    interiors = morphology.remove_small_holes(interiors, area_threshold=8)
    boundary_mask_clean = (~interiors).astype(np.uint8)

    # Distance transform of the interiors.
    dist = cv2.distanceTransform(interiors.astype(np.uint8), cv2.DIST_L2, 5)

    # Min distance between seeds (adaptive to band height if 0).
    if p.watershed_min_distance_px > 0:
        min_distance = p.watershed_min_distance_px
    else:
        min_distance = max(3, int(h * p.watershed_min_distance_factor))

    # Min intensity threshold for peaks.
    dist_max = float(dist.max()) if dist.size else 0.0
    threshold_abs = dist_max * p.watershed_min_intensity_frac

    # Find regional maxima.
    footprint = np.ones((3, 3), dtype=bool)
    coords = feature.peak_local_max(
        dist,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        footprint=footprint,
    )
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    if markers.max() == 0:
        # No markers found — return a single region covering everything.
        return np.ones_like(image, dtype=np.int32), boundary_mask_clean

    # cv2.watershed expects a 3-channel uint8 image and int32 markers.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_rgb, markers.copy())
    # cv2.watershed sets unknown/boundary pixels to -1.
    markers[markers == -1] = 0
    return markers, boundary_mask_clean


def _filter_regions(labels: np.ndarray, pixel_size_nm: float,
                    p: SegmentationParams) -> tuple[np.ndarray, list[int], dict[int, str]]:
    """Filter labeled regions by area and border-touching."""
    px_area_to_nm2 = pixel_size_nm * pixel_size_nm
    h, w = labels.shape
    margin = p.border_margin_px

    # Border labels — only X-axis (left/right) by default.
    border_labels = set()
    if p.exclude_left_right_borders:
        for y in range(h):
            for x in range(margin):
                border_labels.add(int(labels[y, x]))
                border_labels.add(int(labels[y, w - 1 - x]))
        # Also the top/bottom of the X-axis edge columns.
    if p.exclude_top_bottom_borders:
        for x in range(w):
            for y in range(margin):
                border_labels.add(int(labels[y, x]))
                border_labels.add(int(labels[h - 1 - y, x]))
    border_labels.discard(0)  # background

    removed: list[int] = []
    reasons: dict[int, str] = {}
    new_labels = labels.copy()

    for region in measure.regionprops(labels):
        label = region.label
        area_nm2 = region.area * px_area_to_nm2
        if area_nm2 < p.min_grain_area_nm2:
            removed.append(int(label))
            reasons[int(label)] = f"too_small (area={area_nm2:.1f} nm² < {p.min_grain_area_nm2:.1f})"
            new_labels[labels == label] = 0
        elif area_nm2 > p.max_grain_area_nm2:
            removed.append(int(label))
            reasons[int(label)] = f"too_large (area={area_nm2:.1f} nm² > {p.max_grain_area_nm2:.1f})"
            new_labels[labels == label] = 0
        elif int(label) in border_labels:
            removed.append(int(label))
            reasons[int(label)] = "border_touching"
            new_labels[labels == label] = 0

    return new_labels, removed, reasons


def segment_grains(band_image: np.ndarray, pixel_size_nm: float,
                   params: SegmentationParams | None = None) -> SegmentationResult:
    """Run the full grain-segmentation pipeline on a single layer band."""
    if params is None:
        params = SegmentationParams()

    if band_image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    if band_image.dtype != np.uint8:
        band_image = np.clip(band_image, 0, 255).astype(np.uint8)

    # Step 1: bilateral filter.
    denoised = _bilateral_denoise(band_image, params)

    # Step 2: CLAHE.
    enhanced = _clahe(denoised, params)

    # Step 3: gradient magnitude.
    gradient = _gradient_magnitude(enhanced, params)

    # Step 4 + 5: boundary mask.
    thr_value = _compute_gradient_threshold(gradient, params)
    boundary_mask = _boundary_mask(gradient, params, thr_value)

    # Step 7 + 8: watershed.
    labels, boundary_mask_clean = _watershed(enhanced, boundary_mask, params)
    n_raw = int(labels.max())

    # Step 9: filter regions.
    labels_filtered, removed, reasons = _filter_regions(labels, pixel_size_nm, params)
    n_filtered = int(labels_filtered.max())

    return SegmentationResult(
        labels=labels_filtered.astype(np.int32),
        boundaries=boundary_mask_clean.astype(bool),
        n_grains_raw=n_raw,
        n_grains_filtered=n_filtered,
        gradient=gradient.astype(np.float32),
        preprocessed=enhanced,
        gradient_threshold_value=float(thr_value),
        removed_labels=removed,
        removal_reasons=reasons,
    )


def segment_grains_in_mask(image: np.ndarray, layer_mask: np.ndarray,
                            pixel_size_nm: float,
                            params: SegmentationParams | None = None) -> SegmentationResult:
    """Run grain segmentation restricted to a layer mask.

    This is the mask-aware version of ``segment_grains``. The layer mask
    defines the region of interest (any shape — rectangle, polygon, or a
    SAM-generated mask). Grains are segmented only inside the mask; pixels
    outside the mask are set to 0 in the output labels.

    The image is cropped to the mask's bounding box for efficiency, and the
    labels are returned in the **cropped** coordinate frame (so they align
    with the cropped image). The caller is responsible for placing them back
    into the full image if needed.

    Parameters
    ----------
    image : np.ndarray (H, W) uint8
        Full grayscale image.
    layer_mask : np.ndarray (H, W) bool
        Boolean mask of the selected layer (True = inside the layer).
    pixel_size_nm : float
    params : SegmentationParams, optional

    Returns
    -------
    SegmentationResult
        ``labels`` and ``boundaries`` are in the cropped frame (size = bbox
        of the mask). ``preprocessed`` and ``gradient`` are also cropped.
    """
    if params is None:
        params = SegmentationParams()
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    if layer_mask.shape != image.shape:
        raise ValueError(f"Mask shape {layer_mask.shape} != image shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Crop to the mask's bounding box (with a small padding for the bilateral
    # filter not to introduce edge artifacts).
    from .polygon import crop_to_mask_bbox
    cropped_img, cropped_mask, bbox = crop_to_mask_bbox(image, layer_mask, padding=2)
    if cropped_img.size == 0 or not cropped_mask.any():
        # Empty mask — return an empty result.
        return SegmentationResult(
            labels=np.zeros(cropped_img.shape, dtype=np.int32),
            boundaries=np.zeros(cropped_img.shape, dtype=bool),
            n_grains_raw=0, n_grains_filtered=0,
            gradient=np.zeros(cropped_img.shape, dtype=np.float32),
            preprocessed=cropped_img,
            gradient_threshold_value=0.0,
        )

    # Run the standard pipeline on the cropped image.
    result = segment_grains(cropped_img, pixel_size_nm, params)
    result.bbox = bbox

    # Mask out labels outside the layer mask.
    # The labels are in the cropped frame; cropped_mask is also in that frame.
    result.labels[~cropped_mask] = 0
    # Recount grains after masking.
    n_filtered = int(result.labels.max()) if result.labels.max() > 0 else 0
    result.n_grains_filtered = n_filtered

    return result
