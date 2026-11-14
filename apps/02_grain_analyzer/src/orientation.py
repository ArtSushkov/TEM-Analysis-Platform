"""
orientation.py — Detect layer orientation in TEM cross-section images and
derotate the image so layers become horizontal.

Method: brute-force row-mean variance
--------------------------------------
For each candidate layer angle θ ∈ [−45°, +45°], we rotate the image by θ and
compute the variance of the row-mean intensity profile. When the layers are
horizontal, the row-mean profile has sharp transitions between layers of
different brightness → **maximum variance**. When the layers are tilted, the
row-mean profile is "smudged" (each row averages across multiple layers) →
**low variance**.

So: the layer angle = the rotation θ that maximises row-mean variance.

Why not the Radon transform?
----------------------------
The Radon transform computes projection variance, which is symmetric in θ
(Radon(θ) ≡ Radon(180°−θ)) and confounded by intra-layer grain texture — for
TEM cross-sections with vertically-elongated grains inside layers, the
Radon variance peak often lands on the wrong angle (e.g. 0° instead of ±15°).
The brute-force row-mean variance directly measures "how horizontal are the
layers" and gives the correct sign.

Performance
-----------
~45 rotations of a 512×512 image with 1° step → <0.5 s on a typical TEM image.
A 2-stage refinement (1° coarse + 0.1° fine around the peak) gives sub-degree
accuracy without significantly increasing runtime.

Edge handling after rotation
----------------------------
After derotation, the four corners of the image contain "synthetic" black
pixels (the `cval` fill). These are flagged in the returned image as a boolean
mask so downstream code (info-strip detection, layer detection) can exclude
them. In practice, the row-mean profile is robust to a small number of black
rows because they appear as a low-intensity plateau at the edges and do not
produce spurious high-gradient boundaries in the bulk.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional

import cv2
import numpy as np
from skimage.transform import rotate


@dataclass
class AngleDetectionResult:
    """Result of layer-orientation detection.

    Fields
    ------
    angle_deg : float
        Detected layer angle w.r.t. the horizontal axis, normalised to
        [-45, +45]. Positive = layers tilted counter-clockwise from horizontal;
        the image should be derotated by ``-angle_deg`` to make layers
        horizontal.
    confidence : float
        Heuristic confidence in [0, 1]. Computed as
        (peak_variance - min_variance) / (peak_variance + eps).
        Higher = sharper peak in the variance curve = more confident.
    variance_curve : np.ndarray
        Row-mean variance at each candidate angle (length = len(theta)).
    theta_deg : np.ndarray
        Candidate layer angles in degrees (the x-axis of the variance curve).
    peak_theta_deg : float
        The angle at which variance is maximum (= the detected layer angle).
    method : str
        Detection method name.
    """
    angle_deg: float
    confidence: float
    variance_curve: np.ndarray
    theta_deg: np.ndarray
    peak_theta_deg: float
    method: str = "row_mean_variance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_deg": float(self.angle_deg),
            "confidence": float(self.confidence),
            "peak_theta_deg": float(self.peak_theta_deg),
            "method": self.method,
        }


# === Detection ===

def _row_mean_variance(rotated: np.ndarray, cval: float = 0.0) -> float:
    """Compute the variance of the row-mean intensity profile.

    When ``cval`` is set to the image's mean intensity (as done in
    ``detect_layer_angle``), the rotation fill pixels blend in and don't
    create a systematic gradient, so we can simply compute the variance on
    the full row-mean profile.

    A high variance means the rows have distinct mean intensities → the layers
    are horizontal. A low variance means each row averages across multiple
    layers → the layers are tilted.

    Parameters
    ----------
    rotated : np.ndarray
        The rotated image (2-D, float64).
    cval : float
        The fill value used during rotation (unused when it equals the image
        mean, but kept for API compatibility).

    Returns
    -------
    float
    """
    row_mean = rotated.mean(axis=1)
    return float(np.var(row_mean))


def detect_layer_angle(
    image: np.ndarray,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
    fine_half_width: float = 2.0,
    downscale_max_dim: int = 512,
    angle_search_range: tuple[float, float] = (-45.0, 45.0),
    central_crop_frac: float = 0.6,
) -> AngleDetectionResult:
    """Detect the dominant layer orientation by maximising row-mean variance.

    Two-stage search: a coarse pass over ``angle_search_range`` with step
    ``coarse_step`` finds the approximate peak, then a fine pass with step
    ``fine_step`` in a ±``fine_half_width`` window around the coarse peak
    refines the result.

    To avoid the microscope's bottom info-strip (text + scale bar, which has a
    strong horizontal pattern that biases the detected angle), the variance is
    computed on the **central horizontal band** of the image (rows in the
    middle ``central_crop_frac`` fraction of the image height).

    **Black-corner handling:** When the image is rotated during the search,
    the four corners are exposed as synthetic fill pixels. If we fill them
    with 0 (black), they create a smooth intensity gradient that dominates
    the row-mean variance and biases the detected angle. Instead, we fill
    them with the **mean image intensity** — this way, the fill pixels blend
    in and don't create a systematic gradient, so the row-mean variance
    reflects only the real layer structure.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale uint8 image.
    coarse_step : float
        Step between candidate angles in the coarse pass (degrees).
    fine_step : float
        Step between candidate angles in the fine pass (degrees).
    fine_half_width : float
        Half-width of the fine-search window around the coarse peak (degrees).
    downscale_max_dim : int
        Downscale the image to this max dimension before rotation (for speed).
        The detected angle is independent of scale.
    angle_search_range : (float, float)
        Range of layer angles to search, in degrees. Default (-45, 45) covers
        all realistic TEM-cross-section tilts.
    central_crop_frac : float
        Fraction of image height (centred) to use for angle detection.
        Default 0.6 = middle 60%, which excludes both the top and bottom
        info-strips. Set to 1.0 to use the full image.

    Returns
    -------
    AngleDetectionResult
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")

    h_full, w = image.shape

    # Crop to the central horizontal band to exclude info-strips at top/bottom.
    if central_crop_frac < 1.0:
        crop_h = int(h_full * central_crop_frac)
        y0 = (h_full - crop_h) // 2
        cropped = image[y0:y0 + crop_h, :]
    else:
        cropped = image

    # Downscale for speed (rotation is O(N^2) per angle).
    h_crop, w_crop = cropped.shape
    scale = min(1.0, float(downscale_max_dim) / max(h_crop, w_crop))
    if scale < 1.0:
        small = cv2.resize(cropped, (int(w_crop * scale), int(h_crop * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = cropped
    small_f = small.astype(np.float64)

    # Use the mean intensity as the rotation fill value. This prevents the
    # black-corner gradient that would otherwise dominate the row-mean variance.
    fill_value = float(small_f.mean())

    # === Coarse pass ===
    lo, hi = angle_search_range
    coarse_angles = np.arange(lo, hi + coarse_step * 0.5, coarse_step)
    coarse_variances = np.empty(coarse_angles.shape[0], dtype=np.float64)
    for i, ang in enumerate(coarse_angles):
        rot = rotate(small_f, float(ang), resize=False, order=1,
                     mode="constant", cval=fill_value, preserve_range=True)
        coarse_variances[i] = _row_mean_variance(rot, cval=fill_value)

    coarse_max_idx = int(np.argmax(coarse_variances))
    coarse_peak = float(coarse_angles[coarse_max_idx])

    # === Fine pass around the coarse peak ===
    fine_lo = max(lo, coarse_peak - fine_half_width)
    fine_hi = min(hi, coarse_peak + fine_half_width)
    fine_angles = np.arange(fine_lo, fine_hi + fine_step * 0.5, fine_step)
    fine_variances = np.empty(fine_angles.shape[0], dtype=np.float64)
    for i, ang in enumerate(fine_angles):
        rot = rotate(small_f, float(ang), resize=False, order=1,
                     mode="constant", cval=fill_value, preserve_range=True)
        fine_variances[i] = _row_mean_variance(rot, cval=fill_value)

    fine_max_idx = int(np.argmax(fine_variances))
    peak_angle = float(fine_angles[fine_max_idx])
    peak_variance = float(fine_variances[fine_max_idx])

    # Combine the coarse and fine curves for plotting.
    # (Drop the coarse samples that fall inside the fine window to avoid duplicates.)
    in_fine_window = (coarse_angles >= fine_lo) & (coarse_angles <= fine_hi)
    combined_theta = np.concatenate([coarse_angles[~in_fine_window], fine_angles])
    combined_var = np.concatenate([coarse_variances[~in_fine_window], fine_variances])
    # Sort by angle for a clean plot.
    sort_idx = np.argsort(combined_theta)
    combined_theta = combined_theta[sort_idx]
    combined_var = combined_var[sort_idx]

    # Confidence: peak prominence over the minimum.
    min_var = float(np.min(combined_var)) if combined_var.size else 0.0
    confidence = (peak_variance - min_var) / (peak_variance + 1e-9) if peak_variance > 0 else 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return AngleDetectionResult(
        angle_deg=float(peak_angle),
        confidence=float(confidence),
        variance_curve=combined_var.astype(np.float32),
        theta_deg=combined_theta.astype(np.float32),
        peak_theta_deg=float(peak_angle),
    )


# === Derotation ===

def derotate_image(image: np.ndarray, angle_deg: float,
                   cval: float = 0.0, order: int = 1) -> np.ndarray:
    """Rotate the image by ``-angle_deg`` so layers become horizontal.

    Uses ``skimage.transform.rotate`` with ``resize=False`` (output dimensions
    match input). The four corners are filled with ``cval`` (default 0 = black).

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale uint8 (or float) image.
    angle_deg : float
        Layer angle w.r.t. horizontal, in degrees. The image is rotated by
        ``-angle_deg`` so the layers become horizontal.
    cval : float
        Fill value for the corners exposed by rotation.
    order : int
        Spline order for interpolation (1 = bilinear, 3 = bicubic).

    Returns
    -------
    np.ndarray (same dtype as input)
    """
    if abs(angle_deg) < 1e-3:
        return image
    img_f = image.astype(np.float64)
    rotated = rotate(img_f, -float(angle_deg),
                     resize=False, order=order, mode="constant", cval=cval,
                     preserve_range=True)
    if image.dtype == np.uint8:
        return np.clip(rotated, 0, 255).astype(np.uint8)
    return rotated.astype(image.dtype)


def derotate_image_with_mask(image: np.ndarray, angle_deg: float,
                              cval: float = 0.0, order: int = 1
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Derotate and also return a boolean mask of valid (non-corner) pixels.

    The mask is True where the image contains real data and False where the
    rotation exposed synthetic ``cval`` pixels.

    Parameters
    ----------
    image, angle_deg, cval, order : see ``derotate_image``.

    Returns
    -------
    (rotated_image, valid_mask) : (np.ndarray, np.ndarray of bool)
    """
    if abs(angle_deg) < 1e-3:
        return image, np.ones(image.shape, dtype=bool)
    rotated = derotate_image(image, angle_deg, cval=cval, order=order)
    # The mask is True where the rotated image differs from cval.
    # Use a small tolerance for floating-point comparison.
    if cval == 0.0:
        valid_mask = rotated > 1
    else:
        valid_mask = np.abs(rotated.astype(np.float64) - cval) > 1.0
    return rotated, valid_mask


# === Plotting ===

def plot_angle_curve(result: AngleDetectionResult):
    """Plot the row-mean-variance curve with the detected peak marked.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot(result.theta_deg, result.variance_curve, color="steelblue",
            linewidth=1.0, label="Row-mean variance")
    ax.axvline(result.angle_deg, color="red", linewidth=2,
               label=f"detected layer angle = {result.angle_deg:.2f}°")
    ax.set_xlabel("Candidate layer angle (°)")
    ax.set_ylabel("Row-mean variance")
    ax.set_title(f"Layer orientation detection (confidence = {result.confidence:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return fig
