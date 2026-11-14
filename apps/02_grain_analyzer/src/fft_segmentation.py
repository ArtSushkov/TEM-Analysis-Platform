"""
fft_segmentation.py — FFT-based grain segmentation for TEM cross-section images.

Based on the method described in Grachev's dissertation (Chapter 2.4):
"Процедура визуализации нанокристаллов" (sections 2.4.1, 2.4.2).

Adapted for **grains in a polycrystalline layer** (vs. nanocrystals in an
amorphous matrix in the original work). The core idea is the same: work in
**reciprocal (Fourier) space**, where the signal from crystalline regions
(Bragg reflections) is separated from noise.

Pipeline
--------
1. **Optional preprocessing** — CLAHE for contrast, optional denoise.
2. **FFT** — 2-D Fourier transform of the layer image.
3. **Reflection detection** — find local maxima (peaks) in the power spectrum,
   excluding the central low-frequency region and the high-frequency noise.
4. **DBScan clustering** — group reflections that are close in reciprocal
   space. Each cluster corresponds to a (interplanar distance, orientation)
   pair — i.e. one crystallographic variant present in the field of view.
5. **Per-cluster orientation map** — for each cluster, build a mask in
   reciprocal space (small disks around each reflection in the cluster),
   inverse-FFT, take the amplitude. Bright pixels = regions whose lattice
   matches this cluster.
6. **Pixel labelling** — assign each pixel to the cluster with the maximum
   amplitude (argmax across the per-cluster maps). Pixels where all maps are
   below a threshold are labelled as background (amorphous / unknown).
7. **Post-processing** — remove small regions, fill holes, optionally merge
   adjacent regions with very similar orientation.

Why this works for TEM
---------------------
Classical gradient/watershed methods fail on TEM because grain boundaries are
not sharp intensity steps — they are smooth contrast changes due to different
crystallographic orientations of adjacent grains relative to the electron beam.
The FFT method does not look for "edges"; it groups pixels by their **local
lattice fringes**, which directly reflect the grain orientation.

Limitations (from Grachev §3.2.1)
--------------------------------
* **2-D projection** — grains at different depths overlap in projection; the
  measured grain sizes are projected, not true 3-D sizes.
* **Moiré** — overlapping grains produce moiré fringes that create spurious
  reflections.
* **Completeness vs. accuracy** — strict thresholds give high accuracy but
  miss some grains; loose thresholds give more grains but more artefacts.

References
----------
* Grachev D.A., "Formation, structural and optical properties of thin-film
  heterogeneous systems with arrays of Ge nanoinclusions in a dielectric",
  PhD dissertation, UNN, 2017, §2.4.
* Hÿtch M.J., Kilaas R. "Geometric phase analysis of HREM images" (1998).
* Ester M. et al. "DBScan: a density-based algorithm for discovering clusters"
  (1996).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.signal import find_peaks
from skimage import feature, measure, morphology, filters
from sklearn.cluster import DBSCAN


# === Parameters ===

@dataclass
class FFTSegmentationParams:
    """Parameters for the FFT-based grain segmentation pipeline."""
    # --- Preprocessing ---
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8

    # --- FFT window (apodization to reduce edge artifacts) ---
    # A Hann window is applied before FFT to suppress the cross-shaped
    # artifacts from the image borders.
    use_hann_window: bool = True

    # --- Wiener filter (Grachev §2.4.1) ---
    # Separates crystalline reflections (sharp peaks) from the amorphous-matrix
    # noise (radially symmetric halo). The filter computes the radial average
    # of the power spectrum (= noise model) and suppresses it, keeping only
    # the peaks that rise above the halo.
    use_wiener_filter: bool = True
    # Radial bin size (in pixels) for computing the noise profile.
    # Smaller = finer noise estimation but noisier. Default 2.
    wiener_radial_bin_px: int = 2
    # Multiplicative factor on the radial noise estimate. >1 = more aggressive
    # suppression (fewer reflections, cleaner but may miss weak ones).
    # <1 = less aggressive (more reflections, noisier). Default 1.0.
    wiener_noise_scale: float = 1.0
    # Smoothing of the radial noise profile (sigma in pixels). Prevents the
    # noise estimate from dipping at reflection positions. Default 5.
    wiener_smoothing_sigma: float = 5.0

    # --- Reflection detection ---
    # In the power spectrum, only consider pixels whose distance from the
    # center is in [min_freq_frac, max_freq_frac] * (image_min_dim / 2).
    # This excludes the central DC + low frequencies (illumination) and the
    # high-frequency noise.
    min_freq_frac: float = 0.05   # min spatial frequency as a fraction of Nyquist
    max_freq_frac: float = 0.49   # max spatial frequency as a fraction of Nyquist

    # Peak detection: reflections are local maxima in the power spectrum
    # above (peak_threshold) * max(power). Typical: 0.01–0.1.
    peak_threshold_frac: float = 0.05

    # Minimum distance (in pixels of the FFT array) between detected peaks.
    min_peak_distance_px: int = 5

    # --- Radial profile detection (recommended, default ON) ---
    # Robust alternative to 2D peak_local_max: integrates the power spectrum
    # azimuthally to get a 1D S(r) profile, finds peaks in 1D (each peak = one
    # d-spacing), then looks for 2D local maxima ONLY within rings at those
    # radii. Much less false positives than the legacy 2D method because the
    # noise is averaged out azimuthally.
    use_radial_profile_detection: bool = True
    # Gaussian smoothing sigma (in radius-pixels) for the 1D profile.
    radial_smoothing_sigma: float = 2.0
    # Quantile of the (smoothed) radial profile used as the baseline.
    # 0.50 (median) is robust to strong peaks.
    radial_baseline_quantile: float = 0.50
    # Peak height threshold = baseline + height_factor * (high_q - baseline),
    # where high_q is the high_quantile-th percentile of the profile.
    radial_high_quantile: float = 0.90
    radial_height_factor: float = 0.40
    # Peak prominence threshold = prominence_factor * (prom_q - baseline),
    # where prom_q is the prom_quantile-th percentile.
    radial_prom_quantile: float = 0.75
    radial_prominence_factor: float = 0.40
    # Min distance (in radius-pixels) between adjacent peaks in the 1D profile.
    radial_peak_min_distance_px: int = 5
    # Within each ring (at a detected peak radius), find 2D local maxima above
    # this fraction of the ring's max. Higher = fewer reflections per ring.
    radial_ring_local_max_frac: float = 0.30

    # --- DBScan clustering ---
    # eps: max distance (in FFT-pixels) between two reflections to be in the
    # same cluster. Smaller = more clusters (more orientations).
    dbscan_eps: float = 3.0
    # min_samples: min number of reflections in a cluster.
    dbscan_min_samples: int = 2

    # --- Per-cluster mask ---
    # Radius (in FFT-pixels) of the disk mask around each reflection.
    mask_radius_px: int = 4

    # --- Ring mask (Grachev §2.4.2 кольцевая маска) ---
    # Width of the ring mask around each d-spacing's radius.
    # Used instead of individual disk masks — a ring captures ALL reflections
    # at that d-spacing, regardless of orientation (Grachev's approach).
    ring_mask_width_px: float = 6.0

    # --- Reflection quality (Gaussian fit, Scherrer) ---
    # Window size (in FFT pixels) for fitting a 2D Gaussian around each
    # manually-clicked reflection. Must be odd. 15 = good for typical FFT
    # sizes (512–2048); 21 if very broad peaks are expected (small grains).
    reflection_fit_window_px: int = 15
    # Scherrer constant K (dimensionless). 0.94 = spherical domains (standard).
    # 0.89 = cubic, 1.0 = needle-shaped along the beam. Default 0.94.
    scherrer_K: float = 0.94
    # Classification thresholds on the fitted σ (in FFT pixels, FWHM=2.355·σ):
    # σ < sigma_good_px → "good" (narrow Bragg peak, large coherent domain)
    # sigma_good_px ≤ σ < sigma_borderline_px → "borderline" (real but broad)
    # σ ≥ sigma_borderline_px → "noise" (too wide to be a real reflection)
    # Note: for very small grains (3–7 nm), peaks are intrinsically broad —
    # you may need to raise sigma_borderline_px to 8–10.
    sigma_good_px: float = 3.0
    sigma_borderline_px: float = 6.0
    # SNR (peak amplitude / background) thresholds:
    snr_good: float = 5.0
    snr_min: float = 2.0
    # Apply the Gaussian quality filter to AUTO-detected reflections too
    # (not just manual clicks). When ON, each 2D maximum inside a ring is
    # checked via evaluate_reflection_quality; those classified as "noise"
    # are dropped. If ALL maxima of a ring are noise, the whole ring (i.e.
    # the 1D peak) is dropped.
    apply_quality_filter_to_auto: bool = False

    # --- Reference d-spacings (mode B: restrict detection to known planes) ---
    # When use_reference_d_spacings is True, the detector searches for 1D
    # peaks ONLY inside the radial bands corresponding to the user-specified
    # (d_min, d_max) ranges — instead of scanning the whole spectrum.
    # Each entry: (label, d_min_nm, d_max_nm). label is a free-form string
    # (e.g. "{111}"). The label of the closest range is attached to each
    # detected d-spacing in the result.
    use_reference_d_spacings: bool = False
    reference_d_spacings: list = field(default_factory=list)  # list[tuple[str, float, float]]
    # Tolerance (%) applied to each (d_min, d_max) range — widens the band
    # so that peaks slightly outside the user range are still caught (useful
    # for noisy / calibration-drift data). 3% = typical.
    reference_tolerance_pct: float = 3.0

    # --- Contrast normalization (Grachev §2.4.2) ---
    # Target ratio mean/max for amplitude maps. Grachev found 0.3 optimal.
    contrast_target_ratio: float = 0.3

    # --- Pixel labelling ---
    # Pixels whose max amplitude (across all cluster maps) is below this
    # fraction of the global max are labelled as background.
    # Typical: 0.02–0.15. Higher = stricter (fewer grains, more background).
    min_amplitude_frac: float = 0.05

    # --- Post-processing ---
    # Remove grain regions smaller than this (in pixels of the FULL-RES image).
    # Default 200 = ~14×14 px grain at full resolution; reasonable for TEM at
    # 20–35 pm/px (≈3–7 nm grain).
    min_grain_area_px: int = 500
    # Fill holes smaller than this (in pixels).
    fill_holes_max_px: int = 100
    # Exclude grains touching the left/right (X-axis) image borders.
    exclude_left_right_borders: bool = True
    # Exclude grains touching the top/bottom (Y-axis) borders.
    exclude_top_bottom_borders: bool = False
    border_margin_px: int = 1

    # --- Max clusters ---
    # Hard limit on the number of clusters (orientations) to keep. Clusters
    # are sorted by total reflection intensity; only the top N are used.
    # IMPORTANT: amplitude maps (inverse FFT per cluster) are expensive
    # (O(H*W) memory each), so this limit must be kept low (≤50).
    max_clusters: int = 30

    # --- Image downscale ---
    # Downscale the image to this max dimension before FFT (for speed/memory).
    # The segmentation result is upscaled back to the original resolution.
    # 512 is a good balance; 1024 gives finer detail but 4× more memory/time.
    max_image_dim: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# === Result ===

@dataclass
class FFTSegmentationResult:
    """Output of the FFT-based segmentation pipeline."""
    labels: np.ndarray              # int32 label image (0 = background)
    n_grains: int
    n_clusters: int                 # number of crystallographic orientations found
    power_spectrum: np.ndarray      # log-scaled power spectrum (for display)
    reflection_coords: np.ndarray   # (N, 2) of (row, col) in FFT array
    cluster_labels: np.ndarray      # (N,) cluster id for each reflection (-1 = noise)
    orientation_map: np.ndarray     # (H, W) int32 — cluster id per pixel (0 = background)
    amplitude_maps: np.ndarray      # (C, H, W) float32 — per-cluster amplitude maps
    d_spacings_nm: list = field(default_factory=list)  # d-spacings in nm per cluster
    d_spacing_labels: list = field(default_factory=list)  # labels per cluster (e.g. "{111}") — empty unless reference mode
    method: str = "fft_grachev"
    # --- Radial profile data (always populated when radial detection is on) ---
    radial_radii: np.ndarray | None = None       # 1D array of radii (px)
    radial_intensities: np.ndarray | None = None # 1D array of S(r) values
    peak_radii: list = field(default_factory=list)  # list of detected peak radii (px)
    scale: float = 1.0  # small→original downscale factor (for d-spacing conversion)
    reference_d_spacings_used: list = field(default_factory=list)  # snapshot of reference ranges used (for UI bands)


# === Pipeline ===

def _preprocess(image: np.ndarray, p: FFTSegmentationParams) -> np.ndarray:
    """CLAHE + optional Hann window."""
    img = image.astype(np.float64)
    if p.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip_limit,
                                tileGridSize=(p.clahe_tile_grid, p.clahe_tile_grid))
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        img = clahe.apply(img_u8).astype(np.float64)
    if p.use_hann_window:
        hann = np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]
        img = img * hann
    return img


def _compute_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the shifted FFT and log-scaled power spectrum.

    Returns
    -------
    (fft_shifted, log_power)
        fft_shifted: complex128, shape (H, W) — shifted FFT (DC at center).
        log_power: float64, shape (H, W) — log(1 + |fft|^2), for display & peak detection.
    """
    fft = fft2(image)
    fft_shifted = fftshift(fft)
    power = np.abs(fft_shifted) ** 2
    log_power = np.log1p(power)
    return fft_shifted, log_power


def _apply_wiener_filter(fft_shifted: np.ndarray, p: FFTSegmentationParams
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Wiener filter to suppress amorphous-matrix noise (Grachev §2.4.1).

    The amorphous matrix produces a diffuse, **radially symmetric** halo in
    Fourier space. Crystalline regions produce sharp, localised reflections.
    The Wiener filter estimates the radial noise profile and suppresses it,
    keeping only the peaks that rise above the halo.

    Method
    ------
    1. Compute the power spectrum ``P(k) = |FFT(k)|²``.
    2. Compute the radial average ``R(r)`` — for each radius ``r`` from the
       center, average ``P`` over all pixels at that radius. ``R(r)`` is the
       noise model (amorphous halo).
    3. Smooth ``R(r)`` with a Gaussian to prevent the noise estimate from
       dipping at reflection positions.
    4. Build the Wiener filter: ``W(k) = max(0, 1 - noise_scale · R(|k|) / P(k))``.
       Where ``P(k) > R(|k|)`` (reflection) → ``W ≈ 1`` (keep).
       Where ``P(k) ≈ R(|k|)`` (noise) → ``W ≈ 0`` (suppress).
    5. Apply: ``filtered_fft = fft · W``.
    6. Recompute the log power spectrum from the filtered FFT.

    Parameters
    ----------
    fft_shifted : np.ndarray (H, W) complex128
        Shifted FFT (DC at center).
    p : FFTSegmentationParams

    Returns
    -------
    (filtered_fft, filtered_log_power)
        filtered_fft: complex128, shape (H, W) — Wiener-filtered FFT.
        filtered_log_power: float64, shape (H, W) — log power of filtered FFT.
    """
    h, w = fft_shifted.shape
    center_r, center_c = h // 2, w // 2

    # Power spectrum.
    power = np.abs(fft_shifted) ** 2

    # 1. Compute radial average (noise profile).
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    max_r = int(dist.max()) + 1
    bin_size = max(1, p.wiener_radial_bin_px)
    n_bins = max_r // bin_size + 1

    # Use np.histogram to compute the radial average efficiently.
    dist_flat = dist.ravel().astype(np.int32)
    power_flat = power.ravel()
    radial_sum = np.bincount(dist_flat, weights=power_flat, minlength=n_bins)
    radial_count = np.bincount(dist_flat, minlength=n_bins)
    radial_avg = np.divide(radial_sum, radial_count,
                            out=np.zeros_like(radial_sum),
                            where=radial_count > 0)

    # 2. Smooth the radial profile (Gaussian) to prevent dips at reflections.
    if p.wiener_smoothing_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        radial_avg = gaussian_filter1d(radial_avg, sigma=p.wiener_smoothing_sigma)

    # 3. Build the radial noise map (same shape as power).
    noise = radial_avg[dist_flat].reshape(h, w) * p.wiener_noise_scale

    # 4. Wiener filter: W = max(0, 1 - noise / power).
    # Guard against division by zero.
    ratio = np.divide(noise, power, out=np.zeros_like(noise), where=power > 0)
    wiener = np.clip(1.0 - ratio, 0.0, 1.0).astype(np.float64)

    # 5. Apply filter.
    filtered_fft = fft_shifted * wiener
    filtered_power = np.abs(filtered_fft) ** 2
    filtered_log_power = np.log1p(filtered_power)

    return filtered_fft, filtered_log_power


def _find_reflections(log_power: np.ndarray, p: FFTSegmentationParams) -> np.ndarray:
    """Find reflection peaks in the log power spectrum.

    Returns
    -------
    np.ndarray (N, 2) of (row, col) — coordinates of detected reflections in
    the FFT array. Excludes the central low-frequency region and the
    high-frequency noise region.
    """
    h, w = log_power.shape
    center_r, center_c = h // 2, w // 2
    # Use max(h, w) for max_radius so that reflections along the long axis
    # of non-square images are not excluded.
    max_radius = max(h, w) // 2

    # Build a frequency mask: keep only the annulus [min_freq, max_freq].
    # Use a HARD minimum of 5px from center to always exclude the DC component,
    # even on small images where min_freq_frac * max_radius is tiny.
    min_dist_from_center = max(5, int(p.min_freq_frac * max_radius))
    max_dist_from_center = int(p.max_freq_frac * max_radius)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    freq_mask = (dist >= min_dist_from_center) & (dist <= max_dist_from_center)

    # Threshold the power spectrum.
    power_masked = np.where(freq_mask, log_power, 0)
    threshold = p.peak_threshold_frac * power_masked.max()

    # Find local maxima.
    coords = feature.peak_local_max(
        power_masked,
        min_distance=p.min_peak_distance_px,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return coords  # (N, 2) of (row, col)


# === Radial profile detection (recommended) ===

def compute_radial_spectrum(log_power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average of the power spectrum: S(r) = mean(log_power over ring at radius r).

    This collapses the 2D power spectrum into a 1D radial profile. Each value
    S(r) is the average log-power over all pixels at distance r from the FFT
    center. Noise (which is roughly azimuthally uniform) survives; Bragg
    reflections (which are localised in angle) appear as small bumps that may
    or may not rise above the noise floor — but only the *strongest* d-spacings
    show up, which is exactly what we want.

    Returns
    -------
    (radii, intensities) : 1D float64 arrays, length max_radius + 1.
        radii[i] = i  (radius in FFT-pixels)
        intensities[i] = mean log_power over the ring at radius i
    """
    h, w = log_power.shape
    center_r, center_c = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    max_r = int(dist.max()) + 1

    dist_flat = dist.ravel().astype(np.int32)
    power_flat = log_power.ravel()
    radial_sum = np.bincount(dist_flat, weights=power_flat, minlength=max_r)
    radial_count = np.bincount(dist_flat, minlength=max_r)
    radial_avg = np.divide(radial_sum, radial_count,
                            out=np.zeros_like(radial_sum, dtype=np.float64),
                            where=radial_count > 0)
    radii = np.arange(max_r, dtype=np.float64)
    return radii, radial_avg


def find_radial_peaks(
    radii: np.ndarray,
    intensities: np.ndarray,
    fft_shape: tuple[int, int],
    p: FFTSegmentationParams,
) -> list[int]:
    """Find peaks in the 1D radial spectrum S(r).

    Each peak corresponds to one d-spacing present in the image. Detection is
    adaptive: the threshold scales with the baseline (median) and the high
    percentile of the smoothed profile, so it works on both clean and noisy
    spectra without manual tuning.

    Returns
    -------
    list of integer radii (in FFT-pixels) where peaks were found, sorted ascending.
    """
    if len(radii) == 0:
        return []
    h, w = fft_shape
    max_radius = max(h, w) // 2
    min_dist = max(5, int(p.min_freq_frac * max_radius))
    max_dist = int(p.max_freq_frac * max_radius)

    # Restrict to the valid radial range.
    valid = (radii >= min_dist) & (radii <= max_dist)
    if not valid.any():
        return []
    r_in = radii[valid]
    s_in = intensities[valid]

    # Smooth to suppress pixel-level noise.
    if p.radial_smoothing_sigma > 0:
        s_smooth = ndi.gaussian_filter1d(s_in, sigma=p.radial_smoothing_sigma)
    else:
        s_smooth = s_in.copy()

    # Adaptive thresholds.
    baseline = float(np.quantile(s_smooth, p.radial_baseline_quantile))
    high_q = float(np.quantile(s_smooth, p.radial_high_quantile))
    prom_q = float(np.quantile(s_smooth, p.radial_prom_quantile))
    height_threshold = baseline + p.radial_height_factor * (high_q - baseline)
    prominence_threshold = p.radial_prominence_factor * max(prom_q - baseline, 1e-12)

    peaks, _ = find_peaks(
        s_smooth,
        height=max(height_threshold, baseline + 1e-9),
        prominence=max(prominence_threshold, 1e-9),
        distance=max(1, p.radial_peak_min_distance_px),
    )
    return [int(r_in[i]) for i in peaks]


def find_reflections_via_radial_profile(
    log_power: np.ndarray,
    p: FFTSegmentationParams,
    pixel_size_nm: float = 1.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Detect reflections using a 1D radial profile + 2D local maxima at peak radii.

    This is the recommended detector (default ON). It is much more robust than
    2D peak_local_max because the noise is averaged out azimuthally before
    peak detection.

    Each detected peak in the 1D profile defines one cluster (one d-spacing).
    All 2D local maxima within the ring at that radius are assigned to that
    cluster — no DBScan needed.

    Quality filter (Q1)
    -------------------
    If ``p.apply_quality_filter_to_auto`` is True, each 2D maximum is checked
    via :func:`evaluate_reflection_quality`. Maxima classified as "noise" are
    dropped. If ALL maxima of a ring are noise, the whole ring (i.e. the 1D
    peak) is dropped from the output.

    Parameters
    ----------
    log_power : (H, W) float — log power spectrum (Wiener-filtered).
    p : FFTSegmentationParams.
    pixel_size_nm, scale : needed for the quality filter (Scherrer estimate).

    Returns
    -------
    (reflection_coords, cluster_labels_per_reflection, peak_radii)
        reflection_coords: (N, 2) of (row, col) in the FFT array.
        cluster_labels_per_reflection: (N,) int — cluster id matching the peak index.
        peak_radii: list of peak radii (in FFT-pixels), one per cluster.
    """
    h, w = log_power.shape
    center_r, center_c = h / 2.0, w / 2.0

    radii, intensities = compute_radial_spectrum(log_power)
    peak_radii = find_radial_peaks(radii, intensities, (h, w), p)

    if not peak_radii:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.int32),
                peak_radii)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)

    all_coords: list[np.ndarray] = []
    all_labels: list[int] = []
    kept_peak_radii: list[int] = []
    ring_half_width = max(2.0, p.ring_mask_width_px / 2.0)

    # We renumber cluster ids sequentially as we keep rings (so that dropping
    # a ring doesn't leave a gap in cluster ids).
    next_cid = 0
    for r_peak in peak_radii:
        ring_mask = (dist >= r_peak - ring_half_width) & (dist <= r_peak + ring_half_width)
        ring_power = np.where(ring_mask, log_power, 0.0)
        if ring_power.max() <= 0:
            continue
        # 2D local maxima within the ring — these are individual reflections
        # at this d-spacing (different orientations).
        coords = feature.peak_local_max(
            ring_power,
            min_distance=max(2, p.min_peak_distance_px),
            threshold_abs=p.radial_ring_local_max_frac * float(ring_power.max()),
            exclude_border=False,
        )

        # --- Q1: quality filter on auto-reflections ---
        if p.apply_quality_filter_to_auto and len(coords) > 0:
            good_coords = []
            for c in coords:
                q = evaluate_reflection_quality(
                    log_power, float(c[0]), float(c[1]),
                    (h, w), pixel_size_nm, scale, p,
                )
                if q.status != "noise":
                    good_coords.append(c)
            coords = np.array(good_coords, dtype=np.int64) if good_coords else \
                np.empty((0, 2), dtype=np.int64)

        if len(coords) == 0:
            # Whole ring was filtered out as noise → drop this peak.
            continue

        for c in coords:
            all_coords.append(c)
            all_labels.append(next_cid)
        kept_peak_radii.append(r_peak)
        next_cid += 1

    if not all_coords:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.int32),
                kept_peak_radii)

    return (np.array(all_coords, dtype=np.int64),
            np.array(all_labels, dtype=np.int32),
            kept_peak_radii)


def compute_d_spacing_for_radius(
    radius_px: float,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
) -> float:
    """Convert a radius (in FFT-pixels) to a d-spacing (in nm).

    For a non-square image, uses the geometric mean of the dimensions, which
    is exact for square images and a good approximation for nearly-square
    images (the typical case after downscaling to max_image_dim).

    The formula is d = 1 / |g|, where |g| = r / (sqrt(H·W) · pixel_size_small)
    and pixel_size_small = pixel_size_nm / scale.
    """
    h, w = fft_shape
    pixel_size_small = pixel_size_nm / scale
    avg_dim = float(np.sqrt(h * w))
    g_mag = float(radius_px) / (avg_dim * pixel_size_small)
    return 1.0 / g_mag if g_mag > 0 else 0.0


def compute_amplitude_preview(
    fft_shifted: np.ndarray,
    click_r: float,
    click_c: float,
    p: FFTSegmentationParams,
    target_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Compute the inverse-FFT amplitude map for a ring at the click radius.

    This is the **interactive preview** of "what does this reflection
    correspond to in real space?". The user clicks on a reflection in the
    power spectrum; this function builds a ring mask at that radius,
    inverse-FFTs it, and returns the amplitude map — bright pixels are the
    regions of the image whose lattice has this d-spacing.

    Parameters
    ----------
    fft_shifted : (H, W) complex128 — Wiener-filtered shifted FFT.
    click_r, click_c : float — click position in small-FFT coords.
    p : FFTSegmentationParams.
    target_shape : optional (H_full, W_full) — upscale the amplitude map to
        this size (e.g., the cropped image shape) for overlay.

    Returns
    -------
    (amp_map, radius_px)
        amp_map: 2D float64 in [0, ~1] (contrast-normalized).
        radius_px: float — the ring radius used (px in small-FFT space).
    """
    h, w = fft_shifted.shape
    center = (h / 2.0, w / 2.0)
    radius = float(np.sqrt((click_r - center[0]) ** 2 + (click_c - center[1]) ** 2))
    if radius < 1.0:
        # Click too close to the center — DC component, no useful info.
        return np.zeros((h, w), dtype=np.float64), 0.0

    mask = _create_ring_mask((h, w), center, radius, p.ring_mask_width_px)
    amp = _compute_amplitude_map(fft_shifted, mask)
    amp = _normalize_contrast(amp, p.contrast_target_ratio)

    if target_shape is not None and amp.shape != target_shape:
        amp = cv2.resize(amp, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return amp, radius


def render_amplitude_overlay(
    cropped_img: np.ndarray,
    amp_map: np.ndarray,
    layer_mask: np.ndarray | None = None,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay an amplitude map on the cropped image as a heatmap.

    Parameters
    ----------
    cropped_img : (H, W) uint8 — cropped layer image (grayscale or RGB).
    amp_map : (h, w) float — amplitude map (will be resized to H, W if needed).
    layer_mask : (H, W) bool — where False, the original image is shown unchanged.
    alpha : blend weight for the heatmap (0 = original only, 1 = heatmap only).
    colormap : OpenCV colormap id (default JET).

    Returns
    -------
    (H, W, 3) uint8 — overlay image.
    """
    h_full, w_full = cropped_img.shape[:2]
    if amp_map.shape != (h_full, w_full):
        amp_full = cv2.resize(amp_map, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
    else:
        amp_full = amp_map.astype(np.float64, copy=False)

    amp_max = float(amp_full.max())
    if amp_max > 0:
        amp_u8 = np.clip(amp_full / amp_max * 255.0, 0, 255).astype(np.uint8)
    else:
        amp_u8 = np.zeros_like(amp_full, dtype=np.uint8)
    heatmap = cv2.applyColorMap(amp_u8, colormap)

    if cropped_img.ndim == 2:
        orig_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = cropped_img.copy()
        if orig_rgb.shape[2] == 4:
            orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_RGBA2RGB)

    overlay = cv2.addWeighted(orig_rgb, 1.0 - alpha, heatmap, alpha, 0)
    if layer_mask is not None:
        overlay[~layer_mask] = orig_rgb[~layer_mask]
    return overlay


def render_radial_spectrum_plotly(
    radii: np.ndarray,
    intensities: np.ndarray,
    peak_radii: list[int],
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    clicked_radius: float | None = None,
    reference_d_spacings: list | None = None,
    reference_tolerance_pct: float = 0.0,
):
    """Build an interactive Plotly figure of the 1D radial spectrum.

    Why Plotly (not matplotlib):
    - Hover on the curve shows radius + power + d-spacing.
    - Peak markers show d-spacing as hovertext (not as overlapping labels).
    - The user's last click is a separate green vertical line.
    - Reference d-spacing bands are shown as green shaded vertical rectangles.

    Parameters
    ----------
    radii, intensities : 1D arrays (from compute_radial_spectrum).
    peak_radii : list of int — auto-detected peak radii (FFT px).
    fft_shape, pixel_size_nm, scale : for d↔r conversion.
    clicked_radius : optional float — last manual click radius (green line).
    reference_d_spacings : optional list of (label, d_min, d_max) — when
        provided, green shaded bands are drawn at the corresponding radii.
    reference_tolerance_pct : tolerance to apply to each band (matches the
        detector's behavior).

    Returns
    -------
    plotly.graph_objects.Figure
        Pass to ``st.plotly_chart(fig, use_container_width=True)``.
    """
    from scipy.ndimage import gaussian_filter1d
    import plotly.graph_objects as go

    h, w = fft_shape
    pixel_size_small = pixel_size_nm / scale
    avg_dim = float(np.sqrt(h * w))

    s_smooth = (gaussian_filter1d(intensities, sigma=2.0)
                if len(intensities) > 0 else intensities)

    # d-spacing for each radius in the curve (for hover).
    d_per_r = np.where(radii > 0, avg_dim * pixel_size_small / np.maximum(radii, 1e-9), 0.0)

    max_radius = max(h, w) // 2
    min_dist = max(5, int(0.05 * max_radius))
    max_dist = int(0.49 * max_radius)

    # Build hover text per point: "r=12 px | d=0.532 nm | S=3.21".
    hover_raw = [f"r={r:.0f} px<br>d={d:.4f} nm<br>S(raw)={v:.3f}"
                 for r, d, v in zip(radii, d_per_r, intensities)]
    hover_smooth = [f"r={r:.0f} px<br>d={d:.4f} nm<br>S(smoothed)={v:.3f}"
                    for r, d, v in zip(radii, d_per_r, s_smooth)]

    fig = go.Figure()

    # Raw profile (faint).
    fig.add_trace(go.Scatter(
        x=radii, y=intensities,
        mode="lines",
        line=dict(color="#8888aa", width=1),
        opacity=0.5,
        name="raw S(r)",
        hoverinfo="text",
        hovertext=hover_raw,
    ))

    # Smoothed profile (main curve).
    fig.add_trace(go.Scatter(
        x=radii, y=s_smooth,
        mode="lines",
        line=dict(color="#1f3b8a", width=1.6),
        name="smoothed S(r)",
        hoverinfo="text",
        hovertext=hover_smooth,
    ))

    # Peak markers — d-spacing shown as hovertext, not as labels (no overlap).
    if peak_radii:
        peak_y = [float(s_smooth[int(round(r))]) if 0 <= int(round(r)) < len(s_smooth) else 0.0
                  for r in peak_radii]
        peak_d = [avg_dim * pixel_size_small / r if r > 0 else 0.0 for r in peak_radii]
        peak_hover = [f"<b>peak #{i+1}</b><br>r={r:.0f} px<br>d={d:.4f} nm"
                      for i, (r, d) in enumerate(zip(peak_radii, peak_d))]
        fig.add_trace(go.Scatter(
            x=[float(r) for r in peak_radii],
            y=peak_y,
            mode="markers",
            marker=dict(color="#c0392b", size=10,
                        line=dict(color="white", width=1)),
            name=f"peaks ({len(peak_radii)})",
            hoverinfo="text",
            hovertext=peak_hover,
        ))
        # Vertical dashed lines for each peak.
        for r in peak_radii:
            fig.add_vline(x=float(r), line=dict(color="#c0392b", dash="dash",
                                                width=1), opacity=0.5)

    # User's last click — green solid line.
    if clicked_radius is not None and clicked_radius > 0:
        d_click = avg_dim * pixel_size_small / clicked_radius
        fig.add_vline(x=float(clicked_radius),
                      line=dict(color="#27ae60", dash="solid", width=2))
        # Annotation at the top of the plot.
        fig.add_annotation(
            x=float(clicked_radius), y=1.0, yref="paper",
            text=f"your click<br>d={d_click:.4f} nm",
            showarrow=False, yshift=10,
            font=dict(color="#27ae60", size=11),
            bgcolor="white", bordercolor="#27ae60", borderwidth=1,
        )

    # Reference d-spacing bands — green shaded vertical rectangles.
    if reference_d_spacings:
        for (label, d_min, d_max) in reference_d_spacings:
            if d_max <= d_min or d_min <= 0:
                continue
            r_min, r_max = d_spacing_to_radius_range(
                d_min, d_max, fft_shape, pixel_size_nm, scale, reference_tolerance_pct
            )
            if r_max <= 0:
                continue
            fig.add_vrect(
                x0=r_min, x1=r_max,
                fillcolor="rgba(39, 174, 96, 0.18)",
                line=dict(color="#27ae60", width=1, dash="dot"),
                layer="below",
                annotation_text=str(label),
                annotation_position="top left",
                annotation=dict(font=dict(color="#27ae60", size=10),
                                bgcolor="rgba(255,255,255,0.85)"),
            )

    fig.update_layout(
        title="1D radial power spectrum — peaks = d-spacings present in the layer",
        xaxis_title="Radius in FFT (pixels from center)",
        yaxis_title="Power (log scale)",
        yaxis_type="log",
        xaxis_range=[min_dist, max_dist],
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        margin=dict(l=60, r=20, t=60, b=50),
        height=380,
        template="plotly_white",
    )
    return fig


# === Reflection quality assessment (Gaussian fit + Scherrer) ===

@dataclass
class ReflectionQuality:
    """Quality assessment of a manually-clicked reflection.

    A 2D Gaussian is fitted in a small window around the click; from the fit
    we extract σ, SNR, and — via the Scherrer formula — an estimate of the
    coherent-domain size D.

    Fields
    ------
    status : "good" | "borderline" | "noise"
    sigma_mean_px : float — geometric mean σ = sqrt(σ_r · σ_c) (FFT pixels)
    sigma_r_px, sigma_c_px : float — Gaussian σ along the two principal axes
    amplitude : float — fitted peak amplitude above background
    background : float — fitted background level
    snr : float — amplitude / |background| (peak signal-to-noise)
    d_spacing_nm : float — d-spacing at the click radius
    coherent_size_nm : float — Scherrer estimate D = K / FWHM_g
    fit_success : bool — whether curve_fit converged
    center_r_fft, center_c_fft : float — fitted peak center (FFT coords)
    message : str — human-readable status description
    """
    status: str
    sigma_mean_px: float = 0.0
    sigma_r_px: float = 0.0
    sigma_c_px: float = 0.0
    amplitude: float = 0.0
    background: float = 0.0
    snr: float = 0.0
    d_spacing_nm: float = 0.0
    coherent_size_nm: float = 0.0
    fit_success: bool = False
    center_r_fft: float = 0.0
    center_c_fft: float = 0.0
    message: str = ""


def fit_gaussian_2d(
    log_power: np.ndarray,
    click_r: float,
    click_c: float,
    window_size: int = 15,
) -> dict | None:
    """Fit a 2D elliptical Gaussian in a window around (click_r, click_c).

    Model: f(r, c) = bg + amp · exp(-0.5 · (dr_rot²/σ_r² + dc_rot²/σ_c²))
    where (dr_rot, dc_rot) is the displacement from the center, rotated by θ.

    Parameters
    ----------
    log_power : (H, W) float — log power spectrum.
    click_r, click_c : float — click position in FFT coords (row, col).
    window_size : odd int — side of the square window (default 15).

    Returns
    -------
    dict with keys: amplitude, background, sr, sc, theta, center_r_fft,
    center_c_fft; or None if the fit failed (window out of bounds, flat
    window, or curve_fit did not converge).
    """
    from scipy.optimize import curve_fit

    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    h, w = log_power.shape
    r0 = int(round(click_r))
    c0 = int(round(click_c))
    if r0 - half < 0 or r0 + half + 1 > h or c0 - half < 0 or c0 + half + 1 > w:
        return None  # too close to border

    window = log_power[r0 - half:r0 + half + 1, c0 - half:c0 + half + 1].astype(np.float64)

    bg0 = float(window.min())
    amp0 = float(window.max() - bg0)
    if amp0 < 1e-9:
        return None  # flat window — no peak

    rows, cols = np.mgrid[0:window_size, 0:window_size]

    def gauss2d(coords, amp, bg, cr, cc, sr, sc, theta):
        r, c = coords
        dr = r - cr
        dc = c - cc
        # Rotate by θ so the principal axes align with the peak's elongation.
        dr_r = dr * np.cos(theta) - dc * np.sin(theta)
        dc_r = dr * np.sin(theta) + dc * np.cos(theta)
        return bg + amp * np.exp(-0.5 * (dr_r ** 2 / sr ** 2 + dc_r ** 2 / sc ** 2))

    try:
        popt, _ = curve_fit(
            gauss2d,
            (rows.ravel(), cols.ravel()),
            window.ravel(),
            p0=[amp0, bg0, half, half, 1.5, 1.5, 0.0],
            bounds=(
                [0, -np.inf, 0, 0, 0.3, 0.3, -np.pi],
                [np.inf, np.inf, window_size, window_size, half, half, np.pi],
            ),
            maxfev=3000,
        )
    except Exception:
        return None

    return {
        "amplitude": float(popt[0]),
        "background": float(popt[1]),
        "sr": float(abs(popt[4])),
        "sc": float(abs(popt[5])),
        "theta": float(popt[6]),
        "center_r_fft": r0 - half + float(popt[2]),
        "center_c_fft": c0 - half + float(popt[3]),
    }


def estimate_coherent_size_scherrer(
    sigma_px: float,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    K: float = 0.94,
) -> float:
    """Estimate coherent-domain size D from peak σ using the Scherrer formula.

    For electron diffraction (small-angle approximation, λ → 0):

        D = K / Δg

    where Δg is the FWHM of the peak in reciprocal-space units (1/nm) and
    K ≈ 0.94 for spherical domains.

    Conversion chain
    ----------------
    1. The FFT was computed on a small image of shape (H, W) with pixel size
       ``pixel_size_small = pixel_size_nm / scale`` (nm/px).
    2. A frequency offset of r FFT-pixels corresponds to |g| = r / (sqrt(H·W) · pixel_size_small)
       in 1/nm.
    3. The Gaussian σ in FFT-pixels converts to σ_g = σ_px / (sqrt(H·W) · pixel_size_small) in 1/nm.
    4. FWHM_g = 2·sqrt(2·ln 2) · σ_g ≈ 2.355 · σ_g.
    5. D = K / FWHM_g.

    Parameters
    ----------
    sigma_px : float — fitted Gaussian σ (in FFT pixels).
    fft_shape : (H, W) — shape of the FFT array.
    pixel_size_nm : float — pixel size of the ORIGINAL image (nm/px).
    scale : float — downscale factor (small → original).
    K : float — Scherrer constant (default 0.94).

    Returns
    -------
    float — coherent-domain size D in nm.
    """
    if sigma_px <= 0:
        return float("inf")
    h, w = fft_shape
    pixel_size_small = pixel_size_nm / scale
    avg_dim = float(np.sqrt(h * w))
    sigma_g = sigma_px / (avg_dim * pixel_size_small)  # 1/nm
    if sigma_g <= 0:
        return float("inf")
    fwhm_g = 2.35482 * sigma_g  # 2·sqrt(2·ln 2)
    return K / fwhm_g


def evaluate_reflection_quality(
    log_power: np.ndarray,
    click_r: float,
    click_c: float,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    p: FFTSegmentationParams | None = None,
) -> ReflectionQuality:
    """Evaluate the quality of a manually-clicked reflection.

    Fits a 2D Gaussian in a window around the click, computes σ, SNR, the
    d-spacing at the click radius, and the Scherrer coherent-domain size.
    Classifies the click as "good", "borderline", or "noise" based on σ, SNR
    and the sanity of D_coh.

    The fitted center (which may differ from the click by a few pixels) is
    also returned so the UI can show where the actual peak is.
    """
    if p is None:
        p = FFTSegmentationParams()
    h, w = fft_shape

    # d-spacing at the click radius (used for display even if fit fails).
    center_r, center_c = h / 2.0, w / 2.0
    radius = float(np.sqrt((click_r - center_r) ** 2 + (click_c - center_c) ** 2))
    d_nm = compute_d_spacing_for_radius(radius, fft_shape, pixel_size_nm, scale) if radius > 0 else 0.0

    fit = fit_gaussian_2d(log_power, click_r, click_c, p.reflection_fit_window_px)
    if fit is None:
        return ReflectionQuality(
            status="noise",
            d_spacing_nm=d_nm,
            fit_success=False,
            message="Fit failed (window out of bounds or flat). Likely noise.",
        )

    sigma_mean = float(np.sqrt(fit["sr"] * fit["sc"]))
    snr = fit["amplitude"] / max(abs(fit["background"]), 1e-9)
    D_coh = estimate_coherent_size_scherrer(
        sigma_mean, fft_shape, pixel_size_nm, scale, p.scherrer_K
    )

    # Sanity: image size in nm (upper bound on D_coh).
    image_size_nm = max(h, w) * pixel_size_nm / scale

    # Classification (combined σ + SNR + D_coh sanity checks).
    if snr < p.snr_min:
        status = "noise"
        msg = (f"Low SNR ({snr:.1f} < {p.snr_min:.0f}) — likely noise. "
               f"Undo recommended.")
    elif sigma_mean >= p.sigma_borderline_px and D_coh < 0.3:
        # Both wide AND tiny coherent size → noise.
        status = "noise"
        msg = (f"Too broad (σ={sigma_mean:.1f}px, D≈{D_coh:.2f} nm). "
               f"Not a Bragg reflection.")
    elif sigma_mean >= p.reflection_fit_window_px / 2.5:
        # σ approaches the window half-size → fit captured the whole window,
        # no localized peak.
        status = "noise"
        msg = (f"Peak fills the whole window (σ={sigma_mean:.1f}px). "
               f"No localized maximum — likely noise.")
    elif D_coh > image_size_nm:
        # Unreasonably narrow peak — likely a single-pixel artifact.
        status = "borderline"
        msg = (f"Very narrow (D≈{D_coh:.0f} nm > image). Possible artifact; "
               f"check the 2D spectrum.")
    elif sigma_mean >= p.sigma_borderline_px:
        # Wide peak — could be small grains (Scherrer broadening) OR noise.
        # Decide by SNR: high SNR + wide = small grains (real); low SNR + wide = noise.
        if snr >= p.snr_good:
            status = "borderline"
            msg = (f"Broad peak (σ={sigma_mean:.1f}px) but high SNR ({snr:.1f}) — "
                   f"small grains, D≈{D_coh:.1f} nm.")
        else:
            status = "noise"
            msg = (f"Broad and weak (σ={sigma_mean:.1f}px, SNR={snr:.1f}) — "
                   f"likely noise.")
    elif sigma_mean >= p.sigma_good_px:
        # Moderately narrow peak.
        if snr >= p.snr_good:
            status = "good"
            msg = (f"Good Bragg peak (σ={sigma_mean:.1f}px, SNR={snr:.1f}, "
                   f"D≈{D_coh:.1f} nm).")
        else:
            status = "borderline"
            msg = (f"Marginal SNR ({snr:.1f}). σ={sigma_mean:.1f}px, "
                   f"D≈{D_coh:.1f} nm.")
    else:
        # Narrow peak — looks like a clean Bragg reflection.
        if snr >= p.snr_good:
            status = "good"
            msg = (f"Sharp Bragg peak (σ={sigma_mean:.1f}px, SNR={snr:.1f}, "
                   f"D≈{D_coh:.1f} nm).")
        else:
            status = "borderline"
            msg = (f"Sharp but weak (σ={sigma_mean:.1f}px, SNR={snr:.1f}). "
                   f"Check the 2D spectrum.")

    return ReflectionQuality(
        status=status,
        sigma_mean_px=sigma_mean,
        sigma_r_px=fit["sr"],
        sigma_c_px=fit["sc"],
        amplitude=fit["amplitude"],
        background=fit["background"],
        snr=snr,
        d_spacing_nm=d_nm,
        coherent_size_nm=D_coh,
        fit_success=True,
        center_r_fft=fit["center_r_fft"],
        center_c_fft=fit["center_c_fft"],
        message=msg,
    )


# === Reference d-spacings (mode B) ===

# Material presets — common TEM materials with their strongest reflections.
# Values are in nm; for cubic materials d_hkl = a / sqrt(h²+k²+l²).
# Sources: standard crystallographic tables (ICDD cards, Kittel).
MATERIAL_PRESETS: dict[str, dict] = {
    "Si": {
        "structure": "Diamond cubic",
        "a_nm": 0.5431,
        "d_spacings": [
            ("{111}", 0.3135, 0.3150),
            ("{220}", 0.1919, 0.1927),
            ("{311}", 0.1637, 0.1645),
            ("{400}", 0.1357, 0.1362),
            ("{331}", 0.1245, 0.1250),
            ("{422}", 0.1108, 0.1114),
        ],
    },
    "Ge": {
        "structure": "Diamond cubic",
        "a_nm": 0.5658,
        "d_spacings": [
            ("{111}", 0.3262, 0.3280),
            ("{220}", 0.1997, 0.2008),
            ("{311}", 0.1705, 0.1715),
            ("{400}", 0.1413, 0.1422),
            ("{331}", 0.1295, 0.1303),
            ("{422}", 0.1154, 0.1162),
        ],
    },
    "TiN": {
        "structure": "NaCl-type (FCC)",
        "a_nm": 0.4242,
        "d_spacings": [
            ("{111}", 0.2443, 0.2454),
            ("{200}", 0.2116, 0.2126),
            ("{220}", 0.1497, 0.1504),
            ("{311}", 0.1277, 0.1283),
            ("{222}", 0.1222, 0.1227),
        ],
    },
    "Al": {
        "structure": "FCC",
        "a_nm": 0.4049,
        "d_spacings": [
            ("{111}", 0.2336, 0.2346),
            ("{200}", 0.2022, 0.2031),
            ("{220}", 0.1430, 0.1437),
            ("{311}", 0.1220, 0.1226),
            ("{222}", 0.1168, 0.1173),
        ],
    },
    "Au": {
        "structure": "FCC",
        "a_nm": 0.4078,
        "d_spacings": [
            ("{111}", 0.2352, 0.2362),
            ("{200}", 0.2036, 0.2045),
            ("{220}", 0.1440, 0.1447),
            ("{311}", 0.1228, 0.1234),
            ("{222}", 0.1176, 0.1181),
        ],
    },
    "Pt": {
        "structure": "FCC",
        "a_nm": 0.3923,
        "d_spacings": [
            ("{111}", 0.2263, 0.2273),
            ("{200}", 0.1961, 0.1969),
            ("{220}", 0.1387, 0.1393),
            ("{311}", 0.1183, 0.1189),
            ("{222}", 0.1132, 0.1137),
        ],
    },
    "Custom": {
        "structure": "User-defined",
        "a_nm": 0.0,
        "d_spacings": [],
    },
}


def d_spacing_to_radius_range(
    d_min_nm: float,
    d_max_nm: float,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    tolerance_pct: float = 0.0,
) -> tuple[float, float]:
    """Convert a d-spacing range [d_min, d_max] to an FFT-radius range.

    The conversion uses the same formula as compute_d_spacing_for_radius but
    inverted: r = avg_dim × pixel_size_small / d.

    The tolerance is applied symmetrically: the resulting range is widened by
    tolerance_pct on each side, so a 3% tolerance makes the band ~6% wider.

    Returns
    -------
    (r_min_px, r_max_px) in FFT pixels.
    """
    h, w = fft_shape
    pixel_size_small = pixel_size_nm / scale
    avg_dim = float(np.sqrt(h * w))
    if d_max_nm <= 0 or d_min_nm <= 0:
        return (0.0, 0.0)
    # Larger d → smaller r, so d_max → r_min, d_min → r_max.
    r_min = avg_dim * pixel_size_small / d_max_nm
    r_max = avg_dim * pixel_size_small / d_min_nm
    # Apply tolerance.
    if tolerance_pct > 0:
        tol = tolerance_pct / 100.0
        r_min = max(0.0, r_min * (1.0 - tol))
        r_max = r_max * (1.0 + tol)
    return (r_min, r_max)


def find_reflections_via_reference(
    log_power: np.ndarray,
    reference_d_spacings: list,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    p: FFTSegmentationParams,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Detect reflections ONLY inside user-specified d-spacing bands (mode B).

    For each (label, d_min, d_max) entry:
      1. Convert to a radial band [r_min, r_max] (with tolerance).
      2. Compute the 1D radial spectrum S(r) restricted to [r_min, r_max].
      3. Find 1D peaks INSIDE this band — each 1D peak = one sub-cluster
         (one real d-spacing). This is critical: a wide user range like
         0.13–0.35 nm typically contains SEVERAL different d-spacings, and
         treating them all as one cluster gives an empty amplitude map (the
         midpoint radius usually doesn't coincide with any real reflection).
      4. For each 1D peak, find 2D local maxima in a thin ring around it
         (width = ring_mask_width_px). Each 2D maximum is a reflection.
      5. (Optional) Apply the Gaussian quality filter to drop noise maxima.
      6. Attach the user's label to all sub-clusters of this band.

    If NO 1D peaks are found inside the band (e.g. the user's range doesn't
    contain any real reflection), the band is skipped silently.

    Parameters
    ----------
    log_power : (H, W) float — Wiener-filtered log power spectrum.
    reference_d_spacings : list of (label, d_min_nm, d_max_nm).
    fft_shape : (H, W).
    pixel_size_nm : pixel size of the ORIGINAL image (nm/px).
    scale : downscale factor (small → original).
    p : FFTSegmentationParams.

    Returns
    -------
    (reflection_coords, cluster_labels_per_reflection, peak_radii, labels)
        reflection_coords: (N, 2) of (row, col).
        cluster_labels_per_reflection: (N,) int — sequential cluster id.
        peak_radii: list of 1D-peak radii (FFT px), one per cluster.
        labels: list of strings (the user label, e.g. "{111}"), one per cluster.
    """
    h, w = fft_shape
    center_r, center_c = h / 2.0, w / 2.0

    # Pre-compute the global 1D radial spectrum (used for peak detection
    # inside each band — restrict the search afterwards).
    radii_full, intens_full = compute_radial_spectrum(log_power)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)

    all_coords: list[np.ndarray] = []
    all_labels: list[int] = []
    peak_radii_out: list[int] = []
    labels_out: list[str] = []
    next_cid = 0

    for band_idx, (label, d_min, d_max) in enumerate(reference_d_spacings):
        if d_max <= d_min or d_min <= 0:
            continue
        r_min, r_max = d_spacing_to_radius_range(
            d_min, d_max, fft_shape, pixel_size_nm, scale, p.reference_tolerance_pct
        )
        if r_max <= 0 or r_min <= 0:
            continue
        # Cap r_max to a sane value (avoid going beyond the FFT array).
        max_radius = max(h, w) // 2
        r_max = min(r_max, max_radius - 1)
        if r_min >= r_max:
            continue

        # --- Subdivide the wide band by 1D peaks inside [r_min, r_max] ---
        # Restrict the radial spectrum to this band.
        valid = (radii_full >= r_min) & (radii_full <= r_max)
        if not valid.any():
            continue
        r_in = radii_full[valid]
        s_in = intens_full[valid]
        # Smooth (same as find_radial_peaks).
        from scipy.ndimage import gaussian_filter1d
        s_smooth = gaussian_filter1d(s_in, sigma=p.radial_smoothing_sigma) \
            if p.radial_smoothing_sigma > 0 else s_in
        # Adaptive thresholds (same logic as find_radial_peaks but local).
        baseline = float(np.quantile(s_smooth, p.radial_baseline_quantile))
        high_q = float(np.quantile(s_smooth, p.radial_high_quantile))
        prom_q = float(np.quantile(s_smooth, p.radial_prom_quantile))
        height_threshold = baseline + p.radial_height_factor * (high_q - baseline)
        prominence_threshold = p.radial_prominence_factor * max(prom_q - baseline, 1e-12)
        peaks_1d, _ = find_peaks(
            s_smooth,
            height=max(height_threshold, baseline + 1e-9),
            prominence=max(prominence_threshold, 1e-9),
            distance=max(1, p.radial_peak_min_distance_px),
        )
        band_peak_radii = [int(r_in[i]) for i in peaks_1d]

        # If NO 1D peak found in the band, fall back to the band's midpoint
        # (this preserves the previous behavior for narrow bands where the
        # 1D detector might miss a single reflection due to threshold).
        if not band_peak_radii:
            band_peak_radii = [int(round((r_min + r_max) / 2.0))]

        ring_half_width = max(2.0, p.ring_mask_width_px / 2.0)
        for r_peak in band_peak_radii:
            # Thin ring around the 1D peak (NOT the whole band).
            ring_mask = (dist >= r_peak - ring_half_width) & (dist <= r_peak + ring_half_width)
            ring_power = np.where(ring_mask, log_power, 0.0)
            if ring_power.max() <= 0:
                continue
            coords = feature.peak_local_max(
                ring_power,
                min_distance=max(2, p.min_peak_distance_px),
                threshold_abs=p.radial_ring_local_max_frac * float(ring_power.max()),
                exclude_border=False,
            )
            # --- Q1: quality filter on auto-reflections ---
            if p.apply_quality_filter_to_auto and len(coords) > 0:
                good_coords = []
                for c in coords:
                    q = evaluate_reflection_quality(
                        log_power, float(c[0]), float(c[1]),
                        (h, w), pixel_size_nm, scale, p,
                    )
                    if q.status != "noise":
                        good_coords.append(c)
                coords = np.array(good_coords, dtype=np.int64) if good_coords else \
                    np.empty((0, 2), dtype=np.int64)
            if len(coords) == 0:
                continue
            for c in coords:
                all_coords.append(c)
                all_labels.append(next_cid)
            peak_radii_out.append(int(r_peak))
            labels_out.append(str(label))
            next_cid += 1

    if not all_coords:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.int32),
                [], [])

    return (np.array(all_coords, dtype=np.int64),
            np.array(all_labels, dtype=np.int32),
            peak_radii_out,
            labels_out)


def label_for_d_spacing(
    d_nm: float,
    reference_d_spacings: list,
) -> str:
    """Find the label of the reference range closest to a given d-spacing.

    Returns the label of the range [d_min, d_max] that contains d_nm, or
    the closest one if none strictly contains it. Returns "" if no
    reference ranges are provided.
    """
    if not reference_d_spacings or d_nm <= 0:
        return ""
    best_label = ""
    best_dist = float("inf")
    for label, d_min, d_max in reference_d_spacings:
        if d_min <= d_nm <= d_max:
            return str(label)
        # Closest edge.
        d_edge = d_min if d_nm < d_min else d_max
        dist = abs(d_nm - d_edge)
        if dist < best_dist:
            best_dist = dist
            best_label = str(label)
    return best_label


def _cluster_reflections_by_radius(
    reflection_coords: np.ndarray,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    p: FFTSegmentationParams,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Cluster reflections by **radius from FFT center** (1D DBScan).

    This groups reflections with the same spatial frequency |g| = 1/d,
    regardless of their angular position. Each cluster = one d-spacing.

    Parameters
    ----------
    reflection_coords : (N, 2) array of (row, col) in FFT array.
    fft_shape : (H, W) shape of the FFT array.
    pixel_size_nm : pixel size of the ORIGINAL image in nm/px.
    scale : downscale factor (small→original).
    p : FFTSegmentationParams.

    Returns
    -------
    (cluster_labels, d_spacings_nm, cluster_radii_px)
        cluster_labels: (N,) int — cluster id per reflection (-1 = noise).
        d_spacings_nm: list of d-spacings in nm, one per cluster (sorted by radius).
        cluster_radii_px: list of mean radii in FFT-pixels, one per cluster.
    """
    if len(reflection_coords) == 0:
        return np.array([], dtype=np.int32), [], []

    h, w = fft_shape
    center_r, center_c = h / 2.0, w / 2.0

    # Compute radius of each reflection from center.
    radii = np.sqrt(
        (reflection_coords[:, 0] - center_r) ** 2
        + (reflection_coords[:, 1] - center_c) ** 2
    )

    # Compute radius and d-spacing per reflection.
    # d = 1/|g| where |g| = sqrt((dr/(H*Δ))² + (dc/(W*Δ))²)
    # Δ = pixel_size_nm / scale (downscaled image has LARGER pixels)
    pixel_size_small = pixel_size_nm / scale  # nm/px in the small image
    fy = (reflection_coords[:, 0] - center_r) / (h * pixel_size_small)
    fx = (reflection_coords[:, 1] - center_c) / (w * pixel_size_small)
    g_mag = np.sqrt(fy ** 2 + fx ** 2)  # spatial frequency in 1/nm
    # Avoid division by zero.
    g_mag = np.where(g_mag > 0, g_mag, 1e-12)
    d_per_refl = 1.0 / g_mag  # d-spacing in nm per reflection

    # 1D DBScan on radii.
    radii_2d = radii.reshape(-1, 1)
    db = DBSCAN(eps=p.dbscan_eps, min_samples=p.dbscan_min_samples)
    labels = db.fit_predict(radii_2d)

    # Compute mean radius and d-spacing per cluster.
    unique_clusters = sorted(set(labels[labels >= 0].tolist()))
    d_spacings_nm = []
    cluster_radii_px = []
    for cid in unique_clusters:
        mask = labels == cid
        mean_radius = float(radii[mask].mean())
        cluster_radii_px.append(mean_radius)
        # Mean d-spacing of reflections in this cluster.
        d_spacings_nm.append(float(d_per_refl[mask].mean()))

    return labels.astype(np.int32), d_spacings_nm, cluster_radii_px


def _create_ring_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    width: float,
) -> np.ndarray:
    """Create a ring (annulus) mask at a given radius and width.

    This is the **ring filter** from Grachev §2.4.2 (рис. 2.15б).
    All FFT pixels within [radius - width/2, radius + width/2] from center
    are included.

    Returns
    -------
    np.ndarray of bool, shape (H, W)
    """
    h, w = shape
    cr, cc = center
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cr) ** 2 + (xx - cc) ** 2)
    return (dist >= radius - width / 2) & (dist <= radius + width / 2)


def _normalize_contrast(amp_map: np.ndarray, target_ratio: float = 0.3) -> np.ndarray:
    """Normalize contrast so that mean/max ≈ target_ratio (Grachev §2.4.2).

    Grachev empirically found that the image is most contrastful when
    the ratio of mean intensity to max intensity is close to 0.3.

    Parameters
    ----------
    amp_map : (H, W) float64 — amplitude map.
    target_ratio : target mean/max ratio.

    Returns
    -------
    np.ndarray (H, W) float64 — normalized amplitude map.
    """
    max_val = float(amp_map.max())
    if max_val <= 0:
        return amp_map
    mean_val = float(amp_map.mean())
    if mean_val <= 0:
        return amp_map
    current_ratio = mean_val / max_val
    if current_ratio < 1e-6:
        return amp_map
    # Gamma correction to achieve target_ratio.
    gamma = np.log(target_ratio) / np.log(max(current_ratio, 1e-6))
    gamma = np.clip(gamma, 0.1, 10.0)
    normalized = np.power(amp_map / max_val, gamma) * max_val
    return normalized


def _compute_amplitude_map(fft_shifted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inverse-FFT with a mask applied; return the amplitude.

    Returns
    -------
    np.ndarray (H, W) float64 — amplitude of the filtered image.
    """
    filtered = fft_shifted * mask
    img_filtered = ifft2(ifftshift(filtered))
    return np.abs(img_filtered)


def _label_pixels(amplitude_maps: np.ndarray, p: FFTSegmentationParams) -> np.ndarray:
    """Assign each pixel to the cluster with the maximum amplitude.

    Parameters
    ----------
    amplitude_maps : np.ndarray (C, H, W) float64
    p : FFTSegmentationParams

    Returns
    -------
    np.ndarray (H, W) int32 — cluster id per pixel (0 = background).
    """
    if amplitude_maps.shape[0] == 0:
        return np.zeros(amplitude_maps.shape[1:], dtype=np.int32)
    # Global max across all maps (for thresholding).
    global_max = amplitude_maps.max()
    if global_max <= 0:
        return np.zeros(amplitude_maps.shape[1:], dtype=np.int32)
    # Per-pixel argmax → cluster index (0-based).
    argmax = np.argmax(amplitude_maps, axis=0)  # (H, W)
    # Per-pixel max value (for thresholding).
    pixel_max = amplitude_maps.max(axis=0)  # (H, W)
    # Label: cluster id + 1 (so 0 = background); 0 where below threshold.
    threshold = p.min_amplitude_frac * global_max
    labels = np.where(pixel_max >= threshold, argmax + 1, 0).astype(np.int32)
    return labels


def _postprocess_labels(labels: np.ndarray, p: FFTSegmentationParams) -> np.ndarray:
    """Remove small regions, fill holes, exclude border-touching grains.

    Vectorized for speed (avoids regionprops loop on images with many regions).

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
    p : FFTSegmentationParams

    Returns
    -------
    np.ndarray (H, W) int32 — cleaned labels.
    """
    if labels.max() == 0:
        return labels
    h, w = labels.shape

    # 1. Remove small regions (vectorized via np.bincount).
    counts = np.bincount(labels.ravel())
    small_labels = np.where(counts < p.min_grain_area_px)[0]
    small_labels = small_labels[small_labels > 0]  # exclude background
    if len(small_labels) > 0:
        mask_remove = np.isin(labels, small_labels)
        labels = labels.copy()
        labels[mask_remove] = 0

    if labels.max() == 0:
        return labels

    # 2. Fill holes (global, not per-label, for speed).
    # Build a binary mask of all grain pixels, fill holes, add back.
    grain_mask = labels > 0
    filled_mask = ndi.binary_fill_holes(grain_mask)
    # New pixels (filled holes) get assigned to the nearest grain label.
    new_pixels = filled_mask & ~grain_mask
    if new_pixels.any():
        # Assign each new pixel the label of the nearest grain pixel.
        # Use nearest-neighbor interpolation via distance transform.
        # Simple approach: dilate labels into the holes.
        # For speed, use cv2.dilate with a small kernel iteratively.
        labels_u8 = labels.astype(np.uint16)
        for _ in range(3):  # a few dilation iterations to fill small holes
            dilated = cv2.dilate(labels_u8, np.ones((3, 3), np.uint8), iterations=1)
            # Only fill where there are holes.
            fill_mask = new_pixels & (labels == 0)
            labels[fill_mask] = dilated[fill_mask].astype(labels.dtype)
            new_pixels = filled_mask & (labels == 0)
            if not new_pixels.any():
                break

    # 3. Exclude border-touching grains (vectorized).
    margin = p.border_margin_px
    border_labels = set()
    if p.exclude_left_right_borders:
        border_labels.update(labels[:, :margin].ravel().tolist())
        border_labels.update(labels[:, w - margin:].ravel().tolist())
    if p.exclude_top_bottom_borders:
        border_labels.update(labels[:margin, :].ravel().tolist())
        border_labels.update(labels[h - margin:, :].ravel().tolist())
    border_labels.discard(0)
    if border_labels:
        mask_remove = np.isin(labels, list(border_labels))
        labels[mask_remove] = 0

    # 4. Relabel compactly (vectorized).
    unique = np.unique(labels)
    unique = unique[unique > 0]
    relabel_lut = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    for new, old in enumerate(unique, start=1):
        relabel_lut[old] = new
    labels = relabel_lut[labels]

    return labels


# === Main entry point ===

def segment_grains_fft(image: np.ndarray, pixel_size_nm: float,
                        params: FFTSegmentationParams | None = None) -> FFTSegmentationResult:
    """Run the FFT-based grain segmentation pipeline.

    Parameters
    ----------
    image : np.ndarray (H, W) uint8
        Grayscale image of the layer to analyse.
    pixel_size_nm : float
        Pixel size in nm/px (used only for documentation; the segmentation
        itself works in pixel units).
    params : FFTSegmentationParams, optional

    Returns
    -------
    FFTSegmentationResult
    """
    if params is None:
        params = FFTSegmentationParams()
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h_orig, w_orig = image.shape

    # 0. Downscale for speed/memory (FFT + amplitude maps are O(H*W) per cluster).
    scale = min(1.0, float(params.max_image_dim) / max(h_orig, w_orig))
    if scale < 1.0:
        h_s = int(h_orig * scale)
        w_s = int(w_orig * scale)
        img_small = cv2.resize(image, (w_s, h_s), interpolation=cv2.INTER_AREA)
    else:
        img_small = image
        scale = 1.0

    # 1. Preprocess.
    img = _preprocess(img_small, params)

    # 2. FFT.
    fft_shifted, log_power = _compute_power_spectrum(img)

    # 2b. Wiener filter — suppress amorphous-matrix noise (Grachev §2.4.1).
    # This cleans the power spectrum so reflections stand out clearly.
    if params.use_wiener_filter:
        fft_shifted, log_power = _apply_wiener_filter(fft_shifted, params)

    # Always compute the 1D radial spectrum (cheap; used for visualization
    # even in legacy detection mode).
    radial_radii, radial_intensities = compute_radial_spectrum(log_power)

    # 3. Detect reflections.
    d_spacing_labels: list[str] = []  # populated only in reference mode
    if params.use_reference_d_spacings and params.reference_d_spacings:
        # Mode B: probe ONLY the user-specified d-spacing bands.
        reflection_coords, cluster_labels, peak_radii, d_spacing_labels = \
            find_reflections_via_reference(
                log_power, params.reference_d_spacings,
                img_small.shape, pixel_size_nm, scale, params,
            )
        unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0].tolist())) if len(cluster_labels) > 0 else []
        cluster_radii_px = [float(peak_radii[c]) for c in unique_clusters] if peak_radii else []
        d_spacings_nm = [
            compute_d_spacing_for_radius(r, img_small.shape, pixel_size_nm, scale)
            for r in cluster_radii_px
        ]
        # d_spacing_labels is already indexed by band (= cluster id).
    elif params.use_radial_profile_detection:
        # Recommended: 1D radial profile + 2D local maxima at peak radii.
        reflection_coords, cluster_labels, peak_radii = find_reflections_via_radial_profile(
            log_power, params, pixel_size_nm, scale
        )
        # Build cluster-level lists directly from peak_radii.
        unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0].tolist())) if len(cluster_labels) > 0 else []
        cluster_radii_px = [float(peak_radii[c]) for c in unique_clusters] if peak_radii else []
        d_spacings_nm = [
            compute_d_spacing_for_radius(r, img_small.shape, pixel_size_nm, scale)
            for r in cluster_radii_px
        ]
        # Try to attach labels from reference ranges (if any) by closest match.
        if params.reference_d_spacings:
            d_spacing_labels = [
                label_for_d_spacing(d, params.reference_d_spacings)
                for d in d_spacings_nm
            ]
    else:
        # Legacy: 2D peak_local_max + 1D DBScan by radius.
        reflection_coords = _find_reflections(log_power, params)
        peak_radii = []
        cluster_labels, d_spacings_nm, cluster_radii_px = (
            np.array([], dtype=np.int32), [], []
        )
        if len(reflection_coords) > 0:
            cluster_labels, d_spacings_nm, cluster_radii_px = _cluster_reflections_by_radius(
                reflection_coords, img_small.shape, pixel_size_nm, scale, params
            )
        unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0].tolist())) if len(cluster_labels) > 0 else []
        if params.reference_d_spacings:
            d_spacing_labels = [
                label_for_d_spacing(d, params.reference_d_spacings)
                for d in d_spacings_nm
            ]

    if len(reflection_coords) == 0:
        return FFTSegmentationResult(
            labels=np.zeros(image.shape, dtype=np.int32),
            n_grains=0, n_clusters=0,
            power_spectrum=log_power,
            reflection_coords=reflection_coords,
            cluster_labels=np.array([], dtype=np.int32),
            orientation_map=np.zeros(img_small.shape, dtype=np.int32),
            amplitude_maps=np.zeros((0,) + img_small.shape, dtype=np.float32),
            radial_radii=radial_radii,
            radial_intensities=radial_intensities,
            peak_radii=peak_radii,
            scale=scale,
            d_spacing_labels=d_spacing_labels,
            reference_d_spacings_used=list(params.reference_d_spacings) if params.reference_d_spacings else [],
        )

    # 4b. Limit to max_clusters BEFORE computing amplitude maps (memory!).
    if len(unique_clusters) > params.max_clusters:
        cluster_intensities = []
        for cid in unique_clusters:
            mask = cluster_labels == cid
            coords = reflection_coords[mask]
            intensities = log_power[coords[:, 0], coords[:, 1]]
            cluster_intensities.append((cid, float(intensities.sum())))
        cluster_intensities.sort(key=lambda x: x[1], reverse=True)
        unique_clusters = [cid for cid, _ in cluster_intensities[:params.max_clusters]]
        # Filter d_spacings and radii to match.
        d_spacings_nm = [d_spacings_nm[c] for c in range(len(d_spacings_nm))
                         if c in unique_clusters]
        cluster_radii_px = [cluster_radii_px[c] for c in range(len(cluster_radii_px))
                            if c in unique_clusters]
        if d_spacing_labels:
            d_spacing_labels = [d_spacing_labels[c] for c in range(len(d_spacing_labels))
                                if c in unique_clusters]

    # 5. Per-cluster amplitude maps using RING masks (Grachev §2.4.2).
    center = (img_small.shape[0] / 2.0, img_small.shape[1] / 2.0)
    amplitude_maps = []
    for i, cid in enumerate(unique_clusters):
        radius = cluster_radii_px[i] if i < len(cluster_radii_px) else 0
        mask = _create_ring_mask(img_small.shape, center, radius, params.ring_mask_width_px)
        amp = _compute_amplitude_map(fft_shifted, mask)
        # Contrast normalization (Grachev §2.4.2: mean/max ≈ 0.3).
        amp = _normalize_contrast(amp, params.contrast_target_ratio)
        amplitude_maps.append(amp)
    amplitude_maps = np.stack(amplitude_maps, axis=0) if amplitude_maps else np.zeros(
        (0,) + img_small.shape, dtype=np.float64
    )

    # 6. Label pixels by argmax (on the small image).
    orientation_map_small = _label_pixels(amplitude_maps, params)

    # 7. Connected-component labeling to split grains with the same orientation.
    labels_small = np.zeros_like(orientation_map_small)
    next_label = 1
    for cid in range(1, orientation_map_small.max() + 1):
        region_mask = (orientation_map_small == cid)
        if not region_mask.any():
            continue
        cc_labels, n_cc = ndi.label(region_mask)
        for cc_id in range(1, n_cc + 1):
            labels_small[cc_labels == cc_id] = next_label
            next_label += 1

    # 7b. Post-process on the SMALL image (fast), then upscale.
    # Scale min_grain_area_px and fill_holes_max_px to the small image.
    small_params = FFTSegmentationParams(**{**asdict(params),
        "min_grain_area_px": max(1, int(params.min_grain_area_px * scale * scale)),
        "fill_holes_max_px": max(1, int(params.fill_holes_max_px * scale * scale)),
    })
    labels_small = _postprocess_labels(labels_small, small_params)

    # 7c. Upscale labels to original resolution (nearest-neighbor).
    if scale < 1.0:
        labels_full = cv2.resize(
            labels_small.astype(np.uint16),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        orientation_map_full = cv2.resize(
            orientation_map_small.astype(np.uint16),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    else:
        labels_full = labels_small
        orientation_map_full = orientation_map_small

    # 8. Final cleanup on full-res: remove tiny upscale artifacts.
    # Use np.bincount for speed (regionprops loop is too slow on full-res
    # images with thousands of regions).
    if labels_full.max() > 0 and scale < 1.0:
        # Count pixels per label (vectorized).
        counts = np.bincount(labels_full.ravel())
        # Labels to remove: 0 (background, skip) and those with count < min_grain_area_px.
        small_labels = np.where(counts < params.min_grain_area_px)[0]
        small_labels = small_labels[small_labels > 0]  # exclude background
        if len(small_labels) > 0:
            mask_remove = np.isin(labels_full, small_labels)
            labels_full[mask_remove] = 0
        # Relabel compactly using np.unique (faster than dict loop).
        unique = np.unique(labels_full)
        unique = unique[unique > 0]
        relabel_lut = np.zeros(int(labels_full.max()) + 1, dtype=labels_full.dtype)
        for new, old in enumerate(unique, start=1):
            relabel_lut[old] = new
        labels_full = relabel_lut[labels_full]

    n_grains = int(labels_full.max()) if labels_full.max() > 0 else 0

    return FFTSegmentationResult(
        labels=labels_full.astype(np.int32),
        n_grains=n_grains,
        n_clusters=len(unique_clusters),
        power_spectrum=log_power,
        reflection_coords=reflection_coords,
        cluster_labels=cluster_labels,
        orientation_map=orientation_map_full,
        amplitude_maps=amplitude_maps.astype(np.float32),
        d_spacings_nm=d_spacings_nm,
        d_spacing_labels=d_spacing_labels,
        radial_radii=radial_radii,
        radial_intensities=radial_intensities,
        peak_radii=peak_radii,
        scale=scale,
        reference_d_spacings_used=list(params.reference_d_spacings) if params.reference_d_spacings else [],
    )


def visualize_d_spacings(
    image: np.ndarray,
    pixel_size_nm: float,
    params: FFTSegmentationParams | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Create a Grachev-style color-coded d-spacing visualization.

    This is **Etape 1** (visualization, not segmentation): produces a color
    image where each color = a different interplanar distance (d-spacing).
    Bright pixels = regions where that d-spacing is present; dark = amorphous/noise.

    Parameters
    ----------
    image : (H, W) uint8 — grayscale layer image.
    pixel_size_nm : pixel size in nm/px.
    params : FFTSegmentationParams, optional.

    Returns
    -------
    (visualization_rgb, d_spacings_nm)
        visualization_rgb: (H, W, 3) uint8 — color-coded d-spacing map.
        d_spacings_nm: list of d-spacings found, in nm.
    """
    if params is None:
        params = FFTSegmentationParams()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h_orig, w_orig = image.shape
    scale = min(1.0, float(params.max_image_dim) / max(h_orig, w_orig))
    if scale < 1.0:
        img_small = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)),
                                interpolation=cv2.INTER_AREA)
    else:
        img_small = image
        scale = 1.0

    img = _preprocess(img_small, params)
    fft_shifted, log_power = _compute_power_spectrum(img)
    if params.use_wiener_filter:
        fft_shifted, log_power = _apply_wiener_filter(fft_shifted, params)

    reflection_coords = _find_reflections(log_power, params)
    if len(reflection_coords) == 0:
        return np.zeros(img_small.shape + (3,), dtype=np.uint8), []

    cluster_labels, d_spacings_nm, cluster_radii_px = _cluster_reflections_by_radius(
        reflection_coords, img_small.shape, pixel_size_nm, scale, params
    )
    unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0].tolist()))

    # Limit clusters.
    if len(unique_clusters) > params.max_clusters:
        cluster_intensities = []
        for cid in unique_clusters:
            mask = cluster_labels == cid
            coords = reflection_coords[mask]
            intensities = log_power[coords[:, 0], coords[:, 1]]
            cluster_intensities.append((cid, float(intensities.sum())))
        cluster_intensities.sort(key=lambda x: x[1], reverse=True)
        unique_clusters = [cid for cid, _ in cluster_intensities[:params.max_clusters]]
        d_spacings_nm = [d_spacings_nm[c] for c in range(len(d_spacings_nm))
                         if c in unique_clusters]
        cluster_radii_px = [cluster_radii_px[c] for c in range(len(cluster_radii_px))
                            if c in unique_clusters]

    # Build color-coded visualization (Grachev §2.4.2).
    from skimage import color as skcolor
    n = len(unique_clusters)
    if n == 0:
        return np.zeros(img_small.shape + (3,), dtype=np.uint8), []

    # Assign distinct colors per cluster.
    labels_img = np.arange(n).reshape(-1, 1)
    colored = skcolor.label2rgb(labels_img, bg_label=-1, kind="overlay", alpha=1.0)
    colors = (colored[:, 0, :] * 255).astype(np.uint8)

    center = (img_small.shape[0] / 2.0, img_small.shape[1] / 2.0)
    viz = np.zeros(img_small.shape + (3,), dtype=np.float64)

    for i, cid in enumerate(unique_clusters):
        radius = cluster_radii_px[i] if i < len(cluster_radii_px) else 0
        mask = _create_ring_mask(img_small.shape, center, radius, params.ring_mask_width_px)
        amp = _compute_amplitude_map(fft_shifted, mask)
        amp = _normalize_contrast(amp, params.contrast_target_ratio)
        # Normalize to [0, 1].
        amp_norm = amp / (amp.max() + 1e-12)
        # Add color layer.
        color = colors[i % len(colors)].astype(np.float64)
        for c in range(3):
            viz[:, :, c] += amp_norm * color[c]

    # Clip and convert.
    viz = np.clip(viz, 0, 255).astype(np.uint8)

    # Upscale to original resolution.
    if scale < 1.0:
        viz = cv2.resize(viz, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    return viz, d_spacings_nm


def compute_fft_and_spectrum(image: np.ndarray, params: FFTSegmentationParams | None = None
                              ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute the FFT and Wiener-filtered power spectrum (steps 0–2b).

    This is used by the UI to show the power spectrum for manual reflection
    selection BEFORE running the full segmentation.

    Returns
    -------
    (fft_shifted, log_power, scale, img_small_shape)
        fft_shifted: complex128 — Wiener-filtered shifted FFT.
        log_power: float64 — log power spectrum for display.
        scale: float — downscale factor (small→original).
        img_small_shape: (h, w) — shape of the small image.
    """
    if params is None:
        params = FFTSegmentationParams()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h_orig, w_orig = image.shape
    scale = min(1.0, float(params.max_image_dim) / max(h_orig, w_orig))
    if scale < 1.0:
        img_small = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)),
                                interpolation=cv2.INTER_AREA)
    else:
        img_small = image
        scale = 1.0

    img = _preprocess(img_small, params)
    fft_shifted, log_power = _compute_power_spectrum(img)
    if params.use_wiener_filter:
        fft_shifted, log_power = _apply_wiener_filter(fft_shifted, params)

    return fft_shifted, log_power, scale, img_small.shape


def segment_grains_fft_with_reflections(
    fft_shifted: np.ndarray,
    log_power: np.ndarray,
    reflection_coords: np.ndarray,
    image_shape: tuple[int, int],
    scale: float,
    pixel_size_nm: float = 1.0,
    params: FFTSegmentationParams | None = None,
) -> FFTSegmentationResult:
    """Run FFT segmentation with **manually-specified** reflections.

    This is the second half of the pipeline (steps 4–8), called after the user
    has manually clicked on reflections in the power spectrum. The FFT and
    power spectrum are pre-computed by ``compute_fft_and_spectrum``.

    Parameters
    ----------
    fft_shifted : np.ndarray (H_small, W_small) complex128
        Wiener-filtered shifted FFT (from ``compute_fft_and_spectrum``).
    log_power : np.ndarray (H_small, W_small) float64
        Log power spectrum (from ``compute_fft_and_spectrum``).
    reflection_coords : np.ndarray (N, 2) of (row, col)
        Manually-selected reflection coordinates in the SMALL FFT array.
    image_shape : (H_orig, W_orig)
        Original image shape (for upscaling labels).
    scale : float
        Downscale factor (small→original), from ``compute_fft_and_spectrum``.
    params : FFTSegmentationParams, optional

    Returns
    -------
    FFTSegmentationResult
    """
    if params is None:
        params = FFTSegmentationParams()

    h_orig, w_orig = image_shape
    img_small_shape = fft_shifted.shape

    # Compute the radial spectrum (for the UI plot, even in manual mode).
    radial_radii, radial_intensities = compute_radial_spectrum(log_power)
    # Also run the auto-detector to get peak_radii for the UI plot.
    _, _, auto_peak_radii = find_reflections_via_radial_profile(
        log_power, params, pixel_size_nm, scale
    )

    if len(reflection_coords) == 0:
        return FFTSegmentationResult(
            labels=np.zeros(image_shape, dtype=np.int32),
            n_grains=0, n_clusters=0,
            power_spectrum=log_power,
            reflection_coords=reflection_coords,
            cluster_labels=np.array([], dtype=np.int32),
            orientation_map=np.zeros(img_small_shape, dtype=np.int32),
            amplitude_maps=np.zeros((0,) + img_small_shape, dtype=np.float32),
            radial_radii=radial_radii,
            radial_intensities=radial_intensities,
            peak_radii=auto_peak_radii,
            scale=scale,
        )

    # 4. Cluster reflections by RADIUS (1D DBScan) — same as main pipeline.
    cluster_labels, d_spacings_nm, cluster_radii_px = _cluster_reflections_by_radius(
        reflection_coords, img_small_shape, pixel_size_nm, scale, params
    )
    unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0].tolist()))

    # 4b. Limit to max_clusters.
    if len(unique_clusters) > params.max_clusters:
        cluster_intensities = []
        for cid in unique_clusters:
            mask_cl = cluster_labels == cid
            coords = reflection_coords[mask_cl]
            intensities = log_power[coords[:, 0], coords[:, 1]]
            cluster_intensities.append((cid, float(intensities.sum())))
        cluster_intensities.sort(key=lambda x: x[1], reverse=True)
        unique_clusters = [cid for cid, _ in cluster_intensities[:params.max_clusters]]
        d_spacings_nm = [d_spacings_nm[c] for c in range(len(d_spacings_nm))
                         if c in unique_clusters]
        cluster_radii_px = [cluster_radii_px[c] for c in range(len(cluster_radii_px))
                            if c in unique_clusters]

    # 5. Per-cluster amplitude maps using RING masks.
    center = (img_small_shape[0] / 2.0, img_small_shape[1] / 2.0)
    amplitude_maps = []
    for i, cid in enumerate(unique_clusters):
        radius = cluster_radii_px[i] if i < len(cluster_radii_px) else 0
        mask = _create_ring_mask(img_small_shape, center, radius, params.ring_mask_width_px)
        amp = _compute_amplitude_map(fft_shifted, mask)
        amp = _normalize_contrast(amp, params.contrast_target_ratio)
        amplitude_maps.append(amp)
    amplitude_maps = np.stack(amplitude_maps, axis=0) if amplitude_maps else np.zeros(
        (0,) + img_small_shape, dtype=np.float64
    )

    # 6. Label pixels by argmax.
    orientation_map_small = _label_pixels(amplitude_maps, params)

    # 7. Connected-component labeling.
    labels_small = np.zeros_like(orientation_map_small)
    next_label = 1
    for cid in range(1, orientation_map_small.max() + 1):
        region_mask = (orientation_map_small == cid)
        if not region_mask.any():
            continue
        cc_labels, n_cc = ndi.label(region_mask)
        for cc_id in range(1, n_cc + 1):
            labels_small[cc_labels == cc_id] = next_label
            next_label += 1

    # 7b. Post-process on small image.
    from dataclasses import asdict as _asdict
    small_params = FFTSegmentationParams(**{**_asdict(params),
        "min_grain_area_px": max(1, int(params.min_grain_area_px * scale * scale)),
        "fill_holes_max_px": max(1, int(params.fill_holes_max_px * scale * scale)),
    })
    labels_small = _postprocess_labels(labels_small, small_params)

    # 7c. Upscale to original resolution.
    if scale < 1.0:
        labels_full = cv2.resize(
            labels_small.astype(np.uint16), (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        orientation_map_full = cv2.resize(
            orientation_map_small.astype(np.uint16), (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    else:
        labels_full = labels_small
        orientation_map_full = orientation_map_small

    # 8. Final cleanup.
    if labels_full.max() > 0 and scale < 1.0:
        counts = np.bincount(labels_full.ravel())
        small_labels = np.where(counts < params.min_grain_area_px)[0]
        small_labels = small_labels[small_labels > 0]
        if len(small_labels) > 0:
            mask_remove = np.isin(labels_full, small_labels)
            labels_full[mask_remove] = 0
        unique = np.unique(labels_full)
        unique = unique[unique > 0]
        relabel_lut = np.zeros(int(labels_full.max()) + 1, dtype=labels_full.dtype)
        for new, old in enumerate(unique, start=1):
            relabel_lut[old] = new
        labels_full = relabel_lut[labels_full]

    n_grains = int(labels_full.max()) if labels_full.max() > 0 else 0

    return FFTSegmentationResult(
        labels=labels_full.astype(np.int32),
        n_grains=n_grains,
        n_clusters=len(unique_clusters),
        power_spectrum=log_power,
        reflection_coords=reflection_coords,
        cluster_labels=cluster_labels,
        orientation_map=orientation_map_full,
        amplitude_maps=amplitude_maps.astype(np.float32),
        d_spacings_nm=d_spacings_nm,
        radial_radii=radial_radii,
        radial_intensities=radial_intensities,
        peak_radii=auto_peak_radii,
        scale=scale,
    )


# === Visualization helpers ===

def colorize_orientation_map(orientation_map: np.ndarray) -> np.ndarray:
    """Render the orientation map as an RGB image (one color per orientation).

    Parameters
    ----------
    orientation_map : np.ndarray (H, W) int32
        0 = background, 1..N = orientation clusters.

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    from skimage import color as skcolor
    n = int(orientation_map.max())
    if n == 0:
        return np.zeros(orientation_map.shape + (3,), dtype=np.uint8)
    colored = skcolor.label2rgb(
        orientation_map, bg_label=0, kind="overlay", alpha=1.0, bg_color=(0, 0, 0),
    )
    return (colored * 255).astype(np.uint8)


def render_power_spectrum(log_power: np.ndarray, reflection_coords: np.ndarray | None = None,
                            cluster_labels: np.ndarray | None = None,
                            max_dim: int = 512) -> np.ndarray:
    """Render the log power spectrum as an RGB image (for display).

    Reflections are marked with colored dots, colored by cluster.
    Coordinates are scaled to match the downscaled display image.

    Parameters
    ----------
    log_power : np.ndarray (H, W) float64
    reflection_coords : np.ndarray (N, 2), optional
    cluster_labels : np.ndarray (N,), optional
    max_dim : int
        Downscale the image to this max dimension for display.

    Returns
    -------
    np.ndarray (h, w, 3) uint8
    """
    # Normalize to 0–255.
    p = log_power.copy()
    pmin, pmax = p.min(), p.max()
    if pmax > pmin:
        p = (p - pmin) * 255.0 / (pmax - pmin)
    p = np.clip(p, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)

    # Downscale for display.
    h_orig, w_orig = rgb.shape[:2]
    scale = min(1.0, float(max_dim) / max(h_orig, w_orig))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w_orig * scale), int(h_orig * scale)),
                          interpolation=cv2.INTER_AREA)

    # Draw reflections — SCALE coordinates to match the downscaled image.
    if reflection_coords is not None and len(reflection_coords) > 0:
        # Generate distinct colors per cluster using a fixed colormap.
        n_clusters = 0
        if cluster_labels is not None and len(cluster_labels) > 0:
            n_clusters = int(cluster_labels.max()) + 1
        if n_clusters <= 1:
            # All same color (red).
            colors = np.array([[255, 0, 0]], dtype=np.uint8)
        else:
            # Use a colormap with distinct colors (HSV with varying hue + saturation).
            from skimage import color as skcolor
            labels_img = np.arange(n_clusters).reshape(-1, 1)
            colored = skcolor.label2rgb(labels_img, bg_label=-1, kind="overlay", alpha=1.0)
            colors = (colored[:, 0, :] * 255).astype(np.uint8)

        for i, (r, c) in enumerate(reflection_coords):
            # Scale coordinates to the display image.
            r_s = int(r * scale)
            c_s = int(c * scale)
            cid = int(cluster_labels[i]) if cluster_labels is not None and i < len(cluster_labels) else 0
            if cid < 0:
                color = (128, 128, 128)  # noise = gray
            else:
                color = colors[cid % len(colors)].tolist()
            # Larger circles for visibility (scaled with display size).
            radius = max(5, int(8 * scale))
            cv2.circle(rgb, (c_s, r_s), radius, color, -1)
            # White outline for contrast.
            cv2.circle(rgb, (c_s, r_s), radius, (255, 255, 255), 2)
            # Crosshair lines for precise position visibility.
            cv2.line(rgb, (c_s - radius - 4, r_s), (c_s - radius, r_s), (255, 255, 255), 1)
            cv2.line(rgb, (c_s + radius, r_s), (c_s + radius + 4, r_s), (255, 255, 255), 1)
            cv2.line(rgb, (c_s, r_s - radius - 4), (c_s, r_s - radius), (255, 255, 255), 1)
            cv2.line(rgb, (c_s, r_s + radius), (c_s, r_s + radius + 4), (255, 255, 255), 1)

    return rgb
