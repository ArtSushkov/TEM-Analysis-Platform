"""
click_segmentation.py — Click-to-segment grain analysis (mode B).

User clicks on a grain in the real-space image; the app:
  1. Takes a local window around the click.
  2. Computes a local FFT + Wiener filter.
  3. Finds the d-spacings present in THIS grain (radial profile + quality filter).
  4. For each found d-spacing, computes an amplitude map (via the GLOBAL FFT
     for better SNR, falling back to local FFT if the global detector missed it).
  5. Region-growing from the click point with HYSTERESIS (high seed threshold,
     low growth threshold) AND an AND-condition: a pixel belongs to the grain
     only if ALL its d-spacings have high amplitude simultaneously.
  6. The connected component containing the seed = one grain.

Optional: after one grain is segmented, automatically find all OTHER connected
regions with the same d-spacing signature (so the user can add all similar
grains in one click).

This solves two problems that the global FFT method has:
  - Grains with the same single d-spacing but different orientations are not
    separated (here: the AND-condition on multiple d-spacings separates them).
  - Grains containing multiple plane families are not recognized as one unit
    (here: the local FFT finds all the plane families of the clicked grain).

References
----------
- Grachev D.A., PhD dissertation, §2.4 (FFT-based visualization).
- Hysteresis thresholding: Canny J. (1986).
- Region growing: Zucker S. (1976).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# Reuse helpers from fft_segmentation to keep behaviour consistent.
from .fft_segmentation import (
    FFTSegmentationParams,
    _preprocess,
    _compute_power_spectrum,
    _apply_wiener_filter,
    _create_ring_mask,
    _normalize_contrast,
    _compute_amplitude_map,
    compute_radial_spectrum,
    find_radial_peaks,
    find_reflections_via_radial_profile,
    compute_d_spacing_for_radius,
    evaluate_reflection_quality,
    d_spacing_to_radius_range,
)


# === Parameters ===

@dataclass
class ClickSegmentationParams:
    """Parameters for click-to-segment grain analysis."""
    # --- Local FFT window ---
    # Side of the square window (in original-image pixels) around the click.
    # Must be even. 128 = good default for medium grains; 96 for small grains;
    # 192–256 for large grains. Larger = better FFT SNR but may cross grain
    # boundaries.
    window_size_px: int = 128
    # Zero-pad the local window to this size before FFT (improves frequency
    # resolution without increasing window size). 0 = no padding.
    # 256 = pad to 256×256 (good balance).
    zero_pad_size_px: int = 256

    # --- Local d-spacing detection ---
    # Min relative intensity (frac of local 1D spectrum max) for a peak to
    # qualify as a d-spacing of this grain. Higher = fewer d-spacings (stricter).
    local_peak_height_factor: float = 0.30
    # Smoothing sigma for the local 1D radial spectrum.
    local_smoothing_sigma: float = 2.0
    # Max number of d-spacings to keep per grain (sorted by intensity).
    # 1 = only the strongest d-spacing (relaxes the AND condition).
    max_d_spacings_per_grain: int = 5

    # --- Amplitude maps: sliding local FFT (mode B, recommended) ---
    # When use_sliding_local_fft is True, amplitude maps are built by computing
    # a local FFT in a window sliding over the image (with step sliding_step_px).
    # For each window, we check whether the local FFT has a peak at the
    # d-spacing AND orientation of the clicked grain. This gives per-pixel
    # selectivity by BOTH d-spacing AND orientation — solving the leak problem
    # that arises when only the d-spacing is used (with global FFT + ring/disk
    # masks, neighboring grains with the same d but different orientation get
    # high amplitude and region growing leaks).
    use_sliding_local_fft: bool = True
    # Side of the sliding window (in original-image pixels). 128 = good default
    # (large enough to resolve small d-spacings like 0.2–0.3 nm at typical TEM
    # pixel sizes 0.02–0.05 nm/px). 64 = too small for small d-spacings — the
    # peak falls in the DC region. 192–256 for very large grains.
    # Must be even.
    sliding_window_size_px: int = 128
    # Step of the sliding window (in original-image pixels). Smaller = finer
    # spatial resolution but slower. 16 = good balance. The amplitude map is
    # interpolated to full resolution after.
    sliding_step_px: int = 16
    # Zero-pad the sliding window to this size before FFT (improves frequency
    # resolution without increasing window size). 0 = no padding.
    # NOTE: zero-padding can create artifacts that amplify noise peaks; for
    # the sliding mode, 0 (no padding) often works better than 128.
    sliding_zero_pad_size_px: int = 0
    # Orientation tolerance Δ (in degrees). A pixel is considered to belong to
    # the grain if its local FFT has a peak at the grain's d-spacing AND the
    # peak's angle is within ±Δ of the grain's orientation θ₀.
    # 10° = good default. Larger = more tolerant (catches mosaic spread within
    # a grain) but may merge neighboring grains with similar orientations.
    # Smaller = stricter (better separation) but may fragment a single grain
    # with internal orientation spread.
    orientation_tolerance_deg: float = 10.0

    # --- Amplitude maps (legacy: global FFT with ring/disk masks) ---
    # Used only when use_sliding_local_fft is False.
    use_global_fft_for_amplitude: bool = True
    # Width of the ring mask used for amplitude maps (in FFT pixels).
    ring_mask_width_px: float = 6.0
    # CRITICAL: use DISK masks around individual 2D reflections (with their
    # angle) instead of a RING mask (which averages over all angles).
    # Disk masks separate grains by ORIENTATION as well as by d-spacing —
    # without this, all grains with the same d-spacing get high amplitude
    # and region growing leaks across grain boundaries.
    # When True: each d-spacing's amplitude map is the SUM of disk-mask
    # amplitude maps around each 2D reflection found in the local FFT.
    use_orientation_specific_masks: bool = True
    # Radius of the disk mask around each 2D reflection (in FFT pixels).
    # Smaller = more orientation-selective but may miss the reflection if
    # calibration drifts. 5 = good default (with auto-align, this is robust).
    disk_mask_radius_px: float = 5.0

    # --- Region growing (hysteresis) ---
    # High threshold (for the SEED pixel = the click point). The click must
    # be on a pixel whose min-amplitude across all d-spacings exceeds this.
    # If not, the click is rejected as "off-grain".
    # NOTE: with strict AND (min across d-spacings) + orientation-specific
    # disk masks, the seed amplitude can be lower than expected (one d-spacing
    # may be noisy). 0.15 is more forgiving than 0.30 for real TEM data.
    seed_threshold_frac: float = 0.15
    # Low threshold (for GROWTH). A neighbor pixel is added to the grain if
    # its min-amplitude across all d-spacings exceeds this AND it is adjacent
    # to an already-accepted pixel. Lower than seed_threshold_frac.
    # NOTE: with orientation-specific (disk) masks, this can be lower because
    # the amplitude naturally drops to ~0 outside the grain.
    growth_threshold_frac: float = 0.10

    # --- AND condition ---
    # If True, a pixel is accepted only if ALL d-spacings have amplitude >=
    # threshold (strict AND = MIN across d-spacings). If False, accepts pixels
    # where the MEAN amplitude >= threshold (more forgiving, useful when one
    # d-spacing is weak).
    # DEFAULT: True (strict AND). With orientation-specific disk masks, the MIN
    # is essential — otherwise a single "leaky" d-spacing (e.g. a harmonic
    # whose amplitude map doesn't separate grains by orientation) causes the
    # mean to be high everywhere and region growing leaks.
    strict_and: bool = True

    # --- Safety: max grain area ---
    # If the grown region exceeds this fraction of the layer mask area, the
    # grain is likely a "leak" (region growing escaped the grain boundary).
    # The grain is then cropped to the largest connected component containing
    # the seed that fits within max_grain_area_frac × layer_area.
    max_grain_area_frac: float = 0.30  # 30% of the layer

    # --- Auto-find similar grains ---
    # After segmenting one grain, scan the rest of the layer mask for other
    # connected regions with the same d-spacing signature.
    auto_find_similar: bool = True
    # When auto-finding, min area (in original-image pixels) for a region to
    # qualify as a separate grain.
    min_grain_area_px: int = 100

    # --- Post-processing ---
    fill_holes: bool = True
    # Smooth grain boundaries with morphological open/close (kernel size, 0 = off).
    boundary_smooth_kernel: int = 3

    # --- Structure tensor cleanup (optional post-filter) ---
    # After region growing, judge every pixel of the grain mask with the
    # structure tensor and remove amorphous / wrongly-oriented regions:
    # (1) coherence gate — isotropic or textureless pixels (amorphous,
    #     vacuum) are removed;
    # (2) orientation gate — textured pixels whose fringe direction does not
    #     match ANY of the grain's reflections (within tolerance) are removed;
    # (3) hysteresis — only connected candidate regions larger than
    #     cleanup_min_remove_area_px are actually removed, so fringe gaps and
    #     single noisy pixels survive.
    use_structure_cleanup: bool = False
    # Structure tensor window (px). ~1.5× the lattice-fringe period; 3.0 is
    # a good default for typical TEM d-spacings (2–8 px fringes).
    cleanup_sigma_px: float = 3.0
    # Anisotropy below this = amorphous/textureless → removal candidate.
    cleanup_coherence_threshold: float = 0.35
    # ± tolerance (degrees) for matching the local fringe direction against
    # the selected reflections' FFT angles (mod 180°). 25° is forgiving
    # enough for mosaic spread; 0 disables the orientation gate.
    cleanup_orientation_tolerance_deg: float = 25.0
    # Smallest connected candidate region (px) that is actually removed.
    cleanup_min_remove_area_px: int = 25

    def to_dict(self) -> dict:
        """Serialize to a dict (for JSON report)."""
        from dataclasses import asdict
        return asdict(self)


# === Result dataclasses ===

@dataclass
class GrainSignature:
    """The crystallographic 'fingerprint' of a single grain.

    Two grains with the same signature (within tolerance) are considered the
    same crystallographic variant.
    """
    d_spacings_nm: list[float] = field(default_factory=list)
    # The FFT radius (in the global FFT array) for each d-spacing — used to
    # recompute amplitude maps and to compare signatures.
    radii_px: list[float] = field(default_factory=list)
    # Quality assessments (one per d-spacing).
    qualities: list = field(default_factory=list)  # list[ReflectionQuality]
    # Orientation of each d-spacing (degrees, 0–180°, Friedel-invariant).
    # Populated only in sliding mode; empty in legacy mode.
    orientations_deg: list[float] = field(default_factory=list)

    def matches(self, other: "GrainSignature", tol_nm: float = 0.01,
                tol_deg: float = 10.0) -> bool:
        """Check whether two signatures match (each d within tol_nm of a d in the other)."""
        if not self.d_spacings_nm or not other.d_spacings_nm:
            return False
        if len(self.d_spacings_nm) != len(other.d_spacings_nm):
            return False
        # Greedy matching: for each d in self, find the closest unmatched d in other.
        other_remaining = list(other.d_spacings_nm)
        for d_self in self.d_spacings_nm:
            best_idx = -1
            best_dist = float("inf")
            for j, d_other in enumerate(other_remaining):
                dist = abs(d_self - d_other)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if best_idx < 0 or best_dist > tol_nm:
                return False
            other_remaining.pop(best_idx)
        return True


@dataclass
class GrainClickResult:
    """Result of a single click-to-segment operation."""
    # The grain mask (same shape as the cropped layer image, bool).
    grain_mask: np.ndarray
    # The signature of this grain.
    signature: GrainSignature
    # The click point (row, col) in the cropped-image coords.
    click_point: tuple[int, int]
    # The local FFT window (row0, col0, row1, col1) in cropped-image coords.
    window_bbox: tuple[int, int, int, int]
    # Local power spectrum (for display).
    local_power_spectrum: np.ndarray | None = None
    # Auto-found similar grains (if auto_find_similar was True). Each is a
    # grain mask (bool, same shape as grain_mask).
    similar_grain_masks: list = field(default_factory=list)
    # Status message for the UI.
    message: str = ""
    success: bool = True
    # --- Diagnostics (for troubleshooting when success=False) ---
    # Which step failed: "local_fft" | "find_d_spacings" | "amplitude_maps"
    # | "region_growing" | "safety_check" | None (success).
    failed_step: str | None = None
    # d-spacings found in the local FFT (before quality filter).
    d_spacings_raw: list = field(default_factory=list)
    # d-spacings kept after quality filter.
    d_spacings_kept: list = field(default_factory=list)
    # Per-d-spacing amplitude at the click point (after normalization).
    seed_amplitudes: list = field(default_factory=list)
    # Combined amplitude (min or mean) at the click point.
    seed_combined: float = 0.0
    # Final grown area (px) — even if rejected by safety check.
    grown_area_px: int = 0
    # Layer area (px).
    layer_area_px: int = 0
    # Original grown area BEFORE safety check (px). If much larger than
    # grown_area_px, the safety check shrank the region.
    original_grown_area_px: int = 0
    # Whether the safety check was applied.
    safety_check_applied: bool = False
    # Thresholds tried by the safety check: list of (threshold, area) tuples.
    safety_thresholds_tried: list = field(default_factory=list)
    # Description of any fallback used (e.g. "single-d fallback").
    fallback_used: str = ""
    # Per-d-spacing amplitude map diagnostics: list of dicts with keys
    # d_nm, r_global, reflections (list of dicts with expected_pos, actual_pos,
    # shift_px, peak_intensity, background_intensity, snr), best_snr, used_disk.
    amplitude_diag: list = field(default_factory=list)
    # Structure tensor cleanup diagnostics (None = cleanup disabled/unused).
    cleanup_diag: dict | None = None


# === Core functions ===

def compute_local_fft(
    cropped_img: np.ndarray,
    click_r: int,
    click_c: int,
    window_size: int,
    zero_pad_size: int = 0,
    params: FFTSegmentationParams | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], float]:
    """Compute a local FFT in a window around (click_r, click_c).

    Parameters
    ----------
    cropped_img : (H, W) uint8 — the cropped layer image.
    click_r, click_c : int — click position in cropped-img coords.
    window_size : int — side of the square window (will be made even).
    zero_pad_size : int — pad the window to this size before FFT (0 = no pad).
    params : FFTSegmentationParams (uses use_clahe, use_hann_window, use_wiener_filter).

    Returns
    -------
    (fft_shifted, log_power, window_bbox, scale)
        fft_shifted: complex128 — Wiener-filtered shifted FFT.
        log_power: float64 — log power spectrum.
        window_bbox: (r0, c0, r1, c1) — the actual window in cropped-img coords.
        scale: float — always 1.0 (local FFT is at full resolution).
    """
    if params is None:
        params = FFTSegmentationParams()
    if window_size % 2 == 1:
        window_size += 1
    half = window_size // 2

    h, w = cropped_img.shape[:2]
    r0 = max(0, click_r - half)
    r1 = min(h, click_r + half)
    c0 = max(0, click_c - half)
    c1 = min(w, click_c + half)
    if r1 - r0 < 8 or c1 - c0 < 8:
        # Window too small (click near image edge).
        return (np.empty((0, 0), dtype=np.complex128),
                np.empty((0, 0), dtype=np.float64),
                (r0, c0, r1, c1), 1.0)

    window = cropped_img[r0:r1, c0:c1]
    if window.ndim == 3:
        window = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
    if window.dtype != np.uint8:
        window = np.clip(window, 0, 255).astype(np.uint8)

    # Preprocess (CLAHE + Hann) FIRST — on the real window, BEFORE any
    # zero-padding. If padding were applied first, the sharp edges of the
    # content square sitting inside the black canvas act as two perpendicular
    # straight edges and produce a bright cross ("crosshair") through the FFT
    # center; the Hann window would NOT suppress it, because it only tapers
    # the outer boundary of the padded array, not the content square inside.
    img = _preprocess(window, params)

    # Optional zero-padding (improves frequency resolution).
    if zero_pad_size > max(img.shape):
        padded = np.zeros((zero_pad_size, zero_pad_size), dtype=img.dtype)
        # Center the window in the padded array.
        pr0 = (zero_pad_size - img.shape[0]) // 2
        pc0 = (zero_pad_size - img.shape[1]) // 2
        padded[pr0:pr0 + img.shape[0], pc0:pc0 + img.shape[1]] = img
        img = padded

    fft_shifted, log_power = _compute_power_spectrum(img)
    if params.use_wiener_filter:
        fft_shifted, log_power = _apply_wiener_filter(fft_shifted, params)

    return fft_shifted, log_power, (r0, c0, r1, c1), 1.0


def find_grain_d_spacings(
    local_log_power: np.ndarray,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    click_r: int,
    click_c: int,
    fft_params: FFTSegmentationParams,
    click_params: ClickSegmentationParams,
) -> tuple[list[float], list[float], list, list, np.ndarray, np.ndarray]:
    """Find the d-spacings present in the local window around the click.

    Uses the 1D radial profile detector (same as the global one) restricted
    to the local spectrum, then keeps the top-N d-spacings by intensity.

    For each d-spacing, ALL 2D reflections (Friedel pairs + different
    orientations) are recorded — these are used later to build
    orientation-specific disk masks in the global FFT.

    Returns
    -------
    (d_spacings_nm, radii_px, qualities, reflections_per_d, radii_1d, intensities_1d)
        d_spacings_nm: list of d-spacings (nm), one per detected peak.
        radii_px: list of FFT radii (px), one per detected peak.
        qualities: list of ReflectionQuality (one per d-spacing, evaluated at
            the strongest 2D maximum of each ring).
        reflections_per_d: list of lists of (row, col) — for each d-spacing,
            all 2D reflection coordinates in the LOCAL FFT array.
        radii_1d, intensities_1d: the 1D radial spectrum (for UI plotting).
    """
    h, w = local_log_power.shape
    radii_1d, intens_1d = compute_radial_spectrum(local_log_power)
    if len(radii_1d) == 0:
        return [], [], [], [], radii_1d, intens_1d

    # Smooth + adaptive thresholds (local version).
    from scipy.ndimage import gaussian_filter1d
    s_smooth = gaussian_filter1d(intens_1d, sigma=click_params.local_smoothing_sigma)
    max_radius = max(h, w) // 2
    min_dist = max(5, int(fft_params.min_freq_frac * max_radius))
    max_dist = int(fft_params.max_freq_frac * max_radius)
    valid = (radii_1d >= min_dist) & (radii_1d <= max_dist)
    if not valid.any():
        return [], [], [], [], radii_1d, intens_1d
    r_in = radii_1d[valid]
    s_in = s_smooth[valid]
    baseline = float(np.quantile(s_in, 0.50))
    high_q = float(np.quantile(s_in, 0.90))
    height_threshold = baseline + click_params.local_peak_height_factor * (high_q - baseline)
    prominence_threshold = 0.40 * (high_q - baseline)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(
        s_in,
        height=max(height_threshold, baseline + 1e-9),
        prominence=max(prominence_threshold, 1e-9),
        distance=max(1, fft_params.radial_peak_min_distance_px),
    )
    peak_radii = [int(r_in[i]) for i in peaks]

    if not peak_radii:
        return [], [], [], [], radii_1d, intens_1d

    # For each peak: find ALL 2D maxima (these are individual reflections
    # at this d-spacing, each with a specific angle). Evaluate quality at
    # the strongest one.
    center_r, center_c = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    ring_half = max(2.0, click_params.ring_mask_width_px / 2.0)

    d_spacings = []
    radii_px = []
    qualities = []
    reflections_per_d: list[list[tuple[int, int]]] = []
    for r_pk in peak_radii:
        ring_mask = (dist >= r_pk - ring_half) & (dist <= r_pk + ring_half)
        ring_power = np.where(ring_mask, local_log_power, 0.0)
        if ring_power.max() <= 0:
            continue
        coords = __import__("skimage").feature.peak_local_max(
            ring_power, min_distance=2,
            threshold_abs=0.30 * float(ring_power.max()),
            exclude_border=False,
        )
        if len(coords) == 0:
            continue
        # Evaluate quality at the strongest maximum.
        intensities = local_log_power[coords[:, 0], coords[:, 1]]
        best_idx = int(np.argmax(intensities))
        best_coord = coords[best_idx]
        q = evaluate_reflection_quality(
            local_log_power, float(best_coord[0]), float(best_coord[1]),
            (h, w), pixel_size_nm, scale, fft_params,
        )
        if q.status == "noise":
            continue
        d_nm = compute_d_spacing_for_radius(float(r_pk), (h, w), pixel_size_nm, scale)
        d_spacings.append(d_nm)
        radii_px.append(float(r_pk))
        qualities.append(q)
        # Keep ALL 2D reflections (each is a (row, col) in the LOCAL FFT array).
        # These will be mapped to the global FFT later.
        refl_list = [(int(c[0]), int(c[1])) for c in coords]
        reflections_per_d.append(refl_list)

    # Limit to max_d_spacings_per_grain (sorted by SNR).
    if len(d_spacings) > click_params.max_d_spacings_per_grain:
        order = np.argsort([q.snr for q in qualities])[::-1]
        keep = order[:click_params.max_d_spacings_per_grain]
        d_spacings = [d_spacings[i] for i in keep]
        radii_px = [radii_px[i] for i in keep]
        qualities = [qualities[i] for i in keep]
        reflections_per_d = [reflections_per_d[i] for i in keep]

    return d_spacings, radii_px, qualities, reflections_per_d, radii_1d, intens_1d


# === Sliding local FFT (mode B) ===

def compute_sliding_local_ffts(
    cropped_img: np.ndarray,
    cropped_mask: np.ndarray,
    window_size: int,
    step: int,
    zero_pad_size: int,
    fft_params: FFTSegmentationParams,
) -> dict:
    """Compute local FFTs in a sliding window over the image (cached).

    This is the preprocessing step for mode B (sliding local FFT). It is
    independent of the click position — only depends on the image and the
    window/step parameters — so it can be computed once and reused for all
    clicks.

    Returns
    -------
    dict with keys:
        - "log_powers": (N_windows, H_pad, W_pad) float32 — log power spectrum
          for each window. (H_pad, W_pad) = (zero_pad_size, zero_pad_size) if
          zero_pad_size > 0, else (window_size, window_size).
        - "window_centers": (N_windows, 2) float — (row, col) of each window's
          center, in cropped-img coords.
        - "window_bboxes": (N_windows, 4) int — (r0, c0, r1, c1) of each window
          in cropped-img coords.
        - "h_img, w_img": image shape.
        - "window_size, step, zero_pad_size": parameters used.
    """
    h_img, w_img = cropped_img.shape[:2]
    if window_size % 2 == 1:
        window_size += 1
    half = window_size // 2

    # Pad size for the FFT.
    pad_size = max(zero_pad_size, window_size) if zero_pad_size > 0 else window_size

    # Generate window centers on a regular grid.
    centers_r = list(range(half, h_img - half + 1, step))
    centers_c = list(range(half, w_img - half + 1, step))
    if not centers_r or not centers_c:
        return {
            "log_powers": np.empty((0, pad_size, pad_size), dtype=np.float32),
            "window_centers": np.empty((0, 2), dtype=np.float32),
            "window_bboxes": np.empty((0, 4), dtype=np.int32),
            "h_img": h_img, "w_img": w_img,
            "window_size": window_size, "step": step, "zero_pad_size": zero_pad_size,
        }

    log_powers = []
    centers = []
    bboxes = []
    for cr in centers_r:
        for cc in centers_c:
            r0, r1 = cr - half, cr + half
            c0, c1 = cc - half, cc + half
            window = cropped_img[r0:r1, c0:c1]
            if window.ndim == 3:
                window = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
            if window.dtype != np.uint8:
                window = np.clip(window, 0, 255).astype(np.uint8)
            # Skip windows that are mostly outside the layer mask.
            win_mask = cropped_mask[r0:r1, c0:c1]
            if win_mask.sum() < 0.25 * win_mask.size:
                continue
            # Preprocess (CLAHE + Hann) BEFORE zero-padding — same reason as
            # in compute_local_fft: padding first would make the content
            # square edges act as straight edges and paint a crosshair
            # through the FFT center.
            img = _preprocess(window, fft_params)
            # Zero-pad.
            if zero_pad_size > max(img.shape):
                padded = np.zeros((zero_pad_size, zero_pad_size), dtype=img.dtype)
                pr0 = (zero_pad_size - img.shape[0]) // 2
                pc0 = (zero_pad_size - img.shape[1]) // 2
                padded[pr0:pr0 + img.shape[0], pc0:pc0 + img.shape[1]] = img
                img = padded
            # FFT + Wiener.
            fft_shifted, log_power = _compute_power_spectrum(img)
            if fft_params.use_wiener_filter:
                fft_shifted, log_power = _apply_wiener_filter(fft_shifted, fft_params)
            log_powers.append(log_power.astype(np.float32))
            centers.append((float(cr), float(cc)))
            bboxes.append((r0, c0, r1, c1))

    return {
        "log_powers": (np.array(log_powers, dtype=np.float32) if log_powers
                       else np.empty((0, pad_size, pad_size), dtype=np.float32)),
        "window_centers": np.array(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32),
        "window_bboxes": np.array(bboxes, dtype=np.int32) if bboxes else np.empty((0, 4), dtype=np.int32),
        "h_img": h_img, "w_img": w_img,
        "window_size": window_size, "step": step, "zero_pad_size": zero_pad_size,
    }


def find_d_spacing_and_orientation_in_local_fft(
    local_log_power: np.ndarray,
    fft_shape: tuple[int, int],
    pixel_size_nm: float,
    scale: float,
    fft_params: FFTSegmentationParams,
    click_params: ClickSegmentationParams,
) -> tuple[list[float], list[float], list[float], list, np.ndarray, np.ndarray]:
    """Find d-spacings AND their orientations in a local FFT.

    This is the per-click analysis: given the local FFT at the click point,
    find the strongest d-spacings and, for each, the angle θ of its strongest
    2D reflection. The (d, θ) pair is the "signature" of the clicked grain.

    Returns
    -------
    (d_spacings_nm, radii_px, orientations_deg, qualities, radii_1d, intensities_1d)
        d_spacings_nm: list of d-spacings (nm).
        radii_px: list of FFT radii (px).
        orientations_deg: list of orientations (degrees, 0–180°, Friedel-invariant).
        qualities: list of ReflectionQuality (one per d-spacing).
        radii_1d, intensities_1d: the 1D radial spectrum.
    """
    h, w = local_log_power.shape
    radii_1d, intens_1d = compute_radial_spectrum(local_log_power)
    if len(radii_1d) == 0:
        return [], [], [], [], radii_1d, intens_1d

    from scipy.ndimage import gaussian_filter1d
    s_smooth = gaussian_filter1d(intens_1d, sigma=click_params.local_smoothing_sigma)
    max_radius = max(h, w) // 2
    min_dist = max(5, int(fft_params.min_freq_frac * max_radius))
    max_dist = int(fft_params.max_freq_frac * max_radius)
    valid = (radii_1d >= min_dist) & (radii_1d <= max_dist)
    if not valid.any():
        return [], [], [], [], radii_1d, intens_1d
    r_in = radii_1d[valid]
    s_in = s_smooth[valid]
    baseline = float(np.quantile(s_in, 0.50))
    high_q = float(np.quantile(s_in, 0.90))
    height_threshold = baseline + click_params.local_peak_height_factor * (high_q - baseline)
    prominence_threshold = 0.40 * (high_q - baseline)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(
        s_in,
        height=max(height_threshold, baseline + 1e-9),
        prominence=max(prominence_threshold, 1e-9),
        distance=max(1, fft_params.radial_peak_min_distance_px),
    )
    peak_radii = [int(r_in[i]) for i in peaks]
    if not peak_radii:
        return [], [], [], [], radii_1d, intens_1d

    center_r, center_c = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    ring_half = max(2.0, click_params.ring_mask_width_px / 2.0)

    d_spacings = []
    radii_px = []
    orientations = []
    qualities = []
    for r_pk in peak_radii:
        ring_mask = (dist >= r_pk - ring_half) & (dist <= r_pk + ring_half)
        ring_power = np.where(ring_mask, local_log_power, 0.0)
        if ring_power.max() <= 0:
            continue
        coords = __import__("skimage").feature.peak_local_max(
            ring_power, min_distance=2,
            threshold_abs=0.30 * float(ring_power.max()),
            exclude_border=False,
        )
        if len(coords) == 0:
            continue
        intensities = local_log_power[coords[:, 0], coords[:, 1]]
        best_idx = int(np.argmax(intensities))
        best_coord = coords[best_idx]
        q = evaluate_reflection_quality(
            local_log_power, float(best_coord[0]), float(best_coord[1]),
            (h, w), pixel_size_nm, scale, fft_params,
        )
        if q.status == "noise":
            continue
        d_nm = compute_d_spacing_for_radius(float(r_pk), (h, w), pixel_size_nm, scale)
        # Orientation: angle of the strongest reflection, in degrees, 0–180°
        # (Friedel-invariant: a reflection at angle θ and its Friedel pair at
        # θ+180° belong to the same grain, so we map to [0, 180°)).
        dy = best_coord[0] - center_r
        dx = best_coord[1] - center_c
        theta_rad = np.arctan2(dy, dx)
        theta_deg = np.degrees(theta_rad) % 180.0
        d_spacings.append(d_nm)
        radii_px.append(float(r_pk))
        orientations.append(float(theta_deg))
        qualities.append(q)

    # Limit to max_d_spacings_per_grain (sorted by SNR).
    if len(d_spacings) > click_params.max_d_spacings_per_grain:
        order = np.argsort([q.snr for q in qualities])[::-1]
        keep = order[:click_params.max_d_spacings_per_grain]
        d_spacings = [d_spacings[i] for i in keep]
        radii_px = [radii_px[i] for i in keep]
        orientations = [orientations[i] for i in keep]
        qualities = [qualities[i] for i in keep]

    return d_spacings, radii_px, orientations, qualities, radii_1d, intens_1d


def build_amplitude_map_from_sliding(
    sliding_data: dict,
    d_nm: float,
    orientation_deg: float,
    pixel_size_nm: float,
    fft_params: FFTSegmentationParams,
    click_params: ClickSegmentationParams,
) -> tuple[np.ndarray, dict]:
    """Build an amplitude map for a (d, θ) signature using sliding local FFTs.

    For each sliding window, checks whether the local FFT has a peak at the
    d-spacing d_nm AND within the orientation tolerance Δ of θ. The amplitude
    is the peak intensity (normalized by the window's max). The amplitudes are
    placed at the window centers and interpolated to full resolution.

    Returns
    -------
    (amplitude_map, diag)
        amplitude_map: (H, W) float, ~[0, 1].
        diag: dict with keys:
            - "n_windows": total number of sliding windows.
            - "n_with_peak": number of windows where a peak was found in the
              angular sector at radius r(d).
            - "n_in_layer": number of windows inside the layer.
            - "max_amplitude": max amplitude value (before normalization).
    """
    h_img = sliding_data["h_img"]
    w_img = sliding_data["w_img"]
    log_powers = sliding_data["log_powers"]
    window_centers = sliding_data["window_centers"]
    window_bboxes = sliding_data["window_bboxes"]
    n_windows = log_powers.shape[0]
    if n_windows == 0:
        return np.zeros((h_img, w_img), dtype=np.float32), {
            "n_windows": 0, "n_with_peak": 0, "n_in_layer": 0, "max_amplitude": 0.0,
        }

    # FFT shape (per window).
    h_fft, w_fft = log_powers.shape[1], log_powers.shape[2]
    center_r, center_c = h_fft / 2.0, w_fft / 2.0
    avg_dim = float(np.sqrt(h_fft * w_fft))
    pixel_size_small = pixel_size_nm  # local FFT is at full resolution (scale=1)
    r_target = avg_dim * pixel_size_small / d_nm  # FFT radius for this d-spacing

    # Orientation tolerance in radians.
    tol_rad = np.radians(click_params.orientation_tolerance_deg)

    # Pre-compute distance and angle arrays for the FFT.
    yy, xx = np.ogrid[:h_fft, :w_fft]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    # Angle of each FFT pixel (in radians, [0, 2π)). Note: atan2(dy, dx).
    angle = np.arctan2(yy - center_r, xx - center_c)

    # Build the radial+angular mask ONCE (same for all windows since they all
    # use the same FFT shape).
    ring_half = max(2.0, click_params.ring_mask_width_px / 2.0)
    radial_mask = (dist >= r_target - ring_half) & (dist <= r_target + ring_half)
    # Angular mask: accept angles within ±Δ of θ OR within ±Δ of θ+π (Friedel).
    theta_rad = np.radians(orientation_deg)
    # Normalize angle difference to [0, π] (Friedel-invariant).
    diff = np.abs(((angle - theta_rad + np.pi) % (2 * np.pi)) - np.pi)
    angular_mask = diff <= tol_rad

    combined_mask = radial_mask & angular_mask

    # For each window, find the max intensity in the masked region.
    # Vectorize: log_powers is (N, H, W); mask is (H, W).
    masked = np.where(combined_mask[None, :, :], log_powers, 0.0)
    raw_peak = masked.max(axis=(1, 2))  # (N,)
    has_peak = raw_peak > 0

    # Compute the GLOBAL peak intensity (max over all windows) — used as an
    # absolute reference. A window's peak is "significant" only if it's a
    # meaningful fraction of the global peak (otherwise it's just noise that
    # happens to fall in the mask).
    global_peak = float(raw_peak.max()) if has_peak.any() else 1.0
    global_peak_safe = max(global_peak, 1e-9)

    # Compute the BACKGROUND level for each window = mean of log power in an
    # annulus at the same radius as the target ring but OUTSIDE the angular
    # sector. This represents what the noise floor looks like at this radius.
    background_annulus_mask = radial_mask & ~angular_mask
    bg_at_radius_per_window = np.array([
        float(lp[background_annulus_mask].mean()) if background_annulus_mask.sum() > 0 else 0.0
        for lp in log_powers
    ])  # (N,)

    # Two criteria for a window to have a "real" peak:
    # 1. ABSOLUTE: raw_peak >= 0.50 * global_peak (the peak is at least 50% of
    #    the strongest peak across all windows — drops noise-only windows).
    #    50% is strict but necessary: in a typical image, noise windows have
    #    peaks at ~30-50% of the global peak, while real-grain windows have
    #    peaks at ~80-100%.
    # 2. RELATIVE (contrast): raw_peak >= 2.0 * bg_at_radius (the peak is at
    #    least 2× brighter than the surrounding noise at the same radius).
    abs_threshold = 0.50 * global_peak_safe
    bg_safe = np.where(bg_at_radius_per_window > 1e-9, bg_at_radius_per_window, 1.0)
    contrast = raw_peak / bg_safe
    significant = (raw_peak >= abs_threshold) & (contrast >= 2.0) & has_peak

    # Amplitude = raw_peak / global_peak (in [0, 1]). Only significant windows
    # get non-zero amplitude.
    peak_intensities = np.where(significant, raw_peak / global_peak_safe, 0.0)
    peak_intensities = np.clip(peak_intensities, 0.0, 1.0)
    valid = significant
    # Build a low-res amplitude map at the window grid, then resize with
    # bilinear interpolation (NOT cubic + Gaussian — that over-smooths and
    # causes amplitude to leak into neighboring regions).
    if valid.any():
        step = sliding_data["step"]
        # Build a low-res grid: rows = centers_r, cols = centers_c.
        cr_unique = np.unique(window_centers[:, 0].astype(int))
        cc_unique = np.unique(window_centers[:, 1].astype(int))
        low_res = np.zeros((len(cr_unique), len(cc_unique)), dtype=np.float32)
        cr_to_idx = {int(v): i for i, v in enumerate(cr_unique)}
        cc_to_idx = {int(v): i for i, v in enumerate(cc_unique)}
        for i in range(n_windows):
            if not valid[i]:
                continue
            cr_i = int(window_centers[i, 0])
            cc_i = int(window_centers[i, 1])
            if cr_i in cr_to_idx and cc_i in cc_to_idx:
                low_res[cr_to_idx[cr_i], cc_to_idx[cc_i]] = peak_intensities[i]
        # Resize to full resolution with bilinear interpolation (preserves
        # sharp boundaries between grains).
        amplitude_map = cv2.resize(low_res, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        # Light smoothing with a small kernel (3×3) to remove blocky
        # interpolation artifacts — NOT a large Gaussian (that would leak).
        amplitude_map = cv2.GaussianBlur(amplitude_map, (3, 3), 0)
    else:
        amplitude_map = np.zeros((h_img, w_img), dtype=np.float32)

    # Normalize to [0, 1].
    amp_max = float(amplitude_map.max())
    if amp_max > 0:
        amplitude_map = amplitude_map / amp_max

    diag = {
        "n_windows": int(n_windows),
        "n_with_peak": int(valid.sum()),
        "n_in_layer": int(n_windows),  # all windows passed the layer-mask filter
        "max_amplitude": float(amp_max),
    }
    return amplitude_map.astype(np.float32), diag


def build_amplitude_map_from_manual_reflection(
    sliding_data: dict,
    reflection_r: float,
    reflection_c: float,
    pixel_size_nm: float,
    fft_params: FFTSegmentationParams,
    click_params: ClickSegmentationParams,
) -> tuple[np.ndarray, dict]:
    """Build an amplitude map from a MANUALLY-clicked reflection in the local
    power spectrum.

    The user clicks on a reflection in the local FFT (which has shape
    (H_local, W_local)). From the click position (reflection_r, reflection_c),
    we compute:
      - radius r = distance from the FFT center → d-spacing.
      - angle θ = atan2(dr, dc) → orientation.

    Then we build the amplitude map using the same sliding-FFT approach as
    build_amplitude_map_from_sliding, but with the EXACT (r, θ) from the user's
    click (no auto-detection, no ambiguity).

    Parameters
    ----------
    sliding_data : dict (from compute_sliding_local_ffts).
    reflection_r, reflection_c : float — click position in the LOCAL FFT array
        (the same array shown to the user as the local power spectrum).
    pixel_size_nm : pixel size of the original image (nm/px).
    fft_params, click_params : parameters.

    Returns
    -------
    (amplitude_map, diag)
        amplitude_map: (H, W) float, ~[0, 1].
        diag: dict with d_nm, orientation_deg, r_target_sliding, n_with_peak,
            n_windows, r_warning.
    """
    h_img = sliding_data["h_img"]
    w_img = sliding_data["w_img"]
    log_powers = sliding_data["log_powers"]
    n_windows = log_powers.shape[0]
    if n_windows == 0:
        return np.zeros((h_img, w_img), dtype=np.float32), {
            "d_nm": 0.0, "orientation_deg": 0.0, "r_target_sliding": 0.0,
            "n_windows": 0, "n_with_peak": 0, "r_warning": "no sliding windows",
        }

    # The LOCAL FFT shape (the one the user clicked on). This is the click
    # window's FFT, which may be different from the sliding window's FFT.
    # We need to convert the click position to (d, θ), then re-compute r_target
    # in the SLIDING FFT.
    # The local FFT shape is stored in sliding_data? No — it's separate.
    # We compute (d, θ) from the local FFT, then use the sliding FFT's shape
    # for r_target.
    # IMPORTANT: the caller must pass the local FFT shape via fft_params or
    # we infer it from the click_params (window_size + zero_pad).
    local_fft_size = max(click_params.window_size_px, click_params.zero_pad_size_px)
    if click_params.zero_pad_size_px > click_params.window_size_px:
        local_fft_size = click_params.zero_pad_size_px
    else:
        local_fft_size = click_params.window_size_px
    local_center = local_fft_size / 2.0
    local_avg_dim = float(local_fft_size)

    # Compute (d, θ) from the click position in the local FFT.
    dr = reflection_r - local_center
    dc = reflection_c - local_center
    r_local = float(np.sqrt(dr ** 2 + dc ** 2))
    theta_rad = np.arctan2(dr, dc)
    theta_deg = float(np.degrees(theta_rad) % 180.0)
    # d = local_avg_dim * pixel_size_nm / r_local (scale=1 for local FFT).
    if r_local > 0:
        d_nm = local_avg_dim * pixel_size_nm / r_local
    else:
        d_nm = 0.0

    # Now build the amplitude map using the sliding FFT with this (d, θ).
    # Reuse build_amplitude_map_from_sliding.
    amp, slide_diag = build_amplitude_map_from_sliding(
        sliding_data, d_nm, theta_deg, pixel_size_nm, fft_params, click_params,
    )

    # Compute r_target in the sliding FFT for diagnostics.
    slide_h = log_powers.shape[1]
    slide_avg_dim = float(np.sqrt(slide_h * slide_h))
    r_target_sliding = slide_avg_dim * pixel_size_nm / d_nm if d_nm > 0 else 0.0
    slide_half = slide_h // 2
    r_warning = ""
    if r_target_sliding < 5.0:
        r_warning = (f"r_target={r_target_sliding:.1f}px is in the DC region (<5px). "
                     "INCREASE the sliding window size.")
    elif r_target_sliding > slide_half - 5:
        r_warning = (f"r_target={r_target_sliding:.1f}px is at the FFT edge. "
                     "DECREASE the sliding window size.")

    diag = {
        "d_nm": float(d_nm),
        "orientation_deg": theta_deg,
        "r_local": r_local,
        "r_target_sliding": float(r_target_sliding),
        "n_windows": int(slide_diag.get("n_windows", 0)),
        "n_with_peak": int(slide_diag.get("n_with_peak", 0)),
        "r_warning": r_warning,
        "mode": "manual_reflection",
    }
    return amp, diag


def _apply_structure_cleanup(
    cropped_img: np.ndarray,
    grain_mask: np.ndarray,
    click_params: ClickSegmentationParams,
    orientation_targets_deg: list | None = None,
) -> tuple[np.ndarray, dict]:
    """Optional post-filter: remove amorphous / wrongly-oriented pixels from
    a grown grain mask using the structure tensor.

    The FFT angles (0–180°, mod 180) of the grain's reflections are used
    directly as orientation targets: the structure tensor's dominant GRADIENT
    direction of a fringe pattern equals the scattering-vector direction of
    the corresponding reflection (both are normal to the fringes).
    """
    if not click_params.use_structure_cleanup or not grain_mask.any():
        return grain_mask, None
    try:
        from .structure_tensor_segmentation import (
            cleanup_grain_mask_with_structure_tensor,
        )
    except Exception:
        # skimage missing etc. — cleanup is optional, never break segmentation.
        return grain_mask, None
    cleaned, diag = cleanup_grain_mask_with_structure_tensor(
        cropped_img, grain_mask,
        sigma=click_params.cleanup_sigma_px,
        coherence_threshold=click_params.cleanup_coherence_threshold,
        orientation_targets_deg=list(orientation_targets_deg or []),
        orientation_tolerance_deg=click_params.cleanup_orientation_tolerance_deg,
        min_remove_area_px=click_params.cleanup_min_remove_area_px,
    )
    return cleaned, diag


def _cleanup_similar_masks(
    cropped_img: np.ndarray,
    similar_masks: list,
    click_params: ClickSegmentationParams,
    orientation_targets_deg: list | None = None,
) -> list:
    """Apply the structure tensor cleanup to auto-found similar-grain masks."""
    if not click_params.use_structure_cleanup or not similar_masks:
        return similar_masks
    cleaned = []
    for sm in similar_masks:
        sm_c, _ = _apply_structure_cleanup(
            cropped_img, sm, click_params, orientation_targets_deg,
        )
        if sm_c.any():
            cleaned.append(sm_c)
    return cleaned


def segment_grain_from_click_with_reflections(
    cropped_img: np.ndarray,
    cropped_mask: np.ndarray,
    click_r: int,
    click_c: int,
    pixel_size_nm: float,
    fft_params: FFTSegmentationParams,
    click_params: ClickSegmentationParams,
    sliding_data: dict,
    manual_reflections: list[tuple[float, float]],
) -> GrainClickResult:
    """Segment a grain using MANUALLY-selected reflections on the local power
    spectrum.

    This is the manual version of segment_grain_from_click: instead of
    auto-detecting d-spacings and orientations, the user has clicked on
    specific reflections in the local power spectrum. Each reflection defines
    a (d, θ) pair. We build amplitude maps for each via sliding FFT, then
    region-grow with the AND condition.

    Parameters
    ----------
    cropped_img, cropped_mask, click_r, click_c, pixel_size_nm : as usual.
    fft_params, click_params : parameters.
    sliding_data : precomputed sliding local FFTs.
    manual_reflections : list of (r, c) tuples — positions in the LOCAL FFT
        array (the power spectrum shown to the user). Each is a reflection the
        user wants to use for this grain.

    Returns
    -------
    GrainClickResult
    """
    h_full, w_full = cropped_img.shape[:2]
    layer_area = int(cropped_mask.sum())

    # 1. Compute the local FFT (for the power spectrum display + to get the
    # local FFT shape).
    local_fft, local_log_power, window_bbox, local_scale = compute_local_fft(
        cropped_img, click_r, click_c,
        click_params.window_size_px, click_params.zero_pad_size_px, fft_params,
    )
    if local_fft.size == 0:
        return GrainClickResult(
            grain_mask=np.zeros((h_full, w_full), dtype=bool),
            signature=GrainSignature(),
            click_point=(click_r, click_c),
            window_bbox=window_bbox,
            message="Window too small (click near image edge).",
            success=False,
            failed_step="local_fft",
            layer_area_px=layer_area,
        )

    if not manual_reflections:
        return GrainClickResult(
            grain_mask=np.zeros((h_full, w_full), dtype=bool),
            signature=GrainSignature(),
            click_point=(click_r, click_c),
            window_bbox=window_bbox,
            local_power_spectrum=local_log_power,
            message="No reflections selected. Click on reflections in the "
                    "local power spectrum first.",
            success=False,
            failed_step="no_reflections",
            layer_area_px=layer_area,
        )

    # 2. Build amplitude maps for each manually-selected reflection.
    amplitude_maps = []
    radii_global = []
    seed_amplitudes = []
    orientations_deg = []
    d_spacings_nm = []
    qualities = []  # placeholder (no quality assessment for manual clicks)
    amplitude_diag = []

    for (refl_r, refl_c) in manual_reflections:
        amp, m_diag = build_amplitude_map_from_manual_reflection(
            sliding_data, float(refl_r), float(refl_c),
            pixel_size_nm, fft_params, click_params,
        )
        amplitude_maps.append(amp)
        radii_global.append(0.0)
        orientations_deg.append(m_diag["orientation_deg"])
        d_spacings_nm.append(m_diag["d_nm"])
        seed_amplitudes.append(float(amp[click_r, click_c]))
        # Create a placeholder quality (no Gaussian fit for manual clicks).
        from .fft_segmentation import ReflectionQuality
        q = ReflectionQuality(
            status="good",  # manual = user-validated
            d_spacing_nm=m_diag["d_nm"],
            message="Manually selected by user.",
        )
        qualities.append(q)
        amplitude_diag.append({
            "d_nm": m_diag["d_nm"],
            "orientation_deg": m_diag["orientation_deg"],
            "r_local": m_diag.get("r_local", 0.0),
            "r_target_sliding": m_diag.get("r_target_sliding", 0.0),
            "mode": "manual_reflection",
            "n_windows": m_diag.get("n_windows", 0),
            "n_with_peak": m_diag.get("n_with_peak", 0),
            "r_warning": m_diag.get("r_warning", ""),
        })

    # 3. Region growing with hysteresis + AND.
    grain_mask, grow_diag = grow_grain_region(
        amplitude_maps, click_r, click_c, cropped_mask, click_params,
    )
    if click_params.strict_and and seed_amplitudes:
        seed_combined = float(min(seed_amplitudes))
    elif seed_amplitudes:
        seed_combined = float(np.mean(seed_amplitudes))
    else:
        seed_combined = 0.0
    grown_area = int(grain_mask.sum())

    # Fallback: single-d if multi-d failed.
    if grain_mask.sum() == 0 and len(amplitude_maps) > 1:
        best_idx = int(np.argmax(seed_amplitudes))
        single_amp = [amplitude_maps[best_idx]]
        grain_mask_fb, fb_diag = grow_grain_region(
            single_amp, click_r, click_c, cropped_mask, click_params,
        )
        if grain_mask_fb.sum() > 0:
            grain_mask = grain_mask_fb
            amplitude_maps = single_amp
            d_spacings_nm = [d_spacings_nm[best_idx]]
            orientations_deg = [orientations_deg[best_idx]]
            qualities = [qualities[best_idx]]
            radii_global = [radii_global[best_idx]]
            seed_amplitudes = [seed_amplitudes[best_idx]]
            grown_area = int(grain_mask.sum())
            grow_diag = fb_diag
            grow_diag["fallback_used"] = ("single-d fallback: used only the "
                                          "strongest reflection.")

    # 3.5 Structure tensor cleanup (optional post-filter): remove amorphous /
    # wrongly-oriented regions from the grown mask using the selected
    # reflections' FFT angles as orientation targets.
    cleanup_diag = None
    if click_params.use_structure_cleanup and grain_mask.sum() > 0:
        grain_mask, cleanup_diag = _apply_structure_cleanup(
            cropped_img, grain_mask, click_params, orientations_deg,
        )
        grown_area = int(grain_mask.sum())

    if grain_mask.sum() == 0:
        amp_strs = [f"d={d:.3f}nm θ={theta:.1f}° amp={a:.3f}"
                    for d, theta, a in zip(d_spacings_nm, orientations_deg, seed_amplitudes)]
        return GrainClickResult(
            grain_mask=grain_mask,
            signature=GrainSignature(d_spacings_nm=d_spacings_nm,
                                      radii_px=radii_global, qualities=qualities,
                                      orientations_deg=orientations_deg),
            click_point=(click_r, click_c),
            window_bbox=window_bbox,
            local_power_spectrum=local_log_power,
            message=(f"Region growing failed. {len(manual_reflections)} reflection(s): "
                     f"{'; '.join(amp_strs)}. Combined seed amp = {seed_combined:.3f} "
                     f"(need ≥ {click_params.seed_threshold_frac:.2f}). "
                     f"Grown area = {grown_area} px."),
            success=False,
            failed_step="region_growing",
            d_spacings_kept=list(d_spacings_nm),
            seed_amplitudes=seed_amplitudes,
            seed_combined=seed_combined,
            grown_area_px=grown_area,
            layer_area_px=layer_area,
            original_grown_area_px=int(grow_diag.get("original_grown_area", 0)),
            safety_check_applied=bool(grow_diag.get("safety_check_applied", False)),
            safety_thresholds_tried=list(grow_diag.get("safety_thresholds_tried", [])),
            fallback_used=str(grow_diag.get("fallback_used", "")),
            amplitude_diag=amplitude_diag,
        )

    # 4. Auto-find similar grains.
    similar = []
    if click_params.auto_find_similar:
        similar = find_similar_grains(
            amplitude_maps, grain_mask, cropped_mask, click_params,
        )
        similar = _cleanup_similar_masks(
            cropped_img, similar, click_params, orientations_deg,
        )

    signature = GrainSignature(
        d_spacings_nm=d_spacings_nm, radii_px=radii_global, qualities=qualities,
        orientations_deg=orientations_deg,
    )
    n_d = len(d_spacings_nm)
    n_similar = len(similar)
    d_str = ", ".join(f"{d:.3f}@{theta:.0f}°" for d, theta in zip(d_spacings_nm, orientations_deg))
    msg = (f"Grain segmented (manual reflections): {n_d} reflection(s) ({d_str}), "
           f"area = {int(grain_mask.sum())} px.")
    if n_similar > 0:
        msg += f" Found {n_similar} similar grain(s)."
    if grow_diag.get("fallback_used"):
        msg += f" Note: {grow_diag['fallback_used']}"
    if cleanup_diag and cleanup_diag.get("removed_px"):
        msg += (f" Structure-tensor cleanup removed {cleanup_diag['removed_px']} px "
                f"({cleanup_diag['removed_frac'] * 100:.0f}%) as amorphous/"
                f"wrong-orientation.")
    elif cleanup_diag and cleanup_diag.get("note"):
        msg += f" Cleanup note: {cleanup_diag['note']}"

    return GrainClickResult(
        grain_mask=grain_mask,
        signature=signature,
        click_point=(click_r, click_c),
        window_bbox=window_bbox,
        local_power_spectrum=local_log_power,
        similar_grain_masks=similar,
        message=msg,
        success=True,
        d_spacings_kept=list(d_spacings_nm),
        seed_amplitudes=seed_amplitudes,
        seed_combined=seed_combined,
        grown_area_px=int(grain_mask.sum()),
        layer_area_px=layer_area,
        original_grown_area_px=int(grow_diag.get("original_grown_area", 0)),
        safety_check_applied=bool(grow_diag.get("safety_check_applied", False)),
        safety_thresholds_tried=list(grow_diag.get("safety_thresholds_tried", [])),
        fallback_used=str(grow_diag.get("fallback_used", "")),
        amplitude_diag=amplitude_diag,
        cleanup_diag=cleanup_diag,
    )


def grow_grain_region(
    amplitude_maps: list[np.ndarray],
    seed_r: int,
    seed_c: int,
    layer_mask: np.ndarray,
    params: ClickSegmentationParams,
) -> tuple[np.ndarray, dict]:
    """Region growing with hysteresis + AND-condition.

    Returns
    -------
    (grain_mask, diagnostics)
        grain_mask: (H, W) bool — the grown grain mask.
        diagnostics: dict with keys:
            - combined_max: float — max of combined amplitude map
            - combined_at_seed: float — combined amplitude at the seed point
            - original_grown_area: int — area BEFORE safety check
            - safety_check_applied: bool
            - safety_thresholds_tried: list of (threshold, area) tuples
            - fallback_used: str — description of any fallback used
    """
    diag = {
        "combined_max": 0.0,
        "combined_at_seed": 0.0,
        "original_grown_area": 0,
        "safety_check_applied": False,
        "safety_thresholds_tried": [],
        "fallback_used": "",
    }
    if not amplitude_maps:
        return np.zeros_like(layer_mask, dtype=bool), diag
    h, w = amplitude_maps[0].shape
    if seed_r < 0 or seed_r >= h or seed_c < 0 or seed_c >= w:
        return np.zeros_like(layer_mask, dtype=bool), diag

    # Stack amplitude maps and normalize each to [0, 1].
    stack = np.stack(amplitude_maps, axis=0).astype(np.float64)
    for i in range(stack.shape[0]):
        m = float(stack[i].max())
        if m > 0:
            stack[i] = stack[i] / m

    # Combined metric: min (strict AND) or mean (loose AND).
    if params.strict_and:
        combined = stack.min(axis=0)
    else:
        combined = stack.mean(axis=0)
    diag["combined_max"] = float(combined.max())
    diag["combined_at_seed"] = float(combined[seed_r, seed_c])

    # Hysteresis: seed must pass high threshold; growth can pass low threshold.
    high_thr = params.seed_threshold_frac
    low_thr = params.growth_threshold_frac
    if combined[seed_r, seed_c] < high_thr:
        # Seed pixel doesn't pass high threshold — try a small neighborhood.
        r0 = max(0, seed_r - 2)
        r1 = min(h, seed_r + 3)
        c0 = max(0, seed_c - 2)
        c1 = min(w, seed_c + 3)
        local_max = float(combined[r0:r1, c0:c1].max())
        if local_max < high_thr:
            diag["fallback_used"] = (f"seed amp {combined[seed_r, seed_c]:.3f} < "
                                      f"seed_threshold {high_thr:.3f}; local max "
                                      f"{local_max:.3f} also below threshold")
            return np.zeros_like(layer_mask, dtype=bool), diag
        # Move seed to the local max.
        flat_idx = int(np.argmax(combined[r0:r1, c0:c1]))
        seed_r = r0 + flat_idx // (c1 - c0)
        seed_c = c0 + flat_idx % (c1 - c0)
        diag["combined_at_seed"] = float(combined[seed_r, seed_c])

    # Region growing (flood fill) with the low threshold.
    candidate_mask = (combined >= low_thr) & layer_mask
    if not candidate_mask[seed_r, seed_c]:
        candidate_mask[seed_r, seed_c] = True
    labeled, n_comp = ndi.label(candidate_mask)
    if n_comp == 0:
        return np.zeros_like(layer_mask, dtype=bool), diag
    seed_label = labeled[seed_r, seed_c]
    if seed_label == 0:
        return np.zeros_like(layer_mask, dtype=bool), diag
    grain_mask = (labeled == seed_label)

    # Post-process: fill holes, smooth boundaries.
    if params.fill_holes:
        grain_mask = ndi.binary_fill_holes(grain_mask)
    if params.boundary_smooth_kernel > 0:
        k = params.boundary_smooth_kernel
        kernel = np.ones((k, k), np.uint8)
        grain_mask = cv2.morphologyEx(grain_mask.astype(np.uint8),
                                       cv2.MORPH_CLOSE, kernel).astype(bool)
        grain_mask = cv2.morphologyEx(grain_mask.astype(np.uint8),
                                       cv2.MORPH_OPEN, kernel).astype(bool)

    # Restrict to layer mask.
    grain_mask = grain_mask & layer_mask
    diag["original_grown_area"] = int(grain_mask.sum())

    # Safety: if the grown region exceeds max_grain_area_frac of the layer,
    # it's likely a "leak". Try raising the growth threshold to shrink it.
    layer_area = int(layer_mask.sum())
    max_grain_area = int(layer_area * params.max_grain_area_frac)
    if grain_mask.sum() > max_grain_area and max_grain_area > 0:
        diag["safety_check_applied"] = True
        # Try raising the growth threshold up to high_thr.
        best_shrunk = None
        best_area = grain_mask.sum()
        for raised_thr in np.linspace(low_thr + 0.05, min(0.95, high_thr + 0.30), 8):
            candidate_mask = (combined >= raised_thr) & layer_mask
            if not candidate_mask[seed_r, seed_c]:
                candidate_mask[seed_r, seed_c] = True
            labeled_t, n_comp_t = ndi.label(candidate_mask)
            if n_comp_t == 0:
                diag["safety_thresholds_tried"].append((float(raised_thr), 0))
                continue
            seed_label_t = labeled_t[seed_r, seed_c]
            if seed_label_t == 0:
                diag["safety_thresholds_tried"].append((float(raised_thr), 0))
                continue
            shrunk = (labeled_t == seed_label_t)
            if params.fill_holes:
                shrunk = ndi.binary_fill_holes(shrunk)
            shrunk = shrunk & layer_mask
            area = int(shrunk.sum())
            diag["safety_thresholds_tried"].append((float(raised_thr), area))
            if 0 < area <= max_grain_area:
                grain_mask = shrunk
                diag["fallback_used"] = (f"safety check: raised growth threshold "
                                          f"to {raised_thr:.3f}, area shrunk to {area} px")
                return grain_mask, diag
            # Track the smallest non-zero shrunk region (best effort).
            if 0 < area < best_area:
                best_area = area
                best_shrunk = shrunk

        # Could not shrink below max_grain_area — return the smallest found
        # (best effort) instead of 0. This is better than failing entirely.
        if best_shrunk is not None and best_shrunk.sum() > 0:
            grain_mask = best_shrunk
            diag["fallback_used"] = (f"safety check could not shrink below "
                                      f"max_grain_area ({max_grain_area} px); "
                                      f"returning best-effort region of "
                                      f"{best_area} px (you may want to Undo "
                                      f"and try a different click position)")
            return grain_mask, diag
        # If even the smallest region is the original (huge) one, return it
        # with a warning — let the user see and decide.
        diag["fallback_used"] = (f"safety check could not shrink the region "
                                  f"(area = {grain_mask.sum()} px > max "
                                  f"{max_grain_area} px). Returning as-is — "
                                  f"you may want to Undo and try a different "
                                  f"click position, or raise the growth threshold.")
        return grain_mask, diag

    return grain_mask, diag


def find_similar_grains(
    amplitude_maps: list[np.ndarray],
    existing_grain_mask: np.ndarray,
    layer_mask: np.ndarray,
    params: ClickSegmentationParams,
) -> list[np.ndarray]:
    """Find OTHER connected regions with the same d-spacing signature.

    After growing one grain, scan the rest of the layer for other connected
    components of the candidate_mask (pixels passing the low threshold with
    the AND condition).

    Parameters
    ----------
    amplitude_maps : list of (H, W) float — the same maps used for the main grain.
    existing_grain_mask : (H, W) bool — the already-found grain (to exclude).
    layer_mask : (H, W) bool — restrict search to this mask.
    params : ClickSegmentationParams.

    Returns
    -------
    list of (H, W) bool — each is a separate similar grain.
    """
    if not amplitude_maps:
        return []
    stack = np.stack(amplitude_maps, axis=0).astype(np.float64)
    for i in range(stack.shape[0]):
        m = float(stack[i].max())
        if m > 0:
            stack[i] = stack[i] / m
    if params.strict_and:
        combined = stack.min(axis=0)
    else:
        combined = stack.mean(axis=0)

    candidate_mask = (combined >= params.growth_threshold_frac) & layer_mask
    # Exclude the existing grain (and a small dilated margin).
    excluded = ndi.binary_dilation(existing_grain_mask, iterations=2)
    candidate_mask = candidate_mask & ~excluded

    labeled, n_comp = ndi.label(candidate_mask)
    if n_comp == 0:
        return []

    similar = []
    for cid in range(1, n_comp + 1):
        comp = (labeled == cid)
        if comp.sum() < params.min_grain_area_px:
            continue
        # Fill holes + smooth.
        if params.fill_holes:
            comp = ndi.binary_fill_holes(comp)
        if params.boundary_smooth_kernel > 0:
            k = params.boundary_smooth_kernel
            kernel = np.ones((k, k), np.uint8)
            comp = cv2.morphologyEx(comp.astype(np.uint8),
                                     cv2.MORPH_CLOSE, kernel).astype(bool)
            comp = cv2.morphologyEx(comp.astype(np.uint8),
                                     cv2.MORPH_OPEN, kernel).astype(bool)
        comp = comp & layer_mask
        if comp.sum() >= params.min_grain_area_px:
            similar.append(comp)
    return similar


def segment_grain_from_click(
    cropped_img: np.ndarray,
    cropped_mask: np.ndarray,
    click_r: int,
    click_c: int,
    pixel_size_nm: float,
    global_fft_shifted: np.ndarray | None,
    global_fft_shape: tuple[int, int],
    global_scale: float,
    fft_params: FFTSegmentationParams,
    click_params: ClickSegmentationParams,
    sliding_data: dict | None = None,
) -> GrainClickResult:
    """Full click-to-segment pipeline for one click.

    Parameters
    ----------
    cropped_img : (H, W) uint8 — cropped layer image.
    cropped_mask : (H, W) bool — layer mask (True inside the layer).
    click_r, click_c : int — click position in cropped-img coords.
    pixel_size_nm : float — pixel size of the ORIGINAL image (nm/px).
    global_fft_shifted : (H_small, W_small) complex128 or None — the global
        Wiener-filtered FFT (used for amplitude maps if use_global_fft_for_amplitude
        and use_sliding_local_fft is False).
    global_fft_shape : (H_small, W_small) — shape of the global FFT array.
    global_scale : float — downscale factor of the global FFT.
    fft_params : FFTSegmentationParams.
    click_params : ClickSegmentationParams.
    sliding_data : dict or None — precomputed sliding local FFTs (from
        compute_sliding_local_ffts). Required when
        click_params.use_sliding_local_fft is True; ignored otherwise.

    Returns
    -------
    GrainClickResult
    """
    h_full, w_full = cropped_img.shape[:2]

    # 1. Local FFT around the click.
    local_fft, local_log_power, window_bbox, local_scale = compute_local_fft(
        cropped_img, click_r, click_c,
        click_params.window_size_px, click_params.zero_pad_size_px, fft_params,
    )
    if local_fft.size == 0:
        return GrainClickResult(
            grain_mask=np.zeros((h_full, w_full), dtype=bool),
            signature=GrainSignature(),
            click_point=(click_r, click_c),
            window_bbox=window_bbox,
            message="Window too small (click near image edge).",
            success=False,
        )

    # 2. Find d-spacings (AND orientations, if sliding mode) in the local window.
    layer_area = int(cropped_mask.sum())
    orientations_deg: list[float] = []  # only populated in sliding mode
    if click_params.use_sliding_local_fft:
        # Sliding mode: find d-spacings AND their orientations.
        d_spacings, radii_local, orientations_deg, qualities, radii_1d, intens_1d = \
            find_d_spacing_and_orientation_in_local_fft(
                local_log_power, local_fft.shape, pixel_size_nm, local_scale,
                fft_params, click_params,
            )
        # In sliding mode we don't use reflections_per_d (the per-reflection
        # 2D coords in the local FFT) — we use the orientation directly.
        reflections_per_d = [[] for _ in d_spacings]
    else:
        # Legacy mode: find d-spacings + per-reflection 2D coords (for disk masks).
        d_spacings, radii_local, qualities, reflections_per_d, radii_1d, intens_1d = \
            find_grain_d_spacings(
                local_log_power, local_fft.shape, pixel_size_nm, local_scale,
                click_r - window_bbox[0], click_c - window_bbox[1],
                fft_params, click_params,
            )
    # Also get the RAW d-spacings (before quality filter) for diagnostics.
    d_spacings_raw = list(d_spacings)  # placeholder; find_grain_d_spacings already filtered

    if not d_spacings:
        return GrainClickResult(
            grain_mask=np.zeros((h_full, w_full), dtype=bool),
            signature=GrainSignature(),
            click_point=(click_r, click_c),
            window_bbox=window_bbox,
            local_power_spectrum=local_log_power,
            message="No d-spacings found in the local window (all were filtered "
                    "as noise). Try: (1) increasing the local FFT window size, "
                    "(2) raising sigma_borderline_px in the sidebar, "
                    "(3) disabling 'Apply Gaussian quality filter' if it's on.",
            success=False,
            failed_step="find_d_spacings",
            layer_area_px=layer_area,
        )

    # 3. Compute amplitude maps for each d-spacing.
    amplitude_maps = []
    radii_global = []
    seed_amplitudes = []
    amplitude_diag = []

    if click_params.use_sliding_local_fft:
        # MODE B (sliding local FFT): for each (d, θ) of the clicked grain,
        # build an amplitude map by checking each sliding window's local FFT
        # for a peak at d within the angular sector [θ−Δ, θ+Δ]. This is fully
        # local — no local→global mapping, no calibration drift issues.
        if sliding_data is None:
            return GrainClickResult(
                grain_mask=np.zeros((h_full, w_full), dtype=bool),
                signature=GrainSignature(),
                click_point=(click_r, click_c),
                window_bbox=window_bbox,
                local_power_spectrum=local_log_power,
                message="Sliding mode is ON but sliding_data is None. "
                        "Call compute_sliding_local_ffts first.",
                success=False,
                failed_step="amplitude_maps",
                layer_area_px=layer_area,
            )
        # Compute the FFT shape used by the sliding windows (for r_target calc).
        slide_h = sliding_data["log_powers"].shape[1] if sliding_data["log_powers"].size > 0 else click_params.sliding_window_size_px
        slide_w = slide_h
        slide_avg_dim = float(np.sqrt(slide_h * slide_w))
        slide_half = min(slide_h, slide_w) // 2
        for d_idx, d_nm in enumerate(d_spacings):
            theta_deg = orientations_deg[d_idx] if d_idx < len(orientations_deg) else 0.0
            amp, slide_diag = build_amplitude_map_from_sliding(
                sliding_data, d_nm, theta_deg, pixel_size_nm, fft_params, click_params,
            )
            # Compute r_target in the sliding FFT (for diagnostics).
            r_target = slide_avg_dim * pixel_size_nm / d_nm if d_nm > 0 else 0.0
            # Warning: if r_target is too small (<5) the peak falls in the DC
            # region; if too large (>slide_half - 5) it's at the FFT edge.
            r_warning = ""
            if r_target < 5.0:
                r_warning = (f"r_target={r_target:.1f}px is in the DC region (<5px). "
                             "INCREASE the sliding window size — the d-spacing "
                             f"{d_nm:.3f} nm is too small for this window.")
            elif r_target > slide_half - 5:
                r_warning = (f"r_target={r_target:.1f}px is at the FFT edge "
                             f"(max {slide_half - 5}px). DECREASE the sliding "
                             "window size or check the d-spacing calibration.")
            amplitude_maps.append(amp)
            radii_global.append(0.0)  # not used in sliding mode
            seed_amplitudes.append(float(amp[click_r, click_c]))
            amplitude_diag.append({
                "d_nm": d_nm,
                "orientation_deg": theta_deg,
                "r_global": 0.0,  # not applicable in sliding mode
                "r_target_sliding": float(r_target),
                "reflections": [],
                "best_snr": float(slide_diag.get("n_with_peak", 0))
                              / max(1, slide_diag.get("n_windows", 1)),
                "used_disk": True,  # sliding mode = orientation-specific
                "mode": "sliding",
                "n_windows": slide_diag.get("n_windows", 0),
                "n_with_peak": slide_diag.get("n_with_peak", 0),
                "sliding_window_size": click_params.sliding_window_size_px,
                "sliding_fft_size": int(slide_h),
                "r_warning": r_warning,
            })
    else:
        # LEGACY MODE (global FFT with disk/ring masks).
        # CRITICAL: when use_orientation_specific_masks is True, we use
        # DISK masks around each individual 2D reflection (with its angle)
        # instead of a RING mask. AUTO-ALIGN searches for the actual 2D max
        # in a neighborhood of the expected position.
        lh, lw = local_fft.shape
        avg_dim_local = float(np.sqrt(lh * lw))
        gh, gw = global_fft_shape
        avg_dim_global = float(np.sqrt(gh * gw))
        pixel_size_small_global = pixel_size_nm / global_scale
        center_local = (lh / 2.0, lw / 2.0)
        center_global = (gh / 2.0, gw / 2.0)

        # Pre-compute the global FFT power (for auto-align peak search).
        global_power = np.abs(global_fft_shifted) ** 2
        global_log_power = np.log1p(global_power)
        yy_g, xx_g = np.ogrid[:gh, :gw]
        # Auto-align search radius: ±5 px around the expected position.
        align_search_radius = 5

        for d_idx, d_nm in enumerate(d_spacings):
            # Global radius for this d-spacing.
            r_global = avg_dim_global * pixel_size_small_global / d_nm
            radii_global.append(r_global)

            if click_params.use_orientation_specific_masks and reflections_per_d[d_idx]:
                # ORIENTATION-SPECIFIC with AUTO-ALIGN.
                disk_radius = click_params.disk_mask_radius_px
                combined_amp = np.zeros((gh, gw), dtype=np.float64)
                refl_diag = []
                for refl_idx, (lr, lc) in enumerate(reflections_per_d[d_idx]):
                    dy_local = lr - center_local[0]
                    dx_local = lc - center_local[1]
                    theta = np.arctan2(dy_local, dx_local)
                    gr_exp = center_global[0] + r_global * np.sin(theta)
                    gc_exp = center_global[1] + r_global * np.cos(theta)
                    r0 = max(0, int(gr_exp) - align_search_radius)
                    r1 = min(gh, int(gr_exp) + align_search_radius + 1)
                    c0 = max(0, int(gc_exp) - align_search_radius)
                    c1 = min(gw, int(gc_exp) + align_search_radius + 1)
                    window = global_log_power[r0:r1, c0:c1]
                    if window.size == 0:
                        continue
                    local_max_idx = int(np.argmax(window))
                    win_h, win_w = window.shape
                    dr_max = local_max_idx // win_w
                    dc_max = local_max_idx % win_w
                    gr_actual = r0 + dr_max
                    gc_actual = c0 + dc_max
                    peak_intensity = float(window[dr_max, dc_max])
                    bg_mask = np.ones_like(window, dtype=bool)
                    bg_mask[dr_max, dc_max] = False
                    background_intensity = float(np.median(window[bg_mask])) if bg_mask.sum() > 0 else 0.0
                    refl_snr = peak_intensity / max(background_intensity, 1e-9)
                    refl_diag.append({
                        "refl_idx": refl_idx,
                        "expected_pos": (float(gr_exp), float(gc_exp)),
                        "actual_pos": (float(gr_actual), float(gc_actual)),
                        "shift_px": float(np.sqrt((gr_actual - gr_exp) ** 2 + (gc_actual - gc_exp) ** 2)),
                        "peak_intensity": peak_intensity,
                        "background_intensity": background_intensity,
                        "snr": refl_snr,
                    })
                    disk_mask = ((yy_g - gr_actual) ** 2 + (xx_g - gc_actual) ** 2) <= disk_radius ** 2
                    amp = _compute_amplitude_map(global_fft_shifted, disk_mask)
                    combined_amp += amp
                amp = combined_amp
                amp_max = float(amp.max())
                if amp_max > 0:
                    amp = amp / amp_max
                best_refl_snr = max((d["snr"] for d in refl_diag), default=0.0)
                used_disk = True
                if best_refl_snr < 2.0:
                    mask = _create_ring_mask((gh, gw), center_global, r_global,
                                              click_params.ring_mask_width_px)
                    amp = _compute_amplitude_map(global_fft_shifted, mask)
                    amp_max = float(amp.max())
                    if amp_max > 0:
                        amp = amp / amp_max
                    used_disk = False
                amplitude_diag.append({
                    "d_nm": d_nm,
                    "r_global": r_global,
                    "reflections": refl_diag,
                    "best_snr": best_refl_snr,
                    "used_disk": used_disk,
                    "mode": "global_disk" if used_disk else "global_ring",
                })
            else:
                # FALLBACK: ring mask (averages over all orientations).
                mask = _create_ring_mask((gh, gw), center_global, r_global,
                                          click_params.ring_mask_width_px)
                amp = _compute_amplitude_map(global_fft_shifted, mask)
                amp_max = float(amp.max())
                if amp_max > 0:
                    amp = amp / amp_max
                amplitude_diag.append({
                    "d_nm": d_nm,
                    "r_global": r_global,
                    "reflections": [],
                    "best_snr": 0.0,
                    "used_disk": False,
                    "mode": "global_ring",
                })

            # Upscale to full-res.
            if amp.shape != (h_full, w_full):
                amp = cv2.resize(amp, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
            amplitude_maps.append(amp)
            seed_amplitudes.append(float(amp[click_r, click_c]))

    # 4. Region growing with hysteresis + AND.
    grain_mask, grow_diag = grow_grain_region(
        amplitude_maps, click_r, click_c, cropped_mask, click_params,
    )
    # Compute combined seed amplitude (for diagnostics).
    if click_params.strict_and and seed_amplitudes:
        seed_combined = float(min(seed_amplitudes))
    elif seed_amplitudes:
        seed_combined = float(np.mean(seed_amplitudes))
    else:
        seed_combined = 0.0
    grown_area = int(grain_mask.sum())

    if grain_mask.sum() == 0:
        # FALLBACK: try with only the STRONGEST d-spacing (single-d mode).
        # This relaxes the AND condition — useful when one d-spacing is noisy
        # or doesn't separate grains well.
        if len(amplitude_maps) > 1:
            # Find the d-spacing with the highest seed amplitude.
            best_idx = int(np.argmax(seed_amplitudes))
            single_amp = [amplitude_maps[best_idx]]
            single_d = [d_spacings[best_idx]]
            single_r = [radii_global[best_idx]]
            single_q = [qualities[best_idx]]
            single_refl = [reflections_per_d[best_idx]]
            grain_mask_fb, fb_diag = grow_grain_region(
                single_amp, click_r, click_c, cropped_mask, click_params,
            )
            if grain_mask_fb.sum() > 0:
                # Fallback succeeded — use it.
                grain_mask = grain_mask_fb
                amplitude_maps = single_amp
                d_spacings = single_d
                radii_global = single_r
                qualities = single_q
                reflections_per_d = single_refl
                grown_area = int(grain_mask.sum())
                grow_diag = fb_diag
                grow_diag["fallback_used"] = ("single-d fallback: used only the "
                                              "strongest d-spacing (idx=" + str(best_idx)
                                              + ", d=" + f"{single_d[0]:.3f}" + " nm). "
                                              + grow_diag.get("fallback_used", ""))

    # 4.5 Structure tensor cleanup (optional post-filter): remove amorphous /
    # wrongly-oriented regions from the grown mask. Orientation targets are
    # the grain's detected reflection angles (sliding mode); in legacy mode
    # they may be empty → coherence-only gate.
    cleanup_diag = None
    if click_params.use_structure_cleanup and grain_mask.sum() > 0:
        grain_mask, cleanup_diag = _apply_structure_cleanup(
            cropped_img, grain_mask, click_params, orientations_deg,
        )
        grown_area = int(grain_mask.sum())

    if grain_mask.sum() == 0:
        # Build a detailed diagnostic message.
        amp_strs = [f"d={d:.3f}nm amp={a:.3f}" for d, a in zip(d_spacings, seed_amplitudes)]
        # Build a more informative message based on diagnostics.
        if grow_diag.get("safety_check_applied"):
            msg = (f"Region growing failed: the grown region was too large "
                   f"(leaked across grain boundaries) and the safety check "
                   f"could not shrink it below {int(layer_area * click_params.max_grain_area_frac)} px. "
                   f"Original grown area = {grow_diag.get('original_grown_area', 0)} px. "
                   f"Found {len(d_spacings)} d-spacing(s): {'; '.join(amp_strs)}. "
                   f"Try: (1) raise the growth threshold, (2) increase disk mask radius, "
                   f"(3) increase max grain area fraction, (4) try a different click position.")
        else:
            msg = (f"Region growing failed. Found {len(d_spacings)} d-spacing(s): "
                   f"{'; '.join(amp_strs)}. Combined seed amplitude = {seed_combined:.3f} "
                   f"(need ≥ {click_params.seed_threshold_frac:.2f}). "
                   f"Grown area = {grown_area} px (layer = {layer_area} px). "
                   f"Try: (1) lowering seed threshold, (2) increasing the local "
                   f"FFT window, (3) disabling 'Strict AND' (use single-d fallback).")
        return GrainClickResult(
            grain_mask=grain_mask,
            signature=GrainSignature(d_spacings_nm=d_spacings, radii_px=radii_global,
                                      qualities=qualities, orientations_deg=orientations_deg),
            click_point=(click_r, click_c),
            window_bbox=window_bbox,
            local_power_spectrum=local_log_power,
            message=msg,
            success=False,
            failed_step="region_growing",
            d_spacings_kept=list(d_spacings),
            seed_amplitudes=seed_amplitudes,
            seed_combined=seed_combined,
            grown_area_px=grown_area,
            layer_area_px=layer_area,
            original_grown_area_px=int(grow_diag.get("original_grown_area", 0)),
            safety_check_applied=bool(grow_diag.get("safety_check_applied", False)),
            safety_thresholds_tried=list(grow_diag.get("safety_thresholds_tried", [])),
            fallback_used=str(grow_diag.get("fallback_used", "")),
            amplitude_diag=amplitude_diag,
        )

    # 5. Auto-find similar grains.
    similar = []
    if click_params.auto_find_similar:
        similar = find_similar_grains(
            amplitude_maps, grain_mask, cropped_mask, click_params,
        )
        similar = _cleanup_similar_masks(
            cropped_img, similar, click_params, orientations_deg,
        )

    signature = GrainSignature(
        d_spacings_nm=d_spacings, radii_px=radii_global, qualities=qualities,
        orientations_deg=orientations_deg,
    )
    n_d = len(d_spacings)
    n_similar = len(similar)
    d_str = ", ".join(f"{d:.3f}" for d in d_spacings)
    msg = (f"Grain segmented: {n_d} d-spacing(s) found ({d_str} nm), "
           f"area = {int(grain_mask.sum())} px.")
    if n_similar > 0:
        msg += f" Found {n_similar} similar grain(s) — review below."
    if grow_diag.get("fallback_used"):
        msg += f" Note: {grow_diag['fallback_used']}"
    if cleanup_diag and cleanup_diag.get("removed_px"):
        msg += (f" Structure-tensor cleanup removed {cleanup_diag['removed_px']} px "
                f"({cleanup_diag['removed_frac'] * 100:.0f}%) as amorphous/"
                f"wrong-orientation.")
    elif cleanup_diag and cleanup_diag.get("note"):
        msg += f" Cleanup note: {cleanup_diag['note']}"

    return GrainClickResult(
        grain_mask=grain_mask,
        signature=signature,
        click_point=(click_r, click_c),
        window_bbox=window_bbox,
        local_power_spectrum=local_log_power,
        similar_grain_masks=similar,
        message=msg,
        success=True,
        d_spacings_kept=list(d_spacings),
        seed_amplitudes=seed_amplitudes,
        seed_combined=seed_combined,
        grown_area_px=int(grain_mask.sum()),
        layer_area_px=layer_area,
        original_grown_area_px=int(grow_diag.get("original_grown_area", 0)),
        safety_check_applied=bool(grow_diag.get("safety_check_applied", False)),
        safety_thresholds_tried=list(grow_diag.get("safety_thresholds_tried", [])),
        fallback_used=str(grow_diag.get("fallback_used", "")),
        amplitude_diag=amplitude_diag,
        cleanup_diag=cleanup_diag,
    )
