"""
layer_detect.py — Detection of horizontal layers and the microscope info-strip
in TEM cross-section images.

Conventions
-----------
* Rows are indexed from 0 (top) to H-1 (bottom).
* A "band" is a contiguous range of rows [y0, y1) representing a layer candidate.
* The "info-strip" is the bottom (and sometimes top) region of the image where
  the microscope software embeds the scale bar, magnification label, etc. It is
  not part of the specimen and must be excluded from analysis.

The detection is intentionally heuristic and robust:
1. Compute the row-mean intensity profile (smoothed).
2. Find the largest gradient magnitudes in the profile → these are layer
   boundaries.
3. Group consecutive rows between boundaries into zones. Each zone with a
   distinct mean intensity is a candidate band.
4. Detect the info-strip by finding strong boundaries near the top and bottom
   edges of the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Band:
    """A horizontal layer candidate: rows [y0, y1)."""
    y0: int
    y1: int
    mean_intensity: float
    std_intensity: float
    height_px: int
    height_nm: float = 0.0

    def label(self, pixel_size_nm: float) -> str:
        return (f"rows {self.y0}–{self.y1}  "
                f"(h={self.height_px}px = {self.height_nm:.1f}nm, "
                f"mean={self.mean_intensity:.0f})")


@dataclass
class InfoStrip:
    """Detected microscope info-strip regions (rows to exclude)."""
    top: Optional[tuple[int, int]] = None  # (y0, y1) or None
    bottom: Optional[tuple[int, int]] = None

    def excluded_rows(self) -> set[int]:
        rows: set[int] = set()
        if self.top is not None:
            rows.update(range(self.top[0], self.top[1]))
        if self.bottom is not None:
            rows.update(range(self.bottom[0], self.bottom[1]))
        return rows

    def is_excluded(self, y: int) -> bool:
        if self.top is not None and self.top[0] <= y < self.top[1]:
            return True
        if self.bottom is not None and self.bottom[0] <= y < self.bottom[1]:
            return True
        return False


# === Row profile ===

def compute_row_profile(image: np.ndarray, smooth_px: int = 21) -> np.ndarray:
    """Compute the smoothed row-mean intensity profile.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    smooth_px : int
        Smoothing window size in rows (odd). Set to 1 for no smoothing.

    Returns
    -------
    np.ndarray of shape (H,) — mean intensity per row.
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    profile = image.astype(np.float64).mean(axis=1)
    if smooth_px > 1:
        if smooth_px % 2 == 0:
            smooth_px += 1
        kernel = np.ones(smooth_px, dtype=np.float64) / smooth_px
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def compute_col_profile(image: np.ndarray, smooth_px: int = 21) -> np.ndarray:
    """Column-mean profile (used to check whether the layering is horizontal)."""
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    profile = image.astype(np.float64).mean(axis=0)
    if smooth_px > 1:
        if smooth_px % 2 == 0:
            smooth_px += 1
        kernel = np.ones(smooth_px, dtype=np.float64) / smooth_px
        profile = np.convolve(profile, kernel, mode="same")
    return profile


# === Info-strip detection ===

def detect_info_strip(
    image: np.ndarray,
    profile: Optional[np.ndarray] = None,
    edge_fraction: float = 0.25,
    min_strip_height_px: int = 5,
    window_px: int = 151,
    ratio: float = 1.8,
) -> InfoStrip:
    """Detect microscope info-strips at the top and bottom of the image.

    The info-strip contains text, scale bar and labels → it has much higher
    **row-to-row intensity variation** than the bulk sample. We use the
    rolling standard deviation of the row-mean profile (over a large window)
    as the discriminating signal: any window that overlaps the strip has
    elevated std; pure-bulk windows have low std.

    The strip boundary is the row where the rolling std first drops below
    ``ratio × median bulk rolling std`` (when scanning inward from each edge).

    Parameters
    ----------
    image : np.ndarray
    profile : np.ndarray, optional
        Pre-computed **un-smoothed** row profile. If None, computed internally.
    edge_fraction : float
        Fraction of image height (from each edge) to scan (default 25%).
    min_strip_height_px : int
        Minimum strip height in pixels.
    window_px : int
        Rolling-window size (odd) for std estimation. Large windows (~150 px)
        ensure that any window overlapping the strip has elevated std.
    ratio : float
        Strip-vs-bulk threshold multiplier.

    Returns
    -------
    InfoStrip
    """
    if profile is None:
        # Use un-smoothed profile for std estimation (smoothing would
        # suppress exactly the row-to-row variation we want to detect).
        profile = compute_row_profile(image, smooth_px=1)
    h = image.shape[0]
    max_edge = max(int(h * edge_fraction), min_strip_height_px + 1)

    # Rolling std of the row-mean profile.
    if window_px % 2 == 0:
        window_px += 1
    half = window_px // 2
    rolling_std = np.zeros(h, dtype=np.float64)
    for y in range(h):
        lo = max(0, y - half)
        hi = min(h, y + half + 1)
        rolling_std[y] = float(np.std(profile[lo:hi]))

    # Bulk rolling-std statistics — middle 60% of the image.
    mid_lo = int(h * 0.2)
    mid_hi = int(h * 0.8)
    bulk_std_median = float(np.median(rolling_std[mid_lo:mid_hi])) if mid_hi > mid_lo else 0.0
    if bulk_std_median < 1e-6:
        bulk_std_median = float(np.median(rolling_std)) if rolling_std.size else 1.0
    std_thr = bulk_std_median * ratio

    # --- Top strip: scan down from row 0; strip continues while rolling std >= thr. ---
    top_strip: Optional[tuple[int, int]] = None
    end_y = 0
    for y in range(max_edge):
        if rolling_std[y] >= std_thr:
            end_y = y + 1
        else:
            # Allow short gaps inside the strip (quiet rows between labels).
            if end_y > 0 and y + 1 < max_edge and rolling_std[y + 1] >= std_thr:
                continue
            if end_y >= min_strip_height_px:
                break
            end_y = 0
    if end_y >= min_strip_height_px and end_y < max_edge:
        top_strip = (0, end_y)

    # --- Bottom strip: scan up from row h-1; strip continues while rolling std >= thr. ---
    bottom_strip: Optional[tuple[int, int]] = None
    start_y = h
    for y in range(h - 1, h - max_edge, -1):
        if rolling_std[y] >= std_thr:
            start_y = y
        else:
            if start_y < h and y - 1 > h - max_edge and rolling_std[y - 1] >= std_thr:
                continue
            if (h - start_y) >= min_strip_height_px:
                break
            start_y = h
    if (h - start_y) >= min_strip_height_px and start_y > (h - max_edge):
        bottom_strip = (start_y, h)

    return InfoStrip(top=top_strip, bottom=bottom_strip)


# === Band suggestion ===

def _gradient_boundaries(profile: np.ndarray, min_zone_height: int = 20,
                          grad_quantile: float = 0.85) -> list[int]:
    """Find row indices where the profile gradient is locally large.

    Returns a sorted list of boundary row indices.
    """
    grad = np.gradient(profile)
    abs_grad = np.abs(grad)
    if abs_grad.size == 0:
        return []
    thr = float(np.quantile(abs_grad, grad_quantile))
    # Find local maxima above the threshold, separated by at least min_zone_height.
    boundaries: list[int] = []
    in_peak = False
    peak_start = 0
    peak_max_val = 0.0
    peak_max_idx = 0
    for i, g in enumerate(abs_grad):
        if g > thr:
            if not in_peak:
                in_peak = True
                peak_start = i
                peak_max_val = g
                peak_max_idx = i
            else:
                if g > peak_max_val:
                    peak_max_val = g
                    peak_max_idx = i
        else:
            if in_peak:
                in_peak = False
                if boundaries and peak_max_idx - boundaries[-1] < min_zone_height:
                    # Too close to the previous boundary — keep the larger one.
                    if peak_max_val > abs_grad[boundaries[-1]]:
                        boundaries[-1] = peak_max_idx
                else:
                    boundaries.append(peak_max_idx)
    return boundaries


def suggest_bands(
    image: np.ndarray,
    pixel_size_nm: float,
    info_strip: Optional[InfoStrip] = None,
    profile: Optional[np.ndarray] = None,
    min_band_height_px: int = 30,
    max_bands: int = 10,
) -> list[Band]:
    """Suggest horizontal layer candidates based on the row-mean profile.

    The algorithm:
    1. Find rows where the profile gradient is large (boundaries).
    2. Slice the image into zones between consecutive boundaries.
    3. Drop zones that overlap with the info-strip.
    4. Drop zones shorter than `min_band_height_px`.
    5. Sort the remaining zones by height descending; return up to `max_bands`.

    Parameters
    ----------
    image : np.ndarray
    pixel_size_nm : float
        Pixel size in nanometres (used to compute band height in nm).
    info_strip : InfoStrip, optional
        Detected info-strip; bands overlapping it are dropped.
    profile : np.ndarray, optional
        Pre-computed smoothed row profile.
    min_band_height_px : int
        Minimum band height in pixels.
    max_bands : int
        Maximum number of bands to return.

    Returns
    -------
    list[Band]
        Sorted by descending height.
    """
    if profile is None:
        profile = compute_row_profile(image)
    h = image.shape[0]

    boundaries = _gradient_boundaries(profile, min_zone_height=min_band_height_px)
    # Add explicit image top and bottom as boundaries.
    edges = [0] + boundaries + [h]
    edges = sorted(set(edges))

    bands: list[Band] = []
    for i in range(len(edges) - 1):
        y0 = edges[i]
        y1 = edges[i + 1]
        # Trim to avoid overlap with the info-strip.
        if info_strip is not None:
            if info_strip.top is not None:
                # Strip occupies [top[0], top[1]).
                if y1 <= info_strip.top[1]:
                    continue  # band entirely inside top strip
                if y0 < info_strip.top[1]:
                    y0 = info_strip.top[1]
            if info_strip.bottom is not None:
                # Strip occupies [bottom[0], h).
                if y0 >= info_strip.bottom[0]:
                    continue  # band entirely inside bottom strip
                if y1 > info_strip.bottom[0]:
                    y1 = info_strip.bottom[0]
        if y1 - y0 < min_band_height_px:
            continue
        sub = image[y0:y1]
        mean_i = float(sub.mean())
        std_i = float(sub.std())
        height_px = y1 - y0
        height_nm = height_px * pixel_size_nm
        bands.append(Band(
            y0=y0, y1=y1,
            mean_intensity=mean_i,
            std_intensity=std_i,
            height_px=height_px,
            height_nm=height_nm,
        ))

    # Sort by descending height (most prominent layers first).
    bands.sort(key=lambda b: b.height_px, reverse=True)
    return bands[:max_bands]


def band_for_row_range(y0: int, y1: int, image: np.ndarray, pixel_size_nm: float) -> Band:
    """Build a Band object for an explicit [y0, y1) row range."""
    if y0 < 0:
        y0 = 0
    if y1 > image.shape[0]:
        y1 = image.shape[0]
    if y1 <= y0:
        y0, y1 = 0, image.shape[0]
    sub = image[y0:y1]
    return Band(
        y0=int(y0), y1=int(y1),
        mean_intensity=float(sub.mean()),
        std_intensity=float(sub.std()),
        height_px=int(y1 - y0),
        height_nm=float((y1 - y0) * pixel_size_nm),
    )
