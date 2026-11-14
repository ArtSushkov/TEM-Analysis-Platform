"""
measurements.py — Per-grain measurements, lognormal fitting, summary statistics.

For each labeled grain (output of `segmentation.segment_grains`), computes:
* area (px² and nm²)
* equivalent circular diameter (nm) — diameter of a circle with the same area
* perimeter (nm)
* circularity (4πA/P²; 1 = perfect circle, →0 = elongated)
* eccentricity (0 = circular, →1 = elongated; from fitted ellipse)
* major & minor axis lengths (nm) of the fitted ellipse
* aspect ratio (minor / major)
* orientation (degrees of the major axis w.r.t. the X-axis, in [-90, 90])
* centroid (px and nm coordinates)

For a collection of grains, also fits a lognormal distribution to the
equivalent-diameter distribution (standard for grain sizes per Grachev's
dissertation) and computes summary statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional

import numpy as np
from scipy import stats
from skimage import measure


# === Per-grain measurement ===

@dataclass
class Grain:
    """Per-grain measurement results."""
    label: int
    # Areas.
    area_px2: float
    area_nm2: float
    # Equivalent circular diameter (= 2 * sqrt(A/π)).
    equivalent_diameter_nm: float
    # Perimeter.
    perimeter_nm: float
    # Shape.
    circularity: float           # 4πA/P² (1 = circle)
    eccentricity: float          # 0 = circle, 1 = line
    major_axis_nm: float
    minor_axis_nm: float
    aspect_ratio: float          # minor / major (in (0, 1])
    orientation_deg: float       # major-axis angle w.r.t. X-axis, [-90, 90]
    # Position.
    centroid_px: tuple[float, float]   # (row, col)
    centroid_nm: tuple[float, float]   # (y_nm, x_nm) within the band
    bbox: tuple[int, int, int, int]    # (min_row, min_col, max_row, max_col)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["centroid_px"] = list(self.centroid_px)
        d["centroid_nm"] = list(self.centroid_nm)
        d["bbox"] = list(self.bbox)
        return d


def measure_grains(labels: np.ndarray, pixel_size_nm: float) -> list[Grain]:
    """Measure every grain in a labeled image.

    Parameters
    ----------
    labels : np.ndarray
        Int label image (0 = background). Output of `segment_grains`.
    pixel_size_nm : float
        Pixel size in nanometres per pixel.

    Returns
    -------
    list[Grain]
        One `Grain` per labeled region, sorted by label.
    """
    if labels.ndim != 2:
        raise ValueError("Expected a 2-D label image.")

    px_area_to_nm2 = float(pixel_size_nm * pixel_size_nm)
    px_to_nm = float(pixel_size_nm)

    grains: list[Grain] = []
    for region in measure.regionprops(labels):
        label = int(region.label)
        area_px2 = float(region.area)
        area_nm2 = area_px2 * px_area_to_nm2
        eq_diameter_nm = float(region.equivalent_diameter) * px_to_nm
        perimeter_nm = float(region.perimeter) * px_to_nm
        # Circularity: 4πA/P². Guard against P = 0.
        if perimeter_nm > 0:
            circularity = float(4.0 * np.pi * area_nm2 / (perimeter_nm * perimeter_nm))
            circularity = min(circularity, 1.0)  # numerical guard
        else:
            circularity = 0.0
        eccentricity = float(region.eccentricity)
        major_nm = float(region.major_axis_length) * px_to_nm
        minor_nm = float(region.minor_axis_length) * px_to_nm
        aspect_ratio = float(minor_nm / major_nm) if major_nm > 0 else 0.0
        # skimage's orientation is in radians, in [-π/2, π/2], measured
        # counter-clockwise from the X-axis (column axis). Convert to degrees.
        orientation_deg = float(np.degrees(region.orientation))
        cy, cx = region.centroid
        centroid_px = (float(cy), float(cx))
        centroid_nm = (float(cy) * px_to_nm, float(cx) * px_to_nm)
        min_row, min_col, max_row, max_col = region.bbox
        bbox = (int(min_row), int(min_col), int(max_row), int(max_col))

        grains.append(Grain(
            label=label,
            area_px2=area_px2,
            area_nm2=area_nm2,
            equivalent_diameter_nm=eq_diameter_nm,
            perimeter_nm=perimeter_nm,
            circularity=circularity,
            eccentricity=eccentricity,
            major_axis_nm=major_nm,
            minor_axis_nm=minor_nm,
            aspect_ratio=aspect_ratio,
            orientation_deg=orientation_deg,
            centroid_px=centroid_px,
            centroid_nm=centroid_nm,
            bbox=bbox,
        ))

    grains.sort(key=lambda g: g.label)
    return grains


# === Lognormal fit ===

@dataclass
class LognormalFit:
    """Lognormal fit results for a 1-D sample.

    A lognormal distribution has the form:
        f(x) = 1/(x σ sqrt(2π)) · exp(-(ln x - μ)² / (2 σ²))

    where μ (mu) and σ (sigma) are the mean and std of the underlying normal
    distribution of ln(x). The median is exp(μ), the mode is exp(μ - σ²),
    the mean is exp(μ + σ²/2).
    """
    mu: float                 # mean of ln(x)
    sigma: float              # std of ln(x)
    median: float             # exp(μ)
    mode: float               # exp(μ - σ²)
    mean: float               # exp(μ + σ²/2)
    std: float                # sqrt(exp(σ²) - 1) · exp(μ + σ²/2)
    n_samples: int
    success: bool
    method: str = "scipy.stats.lognorm.fit (MLE)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_lognormal(values: np.ndarray | list[float]) -> Optional[LognormalFit]:
    """Fit a lognormal distribution to a 1-D sample via MLE.

    Parameters
    ----------
    values : array-like of float
        Positive values to fit (e.g. equivalent diameters in nm).

    Returns
    -------
    LognormalFit | None
        None if the fit failed (e.g. fewer than 5 positive values).
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = arr.size
    if n < 5:
        return None

    try:
        # scipy.stats.lognorm.fit returns (sigma, loc, scale).
        # We fix loc=0 (two-parameter lognormal).
        sigma, loc, scale = stats.lognorm.fit(arr, floc=0.0)
        mu = float(np.log(scale))   # scale = exp(mu)
        sigma = float(sigma)
        # Theoretical moments.
        median = float(np.exp(mu))
        mode = float(np.exp(mu - sigma * sigma))
        mean = float(np.exp(mu + sigma * sigma / 2.0))
        variance = float((np.exp(sigma * sigma) - 1.0) * np.exp(2.0 * mu + sigma * sigma))
        std = float(np.sqrt(max(variance, 0.0)))
        return LognormalFit(
            mu=mu, sigma=sigma, median=median, mode=mode, mean=mean,
            std=std, n_samples=n, success=True,
        )
    except Exception:
        return None


def lognormal_pdf(x: np.ndarray, fit: LognormalFit) -> np.ndarray:
    """Lognormal probability density function evaluated at x."""
    if fit is None or fit.sigma <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return stats.lognorm.pdf(x, s=fit.sigma, scale=np.exp(fit.mu))


# === Summary statistics ===

@dataclass
class SummaryStats:
    """Summary statistics for a 1-D sample."""
    n: int
    mean: float
    std: float
    median: float
    q1: float           # 25th percentile
    q3: float           # 75th percentile
    iqr: float          # Q3 - Q1
    min: float
    max: float
    total: float        # sum of all values
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summary_stats(values: np.ndarray | list[float], unit: str = "") -> SummaryStats:
    """Compute summary statistics for a 1-D sample.

    Returns an empty-stats object (n=0) if the input is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return SummaryStats(n=0, mean=0.0, std=0.0, median=0.0, q1=0.0, q3=0.0,
                            iqr=0.0, min=0.0, max=0.0, total=0.0, unit=unit)
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    return SummaryStats(
        n=int(n),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if n > 1 else 0.0,
        median=float(np.median(arr)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        min=float(arr.min()),
        max=float(arr.max()),
        total=float(arr.sum()),
        unit=unit,
    )


# === Aggregate per-image statistics ===

@dataclass
class ImageGrainStats:
    """Aggregate grain statistics for one image."""
    n_grains: int
    total_grain_area_nm2: float
    layer_area_nm2: float
    coverage_fraction: float       # total_grain_area / layer_area
    area_stats: SummaryStats       # in nm²
    eq_diameter_stats: SummaryStats  # in nm
    perimeter_stats: SummaryStats  # in nm
    circularity_stats: SummaryStats
    aspect_ratio_stats: SummaryStats
    orientation_stats: SummaryStats  # in degrees
    lognormal_fit: Optional[LognormalFit] = None  # fit on eq_diameter

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_grains": self.n_grains,
            "total_grain_area_nm2": self.total_grain_area_nm2,
            "layer_area_nm2": self.layer_area_nm2,
            "coverage_fraction": self.coverage_fraction,
            "area_stats": self.area_stats.to_dict(),
            "eq_diameter_stats": self.eq_diameter_stats.to_dict(),
            "perimeter_stats": self.perimeter_stats.to_dict(),
            "circularity_stats": self.circularity_stats.to_dict(),
            "aspect_ratio_stats": self.aspect_ratio_stats.to_dict(),
            "orientation_stats": self.orientation_stats.to_dict(),
            "lognormal_fit": self.lognormal_fit.to_dict() if self.lognormal_fit else None,
        }


def compute_image_stats(grains: list[Grain], layer_height_nm: float,
                        layer_width_nm: float) -> ImageGrainStats:
    """Compute aggregate statistics for a list of grains in one image."""
    if not grains:
        empty = SummaryStats(n=0, mean=0.0, std=0.0, median=0.0, q1=0.0, q3=0.0,
                              iqr=0.0, min=0.0, max=0.0, total=0.0)
        return ImageGrainStats(
            n_grains=0,
            total_grain_area_nm2=0.0,
            layer_area_nm2=float(layer_height_nm * layer_width_nm),
            coverage_fraction=0.0,
            area_stats=empty,
            eq_diameter_stats=SummaryStats(n=0, mean=0.0, std=0.0, median=0.0, q1=0.0, q3=0.0, iqr=0.0, min=0.0, max=0.0, total=0.0, unit="nm"),
            perimeter_stats=SummaryStats(n=0, mean=0.0, std=0.0, median=0.0, q1=0.0, q3=0.0, iqr=0.0, min=0.0, max=0.0, total=0.0, unit="nm"),
            circularity_stats=empty,
            aspect_ratio_stats=empty,
            orientation_stats=SummaryStats(n=0, mean=0.0, std=0.0, median=0.0, q1=0.0, q3=0.0, iqr=0.0, min=0.0, max=0.0, total=0.0, unit="deg"),
            lognormal_fit=None,
        )

    areas = np.array([g.area_nm2 for g in grains])
    diams = np.array([g.equivalent_diameter_nm for g in grains])
    perims = np.array([g.perimeter_nm for g in grains])
    circs = np.array([g.circularity for g in grains])
    asps = np.array([g.aspect_ratio for g in grains])
    oris = np.array([g.orientation_deg for g in grains])

    total_area = float(areas.sum())
    layer_area = float(layer_height_nm * layer_width_nm)
    coverage = total_area / layer_area if layer_area > 0 else 0.0

    return ImageGrainStats(
        n_grains=len(grains),
        total_grain_area_nm2=total_area,
        layer_area_nm2=layer_area,
        coverage_fraction=float(coverage),
        area_stats=summary_stats(areas, unit="nm²"),
        eq_diameter_stats=summary_stats(diams, unit="nm"),
        perimeter_stats=summary_stats(perims, unit="nm"),
        circularity_stats=summary_stats(circs, unit=""),
        aspect_ratio_stats=summary_stats(asps, unit=""),
        orientation_stats=summary_stats(oris, unit="deg"),
        lognormal_fit=fit_lognormal(diams),
    )
