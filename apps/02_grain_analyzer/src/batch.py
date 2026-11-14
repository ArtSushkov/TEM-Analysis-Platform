"""
batch.py — Aggregate per-image JSON reports into combined statistics.

The Batch tab takes one or more JSON reports (produced by the Single Image
tab) and pools all grain measurements into a single dataset. From the pooled
dataset it computes:

* Per-image summary table (one row per image).
* Pooled summary statistics for area, equivalent diameter, perimeter,
  circularity, aspect ratio, orientation.
* Pooled lognormal fit on equivalent diameter (the "more statistics for the
  same layer" requested by the user).
* Total grain count and total analysed area.

The JSON reports are self-contained (carry pixel size, layer y-range, params,
and the per-grain measurements), so this aggregation requires **no extra user
actions** beyond dropping the JSON files in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .measurements import (
    Grain, ImageGrainStats, LognormalFit, SummaryStats,
    compute_image_stats, fit_lognormal, summary_stats,
)


# === Loading ===

def _grain_from_dict(d: dict[str, Any]) -> Grain:
    """Reconstruct a Grain from a report dict."""
    return Grain(
        label=int(d["label"]),
        area_px2=float(d["area_px2"]),
        area_nm2=float(d["area_nm2"]),
        equivalent_diameter_nm=float(d["equivalent_diameter_nm"]),
        perimeter_nm=float(d["perimeter_nm"]),
        circularity=float(d["circularity"]),
        eccentricity=float(d["eccentricity"]),
        major_axis_nm=float(d["major_axis_nm"]),
        minor_axis_nm=float(d["minor_axis_nm"]),
        aspect_ratio=float(d["aspect_ratio"]),
        orientation_deg=float(d["orientation_deg"]),
        centroid_px=tuple(d["centroid_px"]),
        centroid_nm=tuple(d["centroid_nm"]),
        bbox=tuple(d["bbox"]),
    )


@dataclass
class PerImageRow:
    """One row of the per-image summary table."""
    file_name: str
    pixel_size_nm: float
    magnification: float | None
    layer_source: str | None        # "sam" | "polygon" | "manual_rows"
    layer_area_nm2: float
    layer_height_nm: float
    layer_width_nm: float
    n_grains: int
    mean_eq_diameter_nm: float
    std_eq_diameter_nm: float
    median_eq_diameter_nm: float
    mean_area_nm2: float
    total_grain_area_nm2: float
    coverage_fraction: float
    lognormal_mu: float | None
    lognormal_sigma: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "pixel_size_nm": self.pixel_size_nm,
            "magnification": self.magnification,
            "layer_source": self.layer_source,
            "layer_area_nm2": self.layer_area_nm2,
            "layer_height_nm": self.layer_height_nm,
            "layer_width_nm": self.layer_width_nm,
            "n_grains": self.n_grains,
            "mean_eq_diameter_nm": self.mean_eq_diameter_nm,
            "std_eq_diameter_nm": self.std_eq_diameter_nm,
            "median_eq_diameter_nm": self.median_eq_diameter_nm,
            "mean_area_nm2": self.mean_area_nm2,
            "total_grain_area_nm2": self.total_grain_area_nm2,
            "coverage_fraction": self.coverage_fraction,
            "lognormal_mu": self.lognormal_mu,
            "lognormal_sigma": self.lognormal_sigma,
        }


@dataclass
class BatchResult:
    """Result of aggregating several per-image reports."""
    n_reports: int
    per_image: list[PerImageRow]
    pooled_grains: list[Grain]
    pooled_stats: ImageGrainStats
    total_layer_area_nm2: float
    total_grain_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reports": self.n_reports,
            "per_image": [r.to_dict() for r in self.per_image],
            "pooled_stats": self.pooled_stats.to_dict(),
            "total_layer_area_nm2": self.total_layer_area_nm2,
            "total_grain_count": self.total_grain_count,
        }


# === Aggregation ===

def aggregate_reports(reports: list[dict[str, Any]]) -> BatchResult:
    """Aggregate a list of per-image report dicts.

    Parameters
    ----------
    reports : list[dict]
        Each dict is the parsed JSON of a per-image report (see `report.py`).

    Returns
    -------
    BatchResult
    """
    per_image: list[PerImageRow] = []
    pooled_grains: list[Grain] = []
    total_layer_area = 0.0
    # Use a running label counter so labels are unique across images.
    label_counter = 0

    for rep in reports:
        image_info = rep.get("image", {})
        layer_info = rep.get("layer", {})
        stats_info = rep.get("statistics", {})

        # Per-image row.
        eq_diam_stats = stats_info.get("eq_diameter_stats", {})
        area_stats = stats_info.get("area_stats", {})
        lf = stats_info.get("lognormal_fit")
        row = PerImageRow(
            file_name=image_info.get("file_name", "?"),
            pixel_size_nm=float(image_info.get("pixel_size_nm", 0.0)),
            magnification=image_info.get("magnification"),
            layer_source=layer_info.get("source"),
            layer_area_nm2=float(layer_info.get("area_nm2", 0.0)),
            layer_height_nm=float(layer_info.get("height_nm", 0.0)),
            layer_width_nm=float(layer_info.get("width_nm", 0.0)),
            n_grains=int(stats_info.get("n_grains", 0)),
            mean_eq_diameter_nm=float(eq_diam_stats.get("mean", 0.0)),
            std_eq_diameter_nm=float(eq_diam_stats.get("std", 0.0)),
            median_eq_diameter_nm=float(eq_diam_stats.get("median", 0.0)),
            mean_area_nm2=float(area_stats.get("mean", 0.0)),
            total_grain_area_nm2=float(stats_info.get("total_grain_area_nm2", 0.0)),
            coverage_fraction=float(stats_info.get("coverage_fraction", 0.0)),
            lognormal_mu=float(lf["mu"]) if lf else None,
            lognormal_sigma=float(lf["sigma"]) if lf else None,
        )
        per_image.append(row)

        # Pool grains.
        for g_dict in rep.get("grains", []):
            g = _grain_from_dict(g_dict)
            label_counter += 1
            g.label = label_counter  # ensure unique label
            pooled_grains.append(g)

        # Accumulate total layer area (use area_nm2 if present, else height*width).
        layer_area = float(layer_info.get("area_nm2", 0.0))
        if layer_area <= 0:
            layer_area = float(layer_info.get("height_nm", 0.0)) * float(layer_info.get("width_nm", 0.0))
        total_layer_area += layer_area

    # Compute pooled stats.
    # We need a "virtual" layer height/width for `compute_image_stats` to
    # compute coverage_fraction. Use a 1×total_layer_area virtual layer.
    pooled_stats = compute_image_stats(
        pooled_grains,
        layer_height_nm=1.0,
        layer_width_nm=total_layer_area if total_layer_area > 0 else 1.0,
    )

    return BatchResult(
        n_reports=len(reports),
        per_image=per_image,
        pooled_grains=pooled_grains,
        pooled_stats=pooled_stats,
        total_layer_area_nm2=total_layer_area,
        total_grain_count=len(pooled_grains),
    )


# === Pooled lognormal fit (convenience) ===

def pooled_eq_diameters_nm(batch: BatchResult) -> np.ndarray:
    """Return all pooled equivalent diameters in nm."""
    return np.array([g.equivalent_diameter_nm for g in batch.pooled_grains],
                    dtype=np.float64)


def pooled_areas_nm2(batch: BatchResult) -> np.ndarray:
    """Return all pooled grain areas in nm²."""
    return np.array([g.area_nm2 for g in batch.pooled_grains], dtype=np.float64)
