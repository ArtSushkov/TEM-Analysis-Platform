"""
report.py — Build a self-contained JSON report for one image analysis.

The report is the **single artefact** produced by the Single Image tab and
the **single input** consumed by the Batch tab. To make the JSON fully
self-contained (no separate image files to keep track of), the colour-labeled
grain map is embedded as base64-encoded PNG.

Report schema (v1.1)
--------------------
```json
{
  "schema_version": "1.1",
  "app": "02_grain_analyzer",
  "generated_at": "2025-10-31T12:34:56",
  "image": {
    "file_name": "TEM [DATE]28005 [1].TIF",
    "width_px": 4008,
    "height_px": 2664,
    "pixel_size_nm": 0.0231,
    "pixel_size_source": "OlympusSIS",
    "magnification": 600000.0,
    "acquisition_datetime": "2025-12-23T19:25:00",
    "camera_name": "Quemesa",
    "software": "analySIS 5.0"
  },
  "layer": {
    "source": "sam",
    "polygon": [[y0,x0], [y1,x1], ...],
    "bbox": [min_row, min_col, max_row, max_col],
    "area_nm2": 4567.8,
    "height_nm": 7.43,
    "width_nm": 92.58,
    "mean_intensity": 104.1,
    "std_intensity": 23.4,
    "pixel_size_nm": 0.0231
  },
  "segmentation_params": {...},
  "statistics": {
    "n_grains": 49,
    "total_grain_area_nm2": 4567.8,
    "layer_area_nm2": 688.4,
    "coverage_fraction": 6.63,
    "area_stats": {...},
    "eq_diameter_stats": {...},
    "lognormal_fit": {"mu": 2.29, "sigma": 0.46, ...}
  },
  "grains": [
    {"label": 1, "area_nm2": 29.5, "equivalent_diameter_nm": 6.13, ...},
    ...
  ],
  "visualization_png_base64": "iVBORw0KGgo..."
}
```
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np

from .io_utils import ImageMetadata
from .measurements import Grain, ImageGrainStats
from .polygon import LayerMask
from .segmentation import SegmentationParams
from .visualization import image_to_png_b64


SCHEMA_VERSION = "1.1"
APP_NAME = "02_grain_analyzer"


def build_report(
    image_metadata: ImageMetadata,
    layer_mask: LayerMask,
    segmentation_params: SegmentationParams,
    stats: ImageGrainStats,
    grains: list[Grain],
    labeled_image_rgb: np.ndarray,
) -> dict[str, Any]:
    """Build a self-contained report dict.

    Parameters
    ----------
    image_metadata : ImageMetadata
        From `io_utils.load_tiff`.
    layer_mask : LayerMask
        The selected layer (from SAM, polygon, or manual rows).
    segmentation_params : SegmentationParams
        Parameters used for segmentation.
    stats : ImageGrainStats
        From `measurements.compute_image_stats`.
    grains : list[Grain]
        From `measurements.measure_grains`.
    labeled_image_rgb : np.ndarray (H, W, 3) uint8
        Colour-labeled grain image (will be embedded as base64 PNG).

    Returns
    -------
    dict
        JSON-serializable report.
    """
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "app": APP_NAME,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "image": {
            "file_name": image_metadata.file_name,
            "width_px": image_metadata.width_px,
            "height_px": image_metadata.height_px,
            "dtype": image_metadata.dtype,
            "pixel_size_nm": image_metadata.pixel_size_nm,
            "pixel_size_source": image_metadata.pixel_size_source,
            "magnification": image_metadata.magnification,
            "acquisition_datetime": image_metadata.acquisition_datetime,
            "camera_name": image_metadata.camera_name,
            "picture_type": image_metadata.picture_type,
            "software": image_metadata.software,
            "make": image_metadata.make,
        },
        "layer": layer_mask.to_dict(image_metadata.pixel_size_nm),
        "segmentation_params": segmentation_params.to_dict(),
        "statistics": stats.to_dict(),
        "grains": [g.to_dict() for g in grains],
        "visualization_png_base64": image_to_png_b64(labeled_image_rgb),
    }
    return report


def serialize_report(report: dict[str, Any]) -> bytes:
    """Serialize a report dict to UTF-8 JSON bytes (indented)."""
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def suggested_report_filename(image_file_name: str) -> str:
    """Suggested filename for a downloaded report.

    Strips the extension from the image filename and appends `_grain_report.json`.
    """
    base = image_file_name.rsplit(".", 1)[0] if "." in image_file_name else image_file_name
    # Sanitize: replace problematic characters.
    base = base.replace(" ", "_").replace("[", "").replace("]", "")
    return f"{base}_grain_report.json"
