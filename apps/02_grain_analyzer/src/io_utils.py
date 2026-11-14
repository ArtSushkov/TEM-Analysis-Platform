"""
io_utils.py — TIFF loading and metadata extraction for TEM cross-section images.

The pixel size is read from the private OlympusSIS TIFF tag (code 33560),
which `tifffile` parses into a dict with `pixelsizex`/`pixelsizey` in metres.
The standard XResolution/YResolution tags are unreliable for these images
(they carry a dummy 200/1 value).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile


# === Data classes ===

@dataclass
class ImageMetadata:
    """Metadata extracted from a TIFF file."""
    file_name: str
    file_path: str
    width_px: int
    height_px: int
    dtype: str
    # Pixel size in nanometres (the working unit for this app).
    pixel_size_nm: float
    # Source of the pixel size value.
    pixel_size_source: str  # "OlympusSIS" | "XResolution" | "manual" | "unknown"
    # Optional OlympusSIS fields (None when absent).
    magnification: Optional[float] = None
    acquisition_datetime: Optional[str] = None
    camera_name: Optional[str] = None
    picture_type: Optional[str] = None
    software: Optional[str] = None
    make: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedImage:
    """A loaded TEM image with its metadata."""
    image: np.ndarray  # 2-D uint8 grayscale
    metadata: ImageMetadata

    @property
    def pixel_size_nm(self) -> float:
        return self.metadata.pixel_size_nm


# === Loading ===

# Private TIFF tag used by Olympus Soft Imaging Solutions (analySIS software).
# tifffile knows about it and parses it into a dict.
_OLYMPUS_SIS_TAG_CODE = 33560


def _extract_olympus_sis(tif_page) -> Optional[dict[str, Any]]:
    """Return the parsed OlympusSIS dict from a TIFF page, or None if absent."""
    tag = tif_page.tags.get(_OLYMPUS_SIS_TAG_CODE)
    if tag is None:
        return None
    val = tag.value
    # tifffile returns a dict for this tag; older versions may return a tuple/struct.
    if isinstance(val, dict):
        return val
    return None


def _parse_resolution_tag(value) -> Optional[float]:
    """Parse an XResolution/YResolution tag value (numerator, denominator) → pixels per unit.

    Returns None if the value is invalid or zero.
    """
    try:
        num, den = value
        if den == 0:
            return None
        return float(num) / float(den)
    except (TypeError, ValueError):
        return None


def load_tiff(path: str | Path, manual_pixel_size_nm: Optional[float] = None) -> LoadedImage:
    """Load a TIFF file and extract metadata.

    Pixel-size resolution order:
      1. Manual override (if `manual_pixel_size_nm` is provided)
      2. OlympusSIS private tag (33560) — `pixelsizex` in metres
      3. XResolution tag — interpreted as pixels per centimetre (TIFF ResolutionUnit)
      4. Fallback: 1.0 nm/px with `pixel_size_source = "unknown"`

    Parameters
    ----------
    path : str | Path
        Path to the TIFF file.
    manual_pixel_size_nm : float, optional
        Manual pixel size in nanometres per pixel. If provided, overrides metadata.

    Returns
    -------
    LoadedImage
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read the entire file into memory first and parse from a BytesIO buffer.
    # This avoids Windows file-locking issues: tifffile may keep the file
    # handle open (via memory mapping) even after the TiffFile context manager
    # exits, which prevents the caller from deleting the temp file.
    data = path.read_bytes()
    bio = io.BytesIO(data)

    with tifffile.TiffFile(bio) as tif:
        page = tif.pages[0]
        # Force a contiguous in-memory copy so the returned array does not
        # reference the BytesIO buffer (which would be freed on exit).
        image = np.ascontiguousarray(page.asarray())

        # Some TEM images are stored as multi-channel; reduce to grayscale if needed.
        if image.ndim == 3:
            # Take the first channel or convert RGB → luminance.
            if image.shape[-1] == 3:
                image = (0.299 * image[..., 0] +
                         0.587 * image[..., 1] +
                         0.114 * image[..., 2]).astype(np.uint8)
            else:
                image = image[..., 0]

        if image.dtype != np.uint8:
            # Normalise to uint8 for downstream processing.
            img_f = image.astype(np.float64)
            lo, hi = np.percentile(img_f, [1.0, 99.0])
            if hi > lo:
                img_f = np.clip((img_f - lo) * 255.0 / (hi - lo), 0, 255)
            image = img_f.astype(np.uint8)

        # OlympusSIS metadata.
        sis = _extract_olympus_sis(page)
        magnification = None
        acquisition_datetime = None
        camera_name = None
        picture_type = None
        pixel_size_nm: Optional[float] = None
        pixel_size_source = "unknown"

        if sis is not None:
            px = sis.get("pixelsizex")
            if px is not None and px > 0:
                # Stored in metres → convert to nm.
                pixel_size_nm = float(px) * 1e9
                pixel_size_source = "OlympusSIS"
            magnification = sis.get("magnification")
            dt = sis.get("datetime")
            acquisition_datetime = dt.isoformat() if hasattr(dt, "isoformat") else str(dt) if dt else None
            camera_name = sis.get("cameraname")
            picture_type = sis.get("picturetype")

        # Fallback to XResolution tag (rarely meaningful for these images, but check).
        if pixel_size_nm is None:
            xres = page.tags.get("XResolution")
            unit = page.tags.get("ResolutionUnit")
            if xres is not None and unit is not None:
                ppu = _parse_resolution_tag(xres.value)  # pixels per ResolutionUnit
                unit_val = unit.value
                # ResolutionUnit: 1 = none, 2 = inch, 3 = centimetre.
                if ppu is not None and ppu > 0 and unit_val in (2, 3):
                    metres_per_unit = 0.0254 if unit_val == 2 else 0.01
                    pixel_size_nm = (metres_per_unit / ppu) * 1e9
                    pixel_size_source = "XResolution"

        # Manual override.
        if manual_pixel_size_nm is not None and manual_pixel_size_nm > 0:
            pixel_size_nm = float(manual_pixel_size_nm)
            pixel_size_source = "manual"

        # Final fallback.
        if pixel_size_nm is None or pixel_size_nm <= 0:
            pixel_size_nm = 1.0
            pixel_size_source = "unknown"

        software = page.tags.get("Software")
        software_val = software.value if software is not None else None
        if isinstance(software_val, bytes):
            software_val = software_val.decode("ascii", errors="replace")

        make = page.tags.get("Make")
        make_val = make.value if make is not None else None
        if isinstance(make_val, bytes):
            make_val = make_val.decode("ascii", errors="replace")

        meta = ImageMetadata(
            file_name=path.name,
            file_path=str(path),
            width_px=int(image.shape[1]),
            height_px=int(image.shape[0]),
            dtype=str(image.dtype),
            pixel_size_nm=float(pixel_size_nm),
            pixel_size_source=pixel_size_source,
            magnification=float(magnification) if magnification is not None else None,
            acquisition_datetime=acquisition_datetime,
            camera_name=camera_name,
            picture_type=picture_type,
            software=software_val,
            make=make_val,
        )

    return LoadedImage(image=image, metadata=meta)


# === Report I/O (used by the Batch tab) ===

def save_report_json(report: dict[str, Any], path: str | Path) -> None:
    """Write a report dict to a JSON file (UTF-8, indented)."""
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_report_json(path: str | Path) -> dict[str, Any]:
    """Load a report dict from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_report_from_bytes(data: bytes) -> dict[str, Any]:
    """Load a report dict from raw JSON bytes."""
    return json.loads(data.decode("utf-8"))
