"""
visualization.py — Visual outputs for the Grain Analyzer app.

Functions:
* `color_label_image`  — distinct colour per grain on a black or original background.
* `boundary_overlay`   — original band with red grain-boundary contours.
* `plot_row_profile`   — row-mean profile with info-strip + suggested bands.
* `plot_histogram`     — histogram with optional lognormal fit curve.
* `figure_to_png_bytes` — encode a matplotlib figure as PNG bytes.
"""
from __future__ import annotations

import io
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from skimage import color as skcolor
from skimage import measure, segmentation as skseg

from .layer_detect import Band, InfoStrip
from .measurements import LognormalFit, lognormal_pdf


# === Color label image ===

def color_label_image(labels: np.ndarray, background: Optional[np.ndarray] = None,
                      alpha: float = 0.7, bg_brightness: float = 0.4) -> np.ndarray:
    """Render a labeled image with a distinct colour per grain.

    Parameters
    ----------
    labels : np.ndarray (int)
        Label image (0 = background).
    background : np.ndarray, optional
        Original grayscale band image to use as the background (improves
        context). If None, a black background is used.
    alpha : float
        Opacity of the colour overlay (0 = pure background, 1 = pure colour).
    bg_brightness : float
        Multiplier applied to the background (only used when `background` is
        provided). Lower values darken the background so colours pop.

    Returns
    -------
    np.ndarray (uint8, shape (H, W, 3))
        RGB image.
    """
    # Use a perceptually-uniform colormap with many distinct colours.
    n_labels = int(labels.max()) + 1
    colored = skcolor.label2rgb(
        labels, bg_label=0, kind="overlay",
        colors=None, alpha=1.0,
        bg_color=(0, 0, 0),
    )
    # `label2rgb` returns float in [0, 1].
    colored = (colored * 255).astype(np.uint8)

    if background is not None:
        bg = background.astype(np.float64) * bg_brightness
        bg_rgb = np.stack([bg, bg, bg], axis=-1)
        # Apply overlay only where labels > 0.
        mask = (labels > 0)[..., None]
        out = np.where(mask, colored.astype(np.float64) * alpha + bg_rgb * (1 - alpha), bg_rgb)
        return np.clip(out, 0, 255).astype(np.uint8)
    return colored


# === Boundary overlay ===

def boundary_overlay(image: np.ndarray, labels: np.ndarray,
                     boundary_color: tuple[int, int, int] = (255, 0, 0),
                     boundary_thickness: int = 1) -> np.ndarray:
    """Draw grain boundaries on the original image.

    Parameters
    ----------
    image : np.ndarray (uint8, 2-D)
        Original grayscale band.
    labels : np.ndarray (int)
        Label image (0 = background).
    boundary_color : (R, G, B)
        Colour for the boundary contours.
    boundary_thickness : int
        Thickness in pixels.

    Returns
    -------
    np.ndarray (uint8, shape (H, W, 3))
    """
    # Find boundaries between regions.
    boundaries = skseg.find_boundaries(labels, mode="outer", background=0)
    # Convert grayscale to RGB.
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    out[boundaries] = boundary_color
    # Also draw region contours slightly thicker for visibility.
    if boundary_thickness > 1:
        contours = []
        for region in measure.regionprops(labels):
            contours.append(region.coords)
        # Dilate boundaries.
        kernel = np.ones((boundary_thickness, boundary_thickness), np.uint8)
        bnd_dilated = cv2.dilate(boundaries.astype(np.uint8), kernel, iterations=1)
        out[bnd_dilated > 0] = boundary_color
    return out


# === Row profile plot ===

def plot_row_profile(profile: np.ndarray, image_height_px: int,
                     info_strip: Optional[InfoStrip] = None,
                     suggested_bands: Optional[list[Band]] = None,
                     selected_band: Optional[Band] = None,
                     pixel_size_nm: float = 1.0) -> Figure:
    """Plot the row-mean intensity profile with overlays.

    Parameters
    ----------
    profile : np.ndarray
        Smoothed row-mean intensity (length H).
    image_height_px : int
    info_strip : InfoStrip, optional
        Detected info-strip regions, shaded grey.
    suggested_bands : list[Band], optional
        Suggested layer candidates, shaded light blue.
    selected_band : Band, optional
        Currently selected layer, outlined in red.
    pixel_size_nm : float
        Used for the secondary Y axis (height in nm).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    h = image_height_px
    y_px = np.arange(h)
    ax.plot(profile, y_px, color="black", linewidth=0.8, label="row-mean intensity")
    ax.set_xlabel("Mean intensity (0–255)")
    ax.set_ylabel("Row (px)")
    ax.set_ylim(h, 0)  # invert Y so row 0 is at top
    ax.grid(True, alpha=0.3)

    # Shade info-strips.
    if info_strip is not None:
        if info_strip.top is not None:
            ax.axhspan(info_strip.top[0], info_strip.top[1], color="grey", alpha=0.3,
                       label="info strip")
        if info_strip.bottom is not None:
            ax.axhspan(info_strip.bottom[0], info_strip.bottom[1], color="grey", alpha=0.3)
    # Shade suggested bands.
    if suggested_bands:
        for i, band in enumerate(suggested_bands):
            ax.axhspan(band.y0, band.y1, color="lightblue", alpha=0.3,
                       label="suggested band" if i == 0 else None)
    # Outline selected band.
    if selected_band is not None:
        ax.axhspan(selected_band.y0, selected_band.y1, facecolor="none",
                   edgecolor="red", linewidth=2, label="selected band")

    ax.legend(loc="best", fontsize=8)

    # Secondary Y axis (nm).
    secax = ax.twinx()
    secax.set_ylim(h * pixel_size_nm, 0)
    secax.set_ylabel("Height (nm)")
    return fig


# === Histogram with lognormal fit ===

def plot_histogram(values: np.ndarray, unit: str, title: str,
                   bins: int = 20, color: str = "steelblue",
                   lognormal_fit: Optional[LognormalFit] = None,
                   show_fit: bool = True,
                   xlabel: Optional[str] = None,
                   ylabel: str = "Count") -> Figure:
    """Plot a histogram with optional lognormal fit overlay.

    Parameters
    ----------
    values : array-like of float
        Values to histogram.
    unit : str
        Unit string for the X-axis label (e.g. "nm").
    title : str
    bins : int
    color : str
    lognormal_fit : LognormalFit, optional
    show_fit : bool
        If True and `lognormal_fit` is provided, draws the lognormal PDF
        scaled to the histogram.
    xlabel, ylabel : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = arr.size
    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return fig

    counts, bin_edges, patches = ax.hist(
        arr, bins=bins, color=color, edgecolor="white", alpha=0.8,
        label=f"data (n={n})",
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f"Value ({unit})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if show_fit and lognormal_fit is not None and lognormal_fit.success:
        # Plot the lognormal PDF scaled to the histogram.
        x = np.linspace(max(arr.min(), 1e-6), arr.max() * 1.05, 300)
        pdf = lognormal_pdf(x, lognormal_fit)
        bin_width = float(bin_edges[1] - bin_edges[0])
        scaled_pdf = pdf * n * bin_width
        ax.plot(x, scaled_pdf, color="red", linewidth=2,
                label=f"lognormal fit\nμ={lognormal_fit.mu:.3f}, σ={lognormal_fit.sigma:.3f}")
        ax.legend(loc="best", fontsize=8)
    elif lognormal_fit is None and n >= 5:
        # Don't draw fit but show the data label.
        ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

    return fig


# === Figure → PNG bytes ===

def figure_to_png_bytes(fig: Figure) -> bytes:
    """Encode a matplotlib figure as PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def figure_to_png_b64(fig: Figure) -> str:
    """Encode a matplotlib figure as base64-encoded PNG (for JSON embedding)."""
    import base64
    return base64.b64encode(figure_to_png_bytes(fig)).decode("ascii")


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 image as PNG bytes."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def image_to_png_b64(image_rgb: np.ndarray) -> str:
    """Encode an RGB uint8 image as base64-encoded PNG (for JSON embedding)."""
    import base64
    return base64.b64encode(image_to_png_bytes(image_rgb)).decode("ascii")
