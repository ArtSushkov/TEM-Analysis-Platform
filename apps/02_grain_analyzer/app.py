"""
app.py — Streamlit UI for TEM cross-section grain-size analysis.

Three tabs:
1. **Single Image** — upload one TIFF, select a layer (SAM / polygon / manual rows),
   analyse grains, download a self-contained JSON report.
2. **Batch Statistics** — drop several JSON reports (or a ZIP), get pooled
   grain-size statistics for the same layer across images.
3. **Help** — usage guide, method, parameter reference, troubleshooting.
"""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

# Make `src` importable.
sys.path.insert(0, str(Path(__file__).parent))
from src import batch as batch_mod
from src import click_segmentation
from src import fft_segmentation
from src import io_utils
from src import measurements
from src import polygon as polygon_mod
from src import report
from src import sam_segmenter
from src import segmentation
from src import structure_tensor_segmentation
from src import visualization


# === Helper: wrap a dict as a params object with to_dict() ===

class _DictParams:
    """Wrap a plain dict so it has a to_dict() method (for report.build_report)."""
    def __init__(self, d):
        self._d = d
    def to_dict(self):
        return self._d


# === Page configuration ===

st.set_page_config(
    page_title="TEM Grain Analyzer",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === Session state defaults ===

def _init_state() -> None:
    defaults = {
        # Single Image tab.
        "loaded_image": None,             # io_utils.LoadedImage
        "loaded_image_bytes": None,
        "loaded_image_name": None,
        "manual_pixel_size_nm": None,
        # Layer selection.
        "layer_mode": "SAM (click on layer)",  # see options below
        "sam_auto_result": None,          # sam_segmenter.SAMResult (automatic mode)
        "sam_auto_selected_idx": None,
        "sam_click_result": None,         # sam_segmenter.SAMResult (point-prompt mode)
        "sam_click_last_click": None,     # (row, col) of last processed click (dedup)
        "sam_click_pending": None,        # (row, col, max_dim) — click queued for SAM processing
        "sam_predictor": None,            # cached sam_segmenter.Segmenter instance
        "polygon_vertices": None,         # list of (row, col) tuples (selector-preview coords)
        "polygon_last_click": None,       # (row, col) of last processed click (dedup)
        "manual_y0": 0,
        "manual_y1": 0,
        "selected_layer_mask": None,      # polygon_mod.LayerMask
        # Analysis.
        "analysis_result": None,
        "analysis_grains": None,
        "analysis_stats": None,
        "analysis_report": None,
        "analysis_labeled_image": None,
        "analysis_overlay_image": None,
        "fft_result": None,               # fft_segmentation.FFTSegmentationResult (for FFT viz)
        "st_result": None,                # structure_tensor_segmentation.StructureTensorResult
        "fft_cached": None,               # (fft_shifted, log_power, scale, img_small_shape) — cached FFT for manual refl
        "manual_reflections": None,       # list of (row, col) in small-FFT coords
        "manual_refl_last_click": None,   # dedup
        "manual_refl_mode": False,        # whether user is in manual reflection selection mode
        "manual_refl_preview": None,      # (overlay_rgb, d_nm, radius_px) of the LAST click
        "manual_refl_last_radius": None,  # last click radius (px) — for the radial spectrum plot
        "manual_refl_quality": None,      # list of ReflectionQuality, one per user-click (not symmetry)
        "manual_refl_last_quality": None, # ReflectionQuality of the LAST click (for the preview column)
        # Click-to-segment mode.
        "click_grains": None,             # list of dicts: {mask, signature, click_point, area, d_spacings, label}
        "click_grain_labels_img": None,   # (H, W) int32 labeled image (one label per grain)
        "click_last_result": None,        # GrainClickResult of the last click (for the preview)
        "click_last_click_coord": None,   # dedup
        "click_global_fft": None,         # cached global FFT for amplitude maps
        "click_sliding_data": None,       # cached sliding local FFTs (mode B)
        "click_sliding_params": None,     # (window_size, step, zero_pad) used for the cache
        "click_pending_similar": None,    # list of similar grain masks awaiting user confirmation
        "click_finished": False,          # True after user clicks "Finish"
        # Manual reflection selection (within click mode).
        "click_manual_reflections": None,     # list of (r, c) in local-FFT coords
        "click_manual_refl_last_click": None, # dedup
        "click_manual_mode": False,           # whether user is selecting reflections manually
        "click_manual_refl_img": None,        # the displayed power spectrum image (for click coords)
        # Manual tracing (polygon) mode.
        "mt_grains": None,              # list of dicts: {mask, polygon, area_px, source}
        "mt_current_vertices": None,    # list of (r, c) for the polygon being drawn
        "mt_last_click": None,          # dedup
        "mt_finished": False,           # True after Finish
        # Batch tab.
        "batch_reports": None,
        "batch_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# === Helpers ===

def _to_rgb_for_streamlit(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _downscale_for_preview(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _draw_polygon_on_image(image_rgb: np.ndarray, polygon: np.ndarray,
                            color: tuple[int, int, int] = (255, 0, 0),
                            thickness: int = 2) -> np.ndarray:
    """Draw a polygon outline on an RGB image. Polygon is (N, 2) of (row, col)."""
    if polygon is None or len(polygon) < 2:
        return image_rgb
    out = image_rgb.copy()
    pts = polygon[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def _draw_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray,
                         color: tuple[int, int, int] = (0, 255, 0),
                         alpha: float = 0.3) -> np.ndarray:
    """Overlay a boolean mask on an RGB image with the given colour and alpha."""
    if mask is None or not mask.any():
        return image_rgb
    out = image_rgb.copy()
    overlay = out.copy()
    overlay[mask] = color
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out


# === Single Image tab ===

def render_single_image_tab() -> None:
    st.title("TEM Grain Analyzer")
    st.caption(
        "Upload a TEM cross-section TIFF, select a layer (SAM auto / polygon / manual rows), "
        "and characterise the grain-size distribution. Pixel size is read "
        "automatically from the OlympusSIS TIFF tag."
    )

    # === Sidebar: upload + settings ===
    st.sidebar.header("Single Image")

    with st.sidebar.expander("Upload", expanded=True):
        uploaded = st.file_uploader(
            "Select TIFF image",
            type=["tif", "tiff"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

    # === Load image when uploaded ===
    if uploaded is not None:
        if (st.session_state.loaded_image_name != uploaded.name
                or st.session_state.loaded_image is None):
            try:
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                    f.write(uploaded.getvalue())
                    tmp_path = Path(f.name)
                li = io_utils.load_tiff(tmp_path)
                st.session_state.loaded_image = li
                st.session_state.loaded_image_bytes = uploaded.getvalue()
                st.session_state.loaded_image_name = uploaded.name
                st.session_state.manual_pixel_size_nm = None
                # Reset downstream state.
                st.session_state.sam_auto_result = None
                st.session_state.sam_auto_selected_idx = None
                st.session_state.sam_click_result = None
                st.session_state.sam_click_last_click = None
                st.session_state.sam_click_pending = None
                st.session_state.sam_predictor = None  # force re-encode for new image
                st.session_state.polygon_vertices = None
                st.session_state.polygon_last_click = None
                st.session_state.selected_layer_mask = None
                st.session_state.analysis_result = None
                st.session_state.analysis_report = None
            except Exception as e:
                st.error(f"Failed to load TIFF: {e}")
                return
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    pass

    li: io_utils.LoadedImage | None = st.session_state.loaded_image

    if li is None:
        st.info("Upload a TIFF image in the sidebar to begin.")
        st.markdown("### How it works")
        st.markdown(
            "1. **Pixel size** is auto-detected from the OlympusSIS TIFF tag (33560)."
            "\n"
            "2. **Layer selection** — three modes:"
            "\n   • **SAM (auto)** — Segment Anything Model finds candidate layers automatically."
            "\n   • **Polygon (draw)** — you draw a polygon around the layer."
            "\n   • **Manual rows** — pick a row range (legacy, for horizontal layers)."
            "\n"
            "3. **Grain segmentation** — bilateral filter → CLAHE → Scharr gradient → "
            "Otsu threshold → marker-controlled watershed, restricted to the selected layer."
            "\n"
            "4. **Output** — colour-labeled grain map, boundary overlay, statistics "
            "table, histograms with lognormal fit, and a self-contained JSON report."
        )
        st.markdown("See the **Help** tab for the full guide.")
        return

    # === Metadata panel ===
    md = li.metadata
    with st.sidebar.expander("Image Metadata", expanded=True):
        st.write(f"**File:** {md.file_name}")
        st.write(f"**Size:** {md.width_px} × {md.height_px} px")
        st.write(f"**Pixel size:** {md.pixel_size_nm:.4f} nm/px "
                 f"({md.pixel_size_nm*1000:.2f} pm/px)")
        st.write(f"**Source:** {md.pixel_size_source}")
        if md.magnification:
            st.write(f"**Magnification:** {md.magnification:g}×")
        if md.acquisition_datetime:
            st.write(f"**Acquired:** {md.acquisition_datetime}")
        if md.camera_name:
            st.write(f"**Camera:** {md.camera_name}")
        if md.software:
            st.write(f"**Software:** {md.software}")
        with st.expander("Override pixel size", expanded=False):
            new_ps = st.number_input(
                "Pixel size (nm/px)",
                min_value=0.0001, max_value=100.0,
                value=float(md.pixel_size_nm),
                format="%.4f",
                key="manual_ps_input",
            )
            if st.button("Apply override", key="apply_ps"):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                        f.write(st.session_state.loaded_image_bytes)
                        tmp_path = Path(f.name)
                    li2 = io_utils.load_tiff(tmp_path, manual_pixel_size_nm=new_ps)
                    st.session_state.loaded_image = li2
                    st.session_state.manual_pixel_size_nm = new_ps
                    li = li2
                    md = li2.metadata
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except (PermissionError, OSError):
                        pass
                st.rerun()

    ps = li.pixel_size_nm

    # === Layer selection ===
    with st.sidebar.expander("Layer Selection", expanded=True):
        layer_mode = st.radio(
            "Selection mode",
            ["SAM (click on layer)", "SAM (auto)", "Polygon (click vertices)", "Manual rows"],
            index=0,
            key="layer_mode_radio",
            help=(
                "SAM (click on layer) = click on the layer you want; SAM segments it (recommended). "
                "SAM (auto) = SAM finds all candidate layers automatically. "
                "Polygon (click vertices) = click polygon vertices on the image. "
                "Manual rows = pick a row range (legacy, for horizontal layers)."
            ),
        )
        st.session_state.layer_mode = layer_mode

        if layer_mode == "SAM (click on layer)":
            _render_sam_click_selector(li)
        elif layer_mode == "SAM (auto)":
            _render_sam_auto_selector(li)
        elif layer_mode == "Polygon (click vertices)":
            _render_polygon_click_selector(li)
        else:  # Manual rows
            _render_manual_rows_selector(li)

    # === Segmentation method + parameters ===
    # When the user switches segmentation method, clear ALL session state
    # from the previous method so stale UI elements don't linger.
    def _on_seg_method_change() -> None:
        # Clear analysis results (shared by all methods).
        st.session_state.analysis_result = None
        st.session_state.analysis_grains = None
        st.session_state.analysis_stats = None
        st.session_state.analysis_report = None
        st.session_state.analysis_labeled_image = None
        st.session_state.analysis_overlay_image = None
        st.session_state.fft_result = None
        st.session_state.st_result = None
        # Clear manual tracing state.
        st.session_state.mt_grains = None
        st.session_state.mt_current_vertices = None
        st.session_state.mt_last_click = None
        st.session_state.mt_finished = False
        # Clear click-to-segment state.
        st.session_state.click_grains = None
        st.session_state.click_last_result = None
        st.session_state.click_last_click_coord = None
        st.session_state.click_pending_similar = None
        st.session_state.click_finished = False
        st.session_state.click_manual_reflections = None
        st.session_state.click_manual_refl_last_click = None
        st.session_state.click_global_fft = None
        st.session_state.click_sliding_data = None
        st.session_state.click_sliding_params = None

    seg_method = st.sidebar.selectbox(
        "Grain segmentation method",
        ["Manual tracing (polygon)", "Structure tensor (auto-assist)",
         "Click-to-segment grains (FFT)", "FFT (Grachev)", "Classical watershed"],
        index=0,
        key="seg_method_select",
        on_change=_on_seg_method_change,
        help=("Manual tracing = draw grain outlines by hand (ISO 13067 "
              "compliant, most reliable). "
              "Structure tensor = auto-detect grains by local lattice-fringe "
              "orientation change (fast, good for grains with visible fringes). "
              "Click-to-segment (FFT) = click on each grain, local FFT + "
              "sliding window amplitude maps. "
              "FFT (Grachev) = global FFT + DBScan. "
              "Classical watershed = gradient-based."),
    )
    st.session_state.seg_method = seg_method

    with st.sidebar.expander("Advanced (Segmentation)", expanded=False):
        if seg_method == "Manual tracing (polygon)":
            st.caption("No parameters needed — just draw grain outlines "
                       "with the mouse. ISO 13067 compliant.")
            params = None
        elif seg_method == "Click-to-segment grains (FFT)":
            click_params_obj = click_segmentation.ClickSegmentationParams()
            click_params_obj.window_size_px = st.slider(
                "Local FFT window (px)", 32, 256, 128, 8,
                key="click_winsize",
                help="Side of the square window around each click. "
                     "Larger = better FFT SNR but may cross grain boundaries. "
                     "Smaller = more localized but noisier FFT. "
                     "128 = good default for medium grains.",
            )
            click_params_obj.zero_pad_size_px = st.slider(
                "Zero-pad size (px)", 0, 512, 256, 32,
                key="click_padsize",
                help="Pad the window to this size before FFT (improves "
                     "frequency resolution without increasing window size). "
                     "0 = no padding.",
            )
            click_params_obj.local_peak_height_factor = st.slider(
                "Local peak height factor", 0.10, 0.80, 0.30, 0.05,
                key="click_pkheight",
                help="Min relative intensity (frac of local 1D spectrum max) "
                     "for a peak to qualify as a d-spacing of this grain. "
                     "Higher = fewer d-spacings (stricter).",
            )
            click_params_obj.max_d_spacings_per_grain = st.slider(
                "Max d-spacings per grain", 1, 8, 5, 1,
                key="click_maxd",
                help="Keep the top-N d-spacings by SNR. 1 = only the strongest "
                     "(relaxes the AND condition). 5 = up to 5 plane families.",
            )
            click_params_obj.seed_threshold_frac = st.slider(
                "Seed threshold (frac of max)", 0.05, 0.80, 0.15, 0.05,
                key="click_seedthr",
                help="High threshold for the click point. The click must be on "
                     "a pixel whose min-amplitude across all d-spacings exceeds "
                     "this. If not, the click is rejected as 'off-grain'. "
                     "With strict AND (min across d-spacings) + orientation-specific "
                     "disk masks, the seed amplitude can be lower than expected "
                     "(one d-spacing may be noisy). 0.15 is forgiving.",
            )
            click_params_obj.growth_threshold_frac = st.slider(
                "Growth threshold (frac of max)", 0.02, 0.50, 0.10, 0.02,
                key="click_growththr",
                help="Low threshold for region growing. A neighbor pixel is "
                     "added to the grain if its min-amplitude exceeds this AND "
                     "it is adjacent to an accepted pixel. Lower than seed threshold. "
                     "With orientation-specific (disk) masks, can be low because "
                     "amplitude naturally drops to ~0 outside the grain.",
            )
            click_params_obj.use_sliding_local_fft = st.checkbox(
                "Sliding local FFT (RECOMMENDED — selects by d AND orientation)",
                value=True,
                key="click_usesliding",
                help="ON = compute a local FFT in a sliding window over the "
                     "image; for each pixel, check whether its local FFT has "
                     "a peak at the grain's d-spacing AND orientation. This "
                     "gives per-pixel selectivity by BOTH d-spacing AND "
                     "orientation — solving the leak problem that arises when "
                     "only the d-spacing is used. OFF = use the legacy global "
                     "FFT with disk/ring masks (faster but prone to leaking).",
            )
            if click_params_obj.use_sliding_local_fft:
                click_params_obj.sliding_window_size_px = st.slider(
                    "Sliding window size (px)", 32, 256, 128, 8,
                    key="click_swin",
                    help="Side of the sliding window for local FFTs. "
                         "128 = good default (large enough to resolve small "
                         "d-spacings like 0.2–0.3 nm at typical TEM pixel "
                         "sizes 0.02–0.05 nm/px). 64 = too small for small "
                         "d-spacings — the peak falls in the DC region. "
                         "192–256 for very large grains. Larger = better FFT "
                         "SNR but lower spatial resolution.",
                )
                click_params_obj.sliding_step_px = st.slider(
                    "Sliding step (px)", 4, 32, 16, 2,
                    key="click_sstep",
                    help="Step of the sliding window. Smaller = finer spatial "
                         "resolution but slower. 16 = good balance.",
                )
                click_params_obj.sliding_zero_pad_size_px = st.slider(
                    "Sliding zero-pad (px)", 0, 256, 0, 16,
                    key="click_spad",
                    help="Zero-pad each sliding window to this size before FFT "
                         "(improves frequency resolution without increasing "
                         "window size). 0 = no padding (often works best for "
                         "sliding mode — zero-padding can amplify noise peaks).",
                )
                click_params_obj.orientation_tolerance_deg = st.slider(
                    "Orientation tolerance Δ (degrees)", 1.0, 30.0, 10.0, 1.0,
                    key="click_otol",
                    help="A pixel belongs to the grain if its local FFT has a "
                         "peak at the grain's d-spacing AND the peak's angle "
                         "is within ±Δ of the grain's orientation θ₀. "
                         "10° = good default. Larger = more tolerant (catches "
                         "mosaic spread) but may merge neighboring grains. "
                         "Smaller = stricter (better separation) but may "
                         "fragment a single grain with internal orientation spread.",
                )
            click_params_obj.use_orientation_specific_masks = st.checkbox(
                "Orientation-specific disk masks (legacy, when sliding is OFF)",
                value=True,
                key="click_orientspec",
                help="Used only when sliding local FFT is OFF. ON = use DISK "
                     "masks around individual 2D reflections (with their angle) "
                     "instead of RING masks. This separates grains by "
                     "ORIENTATION as well as by d-spacing — without it, all "
                     "grains with the same d-spacing get high amplitude and "
                     "region growing leaks across grain boundaries.",
            )
            click_params_obj.disk_mask_radius_px = st.slider(
                "Disk mask radius (FFT px)", 1.0, 8.0, 5.0, 0.5,
                key="click_diskr",
                help="Radius of the disk mask around each 2D reflection. "
                     "Smaller = more orientation-selective but may miss the "
                     "reflection if calibration drifts. 5 = good default "
                     "(with auto-align, this is robust).",
            )
            click_params_obj.strict_and = st.checkbox(
                "Strict AND (all d-spacings required)", value=True,
                key="click_strictand",
                help="ON (recommended) = a pixel is accepted only if ALL "
                     "d-spacings have high amplitude (MIN across d-spacings). "
                     "This is essential with orientation-specific disk masks — "
                     "otherwise a single 'leaky' d-spacing (e.g. a harmonic "
                     "whose amplitude map doesn't separate grains by "
                     "orientation) causes region growing to leak across grain "
                     "boundaries. OFF = accept pixels where the MEAN amplitude "
                     "is high (more forgiving but prone to leaking).",
            )
            click_params_obj.max_grain_area_frac = st.slider(
                "Max grain area (frac of layer)", 0.05, 0.95, 0.30, 0.05,
                key="click_maxarea",
                help="Safety: if the grown region exceeds this fraction of "
                     "the layer area, it's likely a 'leak' (region growing "
                     "escaped the grain boundary). The app tries raising the "
                     "growth threshold; if that doesn't help, the grain is "
                     "rejected. 0.30 = 30% of the layer (reasonable for "
                     "medium grains).",
            )
            click_params_obj.auto_find_similar = st.checkbox(
                "Auto-find similar grains", value=True,
                key="click_autofind",
                help="After each click, scan the rest of the layer for other "
                     "connected regions with the same d-spacing signature.",
            )
            click_params_obj.min_grain_area_px = st.slider(
                "Min grain area (px)", 10, 5000, 100, 10,
                key="click_minarea",
                help="Min area (in original-image pixels) for a region to "
                     "qualify as a separate grain (used by auto-find similar).",
            )
            st.session_state._click_params = click_params_obj
            # No 'params' alias needed — click mode has its own analyze path.
        elif seg_method == "Structure tensor (auto-assist)":
            st_params = structure_tensor_segmentation.StructureTensorParams()
            st_params.sigma = st.slider("Sigma (window size, px)", 1.0, 20.0, 5.0, 0.5,
                                         key="st_sigma",
                                         help="Structure tensor window. ~1.5× lattice-fringe period. Larger = smoother, smaller = noisier.")
            st_params.coherence_threshold = st.slider("Coherence threshold", 0.01, 0.50, 0.15, 0.01,
                                                       key="st_coh",
                                                       help="Pixels with coherence < this = background. Lower = more grains found (including weak fringes).")
            st_params.orientation_median_filter_px = st.slider("Orientation smoothing (px)", 0, 15, 5, 1,
                                                                key="st_smooth",
                                                                help="Median filter on orientation map. Reduces noise → cleaner grain boundaries. 0 = off.")
            st_params.n_orientation_bins = st.slider("Orientation bins", 6, 144, 72, 6,
                                                      key="st_bins",
                                                      help="Number of orientation bins. 72=2.5°/bin (fine), 36=5°/bin (coarse). More = better discrimination of similar grains.")
            st_params.use_watershed_refinement = st.checkbox("Watershed refinement", value=False,
                                                              key="st_ws",
                                                              help="Split touching grains with same orientation. OFF by default (can over-segment on noisy TEM).")
            st_params.min_grain_area_px = st.slider("Min grain area (px)", 10, 5000, 200, 10,
                                                     key="st_minarea",
                                                     help="Remove grains smaller than this (full-res px).")
            st_params.max_image_dim = st.slider("Max image dimension (px)", 256, 2048, 1024, 64,
                                                 key="st_maxdim",
                                                 help="Downscale before processing (for speed). 1024 = good balance.")
            st_params.merge_adjacent_orientations = st.checkbox("Merge adjacent-orientation grains", value=False,
                                                                 key="st_merge",
                                                                 help="OFF by default. ON = merge grains whose orientations differ by 1 bin.")
            st_params.exclude_left_right_borders = st.checkbox("Exclude left/right border grains", value=True, key="st_elr")
            st_params.exclude_top_bottom_borders = st.checkbox("Exclude top/bottom border grains", value=False, key="st_etb",
                                                                 help="Default OFF — top/bottom are layer interfaces.")
            params = st_params
        elif seg_method == "FFT (Grachev)":
            fft_params = fft_segmentation.FFTSegmentationParams()
            # Wiener filter (Grachev §2.4.1).
            fft_params.use_wiener_filter = st.checkbox("Use Wiener filter (recommended)", value=True,
                                                         key="fft_wiener",
                                                         help="Suppresses amorphous-matrix noise in FFT. ON = cleaner reflections.")
            if fft_params.use_wiener_filter:
                fft_params.wiener_noise_scale = st.slider("Wiener noise scale", 0.5, 3.0, 1.0, 0.1,
                                                           key="fft_wns",
                                                           help=">1 = more aggressive noise suppression (fewer reflections). <1 = less aggressive.")
                fft_params.wiener_smoothing_sigma = st.slider("Wiener smoothing σ (px)", 1.0, 20.0, 5.0, 1.0,
                                                                key="fft_wss",
                                                                help="Smooths the radial noise profile. Higher = less likely to suppress real reflections.")
            # Reflection detection.
            fft_params.use_radial_profile_detection = st.checkbox(
                "Use 1D radial-profile detection (recommended)", value=True,
                key="fft_use_radial",
                help="ON: integrate the power spectrum azimuthally into a 1D S(r) "
                     "profile, find peaks there (each peak = one d-spacing), then "
                     "look for 2D local maxima only within rings at those radii. "
                     "Much more robust than 2D peak_local_max. OFF: legacy 2D detector."
            )
            if fft_params.use_radial_profile_detection:
                fft_params.radial_smoothing_sigma = st.slider(
                    "Radial smoothing σ (px)", 0.5, 8.0, 2.0, 0.5,
                    key="fft_rsmooth",
                    help="Gaussian smoothing of the 1D S(r) profile. Higher = "
                         "fewer but more reliable peaks."
                )
                fft_params.radial_height_factor = st.slider(
                    "Radial peak height factor", 0.1, 1.0, 0.40, 0.05,
                    key="fft_rheight",
                    help="Peak height threshold = baseline + factor × (90th percentile − baseline). "
                         "Higher = fewer peaks (stricter)."
                )
                fft_params.radial_ring_local_max_frac = st.slider(
                    "Ring local-max threshold (frac of ring max)", 0.10, 0.80, 0.30, 0.05,
                    key="fft_ringmax",
                    help="Within each ring (at a detected peak radius), keep 2D "
                         "local maxima above this fraction of the ring's max. "
                         "Higher = fewer reflections per ring."
                )
            else:
                fft_params.peak_threshold_frac = st.slider(
                    "Reflection peak threshold (frac of max)", 0.001, 0.10, 0.02, 0.001,
                    key="fft_peakthr",
                    help="(Legacy 2D detector) Lower = more reflections (more sensitive).")
                fft_params.dbscan_eps = st.slider(
                    "DBScan eps (FFT-px)", 2.0, 30.0, 8.0, 1.0,
                    key="fft_eps",
                    help="(Legacy 2D detector) Max radius difference for reflections in the same d-spacing group.")
            fft_params.min_freq_frac = st.slider("Min spatial frequency (frac of Nyquist)", 0.01, 0.20, 0.05, 0.01,
                                                  key="fft_minfreq", help="Exclude central low frequencies (illumination).")
            fft_params.max_freq_frac = st.slider("Max spatial frequency (frac of Nyquist)", 0.20, 0.50, 0.45, 0.01,
                                                  key="fft_maxfreq", help="Exclude high-frequency noise.")
            fft_params.ring_mask_width_px = st.slider("Ring mask width (FFT-px)", 2.0, 20.0, 6.0, 1.0,
                                                   key="fft_ringw", help="Width of the ring mask around each d-spacing's radius (Grachev §2.4.2). Wider = captures more of the reflection. For small grains (broad peaks), increase to 8–10.")
            fft_params.contrast_target_ratio = st.slider("Contrast target (mean/max)", 0.1, 0.5, 0.3, 0.05,
                                                          key="fft_contrast", help="Grachev's contrast normalization. 0.3 = optimal per Grachev §2.4.2.")
            fft_params.min_amplitude_frac = st.slider("Min amplitude (frac of max)", 0.01, 0.30, 0.05, 0.01,
                                                       key="fft_minamp",
                                                       help="Pixels below this = background. Higher = fewer grains, more background.")
            fft_params.min_grain_area_px = st.slider("Min grain area (px)", 10, 5000, 500, 10,
                                                      key="fft_minarea",
                                                      help="Remove grains smaller than this (in full-res px). Higher = fewer, larger grains.")
            fft_params.max_clusters = st.slider("Max d-spacing groups", 5, 50, 30, 1,
                                                key="fft_maxcl", help="Hard limit on distinct d-spacings.")
            fft_params.max_image_dim = st.slider("FFT image dimension (px)", 256, 2048, 1024, 64,
                                                  key="fft_maxdim", help="Downscale before FFT. 1024 = good balance. 2048 = finer d-spacing resolution but slower.")
            fft_params.exclude_left_right_borders = st.checkbox("Exclude left/right border grains", value=True, key="fft_elr")
            fft_params.exclude_top_bottom_borders = st.checkbox("Exclude top/bottom border grains", value=False, key="fft_etb",
                                                                 help="Default OFF — top/bottom are layer interfaces.")

            # Reflection quality (manual mode).
            with st.expander("Reflection quality (manual clicks)", expanded=False):
                fft_params.reflection_fit_window_px = st.slider(
                    "Gaussian fit window (px, odd)", 7, 31, 15, 2,
                    key="fft_qwin",
                    help="Side of the square window for fitting a 2D Gaussian "
                         "around each click. Larger = better for broad peaks "
                         "(small grains). Must be odd.",
                )
                fft_params.sigma_good_px = st.slider(
                    "σ 'good' threshold (px)", 1.0, 5.0, 3.0, 0.5,
                    key="fft_sigmagood",
                    help="Clicks with σ below this = 'good' (sharp Bragg peak). "
                         "Raise for small grains (which give intrinsically broad peaks).",
                )
                fft_params.sigma_borderline_px = st.slider(
                    "σ 'noise' threshold (px)", 3.0, 15.0, 6.0, 0.5,
                    key="fft_sigmanoise",
                    help="Clicks with σ above this = 'noise'. For very small grains "
                         "(3–7 nm) raise to 8–10.",
                )
                fft_params.snr_good = st.slider(
                    "SNR 'good' threshold", 2.0, 20.0, 5.0, 0.5,
                    key="fft_snrgood",
                    help="Clicks with SNR above this and σ < 'good' threshold = 'good'.",
                )
                fft_params.scherrer_K = st.slider(
                    "Scherrer K", 0.7, 1.2, 0.94, 0.01,
                    key="fft_scherrK",
                    help="Scherrer constant. 0.94 = spherical domains (standard). "
                         "0.89 = cubic, 1.0 = needle along the beam.",
                )
                fft_params.apply_quality_filter_to_auto = st.checkbox(
                    "Apply Gaussian quality filter to AUTO reflections",
                    value=False, key="fft_autoq",
                    help="When ON, each 2D maximum found by the auto-detector is "
                         "checked via the Gaussian fit + Scherrer test; maxima "
                         "classified as 🔴 noise are dropped, and if ALL maxima "
                         "of a ring are noise the whole ring is dropped. Slower "
                         "but much cleaner auto-detection.",
                )

            # Reference d-spacings (mode B) — restrict detection to known planes.
            with st.expander("Reference d-spacings (mode B)", expanded=False):
                fft_params.use_reference_d_spacings = st.checkbox(
                    "Restrict detection to known plane families",
                    value=False, key="fft_useref",
                    help="When ON, the detector searches for reflections ONLY "
                         "inside the radial bands corresponding to the (d_min, d_max) "
                         "ranges listed below — instead of scanning the whole "
                         "spectrum. Much fewer false positives when you know the "
                         "material. When OFF, the table is ignored.",
                )
                fft_params.reference_tolerance_pct = st.slider(
                    "Tolerance (±% per band)", 0.0, 10.0, 3.0, 0.5,
                    key="fft_reftol",
                    help="Widens each (d_min, d_max) band symmetrically so peaks "
                         "slightly outside the user range are still caught. Useful "
                         "for noisy / calibration-drift data.",
                )

                # Preset selector.
                preset_names = list(fft_segmentation.MATERIAL_PRESETS.keys())
                col_pre, col_apply = st.columns([2, 1])
                with col_pre:
                    selected_preset = st.selectbox(
                        "Material preset", preset_names, index=len(preset_names) - 1,
                        key="fft_preset",
                        help="Pick a material to auto-fill the table with its "
                             "strongest reflections. 'Custom' = start from scratch.",
                    )
                with col_apply:
                    st.write("")  # spacing
                    st.write("")  # spacing
                    apply_preset = st.button("Apply preset", key="fft_apply_preset")

                # Manage reference table in session_state.
                if "fft_ref_rows" not in st.session_state:
                    st.session_state.fft_ref_rows = []  # list of [label, d_min, d_max]

                if apply_preset:
                    preset = fft_segmentation.MATERIAL_PRESETS[selected_preset]
                    st.session_state.fft_ref_rows = [
                        [label, float(d_min), float(d_max)]
                        for (label, d_min, d_max) in preset["d_spacings"]
                    ]

                # Render the editable table.
                st.markdown("**Reference d-spacings** (one row per plane family):")
                rows = st.session_state.fft_ref_rows
                # Use a data_editor for inline editing.
                if rows:
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=["label", "d_min (nm)", "d_max (nm)"])
                    edited = st.data_editor(
                        df,
                        num_rows="dynamic",
                        use_container_width=True,
                        key="fft_ref_editor",
                        hide_index=True,
                    )
                    # Convert back to list of tuples.
                    rows = [
                        (str(r["label"]), float(r["d_min (nm)"]), float(r["d_max (nm)"]))
                        for _, r in edited.iterrows()
                    ]
                    st.session_state.fft_ref_rows = list(rows)
                else:
                    st.caption("No rows yet. Click 'Apply preset' or 'Add row' below "
                               "to populate the table.")
                    if st.button("➕ Add empty row", key="fft_addrow"):
                        st.session_state.fft_ref_rows = [["{hkl}", 0.0, 0.0]]
                        st.rerun()

                # Sync back to fft_params.
                valid_rows = [
                    (lbl, dmin, dmax) for (lbl, dmin, dmax) in st.session_state.fft_ref_rows
                    if dmin > 0 and dmax > dmin and lbl
                ]
                fft_params.reference_d_spacings = valid_rows

                if fft_params.use_reference_d_spacings and not valid_rows:
                    st.warning("Reference mode is ON but no valid rows — "
                               "fall back to full-spectrum detection.")
                elif fft_params.use_reference_d_spacings:
                    # Sanity-check: warn on very wide bands (>30% relative width),
                    # which typically contain multiple d-spacings and need
                    # sub-division (handled automatically by the detector, but
                    # the user should know).
                    wide_bands = [
                        (lbl, dmin, dmax) for (lbl, dmin, dmax) in valid_rows
                        if (dmax - dmin) / dmin > 0.30
                    ]
                    if wide_bands:
                        st.info(f"ℹ️ {len(wide_bands)} band(s) are wide "
                                f"(>30% relative width). They will be "
                                f"automatically subdivided by 1D peak "
                                f"detection — each real d-spacing inside the "
                                f"band becomes its own cluster.")
                    else:
                        st.caption(f"✓ {len(valid_rows)} band(s) will be probed; "
                                   "all other radii are ignored.")

            params = fft_params  # alias for the analyze button
        else:  # Classical watershed
            params = segmentation.SegmentationParams()
            params.bilateral_d = st.slider("Bilateral filter d (px)", 1, 25, 9, 1, key="adv_bil_d")
            params.bilateral_sigma_color = st.slider("Bilateral σ_color", 1.0, 200.0, 50.0, 5.0, key="adv_bil_sc")
            params.bilateral_sigma_space = st.slider("Bilateral σ_space", 1.0, 200.0, 50.0, 5.0, key="adv_bil_ss")
            params.clahe_clip_limit = st.slider("CLAHE clip limit", 0.5, 10.0, 3.0, 0.5, key="adv_clahe")
            params.clahe_tile_grid = st.slider("CLAHE tile grid", 2, 32, 8, 1, key="adv_clahe_tg")
            params.use_scharr = st.checkbox("Use Scharr (vs Sobel)", value=True, key="adv_scharr")
            params.gradient_threshold_mode = st.selectbox(
                "Gradient threshold mode", ["otsu", "percentile", "absolute"],
                index=0, key="adv_gmode",
            )
            params.gradient_percentile = st.slider("Gradient percentile", 50.0, 99.5, 90.0, 0.5, key="adv_gpct")
            params.gradient_threshold = st.slider("Gradient absolute threshold", 1.0, 500.0, 100.0, 1.0,
                                                    key="adv_gabs") if params.gradient_threshold_mode == "absolute" else None
            params.gradient_threshold_scale = st.slider("Threshold scale factor", 0.5, 2.0, 1.0, 0.05,
                                                         key="adv_gscale",
                                                         help=">1 = stricter (fewer boundaries, larger grains). <1 = looser.")
            params.close_kernel_px = st.slider("Morphological close kernel (px)", 0, 11, 3, 1, key="adv_close")
            params.open_kernel_px = st.slider("Morphological open kernel (px)", 0, 11, 2, 1, key="adv_open")
            params.watershed_min_distance_px = st.slider(
                "Watershed min distance (px, 0=auto)", 0, 100, 0, 1, key="adv_wmd",
                help="0 = auto (5% of layer height). Larger = fewer, larger grains.")
            params.watershed_min_distance_factor = st.slider("Auto factor (layer-height frac)", 0.01, 0.20, 0.05, 0.01, key="adv_wmdf")
            params.watershed_min_intensity_frac = st.slider("Min peak intensity (frac of max)", 0.0, 0.5, 0.1, 0.01, key="adv_wmi")
            params.min_grain_area_nm2 = st.slider("Min grain area (nm²)", 1.0, 1000.0, 25.0, 1.0, key="adv_minarea")
            params.max_grain_area_nm2 = st.slider("Max grain area (nm²)", 100.0, 1.0e6, 1.0e5, 100.0, key="adv_maxarea")
            params.exclude_left_right_borders = st.checkbox("Exclude left/right border grains", value=True, key="adv_elr")
            params.exclude_top_bottom_borders = st.checkbox("Exclude top/bottom border grains", value=False, key="adv_etb",
                                                              help="Default OFF — top/bottom are layer interfaces.")
            params.border_margin_px = st.slider("Border margin (px)", 0, 20, 1, 1, key="adv_bm")

    # === Main panel: preview + analysis ===
    st.subheader("1. Image Preview & Layer Selection")
    selected_layer = st.session_state.selected_layer_mask

    # Preview: full image with layer overlay.
    preview = _downscale_for_preview(li.image, max_dim=900)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    scale = preview.shape[0] / li.image.shape[0]

    # Draw selected layer.
    if selected_layer is not None:
        # Downscale the mask to match the preview.
        mask_small = cv2.resize(selected_layer.mask.astype(np.uint8), (preview.shape[1], preview.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        preview_rgb = _draw_mask_on_image(preview_rgb, mask_small, color=(0, 255, 0), alpha=0.3)
        # Draw polygon outline.
        poly_small = selected_layer.polygon * scale
        preview_rgb = _draw_polygon_on_image(preview_rgb, poly_small, color=(255, 0, 0), thickness=2)

    # Draw SAM auto candidates (if in SAM auto mode and not yet selected).
    if layer_mode == "SAM (auto)" and st.session_state.sam_auto_result is not None:
        for i, cand in enumerate(st.session_state.sam_auto_result.candidates):
            if cand.polygon is not None:
                color = (0, 200, 255) if i != st.session_state.sam_auto_selected_idx else (255, 255, 0)
                poly_small = cand.polygon * scale
                preview_rgb = _draw_polygon_on_image(preview_rgb, poly_small, color=color, thickness=1)

    # Draw SAM click result (if in SAM click mode).
    if layer_mode == "SAM (click on layer)" and st.session_state.sam_click_result is not None:
        for cand in st.session_state.sam_click_result.candidates:
            if cand.polygon is not None:
                poly_small = cand.polygon * scale
                preview_rgb = _draw_polygon_on_image(preview_rgb, poly_small, color=(0, 255, 0), thickness=2)

    # NOTE: Polygon vertices are NOT drawn here — they are shown only in the
    # Polygon (click vertices) selector below. Drawing them here caused
    # coordinate misalignment (vertices are in selector-preview coords, not
    # main-preview coords). The main preview shows only the SELECTED layer
    # (green mask + red polygon outline) once the user clicks "Use polygon".

    st.image(preview_rgb, use_container_width=True,
             caption=f"Full image ({li.image.shape[1]}×{li.image.shape[0]} px). "
                     "Green fill = selected layer, red outline = selected polygon, "
                     "cyan/yellow = SAM auto candidates.")

    # Show layer info.
    if selected_layer is not None:
        st.success(
            f"Layer selected (source={selected_layer.source}): "
            f"area={selected_layer.area_nm2:.1f} nm², "
            f"bbox height={selected_layer.height_nm:.1f} nm, "
            f"width={selected_layer.width_nm:.1f} nm, "
            f"mean intensity={selected_layer.mean_intensity:.1f}"
        )

        # Analyze button (skipped in click-to-segment and manual tracing modes — they have their own UI).
        if seg_method == "Manual tracing (polygon)":
            _render_manual_tracing_ui(li, md, selected_layer, ps)
        elif seg_method == "Click-to-segment grains (FFT)":
            # Click-to-segment mode: render the click UI directly (no "Analyze" button).
            _render_click_segmentation_ui(li, md, selected_layer, ps)
        elif st.button("Analyze grains", type="primary", key="analyze_btn"):
            with st.spinner(f"Segmenting grains ({seg_method})..."):
                try:
                    # Crop to the layer bbox first.
                    from src.polygon import crop_to_mask_bbox
                    cropped_img, cropped_mask, layer_bbox = crop_to_mask_bbox(
                        li.image, selected_layer.mask, padding=2
                    )
                    if seg_method == "Structure tensor (auto-assist)":
                        # Structure-tensor segmentation.
                        st_result = structure_tensor_segmentation.segment_grains_structure_tensor(
                            cropped_img, ps, params
                        )
                        result = segmentation.SegmentationResult(
                            labels=st_result.labels,
                            boundaries=np.zeros_like(st_result.labels, dtype=bool),
                            n_grains_raw=st_result.n_grains,
                            n_grains_filtered=st_result.n_grains,
                            gradient=np.zeros(cropped_img.shape, dtype=np.float32),
                            preprocessed=cropped_img,
                            gradient_threshold_value=0.0,
                            bbox=layer_bbox,
                        )
                        result.labels[~cropped_mask] = 0
                        result.n_grains_filtered = int(result.labels.max()) if result.labels.max() > 0 else 0
                        st.session_state.fft_result = None  # no FFT viz for this method
                        st.session_state.st_result = st_result
                    elif seg_method == "FFT (Grachev)":
                        # FFT-based segmentation on the cropped layer image.
                        # IMPORTANT: mask out pixels OUTSIDE the layer before FFT,
                        # so the FFT only sees the layer (not the surrounding matrix).
                        # Fill outside-layer pixels with the layer's mean intensity
                        # (minimizes FFT artifacts from the mask boundary).
                        layer_mean = float(cropped_img[cropped_mask].mean())
                        fft_input = cropped_img.copy()
                        fft_input[~cropped_mask] = int(layer_mean)
                        # Run FFT segmentation.
                        fft_result = fft_segmentation.segment_grains_fft(
                            fft_input, ps, params  # params is fft_params here
                        )
                        # Wrap in a SegmentationResult-compatible object.
                        result = segmentation.SegmentationResult(
                            labels=fft_result.labels,
                            boundaries=np.zeros_like(fft_result.labels, dtype=bool),
                            n_grains_raw=fft_result.n_grains,
                            n_grains_filtered=fft_result.n_grains,
                            gradient=np.zeros(cropped_img.shape, dtype=np.float32),
                            preprocessed=cropped_img,
                            gradient_threshold_value=0.0,
                            bbox=layer_bbox,
                        )
                        # Mask out labels outside the layer.
                        result.labels[~cropped_mask] = 0
                        # Recount.
                        result.n_grains_filtered = int(result.labels.max()) if result.labels.max() > 0 else 0
                        # Store FFT-specific viz data.
                        st.session_state.fft_result = fft_result
                        # Cache the FFT data for manual reflection re-segmentation.
                        # Use the same fft_input and params to ensure consistency.
                        fft_shifted_c, log_power_c, scale_c, small_shape_c = \
                            fft_segmentation.compute_fft_and_spectrum(fft_input, params)
                        st.session_state.fft_cached = (fft_shifted_c, log_power_c, scale_c, small_shape_c)
                        st.session_state._fft_params = params
                    else:  # Classical watershed
                        result = segmentation.segment_grains_in_mask(
                            li.image, selected_layer.mask, ps, params
                        )
                        st.session_state.fft_result = None
                    grains = measurements.measure_grains(result.labels, ps)
                    stats = measurements.compute_image_stats(
                        grains, selected_layer.height_nm, selected_layer.width_nm
                    )
                    # Build visualizations from the cropped region.
                    rmin, cmin, rmax, cmax = result.bbox
                    viz_img = li.image[rmin:rmax, cmin:cmax]
                    labeled = visualization.color_label_image(
                        result.labels, background=viz_img
                    )
                    overlay = visualization.boundary_overlay(viz_img, result.labels)
                    rep = report.build_report(
                        image_metadata=md,
                        layer_mask=selected_layer,
                        segmentation_params=params,
                        stats=stats,
                        grains=grains,
                        labeled_image_rgb=labeled,
                    )
                    st.session_state.analysis_result = result
                    st.session_state.analysis_grains = grains
                    st.session_state.analysis_stats = stats
                    st.session_state.analysis_labeled_image = labeled
                    st.session_state.analysis_overlay_image = overlay
                    st.session_state.analysis_report = rep
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

    # === Analysis results ===
    _render_analysis_results(md, selected_layer, ps)


def _check_sam_available() -> bool:
    """Check SAM availability; show warning + install instructions if missing."""
    if not sam_segmenter.is_available():
        missing = sam_segmenter.missing_dependencies()
        st.warning(
            "SAM dependencies are not installed. Install with:\n"
            f"```\npip install {' '.join(missing)}\n```\n\n"
            "Switch to **Polygon (click vertices)** or **Manual rows** mode in the meantime."
        )
        return False
    return True


def _get_sam_predictor(model_key: str, max_img_dim: int) -> sam_segmenter.Segmenter | None:
    """Get or create a cached SAM Segmenter instance.

    The Segmenter is cached in session state so the image embedding is computed
    only once per image (the encoder is the slowest step, ~5 s). If the
    ``model_key`` changes, the cached predictor is discarded and a new one is
    created.

    If the model's checkpoint is not yet cached, a clear warning + progress
    indicator is shown so the user knows a download is happening (~350 MB for
    ViT-B, ~1.2 GB for ViT-L — can take minutes on slow connections).
    """
    cached = st.session_state.sam_predictor
    if cached is not None and getattr(cached, "model_key", None) == model_key:
        return cached

    # Check if the checkpoint needs downloading.
    needs_download = not sam_segmenter.is_checkpoint_cached(model_key)
    model_info = sam_segmenter.get_model_info(model_key)
    size_mb = sam_segmenter.get_model_size_mb(model_key)

    if needs_download:
        st.warning(
            f"⚠️ **Downloading SAM model '{model_key}'** ({size_mb:.0f} MB)\n\n"
            f"{model_info['description']}\n\n"
            f"This is a one-time download — the model will be cached in "
            f"`{sam_segmenter.DEFAULT_CACHE_DIR}` and reused on subsequent runs. "
            f"Please wait, this can take several minutes depending on your connection."
        )

    # Create the Segmenter and load the model (downloads checkpoint if needed).
    try:
        seg = sam_segmenter.Segmenter(model_key=model_key)
        if needs_download:
            with st.spinner(f"Downloading {model_key} (~{size_mb:.0f} MB) and loading model..."):
                seg._ensure_model()
            st.success(f"✅ Model '{model_key}' downloaded and loaded successfully.")
        else:
            with st.spinner(f"Loading cached model '{model_key}'..."):
                seg._ensure_model()
        st.session_state.sam_predictor = seg
    except Exception as e:
        st.error(f"Failed to load SAM model '{model_key}': {e}")
        return None
    return st.session_state.sam_predictor


def _render_sam_click_selector(li: io_utils.LoadedImage) -> None:
    """SAM point-prompt mode: user clicks on the layer, SAM segments it.

    The SAM result mask is drawn on the sidebar image so the user sees the
    segmented region immediately, without having to look at the main preview.
    Uses ``on_click`` callback (same pattern as Polygon mode) so the first
    click registers reliably.
    """
    if not _check_sam_available():
        return

    # SAM parameters.
    with st.expander("SAM parameters", expanded=False):
        # Model selector — show ✓ for cached models, ⬇ for models needing download.
        model_keys = sam_segmenter.list_models()
        model_options = {}
        for k in model_keys:
            info = sam_segmenter.get_model_info(k)
            size_mb = sam_segmenter.get_model_size_mb(k)
            cached = sam_segmenter.is_checkpoint_cached(k)
            status = "✓ cached" if cached else f"⬇ ~{size_mb:.0f} MB download"
            model_options[k] = f"{k} ({status}) — {info['description']}"
        selected_model = st.selectbox(
            "SAM model",
            model_keys,
            format_func=lambda k: model_options[k],
            index=0,  # default: sam_vit_b
            key="sam_click_model",
            help=("Vanilla SAM = Meta's original model (natural images). "
                  "Micro-SAM = fine-tuned for microscopy (better for TEM). "
                  "EM-organelles variant is recommended for electron microscopy. "
                  "✓ = already downloaded; ⬇ = will download on first use."),
        )
        max_img_dim = st.slider(
            "Max image dimension (px)", 512, 2048, 1024, 64,
            key="sam_click_max_dim",
            help=("SAM internally encodes at 1024×1024. Default 1024. "
                  "Reduce to 768 or 512 if you get out-of-memory errors.")
        )
        st.caption(f"Image: {li.image.shape[1]}×{li.image.shape[0]} px. "
                   f"Will be downscaled to ≤{max_img_dim}px before SAM.")

    st.markdown("**Click on the layer** you want to analyse in the image below.")
    st.caption("SAM will segment the region containing your click point (green overlay). "
               "First click loads the model (~350 MB download) and encodes the image (~5 s). "
               "Subsequent clicks are instant. Click **Use this layer** when satisfied.")

    # Downscaled preview for click capture.
    preview = _downscale_for_preview(li.image, max_dim=800)
    h_prev, w_prev = preview.shape[:2]
    scale = h_prev / li.image.shape[0]

    # Draw existing SAM click result on the preview BEFORE rendering the widget,
    # so the segmented region is visible in the sidebar image.
    display = _to_rgb_for_streamlit(preview).copy()
    if st.session_state.sam_click_result is not None:
        for cand in st.session_state.sam_click_result.candidates:
            # Fill mask with green transparency.
            mask_small = cv2.resize(cand.mask.astype(np.uint8), (w_prev, h_prev),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = display.copy()
            overlay[mask_small] = (0, 255, 0)
            display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
            # Draw polygon outline.
            if cand.polygon is not None:
                poly_small = cand.polygon * scale
                pts = poly_small[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Capture clicks via on_click callback (same pattern as Polygon mode).
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error("streamlit-image-coordinates not installed. "
                 "Install with: pip install streamlit-image-coordinates")
        return

    # Read the current model + max_dim from session state (set by the selectbox/slider above).
    click_model_key = st.session_state.get("sam_click_model", sam_segmenter.DEFAULT_MODEL_KEY)
    click_max_dim = st.session_state.get("sam_click_max_dim", 1024)

    def _on_sam_click() -> None:
        """Callback when the SAM image-coordinates widget is clicked."""
        click = st.session_state.get("sam_click_coord")
        if click is None:
            return
        click_col_prev = int(click["x"]) if isinstance(click, dict) else int(click[0])
        click_row_prev = int(click["y"]) if isinstance(click, dict) else int(click[1])
        # Deduplicate: ignore if same as last processed click.
        last = st.session_state.get("sam_click_last_click")
        if last is not None and last[0] == click_row_prev and last[1] == click_col_prev:
            return
        # Store the click + model + max_dim for processing after the rerun.
        st.session_state.sam_click_pending = (
            click_row_prev, click_col_prev, click_max_dim, click_model_key
        )
        st.session_state.sam_click_last_click = (click_row_prev, click_col_prev)

    streamlit_image_coordinates(display, key="sam_click_coord", on_click=_on_sam_click)

    # Process pending SAM click (set by the callback). This runs on the rerun
    # after the click, so the spinner and st.success/st.error are visible.
    pending = st.session_state.get("sam_click_pending")
    if pending is not None:
        st.session_state.sam_click_pending = None  # consume
        click_row_prev, click_col_prev, click_max_dim, click_model_key = pending
        click_row = click_row_prev / scale
        click_col = click_col_prev / scale
        st.info(f"Clicked at (row={click_row:.0f}, col={click_col:.0f}) in original image. "
                f"Running SAM (model: {click_model_key})...")

        with st.spinner(f"Running SAM {click_model_key} (first click encodes image ~5 s)..."):
            seg = _get_sam_predictor(click_model_key, click_max_dim)
            if seg is None:
                return
            try:
                sam_res = seg.detect_layer_at_point(
                    li.image, (int(click_row), int(click_col)),
                    max_image_dim=click_max_dim,
                )
                st.session_state.sam_click_result = sam_res
            except Exception as e:
                st.error(f"SAM failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        # Trigger a rerun so the sidebar image re-renders with the new mask overlay.
        st.rerun()

    # Show the SAM result + "Use this layer" button.
    if st.session_state.sam_click_result is not None:
        sam_res = st.session_state.sam_click_result
        if sam_res.error:
            st.error(f"SAM error: {sam_res.error}")
        elif not sam_res.candidates:
            st.warning("SAM returned no mask at this point. Try clicking elsewhere.")
        else:
            cand = sam_res.candidates[0]
            st.success(f"SAM segmented: area={cand.area_px}px² "
                       f"({cand.area_px * li.pixel_size_nm**2:.1f} nm²), "
                       f"score={cand.predicted_iou:.3f}")
            col_use, col_retry = st.columns(2)
            with col_use:
                if st.button("Use this layer", key="sam_click_use_btn", type="primary"):
                    st.session_state.selected_layer_mask = polygon_mod.layer_mask_from_polygon(
                        cand.polygon, li.image.shape, li.pixel_size_nm, li.image, source="sam"
                    )
                    st.session_state.analysis_result = None
                    st.session_state.analysis_report = None
            with col_retry:
                if st.button("Clear", key="sam_click_clear_btn"):
                    st.session_state.sam_click_result = None
                    st.session_state.sam_click_last_click = None
                    st.rerun()


def _render_sam_auto_selector(li: io_utils.LoadedImage) -> None:
    """SAM automatic mode: finds all candidate layers, user picks from dropdown."""
    if not _check_sam_available():
        return

    with st.expander("SAM parameters", expanded=False):
        # Model selector — show ✓ for cached models, ⬇ for models needing download.
        model_keys = sam_segmenter.list_models()
        model_options = {}
        for k in model_keys:
            info = sam_segmenter.get_model_info(k)
            size_mb = sam_segmenter.get_model_size_mb(k)
            cached = sam_segmenter.is_checkpoint_cached(k)
            status = "✓ cached" if cached else f"⬇ ~{size_mb:.0f} MB download"
            model_options[k] = f"{k} ({status}) — {info['description']}"
        selected_model = st.selectbox(
            "SAM model",
            model_keys,
            format_func=lambda k: model_options[k],
            index=0,  # default: sam_vit_b
            key="sam_auto_model",
            help=("Vanilla SAM = Meta's original model (natural images). "
                  "Micro-SAM = fine-tuned for microscopy (better for TEM). "
                  "EM-organelles variant is recommended for electron microscopy. "
                  "✓ = already downloaded; ⬇ = will download on first use."),
        )
        max_img_dim = st.slider(
            "Max image dimension (px)", 512, 2048, 1024, 64,
            key="sam_auto_max_dim",
            help=("SAM internally encodes at 1024×1024. Default 1024. "
                  "Reduce to 768 or 512 if you get out-of-memory errors.")
        )
        pps = st.slider(
            "Points per side", 12, 48, 24, 4,
            key="sam_auto_pps",
            help=("SAM grid density. More = more candidate masks but slower and "
                  "more memory. Default 24. Reduce to 16 if memory is tight.")
        )
        st.caption(f"Image: {li.image.shape[1]}×{li.image.shape[0]} px. "
                   f"Will be downscaled to ≤{max_img_dim}px before SAM.")

    if st.button("Detect layers (SAM auto)", key="sam_auto_detect_btn", type="primary"):
        # Check if model needs downloading and warn the user.
        needs_download = not sam_segmenter.is_checkpoint_cached(selected_model)
        size_mb = sam_segmenter.get_model_size_mb(selected_model)
        if needs_download:
            st.warning(
                f"⚠️ **Downloading SAM model '{selected_model}'** (~{size_mb:.0f} MB)\n\n"
                f"This is a one-time download — cached for future runs. "
                f"Please wait, this can take several minutes depending on your connection."
            )
        spinner_msg = (
            f"Downloading {selected_model} (~{size_mb:.0f} MB) and running SAM auto..."
            if needs_download
            else f"Running SAM auto ({selected_model}, 5–30 s per image)..."
        )
        with st.spinner(spinner_msg):
            try:
                seg = sam_segmenter.Segmenter(model_key=selected_model)
                sam_res = seg.detect_layers(
                    li.image,
                    points_per_side=pps,
                    max_image_dim=max_img_dim,
                )
                st.session_state.sam_auto_result = sam_res
                st.session_state.sam_auto_selected_idx = None
                st.session_state.selected_layer_mask = None
            except Exception as e:
                st.error(f"SAM failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
        if needs_download:
            st.success(f"✅ Model '{selected_model}' downloaded and cached for future use.")
        if sam_res.error:
            st.error(f"SAM error: {sam_res.error}")
        elif not sam_res.candidates:
            st.warning("SAM found no layer candidates. Try **SAM (click on layer)** mode, "
                       "or **Polygon (click vertices)** mode.")
        else:
            st.success(f"SAM found {sam_res.n_candidates} layer candidates "
                       f"(out of {sam_res.n_total_masks} raw masks).")

    sam_res = st.session_state.sam_auto_result
    if sam_res is not None and sam_res.candidates:
        options = []
        for i, c in enumerate(sam_res.candidates):
            options.append(
                f"#{i+1}: area={c.area_px}px² ({c.area_px * li.pixel_size_nm**2:.1f} nm²), "
                f"ecc={c.eccentricity:.2f}, mean={c.mean_intensity:.0f}, "
                f"IoU={c.predicted_iou:.2f}"
            )
        selected_idx = st.selectbox(
            "SAM candidates (sorted by area)",
            list(range(len(sam_res.candidates))),
            format_func=lambda i: options[i],
            index=0,
            key="sam_auto_cand_idx",
        )
        if st.button("Use this SAM candidate", key="sam_auto_use_btn", type="primary"):
            cand = sam_res.candidates[selected_idx]
            st.session_state.selected_layer_mask = polygon_mod.layer_mask_from_polygon(
                cand.polygon, li.image.shape, li.pixel_size_nm, li.image, source="sam"
            )
            st.session_state.sam_auto_selected_idx = selected_idx
            st.session_state.analysis_result = None
            st.session_state.analysis_report = None


def _render_polygon_click_selector(li: io_utils.LoadedImage) -> None:
    """Polygon mode with click-to-add-vertex using streamlit-image-coordinates.

    Implementation notes:
    - ``streamlit_image_coordinates`` is a stateful widget: Streamlit stores its
      latest value in ``st.session_state[key]``. We use ``on_change`` callback to
      detect NEW clicks (vs. the widget re-rendering with the same stale value).
    - We track ``polygon_last_click`` in session state to detect when the widget
      value actually changes (a real new click), so the first click registers
      immediately and Undo doesn't re-add the popped vertex.
    - No ``st.rerun()`` is called after Undo/Clear — the widget re-renders
      naturally with the updated image (vertices drawn on it), and the
      ``on_change`` callback handles the next click.
    """
    st.markdown("**Click on the image** to add polygon vertices around the layer.")
    st.caption("Click 3+ vertices, then click **Use polygon**. "
               "Buttons: **Clear** (start over), **Undo last** (remove last vertex).")

    # Downscaled preview for click capture.
    preview = _downscale_for_preview(li.image, max_dim=800)
    h_prev, w_prev = preview.shape[:2]
    scale = h_prev / li.image.shape[0]

    if st.session_state.polygon_vertices is None:
        st.session_state.polygon_vertices = []
    vertices_prev = st.session_state.polygon_vertices  # in preview coords

    # Draw current vertices on the preview BEFORE showing it, so the user
    # sees their vertices update in real time.
    display = _to_rgb_for_streamlit(preview).copy()
    for i, (vy, vx) in enumerate(vertices_prev):
        cv2.circle(display, (int(vx), int(vy)), 6, (255, 0, 0), -1)
        cv2.circle(display, (int(vx), int(vy)), 6, (255, 255, 0), 1)
        cv2.putText(display, str(i + 1), (int(vx) + 8, int(vy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if len(vertices_prev) >= 2:
        pts = np.array([[v[1], v[0]] for v in vertices_prev], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [pts], isClosed=(len(vertices_prev) >= 3),
                      color=(0, 255, 0), thickness=2)

    # Capture clicks via on_change callback to reliably detect new clicks.
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error("streamlit-image-coordinates not installed. "
                 "Install with: pip install streamlit-image-coordinates")
        return

    def _on_poly_click() -> None:
        """Callback when the image-coordinates widget value changes."""
        click = st.session_state.get("poly_click_coord")
        if click is None:
            return
        click_col = int(click["x"]) if isinstance(click, dict) else int(click[0])
        click_row = int(click["y"]) if isinstance(click, dict) else int(click[1])
        # Detect if this is a genuinely new click (different from the last one we processed).
        last = st.session_state.get("polygon_last_click")
        if last is not None and last[0] == click_row and last[1] == click_col:
            return  # stale value, ignore
        # New click — add vertex.
        st.session_state.polygon_vertices.append((float(click_row), float(click_col)))
        st.session_state.polygon_last_click = (click_row, click_col)

    # Render the widget with on_click callback.
    # Note: streamlit-image-coordinates uses `on_click` (not `on_change`).
    # Internally it's wired to the same on_change mechanism, so the callback
    # fires whenever the widget value changes (i.e. on each new click).
    streamlit_image_coordinates(display, key="poly_click_coord", on_click=_on_poly_click)

    # Buttons — no st.rerun() needed; the callback + natural re-render handle updates.
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear", key="poly_click_clear"):
            st.session_state.polygon_vertices = []
            st.session_state.polygon_last_click = None
            st.session_state.selected_layer_mask = None
            st.rerun()
    with col2:
        if st.button("Undo last", key="poly_click_undo"):
            if st.session_state.polygon_vertices:
                st.session_state.polygon_vertices.pop()
                # Reset the last-click tracker so the popped vertex doesn't get re-added.
                st.session_state.polygon_last_click = None
                st.rerun()
    with col3:
        if len(vertices_prev) >= 3:
            if st.button("Use polygon", key="poly_click_use", type="primary"):
                # Convert preview coords → original image coords.
                orig_polygon = np.array(
                    [[vy / scale, vx / scale] for vy, vx in vertices_prev],
                    dtype=np.float32
                )
                st.session_state.selected_layer_mask = polygon_mod.layer_mask_from_polygon(
                    orig_polygon, li.image.shape, li.pixel_size_nm, li.image, source="polygon"
                )
                st.session_state.analysis_result = None
                st.session_state.analysis_report = None
        else:
            st.caption(f"Need ≥3 vertices (have {len(vertices_prev)}).")


def _render_manual_rows_selector(li: io_utils.LoadedImage) -> None:
    """Render the manual-row-range layer selector (legacy)."""
    y0 = st.slider("Top row (y0)", 0, li.image.shape[0] - 1, 0, 1, key="man_y0_slider")
    y1 = st.slider("Bottom row (y1)", y0 + 1, li.image.shape[0],
                    li.image.shape[0], 1, key="man_y1_slider")
    if st.button("Set manual layer", key="set_manual_btn", type="primary"):
        st.session_state.selected_layer_mask = polygon_mod.layer_mask_from_rows(
            y0, y1, li.image.shape, li.pixel_size_nm, li.image
        )
        st.session_state.analysis_result = None
        st.session_state.analysis_report = None


def _render_manual_tracing_ui(li, md, selected_layer, ps) -> None:
    """Render the manual tracing (polygon) grain analysis UI.

    The user draws grain outlines by clicking vertices. Each closed polygon
    becomes one grain. After Finish, standard grain-size statistics are
    computed. This is ISO 13067 compliant (manual tracing + area method).
    """
    from streamlit_image_coordinates import streamlit_image_coordinates
    from src.polygon import crop_to_mask_bbox

    st.subheader("2. Manual tracing — draw grain outlines")
    st.caption("Click to add polygon vertices around each grain. Click "
               "'Close grain' to finish the current polygon. Repeat for "
               "each grain. When done, click 'Finish' to compute statistics. "
               "This is the most reliable method (ISO 13067 compliant).")

    # Crop to the layer bbox.
    full_img = li.image
    cropped_img, cropped_mask, layer_bbox = crop_to_mask_bbox(
        full_img, selected_layer.mask, padding=2
    )
    h_crop, w_crop = cropped_img.shape[:2]

    # Build the display image: cropped image + overlay of already-traced grains.
    mt_grains = st.session_state.get("mt_grains") or []
    current_vertices = st.session_state.get("mt_current_vertices") or []

    display_img = _build_mt_display_image(cropped_img, cropped_mask, mt_grains, current_vertices)

    st.markdown(f"**Cropped layer image** ({w_crop}×{h_crop}px). "
                f"Click to add vertices. Close the polygon to finish a grain.")

    # Click callback.
    def _on_mt_click() -> None:
        click = st.session_state.get("mt_click_coord")
        if click is None:
            return
        click_col = int(click["x"]) if isinstance(click, dict) else int(click[0])
        click_row = int(click["y"]) if isinstance(click, dict) else int(click[1])
        last = st.session_state.get("mt_last_click")
        if last is not None and last[0] == click_row and last[1] == click_col:
            return
        if st.session_state.mt_current_vertices is None:
            st.session_state.mt_current_vertices = []
        st.session_state.mt_current_vertices.append((float(click_row), float(click_col)))
        st.session_state.mt_last_click = (click_row, click_col)

    streamlit_image_coordinates(display_img, key="mt_click_coord", on_click=_on_mt_click)

    # Action buttons.
    col_close, col_undo, col_clear, col_finish = st.columns(4)
    with col_close:
        n_verts = len(current_vertices)
        if st.button(f"✓ Close grain ({n_verts} vertices)",
                      key="mt_close", type="primary",
                      disabled=(n_verts < 3)):
            # Close the polygon → create a grain mask.
            polygon = list(current_vertices)
            mask = np.zeros((h_crop, w_crop), dtype=bool)
            # Use cv2.fillPoly for the polygon mask.
            pts = np.array([[int(c), int(r)] for r, c in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [pts], True)
            mask = mask & cropped_mask  # restrict to layer
            if mask.sum() > 0:
                if st.session_state.mt_grains is None:
                    st.session_state.mt_grains = []
                st.session_state.mt_grains.append({
                    "mask": mask,
                    "polygon": polygon,
                    "area_px": int(mask.sum()),
                    "source": "manual-tracing",
                })
            st.session_state.mt_current_vertices = []
            st.session_state.mt_last_click = None
            st.rerun()
    with col_undo:
        if st.button("↩ Undo last vertex", key="mt_undo",
                      disabled=(n_verts == 0)):
            if current_vertices:
                current_vertices.pop()
                st.session_state.mt_current_vertices = current_vertices
                st.session_state.mt_last_click = None
                st.rerun()
    with col_clear:
        if st.button("🗑 Clear all", key="mt_clear", disabled=(not mt_grains and n_verts == 0)):
            st.session_state.mt_grains = []
            st.session_state.mt_current_vertices = []
            st.session_state.mt_last_click = None
            st.session_state.mt_finished = False
            st.rerun()
    with col_finish:
        if st.button("✓ Finish & compute statistics", key="mt_finish",
                      type="primary", disabled=(not mt_grains)):
            _finalize_mt_grains(li, md, selected_layer, ps, mt_grains,
                                 layer_bbox, cropped_img, cropped_mask)

    # Grains table.
    if mt_grains:
        st.markdown(f"### Grains traced: {len(mt_grains)}")
        table_rows = []
        for i, g in enumerate(mt_grains):
            area_px = g["area_px"]
            area_nm2 = area_px * ps * ps
            eq_diam_nm = 2 * np.sqrt(area_nm2 / np.pi)
            table_rows.append({
                "#": i + 1,
                "vertices": len(g["polygon"]),
                "area (px)": area_px,
                "area (nm²)": f"{area_nm2:.1f}",
                "eq. diam (nm)": f"{eq_diam_nm:.2f}",
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Grain selection + highlight + delete.
        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            grain_options = [f"Grain #{i+1} (area={g['area_px']} px, "
                             f"d={2*np.sqrt(g['area_px']*ps*ps/np.pi):.1f} nm)"
                             for i, g in enumerate(mt_grains)]
            selected_grain_idx = st.selectbox(
                "Select a grain to highlight", grain_options,
                key="mt_grain_select",
            )
            sel_idx = grain_options.index(selected_grain_idx) if selected_grain_idx else None
        with col_del:
            st.write("")  # spacing
            st.write("")
            if st.button("🗑 Delete selected", key="mt_del_grain",
                          disabled=(sel_idx is None)):
                if sel_idx is not None and sel_idx < len(mt_grains):
                    mt_grains.pop(sel_idx)
                    st.session_state.mt_grains = mt_grains
                    st.rerun()

        # Color-coded labeled image with highlight.
        labels_img = _build_mt_labels_image(cropped_img, cropped_mask, mt_grains,
                                             highlight_idx=sel_idx)
        st.markdown("**Color-coded grain map**" +
                    (f" (grain #{sel_idx+1} highlighted in yellow)" if sel_idx is not None else ""))
        st.image(labels_img, use_container_width=True)


def _build_mt_display_image(cropped_img, cropped_mask, mt_grains, current_vertices):
    """Build the display image for manual tracing: cropped + overlay of traced grains + current polygon."""
    if cropped_img.ndim == 2:
        disp = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    else:
        disp = cropped_img.copy()
        if disp.shape[2] == 4:
            disp = cv2.cvtColor(disp, cv2.COLOR_RGBA2RGB)
    # Dim pixels outside the layer.
    disp[~cropped_mask] = (disp[~cropped_mask].astype(np.int32) * 0.4).astype(np.uint8)
    # Overlay traced grains.
    if mt_grains:
        from skimage import color as skcolor
        n = len(mt_grains)
        labels = np.arange(n).reshape(-1, 1)
        colored = skcolor.label2rgb(labels, bg_label=-1, kind="overlay", alpha=1.0)
        colors = (colored[:, 0, :] * 255).astype(np.uint8)
        for i, g in enumerate(mt_grains):
            mask = g["mask"]
            color = colors[i % len(colors)]
            disp[mask] = (disp[mask].astype(np.int32) * 0.5
                           + color.astype(np.int32) * 0.5).astype(np.uint8)
            # Outline.
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, contours, -1, color.tolist(), 2)
    # Draw current polygon (being drawn).
    if current_vertices:
        pts = np.array([[int(c), int(r)] for r, c in current_vertices], dtype=np.int32)
        # Draw lines between vertices.
        for j in range(len(pts) - 1):
            cv2.line(disp, tuple(pts[j]), tuple(pts[j + 1]), (0, 255, 0), 2)
        # Draw vertices.
        for j, pt in enumerate(pts):
            cv2.circle(disp, tuple(pt), 4, (0, 255, 0), -1)
            cv2.circle(disp, tuple(pt), 4, (255, 255, 255), 1)
            cv2.putText(disp, str(j + 1), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # If ≥3 vertices, draw a dashed line back to the first (preview).
        if len(pts) >= 3:
            cv2.line(disp, tuple(pts[-1]), tuple(pts[0]), (0, 200, 0), 1)
    return disp


def _build_mt_labels_image(cropped_img, cropped_mask, mt_grains, highlight_idx=None):
    """Build a color-coded labeled image for manual tracing.

    If highlight_idx is given, the corresponding grain is outlined in yellow.
    """
    if cropped_img.ndim == 2:
        bg = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    else:
        bg = cropped_img.copy()
        if bg.shape[2] == 4:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)
    if not mt_grains:
        return bg
    labels = np.zeros(cropped_img.shape[:2], dtype=np.int32)
    for i, g in enumerate(mt_grains, start=1):
        labels[g["mask"]] = i
    from skimage import color as skcolor
    colored = skcolor.label2rgb(labels, bg_label=0, kind="overlay", alpha=0.55,
                                 bg_color=(0, 0, 0))
    colored = (colored * 255).astype(np.uint8)
    no_grain = labels == 0
    colored[no_grain] = bg[no_grain]
    # Highlight the selected grain with a yellow outline.
    if highlight_idx is not None and highlight_idx < len(mt_grains):
        mask = mt_grains[highlight_idx]["mask"]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored, contours, -1, (255, 255, 0), 3)  # yellow
    return colored


def _finalize_mt_grains(li, md, selected_layer, ps, mt_grains,
                         layer_bbox, cropped_img, cropped_mask):
    """Convert manually-traced grains into a labeled image and compute statistics."""
    if not mt_grains:
        st.warning("No grains to finalize.")
        return
    labels = np.zeros(cropped_img.shape[:2], dtype=np.int32)
    for i, g in enumerate(mt_grains, start=1):
        labels[g["mask"]] = i
    grains = measurements.measure_grains(labels, ps)
    stats = measurements.compute_image_stats(
        grains, selected_layer.height_nm, selected_layer.width_nm
    )
    rmin, cmin, rmax, cmax = layer_bbox
    viz_img = li.image[rmin:rmax, cmin:cmax]
    labeled = visualization.color_label_image(labels, background=viz_img)
    overlay = visualization.boundary_overlay(viz_img, labels)
    rep = report.build_report(
        image_metadata=md,
        layer_mask=selected_layer,
        segmentation_params=_DictParams({"method": "manual_tracing"}),
        stats=stats,
        grains=grains,
        labeled_image_rgb=labeled,
    )
    st.session_state.analysis_result = segmentation.SegmentationResult(
        labels=labels,
        boundaries=np.zeros_like(labels, dtype=bool),
        n_grains_raw=len(mt_grains),
        n_grains_filtered=len(mt_grains),
        gradient=np.zeros(viz_img.shape, dtype=np.float32),
        preprocessed=viz_img,
        gradient_threshold_value=0.0,
        bbox=layer_bbox,
    )
    st.session_state.analysis_grains = grains
    st.session_state.analysis_stats = stats
    st.session_state.analysis_labeled_image = labeled
    st.session_state.analysis_overlay_image = overlay
    st.session_state.analysis_report = rep
    st.session_state.fft_result = None
    st.session_state.st_result = None
    st.session_state.mt_finished = True
    st.success(f"Finalized {len(mt_grains)} grains. Scroll down for statistics.")
    st.rerun()


def _render_click_segmentation_ui(li, md, selected_layer, ps) -> None:
    """Render the click-to-segment grain analysis UI.

    The user clicks on grains in the cropped layer image. For each click:
      - A local FFT is computed around the click.
      - d-spacings of the clicked grain are found.
      - Region growing with hysteresis + AND-condition segments the grain.
      - Similar grains (same d-spacing signature) are auto-found.

    The user can Undo / Clear / Finish. After Finish, the standard grain
    statistics (histograms, lognormal fit) are computed from the clicked grains.
    """
    from streamlit_image_coordinates import streamlit_image_coordinates
    from src.polygon import crop_to_mask_bbox

    st.subheader("2. Click-to-segment grain analysis")
    st.caption("Click on each grain in the cropped image below. The app "
               "computes a local FFT around each click, finds the grain's "
               "d-spacings, and grows a region with the AND-condition "
               "(pixels where ALL d-spacings have high amplitude).")

    # Crop to the layer bbox.
    full_img = li.image
    cropped_img, cropped_mask, layer_bbox = crop_to_mask_bbox(
        full_img, selected_layer.mask, padding=2
    )
    h_crop, w_crop = cropped_img.shape[:2]

    # Get cached FFT params (for the local FFT preprocessing).
    fft_params = st.session_state.get("_fft_params", fft_segmentation.FFTSegmentationParams())
    click_params = st.session_state.get("_click_params", click_segmentation.ClickSegmentationParams())

    # Compute (or reuse) the global FFT — used for amplitude maps in legacy mode.
    global_fft_data = st.session_state.get("click_global_fft")
    if global_fft_data is None:
        with st.spinner("Computing global FFT (cached for all clicks)..."):
            try:
                layer_mean = float(cropped_img[cropped_mask].mean())
                fft_input = cropped_img.copy()
                fft_input[~cropped_mask] = int(layer_mean)
                gf, glp, gscale, gshape = fft_segmentation.compute_fft_and_spectrum(fft_input, fft_params)
                st.session_state.click_global_fft = (gf, glp, gscale, gshape)
                global_fft_data = st.session_state.click_global_fft
            except Exception as e:
                st.error(f"Failed to compute global FFT: {e}")
                return
    global_fft, global_log_power, global_scale, global_shape = global_fft_data

    # Compute (or reuse) the sliding local FFTs — used for amplitude maps in
    # sliding mode (mode B). Cache by parameters; recompute if they changed.
    sliding_data = None
    if click_params.use_sliding_local_fft:
        cur_params = (click_params.sliding_window_size_px,
                      click_params.sliding_step_px,
                      click_params.sliding_zero_pad_size_px)
        cached_params = st.session_state.get("click_sliding_params")
        cached_data = st.session_state.get("click_sliding_data")
        if cached_data is None or cached_params != cur_params:
            with st.spinner(f"Computing sliding local FFTs "
                            f"(window={click_params.sliding_window_size_px}, "
                            f"step={click_params.sliding_step_px})... this may take a few seconds..."):
                try:
                    sliding_data = click_segmentation.compute_sliding_local_ffts(
                        cropped_img, cropped_mask,
                        window_size=click_params.sliding_window_size_px,
                        step=click_params.sliding_step_px,
                        zero_pad_size=click_params.sliding_zero_pad_size_px,
                        fft_params=fft_params,
                    )
                    st.session_state.click_sliding_data = sliding_data
                    st.session_state.click_sliding_params = cur_params
                except Exception as e:
                    st.error(f"Failed to compute sliding FFTs: {e}")
                    return
        else:
            sliding_data = cached_data
        n_windows = sliding_data["log_powers"].shape[0] if sliding_data else 0
        st.caption(f"Sliding FFT cache: {n_windows} windows "
                   f"(window={click_params.sliding_window_size_px}px, "
                   f"step={click_params.sliding_step_px}px).")

    # Build the display image: cropped image + overlay of already-found grains.
    click_grains = st.session_state.get("click_grains") or []
    display_img = _build_click_display_image(cropped_img, cropped_mask, click_grains)

    # --- Click capture ---
    st.markdown(f"**Cropped layer image** ({w_crop}×{h_crop}px). "
                f"Click on the **center** of a grain (where lattice fringes are visible).")

    def _on_grain_click() -> None:
        click = st.session_state.get("grain_click_coord")
        if click is None:
            return
        click_col = int(click["x"]) if isinstance(click, dict) else int(click[0])
        click_row = int(click["y"]) if isinstance(click, dict) else int(click[1])
        last = st.session_state.get("click_last_click_coord")
        if last is not None and last[0] == click_row and last[1] == click_col:
            return
        # Run the click-to-segment pipeline.
        try:
            result = click_segmentation.segment_grain_from_click(
                cropped_img, cropped_mask, click_row, click_col, ps,
                global_fft, global_shape, global_scale, fft_params, click_params,
                sliding_data=sliding_data,
            )
            st.session_state.click_last_result = result
            st.session_state.click_last_click_coord = (click_row, click_col)
            if result.success:
                # Add the main grain.
                if st.session_state.click_grains is None:
                    st.session_state.click_grains = []
                st.session_state.click_grains.append({
                    "mask": result.grain_mask,
                    "signature": result.signature,
                    "click_point": result.click_point,
                    "area_px": int(result.grain_mask.sum()),
                    "d_spacings": result.signature.d_spacings_nm,
                    "source": "click",
                })
                # Stage the similar grains for user confirmation.
                if result.similar_grain_masks:
                    st.session_state.click_pending_similar = result.similar_grain_masks
                else:
                    st.session_state.click_pending_similar = None
        except Exception as e:
            import traceback
            st.error(f"Click-to-segment failed: {e}")
            st.code(traceback.format_exc())

    streamlit_image_coordinates(display_img, key="grain_click_coord", on_click=_on_grain_click)

    # --- Last click preview ---
    last_result = st.session_state.get("click_last_result")
    if last_result is not None:
        status_emoji = "✅" if last_result.success else "❌"
        with st.expander(f"Last click result: {status_emoji} {last_result.message}",
                          expanded=True):
            if last_result.local_power_spectrum is not None and last_result.local_power_spectrum.size > 0:
                col_lps, col_info = st.columns([1, 1])
                with col_lps:
                    st.markdown("**Local power spectrum** (FFT of the click window)")
                    ps_disp = fft_segmentation.render_power_spectrum(
                        last_result.local_power_spectrum, None, None, max_dim=700,
                    )
                    # Force to 700×700 so it matches the Manual reflection
                    # selection display below.
                    if ps_disp.shape[0] != 700 or ps_disp.shape[1] != 700:
                        ps_disp = cv2.resize(ps_disp, (700, 700),
                                              interpolation=cv2.INTER_NEAREST)
                    st.image(ps_disp, use_container_width=True)
                    st.caption(f"FFT size: {last_result.local_power_spectrum.shape[0]}×"
                               f"{last_result.local_power_spectrum.shape[1]}px, "
                               f"displayed at 700×700.")
                with col_info:
                    st.markdown("**Grain signature** (d-spacings found in this grain)")
                    if last_result.signature.d_spacings_nm:
                        sig_rows = []
                        for i, (d, q) in enumerate(zip(last_result.signature.d_spacings_nm,
                                                        last_result.signature.qualities)):
                            # Per-d seed amplitude (if available).
                            seed_amp = (last_result.seed_amplitudes[i]
                                        if i < len(last_result.seed_amplitudes) else None)
                            sig_rows.append({
                                "#": i + 1,
                                "d (nm)": f"{d:.4f}",
                                "θ (deg)": (f"{last_result.signature.orientations_deg[i]:.1f}"
                                            if i < len(last_result.signature.orientations_deg) else "—"),
                                "σ (px)": f"{q.sigma_mean_px:.2f}",
                                "SNR": f"{q.snr:.1f}",
                                "D (nm)": (f"{q.coherent_size_nm:.1f}"
                                           if np.isfinite(q.coherent_size_nm) else "∞"),
                                "seed amp": (f"{seed_amp:.3f}" if seed_amp is not None else "—"),
                                "status": {"good": "🟢", "borderline": "🟡", "noise": "🔴"}.get(q.status, "?"),
                            })
                        st.dataframe(sig_rows, use_container_width=True, hide_index=True)
                    # Diagnostics for failed clicks.
                    if not last_result.success:
                        st.markdown("**Diagnostics:**")
                        diag_lines = [
                            f"- Failed step: `{last_result.failed_step}`",
                            f"- d-spacings kept after quality filter: {len(last_result.d_spacings_kept)}",
                            f"- Seed amplitudes (per d-spacing): " +
                            ", ".join(f"{a:.3f}" for a in last_result.seed_amplitudes),
                            f"- Combined seed amplitude: **{last_result.seed_combined:.3f}** "
                            f"(need ≥ {click_params.seed_threshold_frac:.2f})",
                            f"- Original grown area (BEFORE safety check): "
                            f"{last_result.original_grown_area_px} px "
                            f"({(last_result.original_grown_area_px / max(1, last_result.layer_area_px) * 100):.1f}% of layer)",
                            f"- Final grown area: {last_result.grown_area_px} px / "
                            f"layer = {last_result.layer_area_px} px "
                            f"({(last_result.grown_area_px / max(1, last_result.layer_area_px) * 100):.1f}%)",
                        ]
                        if last_result.safety_check_applied:
                            diag_lines.append(
                                f"- **Safety check was APPLIED** (region was too big). "
                                f"Thresholds tried (thr → area): " +
                                ", ".join(f"{t:.2f}→{a}" for t, a in last_result.safety_thresholds_tried[:5])
                            )
                        if last_result.fallback_used:
                            diag_lines.append(
                                f"- **Fallback used:** {last_result.fallback_used}"
                            )
                        for line in diag_lines:
                            st.markdown(line)
                        # Amplitude map diagnostics (auto-align info).
                        if last_result.amplitude_diag:
                            st.markdown("**Amplitude map diagnostics**:")
                            for ad in last_result.amplitude_diag:
                                mode = ad.get("mode", "global_disk" if ad.get("used_disk") else "global_ring")
                                if mode == "sliding":
                                    st.markdown(f"- d={ad['d_nm']:.4f} nm, θ={ad.get('orientation_deg', 0):.1f}°, "
                                                f"mode=SLIDING, window={ad.get('sliding_window_size', '?')}px, "
                                                f"r_target={ad.get('r_target_sliding', 0):.1f}px, "
                                                f"peaks={ad.get('n_with_peak', 0)}/{ad.get('n_windows', 0)} windows")
                                    if ad.get("r_warning"):
                                        st.warning(f"  ⚠️ {ad['r_warning']}")
                                    elif ad.get("n_with_peak", 0) == 0:
                                        st.warning(f"  ⚠️ No sliding windows found a significant peak for this "
                                                   f"d-spacing. The sliding window may be too small, or the "
                                                   f"d-spacing doesn't exist in the sliding windows. "
                                                   f"Try increasing the sliding window size.")
                                else:
                                    mask_type = "DISK (auto-aligned)" if ad.get("used_disk") else "RING (fallback)"
                                    st.markdown(f"- d={ad['d_nm']:.4f} nm, r_global={ad.get('r_global', 0):.1f} px, "
                                                f"mask={mask_type}, best SNR={ad.get('best_snr', 0):.2f}")
                                    for rd in ad.get("reflections", [])[:3]:
                                        st.markdown(f"  - refl #{rd['refl_idx']}: shift={rd['shift_px']:.1f} px, "
                                                    f"peak={rd['peak_intensity']:.2f}, bg={rd['background_intensity']:.2f}, "
                                                    f"SNR={rd['snr']:.2f}")
                        st.caption("**Troubleshooting:**")
                        st.caption("• If **r_target < 5px** → the d-spacing is too small for the sliding "
                                   "window. **INCREASE the sliding window size** (e.g. 128 → 192 or 256). "
                                   "This is the most common cause of seed amp = 0.000 in sliding mode.")
                        st.caption("• If **peaks = 0/N windows** → no sliding window found a significant "
                                   "peak. Either the sliding window is too small, or the d-spacing "
                                   "doesn't exist in the sliding windows. **Increase the sliding window size** "
                                   "or **decrease the local peak height factor**.")
                        st.caption("• If **best SNR < 2** (legacy mode) → the disk mask is hitting noise. "
                                   "**Increase disk mask radius** or **disable orientation-specific masks**.")
                        st.caption("• If **shift_px is large** (>3 px, legacy mode) → calibration drift. "
                                   "**Increase disk mask radius**.")
                        st.caption("• If **safety check was applied** and grown area still big → "
                                   "**decrease max grain area fraction** (0.30 → 0.10).")
                        st.caption("• If **seed amp is low** → lower the seed threshold (0.15 → 0.05).")
                    if last_result.success:
                        st.metric("Grain area", f"{int(last_result.grain_mask.sum())} px")
                        # Show fallback/safety info if applied.
                        if last_result.safety_check_applied or last_result.fallback_used:
                            info_parts = []
                            if last_result.safety_check_applied:
                                info_parts.append(
                                    f"safety check shrank region from "
                                    f"{last_result.original_grown_area_px} → "
                                    f"{last_result.grown_area_px} px"
                                )
                            if last_result.fallback_used:
                                info_parts.append(last_result.fallback_used)
                            st.info(" • ".join(info_parts))
                    if last_result.similar_grain_masks:
                        st.info(f"Found {len(last_result.similar_grain_masks)} similar grain(s) — "
                                "review below.")
            else:
                st.warning("No local power spectrum available.")

    # --- Manual reflection selection on the local power spectrum ---
    # After clicking on a grain, the user can manually select reflections
    # in the local power spectrum. This gives precise (d, θ) for the grain,
    # and the amplitude map is built from these exact reflections.
    last_result = st.session_state.get("click_last_result")
    if (last_result is not None
            and last_result.local_power_spectrum is not None
            and last_result.local_power_spectrum.size > 0):
        with st.expander("Manual reflection selection (advanced — precise (d, θ))",
                          expanded=False):
            st.caption("Click on reflections in the local power spectrum below "
                       "to select them. Each click defines a (d, θ) pair. Then "
                       "click 'Build amplitude maps' to segment the grain using "
                       "ONLY these reflections — this gives precise control "
                       "over which plane families define the grain.")
            from streamlit_image_coordinates import streamlit_image_coordinates

            log_power = last_result.local_power_spectrum
            h_fft, w_fft = log_power.shape

            # Build the display image — SAME size as the Local power spectrum
            # in "Last click result" (both use max_dim=512, then upscaled to
            # a fixed display size so they look identical).
            # NOTE: streamlit_image_coordinates shows the image at its NATIVE
            # pixel size (no use_container_width), while st.image(use_container_width=True)
            # stretches to the container width. To make them look the same size,
            # we use a LARGER fixed_display_size here (700 instead of 512) so
            # that the native display roughly matches the stretched 512 in a
            # half-width column.
            max_display = 700
            ps_base = fft_segmentation.render_power_spectrum(
                log_power, None, None, max_dim=max_display
            )
            # Force to a fixed size.
            fixed_display_size = 700
            if ps_base.shape[0] != fixed_display_size or ps_base.shape[1] != fixed_display_size:
                ps_base = cv2.resize(ps_base, (fixed_display_size, fixed_display_size),
                                      interpolation=cv2.INTER_NEAREST)
            h_disp, w_disp = ps_base.shape[:2]
            # Scale: display_pixels / fft_pixels.
            display_scale_r = h_disp / h_fft
            display_scale_c = w_disp / w_fft
            # FFT center in display coords.
            center_r_disp = (h_fft / 2.0) * display_scale_r
            center_c_disp = (w_fft / 2.0) * display_scale_c

            # Draw manually-selected reflections + their Friedel-symmetric partners.
            manual_refls = st.session_state.get("click_manual_reflections") or []
            ps_display = ps_base.copy()
            for i, (r_fft, c_fft) in enumerate(manual_refls):
                r_disp = int(r_fft * display_scale_r)
                c_disp = int(c_fft * display_scale_c)
                # User click = green.
                cv2.circle(ps_display, (c_disp, r_disp), 7, (0, 255, 0), -1)
                cv2.circle(ps_display, (c_disp, r_disp), 7, (255, 255, 255), 2)
                cv2.putText(ps_display, str(i + 1), (c_disp + 10, r_disp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Friedel-symmetric partner = cyan (point-reflection through center).
                r_sym_disp = int(2 * center_r_disp - r_disp)
                c_sym_disp = int(2 * center_c_disp - c_disp)
                if 0 <= r_sym_disp < h_disp and 0 <= c_sym_disp < w_disp:
                    cv2.circle(ps_display, (c_sym_disp, r_sym_disp), 7, (255, 255, 0), -1)
                    cv2.circle(ps_display, (c_sym_disp, r_sym_disp), 7, (255, 255, 255), 2)

            st.markdown("**Click on reflections** in the power spectrum:")
            # Show the mask size info.
            ring_w = click_params.ring_mask_width_px
            tol_deg = click_params.orientation_tolerance_deg
            slide_win = click_params.sliding_window_size_px
            st.caption(f"Power spectrum: {w_disp}×{h_disp}px. FFT size: "
                       f"{h_fft}×{w_fft}px. "
                       f"Each click = one reflection (d, θ). "
                       f"**Mask per sliding window:** ring width = {ring_w:.0f}px, "
                       f"angular sector = ±{tol_deg:.0f}° (total {2*tol_deg:.0f}°), "
                       f"applied in each {slide_win}×{slide_win}px sliding FFT.")
            st.caption("🟢 green = your click. 🟡 cyan = Friedel-symmetric partner "
                       "(automatically marked for visual reference — the amplitude "
                       "map is already Friedel-invariant, so clicking the partner "
                       "is not needed, but it helps verify you picked the right "
                       "reflection.)")

            # Click callback.
            def _on_manual_refl_click() -> None:
                click = st.session_state.get("click_manual_refl_coord")
                if click is None:
                    return
                click_col = int(click["x"]) if isinstance(click, dict) else int(click[0])
                click_row = int(click["y"]) if isinstance(click, dict) else int(click[1])
                last = st.session_state.get("click_manual_refl_last_click")
                if last is not None and last[0] == click_row and last[1] == click_col:
                    return
                r_fft = click_row / display_scale_r
                c_fft = click_col / display_scale_c
                if st.session_state.click_manual_reflections is None:
                    st.session_state.click_manual_reflections = []
                st.session_state.click_manual_reflections.append((float(r_fft), float(c_fft)))
                st.session_state.click_manual_refl_last_click = (click_row, click_col)

            streamlit_image_coordinates(ps_display, key="click_manual_refl_coord",
                                         on_click=_on_manual_refl_click)

            # Buttons.
            col_clear_r, col_undo_r, col_build = st.columns(3)
            with col_clear_r:
                if st.button("Clear reflections", key="click_refl_clear"):
                    st.session_state.click_manual_reflections = []
                    st.session_state.click_manual_refl_last_click = None
                    st.rerun()
            with col_undo_r:
                if st.button("Undo last", key="click_refl_undo"):
                    if st.session_state.click_manual_reflections:
                        st.session_state.click_manual_reflections.pop()
                        st.session_state.click_manual_refl_last_click = None
                        st.rerun()
            with col_build:
                n_refls = len(manual_refls)
                if st.button(f"Build amplitude maps from {n_refls} reflection(s)",
                              key="click_refl_build", type="primary",
                              disabled=(n_refls == 0)):
                    with st.spinner("Building amplitude maps from manual reflections..."):
                        try:
                            result_manual = click_segmentation.segment_grain_from_click_with_reflections(
                                cropped_img, cropped_mask,
                                last_result.click_point[0], last_result.click_point[1],
                                ps, fft_params, click_params, sliding_data,
                                manual_refls,
                            )
                            st.session_state.click_last_result = result_manual
                            if result_manual.success:
                                if st.session_state.click_grains is None:
                                    st.session_state.click_grains = []
                                st.session_state.click_grains.append({
                                    "mask": result_manual.grain_mask,
                                    "signature": result_manual.signature,
                                    "click_point": result_manual.click_point,
                                    "area_px": int(result_manual.grain_mask.sum()),
                                    "d_spacings": result_manual.signature.d_spacings_nm,
                                    "source": "manual-reflections",
                                })
                                if result_manual.similar_grain_masks:
                                    st.session_state.click_pending_similar = result_manual.similar_grain_masks
                                else:
                                    st.session_state.click_pending_similar = None
                                # Clear reflections for the next grain.
                                st.session_state.click_manual_reflections = []
                                st.session_state.click_manual_refl_last_click = None
                        except Exception as e:
                            import traceback
                            st.error(f"Manual reflection segmentation failed: {e}")
                            st.code(traceback.format_exc())
                    st.rerun()

            # Show selected reflections table.
            if manual_refls:
                st.caption("**Selected reflections:**")
                refl_rows = []
                for i, (r_fft, c_fft) in enumerate(manual_refls):
                    local_center = max(click_params.window_size_px, click_params.zero_pad_size_px) / 2.0
                    local_avg = max(click_params.window_size_px, click_params.zero_pad_size_px)
                    dr = r_fft - local_center
                    dc = c_fft - local_center
                    r_local = float(np.sqrt(dr ** 2 + dc ** 2))
                    theta = float(np.degrees(np.arctan2(dr, dc)) % 180.0)
                    d_nm = local_avg * ps / r_local if r_local > 0 else 0.0
                    refl_rows.append({
                        "#": i + 1,
                        "row (FFT)": f"{r_fft:.1f}",
                        "col (FFT)": f"{c_fft:.1f}",
                        "r (px)": f"{r_local:.1f}",
                        "d (nm)": f"{d_nm:.4f}",
                        "θ (deg)": f"{theta:.1f}",
                    })
                st.dataframe(refl_rows, use_container_width=True, hide_index=True)

    # --- Pending similar grains (user confirmation) ---
    pending = st.session_state.get("click_pending_similar")
    if pending:
        with st.expander(f"Similar grains found ({len(pending)}) — review and add", expanded=True):
            st.caption("These are connected regions with the same d-spacing "
                       "signature as your last click. Review them and add the ones "
                       "you want.")
            col_add_all, col_skip = st.columns(2)
            with col_add_all:
                if st.button("➕ Add all similar grains", key="click_add_all"):
                    if st.session_state.click_grains is None:
                        st.session_state.click_grains = []
                    for sim_mask in pending:
                        st.session_state.click_grains.append({
                            "mask": sim_mask,
                            "signature": last_result.signature if last_result else click_segmentation.GrainSignature(),
                            "click_point": None,
                            "area_px": int(sim_mask.sum()),
                            "d_spacings": last_result.signature.d_spacings_nm if last_result else [],
                            "source": "auto-similar",
                        })
                    st.session_state.click_pending_similar = None
                    st.rerun()
            with col_skip:
                if st.button("⏭ Skip all", key="click_skip_all"):
                    st.session_state.click_pending_similar = None
                    st.rerun()
            # Show each similar grain with an Add button.
            for i, sim_mask in enumerate(pending):
                col_preview, col_btn = st.columns([3, 1])
                with col_preview:
                    # Render a small preview: cropped_img with this grain highlighted.
                    preview = cropped_img.copy()
                    if preview.ndim == 2:
                        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
                    overlay_color = np.array([0, 255, 0], dtype=np.uint8)
                    preview[sim_mask] = (preview[sim_mask].astype(np.int32) * 0.4
                                          + overlay_color.astype(np.int32) * 0.6).astype(np.uint8)
                    st.image(preview, use_container_width=True,
                              caption=f"Similar grain #{i+1}: area={int(sim_mask.sum())} px")
                with col_btn:
                    st.write("")
                    st.write("")
                    if st.button(f"➕ Add #{i+1}", key=f"click_add_sim_{i}"):
                        if st.session_state.click_grains is None:
                            st.session_state.click_grains = []
                        st.session_state.click_grains.append({
                            "mask": sim_mask,
                            "signature": last_result.signature if last_result else click_segmentation.GrainSignature(),
                            "click_point": None,
                            "area_px": int(sim_mask.sum()),
                            "d_spacings": last_result.signature.d_spacings_nm if last_result else [],
                            "source": "auto-similar",
                        })
                        # Remove this one from pending.
                        st.session_state.click_pending_similar.pop(i)
                        if not st.session_state.click_pending_similar:
                            st.session_state.click_pending_similar = None
                        st.rerun()

    # --- Action buttons: Undo / Clear / Finish ---
    st.markdown("---")
    col_undo, col_clear, col_finish = st.columns(3)
    with col_undo:
        if st.button("↩ Undo last grain", key="click_undo", disabled=(not click_grains)):
            if click_grains:
                click_grains.pop()
                st.session_state.click_grains = click_grains
                st.session_state.click_last_result = None
                st.session_state.click_pending_similar = None
                st.rerun()
    with col_clear:
        if st.button("🗑 Clear all", key="click_clear", disabled=(not click_grains)):
            st.session_state.click_grains = []
            st.session_state.click_last_result = None
            st.session_state.click_pending_similar = None
            st.session_state.click_finished = False
            st.rerun()
    with col_finish:
        if st.button("✓ Finish & compute statistics", key="click_finish",
                      type="primary", disabled=(not click_grains)):
            _finalize_click_grains(li, md, selected_layer, ps, click_grains,
                                    layer_bbox, cropped_img, cropped_mask)

    # --- Grains table ---
    if click_grains:
        st.markdown(f"### Grains found: {len(click_grains)}")
        # Group by signature for the "type" column.
        type_labels = _assign_grain_type_labels(click_grains)
        table_rows = []
        for i, g in enumerate(click_grains):
            area_px = g["area_px"]
            area_nm2 = area_px * ps * ps
            eq_diam_nm = 2 * np.sqrt(area_nm2 / np.pi)
            d_str = ", ".join(f"{d:.3f}" for d in g["d_spacings"]) if g["d_spacings"] else "—"
            table_rows.append({
                "#": i + 1,
                "type": type_labels[i],
                "source": g["source"],
                "area (px)": area_px,
                "area (nm²)": f"{area_nm2:.1f}",
                "eq. diam (nm)": f"{eq_diam_nm:.2f}",
                "d-spacings (nm)": d_str,
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Grain selection + highlight + delete.
        col_sel, col_del = st.columns([3, 1])
        with col_sel:
            grain_options = [f"Grain #{i+1} (type={type_labels[i]}, area={g['area_px']} px, "
                             f"d={2*np.sqrt(g['area_px']*ps*ps/np.pi):.1f} nm)"
                             for i, g in enumerate(click_grains)]
            selected_grain_idx = st.selectbox(
                "Select a grain to highlight", grain_options,
                key="click_grain_select",
            )
            sel_idx = grain_options.index(selected_grain_idx) if selected_grain_idx else None
        with col_del:
            st.write("")  # spacing
            st.write("")
            if st.button("🗑 Delete selected", key="click_del_grain",
                          disabled=(sel_idx is None)):
                if sel_idx is not None and sel_idx < len(click_grains):
                    click_grains.pop(sel_idx)
                    st.session_state.click_grains = click_grains
                    st.rerun()

        # Color-coded labeled image with highlight.
        labels_img = _build_click_labels_image(cropped_img, cropped_mask, click_grains,
                                                highlight_idx=sel_idx)
        st.markdown("**Color-coded grain map** (each color = a different grain; "
                    "same hue = same crystallographic type)" +
                    (f" — grain #{sel_idx+1} highlighted in yellow" if sel_idx is not None else ""))
        st.image(labels_img, use_container_width=True)


def _build_click_display_image(cropped_img, cropped_mask, click_grains):
    """Build the display image: cropped + overlay of found grains (semi-transparent colors)."""
    if cropped_img.ndim == 2:
        disp = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    else:
        disp = cropped_img.copy()
        if disp.shape[2] == 4:
            disp = cv2.cvtColor(disp, cv2.COLOR_RGBA2RGB)
    # Dim pixels outside the layer.
    disp[~cropped_mask] = (disp[~cropped_mask].astype(np.int32) * 0.4).astype(np.uint8)
    # Overlay each grain with a distinct color.
    if click_grains:
        from skimage import color as skcolor
        n = len(click_grains)
        labels = np.arange(n).reshape(-1, 1)
        colored = skcolor.label2rgb(labels, bg_label=-1, kind="overlay", alpha=1.0)
        colors = (colored[:, 0, :] * 255).astype(np.uint8)
        for i, g in enumerate(click_grains):
            mask = g["mask"]
            color = colors[i % len(colors)]
            disp[mask] = (disp[mask].astype(np.int32) * 0.5
                           + color.astype(np.int32) * 0.5).astype(np.uint8)
            # Outline.
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, contours, -1, color.tolist(), 2)
            # Mark the click point.
            if g.get("click_point") is not None:
                cy, cx = g["click_point"]
                cv2.circle(disp, (int(cx), int(cy)), 4, (255, 255, 255), -1)
                cv2.circle(disp, (int(cx), int(cy)), 4, color.tolist(), 1)
    return disp


def _build_click_labels_image(cropped_img, cropped_mask, click_grains, highlight_idx=None):
    """Build a color-coded labeled image (each grain = one label, colored by type).

    If highlight_idx is given, the corresponding grain is outlined in yellow.
    """
    if cropped_img.ndim == 2:
        bg = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    else:
        bg = cropped_img.copy()
        if bg.shape[2] == 4:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)
    if not click_grains:
        return bg
    # Build a label image.
    labels = np.zeros(cropped_img.shape[:2], dtype=np.int32)
    type_labels = _assign_grain_type_labels(click_grains)
    for i, g in enumerate(click_grains):
        # Use the type label (not the grain index) so same-type grains share a color.
        labels[g["mask"]] = type_labels[i]
    # Colorize by type.
    from skimage import color as skcolor
    n_types = max(type_labels) if type_labels else 0
    if n_types == 0:
        return bg
    colored = skcolor.label2rgb(labels, bg_label=0, kind="overlay", alpha=0.55,
                                 bg_color=(0, 0, 0), colors=None)
    colored = (colored * 255).astype(np.uint8)
    # Where there's no grain, show the original.
    no_grain = labels == 0
    colored[no_grain] = bg[no_grain]
    # Highlight the selected grain with a yellow outline.
    if highlight_idx is not None and highlight_idx < len(click_grains):
        mask = click_grains[highlight_idx]["mask"]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored, contours, -1, (255, 255, 0), 3)  # yellow
    return colored


def _assign_grain_type_labels(click_grains):
    """Group grains by their d-spacing signature; return a list of type ids (1-based)."""
    if not click_grains:
        return []
    # Greedy clustering by signature similarity.
    type_id = 0
    type_labels = [0] * len(click_grains)
    representative_sigs = []  # list of GrainSignature, one per type
    for i, g in enumerate(click_grains):
        sig = g.get("signature")
        if sig is None or not sig.d_spacings_nm:
            # No signature → own type.
            type_id += 1
            type_labels[i] = type_id
            representative_sigs.append(sig)
            continue
        matched = False
        for j, rep in enumerate(representative_sigs):
            if rep is not None and rep.d_spacings_nm and sig.matches(rep, tol_nm=0.02):
                type_labels[i] = j + 1
                matched = True
                break
        if not matched:
            type_id += 1
            type_labels[i] = type_id
            representative_sigs.append(sig)
    return type_labels


def _finalize_click_grains(li, md, selected_layer, ps, click_grains,
                             layer_bbox, cropped_img, cropped_mask):
    """Convert the click-found grains into a labeled image and compute statistics."""
    if not click_grains:
        st.warning("No grains to finalize.")
        return
    # Build the labels image (each grain = unique label).
    labels = np.zeros(cropped_img.shape[:2], dtype=np.int32)
    for i, g in enumerate(click_grains, start=1):
        labels[g["mask"]] = i
    # Compute measurements.
    grains = measurements.measure_grains(labels, ps)
    stats = measurements.compute_image_stats(
        grains, selected_layer.height_nm, selected_layer.width_nm
    )
    rmin, cmin, rmax, cmax = layer_bbox
    viz_img = li.image[rmin:rmax, cmin:cmax]
    labeled = visualization.color_label_image(labels, background=viz_img)
    overlay = visualization.boundary_overlay(viz_img, labels)
    rep = report.build_report(
        image_metadata=md,
        layer_mask=selected_layer,
        segmentation_params=st.session_state.get("_click_params", click_segmentation.ClickSegmentationParams()),
        stats=stats,
        grains=grains,
        labeled_image_rgb=labeled,
    )
    st.session_state.analysis_result = segmentation.SegmentationResult(
        labels=labels,
        boundaries=np.zeros_like(labels, dtype=bool),
        n_grains_raw=len(click_grains),
        n_grains_filtered=len(click_grains),
        gradient=np.zeros(viz_img.shape, dtype=np.float32),
        preprocessed=viz_img,
        gradient_threshold_value=0.0,
        bbox=layer_bbox,
    )
    st.session_state.analysis_grains = grains
    st.session_state.analysis_stats = stats
    st.session_state.analysis_labeled_image = labeled
    st.session_state.analysis_overlay_image = overlay
    st.session_state.analysis_report = rep
    st.session_state.fft_result = None
    st.session_state.st_result = None
    st.session_state.click_finished = True
    st.success(f"Finalized {len(click_grains)} grains. Scroll down for statistics.")
    st.rerun()


def _render_manual_reflection_selector(fft_res, md, selected_layer, ps) -> None:
    """Render the manual reflection selection UI (click on power spectrum).

    Two-column layout:
      - Left: power spectrum (clickable). Each click adds a reflection + its
        Friedel-symmetry partner. After each click, an **amplitude preview** is
        computed (inverse-FFT of the ring at the click radius) and shown on
        the right column — this lets the user verify that the clicked
        reflection corresponds to a real d-spacing visible in the image.
      - Right: amplitude preview overlay + d-spacing info for the last click.

    When the user clicks "Re-segment", the FFT segmentation is re-run with the
    manually-selected reflections.
    """
    from streamlit_image_coordinates import streamlit_image_coordinates
    from src.polygon import crop_to_mask_bbox

    # --- Get cached FFT data (must match what re-segmentation will use). ---
    cached = st.session_state.get("fft_cached")
    if cached is not None:
        fft_shifted, log_power, fft_scale, small_shape = cached
    else:
        log_power = fft_res.power_spectrum
        fft_shifted = None
        fft_scale = float(getattr(fft_res, 'scale', 1.0) or 1.0)
        small_shape = log_power.shape
    h_fft, w_fft = log_power.shape

    # --- Cropped image + mask (for the amplitude preview overlay). ---
    full_img = st.session_state.loaded_image.image
    cropped_img, cropped_mask, layer_bbox = crop_to_mask_bbox(
        full_img, selected_layer.mask, padding=2
    )

    # --- Build the power-spectrum display image. ---
    max_display = 2048
    ps_base = fft_segmentation.render_power_spectrum(
        log_power, None, None, max_dim=max_display
    )
    h_disp, w_disp = ps_base.shape[:2]
    display_scale_r = h_disp / h_fft
    display_scale_c = w_disp / w_fft

    # --- Draw manually-selected reflections on the display image. ---
    manual_refls = st.session_state.manual_reflections or []
    ps_display = ps_base.copy()
    for i, (r_fft, c_fft) in enumerate(manual_refls):
        r_disp = int(r_fft * display_scale_r)
        c_disp = int(c_fft * display_scale_c)
        if i % 2 == 0:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        cv2.circle(ps_display, (c_disp, r_disp), 8, color, -1)
        cv2.circle(ps_display, (c_disp, r_disp), 8, (255, 255, 255), 2)
        cv2.putText(ps_display, str(i + 1), (c_disp + 10, r_disp - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fft_params = st.session_state.get("_fft_params", fft_segmentation.FFTSegmentationParams())

    st.markdown("**Click on reflections** in the power spectrum (left). "
                "The **amplitude preview** (right) shows where in the image that "
                "d-spacing is present — bright regions = real signal, dim = noise.")
    st.caption(f"Power spectrum: {w_disp}×{h_disp}px. "
               f"Each click adds a reflection + its Friedel-symmetry partner "
               f"(green = your click, cyan = auto-symmetric). "
               f"Use the 1D radial spectrum above as a guide: peaks there = "
               f"d-spacings worth clicking on.")

    # Two-column layout: power spectrum (clickable) + amplitude preview.
    col_ps, col_prev = st.columns([1, 1])

    # --- Click callback: append reflection, compute amplitude preview. ---
    def _on_refl_click() -> None:
        click = st.session_state.get("refl_click_coord")
        if click is None:
            return
        click_col = int(click["x"]) if isinstance(click, dict) else int(click[0])
        click_row = int(click["y"]) if isinstance(click, dict) else int(click[1])
        # Deduplicate.
        last = st.session_state.get("manual_refl_last_click")
        if last is not None and last[0] == click_row and last[1] == click_col:
            return
        # Convert display coords → small-FFT coords.
        r_fft = click_row / display_scale_r
        c_fft = click_col / display_scale_c

        # Compute the amplitude preview for this click.
        # Use the cached fft_shifted if available; otherwise skip preview.
        preview_data = None
        local_fft = fft_shifted
        if local_fft is None:
            # Fallback: recompute FFT just for the preview.
            try:
                layer_mean_fb = float(cropped_img[cropped_mask].mean())
                fft_input_fb = cropped_img.copy()
                fft_input_fb[~cropped_mask] = int(layer_mean_fb)
                local_fft, _, _, _ = fft_segmentation.compute_fft_and_spectrum(fft_input_fb, fft_params)
            except Exception:
                local_fft = None

        if local_fft is not None:
            try:
                amp_map, radius_px = fft_segmentation.compute_amplitude_preview(
                    local_fft, float(r_fft), float(c_fft), fft_params,
                    target_shape=cropped_img.shape,
                )
                if radius_px > 0:
                    overlay = fft_segmentation.render_amplitude_overlay(
                        cropped_img, amp_map, layer_mask=cropped_mask, alpha=0.55
                    )
                    d_nm = fft_segmentation.compute_d_spacing_for_radius(
                        radius_px, local_fft.shape, ps, fft_scale
                    )
                    preview_data = (overlay, d_nm, radius_px)
                    st.session_state.manual_refl_last_radius = float(radius_px)
            except Exception:
                preview_data = None

            # --- Reflection quality assessment (Gaussian fit + Scherrer) ---
            try:
                quality = fft_segmentation.evaluate_reflection_quality(
                    log_power, float(r_fft), float(c_fft),
                    log_power.shape, ps, fft_scale, fft_params,
                )
                if st.session_state.manual_refl_quality is None:
                    st.session_state.manual_refl_quality = []
                st.session_state.manual_refl_quality.append(quality)
                st.session_state.manual_refl_last_quality = quality
            except Exception:
                st.session_state.manual_refl_last_quality = None
        else:
            st.session_state.manual_refl_last_quality = None

        st.session_state.manual_refl_preview = preview_data

        # Append the click + its Friedel-symmetry partner.
        st.session_state.manual_reflections.append((float(r_fft), float(c_fft)))
        center_r = h_fft / 2.0
        center_c = w_fft / 2.0
        r_sym = 2.0 * center_r - r_fft
        c_sym = 2.0 * center_c - c_fft
        if 0 <= r_sym < h_fft and 0 <= c_sym < w_fft:
            st.session_state.manual_reflections.append((float(r_sym), float(c_sym)))
        st.session_state.manual_refl_last_click = (click_row, click_col)

    # --- Render the two columns. ---
    with col_ps:
        streamlit_image_coordinates(ps_display, key="refl_click_coord", on_click=_on_refl_click)

    with col_prev:
        st.markdown("**Amplitude preview** (last click)")
        preview = st.session_state.get("manual_refl_preview")
        quality = st.session_state.get("manual_refl_last_quality")
        if preview is not None:
            overlay_img, d_nm, radius_px = preview
            st.image(overlay_img, use_container_width=True)
            st.markdown(
                f"d-spacing = **{d_nm:.4f} nm**  "
                f"·  radius = {radius_px:.0f} px"
            )
            st.caption("Color scale (JET, low→high): dark blue → cyan → green → yellow → red. "
                       "Red/yellow = pixels whose lattice HAS this d-spacing (real grains "
                       "with this plane family). Cyan/blue = low amplitude (this d-spacing "
                       "is NOT present here — either a different orientation, amorphous "
                       "matrix, or noise). If you see clear red/yellow grains on a "
                       "blue background → good click. If everything is uniformly cyan → "
                       "likely noise; undo and try again.")

            # --- Reflection quality (Gaussian fit + Scherrer estimate) ---
            if quality is not None:
                st.markdown("**Reflection quality** (2D Gaussian fit)")
                # Status badge with color.
                status_colors = {
                    "good": ("🟢 GOOD", "green"),
                    "borderline": ("🟡 BORDERLINE", "orange"),
                    "noise": ("🔴 NOISE", "red"),
                }
                badge_text, badge_color = status_colors.get(
                    quality.status, ("❓ UNKNOWN", "gray")
                )
                st.markdown(
                    f"<span style='background-color:{badge_color}; color:white; "
                    f"padding:2px 10px; border-radius:6px; font-weight:bold;'>"
                    f"{badge_text}</span>",
                    unsafe_allow_html=True,
                )
                # Numeric metrics in 2 columns.
                qcol1, qcol2 = st.columns(2)
                with qcol1:
                    st.metric("σ (FWHM/2.355)", f"{quality.sigma_mean_px:.2f} px")
                    st.metric("SNR", f"{quality.snr:.1f}")
                with qcol2:
                    if np.isfinite(quality.coherent_size_nm):
                        if quality.coherent_size_nm >= 100:
                            st.metric("D (Scherrer)", f"{quality.coherent_size_nm:.0f} nm")
                        elif quality.coherent_size_nm >= 10:
                            st.metric("D (Scherrer)", f"{quality.coherent_size_nm:.1f} nm")
                        else:
                            st.metric("D (Scherrer)", f"{quality.coherent_size_nm:.2f} nm")
                    else:
                        st.metric("D (Scherrer)", "∞ (σ→0)")
                    st.metric("d-spacing", f"{quality.d_spacing_nm:.4f} nm")
                # Status message (human-readable).
                st.caption(quality.message)
                st.caption(
                    "D = K / FWHM_g is the **Scherrer coherent-domain size** — "
                    "the average size of the region over which the lattice is "
                    "coherent. For real grains this should match the grain size "
                    "seen in the image; if D is much smaller, the peak is "
                    "noise-broadened; if D is much larger than the image, the "
                    "peak is suspiciously narrow (single-pixel artifact)."
                )
        else:
            st.info("Click on a reflection in the power spectrum to see its "
                    "amplitude preview + quality assessment here.")

    # --- Buttons: Clear / Undo / Re-segment. ---
    col_clear, col_undo, col_run = st.columns(3)
    with col_clear:
        if st.button("Clear", key="refl_clear"):
            st.session_state.manual_reflections = []
            st.session_state.manual_refl_last_click = None
            st.session_state.manual_refl_preview = None
            st.session_state.manual_refl_last_radius = None
            st.session_state.manual_refl_quality = None
            st.session_state.manual_refl_last_quality = None
            st.rerun()
    with col_undo:
        if st.button("Undo last", key="refl_undo"):
            if st.session_state.manual_reflections:
                st.session_state.manual_reflections.pop()  # symmetry partner
                if st.session_state.manual_reflections:
                    st.session_state.manual_reflections.pop()  # user click
                st.session_state.manual_refl_last_click = None
                st.session_state.manual_refl_preview = None
                st.session_state.manual_refl_last_radius = None
                # Also pop the quality assessment for the undone click.
                if st.session_state.manual_refl_quality:
                    st.session_state.manual_refl_quality.pop()
                    st.session_state.manual_refl_last_quality = (
                        st.session_state.manual_refl_quality[-1]
                        if st.session_state.manual_refl_quality else None
                    )
                st.rerun()
    with col_run:
        # Auto-reject option (only meaningful if we have quality assessments).
        qualities = st.session_state.get("manual_refl_quality") or []
        n_noise = sum(1 for q in qualities if q.status == "noise")
        auto_reject = st.checkbox(
            "Auto-reject noise clicks",
            value=True,
            key="refl_auto_reject",
            help=f"Drop clicks classified as 'noise' before re-segmentation. "
                 f"Currently {n_noise}/{len(qualities)} clicks are noise.",
            disabled=(len(qualities) == 0),
        )

        if len(manual_refls) >= 2:
            if st.button("Re-segment with these reflections", key="refl_run", type="primary"):
                with st.spinner("Re-running FFT segmentation with manual reflections..."):
                    try:
                        # Get cached FFT data (recompute if missing).
                        cached = st.session_state.get("fft_cached")
                        if cached is None:
                            layer_mean_fb = float(cropped_img[cropped_mask].mean())
                            fft_input_fb = cropped_img.copy()
                            fft_input_fb[~cropped_mask] = int(layer_mean_fb)
                            fft_shifted_r, log_power_r, scale_r, small_shape_r = \
                                fft_segmentation.compute_fft_and_spectrum(fft_input_fb, fft_params)
                            st.session_state.fft_cached = (fft_shifted_r, log_power_r, scale_r, small_shape_r)
                            cached = st.session_state.fft_cached

                        fft_shifted_r, log_power_r, scale_r, small_shape_r = cached

                        # Filter reflections: if auto_reject is on, drop user-clicked
                        # reflections whose status == "noise" (and their symmetry partners).
                        # manual_reflections is a flat list of pairs:
                        # [click1, sym1, click2, sym2, ...] (each user click adds 2).
                        # qualities has one entry per user click (not per pair).
                        if auto_reject and qualities:
                            filtered = []
                            kept = 0
                            rejected = 0
                            for i in range(0, len(manual_refls), 2):
                                # The qualities list may be shorter than manual_reflections//2
                                # if some clicks didn't get a quality assessment.
                                qidx = i // 2
                                if qidx < len(qualities):
                                    q = qualities[qidx]
                                    if q.status == "noise":
                                        rejected += 1
                                        continue  # skip this click + its symmetry partner
                                filtered.append(manual_refls[i])  # user click
                                if i + 1 < len(manual_refls):
                                    filtered.append(manual_refls[i + 1])  # symmetry partner
                                kept += 1
                            refl_coords = np.array(filtered, dtype=np.float32)
                            if len(refl_coords) < 2:
                                st.error(
                                    f"After rejecting {rejected} noise click(s), only "
                                    f"{kept} click(s) remain — need ≥1. Either disable "
                                    f"'Auto-reject' or click on more reflections."
                                )
                                return
                        else:
                            refl_coords = np.array(manual_refls, dtype=np.float32)
                            rejected = 0

                        new_fft_result = fft_segmentation.segment_grains_fft_with_reflections(
                            fft_shifted_r, log_power_r, refl_coords,
                            cropped_img.shape,
                            scale_r,
                            ps,
                            fft_params,
                        )

                        new_fft_result.labels[~cropped_mask] = 0

                        grains = measurements.measure_grains(new_fft_result.labels, ps)
                        stats = measurements.compute_image_stats(
                            grains, selected_layer.height_nm, selected_layer.width_nm
                        )
                        rmin, cmin, rmax, cmax = layer_bbox
                        viz_img = full_img[rmin:rmax, cmin:cmax]
                        labeled = visualization.color_label_image(
                            new_fft_result.labels, background=viz_img
                        )
                        overlay_viz = visualization.boundary_overlay(viz_img, new_fft_result.labels)
                        rep = report.build_report(
                            image_metadata=md,
                            layer_mask=selected_layer,
                            segmentation_params=fft_params,
                            stats=stats,
                            grains=grains,
                            labeled_image_rgb=labeled,
                        )
                        st.session_state.analysis_result = segmentation.SegmentationResult(
                            labels=new_fft_result.labels,
                            boundaries=np.zeros_like(new_fft_result.labels, dtype=bool),
                            n_grains_raw=new_fft_result.n_grains,
                            n_grains_filtered=new_fft_result.n_grains,
                            gradient=np.zeros(viz_img.shape, dtype=np.float32),
                            preprocessed=viz_img,
                            gradient_threshold_value=0.0,
                            bbox=layer_bbox,
                        )
                        st.session_state.analysis_grains = grains
                        st.session_state.analysis_stats = stats
                        st.session_state.analysis_labeled_image = labeled
                        st.session_state.analysis_overlay_image = overlay_viz
                        st.session_state.analysis_report = rep
                        st.session_state.fft_result = new_fft_result
                    except Exception as e:
                        st.error(f"Re-segmentation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        return
                msg = f"Re-segmented with {len(refl_coords)} reflections: " \
                      f"{st.session_state.analysis_stats.n_grains} grains found."
                if auto_reject and rejected > 0:
                    msg += f" ({rejected} noise click(s) auto-rejected)"
                st.success(msg)
                st.rerun()
        else:
            st.caption(f"Need ≥2 reflections (have {len(manual_refls)}).")

    # --- Click history table. ---
    if manual_refls:
        with st.expander(f"Click history ({len(manual_refls)} reflections)", expanded=False):
            qualities = st.session_state.get("manual_refl_quality") or []
            hist_rows = []
            for i, (r_fft, c_fft) in enumerate(manual_refls):
                role = "your click" if i % 2 == 0 else "auto-symmetry"
                radius_px = float(np.sqrt(
                    (r_fft - h_fft / 2.0) ** 2 + (c_fft - w_fft / 2.0) ** 2
                ))
                d_nm = fft_segmentation.compute_d_spacing_for_radius(
                    radius_px, (h_fft, w_fft), ps, fft_scale
                ) if radius_px > 0 else 0.0
                row = {
                    "#": i + 1,
                    "role": role,
                    "row (FFT)": f"{r_fft:.1f}",
                    "col (FFT)": f"{c_fft:.1f}",
                    "radius (px)": f"{radius_px:.1f}",
                    "d (nm)": f"{d_nm:.4f}",
                    "σ (px)": "—",
                    "SNR": "—",
                    "D (nm)": "—",
                    "status": "—",
                }
                # Quality is only attached to user-clicked reflections (even indices).
                if i % 2 == 0:
                    qidx = i // 2
                    if qidx < len(qualities):
                        q = qualities[qidx]
                        row["σ (px)"] = f"{q.sigma_mean_px:.2f}"
                        row["SNR"] = f"{q.snr:.1f}"
                        if np.isfinite(q.coherent_size_nm):
                            if q.coherent_size_nm >= 100:
                                row["D (nm)"] = f"{q.coherent_size_nm:.0f}"
                            elif q.coherent_size_nm >= 10:
                                row["D (nm)"] = f"{q.coherent_size_nm:.1f}"
                            else:
                                row["D (nm)"] = f"{q.coherent_size_nm:.2f}"
                        else:
                            row["D (nm)"] = "∞"
                        status_emoji = {
                            "good": "🟢 good",
                            "borderline": "🟡 borderline",
                            "noise": "🔴 noise",
                        }.get(q.status, q.status)
                        row["status"] = status_emoji
                hist_rows.append(row)
            st.dataframe(hist_rows, use_container_width=True, hide_index=True)
            st.caption("σ = Gaussian width (FWHM = 2.355·σ). "
                       "SNR = peak amplitude / background. "
                       "D = Scherrer coherent-domain size (K = 0.94). "
                       "Auto-rejection drops rows with status 🔴 noise before re-segmentation.")


def _render_analysis_results(md, selected_layer, ps) -> None:
    """Render the analysis results section (step 3)."""
    if st.session_state.analysis_result is None:
        return
    st.subheader("2. Grain Analysis Results")
    result = st.session_state.analysis_result
    grains = st.session_state.analysis_grains
    stats = st.session_state.analysis_stats
    labeled = st.session_state.analysis_labeled_image
    overlay = st.session_state.analysis_overlay_image
    rep = st.session_state.analysis_report
    fft_params = st.session_state.get("_fft_params", fft_segmentation.FFTSegmentationParams())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grains (raw)", result.n_grains_raw)
    c2.metric("Grains (filtered)", result.n_grains_filtered)
    c3.metric("Total grain area", f"{stats.total_grain_area_nm2:.1f} nm²")
    c4.metric("Coverage", f"{stats.coverage_fraction*100:.1f}%")

    col_lab, col_ov = st.columns(2)
    with col_lab:
        st.markdown("**Colour-labeled grain map**")
        st.image(_downscale_for_preview(labeled, max_dim=1500), use_container_width=True)
    with col_ov:
        st.markdown("**Grain boundaries on layer**")
        st.image(_downscale_for_preview(overlay, max_dim=1500), use_container_width=True)

    # --- Grain selection + highlight + delete (for automatic methods) ---
    if grains and len(grains) > 0:
        with st.expander(f"Grain review ({len(grains)} grains — select, highlight, delete)",
                          expanded=False):
            # Build grain options.
            grain_options = []
            for i, g in enumerate(grains):
                d_nm = g.equivalent_diameter_nm
                area_nm2 = g.area_nm2
                grain_options.append(
                    f"Grain #{i+1} (area={area_nm2:.0f} nm², d={d_nm:.1f} nm)"
                )
            col_sel, col_del = st.columns([3, 1])
            with col_sel:
                selected_grain = st.selectbox(
                    "Select a grain to highlight on the map",
                    grain_options,
                    key="auto_grain_select",
                )
                sel_idx = grain_options.index(selected_grain) if selected_grain else None
            with col_del:
                st.write("")  # spacing
                st.write("")
                if st.button("🗑 Delete selected grain", key="auto_del_grain",
                              disabled=(sel_idx is None)):
                    # Delete the grain: set its label to 0 in the labels array.
                    if sel_idx is not None and sel_idx < len(grains):
                        grain_label = grains[sel_idx].label  # ACTUAL label (may have gaps)
                        result.labels[result.labels == grain_label] = 0
                        # Recompute measurements + stats.
                        new_grains = measurements.measure_grains(result.labels, ps)
                        new_stats = measurements.compute_image_stats(
                            new_grains, selected_layer.height_nm, selected_layer.width_nm
                        )
                        rmin, cmin, rmax, cmax = result.bbox
                        li = st.session_state.loaded_image
                        viz_img = li.image[rmin:rmax, cmin:cmax]
                        new_labeled = visualization.color_label_image(
                            result.labels, background=viz_img
                        )
                        new_overlay = visualization.boundary_overlay(viz_img, result.labels)
                        fft_params = st.session_state.get("_fft_params", _DictParams({"method": "auto"}))
                        new_rep = report.build_report(
                            image_metadata=md,
                            layer_mask=selected_layer,
                            segmentation_params=fft_params,
                            stats=new_stats,
                            grains=new_grains,
                            labeled_image_rgb=new_labeled,
                        )
                        st.session_state.analysis_result = segmentation.SegmentationResult(
                            labels=result.labels,
                            boundaries=np.zeros_like(result.labels, dtype=bool),
                            n_grains_raw=result.n_grains_raw,
                            n_grains_filtered=int(result.labels.max()) if result.labels.max() > 0 else 0,
                            gradient=result.gradient,
                            preprocessed=result.preprocessed,
                            gradient_threshold_value=result.gradient_threshold_value,
                            bbox=result.bbox,
                        )
                        st.session_state.analysis_grains = new_grains
                        st.session_state.analysis_stats = new_stats
                        st.session_state.analysis_labeled_image = new_labeled
                        st.session_state.analysis_overlay_image = new_overlay
                        st.session_state.analysis_report = new_rep
                        st.rerun()

            # Show the labeled image with the selected grain highlighted.
            if sel_idx is not None and sel_idx < len(grains):
                grain_label = grains[sel_idx].label  # ACTUAL label (may have gaps)
                highlight_img = labeled.copy()
                # Find the grain mask and outline it in yellow.
                grain_mask = (result.labels == grain_label).astype(np.uint8)
                contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(highlight_img, contours, -1, (255, 255, 0), 3)
                st.markdown(f"**Grain #{sel_idx+1} highlighted** (yellow outline)")
                st.image(_downscale_for_preview(highlight_img, max_dim=1500),
                          use_container_width=True)
                # Show grain details.
                g = grains[sel_idx]
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Area", f"{g.area_nm2:.1f} nm²")
                dc2.metric("Eq. diameter", f"{g.equivalent_diameter_nm:.2f} nm")
                dc3.metric("Perimeter", f"{g.perimeter_nm:.1f} nm")

    # Structure-tensor-specific visualizations.
    st_res_viz = st.session_state.get("st_result")
    if st_res_viz is not None:
        with st.expander("Structure tensor analysis (orientation + coherence maps)", expanded=False):
            st.markdown("**Orientation map** (hue = lattice-fringe direction, brightness = coherence)")
            orient_rgb = structure_tensor_segmentation.colorize_orientation_map(
                st_res_viz.orientation_map, st_res_viz.coherence_map,
                st.session_state.get("_st_params", structure_tensor_segmentation.StructureTensorParams()).coherence_threshold
            )
            st.image(orient_rgb, use_container_width=True)
            st.caption(f"Each color = a different crystallographic orientation. "
                       f"Dim pixels = low coherence (amorphous/noise).")
            st.markdown("**Coherence map** (bright = high anisotropy = crystalline)")
            coh_img = structure_tensor_segmentation.colorize_coherence_map(st_res_viz.coherence_map)
            st.image(coh_img, use_container_width=True)
            st.caption(f"Coherence range: {st_res_viz.coherence_map.min():.3f} – {st_res_viz.coherence_map.max():.3f}")

    # FFT-specific visualizations (only if FFT method was used).
    fft_res = st.session_state.get("fft_result")
    if fft_res is not None:
        with st.expander("FFT analysis (power spectrum + orientation map)", expanded=False):
            # --- Radial spectrum (1D) — shows which d-spacings are actually present.
            # This is the key diagnostic plot: peaks = d-spacings. Much easier
            # to interpret than the 2D power spectrum for noisy TEM data.
            if (getattr(fft_res, 'radial_radii', None) is not None
                    and getattr(fft_res, 'radial_intensities', None) is not None):
                st.markdown("**1D radial power spectrum** "
                            "(peaks = d-spacings present in the layer)")
                st.caption("Interactive plot — hover over the curve to see radius + "
                           "d-spacing + power; hover over a red peak marker to see "
                           "the d-spacing of that peak. Each peak = one d-spacing "
                           "present in the layer. Use this as a guide: if you see "
                           "a peak here, look for the corresponding ring in the 2D "
                           "spectrum below.")
                fft_scale = float(getattr(fft_res, 'scale', 1.0) or 1.0)
                # The clicked_radius is the LAST manual click (for the manual mode plot).
                clicked_radius = st.session_state.get("manual_refl_last_radius")
                # Pass reference d-spacing bands (if any) so they show as green
                # shaded regions on the plot.
                ref_bands = getattr(fft_res, 'reference_d_spacings_used', None) or []
                ref_tol = float(getattr(fft_params, 'reference_tolerance_pct', 0.0) or 0.0)
                radial_fig = fft_segmentation.render_radial_spectrum_plotly(
                    fft_res.radial_radii, fft_res.radial_intensities,
                    fft_res.peak_radii, fft_res.power_spectrum.shape,
                    ps, fft_scale,
                    clicked_radius=clicked_radius,
                    reference_d_spacings=ref_bands,
                    reference_tolerance_pct=ref_tol,
                )
                st.plotly_chart(radial_fig, use_container_width=True)
                if ref_bands:
                    st.caption("🟢 Green shaded bands = your reference d-spacing ranges "
                               "(mode B). The detector only looks inside these bands.")
                if fft_res.peak_radii:
                    st.caption(f"Auto-detected peaks: {len(fft_res.peak_radii)} → "
                               f"d-spacings: " +
                               ", ".join(f"{fft_segmentation.compute_d_spacing_for_radius(r, fft_res.power_spectrum.shape, ps, fft_scale):.3f} nm"
                                         for r in fft_res.peak_radii[:8])
                               + ("…" if len(fft_res.peak_radii) > 8 else ""))

            # Power spectrum — full width for visibility.
            st.markdown("**2D power spectrum** (colored dots = detected reflections, colored by cluster)")
            ps_img = fft_segmentation.render_power_spectrum(
                fft_res.power_spectrum, fft_res.reflection_coords, fft_res.cluster_labels,
                max_dim=2048,
            )
            st.image(ps_img, use_container_width=True)
            st.caption(f"{len(fft_res.reflection_coords)} reflections → "
                       f"{fft_res.n_clusters} clusters (d-spacings)")

            # Show d-spacings table.
            if hasattr(fft_res, 'd_spacings_nm') and fft_res.d_spacings_nm:
                # Sort by d-spacing, but keep labels in sync.
                labels_list = getattr(fft_res, 'd_spacing_labels', []) or []
                paired = list(zip(fft_res.d_spacings_nm, labels_list))
                paired.sort(key=lambda x: x[0])
                has_labels = any(lbl for _, lbl in paired)
                mode_label = ("(reference mode: detection restricted to known plane families)"
                              if getattr(fft_res, 'reference_d_spacings_used', None) else
                              "(no a priori material knowledge required)")
                st.markdown(f"**Detected d-spacings** {mode_label}")
                dsp_data = []
                for i, (d, lbl) in enumerate(paired):
                    row = {"#": i + 1, "d-spacing (nm)": f"{d:.4f}"}
                    if has_labels:
                        row["label"] = lbl or "—"
                    dsp_data.append(row)
                st.dataframe(dsp_data, use_container_width=True, hide_index=True)
                if not has_labels:
                    st.caption("Compare with known values for your material. "
                               "For Ge: d₁₁₁=0.327 nm, d₂₂₀=0.200 nm, d₃₁₁=0.171 nm. "
                               "For Si: d₁₁₁=0.314 nm, d₂₂₀=0.192 nm, d₃₁₁=0.164 nm. "
                               "Use 'Reference d-spacings' in the sidebar to attach labels "
                               "automatically.")

            # --- Manual reflection selection ---
            st.markdown("---")
            st.markdown("**Manual reflection selection**")
            st.caption("If auto-detected reflections are wrong (noise detected as reflections), "
                       "you can manually click on the real reflections in the power spectrum below, "
                       "then re-run segmentation with your selection.")

            col_auto, col_manual = st.columns(2)
            with col_auto:
                if st.button("Use auto-detected reflections", key="fft_use_auto"):
                    st.session_state.manual_refl_mode = False
                    st.session_state.manual_reflections = None
                    st.rerun()
            with col_manual:
                if st.button("Select reflections manually", key="fft_use_manual", type="primary"):
                    st.session_state.manual_refl_mode = True
                    st.session_state.manual_reflections = []
                    st.session_state.manual_refl_last_click = None
                    st.rerun()

            if st.session_state.manual_refl_mode:
                _render_manual_reflection_selector(fft_res, md, selected_layer, ps)

    st.markdown("**Summary statistics**")
    stat_rows = []
    for name, s in [
        ("Area (nm²)", stats.area_stats),
        ("Equivalent diameter (nm)", stats.eq_diameter_stats),
        ("Perimeter (nm)", stats.perimeter_stats),
        ("Circularity", stats.circularity_stats),
        ("Aspect ratio (minor/major)", stats.aspect_ratio_stats),
        ("Orientation (deg)", stats.orientation_stats),
    ]:
        stat_rows.append({
            "Metric": name, "N": s.n,
            "Mean": f"{s.mean:.3f}", "Std": f"{s.std:.3f}",
            "Median": f"{s.median:.3f}", "Q1": f"{s.q1:.3f}", "Q3": f"{s.q3:.3f}",
            "IQR": f"{s.iqr:.3f}", "Min": f"{s.min:.3f}", "Max": f"{s.max:.3f}",
        })
    st.dataframe(stat_rows, use_container_width=True, hide_index=True)

    if stats.lognormal_fit is not None:
        lf = stats.lognormal_fit
        st.markdown("**Lognormal fit on equivalent diameter**  "
                    f"(μ = {lf.mu:.3f}, σ = {lf.sigma:.3f}, "
                    f"median = {lf.median:.2f} nm, mode = {lf.mode:.2f} nm, "
                    f"mean = {lf.mean:.2f} nm, std = {lf.std:.2f} nm)")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        diams = np.array([g.equivalent_diameter_nm for g in grains])
        hist_fig1 = visualization.plot_histogram(
            diams, "nm", "Equivalent diameter distribution",
            bins=min(20, max(5, len(grains) // 2)),
            lognormal_fit=stats.lognormal_fit,
            xlabel="Equivalent diameter (nm)",
        )
        st.pyplot(hist_fig1, use_container_width=True)
    with col_h2:
        areas = np.array([g.area_nm2 for g in grains])
        hist_fig2 = visualization.plot_histogram(
            areas, "nm²", "Grain area distribution",
            bins=min(20, max(5, len(grains) // 2)),
            color="darkorange", xlabel="Area (nm²)",
        )
        st.pyplot(hist_fig2, use_container_width=True)

    if result.removed_labels:
        with st.expander(f"Removed grains ({len(result.removed_labels)})"):
            reasons: dict[str, int] = {}
            for r in result.removal_reasons.values():
                key = r.split(" ")[0]
                reasons[key] = reasons.get(key, 0) + 1
            st.write(reasons)

    st.markdown("---")
    st.subheader("Download Report")
    rep_bytes = report.serialize_report(rep)
    fname = report.suggested_report_filename(md.file_name)
    st.download_button(
        label=f"Download JSON report ({len(rep_bytes)/1024:.0f} KB)",
        data=rep_bytes, file_name=fname, mime="application/json", type="primary",
    )
    st.caption(
        "The JSON report is self-contained — per-grain measurements, statistics, "
        "segmentation parameters, layer polygon, and the labeled image (base64 PNG). "
        "Drop several into the **Batch Statistics** tab to pool grain sizes."
    )


# === Batch Statistics tab ===

def render_batch_tab() -> None:
    st.title("Batch Statistics")
    st.caption(
        "Aggregate grain-size statistics across several per-image JSON reports. "
        "Each report was produced by the Single Image tab and is self-contained."
    )

    with st.sidebar.expander("Upload Reports", expanded=True):
        batch_mode = st.radio(
            "Upload method",
            ["Multiple JSON files", "ZIP archive"],
            index=0, key="batch_mode_radio",
        )
        if batch_mode == "Multiple JSON files":
            uploaded_files = st.file_uploader(
                "Select JSON reports", type=["json"],
                accept_multiple_files=True, label_visibility="collapsed", key="batch_files",
            )
            uploaded_zip = None
        else:
            uploaded_zip = st.file_uploader(
                "Select ZIP archive", type=["zip"],
                accept_multiple_files=False, label_visibility="collapsed", key="batch_zip",
            )
            uploaded_files = None

    has_input = (batch_mode == "Multiple JSON files" and uploaded_files) or \
                (batch_mode == "ZIP archive" and uploaded_zip is not None)
    if not has_input:
        st.info("Upload JSON reports (or a ZIP) in the sidebar to begin.")
        st.markdown("### What you get")
        st.markdown(
            "* **Per-image table** — one row per image with pixel size, layer "
            "source/area, grain count, mean equivalent diameter, lognormal μ/σ."
            "\n* **Pooled histogram** of equivalent diameter with a lognormal fit."
            "\n* **Pooled statistics** — total grain count, total analysed layer area."
            "\n* **Combined CSV** — one row per grain from all images."
        )
        return

    reports: list[dict[str, Any]] = []
    try:
        if batch_mode == "Multiple JSON files":
            for f in uploaded_files:
                reports.append(io_utils.load_report_from_bytes(f.getvalue()))
        else:
            with zipfile.ZipFile(uploaded_zip) as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        reports.append(io_utils.load_report_from_bytes(zf.read(name)))
    except Exception as e:
        st.error(f"Failed to load reports: {e}")
        return

    if not reports:
        st.warning("No JSON reports found in the upload.")
        return

    st.success(f"Loaded {len(reports)} reports.")

    with st.spinner("Aggregating..."):
        batch_result = batch_mod.aggregate_reports(reports)

    st.subheader("Per-image Summary")
    per_image_data = []
    for r in batch_result.per_image:
        per_image_data.append({
            "File": r.file_name,
            "Pixel (nm/px)": f"{r.pixel_size_nm:.4f}",
            "Mag": f"{r.magnification:g}×" if r.magnification else "—",
            "Layer src": r.layer_source or "—",
            "Layer area (nm²)": f"{r.layer_area_nm2:.0f}",
            "N grains": r.n_grains,
            "Mean d (nm)": f"{r.mean_eq_diameter_nm:.2f}",
            "Std d (nm)": f"{r.std_eq_diameter_nm:.2f}",
            "Median d (nm)": f"{r.median_eq_diameter_nm:.2f}",
            "Coverage %": f"{r.coverage_fraction*100:.1f}",
            "LogN μ": f"{r.lognormal_mu:.3f}" if r.lognormal_mu is not None else "—",
            "LogN σ": f"{r.lognormal_sigma:.3f}" if r.lognormal_sigma is not None else "—",
        })
    st.dataframe(per_image_data, use_container_width=True, hide_index=True)

    st.subheader("Pooled Statistics")
    pstats = batch_result.pooled_stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images", batch_result.n_reports)
    c2.metric("Total grains", batch_result.total_grain_count)
    c3.metric("Total layer area", f"{batch_result.total_layer_area_nm2:.0f} nm²")
    c4.metric("Pooled mean d", f"{pstats.eq_diameter_stats.mean:.2f} nm")
    c5.metric("Pooled std d", f"{pstats.eq_diameter_stats.std:.2f} nm")

    if pstats.lognormal_fit is not None:
        lf = pstats.lognormal_fit
        st.markdown(f"**Pooled lognormal fit on equivalent diameter:**  "
                    f"μ = {lf.mu:.3f}, σ = {lf.sigma:.3f}, "
                    f"median = {lf.median:.2f} nm, mode = {lf.mode:.2f} nm, "
                    f"mean = {lf.mean:.2f} nm, std = {lf.std:.2f} nm  "
                    f"(n = {lf.n_samples})")

    st.subheader("Pooled Histograms")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        diams = batch_mod.pooled_eq_diameters_nm(batch_result)
        if diams.size > 0:
            hist_fig = visualization.plot_histogram(
                diams, "nm", "Pooled equivalent diameter",
                bins=min(30, max(5, diams.size // 5)),
                lognormal_fit=pstats.lognormal_fit,
                xlabel="Equivalent diameter (nm)",
            )
            st.pyplot(hist_fig, use_container_width=True)
    with col_h2:
        areas = batch_mod.pooled_areas_nm2(batch_result)
        if areas.size > 0:
            hist_fig2 = visualization.plot_histogram(
                areas, "nm²", "Pooled grain area",
                bins=min(30, max(5, areas.size // 5)),
                color="darkorange", xlabel="Area (nm²)",
            )
            st.pyplot(hist_fig2, use_container_width=True)

    st.subheader("Per-image Comparison")
    try:
        import matplotlib.pyplot as plt
        per_image_diams = []
        labels = []
        for rep in reports:
            ds = [g["equivalent_diameter_nm"] for g in rep.get("grains", [])]
            if ds:
                per_image_diams.append(ds)
                labels.append(rep["image"]["file_name"][:30])
        if per_image_diams:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            ax.boxplot(per_image_diams, labels=labels, showmeans=True,
                       meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
            ax.set_ylabel("Equivalent diameter (nm)")
            ax.set_title("Per-image grain diameter distribution")
            plt.xticks(rotation=15, ha="right")
            st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Box plot failed: {e}")

    st.markdown("---")
    st.subheader("Download Combined Data")
    import csv
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow([
        "source_image", "pixel_size_nm", "magnification",
        "layer_source", "layer_area_nm2",
        "label", "area_nm2", "equivalent_diameter_nm", "perimeter_nm",
        "circularity", "eccentricity", "major_axis_nm", "minor_axis_nm",
        "aspect_ratio", "grain_orientation_deg",
    ])
    for rep in reports:
        img = rep.get("image", {})
        layer = rep.get("layer", {})
        for g in rep.get("grains", []):
            writer.writerow([
                img.get("file_name", "?"),
                img.get("pixel_size_nm", ""),
                img.get("magnification", ""),
                layer.get("source", ""),
                layer.get("area_nm2", ""),
                g["label"], g["area_nm2"], g["equivalent_diameter_nm"],
                g["perimeter_nm"], g["circularity"], g["eccentricity"],
                g["major_axis_nm"], g["minor_axis_nm"], g["aspect_ratio"],
                g["orientation_deg"],
            ])
    csv_data = csv_buf.getvalue().encode("utf-8")
    st.download_button(
        label=f"Download combined CSV ({len(csv_data)/1024:.1f} KB)",
        data=csv_data, file_name="pooled_grains.csv", mime="text/csv",
    )

    combined_json = report.serialize_report({
        "schema_version": report.SCHEMA_VERSION,
        "app": report.APP_NAME, "type": "batch_aggregate",
        "n_reports": batch_result.n_reports,
        "total_grain_count": batch_result.total_grain_count,
        "total_layer_area_nm2": batch_result.total_layer_area_nm2,
        "pooled_statistics": batch_result.pooled_stats.to_dict(),
        "per_image": [r.to_dict() for r in batch_result.per_image],
    })
    st.download_button(
        label=f"Download combined JSON ({len(combined_json)/1024:.1f} KB)",
        data=combined_json, file_name="pooled_statistics.json", mime="application/json",
    )


# === Help tab ===

def render_help_tab() -> None:
    st.markdown("# Help & Usage Guide")

    st.markdown("## Quick Start")
    st.markdown(
        "1. **Upload a TIFF** in the sidebar of the Single Image tab."
        "\n"
        "2. **Check the metadata** — pixel size should be auto-detected from the "
        "OlympusSIS tag. If not, enter it manually under *Image Metadata → Override pixel size*."
        "\n"
        "3. **Select a layer** — four modes (in order of recommendation):"
        "\n   • **SAM (click on layer)** — click on the layer you want; SAM segments it. "
        "Most reliable for TEM images. First click downloads ~350 MB model + encodes image (~5 s), "
        "subsequent clicks are instant."
        "\n   • **SAM (auto)** — SAM finds all candidate layers automatically; pick from dropdown."
        "\n   • **Polygon (click vertices)** — click on the image to add polygon vertices "
        "around the layer. No SAM needed, just `streamlit-image-coordinates`."
        "\n   • **Manual rows** — pick a top/bottom row range (legacy, only for horizontal layers)."
        "\n"
        "4. **Click Analyze grains** — the app segments grains inside the selected layer "
        "and displays the colour-labeled grain map, statistics, and histograms."
        "\n"
        "5. **Download the JSON report** — self-contained (measurements + metadata + "
        "parameters + layer polygon + labeled image as base64 PNG)."
        "\n"
        "6. **For batch statistics** — repeat for each image, then go to the "
        "**Batch Statistics** tab, drop all JSON reports in."
    )

    st.markdown("---")
    st.markdown("## Method")

    st.markdown(
        "The app analyses a **selected layer** (any shape — rectangle, polygon, "
        "or SAM-generated mask) of a TEM cross-section image for grain-size "
        "statistics. The following segmentation methods are available:"
    )

    st.markdown("### Manual tracing (polygon) — MOST RELIABLE (ISO 13067 compliant)")
    st.markdown(
        "The user draws grain outlines by clicking polygon vertices around "
        "each grain. Each closed polygon becomes one grain. After Finish, "
        "standard grain-size statistics are computed (area, equivalent "
        "diameter, circularity, lognormal fit)."
        "\n\nThis is the method recommended by ISO 13067 and NIST for "
        "polycrystalline thin films when automated methods are unreliable. "
        "It is what researchers in leading TEM labs (NCEM Berkeley, ER-C "
        "Jülich, Oxford Materials) actually use for HRTEM cross-sections "
        "when ASTAR / 4D-STEM is not available."
        "\n\n**When to use:** always works, no tuning needed, publishable "
        "results. Takes ~5 minutes per image."
    )

    st.markdown("### Structure tensor (auto-assist) — fast automatic")
    st.markdown(
        "Computes the local lattice-fringe orientation at each pixel via the "
        "structure tensor (gradient-based). Grains = connected regions with "
        "constant orientation (within a tolerance). Faster than FFT methods "
        "and works in real space (no inverse problem)."
        "\n\n**When to use:** good for HRTEM images where lattice fringes "
        "are clearly visible and grains have distinct orientations. May "
        "fail on noisy images or grains without visible fringes."
    )

    st.markdown("### Click-to-segment grains (FFT) — experimental")
    st.markdown(
        "The user clicks on each grain in the cropped layer image. For each "
        "click:"
        "\n1. **Local FFT** is computed in a window (default 96×96 px, "
        "configurable) around the click — zero-padded to 256×256 for better "
        "frequency resolution."
        "\n2. **d-spacings of the clicked grain** are found via the 1D radial "
        "profile detector + Gaussian quality filter (same as the global FFT "
        "method, but local). Each grain can have up to N plane families "
        "(default 5)."
        "\n3. **Amplitude maps** are computed for each d-spacing — using the "
        "GLOBAL FFT for better SNR (the global FFT is computed once and cached)."
        "\n4. **Region growing with hysteresis + AND-condition.** A pixel "
        "belongs to the grain if its min-amplitude (across all d-spacings) "
        "exceeds the growth threshold AND it is connected to the seed (click "
        "point). Hysteresis: the seed must pass a higher threshold than the "
        "growth — this prevents fragmented regions."
        "\n5. **Auto-find similar grains.** After each click, the app scans "
        "the rest of the layer for OTHER connected regions with the same "
        "d-spacing signature. The user can review and add them one-by-one or "
        "all at once."
        "\n6. **Undo / Clear / Finish.** After all grains are marked, click "
        "Finish to compute the standard grain-size statistics (area, equivalent "
        "diameter, circularity, lognormal fit)."
        "\n"
        "**Why this works for TEM grains:** the global FFT method mixes all "
        "grains' d-spacings together and can't separate grains that share a "
        "d-spacing but differ in another. The click-to-segment method looks at "
        "EACH grain individually — its local FFT reveals the FULL set of plane "
        "families in that specific grain, and the AND-condition grows a region "
        "where ALL of them are present simultaneously. This naturally "
        "separates grains by their complete crystallographic signature, not "
        "just one d-spacing."
    )

    st.markdown("### FFT (Grachev) — global FFT method")
    st.markdown(
        "Based on the method described in Grachev's dissertation (§2.4 "
        "\"Процедура визуализации нанокристаллов\"). Adapted for grains in a "
        "polycrystalline layer. The pipeline:"
    )
    st.markdown(
        "1. **Layer selection** — SAM / polygon / manual rows."
        "\n2. **Crop to layer bbox** + optional CLAHE + Hann window."
        "\n3. **FFT** — 2-D Fourier transform of the layer image."
        "\n4. **Wiener filter** — suppress the amorphous-matrix halo (radially "
        "symmetric noise) so Bragg reflections stand out."
        "\n5. **Reflection detection** (default: **1D radial profile**): "
        "azimuthally integrate the power spectrum into a 1D S(r) profile, find "
        "peaks (each peak = one d-spacing), then look for 2D local maxima ONLY "
        "within rings at those radii. Much more robust than 2D `peak_local_max` "
        "because the noise is averaged out azimuthally. The legacy 2D detector "
        "is still available via `use_radial_profile_detection=False`."
        "\n6. **Per-cluster amplitude map** — for each d-spacing, build a RING "
        "mask at that radius (Grachev §2.4.2), inverse-FFT, take amplitude. "
        "Bright pixels = regions whose lattice has this d-spacing."
        "\n7. **Pixel labelling** — assign each pixel to the cluster with max "
        "amplitude (argmax). Pixels below threshold = background."
        "\n8. **Connected-component labeling** — split grains sharing an "
        "orientation into separate regions."
        "\n9. **Post-processing** — remove small regions, fill holes, exclude "
        "border-touching grains."
        "\n"
        "**Manual reflection mode** — if the auto-detector misses a real "
        "d-spacing (or picks up noise), you can click on reflections in the 2D "
        "power spectrum manually. **Each click shows an amplitude preview** "
        "(inverse-FFT of the ring at the click radius overlaid on the layer "
        "image), so you can verify in real time that the clicked reflection "
        "actually highlights real grains. The 1D radial spectrum above the "
        "power spectrum shows which d-spacings are present — use it as a guide "
        "for where to click."
        "\n"
        "**Reflection quality (Gaussian fit + Scherrer).** For every click, a "
        "2D elliptical Gaussian is fitted in a small window (default 15×15 px). "
        "From the fit:"
        "\n  • **σ** (FWHM = 2.355·σ) — peak width. Narrow = sharp Bragg peak "
        "from large coherent domains; broad = either small grains (Scherrer "
        "broadening) or noise."
        "\n  • **SNR** = peak amplitude / background. ≥ 5 = clearly real; "
        "< 2 = likely noise."
        "\n  • **D (Scherrer)** = K / FWHM_g — coherent-domain size in nm "
        "(K = 0.94 for spherical domains). This is the average size of the "
        "region over which the lattice is coherent; for real grains it should "
        "match the grain size seen in the image."
        "\n  Each click is classified 🟢 good / 🟡 borderline / 🔴 noise. "
        "Clicks classified as 🔴 noise can be auto-rejected before "
        "re-segmentation (checkbox under the Re-segment button). The same "
        "quality filter can also be applied to AUTO-detected reflections "
        "(sidebar → 'Reflection quality' → 'Apply Gaussian quality filter to "
        "AUTO reflections'). When ON, every 2D maximum found by the "
        "auto-detector is checked; maxima classified as noise are dropped, and "
        "if ALL maxima of a ring are noise the whole ring is dropped."
        "\n"
        "**Reference d-spacings (mode B).** When you know the material, switch "
        "on 'Restrict detection to known plane families' in the sidebar. The "
        "detector then searches for reflections ONLY inside the radial bands "
        "corresponding to the (d_min, d_max) ranges you list — instead of "
        "scanning the whole spectrum. Use the material presets (Si, Ge, TiN, "
        "Al, Au, Pt) to auto-fill the table, or enter your own. Each detected "
        "d-spacing gets a label (e.g. {111}) shown in the result table and "
        "as a green shaded band on the 1D radial spectrum plot."
        "\n"
        "**Why this works for TEM:** grain boundaries are not sharp intensity "
        "steps but smooth contrast changes due to different crystallographic "
        "orientations. The FFT method does not look for 'edges'; it groups "
        "pixels by their local lattice fringes, which directly reflect grain "
        "orientation. This is fundamentally different from gradient/watershed "
        "methods and is specifically designed for TEM (Grachev §2.4.2)."
    )

    st.markdown("### Classical watershed")
    st.markdown(
        "Traditional image-processing pipeline (faster but less reliable on TEM):"
        "\n1. **Bilateral filter** — edge-preserving denoise."
        "\n2. **CLAHE** — local contrast enhancement."
        "\n3. **Scharr gradient** — boundary detection."
        "\n4. **Otsu threshold** — boundary mask."
        "\n5. **Morphological close + open** — connect boundaries, remove noise."
        "\n6. **Distance transform + watershed** — split touching grains."
        "\n"
        "Use this only if FFT gives poor results or for quick experimentation."
    )

    st.markdown(
        "Both methods produce a labeled image (one label per grain), from which "
        "per-grain measurements are computed with `skimage.measure.regionprops` "
        "(area, equivalent diameter, perimeter, circularity, eccentricity, "
        "orientation), followed by a lognormal fit on the equivalent-diameter "
        "distribution."
    )

    st.markdown("---")
    st.markdown("## Layer Selection Modes")

    st.markdown("### SAM (click on layer) — recommended")
    st.markdown(
        "Uses Segment Anything Model in **point-prompt mode**. "
        "You click on the layer you want, and SAM segments exactly the region "
        "containing your click point. This is more reliable than automatic mode "
        "because you tell SAM which layer to segment."
        "\n\n"
        "**Model selection:** In the *SAM parameters* expander, you can choose between:"
        "\n* **Vanilla SAM** (`sam_vit_b`, `sam_vit_l`) — Meta's original model, "
        "trained on natural images (SA-1B dataset). Works but is not specialised "
        "for microscopy."
        "\n* **Micro-SAM** (`microsam_vit_b_em`, `microsam_vit_l_em`, "
        "`microsam_vit_b_lm`, `microsam_vit_l_lm`) — fine-tuned variants from EMBL, "
        "trained on microscopy data. **For TEM cross-sections, the EM-organelles "
        "variants (`_em`) are recommended** — they were fine-tuned on electron "
        "microscopy and understand TEM textures better than vanilla SAM."
        "\n\n"
        "**How to use:**"
        "\n1. Select **SAM (click on layer)** mode."
        "\n2. (Optional) In *SAM parameters*, choose a Micro-SAM model for better TEM results."
        "\n3. Click on the layer in the image preview."
        "\n4. Wait for SAM (first click loads the model ~350 MB + encodes the image ~5 s; "
        "subsequent clicks on the same image are instant)."
        "\n5. SAM shows the segmented region (green overlay). If it's the right layer, "
        "click **Use this layer**. If not, click **Clear** and click elsewhere."
        "\n\n"
        "**Dependencies:** `pip install torch torchvision` "
        "(or `--index-url https://download.pytorch.org/whl/cpu` for CPU-only), "
        "`pip install git+https://github.com/facebookresearch/segment-anything.git`, "
        "and `pip install streamlit-image-coordinates`."
    )

    st.markdown("### SAM (auto)")
    st.markdown(
        "Uses SAM in **automatic mask generation** mode — finds all candidate "
        "regions, you pick from a dropdown. Useful when you're not sure where "
        "the layer is and want to see all options."
        "\n\n"
        "Same model selection as SAM (click on layer) — Micro-SAM variants are "
        "available and recommended for TEM."
        "\n\n"
        "Masks are filtered to layer candidates by area (1–95%), eccentricity (≥0.85), "
        "and intensity std (≤60). Sorted by area descending, top 10 shown."
        "\n\n"
        "Same dependencies as SAM (click on layer)."
    )

    st.markdown("### Polygon (click vertices)")
    st.markdown(
        "Click on the image to add polygon vertices around the layer. When you "
        "have ≥3 vertices, click **Use polygon**. Buttons: **Clear** (start over), "
        "**Undo last** (remove the last vertex)."
        "\n\n"
        "This is the most reliable fallback when SAM is unavailable or finds no "
        "good candidates. Works for any layer shape (tilted, curved, irregular)."
        "\n\n"
        "**Dependency:** `pip install streamlit-image-coordinates`."
    )

    st.markdown("### Manual rows")
    st.markdown(
        "Legacy mode for horizontal layers. Pick top row (y0) and bottom row (y1) "
        "with sliders. The layer is a full-width horizontal band. Only works when "
        "layers are horizontal — use SAM or Polygon for tilted/curved layers."
    )

    st.markdown("---")
    st.markdown("## Settings Reference")
    st.markdown(
        "See the **Advanced (Segmentation)** expander in the sidebar for the full "
        "parameter list. Key parameters:"
    )
    st.markdown(
        "| Setting | Default | Effect |"
        "\n|---|---|---|"
        "\n| **CLAHE clip limit** | 3.0 | Contrast enhancement. Increase to 4–5 for low-contrast images. |"
        "\n| **Threshold scale factor** | 1.0 | >1 = fewer boundaries (larger grains). <1 = more boundaries. |"
        "\n| **Watershed min distance** | 0 (auto) | 0 = 5% of layer height. Larger = fewer, larger grains. |"
        "\n| **Min grain area** | 25 nm² | Grains smaller than this are dropped. |"
        "\n| **Exclude left/right borders** | ON | Drop grains touching the cropped image's left/right edges. |"
    )

    st.markdown("---")
    st.markdown("## Output Metrics")
    st.markdown(
        "Each grain produces (all in nanometre units):"
        "\n* **Area** (nm²) — pixel count × pixel_size²"
        "\n* **Equivalent diameter** (nm) — = 2·√(A/π)"
        "\n* **Perimeter** (nm)"
        "\n* **Circularity** = 4πA/P² — 1 = circle, →0 = elongated"
        "\n* **Eccentricity** — 0 = circle, →1 = line"
        "\n* **Major / minor axis** (nm)"
        "\n* **Aspect ratio** = minor/major"
        "\n* **Orientation** (deg) — major-axis angle, [-90, 90]"
    )

    st.markdown("---")
    st.markdown("## Troubleshooting")

    with st.expander("SAM dependencies not installed"):
        st.markdown(
            "**Fix:** install torch and segment-anything:"
            "\n```"
            "\npip install torch torchvision"
            "\npip install git+https://github.com/facebookresearch/segment-anything.git"
            "\n```"
            "\nFor CPU-only torch (smaller):"
            "\n```"
            "\npip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            "\n```"
            "\nUntil installed, use **Polygon (draw)** or **Manual rows** mode."
        )

    with st.expander("SAM first run takes very long"):
        st.markdown(
            "**Cause:** the model checkpoint (~350 MB) is being downloaded."
            "\n\n"
            "**Fix:** wait. Subsequent runs use the cached model and take 5–30 s."
        )

    with st.expander("SAM out-of-memory error (DefaultCPUAllocator: not enough memory)"):
        st.markdown(
            "**Cause:** SAM tried to allocate too much RAM. This happens on large "
            "TEM images (e.g. 2664×4008) when the image is not downscaled before SAM."
            "\n\n"
            "**Fixes (in order of effectiveness):**"
            "\n1. **Reduce Max image dimension** in the SAM parameters expander "
            "(sidebar → Layer Selection → SAM parameters). Default 1024. Try 768 or 512."
            "\n2. **Reduce Points per side** to 16 or 12 (fewer masks = less memory)."
            "\n3. **Use CPU-only torch** if you installed GPU torch but have no GPU: "
            "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` "
            "(CPU torch uses less memory than GPU torch on CPU)."
            "\n4. **Switch to Polygon (draw) mode** — it needs zero extra memory."
            "\n\n"
            "**Why this happens:** SAM's automatic mask generator creates one mask "
            "per grid point (points_per_side²). Each mask is a full-resolution bool "
            "array. On a 2664×4008 image with 32×32=1024 points, that's ~10.7M pixels "
            "× 1024 masks = ~11 GB just for masks. The app now downscales to 1024px "
            "by default (reducing masks to ~1M pixels × 576 masks = ~600 MB), but if "
            "you still hit OOM, reduce the parameters above."
        )

    with st.expander("SAM finds no layer candidates"):
        st.markdown(
            "**Cause:** the default filters (area 1–95%, eccentricity ≥0.85, "
            "intensity std ≤60) are too strict for your image."
            "\n\n"
            "**Fixes:**"
            "\n1. Increase **Max image dimension** to 1536–2048 (if RAM allows) — "
            "smaller images may miss thin layers."
            "\n2. Switch to **Polygon (draw)** mode and draw the layer manually."
        )

    with st.expander("Too few grains detected"):
        st.markdown(
            "**Fixes:**"
            "\n1. Lower **Threshold scale factor** to 0.7–0.9 (more boundaries)."
            "\n2. Lower **Watershed min distance** (or set to 0 for auto)."
            "\n3. Lower **Min grain area**."
            "\n4. Disable **Exclude top/bottom borders** if your layer is thin."
        )

    with st.expander("Too many grains (over-segmentation)"):
        st.markdown(
            "**Fixes:**"
            "\n1. Raise **Threshold scale factor** to 1.2–1.5."
            "\n2. Raise **Watershed min distance**."
            "\n3. Raise **Min grain area**."
            "\n4. Increase **Bilateral filter d** for stronger denoising."
        )

    st.markdown("---")
    st.markdown("## Limitations")
    st.markdown(
        "* **2-D projection** — TEM cross-sections show a 2-D projection of a 3-D "
        "structure. Grains at different depths can overlap; measured sizes are "
        "projected, not true 3-D."
        "\n* **Moiré artefacts** — overlapping grains can produce moiré fringes "
        "mistaken for boundaries. Verify the labeled map before trusting statistics."
        "\n* **SAM model size** — ~350 MB download on first use. Cached afterwards."
        "\n* **Polygon mode** — manual vertex entry via numeric inputs (no click-on-image). "
        "A future version may integrate `streamlit-drawable-canvas` for click-drawing."
        "\n* **One layer per analysis** — for multi-layer structures, run the "
        "Single Image tab multiple times with different layers."
    )

    st.markdown("---")
    st.markdown("## JSON Report Schema (v1.1)")
    st.code(
        "{\n"
        "  \"schema_version\": \"1.1\",\n"
        "  \"app\": \"02_grain_analyzer\",\n"
        "  \"generated_at\": \"2025-10-31T12:34:56Z\",\n"
        "  \"image\": { \"file_name\": ..., \"pixel_size_nm\": ..., ... },\n"
        "  \"layer\": {\n"
        "    \"source\": \"sam\",  // or \"polygon\" or \"manual_rows\"\n"
        "    \"polygon\": [[y0,x0], [y1,x1], ...],\n"
        "    \"bbox\": [min_row, min_col, max_row, max_col],\n"
        "    \"area_nm2\": ...,\n"
        "    \"height_nm\": ...,\n"
        "    \"width_nm\": ...,\n"
        "    \"mean_intensity\": ...,\n"
        "    \"pixel_size_nm\": ...\n"
        "  },\n"
        "  \"segmentation_params\": { ... },\n"
        "  \"statistics\": { \"n_grains\": ..., \"eq_diameter_stats\": ..., \"lognormal_fit\": ... },\n"
        "  \"grains\": [ { \"label\": 1, \"area_nm2\": ..., ... }, ... ],\n"
        "  \"visualization_png_base64\": \"iVBORw0KGgo...\"\n"
        "}",
        language="json",
    )


# === Render tabs ===

tab_single, tab_batch, tab_help = st.tabs(["Single Image", "Batch Statistics", "Help"])

with tab_single:
    render_single_image_tab()

with tab_batch:
    render_batch_tab()

with tab_help:
    render_help_tab()
