"""
app.py — Streamlit UI for stitching optical microscopy images.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

# Add path to src
sys.path.insert(0, str(Path(__file__).parent))

from src import io_utils, pipeline, pair_review, preprocessing, stitching


# === Page configuration ===
st.set_page_config(
    page_title="Optical Microscopy Stitcher",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === Global CSS: keep the sidebar reliably scrollable ===
#
# Streamlit styles the sidebar scrollbar as a thin overlay that only appears
# on hover. Make it always visible and force the scroll container to scroll.

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow-y: auto !important;
        scrollbar-width: auto;          /* Firefox: full-width scrollbar */
        scrollbar-gutter: stable;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
        width: 10px;                     /* Chromium/WebKit: always visible */
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
        background: rgba(128, 128, 128, 0.55);
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
        background: rgba(128, 128, 128, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# === Navigation ===
#
# Why not st.tabs? Streamlit evaluates the bodies of *all* tabs on every
# rerun (heavy content included), so switching recomputed hidden sections.
# With an explicit navigation widget only the active section's main content
# renders. The SIDEBAR is different: it is shared by all tabs and rendered
# on every run (see render_sidebar). It used to live inside the Stitcher
# renderer only — on Pair Review/Help the sidebar was then empty (nowhere
# to upload, although the info box told the user to "upload in the
# sidebar"), and Streamlit deleted the uploaded-file session state because
# the uploader widgets stopped rendering. The always-on sidebar fixes both.

NAV_TABS = ["Stitcher", "Pair Review", "Help"]


def _render_navigation() -> str:
    """Render the top navigation and return the active section name."""
    if hasattr(st, "segmented_control"):
        # Streamlit >= 1.40: pill-style control that looks like tabs.
        extra_kwargs = {}
        if "nav_seg" not in st.session_state:
            extra_kwargs["default"] = st.session_state.get("nav_tab", NAV_TABS[0])
        nav = st.segmented_control(
            "App section", NAV_TABS, selection_mode="single",
            key="nav_seg", label_visibility="collapsed",
            **extra_kwargs,
        )
    else:
        nav = st.radio(
            "App section", NAV_TABS, horizontal=True,
            key="nav_seg", label_visibility="collapsed",
        )
    if not nav:  # segmented_control allows deselecting the current pill
        nav = st.session_state.get("nav_tab", NAV_TABS[0])
    st.session_state.nav_tab = nav
    return nav


def _param(key: str, default):
    """Read a sidebar widget value from session state (shared across tabs)."""
    return st.session_state.get(key, default)


def _blend_mode_params() -> dict:
    """Resolve the `Blending weight mode` radio to the underlying pipeline kwargs.

    Returns a dict with the three weight-mode parameters
    (`blend_use_focus_weights`, `blend_focus_priority`,
    `blend_focus_priority_gamma`, `blend_focus_priority_levels`) so both
    `stitch_row()` call sites stay in sync with the sidebar.

    Radio modes:
      "Feather"                      -> feather mask only
      "Focus-weighted (per-tile)"    -> legacy focus_weight_mask
      "Focus-priority (default)"     -> winner-takes-all at high freq
    """
    mode = st.session_state.get("blend_weight_mode", "Focus-priority (default)")
    if mode == "Focus-weighted (per-tile)":
        return {
            "blend_use_focus_weights": True,
            "blend_focus_priority": False,
        }
    if mode == "Focus-priority (default)":
        return {
            "blend_use_focus_weights": False,
            "blend_focus_priority": True,
            "blend_focus_priority_gamma": _param("focus_priority_gamma", 6.0),
            "blend_focus_priority_levels": _param("focus_priority_levels", 2),
        }
    # Feather and any unknown value
    return {
        "blend_use_focus_weights": False,
        "blend_focus_priority": False,
    }


# === ROI mode constants (sidebar radio options — single source of truth) ===
_ROI_MODE_REF1 = "Auto (reference tile 1) — same band on every tile"
_ROI_MODE_PER_TILE = "Auto-detect per tile (advanced)"


def _roi_sig_hash(sig):
    """Short stable hash of the upload signature (widget keys / caches)."""
    return hashlib.md5(repr(sig).encode("utf-8")).hexdigest()[:8]


def _roi_ref_off(sig_hash):
    """Shared reference-band vertical offset (px), keyed by the upload
    signature.

    Edited from BOTH the Stitcher ROI preview slider and the Pair Review
    Step 3 slider — the two stay in sync via on_change callbacks, so the
    preview and the stitch always use the same band.

    The band HEIGHT is deliberately NOT stored here: it is governed by
    the sidebar "ROI height (fraction of tile height)" slider only (the
    old Pair Review "Band height (px)" slider was removed — it
    duplicated the sidebar control). A height value left over in
    session state from an older version is ignored.
    """
    d = st.session_state.get("roi_ref1_adj")
    if not (isinstance(d, dict) and d.get("sig") == sig_hash):
        return 0
    try:
        return int(d.get("off", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _roi_ref_set(sig_hash, **upd):
    """Merge updates into the shared reference-band adjustment."""
    d = st.session_state.get("roi_ref1_adj")
    d = dict(d) if isinstance(d, dict) and d.get("sig") == sig_hash \
        else {"sig": sig_hash}
    d.update(upd)
    st.session_state["roi_ref1_adj"] = d


def _roi_ref_off_cb(sig_hash, src_key):
    """on_change callback for the offset slider: store the value and keep
    the sibling slider (the same control rendered in the other tab) in
    sync. Runs before the rerun instantiates widgets, so mutating
    session_state here is legal."""
    v = int(st.session_state.get(src_key, 0) or 0)
    _roi_ref_set(sig_hash, off=v)
    for k in (f"st_roi_ref_off_{sig_hash}", f"rvp_roi_ref_off_{sig_hash}"):
        if k != src_key:
            st.session_state[k] = v


def _roi_pt_off(sig_hash, k):
    """Shared per-tile vertical offset (px) for the "Auto-detect per
    tile (advanced)" mode, keyed by the upload signature. Tile k's
    offset shifts that tile's own auto-detected band; the band height
    always comes from the sidebar "ROI height (fraction of tile
    height)" slider.

    Edited from BOTH the Stitcher per-tile ROI preview and the Pair
    Review Step 3 sliders — one state, synchronised by on_change
    callbacks, so the previews and the stitch always use the same
    bands.
    """
    d = st.session_state.get("roi_pt_adj")
    if not (isinstance(d, dict) and d.get("sig") == sig_hash):
        return 0
    try:
        return int((d.get("offs") or {}).get(str(k), 0) or 0)
    except (TypeError, ValueError):
        return 0


def _roi_pt_off_cb(sig_hash, k, src_key):
    """on_change callback for a per-tile offset slider: store the value
    and keep the sibling slider (the same control rendered in the other
    tab) in sync. Runs before the rerun instantiates widgets, so
    mutating session_state here is legal."""
    v = int(st.session_state.get(src_key, 0) or 0)
    d = st.session_state.get("roi_pt_adj")
    d = dict(d) if isinstance(d, dict) and d.get("sig") == sig_hash \
        else {"sig": sig_hash}
    offs = dict(d.get("offs") or {})
    offs[str(k)] = v
    d["offs"] = offs
    st.session_state["roi_pt_adj"] = d
    for prefix in (f"st_roi_pt_off_{sig_hash}_",
                   f"rvp_roi_pt_off_{sig_hash}_"):
        kk = f"{prefix}{k}"
        if kk != src_key:
            st.session_state[kk] = v


def _shift_band(band, off, h):
    """Vertically shift an ROI band by `off` px, clamped to the tile.

    The band height is preserved (only the position moves); y0 / y1
    stay inside [0, h] and the band never collapses (y1 > y0).
    """
    y0 = max(0, min(h - 1, int(band[0]) + int(off)))
    y1 = max(y0 + 1, min(h, int(band[1]) + int(off)))
    return (y0, y1)


def _auto_band_bounds(h_frac):
    """Detection clamp bounds used with `auto_detect_roi_band`.

    These mirror `io_utils.roi_band_for_tiles(mode="auto")` so BOTH ROI
    modes honour the sidebar "ROI height (fraction of tile height)"
    slider EXACTLY (up to 1.0). Relying on the helper's own conservative
    default cap (max_fraction=0.50) silently clipped the window at half
    the tile height in the reference-tile-1 mode — the "window never
    grows beyond 0.5" bug.
    """
    return max(0.10, h_frac - 0.10), min(1.0, h_frac + 0.25)


def _roi_ref_band(img, sig_hash, h_frac):
    """Auto-detected tile-1 band plus the shared vertical offset.

    Returns (auto_band, applied_band). Used by the Stitcher preview,
    the Stitch button and Pair Review so all three always agree. The
    band height comes ONLY from `h_frac` (the sidebar "ROI height
    (fraction of tile height)" slider) and is honoured up to the full
    tile height; the offset is the shared user adjustment.
    """
    lo, hi = _auto_band_bounds(h_frac)
    auto = io_utils.auto_detect_roi_band(
        img, target_fraction=h_frac, min_fraction=lo, max_fraction=hi)
    h0 = int(img.shape[0])
    return auto, _shift_band(auto, _roi_ref_off(sig_hash), h0)


def _roi_auto_bands_cached(tiles, h_frac):
    """Auto-detected per-tile ROI bands, cached in session_state.

    Detection (per-row Tenengrad on every tile) depends ONLY on the
    upload and the sidebar fraction — NOT on the per-tile offsets — so
    it runs ONCE per (upload, fraction). Moving one tile's offset
    slider then re-uses these bands instead of re-scanning every tile,
    which is what makes the update touch only the edited tile.
    """
    sig = _current_upload_signature()
    sig_hash = _roi_sig_hash(sig) if sig is not None else "default"
    cache = st.session_state.get("roi_auto_bands_cache")
    if not (isinstance(cache, dict)
            and cache.get("sig") == sig_hash
            and cache.get("h_frac") == h_frac
            and isinstance(cache.get("bands"), list)
            and len(cache["bands"]) == len(tiles)):
        bands = io_utils.roi_band_for_tiles(
            tiles, mode="auto", target_fraction=h_frac)
        cache = {"sig": sig_hash, "h_frac": h_frac, "bands": bands}
        st.session_state["roi_auto_bands_cache"] = cache
    return cache["bands"]


def _roi_overlay_cached(sig_hash, k, image, band, out_width):
    """`pair_review.render_tile_with_roi_overlay` with a session cache
    keyed by (upload, tile, band, width).

    Moving ONE tile's offset slider changes only that tile's band; the
    overlays of all other tiles are reused from this cache instead of
    being re-drawn from the full-resolution frames on every rerun.
    """
    cache = st.session_state.get("roi_overlay_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["roi_overlay_cache"] = cache
    ckey = (sig_hash, int(k), int(band[0]), int(band[1]), int(out_width))
    out = cache.get(ckey)
    if out is None:
        out = pair_review.render_tile_with_roi_overlay(
            image, band, out_width=out_width)
        if len(cache) >= 128:
            cache.clear()
        cache[ckey] = out
    return out


def _resolve_roi_band_for_stitcher(tiles):
    """Read the ROI sidebar settings and compute the ROI band to apply.

    Returns (roi_band, roi_bands_per_tile, roi_label, keep_full) where
    exactly ONE of roi_band / roi_bands_per_tile is not None (or both are
    None when ROI is disabled). roi_label is a short human-readable
    description for the Stitcher preview caption. keep_full echoes the
    sidebar "Also keep full panorama (debug)" checkbox.

    Modes (sidebar radio "ROI detection mode"):
    - "Auto (reference tile 1) ..." (DEFAULT): detect the textured band
      on tile 1 only and apply the SAME band to every tile. The band can
      be shifted interactively (Stitcher ROI preview slider and Pair
      Review Step 3 sliders share ONE state); the shift is applied here
      too, so the preview and the stitched panorama always agree. Single
      band => Pair Review translations transfer exactly (no per-tile dy
      conversion needed), so the seams stay sharp.
    - "Auto-detect per tile (advanced)": per-tile bands, each one
      shifted by its own interactive offset (the Stitcher per-tile ROI
      preview and the Pair Review Step 3 sliders share ONE state — the
      offsets are applied here too, so the previews and the stitched
      panorama always agree). The band heights always come from the
      sidebar "ROI height (fraction of tile height)" slider. When the
      resulting bands differ, roi_bands_per_tile is returned and
      pipeline.stitch_row converts pair dy between the full-tile and
      cropped-tile frames automatically.

    Any other stored mode value (e.g. the "Manual ..." mode removed in
    an older version) safely falls back to the default reference-tile-1
    behaviour.
    """
    if not _param("roi_enabled", True):
        return None, None, "", False
    mode = _param("roi_mode", _ROI_MODE_REF1)
    h_frac = float(_param("roi_height_fraction", 0.30))
    keep_full = bool(_param("keep_full_panorama", False))
    if mode == _ROI_MODE_PER_TILE:
        sig = _current_upload_signature()
        sig_hash = _roi_sig_hash(sig) if sig is not None else "default"
        auto_bands = io_utils.roi_band_for_tiles(
            tiles, mode="auto", target_fraction=h_frac,
        )
        # Apply each tile's shared interactive offset (default 0). These
        # are the SAME values the previews show — single state for the
        # Stitcher preview, Pair Review Step 3 and the stitch itself.
        bands = [
            _shift_band(b, _roi_pt_off(sig_hash, k), int(t.image.shape[0]))
            for k, (b, t) in enumerate(zip(auto_bands, tiles))
        ]
        if len(set(bands)) == 1:
            b = bands[0]
            return b, None, f"auto per tile (all agreed) y=[{b[0]},{b[1]})", keep_full
        return None, bands, "auto per tile (bands differ — dy auto-converted)", keep_full
    # Default: Auto (reference tile 1) + the shared interactive offset /
    # height adjustment (single source of truth for preview and stitch)
    sig = _current_upload_signature()
    sig_hash = _roi_sig_hash(sig) if sig is not None else "default"
    auto_band, band = _roi_ref_band(tiles[0].image, sig_hash, h_frac)
    label = (f"reference tile 1 y=[{band[0]},{band[1]}) "
             f"(auto y=[{auto_band[0]},{auto_band[1]}]) — same on every tile")
    return band, None, label, keep_full


# Scoped reruns: Streamlit >= 1.39 supports @st.fragment — a widget
# inside the fragment then re-renders ONLY the fragment (here: the
# per-tile ROI grids), not the whole page. Older versions (the project
# pins streamlit>=1.30) get a no-op decorator; the caches above still
# keep those interactions fast there.
if hasattr(st, "fragment"):
    _scope_to_widget = st.fragment
else:  # pragma: no cover - legacy Streamlit fallback
    def _scope_to_widget(fn):
        return fn


@_scope_to_widget
def _render_roi_preview_stitcher(tiles, h_frac):
    """Interactive per-tile ROI preview for "Auto-detect per tile" mode.

    Mirrors Pair Review Step 3's advanced layout: one "offset (px)"
    slider per tile shifts that tile's own auto-detected band, and the
    band height always comes from the sidebar "ROI height (fraction of
    tile height)" slider. The sliders share ONE state with Pair Review
    Step 3 (synchronised by on_change callbacks), and
    `_resolve_roi_band_for_stitcher` applies the same offsets when the
    Stitch button runs — the preview and the panorama always agree.

    Runs as a Streamlit fragment (when supported): moving one tile's
    slider re-renders only THIS block. The auto-detected bands and the
    tile overlays are cached (`_roi_auto_bands_cached` /
    `_roi_overlay_cached`), so only the edited tile is actually
    recomputed / re-drawn — the other tiles' previews are reused as-is.
    """
    sig = _current_upload_signature()
    sig_hash = _roi_sig_hash(sig) if sig is not None else "default"
    n = len(tiles)
    st.caption(
        "**Advanced per-tile mode.** Each tile gets its own auto-detected "
        "band (height from **ROI height (fraction of tile height)** in "
        "the sidebar). Use the per-tile offset (px) sliders below only "
        "if a band missed the disk on that tile — the SAME sliders are "
        "available in Pair Review Step 3 and the two stay in sync.")
    n_cols = min(4, max(2, n))
    cols = st.columns(n_cols)
    auto_bands = _roi_auto_bands_cached(tiles, h_frac)
    for k, tile_k in enumerate(tiles):
        with cols[k % n_cols]:
            y0_a, y1_a = auto_bands[k]
            hk = int(tile_k.image.shape[0])
            off_key = f"st_roi_pt_off_{sig_hash}_{k}"
            st.slider(
                f"#{tile_k.index} {tile_k.name} — offset (px)",
                min_value=-hk // 2, max_value=hk // 2,
                value=_roi_pt_off(sig_hash, k), step=5, key=off_key,
                on_change=(lambda s=sig_hash, kk=k, key_=off_key:
                           _roi_pt_off_cb(s, kk, key_)),
                help=("Shift this tile's ROI band up (negative) or down "
                      "(positive). Auto-detected band: "
                      f"y=[{y0_a},{y1_a}) of {hk} px. Synchronised with "
                      "the slider in Pair Review Step 3."),
            )
            band_k = _shift_band(auto_bands[k], _roi_pt_off(sig_hash, k),
                                 hk)
            prev_k = _roi_overlay_cached(sig_hash, k, tile_k.image, band_k,
                                         300)
            st.image(prev_k, use_container_width=True)
            st.caption(f"band y=[{band_k[0]},{band_k[1]}) "
                       f"({band_k[1] - band_k[0]} px of {hk} px)")


@_scope_to_widget
def _render_pair_review_per_tile_grid(tiles, auto_bands_full, sig_hash_pt):
    """Per-tile ROI grid of Pair Review Step 3 (advanced mode).

    Runs as a Streamlit fragment (when supported): moving one tile's
    offset slider re-renders ONLY this grid — the rest of Pair Review
    (pair previews, other Steps) stays untouched. The overlays are
    cached (`_roi_overlay_cached`), so tiles whose band did not change
    are reused as-is; only the edited tile is re-drawn.
    """
    n = len(tiles)
    n_cols_roi = min(4, max(2, n))
    roi_cols = st.columns(n_cols_roi)
    for k, tile_k in enumerate(tiles):
        with roi_cols[k % n_cols_roi]:
            y0_a, y1_a = auto_bands_full[k]
            hk = int(tile_k.image.shape[0])
            off_key = f"rvp_roi_pt_off_{sig_hash_pt}_{k}"
            st.slider(
                f"#{tile_k.index} {tile_k.name} — offset (px)",
                min_value=-hk // 2, max_value=hk // 2,
                value=_roi_pt_off(sig_hash_pt, k), step=5,
                key=off_key,
                on_change=(lambda s=sig_hash_pt, kk=k, key_=off_key:
                           _roi_pt_off_cb(s, kk, key_)),
                help=("Shift this tile's ROI band up (negative) or "
                      "down (positive). Auto-detected band: "
                      f"y=[{y0_a},{y1_a}) of {hk} px. "
                      "Synchronised with the slider in the "
                      "Stitcher tab ROI preview."))
            band_k = _shift_band(auto_bands_full[k],
                                 _roi_pt_off(sig_hash_pt, k), hk)
            prev_roi = _roi_overlay_cached(sig_hash_pt, k, tile_k.image,
                                           band_k, 300)
            st.image(prev_roi, use_container_width=True)
            st.caption(f"band y=[{band_k[0]},{band_k[1]}) "
                       f"({band_k[1] - band_k[0]} px)")


def _render_stitcher_roi_reference_ui(tiles, h_frac):
    """Interactive ROI preview for the "Auto (reference tile 1)" mode.

    A slider shifts the tile-1 band up / down (shared with Pair Review
    Step 3 — one state, synchronised by callbacks). Tile 1 is shown with
    the applied band; tiles 2..N are shown with BOTH the applied band
    and their OWN auto-detected band, so the user can judge how the
    window lands relative to the disk on every tile BEFORE spending
    time on a full stitch.
    """
    sig = _current_upload_signature()
    sig_hash = _roi_sig_hash(sig) if sig is not None else "default"
    h0 = int(tiles[0].image.shape[0])
    off_key = f"st_roi_ref_off_{sig_hash}"
    auto_band, band = _roi_ref_band(tiles[0].image, sig_hash, h_frac)
    st.slider(
        "Vertical offset of the ROI window (px) — shifts the band on "
        "EVERY tile", -h0 // 2, h0 // 2,
        value=_roi_ref_off(sig_hash), step=5, key=off_key,
        on_change=(lambda s=sig_hash, k=off_key: _roi_ref_off_cb(s, k)),
        help=("Shift the ROI window of tile 1 up (negative) or down "
              f"(positive). Auto-detected position: y=[{auto_band[0]},"
              f"{auto_band[1]}) of {h0} px. The SAME y-window is applied "
              "to every tile; the previews below show how it lands "
              "relative to each tile's own auto-detected band."),
    )
    prev = pair_review.render_tile_with_roi_overlay(
        tiles[0].image, band, out_width=520)
    st.image(prev, use_container_width=True)
    st.caption(f"Applied band y=[{band[0]},{band[1]}) "
               f"({band[1] - band[0]} px of {h0} px = "
               f"{100.0 * (band[1] - band[0]) / h0:.0f}% of the tile "
               f"height, set by the sidebar ROI height (fraction) "
               f"slider) — the same band is applied to all "
               f"{len(tiles)} tiles.")
    if len(tiles) > 1:
        with st.expander(
                f"Applied band vs each tile's own auto-detected band "
                f"(tiles 2..{len(tiles)})", expanded=False):
            n_rest = len(tiles) - 1
            cols = st.columns(min(4, max(2, n_rest)))
            own_bands = _roi_auto_bands_cached(tiles, h_frac)[1:]
            for j, tile_k in enumerate(tiles[1:]):
                with cols[j % min(4, max(2, n_rest))]:
                    prev_k = pair_review.render_tile_with_two_roi_overlays(
                        tile_k.image, band, own_bands[j], out_width=220)
                    st.image(prev_k, use_container_width=True)
                    st.caption(
                        f"#{tile_k.index} {tile_k.name}: applied "
                        f"y=[{band[0]},{band[1]}) | own "
                        f"y=[{own_bands[j][0]},{own_bands[j][1]})")


def _current_upload_signature():
    """Stable signature of the current upload (used to cache loaded tiles)."""
    mode = st.session_state.get("upload_mode", "Multiple files")
    if mode == "ZIP archive":
        z = st.session_state.get("uploaded_zip")
        if z is None:
            return None
        return ("zip", z.name, z.size)
    files = st.session_state.get("uploaded_files") or []
    if not files:
        return None
    return ("files", tuple((f.name, f.size) for f in files))


def _get_tiles_cached():
    """Load tiles once per upload, cached in session state.

    Shared by the Stitcher and Pair Review tabs. Returns (tiles, error)
    where exactly one of the two is None.
    """
    sig = _current_upload_signature()
    if sig is None:
        return None, None
    if st.session_state.get("tiles_sig") == sig:
        cached = st.session_state.get("tiles_data")
        if isinstance(cached, dict) and cached.get("error"):
            return None, cached["error"]
        return cached, None

    mode = st.session_state.get("upload_mode", "Multiple files")
    try:
        if mode == "ZIP archive":
            zf = st.session_state["uploaded_zip"]
            data_dir = Path(tempfile.mkdtemp(prefix="micro_zip_"))
            with zipfile.ZipFile(zf) as z:
                z.extractall(data_dir)
        else:
            data_dir = Path(tempfile.mkdtemp(prefix="micro_files_"))
            for f in st.session_state["uploaded_files"]:
                (data_dir / f.name).write_bytes(f.getbuffer())
        tiles = io_utils.load_tiles(data_dir)
        st.session_state["tiles_sig"] = sig
        st.session_state["tiles_data"] = tiles
        return tiles, None
    except Exception as e:
        st.session_state["tiles_sig"] = sig
        st.session_state["tiles_data"] = {"error": str(e)}
        return None, str(e)


def _wipe_pair_widgets():
    """Remove per-pair widget state (positions / approvals) of the review."""
    for k in list(st.session_state.keys()):
        if k.startswith("rvp_"):
            del st.session_state[k]


def render_stitch_result(result, tiles):
    """Render panorama result: diagnostics, warnings, preview, downloads."""
    st.success(f"Panorama assembled: {result.canvas_size[1]}x{result.canvas_size[0]} px")

    # Pairwise diagnostics
    st.subheader("Stitching Diagnostics")
    diag_data = []
    for d in result.diagnostics:
        diag_data.append({
            "Pair": f"{d.pair_index+1} <-> {d.pair_index+2}",
            "Method": d.method,
            "Success": "yes" if d.success else "no",
            "dx": f"{d.translation[0]:.1f}",
            "dy": f"{d.translation[1]:.1f}",
            "rot_deg": f"{d.rotation_deg:.3f}",
            "scale": f"{d.scale:.4f}",
            "matches": d.matches,
            "inliers": d.inliers,
            "ncc": f"{d.score:.3f}" if d.score > 0 else "-",
            "Note": d.message,
        })
    st.dataframe(diag_data, use_container_width=True)
    if result.drift_trend is not None:
        slope, intercept = result.drift_trend
        st.caption(
            f"Vertical drift trend: {slope*1000:+.1f} px per 1000 px of horizontal "
            f"travel (intercept {intercept:+.1f} px). A smooth trend is normal — "
            "microscope stages drift vertically while moving along the row. "
            "Single pairs jumping off the trend are flagged in the warnings below.")

    if result.warnings:
        st.warning("Warnings:")
        for w in result.warnings:
            st.write(f"- {w}")

    # Panorama preview
    st.subheader("Result")
    panorama_rgb = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB)
    h, w = panorama_rgb.shape[:2]
    if max(h, w) > 4000:
        st.info(f"Large image ({w}x{h}), showing downscaled preview. "
                f"Download full size below.")
        s = 4000 / max(h, w)
        preview = cv2.resize(panorama_rgb, None, fx=s, fy=s)
    else:
        preview = panorama_rgb
    st.image(preview, use_container_width=True)

    # Tile positions
    st.subheader("Tile Position Map")
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p[0] for p in result.tile_positions],
            y=[p[1] for p in result.tile_positions],
            mode="lines+markers+text",
            text=[t.name for t in tiles],
            textposition="top center",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            hovertemplate="%{text}<br>x=%{x}, y=%{y}<extra></extra>",
        ))
        fig.update_layout(
            title="Tile offsets on the canvas (px) — a smooth curve is expected",
            xaxis_title="offset_x",
            yaxis_title="offset_y",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_yaxes(autorange="reversed")  # image coordinates
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # plotly not available — text output below still works
    positions_text = "\n".join(
        f"  {t.name}: ({pos[0]}, {pos[1]})"
        for t, pos in zip(tiles, result.tile_positions)
    )
    st.code(positions_text)

    # Download buttons
    st.subheader("Download")
    out_format = _param("out_format", "PNG")
    ext_map = {"PNG": "png", "TIFF (LZW)": "tif", "JPEG": "jpg"}
    ext = ext_map.get(out_format, "png")

    if ext == "tif":
        import tifffile
        rgb = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        tifffile.imwrite(buf, rgb, compression="lzw")
        data = buf.getvalue()
        mime = "image/tiff"
    else:
        if ext == "png":
            ok, buf = cv2.imencode(".png", result.panorama)
            mime = "image/png"
        else:
            ok, buf = cv2.imencode(".jpg", result.panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mime = "image/jpeg"
        data = buf.tobytes()

    fname = f"panorama_{len(tiles)}_tiles.{ext}"
    st.download_button(
        label=f"Download panorama ({out_format})",
        data=data,
        file_name=fname,
        mime=mime,
    )

    # Optional debug full-resolution panorama (when keep_full_panorama was ON
    # AND a single ROI band was applied — see pipeline.stitch_row docs).
    pan_full = getattr(result, "panorama_full", None)
    if pan_full is not None:
        st.caption(
            f"Debug full-resolution panorama (no ROI crop): "
            f"{pan_full.shape[1]}x{pan_full.shape[0]} px.")
        if ext == "tif":
            import tifffile
            rgb_f = cv2.cvtColor(pan_full, cv2.COLOR_BGR2RGB)
            buf_f = io.BytesIO()
            tifffile.imwrite(buf_f, rgb_f, compression="lzw")
            data_f = buf_f.getvalue()
            mime_f = "image/tiff"
        elif ext == "png":
            _, buf_f = cv2.imencode(".png", pan_full)
            data_f = buf_f.tobytes()
            mime_f = "image/png"
        else:
            _, buf_f = cv2.imencode(".jpg", pan_full,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
            data_f = buf_f.tobytes()
            mime_f = "image/jpeg"
        st.download_button(
            label=f"Download panorama (FULL — debug, {out_format})",
            data=data_f,
            file_name=f"panorama_{len(tiles)}_tiles_full.{ext}",
            mime=mime_f,
        )

    # CSV with tile positions
    import csv
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["tile_name", "tile_index", "offset_x", "offset_y", "width", "height"])
    for t, pos in zip(tiles, result.tile_positions):
        h, w = t.image.shape[:2]
        writer.writerow([t.name, t.index, pos[0], pos[1], w, h])
    st.download_button(
        label="Download tile positions (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="tile_positions.csv",
        mime="text/csv",
    )


def render_help_tab():
    """Render the Help & Usage Guide tab."""
    st.markdown("# Help & Usage Guide")

    st.markdown("## Quick Start")
    st.markdown(
        "1. **Prepare images**: name them `<prefix>_N.<ext>` where `N` is the sequential index "
        "(e.g. `sample_1.tif`, `sample_2.tif`, ..., `sample_5.tif`)."
        "\n"
        "2. **Upload**: choose multiple files or a ZIP archive in the sidebar."
        "\n"
        "3. **Run**: click **Stitch panorama** (Stitcher tab) — or use the "
        "**Pair Review** tab to check and correct every pair manually first."
        "\n"
        "4. **Download**: panorama (PNG/TIFF/JPEG) and tile positions CSV."
    )

    st.markdown("---")
    st.markdown("## Pair Review workflow (manual alignment)")
    st.markdown(
        "When a row was captured by hand, automatic alignment can be slightly "
        "off for some pairs (and a precise `Expected overlap` prior is hard to "
        "give). The **Pair Review** tab solves this interactively:"
    )
    st.markdown(
        "1. **Step 1 — Estimate**: the app estimates the shift of every adjacent "
        "pair on downscaled copies (fast, full SIFT + fallback pipeline)."
        "\n"
        "2. **Step 2 — Review**: every pair is rendered as an overlay "
        "(**Blend** 50/50, **Checkerboard**, or **Difference** map) with tile "
        "outlines A (yellow) and B (cyan). Tick **Position is correct** for "
        "pairs that look right. If two neighbouring shots were captured "
        "**without any overlap** (a common operator mistake), tick **No "
        "overlap (black gap)** instead — the pair is placed side by side "
        "with a visible black band on the panorama, and its checkbox counts "
        "as approved."
        "\n"
        "3. **Step 3 — Region of interest**: the tiles are cropped to a "
        "horizontal ROI band — auto-detected, shown on every tile and "
        "interactively adjustable."
        "\n"
        "4. **Step 4 — Adjust**: every unticked pair gets manual controls — "
        "nudge buttons (1–100 px steps) and numeric dx/dy inputs, with a live "
        "overlay that updates on every click."
        "\n"
        "5. **Step 5 — Stitch**: the panorama is blended using **exactly** the "
        "reviewed positions — approved pairs are taken as-is, manually adjusted "
        "pairs are placed exactly where you put them."
    )
    st.markdown(
        "- `dx` is the horizontal offset of tile N+1 relative to tile N: "
        "**increasing dx moves the tiles apart** (less overlap), decreasing dx "
        "pushes them closer together."
        "\n"
        "- **No overlap (black gap)** pairs: when two neighbouring shots share "
        "no common content, no algorithm can position them — any automatic "
        "estimate is a guess. Ticking **No overlap (black gap)** skips the "
        "estimate for that pair entirely: dx = tile width + gap, dy = 0, so "
        "the panorama shows a clear black separator band (gap width is "
        "configurable in Step 2) instead of a wrong forced overlap. Unticking "
        "restores the previous position and the pair goes back to the normal "
        "review flow."
        "\n"
        "- The automatic estimate is skipped entirely in the final stitch — no "
        "algorithm can override what you approved."
        "\n"
        "- **Export positions (JSON)** in Step 5 saves your work (positions, "
        "approvals, no-overlap marks); load it later in Step 1 to continue "
        "where you left off — useful because browser refreshes reset the "
        "interactive session."
        "\n"
        "- Optional *Refine approved pairs at full resolution* re-runs SIFT in "
        "a small window around approved positions for sub-pixel polish — it "
        "never touches manually adjusted pairs."
    )

    st.markdown("---")
    st.markdown("## Best Practices for Capturing Images")

    st.markdown("### Overlap")
    st.markdown(
        "Adjacent images must overlap by **20–40%** (30% is optimal). "
        "Too little overlap → SIFT cannot find enough matching keypoints. "
        "Too much overlap → unnecessary data and slower stitching."
    )

    st.markdown("### Scale Bar (Important!)")
    st.markdown(
        "TIFF files from microscope cameras typically **do not store scale information** in metadata "
        "(`XResolution`, `YResolution` are usually empty). The only way to preserve scale is to **embed "
        "the scale bar visually into one of the images** via the microscope software."
    )
    st.markdown(
        "**Where to place the scale bar** so that it survives stitching:"
    )
    st.markdown(
        "- **First image of the series**: scale bar in the **bottom-left corner**. "
        "This part of the first tile is never overlapped by the second tile, so the scale bar "
        "remains intact in the final panorama."
    )
    st.markdown(
        "- **OR last image of the series**: scale bar in the **bottom-right corner**. "
        "Same reasoning — this part of the last tile is never overlapped."
    )
    st.markdown(
        "**Avoid** placing the scale bar in the right part of intermediate tiles "
        "or the left part of any tile except the first — it will be blended away or "
        "distorted at the seam."
    )

    st.markdown("### Illumination")
    st.markdown(
        "Try to keep illumination **uniform** between shots. "
        "The `Gain compensation` setting corrects for small brightness differences, "
        "but large variations (e.g. lamp warming up) may still cause visible seams."
    )

    st.markdown("### Focus")
    st.markdown(
        "If the specimen is uneven, some frames may be partially out of focus. "
        "For typical dark-field optical microscopy the default **Feather** "
        "blending mode is the most stable choice — it averages the overlap "
        "zone between two tiles, which hides small alignment errors."
    )
    st.markdown(
        "**When to use Focus-priority blending.** If the operator focused "
        "each frame on a different depth of the cross-section and the goal "
        "is to **see the sharpest detail** at every point rather than a "
        "smoothly-averaged picture, switch the **Blending weight mode** in "
        "the sidebar to **Focus-priority (default)**. In the overlap "
        "zone, the tile whose Tenengrad focus measure is highest at a given "
        "pixel wins the high-frequency content (edges, fine detail); low "
        "frequencies are still averaged so brightness and illumination "
        "transitions stay smooth and small residual alignment errors are "
        "not revealed as visible steps. This is the same technique used by "
        "focus-stacking software (Helicon Focus, Zerene Stacker)."
    )
    st.markdown(
        "**Focus-weighted (per-tile)** is a legacy mode: it weights pixels "
        "by their sharpness *within each tile*, but does not compare "
        "sharpness across tiles. Use Focus-priority instead when you want "
        "the sharpest tile to win in the overlap."
    )

    st.markdown("### Number of Images")
    st.markdown(
        "Up to ~100 images per row are supported. "
        "For more, split the series into multiple panoramas. "
        "Memory consumption scales linearly with the number of tiles."
    )

    st.markdown("---")
    st.markdown("## Settings Reference")

    st.markdown("### Region of interest (ROI)")
    st.markdown(
        "When the visible cross-section of the TEM disk occupies only a "
        "horizontal strip of each frame (typically ~1/4 to ~1/3 of the "
        "frame height — the rest is empty microscope background), cropping "
        "each tile to that strip BEFORE stitching is the single biggest "
        "speedup available after the downscale-for-estimation: roughly "
        "(tile height / band height)x faster on SIFT/NCC/blend, with "
        "the same memory reduction on the final panorama. The crop is "
        "vertical only — horizontal overlap between adjacent tiles is "
        "preserved, so pairwise alignment is unaffected."
    )
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Crop tiles to ROI band** | ON | Master switch for the ROI "
        "workflow. OFF = stitch full tiles (legacy behaviour). |"
        "\n"
        "| **ROI detection mode** | Auto (reference tile 1) | **Auto "
        "(reference tile 1)** — DEFAULT and recommended: the app detects "
        "the textured band on tile 1 (a window of the requested height, "
        "centred on the per-row Tenengrad energy centroid of the disk "
        "band, robust to dust specks) and applies the SAME band to every "
        "tile. Because the band is identical on every tile, Pair Review "
        "positions transfer exactly — no vertical blur at the seams. "
        "The band can be shifted interactively in the Stitcher ROI "
        "preview and in Pair Review Step 3 (the same offset slider, "
        "synchronised). **Auto-detect per tile (advanced)** — per-tile "
        "bands for heavy vertical drift, each with its own offset (px) "
        "slider (the same sliders in the Stitcher ROI preview and Pair "
        "Review Step 3 stay in sync); the pipeline converts pair dy "
        "between the full-tile and cropped-tile frames "
        "automatically. |"
        "\n"
        "| **ROI height (fraction)** | 0.30 | Target band height as a "
        "fraction of tile height. 0.30 = central third (typical for a "
        "TEM disk cross-section). Smaller = faster and smaller panorama, "
        "but risks clipping the disk. Larger = safer, slower. |"
        "\n"
        "| **Also keep full panorama (debug)** | OFF | When ON AND a "
        "single ROI band is applied to every tile (Auto reference-tile 1 "
        "mode, with the interactive offset applied), the pipeline ALSO "
        "blends a full-resolution "
        "panorama without cropping, available as a separate download. "
        "Costs one EXTRA blend pass — keep OFF for production runs. "
        "Ignored (with a warning) when per-tile bands differ in the "
        "advanced mode. |"
    )
    st.markdown(
        "In the **Pair Review** tab, **Step 3** shows the reference tile "
        "(tile 1) with its ROI band drawn on top. Tune the **Vertical "
        "offset** slider until the band brackets the disk the way you "
        "want — the band HEIGHT is set by the sidebar **ROI height "
        "(fraction of tile height)** slider — and the SAME band is then "
        "applied to every tile (a preview grid shows the band on tiles "
        "2..N). In the advanced per-tile mode, each tile gets its own "
        "offset slider. The Stitcher tab also shows an **ROI band "
        "preview** expander before the Stitch button with the same "
        "sliders, so you can verify the band before spending time on a "
        "full stitch."
    )

    st.markdown("### Feature Detection")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Estimation resolution (max side, px)** | 1600 | Pairwise estimation "
        "(SIFT + NCC) runs on downscaled copies whose longest side is at most this "
        "size; the estimated shifts are scaled back up and polished at full "
        "resolution with a quick ECC run. Blending ALWAYS happens at full resolution, "
        "so the panorama quality is unchanged. 1600 px gives ~9-13x faster estimation "
        "on 20 MP frames with no visible quality loss. Choose 'Full resolution' only "
        "for tiny source images. |"
        "\n"
        "| **SIFT features (per tile)** | 2000 | Number of keypoints detected per tile. "
        "2000 is enough for typical microscope pairs; raise to 5000-10000 only if "
        "pairs fail to match. |"
        "\n"
        "| **Lowe ratio threshold** | 0.75 | Strictness of match filtering. "
        "Lower (0.7) = stricter, fewer but more reliable matches. "
        "Higher (0.85) = more matches, but more false positives. |"
        "\n"
        "| **RANSAC reprojection threshold (px)** | 5.0 | Allowed geometric error for inliers. "
        "Lower = stricter. Higher = more tolerant to noise. |"
        "\n"
        "| **Min inliers for SIFT** | 8 | If SIFT produces fewer inliers than this, "
        "the app falls back to phase correlation or NCC scan. |"
    )

    st.markdown("### Preprocessing")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Apply CLAHE** | ON | Contrast-Limited Adaptive Histogram Equalization. "
        "**Critical for dark microscope images** — boosts SIFT keypoint count by 5–10x. |"
        "\n"
        "| **CLAHE clip limit** | 3.0 | Contrast amplification strength. "
        "Increase to 4–5 for very dark images. |"
    )

    st.markdown("### Alignment Verification")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Verify SIFT with NCC scan** | ON | Every SIFT result is cross-checked "
        "against a global normalized cross-correlation scan. On samples with long "
        "horizontal scratches or sparse dots, SIFT+RANSAC can lock onto a "
        "self-consistent but **wrong** consensus (e.g. matching one scratch to the "
        "neighboring scratch). The NCC scan is immune to this and catches such "
        "blunders. Adds a few seconds per pair. |"
        "\n"
        "| **Rejection score gap** | 0.07 | SIFT is rejected when the NCC scan finds "
        "an alignment better by this margin. Increase (0.10–0.12) if SIFT answers "
        "are wrongly rejected on low-texture samples; decrease (0.04–0.05) for "
        "stricter control. |"
        "\n"
        "| **Suppress dust / fixed-pattern matches** | ON | Dust and other artifacts "
        "on the optics appear at the **same pixel position in every frame** (they do "
        "not move with the stage). Keypoints on such artifacts match each other with "
        "a ~zero shift and can steal the RANSAC consensus. They are dropped "
        "automatically when the row clearly moves. |"
        "\n"
        "| **Vertical drift guard** | ON | Microscope stages drift vertically while "
        "moving along the row — that is normal and preserved. But a single pair whose "
        "vertical shift deviates sharply from the row-wide trend (e.g. +350 px among "
        "+25 px neighbors) is usually a matching blunder. Such pairs are re-verified "
        "with a constrained correlation scan and corrected when it confirms the trend. |"
    )

    st.markdown("### Blending")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Laplacian pyramid levels** | 3 | Number of frequency bands for multi-band blending. "
        "More = smoother seams, but slower and uses more memory. 3 is a good "
        "speed/quality balance; raise to 5 if seams are visible. |"
        "\n"
        "| **Feather zone width (px)** | 64 | Smooth mask falloff at tile edges. "
        "Larger = softer seam transition. |"
        "\n"
        "| **Blending weight mode** | Focus-priority (default) | Three mutually exclusive modes. "
        "**Feather** — smooth mask, robust on dark images; both tiles contribute to the seam "
        "(averaged). **Focus-weighted (per-tile)** — pixels in sharper zones of EACH tile "
        "get higher weight; per-tile normalization means sharpness is not compared across "
        "tiles. **Focus-priority (default)** — in the overlap zone, the tile with the "
        "highest Tenengrad focus score wins the high-frequency content; low frequencies are "
        "still averaged so brightness stays smooth and residual alignment errors are not "
        "revealed. Use Focus-priority when the operator focused each frame on a different "
        "depth and the goal is to see the sharpest detail. |"
        "\n"
        "| **Focus-priority strength (gamma)** | 6.0 | Exponent on the Tenengrad score. "
        "Higher = stronger winner-takes-all (the sharpest tile contributes ~95% of the "
        "detail when its score is ~2x the runner-up). Lower = softer, more averaged. "
        "Only shown when Focus-priority is selected. |"
        "\n"
        "| **Focus-priority high-frequency levels** | 2 | How many of the top pyramid levels "
        "(finest detail) use the focus winner weight; the rest are averaged with a feather "
        "mask. Increase to push focus selection into coarser structure too (risk: reveals "
        "residual misalignment as visible steps). Only shown when Focus-priority is selected. |"
        "\n"
        "| **Gamma for focus weights** | 1.5 | Contrast of per-tile sharpness weights. "
        "Only shown when Focus-weighted (per-tile) is selected. |"
        "\n"
        "| **Gain compensation** | 0.8 | Equalizes average brightness between tiles. "
        "0 = off, 1 = full correction. Useful when illumination varies between shots. |"
    )

    st.markdown("### Backup Methods")
    st.markdown(
        "When SIFT fails to find enough inliers, the app falls back to alternative methods "
        "(in order):"
    )
    st.markdown(
        "1. **Phase correlation** — fast Fourier-based method. Works well for pure translation, "
        "fails if there is rotation or significant brightness difference."
        "\n"
        "2. **NCC scan** — brute-force search over a range of shifts using normalized "
        "cross-correlation. Slow but very robust. Used as last resort."
    )
    st.markdown(
        "Disable these if you want to enforce SIFT-only stitching (faster but less robust)."
    )

    st.markdown("---")
    st.markdown("## Troubleshooting")

    with st.expander("Some frames are shifted up or down (vertical misalignment)"):
        st.markdown(
            "**Possible causes and what the app does about them:**"
            "\n\n"
            "1. **Wrong SIFT consensus on repetitive texture** (most common). "
            "Polished cross-sections are full of long horizontal scratches; resin "
            "areas contain sparse look-alike dots (dust, pores). SIFT matches "
            "scratch-to-scratch or dot-to-dot, and RANSAC can lock onto a "
            "self-consistent but wrong consensus — one frame lands one scratch "
            "period higher/lower. **Fix**: keep `Verify SIFT with NCC scan` ON "
            "(default) — the wrong consensus is detected and replaced."
            "\n\n"
            "2. **Dust on the optics**. It does not move with the specimen, so "
            "dust-to-dust matches vote for a ~zero shift and can corrupt the "
            "estimate. **Fix**: `Suppress dust / fixed-pattern matches` ON (default). "
            "Also clean the objective lens — it is the cheapest and most effective fix."
            "\n\n"
            "3. **Vertical stage drift**. If the whole panorama is a smooth diagonal "
            "band, the stage drifted vertically while you moved it along the row — "
            "this is **genuine** and preserved by the app (the panorama simply "
            "becomes taller). Only *single pairs* jumping off the trend are treated "
            "as errors. Check the **Tile Position Map**: a smooth curve is normal, "
            "a lone jump indicates a problem pair."
            "\n\n"
            "4. **Coarse fallback quantization** (versions before the drift guard): "
            "the NCC fallback scanned shifts with a 25 px step, so dy values like "
            "25, 50, 75 were step artifacts that accumulated along the row. The "
            "current NCC scan uses a two-stage search with sub-pixel refinement "
            "plus an ECC polish."
            "\n\n"
            "5. **Flat correlation ridge on striped textures**. When the sample is "
            "dominated by long horizontal scratches / layers, the correlation "
            "surface stays almost flat along dx: any horizontal shift in a wide "
            "range fits almost equally well, so dx is only weakly constrained "
            "by the texture. The app scores every candidate shift on a "
            "fixed-width window of the overlap (so a small overlap cannot win "
            "by being easy), aligns each candidate vertically on the band "
            "structure first, then resolves the remaining ambiguity honestly: "
            "(a) if you set the `Expected overlap` in the **Capture** sidebar "
            "section — or the row already has solved pairs — the answer is "
            "anchored to that prior (`flat dx ridge — anchored to ... prior`); "
            "(b) otherwise, when the interface-line excursions (pores / dirt "
            "specks / bumps — the only non-repetitive features of such a "
            "sample) form a decisive \"booking\" in which every excursion "
            "finds a counterpart in the other tile while dust fixed to the "
            "optics is excluded, that booking corrects dx; (c) if neither "
            "works, the best texture agreement is kept and the diagnostics "
            "show `WARNING: dx weakly constrained` — verify that seam "
            "visually."
        )

    with st.expander("Dark zone appears at the edge of the panorama"):
        st.markdown(
            "**Cause**: The `Focus-weighted (per-tile)` blending mode is enabled, "
            "and the dark microscope images produce near-zero sharpness values, "
            "causing blending to fail in non-overlap zones."
            "\n\n"
            "**Fix**: Switch **Blending weight mode** to **Focus-priority (default)** "
            "or **Feather**. Both are safe on dark images — Focus-priority always "
            "falls back to a feather mask in non-overlap zones; the Focus-weighted "
            "(per-tile) mode is a legacy option kept only for backward compatibility."
        )

    with st.expander("\"Unable to allocate ... GiB for an array\" (Out of Memory)"):
        st.markdown(
            "**Cause**: The canvas is too large for available RAM."
            "\n\n"
            "**Fixes**:"
            "\n"
            "1. **Enable the ROI crop** (sidebar: Region of interest → "
            "Crop tiles to ROI band). On 20 MP frames whose disk occupies ~1/4 "
            "of the frame height, the panorama canvas shrinks ~4x in height — "
            "and the per-tile Laplacian/weight pyramids shrink with it."
            "\n"
            "2. Reduce `Laplacian pyramid levels` to 3–4."
            "\n"
            "3. Keep **Blending weight mode** on **Feather** — the legacy "
            "Focus-weighted mode and the Focus-priority mode each build one extra "
            "per-tile weight pyramid (small, tile-sized, but non-zero)."
            "\n"
            "4. For very large sets (>30 tiles), split into multiple panoramas."
        )

    with st.expander("ROI band missed the disk on some tile"):
        st.markdown(
            "**Symptom**: In the Pair Review tab, Step 3 shows the yellow ROI "
            "rectangle not bracketing the visible disk cross-section on one "
            "or more tiles."
            "\n\n"
            "**Fix**: Use the **Vertical offset correction** slider under that "
            "tile in Step 3 to shift the band up or down by up to ±half the "
            "tile height. The auto-detect uses a Tenengrad energy profile and "
            "occasionally picks the wrong band on a frame that is dominated by "
            "background noise or by a single bright scratch; the manual slider "
            "covers that case without forcing a full re-detect."
            "\n\n"
            "If the disk is genuinely in a different position on every tile "
            "(operator hand-positioned the stage without locking Y), keep "
            "**ROI detection mode** on **Auto-detect per tile (advanced)** so "
            "each tile "
            "gets its own band — combined with the per-tile offset sliders you "
            "can recover the disk on every frame."
        )

    with st.expander("Visible seam between tiles"):
        st.markdown(
            "**Fixes**:"
            "\n"
            "1. Increase `Laplacian pyramid levels` to 6–7."
            "\n"
            "2. Increase `Feather zone width` to 100–150."
            "\n"
            "3. Increase `Gain compensation` to 1.0 if brightness differs between tiles."
        )

    with st.expander("All stitching methods failed for a pair"):
        st.markdown(
            "**Fixes**:"
            "\n"
            "1. Verify the two images actually overlap (at least 20%)."
            "\n"
            "2. Increase `SIFT features` to 10000."
            "\n"
            "3. Lower `Lowe ratio threshold` to 0.7."
            "\n"
            "4. Increase `CLAHE clip limit` to 4–5 for very dark images."
        )

    with st.expander("Stitching is slow (>30 seconds)"):
        st.markdown(
            "**Fixes**:"
            "\n"
            "1. Reduce `SIFT features` to 2000–3000."
            "\n"
            "2. Reduce `Laplacian pyramid levels` to 4."
            "\n"
            "3. Disable `NCC scan fallback` (it is the slowest backup method)."
        )

    with st.expander("Cannot extract index from filename"):
        st.markdown(
            "**Cause**: Filenames must end with `_N` where N is an integer."
            "\n\n"
            "**Examples**:"
            "\n"
            "- ✅ `sample_1.tif`, `sample_2.tif`, ..., `sample_5.tif`"
            "\n"
            "- ✅ `x10_001.tif`, `x10_002.tif`, ..., `x10_015.tif`"
            "\n"
            "- ❌ `sample.tif` (no index)"
            "\n"
            "- ❌ `sample_1a.tif` (non-numeric suffix)"
        )

    st.markdown("---")
    st.markdown("## How It Works")
    st.markdown(
        "1. **Load** N images sorted by `_N` index."
        "\n"
        "2. **CLAHE preprocessing** — boost contrast for keypoint detection."
        "\n"
        "3. **Pairwise stitching** for each `i ↔ i+1` pair:"
        "\n"
        "   - SIFT detector + FLANN matcher"
        "\n"
        "   - Lowe ratio test → good matches (fixed-pattern / dust matches dropped)"
        "\n"
        "   - RANSAC + `estimateAffinePartial2D` → translation + small rotation"
        "\n"
        "   - **NCC verification**: the SIFT answer is cross-checked against a global "
        "correlation scan; a wrong consensus on repetitive textures is rejected"
        "\n"
        "   - Fallbacks: phase correlation → NCC scan (two-stage, sub-pixel + ECC polish)"
        "\n"
        "4. **Vertical drift guard** — pairwise shifts that deviate sharply from the "
        "row-wide drift trend are re-verified and corrected."
        "\n"
        "5. **Cumulative position calculation** — chain pairwise translations."
        "\n"
        "6. **Multi-band blending** with focus-aware feather masks:"
        "\n"
        "   - Laplacian pyramid (5 levels by default)"
        "\n"
        "   - Weighted average per level"
        "\n"
        "   - Gain compensation between tiles"
        "\n"
        "7. **Output**: panorama + diagnostics table + tile position map + CSV."
    )

    st.markdown("---")
    st.markdown("## Limitations")
    st.markdown(
        "- **1D row only** — 2D grids are not supported yet."
        "\n"
        "- **Up to ~100 tiles** — for larger sets, split into multiple panoramas."
        "\n"
        "- **Translation + small rotation** — significant perspective distortion is not handled."
        "\n"
        "- **Scale bar must be embedded visually** — TIFF metadata is not used for scale."
    )


def render_sidebar() -> None:
    """Render the shared sidebar (Upload + Method Settings) on EVERY tab.

    Why always render it:
    1. Streamlit deletes a widget's session-state entry when the widget is
       not rendered in the current run. With the Upload expander living only
       in the Stitcher renderer, switching to Pair Review (a) left the
       sidebar completely empty — although the main area said "Upload images
       or a ZIP archive in the sidebar to begin" — and (b) silently wiped
       the already-uploaded files, so going back to Stitcher showed the
       empty upload again.
    2. Pair Review's Step 5 stitches with the Method Settings read via
       ``_param()``; if those widgets do not render, the defaults are used
       and the user cannot change them. Always-on sidebar makes the settings
       genuinely shared between the Stitcher and Pair Review workflows.
    """
    st.sidebar.header("Method Settings")

    with st.sidebar.expander("Upload", expanded=True):
        upload_mode = st.radio(
            "Upload method",
            ["Multiple files", "ZIP archive"],
            index=0,
            key="upload_mode",
        )
        if upload_mode == "Multiple files":
            st.file_uploader(
                "Select images",
                type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="uploaded_files",
            )
        else:
            st.file_uploader(
                "Select ZIP archive",
                type=["zip"],
                accept_multiple_files=False,
                label_visibility="collapsed",
                key="uploaded_zip",
            )

    with st.sidebar.expander("Capture (optional)", expanded=False):
        st.slider(
            "Expected overlap between adjacent frames (%)", 0, 80, 0, step=5,
            key="expected_overlap",
            help="Approximate overlap you aim for when capturing a row of "
                 "frames. Used ONLY when a pair's horizontal shift is "
                 "ambiguous (flat correlation ridge on layered/scratched "
                 "samples): the app then prefers the alignment that matches "
                 "this overlap. 0 = unknown — fully automatic.")

    with st.sidebar.expander("Region of interest (ROI)", expanded=False):
        st.checkbox(
            "Crop tiles to a horizontal ROI band before stitching",
            value=True, key="roi_enabled",
            help=(
                "When ON, each tile is vertically cropped to a horizontal "
                "band that brackets the visible part of the cross-section "
                "(the TEM disk) BEFORE SIFT / NCC / blending run. Saves "
                "roughly (tile height / band height)x on time, memory and "
                "final panorama size. Only the Y axis is cropped — "
                "horizontal overlap between adjacent tiles is preserved, "
                "so pairwise alignment is unaffected. The Pair Review tab "
                "shows the auto-detected band on every tile and lets you "
                "correct the offset per tile if the disk moved vertically "
                "between shots. OFF = stitch full tiles (legacy behaviour)."
            ),
        )
        # Legacy sessions may still hold the removed "Manual ..." mode;
        # drop it so the radio falls back to its default option.
        if str(st.session_state.get("roi_mode", "")).startswith("Manual"):
            st.session_state.pop("roi_mode", None)
        st.radio(
            "ROI detection mode",
            options=[_ROI_MODE_REF1, _ROI_MODE_PER_TILE],
            index=0, key="roi_mode",
            help=(
                "Auto (reference tile 1) \u2014 DEFAULT and recommended: "
                "the app detects the textured band on the FIRST tile "
                "and applies the SAME band to every tile. You can "
                "verify / correct it visually in Pair Review Step 3 "
                "(or in the Stitcher ROI preview, which has the same "
                "offset slider right above the tile-1 preview). Because "
                "the band is identical on every tile, Pair Review "
                "translations transfer exactly \u2014 no vertical blur "
                "at the seams. Auto-detect per tile (advanced): "
                "each tile gets its own band (use ONLY when the disk "
                "moved vertically a LOT between shots \u2014 the per-tile "
                "dy is then converted automatically, but the safest "
                "workflow is still reference tile 1)."
            ),
        )
        st.slider(
            "ROI height (fraction of tile height)", 0.15, 1.0, 0.30,
            step=0.05, key="roi_height_fraction",
            help=(
                "Target band height as a fraction of the tile height. "
                "0.25 = the central quarter (typical: the disk occupies "
                "~1/4 of the frame height). Smaller = faster, smaller "
                "panorama, but risks clipping the disk if the operator "
                "moved it vertically. Larger = safer, slower."
            ),
        )
        st.checkbox(
            "Also keep full panorama (debug)",
            value=False, key="keep_full_panorama",
            help=(
                "When ON AND a single ROI band is applied to every tile "
                "(Auto (reference tile 1) mode, OR Auto-detect per tile "
                "where all bands ended up identical), the pipeline ALSO "
                "blends a full-resolution panorama without cropping, "
                "returned as result.panorama_full. Costs one EXTRA blend "
                "pass — keep OFF for production runs. Ignored (with a "
                "warning) when the per-tile bands differ (the per-tile dy "
                "conversion is incompatible with the full-tile "
                "coordinate frame)."
            ),
        )

    with st.sidebar.expander("Feature Detection", expanded=True):
        st.selectbox(
            "Estimation resolution (max side, px)",
            options=["Full resolution", 2000, 1600, 1200, 800],
            index=2,
            key="estimation_max_side",
            help=(
                "Pairwise estimation (SIFT + NCC) runs on downscaled "
                "copies whose longest side is at most this size; the "
                "estimated shifts are scaled back up and polished at full "
                "resolution with a quick ECC run. Blending ALWAYS happens "
                "at full resolution, so the panorama quality is unchanged. "
                "1600 px gives ~9-13x faster estimation on 20 MP frames "
                "with no visible quality loss. Choose 'Full resolution' "
                "only for tiny source images."
            ),
        )
        st.slider("SIFT features (per tile)", 500, 10000, 2000, step=500,
                  key="n_features",
                  help="More = more robust but slower. 2000 is enough for "
                       "typical microscope pairs; raise to 5000+ only if "
                       "pairs fail to match.")
        st.slider("Lowe ratio threshold", 0.5, 0.9, 0.75, step=0.05,
                  key="ratio_thresh",
                  help="Lower = stricter match filtering. 0.75 is standard.")
        st.slider("RANSAC reprojection threshold (px)", 1.0, 15.0, 5.0, step=0.5,
                  key="ransac_thresh",
                  help="Allowed inlier error. 5 px is standard.")
        st.slider("Min inliers for SIFT", 4, 50, 8, step=1,
                  key="min_inliers",
                  help="Below this — fall back to alternative methods.")

    with st.sidebar.expander("Preprocessing", expanded=False):
        st.checkbox("Apply CLAHE (recommended)", value=True,
                    key="use_clahe",
                    help="Adaptive histogram equalization. Critical for dark microscope images.")
        st.slider("CLAHE clip limit", 1.0, 8.0, 3.0, step=0.5, key="clahe_clip")

    with st.sidebar.expander("Blending", expanded=False):
        st.slider("Laplacian pyramid levels", 2, 8, 3, step=1,
                  key="blend_levels",
                  help="More = smoother seams but slower. 3 is a good "
                       "speed/quality balance for microscope rows; raise "
                       "to 5 if seams are visible.")
        st.slider("Feather zone width (px)", 0, 200, 64, step=8,
                  key="blend_feather",
                  help="Smooth mask falloff toward tile edges.")

        # Unified weight-mode selector. Three mutually exclusive modes:
        # - Feather:               stable baseline, works on dark images
        # - Focus-weighted:         legacy per-tile sharpness weighting
        # - Focus-priority (DEFAULT): cross-tile winner-takes-all at high
        #                          frequencies — keeps the sharpest tile's
        #                          detail in the overlap zone
        st.radio(
            "Blending weight mode",
            options=[
                "Feather",
                "Focus-weighted (per-tile)",
                "Focus-priority (default)",
            ],
            index=2,
            key="blend_weight_mode",
            help=(
                "Feather: smooth mask, robust on dark images; both tiles "
                "contribute to the seam (averaged). "
                "Focus-weighted: pixels in sharper zones of EACH tile get "
                "higher weight (per-tile normalization, does not compare "
                "across tiles). "
                "Focus-priority (DEFAULT): in the overlap zone, the tile "
                "with the highest Tenengrad sharpness score wins the "
                "high-frequency content (edges, fine detail); low "
                "frequencies are still averaged so brightness stays smooth "
                "and residual alignment errors are not revealed. Use this "
                "when the operator focused each frame on a different depth "
                "and the goal is to see the sharpest detail, not a pretty "
                "averaged picture."
            ),
            horizontal=False,
        )

        mode = st.session_state.get("blend_weight_mode", "Focus-priority (default)")
        if mode == "Focus-weighted (per-tile)":
            st.slider("Gamma for focus weights", 0.5, 3.0, 1.5, step=0.1,
                      key="blend_gamma",
                      help=">1 — sharp zones dominate more within each tile.")
        elif mode == "Focus-priority (default)":
            st.slider(
                "Focus-priority strength (gamma)", 2.0, 12.0, 6.0, step=0.5,
                key="focus_priority_gamma",
                help=(
                    "Exponent on the Tenengrad sharpness score used as the "
                    "blend weight at high-frequency pyramid levels. "
                    "Higher = stronger winner-takes-all: the sharpest tile "
                    "contributes ~95% of the detail when its score is ~2x "
                    "the runner-up. Lower = softer, more averaged. "
                    "Default 6 is a good starting point."
                ),
            )
            st.slider(
                "Focus-priority high-frequency levels", 1, 4, 2, step=1,
                key="focus_priority_levels",
                help=(
                    "How many of the TOP pyramid levels (finest detail) "
                    "use the focus winner weight; the rest are averaged "
                    "with a feather mask so brightness and coarse structure "
                    "stay smooth. Default 2 = the two highest-frequency "
                    "bands are chosen by focus. Increase to push the "
                    "focus selection into coarser structure too (risk: "
                    "reveals residual misalignment as visible steps)."
                ),
            )

        st.slider("Gain compensation", 0.0, 1.0, 0.8, step=0.1,
                  key="gain_strength",
                  help="Brightness equalization between tiles. "
                       "0 = off, 1 = full correction.")

    with st.sidebar.expander("Backup Methods", expanded=False):
        st.checkbox("Phase correlation fallback", value=True, key="enable_phase")
        st.checkbox("NCC scan fallback (slow)", value=True, key="enable_ncc")

    with st.sidebar.expander("Alignment Verification", expanded=False):
        st.checkbox(
            "Verify SIFT with NCC scan", value=True, key="verify_with_ncc",
            help="Cross-checks every SIFT result against a global correlation scan. "
                 "Protects against wrong SIFT consensus on samples with repetitive "
                 "scratches or dust dots. Recommended ON. Adds a few seconds per pair.")
        st.slider(
            "Rejection score gap", 0.02, 0.15, 0.07, step=0.01, key="ncc_verify_gap",
            help="SIFT is rejected when the NCC scan finds an alignment better "
                 "by this margin. 0.07 is a balanced value.")
        st.checkbox(
            "Suppress dust / fixed-pattern matches", value=True,
            key="suppress_fixed_pattern",
            help="Dust on the optics stays at the same pixel position in every "
                 "frame. Matches between such artifacts vote for a ~zero shift "
                 "and can corrupt RANSAC — they are dropped automatically.")
        st.checkbox(
            "Vertical drift guard", value=True, key="enable_drift_guard",
            help="Detects pairs whose vertical shift deviates sharply from the "
                 "row-wide drift trend and re-verifies them with a constrained "
                 "correlation scan. Fixes single frames shifted up or down.")

    with st.sidebar.expander("Output Format", expanded=False):
        st.selectbox("Output format", ["PNG", "TIFF (LZW)", "JPEG"], index=0,
                     key="out_format")


def render_stitcher_tab():
    """Render the main Stitcher tab.

    The sidebar (Upload + Method Settings) is rendered separately by
    ``render_sidebar()`` and is shared with the Pair Review tab.
    """
    st.title("Microscope Image Stitcher")
    st.caption(
        "Upload a series of images with `_N` suffix (N = sequential index). "
        "The app automatically determines their relative positions and stitches a panorama."
    )

    # === Main content ===
    tiles, load_error = _get_tiles_cached()

    if tiles is None:
        if load_error:
            st.error(f"Loading error: {load_error}")
            st.error("Verify that filenames contain `_N` (e.g. `sample_1.tif`).")
            return
        st.info("Upload images or a ZIP archive in the sidebar to begin.")
        st.markdown("### Expected filename format")
        st.markdown(
            "Files must be named `<name>_N.<ext>`, where `N` is the sequential index."
            "\n\n"
            "Example: `x10_1.tif`, `x10_2.tif`, ..., `x10_5.tif`."
        )
        st.markdown("### How it works")
        st.markdown(
            "1. **CLAHE** — adaptive brightness equalization (important for dark microscope images)"
            "\n"
            "2. **SIFT + RANSAC** — keypoint detection and affine transform estimation"
            "\n"
            "3. **Multi-band blending** — invisible seams between tiles"
            "\n"
            "4. **Gain compensation** — brightness equalization across tiles"
        )
        st.markdown("See the **Help** tab for detailed usage guide and best practices.")
        return

    st.success(f"Loaded {len(tiles)} images")
    st.write(f"Stitching order: {' -> '.join(t.name for t in tiles)}")

    # Preview uploaded tiles
    with st.expander("Preview uploaded tiles", expanded=False):
        cols = st.columns(min(len(tiles), 5))
        for i, (col, tile) in enumerate(zip(cols, tiles)):
            with col:
                st.caption(f"#{tile.index}: {tile.name}")
                preview = tile.image
                h, w = preview.shape[:2]
                if max(h, w) > 600:
                    s = 600 / max(h, w)
                    preview = cv2.resize(preview, None, fx=s, fy=s)
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

    # ROI preview (before the Stitch button) — visual feedback so the user
    # can see WHICH band will be cropped before spending time on a stitch.
    roi_enabled_preview = bool(_param("roi_enabled", True))
    if roi_enabled_preview:
        with st.expander("ROI band preview (applied before stitching)",
                         expanded=True):
            if _param("roi_mode", _ROI_MODE_REF1) == _ROI_MODE_PER_TILE:
                _render_roi_preview_stitcher(
                    tiles, float(_param("roi_height_fraction", 0.30)))
            else:
                _render_stitcher_roi_reference_ui(
                    tiles, float(_param("roi_height_fraction", 0.30)))

    # Run button
    if not st.button("Stitch panorama", type="primary"):
        return

    # Progress
    progress = st.progress(0.0, "Initializing...")

    def update_progress(msg, p):
        progress.progress(min(max(p, 0.0), 1.0), msg)

    # Run stitching (settings come from the shared sidebar via _param)
    try:
        eo = _param("expected_overlap", 0)
        ems = _param("estimation_max_side", 1600)
        roi_band, roi_bands_per_tile, roi_label, keep_full = \
            _resolve_roi_band_for_stitcher(tiles)
        result = pipeline.stitch_row(
            tiles,
            n_features=_param("n_features", 2000),
            ratio_thresh=_param("ratio_thresh", 0.75),
            ransac_thresh=_param("ransac_thresh", 5.0),
            use_clahe=_param("use_clahe", True),
            clahe_clip=_param("clahe_clip", 3.0),
            min_inliers_sift=_param("min_inliers", 8),
            enable_phase_fallback=_param("enable_phase", True),
            enable_ncc_fallback=_param("enable_ncc", True),
            verify_with_ncc=_param("verify_with_ncc", True),
            ncc_verify_gap=_param("ncc_verify_gap", 0.07),
            suppress_fixed_pattern=_param("suppress_fixed_pattern", True),
            enable_drift_guard=_param("enable_drift_guard", True),
            expected_overlap_pct=(eo if eo > 0 else None),
            estimation_max_side=(ems if isinstance(ems, int) else None),
            blend_levels=_param("blend_levels", 3),
            blend_feather=_param("blend_feather", 64),
            blend_gamma=_param("blend_gamma", 1.5),
            blend_gain_strength=_param("gain_strength", 0.8),
            **_blend_mode_params(),
            roi_band=roi_band,
            roi_bands_per_tile=roi_bands_per_tile,
            keep_full_panorama=keep_full,
            progress_callback=update_progress,
        )
    except Exception as e:
        st.error(f"Stitching error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    render_stitch_result(result, tiles)


def render_pair_review_tab():
    """Render the interactive Pair Review tab (manual alignment workflow)."""
    st.title("Pair Review — Verify & Adjust Pairs Manually")
    st.caption(
        "Interactive workflow: get a fast automatic estimate for every adjacent "
        "pair, visually check how each pair will be positioned, approve the good "
        "pairs, nudge the bad ones by hand in X and Y, then stitch the panorama "
        "using exactly these positions. "
        "The `Expected overlap` prior is NOT needed here — your eyes decide.")

    tiles, load_error = _get_tiles_cached()
    if tiles is None:
        if load_error:
            st.error(f"Loading error: {load_error}")
            st.error("Verify that filenames contain `_N` (e.g. `sample_1.tif`).")
        else:
            st.info("Upload images or a ZIP archive in the sidebar to begin.")
        return

    st.success(f"Loaded {len(tiles)} images")
    st.write(f"Review order: {' -> '.join(t.name for t in tiles)}")

    n = len(tiles)
    if n < 2:
        st.warning("At least 2 images are required.")
        return
    n_pairs = n - 1
    tile_names = [t.name for t in tiles]
    sig = _current_upload_signature()

    # =================== Step 1 — estimate ===================
    st.subheader("Step 1 — Automatic position estimate")
    st.caption(
        "Fast estimate on downscaled copies (the full SIFT + fallback pipeline, "
        "a few seconds for the whole row). Positions are reported in "
        "full-resolution pixels. You verify and correct them visually below.")

    c1, _ = st.columns([1, 2])
    with c1:
        max_side = st.selectbox(
            "Review resolution (max side, px)",
            [800, 1200, 1600, 2200], index=1, key="rv_max_side",
            help="Lower = faster estimation and previews. 1600-2200 gives a "
                 "more precise initial estimate at the cost of speed.")

    estimate_clicked = st.button("Estimate pair positions", type="primary",
                                 key="rv_estimate")

    if estimate_clicked:
        progress = st.progress(0.0, "Estimating...")

        def est_cb(msg, p):
            progress.progress(min(max(p, 0.0), 1.0), msg)

        try:
            imgs, scale = pair_review.downscale_tiles(
                [t.image for t in tiles], max_side)
            _wipe_pair_widgets()
            eo = _param("expected_overlap", 0)
            exp_dx_full = (tiles[0].image.shape[1] * (1.0 - eo / 100.0)
                           if eo > 0 else None)
            records = pair_review.estimate_pairs(
                imgs, scale,
                n_features=_param("n_features", 2000),
                ratio_thresh=_param("ratio_thresh", 0.75),
                ransac_thresh=_param("ransac_thresh", 5.0),
                use_clahe=_param("use_clahe", True),
                clahe_clip=_param("clahe_clip", 3.0),
                min_inliers_sift=_param("min_inliers", 8),
                enable_phase_fallback=_param("enable_phase", True),
                enable_ncc_fallback=_param("enable_ncc", True),
                verify_with_ncc=False,   # the user is the verifier here
                ncc_verify_gap=_param("ncc_verify_gap", 0.07),
                suppress_fixed_pattern=_param("suppress_fixed_pattern", True),
                expected_dx_full=exp_dx_full,
                progress_cb=est_cb,
            )
            st.session_state["review_state"] = {
                "sig": sig, "max_side": max_side, "scale": scale,
                "imgs": imgs, "pairs": records,
            }
            progress.progress(1.0, "Estimation complete")
        except Exception as e:
            st.error(f"Estimation error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Optional import of a previously saved review session
    with st.expander("Load positions from a saved review JSON (optional)"):
        imp = st.file_uploader("Review JSON file", type=["json"], key="rv_import")
        if imp is not None:
            try:
                imported = pair_review.apply_json(imp.getvalue(), tile_names)
            except ValueError as e:
                st.error(str(e))
            else:
                review_now = st.session_state.get("review_state")
                if (review_now is None or review_now.get("sig") != sig):
                    imgs, scale = pair_review.downscale_tiles(
                        [t.image for t in tiles], max_side)
                    records = []
                    for i, p in enumerate(imported):
                        records.append({
                            "pair_index": i, "dx": p["dx"], "dy": p["dy"],
                            "dx0": p["dx"], "dy0": p["dy"],
                            "method": "imported", "success": True,
                            "score": 0.0, "inliers": 0,
                            "message": p.get("note") or "imported from JSON",
                        })
                    _wipe_pair_widgets()
                    st.session_state["review_state"] = {
                        "sig": sig, "max_side": max_side, "scale": scale,
                        "imgs": imgs, "pairs": records,
                    }
                else:
                    _wipe_pair_widgets()
                    for i, p in enumerate(imported):
                        if i < len(review_now["pairs"]):
                            review_now["pairs"][i].update({
                                "dx": p["dx"], "dy": p["dy"],
                                "dx0": p["dx"], "dy0": p["dy"],
                                "method": "imported",
                                "message": p.get("note") or "imported from JSON",
                            })
                for i, p in enumerate(imported):
                    if i >= n_pairs:
                        break
                    st.session_state[f"rvp_ok_{i}"] = p["approved"]
                    st.session_state[f"rvp_no_{i}"] = bool(
                        p.get("no_overlap", False))
                    st.session_state[f"rvp_dx_{i}"] = int(round(p["dx"]))
                    st.session_state[f"rvp_dy_{i}"] = int(round(p["dy"]))
                st.success(f"Imported {len(imported)} pair positions from JSON.")

    review = st.session_state.get("review_state")
    if review is None:
        st.info("Run Step 1 to estimate all pair positions — or import a "
                "previously saved review JSON above.")
        return
    if review.get("sig") != sig:
        st.warning("The uploaded images changed since the last estimate — "
                   "please run Step 1 again.")
        return

    imgs = review["imgs"]
    scale = review["scale"]
    pairs = review["pairs"]
    w_full = int(tiles[0].image.shape[1])
    h_full = int(tiles[0].image.shape[0])

    mode_label = st.radio(
        "Preview mode",
        ["Blend", "Checkerboard", "Difference", "Side-by-side (static)"],
        horizontal=True, key="rv_mode",
        help=("Blend: 50/50 mix in the overlap. Checkerboard: alternating "
              "squares of both tiles — the classic alignment inspector. "
              "Difference: |A-B| map, dark = perfectly aligned. "
              "Side-by-side (static): both tiles shown in full next to "
              "each other with seam markers — use when the overlay "
              "hides a mismatch (one tile's content averages over the "
              "other's, so a wrong dx/dy can look 'fine')."))
    static_mode = mode_label == "Side-by-side (static)"
    mode = {"Blend": "blend", "Checkerboard": "checker",
            "Difference": "difference"}[mode_label if not static_mode
                                         else "Blend"]

    # =================== Step 2 — review grid ===================
    st.subheader("Step 2 — Review every pair")
    qa1, qa2, qa3, _ = st.columns([1, 1, 1, 2])
    if qa1.button("Approve all pairs", key="rv_approve_all"):
        for i in range(n_pairs):
            st.session_state[f"rvp_ok_{i}"] = True
    if qa2.button("Clear all approvals", key="rv_clear_all"):
        for i in range(n_pairs):
            st.session_state[f"rvp_ok_{i}"] = False
    with qa3:
        st.number_input(
            "Gap width (px)",
            min_value=5, max_value=1000, value=pair_review.DEFAULT_GAP_PX,
            step=5, key="rv_gap_px",
            help="Width of the black band inserted between the tiles of a "
                 "pair marked 'No overlap (black gap)'. Applied at the "
                 "moment the checkbox is ticked; the pair's dx can still be "
                 "fine-tuned in Step 4 afterwards.")

    st.caption(
        "Tick **Position is correct** under each pair that is properly aligned. "
        "If two neighbouring shots were captured **without any overlap** (the "
        "automatic estimate is meaningless for them), tick **No overlap (black "
        "gap)** instead — the pair is placed side by side with a visible black "
        "band on the panorama. Unticked pairs are offered for manual adjustment "
        "in Step 4.")

    def _on_no_overlap(i_: int, rec_: dict) -> None:
        """on_change callback for the 'No overlap (black gap)' checkbox.

        Runs at the START of the rerun, before the rvp_* widgets are
        instantiated, so mutating session_state here is legal."""
        pair_review.apply_no_overlap_flag(
            st.session_state, i_, rec_,
            tile_width=w_full,
            gap_px=int(st.session_state.get("rv_gap_px",
                                            pair_review.DEFAULT_GAP_PX)),
        )

    def _on_pos_change(i_: int) -> None:
        """on_change callback for the dx/dy number inputs: sync the widget
        value into review_state['pairs'] — the non-widget source of truth
        (Streamlit deletes widget keys whose widgets stop rendering, e.g.
        after a pair is approved, so positions must live in the record)."""
        review_now = st.session_state.get("review_state")
        if review_now is None or i_ >= len(review_now["pairs"]):
            return
        rec_ = review_now["pairs"][i_]
        if f"rvp_dx_{i_}" in st.session_state:
            rec_["dx"] = float(st.session_state[f"rvp_dx_{i_}"])
        if f"rvp_dy_{i_}" in st.session_state:
            rec_["dy"] = float(st.session_state[f"rvp_dy_{i_}"])

    grid_cols = st.columns(2)
    for i in range(n_pairs):
        with grid_cols[i % 2]:
            rec = pairs[i]
            dx_cur = float(rec.get("dx", rec.get("dx0", 0.0)))
            dy_cur = float(rec.get("dy", rec.get("dy0", 0.0)))
            no_ovl_i = bool(st.session_state.get(f"rvp_no_{i}", False))
            if static_mode:
                prev = pair_review.render_pair_preview_static(
                    imgs[i], imgs[i + 1],
                    dx_cur * scale, dy_cur * scale,
                    out_width=470)
            else:
                prev = pair_review.render_pair_preview(
                    imgs[i], imgs[i + 1],
                    dx_cur * scale, dy_cur * scale,
                    mode=mode, out_width=470)
            st.image(prev, use_container_width=True)
            st.markdown(f"**Pair {i+1} ↔ {i+2}** — `{tile_names[i]}` → "
                        f"`{tile_names[i+1]}`")
            ovl = max(0.0, 100.0 * (w_full - abs(dx_cur)) / w_full)
            caps = [f"dx = {dx_cur:.1f} px", f"dy = {dy_cur:.1f} px",
                    f"x-overlap ≈ {ovl:.0f} %", f"method: {rec['method']}"]
            if rec.get("inliers"):
                caps.append(f"{rec['inliers']} inliers")
            if not rec.get("success", True):
                caps.append("AUTOMATIC ESTIMATE FAILED")
            if no_ovl_i:
                caps.insert(0, "NO OVERLAP — black gap")
            st.caption(" · ".join(caps))
            okc, noc = st.columns(2)
            okc.checkbox("Position is correct", key=f"rvp_ok_{i}")
            noc.checkbox(
                "No overlap (black gap)", key=f"rvp_no_{i}",
                on_change=_on_no_overlap, args=(i, rec),
                help="Tick if these two shots were captured WITHOUT any "
                     "overlap: the tiles are placed side by side with a "
                     "black band between them on the panorama (dx = tile "
                     "width + gap, dy = 0). Unticking restores the previous "
                     "position.")

    # =================== Step 3 — ROI band ===================
    # Reference-tile workflow: the FIRST tile is shown with its auto-detected
    # ROI band, the user tunes vertical offset / height sliders until the
    # band brackets the disk cross-section the way they want, and then THE
    # SAME band is applied to every tile in the row. Because the band is
    # identical on every tile, Pair Review translations transfer exactly
    # (no per-tile dy conversion), which keeps the seams sharp.
    roi_enabled = bool(_param("roi_enabled", True))
    if roi_enabled:
        mode_roi = _param("roi_mode", _ROI_MODE_REF1)
        h_frac = float(_param("roi_height_fraction", 0.30))
        if mode_roi == _ROI_MODE_PER_TILE:
            st.subheader("Step 3 — Region of interest")
        else:
            st.subheader("Step 3 — Region of interest (reference tile 1)")
            st.caption(
                "The FIRST tile is shown with its ROI band highlighted "
                "(yellow rectangle). Tune the **Vertical offset** slider "
                "until the band brackets the visible disk cross-section "
                "the way you want \u2014 the SAME band is then applied to "
                "every tile in the row (preview grid below). The band "
                "HEIGHT is set by the sidebar **ROI height (fraction of "
                "tile height)** slider. Because the band is identical on "
                "every tile, Pair Review positions transfer exactly: no "
                "vertical blur at the seams. The SAME offset slider is "
                "available in the Stitcher tab ROI preview \u2014 the two "
                "stay in sync.")

        # Lazily compute + cache the reference band. The cache signature
        # includes the sidebar ROI settings (mode / height) so a sidebar
        # change recomputes the band IMMEDIATELY (no page reload needed)
        # as well as the upload signature.
        rv_state = st.session_state.get("review_state", {})
        roi_sig = (sig, mode_roi, h_frac)
        auto_bands_full = rv_state.get("roi_auto_bands_full")
        auto_bands_sig = rv_state.get("roi_auto_bands_sig")
        if auto_bands_full is None or auto_bands_sig != roi_sig:
            if mode_roi == _ROI_MODE_PER_TILE:
                ref_band_init = None
                auto_bands_full = io_utils.roi_band_for_tiles(
                    tiles, mode="auto", target_fraction=h_frac,
                )
            else:
                # Auto (reference tile 1) — DEFAULT. The min/max bounds
                # mirror roi_band_for_tiles so the sidebar fraction is
                # honoured up to 1.0 of the tile height (not capped at
                # 0.5 like the helper's defaults).
                lo, hi = _auto_band_bounds(h_frac)
                ref_band_init = io_utils.auto_detect_roi_band(
                    tiles[0].image, target_fraction=h_frac,
                    min_fraction=lo, max_fraction=hi)
            if ref_band_init is not None:
                auto_bands_full = [ref_band_init] * n
            rv_state["roi_auto_bands_full"] = auto_bands_full
            rv_state["roi_auto_bands_sig"] = roi_sig
            st.session_state["review_state"] = rv_state

        if mode_roi == _ROI_MODE_PER_TILE:
            # ADVANCED per-tile path — kept for rows with heavy vertical
            # drift between shots. The pipeline converts pair dy between
            # the full-tile and cropped-tile frames automatically. The
            # per-tile offset sliders share ONE state with the Stitcher
            # tab's per-tile ROI preview (synchronised by on_change
            # callbacks); the band height always comes from the sidebar
            # "ROI height (fraction of tile height)" slider. The grid is
            # rendered as a fragment: moving ONE tile's slider updates
            # only this grid — and, thanks to the overlay cache,
            # effectively only the edited tile's preview.
            sig_hash_pt = _roi_sig_hash(sig) if sig is not None else "default"
            st.caption(
                "**Advanced per-tile mode.** Each tile gets its own "
                "auto-detected band (height from the sidebar **ROI height "
                "(fraction of tile height)** slider); use the per-tile "
                "offset sliders below only if a band missed the disk on "
                "that tile. The SAME sliders are available in the "
                "Stitcher tab ROI preview and the two stay in sync. Pair "
                "Review positions are converted automatically when the "
                "per-tile bands differ.")
            _render_pair_review_per_tile_grid(tiles, auto_bands_full,
                                              sig_hash_pt)
        else:
            # REFERENCE TILE 1 path (default)
            ref_band_auto = auto_bands_full[0]
            h0 = int(tiles[0].image.shape[0])
            # Interactive offset slider (px of tile 1). It shares ONE
            # state with the offset slider in the Stitcher ROI preview
            # (synchronised via on_change callbacks), so both tabs always
            # show and stitch the SAME band. The band HEIGHT is governed
            # by the sidebar "ROI height (fraction of tile height)"
            # slider only — the old "Band height (px)" slider was removed
            # (it duplicated the sidebar control).
            sig_hash = _roi_sig_hash(sig) if sig is not None else "default"
            shared_off = _roi_ref_off(sig_hash)
            off_key = f"rvp_roi_ref_off_{sig_hash}"
            ref_off = st.slider(
                "Vertical offset of the band (px)", -h0 // 2, h0 // 2,
                value=shared_off, step=5, key=off_key,
                on_change=(lambda s=sig_hash, k=off_key:
                           _roi_ref_off_cb(s, k)),
                help=("Shift the reference band up (negative) or down "
                      "(positive). Auto-detected position: "
                      f"y=[{ref_band_auto[0]},{ref_band_auto[1]}). Band "
                      "height: set by the sidebar ROI height (fraction) "
                      "slider. Synchronised with the offset slider in "
                      "the Stitcher ROI preview."))
            ref_band = _shift_band(ref_band_auto, ref_off, h0)
            auto_bands_full = [ref_band] * n

            prev_ref = pair_review.render_tile_with_roi_overlay(
                tiles[0].image, ref_band, out_width=520)
            st.image(prev_ref, use_container_width=True)
            st.caption(f"Reference band y=[{ref_band[0]},{ref_band[1]}) "
                       f"({ref_band[1] - ref_band[0]} px of {h0} px "
                       f"= {100.0 * (ref_band[1] - ref_band[0]) / h0:.0f}% "
                       f"of the tile height, set by the sidebar ROI height "
                       f"(fraction) slider). The same band is applied to "
                       f"all {n} tiles.")
            with st.expander(
                    f"Preview: the same band applied to tiles 2..{n}",
                    expanded=False):
                n_cols_ref = min(4, max(2, n - 1))
                ref_cols = st.columns(n_cols_ref)
                for k, tile_k in enumerate(tiles[1:], start=1):
                    with ref_cols[(k - 1) % n_cols_ref]:
                        prev_k = pair_review.render_tile_with_roi_overlay(
                            tile_k.image, ref_band, out_width=220)
                        st.image(prev_k, use_container_width=True)
                        st.caption(f"#{tile_k.index} {tile_k.name}")
    else:
        auto_bands_full = None

    # =================== Step 4 — manual adjustment ===================
    st.subheader("Step 4 — Manually adjust flagged pairs")
    unchecked = [i for i in range(n_pairs)
                 if not st.session_state.get(f"rvp_ok_{i}", False)]
    if not unchecked:
        st.success("All pairs approved — continue to Step 5.")
    else:
        st.caption(
            f"{len(unchecked)} pair(s) not approved yet. Nudge dx/dy below — "
            "the overlay updates live. **Increasing dx moves the right tile "
            "further right (tiles separate, overlap decreases)**; increasing "
            "dy moves it down.")

        for i in unchecked:
            rec = pairs[i]
            no_ovl_i = bool(st.session_state.get(f"rvp_no_{i}", False))

            def _reset_pair(i_=i, rec_=rec):
                """on_click callback: runs BEFORE the dx/dy widgets of the
                next rerun are instantiated, so mutating session_state is
                legal here (direct mutation after widget creation is not)."""
                st.session_state[f"rvp_dx_{i_}"] = int(round(
                    rec_.get("dx0", rec_["dx"])))
                st.session_state[f"rvp_dy_{i_}"] = int(round(
                    rec_.get("dy0", rec_["dy"])))
                rec_["dx"] = float(st.session_state[f"rvp_dx_{i_}"])
                rec_["dy"] = float(st.session_state[f"rvp_dy_{i_}"])

            title = (f"Pair {i+1} ↔ {i+2} — {tile_names[i]} → {tile_names[i+1]}")
            if no_ovl_i:
                title += " — NO OVERLAP (black gap)"
            elif not rec.get("success", True):
                title += " — automatic estimate FAILED, set the position manually"
            with st.expander(title, expanded=(len(unchecked) <= 3)):
                if no_ovl_i:
                    st.caption(
                        "This pair is marked **No overlap**: dx = tile width + "
                        "gap places the tiles side by side with a black vertical "
                        "band between them. Tweak dx (gap width) and dy below "
                        "if needed — the overlay updates live.")
                lc, rc = st.columns([3, 2])
                with rc:
                    step = st.selectbox("Nudge step (px)",
                                        [1, 2, 5, 10, 25, 50, 100],
                                        index=3, key=f"rvp_step_{i}")
                    bc = st.columns(4)
                    if bc[0].button("dx −", key=f"rvp_dxm_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dx_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dx_{i}", int(round(rec["dx"])))) - step
                        rec["dx"] = float(st.session_state[f"rvp_dx_{i}"])
                    if bc[1].button("dx +", key=f"rvp_dxp_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dx_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dx_{i}", int(round(rec["dx"])))) + step
                        rec["dx"] = float(st.session_state[f"rvp_dx_{i}"])
                    if bc[2].button("dy −", key=f"rvp_dym_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dy_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dy_{i}", int(round(rec["dy"])))) - step
                        rec["dy"] = float(st.session_state[f"rvp_dy_{i}"])
                    if bc[3].button("dy +", key=f"rvp_dyp_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dy_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dy_{i}", int(round(rec["dy"])))) + step
                        rec["dy"] = float(st.session_state[f"rvp_dy_{i}"])

                    if f"rvp_dx_{i}" not in st.session_state:
                        st.session_state[f"rvp_dx_{i}"] = int(round(rec["dx"]))
                    if f"rvp_dy_{i}" not in st.session_state:
                        st.session_state[f"rvp_dy_{i}"] = int(round(rec["dy"]))

                    dx = st.number_input(
                        "dx — horizontal offset of tile N+1 (px)",
                        min_value=-2 * w_full, max_value=3 * w_full, step=1,
                        key=f"rvp_dx_{i}", on_change=_on_pos_change, args=(i,),
                        help="Shift of the right tile relative to the left "
                             "one. Larger dx = tiles separate = less overlap.")
                    dy = st.number_input(
                        "dy — vertical offset of tile N+1 (px)",
                        min_value=-3 * h_full, max_value=3 * h_full, step=1,
                        key=f"rvp_dy_{i}", on_change=_on_pos_change, args=(i,),
                        help="Vertical shift of the right tile. Positive = "
                             "down.")
                    ovl = max(0.0, 100.0 * (w_full - abs(dx)) / w_full)
                    st.caption(f"x-overlap now ≈ {ovl:.1f} %")

                    st.button("Reset to automatic estimate",
                              key=f"rvp_reset_{i}", on_click=_reset_pair)
                    st.caption("When the overlay looks right, also tick "
                               "**Position is correct** for this pair in "
                               "Step 2 above.")
                with lc:
                    if static_mode:
                        prev = pair_review.render_pair_preview_static(
                            imgs[i], imgs[i + 1],
                            dx * scale, dy * scale,
                            out_width=760)
                    else:
                        prev = pair_review.render_pair_preview(
                            imgs[i], imgs[i + 1],
                            dx * scale, dy * scale,
                            mode=mode, out_width=760)
                    st.image(prev, use_container_width=True)

    # =================== Step 5 — stitch ===================
    st.subheader("Step 5 — Stitch the panorama with these positions")
    st.caption(
        "The panorama is blended using exactly the positions shown above "
        "(converted back to full-resolution pixels). Blending options come "
        "from the Method Settings in the sidebar.")

    with st.expander("Options"):
        refine = st.checkbox(
            "Refine approved pairs at full resolution (SIFT polish)",
            value=False, key="rv_refine",
            help="Re-runs SIFT at full resolution in a ±30 px window around "
                 "each approved position for sub-pixel accuracy. Manually "
                 "adjusted and no-overlap pairs are never touched. OFF "
                 "(default) = positions are used exactly as reviewed.")
        approved_flags = [bool(st.session_state.get(f"rvp_ok_{i}", False))
                          for i in range(n_pairs)]
        no_overlap_flags = [bool(st.session_state.get(f"rvp_no_{i}", False))
                            for i in range(n_pairs)]
        # Export the CURRENT positions (user-adjusted / no-overlap), not the
        # original automatic estimates.
        export_pairs = [{"dx": rec["dx"], "dy": rec["dy"],
                         "method": rec.get("method", "")}
                        for rec in pairs]
        st.download_button(
            "Export positions (JSON) — save your review work",
            data=pair_review.to_json(tile_names, export_pairs, approved_flags,
                                     no_overlap_flags),
            file_name="pair_review.json",
            mime="application/json",
            key="rv_export",
            help="Save all positions, approvals and no-overlap marks to a "
                 "JSON file. Load it later via 'Load positions from a saved "
                 "review JSON' above.")

    if st.button("Stitch panorama with these positions", type="primary",
                 key="rv_stitch"):
        final_translations: List[Tuple[float, float]] = []
        manual_flags: List[bool] = []
        no_flags: List[bool] = []
        for i in range(n_pairs):
            rec_i = pairs[i]
            # review_state['pairs'] is the source of truth: widget-backed
            # session keys are deleted by Streamlit once a pair is approved
            # (its number inputs stop rendering), but the record survives.
            dx_i = float(rec_i.get("dx",
                                   st.session_state.get(f"rvp_dx_{i}",
                                                        rec_i.get("dx0", 0.0))))
            dy_i = float(rec_i.get("dy",
                                   st.session_state.get(f"rvp_dy_{i}",
                                                        rec_i.get("dy0", 0.0))))
            dx0 = float(rec_i.get("dx0", dx_i))
            dy0 = float(rec_i.get("dy0", dy_i))
            is_manual = (abs(dx_i - dx0) > 0.5 or abs(dy_i - dy0) > 0.5)
            manual_flags.append(is_manual)
            no_flags.append(bool(st.session_state.get(f"rvp_no_{i}", False)))
            final_translations.append((dx_i, dy_i))

        refined_flags = [False] * n_pairs
        if refine:
            try:
                with st.spinner("Refining approved pairs at full resolution..."):
                    for i in range(n_pairs):
                        if manual_flags[i] or no_flags[i] \
                                or not pairs[i].get("success", True):
                            continue
                        r = stitching.stitch_pair_sift(
                            tiles[i].image, tiles[i + 1].image,
                            n_features=_param("n_features", 2000),
                            ratio_thresh=_param("ratio_thresh", 0.75),
                            ransac_thresh=_param("ransac_thresh", 5.0),
                            use_clahe=_param("use_clahe", True),
                            clahe_clip=_param("clahe_clip", 3.0),
                            suppress_fixed_pattern=True,
                            prior_translation=final_translations[i],
                            prior_window=30.0,
                        )
                        if r.success and r.inliers >= _param("min_inliers", 8):
                            final_translations[i] = r.translation
                            refined_flags[i] = True
            except Exception as e:
                st.warning(f"Refinement failed ({e}) — using reviewed positions as is.")

        methods = ["no-overlap" if no_flags[i]
                   else ("manual" if manual_flags[i]
                         else ("refined" if refined_flags[i] else "reviewed"))
                   for i in range(n_pairs)]

        progress = st.progress(0.0, "Blending panorama...")

        def stitch_cb(msg, p):
            progress.progress(min(max(p, 0.0), 1.0), msg)

        # Resolve ROI for the Pair Review stitch. auto_bands_full holds the
        # band for each tile. In the default reference-tile mode it already
        # includes the user's offset (same band on every tile). In the
        # advanced per-tile mode the shared per-tile offsets (the SAME
        # state the previews use — Stitcher ROI preview + Pair Review
        # Step 3) are applied here. When all tiles share the same band ->
        # pass as a single `roi_band` (Pair Review translations transfer
        # exactly). When the bands differ -> pass `roi_bands_per_tile`;
        # pipeline.stitch_row converts pair dy between the full-tile and
        # cropped-tile frames automatically.
        roi_band_pr = None
        roi_bands_per_tile_pr = None
        keep_full_pr = False
        if bool(_param("roi_enabled", True)) and auto_bands_full is not None:
            keep_full_pr = bool(_param("keep_full_panorama", False))
            if _param("roi_mode", _ROI_MODE_REF1) == _ROI_MODE_PER_TILE:
                sig_hash_pr = (_roi_sig_hash(sig)
                               if sig is not None else "default")
                auto_bands_full = [
                    _shift_band(auto_bands_full[k],
                                _roi_pt_off(sig_hash_pr, k),
                                int(tile_k.image.shape[0]))
                    for k, tile_k in enumerate(tiles)
                ]
            if len(set(auto_bands_full)) == 1:
                roi_band_pr = auto_bands_full[0]
            else:
                roi_bands_per_tile_pr = list(auto_bands_full)

        try:
            result = pipeline.stitch_row(
                tiles,
                blend_levels=_param("blend_levels", 3),
                blend_feather=_param("blend_feather", 64),
                blend_gamma=_param("blend_gamma", 1.5),
                blend_gain_strength=_param("gain_strength", 0.8),
                **_blend_mode_params(),
                pair_translations=final_translations,
                pair_methods=methods,
                roi_band=roi_band_pr,
                roi_bands_per_tile=roi_bands_per_tile_pr,
                keep_full_panorama=keep_full_pr,
                progress_callback=stitch_cb,
            )
        except Exception as e:
            st.error(f"Stitching error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

        render_stitch_result(result, tiles)


# === Render the active section ===

nav_tab = _render_navigation()
render_sidebar()  # shared by ALL tabs — upload + method settings
if nav_tab == "Stitcher":
    render_stitcher_tab()
elif nav_tab == "Pair Review":
    render_pair_review_tab()
else:
    render_help_tab()
