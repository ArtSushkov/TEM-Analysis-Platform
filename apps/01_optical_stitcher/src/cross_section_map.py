"""
Cross-section Map — register an x10 overview panorama with an x100 detail
panorama of the SAME TEM disk and build a standalone interactive HTML report.

Workflow (why this tab exists)
------------------------------
The TEM disk is thinned while the operator controls the position of the IC
cross-section in an optical microscope:

- the x10 objective photographs the WHOLE disk (slightly tilted, the
  cross-section is visible THROUGH the sacrificial glass);
- the x100 objective photographs the cross-section itself from above.

Both image series are stitched by this app.  This tab takes the two stitched
panoramas (saved into the in-session store from the Stitcher / Pair Review
tabs via the "Save to Cross-section Map" button next to the Download
buttons) and:

1. calibrates the physical scale (µm per pixel) of each panorama from the
   microscope scale bar — auto-detected on the first tile with OCR of its
   label, with the result shown to the operator for correction, or entered
   manually by clicking/drawing a line along the bar and typing its length;
2. registers the two panoramas with 2-6 operator-picked tie points (the
   disk edges are visible in BOTH panoramas) using a SIMILARITY transform
   (scale + rotation + translation). Tilt anisotropy and glass parallax are
   deliberately NOT modelled — the mapping is approximate by design;
3. checks calibration consistency: the physical distance between every pair
   of tie points must be the SAME on both panoramas (within a tolerance);
4. lets the operator click any pixel of the x10 panorama and immediately
   see the corresponding field-of-view crop (default 200 µm + a safety
   margin proportional to the registration uncertainty) from the x100
   panorama;
5. builds ONE standalone HTML file with the same interactive click-to-crop
   behaviour (pure inline JS, works offline in any browser), download
   buttons for both panoramas and the full registration metadata.

Everything degrades gracefully:
- without ``pytesseract`` (or the tesseract binary) the scale value is
  entered manually;
- without ``streamlit-image-coordinates`` the points are typed in as
  numbers instead of being picked by clicks.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib as _hl
import html as _html
import inspect
import json as _json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import tifffile

from . import io_utils

_OCR_STATUS = ""      # why OCR is unavailable (shown next to the OCR button)
try:  # OCR of the scale-bar label (optional dependency)
    import pytesseract

    def _tesseract_available() -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    if _tesseract_available():
        _HAS_PYTESSERACT = True
    else:
        # The pip wrapper is installed but the ENGINE binary is missing
        # from PATH. Try the default Windows install locations (UB-Mannheim
        # installer) before giving up — this saves the operator from having
        # to edit PATH manually.
        _tesseract_candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(Path.home() / "AppData" / "Local" / "Programs"
                / "Tesseract-OCR" / "tesseract.exe"),
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
        for _cand in _tesseract_candidates:
            if Path(_cand).exists():
                pytesseract.pytesseract.tesseract_cmd = _cand
                break
        if _tesseract_available():
            _HAS_PYTESSERACT = True
        else:
            _HAS_PYTESSERACT = False
            pytesseract = None
            _OCR_STATUS = ("the Tesseract-OCR ENGINE is not installed or "
                           "not in PATH (the pytesseract wrapper IS "
                           "installed) — see the commented instructions in "
                           "requirements.txt")
except Exception:  # pragma: no cover - depends on environment
    pytesseract = None
    _HAS_PYTESSERACT = False
    _OCR_STATUS = ("pytesseract is not installed "
                   "(pip install pytesseract) — see requirements.txt")

try:  # click-on-image widget (optional dependency, shared with app 02)
    from streamlit_image_coordinates import streamlit_image_coordinates

    _HAS_IMG_COORDS = True
except Exception:  # pragma: no cover - depends on environment
    streamlit_image_coordinates = None
    _HAS_IMG_COORDS = False

# Stretching the clickable preview to the CONTAINER width (instead of
# rendering it at its natural pixel size) needs support on TWO sides:
#
# * widget >= 0.4.1  — the frontend implements width="stretch";
# * widget 0.1.x-0.4.0 — the python signature ALREADY accepts a `width`
#   parameter, but the frontend does NOT understand the string
#   "stretch": it falls into the numeric branch, the assignment
#   img.width = "stretch" is silently ignored (invalid value -> the
#   attribute is dropped) and the image renders at its NATURAL size
#   inside #image-container { overflow-x: scroll } — i.e. CUT OFF
#   whenever the column is narrower than the preview (the operator's
#   "at first opening the Calibration-mode image is cropped and the
#   scale bar is in the hidden part" report). For those versions the
#   SAME stretching is available through use_column_width="always",
#   which every published version (>= 0.1.7) implements in its
#   frontend. We therefore pass BOTH arguments (the python side of
#   0.4.1 documents: use_column_width takes precedence over width).
_IMG_COORDS_HAS_WIDTH = False
_IMG_COORDS_HAS_UCW = False
if _HAS_IMG_COORDS:
    try:
        _sig_params = inspect.signature(
            streamlit_image_coordinates).parameters
        _IMG_COORDS_HAS_WIDTH = ("width" in _sig_params)
        _IMG_COORDS_HAS_UCW = ("use_column_width" in _sig_params)
    except (TypeError, ValueError):  # pragma: no cover
        _IMG_COORDS_HAS_WIDTH = False
        _IMG_COORDS_HAS_UCW = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORE_KEY = "xmap_store"        # list of saved panorama entries
LAST_RESULT_KEY = "xmap_last_result"   # snapshot set by render_stitch_result
STORE_LIMIT = 4                 # keep at most this many saved panoramas
MAX_TIE_POINTS = 6              # registration pairs
CONSISTENCY_WARN = 0.05         # 5 % distance mismatch -> warning

DEFAULT_FOV_UM = 200.0          # crop field of view around a clicked point
DEFAULT_SAFETY_K = 3.0          # crop margin = k * sigma
DEFAULT_EMBED_MAX_SIDE = 3200   # max side of JPEG images embedded in HTML
DEFAULT_JPEG_QUALITY = 87
DEFAULT_PICK_PRECISION_X10_PX = 5.0   # operator tie-point precision on x10

PREVIEW_SIDE_MAIN = 1000        # x10 previews (precision matters here)
# Resolution of the calibration snapshots (mode 1 trace, mode 2 ROI picker
# and mode 2 two-clicks). Raised 900 -> 1500 at the operator's request
# ("the snapshot must be BIGGER — I cannot hit the scale-bar pixel with
# the cursor"): more preview pixels = a sharper, larger clickable image,
# so a thin scale bar spans more screen pixels and is easier to hit.
# The value also RAISES the natural width cap of the numeric display
# sizing, so on tiles up to 1500 px the snapshot is no longer shrunk
# before it reaches the screen.
PREVIEW_SIDE_AUX = 1500         # x100 previews / scale-bar previews

# Display width (CSS px) of the calibration-mode clickable previews (mode 1
# one-click trace, mode 2 ROI picker and mode 2 two-clicks). These previews
# are rendered by the widget at EXACTLY this size with explicit numeric
# width/height (see _click_preview_scaled) — this is the FALLBACK and the
# smallest option of the operator-adjustable "Snapshot display size"
# selector (CAL_WIDTH_OPTIONS below): numeric sizing keeps the FULL tile
# snapshot — scale bar included — always visible from the very first
# render, no matter how the window or sidebar is resized afterwards.
PREVIEW_WIDTH_CAL = 640

# Operator-adjustable display width (CSS px) of the calibration snapshots.
# The operator's request: the DEFAULT snapshot was too small to place the
# cursor ON a scale-bar pixel, so the width is now chosen in the UI and
# remembered per session (larger = each image pixel spans more screen
# pixels = easier to hit the bar). 1280 px is the default since the
# operator's follow-up request ("сделай Snapshot display size по умолчанию
# равный 1280") — the widest option, best for hitting a thin bar; on a
# smaller window the operator picks a smaller option (640 / 800 / 960).
# Numeric sizing keeps the "never cropped" guarantee at ANY of these
# widths.
CAL_WIDTH_OPTIONS = [640, 800, 960, 1120, 1280]
CAL_WIDTH_DEFAULT = 1280

# Two-step tie-point picking (section "3 · Tie points (disk edges)", the
# operator's request: "the first click must NOT fix the pixel — show an
# enlarged fragment around it, and the SECOND click inside that fragment
# gives the exact point"). The magnifier is a square window of
# 2*TIE_ZOOM_HALF_PX full-res panorama pixels around the first click,
# enlarged by an ADAPTIVE factor (see _tie_zoom_pick_block) so the widget
# shows the fragment at ~TIE_ZOOM_DISPLAY_WIDTH CSS px.
# ENLARGED AGAIN (the operator's follow-up request: "the fragment area
# after the first click must be ANOTHER ~3x bigger — it must be clearer
# where to click"): the window grew 440 -> 760 px (area 440^2 -> 760^2 =
# x2.98 of the previous step, x11.9 of the original 220 px window) and
# the widget grew 900 -> 1280 CSS px (the same width the calibration
# snapshots use by default). The effective on-screen zoom is ~1.68 screen
# px per panorama pixel; the click -> full-res mapping stays exact (the
# widget rescales by the displayed size), so precision is not lost — the
# AIMING CONTEXT is three times larger.
TIE_ZOOM_HALF_PX = 380          # full-res px, half-window of the magnifier
TIE_ZOOM_FACTOR = 2.0           # kept for compatibility (adaptive factor
                                # computed per window in _tie_zoom_pick_block)
TIE_ZOOM_DISPLAY_WIDTH = 1280   # CSS px width of the magnifier widget

# Display width (CSS px) of the whole-panorama tie-point previews
# (section "3 · Tie points (disk edges)"). These previews now use the
# EXPLICIT NUMERIC sizing (width/height sent from Python) instead of the
# CSS "stretch to column" mode — the same reliable mode the calibration
# snapshots use. Why it matters for the operator's "the panoramas
# disappear on clicking" report: the stretch mode sizes the component's
# iframe from the BROWSER layout at mount time, so a preview that is
# re-mounted while the page structure changes (the magnifier of the
# other panorama opens/closes, tie points are added) can end up with a
# stale frame height / a blank render. With numeric sizing the frame
# height is a CONSTANT sent from Python, identical on every mount and
# every rerun, so the preview can never be cut off or render empty.
# The previews are also slightly UPSCALED on screen (natural width 1000
# px -> 1280 CSS px), which makes aiming easier.
TIE_PREVIEW_DISPLAY_WIDTH = 1280

# Manual calibration modes (Step 3 of the calibration flow). The two
# auto-search modes were REMOVED at the operator's request, and the two
# click-based modes were RENUMBERED at the operator's request: mode 1
# (one click on the bar — the app traces the full line) is the FIRST
# option and therefore the DEFAULT, mode 2 (draw ROI + click the two
# bar ends) is the precise manual fallback chosen on demand.
CAL_MODE_OPTIONS = [
    "1 · Click once ON the bar — the app traces its full length",
    "2 · Draw ROI on the tile, then click two ends of the bar",
]

_UNIT_TO_UM = {"µm": 1.0, "μm": 1.0, "um": 1.0, "u": 1.0,
               "nm": 1e-3, "mm": 1e3}


# ---------------------------------------------------------------------------
# In-memory panorama store
# ---------------------------------------------------------------------------

def get_store() -> List[dict]:
    """All saved panorama entries (newest first)."""
    return list(st.session_state.get(STORE_KEY, []))


def save_panorama(image_bgr: np.ndarray,
                  first_tile_bgr: Optional[np.ndarray],
                  source: str,
                  calib: Optional[Calib] = None) -> str:
    """Save a stitched panorama (+ its first tile for scale-bar OCR).

    Stores RGB uint8; keeps at most ``STORE_LIMIT`` newest entries.
    ``calib`` (when supplied) is the per-upload calibration produced by
    ``render_calibration_section`` in the Stitcher / Pair Review tab — it
    is attached to the entry so the Cross-section Map tab does NOT need
    to re-calibrate the panorama.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tile_rgb = (cv2.cvtColor(first_tile_bgr, cv2.COLOR_BGR2RGB)
                if first_tile_bgr is not None else None)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, w = img_rgb.shape[:2]
    entry = {
        "name": f"{source} · {w}×{h} · {ts}",
        "image": np.ascontiguousarray(img_rgb.astype(np.uint8)),
        "first_tile": (np.ascontiguousarray(tile_rgb.astype(np.uint8))
                       if tile_rgb is not None else None),
        "w": int(w),
        "h": int(h),
        "ts": ts,
        "source": str(source),
        "calib": (calib if (calib is not None and calib.ok) else None),
    }
    st.session_state.setdefault(STORE_KEY, [])
    st.session_state[STORE_KEY].insert(0, entry)
    del st.session_state[STORE_KEY][STORE_LIMIT:]
    return entry["name"]


def render_save_button(btn_key: str = "xmap_save_btn") -> None:
    """'Save to Cross-section Map' button (rendered next to Download).

    The stitched result exists only in the run that produced it, so the
    panorama snapshot is taken by ``render_stitch_result`` into
    ``st.session_state[LAST_RESULT_KEY]`` and consumed here via on_click.
    The snapshot also carries the upload signature so the per-upload
    calibration (keyed by ``render_calibration_section``) is fetched and
    attached to the saved entry — the Cross-section Map tab then reads
    it without re-calibrating.
    """

    def _on_save() -> None:
        snap = st.session_state.get(LAST_RESULT_KEY)
        if not snap:
            return
        try:
            calib = get_calibration_for_upload(snap.get("upload_sig"))
            name = save_panorama(snap.get("panorama_bgr"),
                                 snap.get("first_tile_bgr"),
                                 snap.get("source", "Stitch"),
                                 calib=calib if calib and calib.ok else None)
            if hasattr(st, "toast"):
                msg = f"Saved to Cross-section Map: {name}"
                if calib and calib.ok:
                    msg += (f" (calibration: "
                            f"{(1.0 / calib.px_per_um):.4g} µm/px)")
                else:
                    msg += " (NO calibration — none was set in this tab)"
                st.toast(msg)
        except Exception as e:  # never break the parent tab
            if hasattr(st, "toast"):
                st.toast(f"Save failed: {e}")

    # Show BEFORE the press what will actually be stored: the panorama
    # snapshot plus the calibration produced above in this tab (it IS
    # saved together with the image — required to register two
    # panoramas against each other).
    snap = st.session_state.get(LAST_RESULT_KEY)
    pending = (get_calibration_for_upload(snap.get("upload_sig"))
               if snap else Calib())
    if pending is not None and pending.ok:
        st.caption("Save stores the stitched panorama **together with "
                   f"the calibration {pending.summary()}** — the "
                   "Cross-section Map tab needs it to register the two "
                   "panoramas.")
    else:
        st.caption("Save stores the stitched panorama; **no calibration "
                   "is set yet** — complete the calibration block above "
                   "first so the entry is saved calibrated.")

    st.button(
        "Save to Cross-section Map",
        key=btn_key,
        on_click=_on_save,
        help="Store this stitched panorama in the session memory so the "
             "'Cross-section Map' tab can register it with the other "
             "panorama (x10 overview / x100 detail). The calibration "
             "produced above in this tab is attached automatically.",
    )


def _clear_store_if_requested() -> None:
    if st.session_state.get("xmap_clear_store"):
        st.session_state[STORE_KEY] = []
        st.session_state["xmap_clear_store"] = False


# ---------------------------------------------------------------------------
# Scale bar: detection + OCR + parsing
# ---------------------------------------------------------------------------

@dataclass
class ScaleBar:
    """A detected horizontal scale-bar candidate (full-res px)."""
    x0: int
    y0: int
    x1: int
    y1: int
    score: float = 0.0

    @property
    def len_px(self) -> float:
        return float(self.x1 - self.x0)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2)


def detect_scale_bar_candidates(img_rgb: np.ndarray,
                                max_candidates: int = 8) -> List[ScaleBar]:
    """Find solid horizontal bar candidates (microscope scale bars).

    Microscope overlays draw the bar as a short solid bright (or dark)
    horizontal line, usually in the bottom / top region of the frame.

    The strategy is robust to ANY global brightness / contrast (the old
    fixed thresholds ``gray >= 200`` / ``gray <= 60`` missed mid-grey
    bars on mid-grey backgrounds):

    1. morphological **top-hat / black-hat** with a long horizontal
       kernel — the response equals the LOCAL contrast of thin
       horizontal structures, so the bar "pops out" of its immediate
       background whatever the overall exposure;
    2. adaptive threshold on the hat response (30 % of its maximum,
       floored at 15) + horizontal opening that drops text glyphs
       (their horizontal runs are shorter than the minimum bar length);
    3. connected components -> RELAXED geometric filters (length
       1.2-50 % of the width, thickness up to ~h/120, aspect >= 3,
       fill >= 0.55) — then components produced by the SAME slightly
       tilted bar are merged (one per-row component of the horizontal
       opening);
    4. scored by length, position (bottom / top of the frame preferred)
       and thinness; the best ``max_candidates`` are returned.

    Recall is deliberately generous — the operator picks the real bar
    among the numbered candidates in Stage A of the calibration flow.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    klen_min = max(15, int(w * 0.012))     # min plausible bar length
    klen_max = int(w * 0.50)               # max plausible bar length
    max_thick = max(10, int(h / 120))
    klen_hat = max(klen_min * 2, int(w * 0.55))   # > any real bar length
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(klen_hat), 1))
    open_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (klen_min, 1))
    raw: List[ScaleBar] = []
    for resp in (cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
                 cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)):
        rmax = float(resp.max())
        if rmax < 15.0:
            continue                       # no contrasty thin structures
        thr = max(15.0, 0.30 * rmax)
        m8 = (resp >= thr).astype(np.uint8)
        opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, open_ker)
        n, _, stats, _ = cv2.connectedComponentsWithStats(opened)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if bw < klen_min or bw > klen_max:
                continue                   # too short / too long
            if bh < 1 or bh > max_thick:
                continue                   # too thick -> not a bar
            if bw / max(1, bh) < 3.0:
                continue                   # not elongated enough
            if area / float(bw * bh) < 0.55:
                continue                   # not solid
            raw.append(ScaleBar(x, y, x + bw, y + bh, 0.0))
    # Merge per-row fragments of the same (slightly tilted) bar: sort by
    # position and fuse neighbours that overlap both in x and y.
    raw.sort(key=lambda b: (b.y0, b.x0))
    merged: List[ScaleBar] = []
    for b in raw:
        if merged:
            m = merged[-1]
            y_ov = min(m.y1, b.y1) - max(m.y0, b.y0)
            x_ov = min(m.x1, b.x1) - max(m.x0, b.x0)
            if y_ov >= 0 and x_ov >= 0.5 * min(m.len_px, b.len_px):
                merged[-1] = ScaleBar(min(m.x0, b.x0), min(m.y0, b.y0),
                                      max(m.x1, b.x1), max(m.y1, b.y1), 0.0)
                continue
        merged.append(b)
    # Score: length (bigger = better), bottom/top position, thinness.
    out: List[ScaleBar] = []
    for b in merged:
        bw = max(1, b.x1 - b.x0)
        bh = max(1, b.y1 - b.y0)
        pos = b.y0 / float(h)
        pos_score = 1.0 if pos > 0.60 else (0.7 if pos < 0.20 else 0.4)
        thick_score = 1.0 if bh <= 6 else (0.8 if bh <= max(10, h // 150)
                                           else 0.6)
        score = (min(1.0, bw / max(1.0, 0.03 * w)) * pos_score
                 * thick_score)
        out.append(ScaleBar(b.x0, b.y0, b.x1, b.y1, float(score)))
    out.sort(key=lambda b: -b.score)
    return out[:max_candidates]


_SCALE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(µm|μm|um|Um|uM|nm|Nm|nM|mm|Mm|mM|µ|μ|u)?",
    re.IGNORECASE,
)


def trace_scale_bar_from_click(img_rgb: np.ndarray,
                               x: int,
                               y: int,
                               tol: float = 40.0,
                               band: int = 3,
                               max_frac: float = 0.9) -> Optional[dict]:
    """Trace a horizontal scale bar from ONE operator click on it.

    The operator clicks ANY pixel that belongs to the bar (the bar may be
    red, blue, white, black — ANY microscope / operator colour scheme).
    The tracing is colour-based region growing along the clicked row:

    1. the reference colour is the per-channel MEDIAN of a 3x3 patch
       around the clicked pixel (small enough to stay centred on the
       bar even when the click lands on its edge row, robust to
       anti-aliasing);
    2. on each row of a small band around the click (bars are 1-6 px
       thick; a small tilt is allowed), pixels whose colour distance to
       the reference is <= ``tol`` form runs; the run that contains /
       overlaps the clicked x is expanded left and right;
    3. runs of ADJACENT rows are merged into the bar when they overlap
       the current span AND are at least 60 % as long (this follows a
       slightly tilted bar but rejects label-glyph rows, whose runs are
       short);
    4. the bar tips are extended through up to 5 anti-aliased pixels per
       side at 1.6x tolerance (the visual end of the bar);
    5. a CONTRAST GUARD rejects the trace when the traced structure does
       not stand out against the rows above/below it (colour distance
       < 15): a click on a flat, uniform background would otherwise
       trace the entire row — no image-wide "everything matches" lines.

    A "train a neural network to find the bar" approach would break as
    soon as another microscope / operator draws the bar in a different
    colour; this detector learns the colour from the operator's click
    itself, so it is colour-agnostic by construction.

    Returns ``None`` when nothing similar to the clicked pixel exists on
    its row, otherwise a dict with the full-res geometry::

        {"x0": int, "x1": int, "y": int, "len_px": float,
         "ref_color": (r, g, b)}
    """
    H, W = img_rgb.shape[:2]
    if W < 8 or H < 4:
        return None
    x = int(max(0, min(W - 1, int(round(x)))))
    y = int(max(0, min(H - 1, int(round(y)))))
    py0, py1 = max(0, y - 1), min(H, y + 2)
    px0, px1 = max(0, x - 1), min(W, x + 2)
    patch = img_rgb[py0:py1, px0:px1].reshape(-1, 3).astype(np.float32)
    ref = np.median(patch, axis=0)                     # reference colour
    max_len = max(8, int(W * max_frac))

    def _row_dist(ry: int) -> np.ndarray:
        row = img_rgb[ry].astype(np.float32)
        return np.sqrt(((row - ref[None, :]) ** 2).sum(axis=1))

    best: Optional[Tuple[int, int]] = None
    for ry in range(max(0, y - band), min(H, y + band + 1)):
        mask = _row_dist(ry) <= tol
        if not mask[x]:
            continue
        xl = x
        while xl > 0 and mask[xl - 1] and (x - xl + 1) < max_len:
            xl -= 1
        xr = x
        while xr < W - 1 and mask[xr + 1] and (xr + 1 - x) < max_len:
            xr += 1
        if best is None or (xr - xl) > (best[1] - best[0]):
            best = (xl, xr)
    if best is None:
        return None
    xl, xr = best
    # Merge runs of adjacent rows (slightly tilted bar). A neighbouring
    # row contributes only when its run overlaps the current span and is
    # at least 60 % of its length — label glyphs (short runs) are skipped.
    min_len = 0.60 * max(1, (xr - xl))
    for ry in range(max(0, y - band), min(H, y + band + 1)):
        if ry == y:
            continue
        mask = _row_dist(ry) <= tol
        w_lo, w_hi = max(0, xl - 6), min(W, xr + 7)
        seg = mask[w_lo:w_hi]
        if not seg.any():
            continue
        idx = np.where(seg)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for g in groups:
            gl, gr = int(w_lo + g[0]), int(w_lo + g[-1])
            overlap = min(gr, xr) - max(gl, xl) + 1
            if (overlap > 0 and (gr - gl + 1) >= min_len
                    and (gr - gl + 1) <= max_len):     # never outgrow max
                xl, xr = min(xl, gl), max(xr, gr)
    # Anti-aliased tips: extend through up to 5 soft pixels per side at a
    # relaxed tolerance (the bar visually ends where the colour fades).
    tol_soft = tol * 1.6
    d_click = _row_dist(y)
    soft = 0
    while xl > 0 and soft < 5 and d_click[xl - 1] <= tol_soft:
        xl -= 1
        soft += 1
    soft = 0
    while xr < W - 1 and soft < 5 and d_click[xr + 1] <= tol_soft:
        xr += 1
        soft += 1
    if xr <= xl:
        return None
    # Contrast guard: the traced structure must STAND OUT against the rows
    # above / below it. A real scale bar does; a flat background (where
    # every pixel matches everything within the tolerance) does not — a
    # click there must NOT produce an image-wide "bar".
    sur_rows = ([r for r in range(max(0, y - band - 4), max(0, y - band))
                 if 0 <= r < H]
                + [r for r in range(min(H, y + band + 1),
                                    min(H, y + band + 5)) if 0 <= r < H])
    if sur_rows:
        sur = np.concatenate([img_rgb[r, xl:xr + 1] for r in sur_rows])
        sur_med = np.median(sur.astype(np.float32).reshape(-1, 3), axis=0)
        contrast = float(np.sqrt(((ref - sur_med) ** 2).sum()))
        if contrast < max(15.0, 0.3 * float(tol)):
            return None
    return {"x0": int(xl), "x1": int(xr), "y": int(y),
            "len_px": float(xr - xl),
            "ref_color": tuple(int(c) for c in ref)}


def parse_scale_text(text: Optional[str]) -> Optional[Tuple[float, str]]:
    """Parse '100 µm' / '50um' / '500 nm' / '0.5 mm' -> (value_in_µm, raw).

    A number with a lost unit is interpreted as µm (the common case).
    """
    if not text:
        return None
    for m in _SCALE_RE.finditer(text):
        val = float(m.group(1).replace(",", "."))
        unit = (m.group(2) or "").strip()
        if val <= 0:
            continue
        um = _UNIT_TO_UM.get(unit, 1.0) * val
        if um > 0:
            return (um, m.group(0).strip())
    return None


def ocr_scale_label(img_rgb: np.ndarray,
                    bar: ScaleBar,
                    ref_color: Optional[Tuple[int, int, int]] = None
                    ) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    """OCR the text label near a detected scale bar (optional pytesseract).

    The label sits directly ABOVE or directly BELOW the bar (rarely next
    to its inner end) and is drawn in the SAME colour as the bar itself
    (a red bar carries a red label, a white bar a white one). The old
    implementation OCR'd the raw RGB window: the sample texture around
    the tiny label forced tesseract to hallucinate single digits, the
    digit whitelist made that WORSE (every noise blob became one of
    "0123456789"), and ANY parsable number won immediately — so a noise
    "3" beat the real "200 µm". The rewritten reader:

    1. builds a COLOUR-ISOLATED binary variant of each label window —
       only pixels within a tolerance of the bar colour (known from the
       mode-4 trace, or re-measured from the bar strip) become black
       glyphs on a white background; the sample texture disappears and
       tesseract reads the overlay text, not the dust;
    2. additionally tries plain-grayscale variants (labels that are not
       bar-coloured), several upscale factors, both polarities and
       several page-segmentation modes; the label windows themselves
       grow PROGRESSIVELY — three levels (base, tall, extra-tall, each
       also wider) — and every level that still yields no trustworthy
       value triggers the next, taller one, so a label sitting far
       above/below the bar is captured too (operator report: the value
       "did not fit" into the first window and OCR found nothing);
    3. collects EVERY reading and RANKS the parses instead of letting
       the first one win: a value with an explicit unit, a plausible
       scale-bar magnitude (5–5000 µm) and 2+ digits beats a bare
       single digit — the "3 instead of 200" failure mode;
    4. returns ``parsed@…`` ONLY for a trustworthy reading. A raw text
       whose every parse is implausible comes back as ``raw@…`` — the
       UI then shows the text but does NOT pre-fill the value field
       with noise.

    Returns ``(text, status, crop)`` so the UI can ALWAYS show the
    operator what happened instead of staying silent:

    * ``text``   — raw OCR text, or None when nothing was read;
    * ``status`` — one of ``"parsed@<window>"`` (a trustworthy scale
      value; ``<window>`` is "above the bar", "below the bar" or
      "around the bar (centre fallback)"), ``"raw@<window>"`` (text
      recognised but no trustworthy value in it), ``"empty"`` (no text
      in the label windows), ``"no-pytesseract"`` (OCR not installed)
      or ``"error:<message>"`` (tesseract broken / missing from PATH);
    * ``crop``   — the label window the result came from (None when
      there is nothing to show), so the UI can display WHAT was OCR'd.
    """
    if not _HAS_PYTESSERACT:
        return None, "no-pytesseract", None
    h, w = img_rgb.shape[:2]
    bh = max(1, bar.y1 - bar.y0)
    bw = max(1, bar.x1 - bar.x0)
    cx, _ = bar.center
    # TALL label windows with a PROGRESSIVE LADDER (operator report: the
    # label sometimes sits FAR above the bar and did not fit into the
    # first window at all, so OCR "could not determine anything").
    # Level 1 keeps the proven base geometry (>=100 px above AND below
    # the bar; the old int(bh*6) + 30 gave a ~36 px window for a
    # one-pixel traced bar — too small when the label sits a few dozen
    # pixels away). When a level produces NO trustworthy reading, the
    # windows are rebuilt TALLER and WIDER — every level accumulates
    # readings, and the first level that yields a trusted value wins,
    # so the common case stays exactly as fast as before.
    _PAD_LADDER = (
        # (pad_y, pad_x, window-name tag)
        (max(int(bh * 6) + 30, 100), int(bw * 1.25) + 30, ""),
        (max(int(bh * 12) + 60, 240), int(bw * 1.6) + 60, " [tall window]"),
        (max(int(bh * 20) + 100, 400), int(bw * 2.0) + 100,
         " [extra-tall window]"),
    )

    def _windows_for(pad_y: int, pad_x: int, tag: str):
        x0, x1 = max(0, cx - pad_x), min(w, cx + pad_x)
        y_top = max(0, bar.y0 - pad_y)
        y_bot = min(h, bar.y1 + pad_y)
        return [
            (f"above the bar{tag}",
             img_rgb[y_top:max(y_top, bar.y0 - 2), x0:x1]),
            (f"below the bar{tag}",
             img_rgb[min(h, bar.y1 + 2):y_bot, x0:x1]),
            (f"around the bar (centre fallback){tag}",
             img_rgb[y_top:y_bot, x0:x1]),
        ]
    # Reference overlay colour: the caller (mode-4 trace) knows it best;
    # otherwise re-measure it from the bar strip itself.
    ref: Optional[np.ndarray] = None
    if ref_color is not None:
        try:
            ref = np.asarray(ref_color, dtype=np.float32)[:3]
        except Exception:
            ref = None
    if ref is None:
        strip = img_rgb[max(0, bar.y0):min(h, bar.y1 + 1),
                        max(0, bar.x0):min(w, bar.x1)]
        if strip.size:
            ref = np.median(
                strip.reshape(-1, 3).astype(np.float32), axis=0)

    readings: List[Tuple[str, Optional[Tuple[float, str]], str, str, bool]]
    readings = []              # (raw, parsed, window, variant, is_mask)
    best_raw: Optional[str] = None
    best_crop: Optional[np.ndarray] = None
    best_where = ""
    err_msg: Optional[str] = None
    # Window crops by name — so the UI can show EXACTLY the window the
    # winning reading came from, even when that window belongs to a
    # taller ladder level.
    crop_by_wname: dict = {}

    def _run_ocr(variants: List[Tuple[str, np.ndarray]], wname: str,
                 crop: Optional[np.ndarray]) -> None:
        """OCR every (label, image) variant; collect raw readings."""
        nonlocal best_raw, best_crop, best_where, err_msg
        for vname, var in variants:
            is_mask = vname.startswith("colour-mask")
            for g in (var, 255 - var):          # dark-on-light + inverse
                for psm in (7, 6, 11):
                    try:
                        raw = pytesseract.image_to_string(
                            g, config=f"--psm {psm}").strip()
                    except Exception as e:
                        # tesseract broken / binary missing -> NEVER silent
                        err_msg = str(e) or e.__class__.__name__
                        return
                    if not raw:
                        continue
                    readings.append((raw, parse_scale_text(raw),
                                     wname, vname, is_mask))
                    if best_raw is None:
                        best_raw = raw
                        best_crop = crop
                        best_where = wname
                if err_msg is not None:
                    return

    def _trusted_hit() -> Optional[Tuple[str, str, Optional[np.ndarray]]]:
        """Rank the readings collected SO FAR.

        Returns ``(raw, status, crop)`` of the best TRUSTED parse, or
        None when nothing is trusted yet. TRUST rules — a reading is
        only applied when it has a physical anchor; bare noise digits
        ("3 instead of 200") are never trusted:
          a) an explicit unit right after the number (tesseract renders
             µ as '?' — "?m" is accepted too);
          b) the value was read from a GLYPH-LIKE colour-mask variant
             (small text blobs in the bar's own colour) AND the magnitude
             is plausible (10–2000 µm without a unit);
          c) 3+ digits AND plausible magnitude;
          d) the SAME value was read in 2+ different windows (above and
             the centre fallback overlap, so a real label agrees with
             itself; random noise rarely repeats a 2+ digit value).
        """
        UNIT_SUFFIX_RE = re.compile(
            r"\s{0,3}(?:µm|μm|um|um|nm|mm|\?{1,2}m|µ|μ|u)", re.IGNORECASE)

        def _has_unit(text: str) -> bool:
            for m in _SCALE_RE.finditer(text):
                if UNIT_SUFFIX_RE.match(text[m.end():m.end() + 6]):
                    return True
            return False

        value_windows: dict = {}
        for raw, parsed, wname, _vn, _im in readings:
            if parsed is not None:
                value_windows.setdefault(round(parsed[0], 3), set()).add(wname)
        candidates: List[Tuple[float, str, Tuple[float, str], str]] = []
        for raw, parsed, wname, _vn, is_mask in readings:
            if parsed is None:
                continue
            unit = _has_unit(raw)
            plausible = (10.0 <= parsed[0] <= 2000.0 if not unit
                         else 0.2 <= parsed[0] <= 20000.0)
            ndig = len(re.sub(r"\D", "", parsed[1]))
            agree = len(value_windows.get(round(parsed[0], 3), ()))
            trusted = (unit or (is_mask and plausible)
                       or (plausible and ndig >= 3)
                       or (plausible and ndig >= 2 and agree >= 2))
            if not trusted:
                continue
            score = ((4.0 if unit else 0.0)
                     + (3.0 if plausible else 0.0)
                     + (2.0 if is_mask else 0.0)
                     + (1.5 if ndig >= 3 else (0.5 if ndig >= 2 else 0.0))
                     + (2.0 if agree >= 2 else 0.0))
            candidates.append((score, raw, parsed, wname))
        if not candidates:
            return None
        candidates.sort(key=lambda c: -c[0])
        _score, raw, _parsed, wname = candidates[0]
        return raw, f"parsed@{wname}", crop_by_wname.get(wname, best_crop)

    # Progressive ladder: level 1 = the proven base geometry (fast);
    # taller/wider windows run only while NO trustworthy reading exists.
    for pad_y, pad_x, tag in _PAD_LADDER:
        for wname, crop in _windows_for(pad_y, pad_x, tag):
            if crop.size == 0:
                continue
            crop_by_wname[wname] = crop
            variants: List[Tuple[str, np.ndarray]] = []
            # 1) Colour-isolated variants — overlay glyphs only. Built at
            #    NATIVE resolution (mask first, upscale afterwards) so the
            #    anti-aliased glyph edges stay crisp. A mask only counts as
            #    a TRUST anchor when it looks like TEXT: several small
            #    glyph-sized blobs. A mask dominated by huge blobs is just
            #    sample texture in a similar colour — its readings get no
            #    extra trust ("colour-weak").
            if ref is not None:
                c32 = crop.astype(np.float32)
                dist = np.sqrt(((c32 - ref[None, None, :]) ** 2).sum(axis=2))
                for tol in (60.0, 90.0):
                    m8 = (dist <= tol).astype(np.uint8)
                    if m8.any() and m8.mean() <= 0.60:
                        m8o = cv2.morphologyEx(
                            m8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                        # glyph-structure check on the opened mask
                        n_lbl, _, st_lbl, _ = \
                            cv2.connectedComponentsWithStats(m8o)
                        glyphs = [s for s in st_lbl[1:]
                                  if 4 <= s[3] <= 80 and 2 <= s[2] <= 120]
                        glyph_like = (len(glyphs) >= 3
                                      and len(glyphs) >= 0.5 * max(1, n_lbl - 1))
                        vname = (f"colour-mask tol{tol:.0f}" if glyph_like
                                 else f"colour-weak tol{tol:.0f}")
                        variants.append(
                            (vname, np.where(m8o.astype(bool),
                                             0, 255).astype(np.uint8)))
            # 2) Plain-grayscale variants (labels in another colour / b/w).
            for fx in (3.0, 5.0):
                big = cv2.resize(crop, None, fx=fx, fy=fx,
                                 interpolation=cv2.INTER_CUBIC)
                variants.append((f"gray x{fx:.0f}",
                                 cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)))
            # Colour masks are upscaled AFTER binarisation, grayscale before:
            # bring the masks to a comparable pixel height too.
            upscaled: List[Tuple[str, np.ndarray]] = []
            for vname, var in variants:
                hh = var.shape[0]
                if hh < 140:
                    f = min(6.0, 140.0 / max(1, hh))
                    var = cv2.resize(var, None, fx=f, fy=f,
                                     interpolation=cv2.INTER_NEAREST)
                upscaled.append((vname, var))
            _run_ocr(upscaled, wname, crop)
            if err_msg is not None:
                return best_raw, f"error:{err_msg}", best_crop
        hit = _trusted_hit()
        if hit is not None:
            return hit

    # No trusted value in ANY window of the ladder — surface the raw
    # text (transparency) without pre-filling the value field with it.
    return (best_raw,
            (f"raw@{best_where}" if best_raw else "empty"),
            best_crop)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calib:
    """Physical scale of one panorama."""
    px_per_um: Optional[float] = None
    method: str = "none"                    # none | metadata | auto | manual
    bar_len_px: Optional[float] = None
    bar_value_um: Optional[float] = None
    ocr_text: str = ""
    metadata_source: str = ""               # short label of the metadata source
    p1: Optional[Tuple[float, float]] = None  # manual line (full-res px)
    p2: Optional[Tuple[float, float]] = None

    @property
    def ok(self) -> bool:
        return self.px_per_um is not None and self.px_per_um > 0

    @property
    def um_per_px(self) -> Optional[float]:
        """The calibration in µm per pixel (DISPLAY unit).

        ``px_per_um`` stores px PER µm (px / µm) — the natural result of
        "bar length in px divided by bar value in µm". This property
        converts to the familiar µm/px for display and export.
        """
        if not self.ok:
            return None
        return 1.0 / self.px_per_um

    def summary(self) -> str:
        if not self.ok:
            return "not calibrated"
        src = {"metadata": "metadata", "auto": "auto (OCR)",
               "manual": "manual"}.get(self.method, self.method)
        extra = ""
        if self.method == "metadata" and self.metadata_source:
            extra = f" · {self.metadata_source}"
        elif self.bar_len_px and self.bar_value_um:
            extra = (f" · bar {self.bar_len_px:.0f} px = "
                     f"{self.bar_value_um:g} µm")
        return f"{1.0 / self.px_per_um:.4g} µm/px ({src}{extra})"


def calib_from_manual(p1: Tuple[float, float],
                      p2: Tuple[float, float],
                      length_value: float,
                      unit: str) -> Optional[Calib]:
    """Manual calibration from a line drawn along the bar + its real length."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist <= 0 or length_value <= 0:
        return None
    um = _UNIT_TO_UM.get(unit, 1.0) * length_value
    if um <= 0:
        return None
    return Calib(px_per_um=dist / um, method="manual",
                 bar_len_px=dist, bar_value_um=um, p1=p1, p2=p2)


def calib_from_ocr(bar: ScaleBar, value_um: float, ocr_text: str) -> Calib:
    return Calib(px_per_um=bar.len_px / value_um, method="auto",
                 bar_len_px=bar.len_px, bar_value_um=value_um,
                 ocr_text=ocr_text,
                 p1=(float(bar.x0), float((bar.y0 + bar.y1) / 2)),
                 p2=(float(bar.x1), float((bar.y0 + bar.y1) / 2)))


def calib_from_metadata(um_per_px: float, source: str,
                        raw: str = "") -> Calib:
    """Calibration accepted from image metadata (TIFF resolution tags,
    ImageJ ImageDescription, OME-XML or vendor TIFF tags).

    ``um_per_px`` is the METADATA value in µm per pixel. It is stored
    INTERNALLY as px per µm (``Calib.px_per_um``) — the SAME convention
    used by ``calib_from_manual`` / ``calib_from_ocr`` (a distance in px
    divided by a length in µm). Storing the metadata value as-is used to
    put an INVERTED scale into the calibration, which silently corrupted
    the consistency check, the fragment scale bar and the µm/px written
    into exported TIFFs.
    """
    um_per_px_f = float(um_per_px)
    if um_per_px_f <= 0:
        return Calib()
    return Calib(px_per_um=1.0 / um_per_px_f, method="metadata",
                 metadata_source=str(source),
                 ocr_text=(raw or source)[:200])


# Plausibility band for a LIGHT-OPTICAL microscope calibration.
# The INTERNAL unit of Calib.px_per_um is px PER µm (px/µm):
#   x100 optical: ~0.05–0.5 µm/px  ->  ~2–20 px/µm;
#   x10 optical:  ~0.5–5 µm/px     ->  ~0.2–2 px/µm.
# The band is deliberately generous (zoom optics, photo adapters), but it
# catches the classic garbage values — scanner-default DPI tags (96 dpi ->
# 25400/96 ≈ 265 µm/px ≈ 0.0038 px/µm), export defaults, or a decimal-point
# slip in a hand-typed bar value. A value outside the band used to be
# ACCEPTED SILENTLY and produced a fragment scale bar "overestimated by
# orders of magnitude".
_UM_PER_PX_MIN = 0.05     # px per µm  (~150 µm/px — beyond any light optics)
_UM_PER_PX_MAX = 150.0    # px per µm  (~0.007 µm/px — extreme digital zoom)
_DEFAULT_DPI = {72, 96, 120, 150, 200, 300, 600, 1200, 2400, 4800}


def calib_plausibility_warning(px_per_um: float,
                               source: str = "",
                               raw: str = "") -> Optional[str]:
    """Return a warning string when a calibration value looks implausible,
    or None when it is within the plausible band for optical microscopy.

    ``px_per_um`` is the INTERNAL unit: px PER µm. Metadata callers hold
    µm/px values — pass ``1.0 / um_per_px`` here.
    """
    try:
        v = float(px_per_um)
    except (TypeError, ValueError):
        return "calibration value is not a number"
    if v <= 0:
        return "calibration value is not positive"
    if not (math.isfinite(v)):
        return "calibration value is not finite"
    um_px = 1.0 / v                            # display value, µm/px
    notes = []
    if v > _UM_PER_PX_MAX:
        notes.append(
            f"{um_px:.4g} µm/px is implausibly SMALL for a light-microscope "
            f"image (< {1.0 / _UM_PER_PX_MAX:g} µm/px) — the fragment scale "
            "bar would be overestimated by orders of magnitude. Typical "
            "values: x10 ≈ 0.5–5 µm/px, x100 ≈ 0.05–0.5 µm/px.")
    if v < _UM_PER_PX_MIN:
        notes.append(
            f"{um_px:.4g} µm/px is implausibly LARGE for a light-microscope "
            f"image (> {1.0 / _UM_PER_PX_MIN:g} µm/px) — check the units of "
            "the source value (µm vs nm vs mm), scanner-DPI metadata or a "
            "misread scale-bar value. Typical values: x10 ≈ 0.5–5 µm/px, "
            "x100 ≈ 0.05–0.5 µm/px.")
    # Classic garbage source: scanner/graphics DEFAULT DPI stored in the
    # TIFF resolution tags by an export filter (96 dpi -> 265 µm/px etc.).
    if "XResolution" in (raw or ""):
        m = re.search(r"XResolution[=:\s]*([0-9]+(?:\.[0-9]+)?)",
                      str(raw), re.IGNORECASE)
        if m:
            try:
                xres = float(m.group(1))
                if ("ResolutionUnit=2" in str(raw) or "unit=2" in str(raw)
                        or ("inch" in str(raw).lower())
                        or xres in _DEFAULT_DPI):
                    if round(xres) in _DEFAULT_DPI or xres in _DEFAULT_DPI:
                        notes.append(
                            f"XResolution={m.group(1)} with ResolutionUnit="
                            "inch looks like a DEFAULT GRAPHICS DPI value "
                            "written by an export filter, not a microscope "
                            "calibration — do NOT trust it; calibrate from "
                            "the scale bar instead.")
            except ValueError:
                pass
    if not notes:
        return None
    src_txt = f" (source: {source})" if source else ""
    return " ".join(notes) + src_txt


# ---------------------------------------------------------------------------
# Calibration export: plain-text sidecar for formats that cannot carry a
# physical scale (PNG/JPEG). The TIFF export was removed at the operator's
# request — PNG is the lossless panorama format.
# ---------------------------------------------------------------------------

def calibration_sidecar_text(image_name: str, width: int, height: int,
                             calib: Optional[Calib]) -> str:
    """Plain-text calibration sidecar accompanying a PNG/JPEG export.

    PNG cannot store a reliable µm/px value: its ``pHYs`` chunk holds
    only INTEGER pixels-per-metre (a rounding of the true scale) and
    most viewers ignore it anyway. The sidecar keeps the exact value in
    a human- and machine-readable ``key = value`` form; the platform
    accepts it next to the image and ImageJ can apply it via
    Image > Properties.
    """
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = bool(calib is not None and calib.ok)
    lines = [
        "# TEM Analysis Platform - calibration sidecar",
        f"# Generated: {ts}",
        f"# Image: {image_name} ({int(width)} x {int(height)} px)",
        "# PNG/JPEG cannot store a reliable um/px value -",
        "# keep this file next to the image.",
        "",
    ]
    if not ok:
        lines.append("um_per_pixel =")
        lines.append("# (this panorama had no calibration; calibrate it in")
        lines.append("#  the Stitcher / Pair Review tab and re-export)")
    else:
        # Calib.px_per_um is px PER µm: um_per_pixel = 1 / px_per_um.
        # (The two lines used to be swapped — the sidecar then contradicted
        # the image it accompanied.)
        lines.append(f"um_per_pixel = {(1.0 / calib.px_per_um):.10g}")
        lines.append(f"px_per_um = {calib.px_per_um:.10g}")
        lines.append(f"method = {calib.method}")
        if calib.bar_len_px and calib.bar_value_um:
            lines.append(f"scale_bar_length_px = {calib.bar_len_px:.6g}")
            lines.append(f"scale_bar_value_um = {calib.bar_value_um:.6g}")
        if calib.p1 is not None and calib.p2 is not None:
            lines.append(
                "scale_bar_line_px = "
                f"({calib.p1[0]:.1f},{calib.p1[1]:.1f}) - "
                f"({calib.p2[0]:.1f},{calib.p2[1]:.1f})")
        if calib.ocr_text:
            lines.append(f"ocr_text = {calib.ocr_text}")
        if calib.metadata_source:
            lines.append(f"metadata_source = {calib.metadata_source}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Registration: similarity transform + consistency + uncertainty
# ---------------------------------------------------------------------------

@dataclass
class Registration:
    """2x3 linear map  q = M · [x, y, 1]^T  (x10 -> x100, full px).

    Two models:
    * ``similarity`` — scale + rotation + shift (4 DOF), fit from >= 2
      tie points as a complex least squares; M holds its matrix form;
    * ``affine``    — adds anisotropic scale + shear (6 DOF), fit from
      >= 3 tie points. The affine part absorbs the TILT ANISOTROPY of
      the x10 overview (the disk is photographed through the tilted
      sacrificial glass, so x and y scale differently) — one of the
      systematic causes of large mid-panorama mapping errors.
    """
    model: str = "similarity"               # "similarity" | "affine"
    a_re: float = 1.0                       # similarity complex factor
    a_im: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    m00: float = 1.0                        # matrix form (used by map_point)
    m01: float = 0.0
    m02: float = 0.0
    m10: float = 0.0
    m11: float = 1.0
    m12: float = 0.0
    s: float = 1.0                          # scale (x100 px per x10 px)
    sx: float = 1.0                         # affine: scale along x / y
    sy: float = 1.0
    theta_deg: float = 0.0
    residuals_px: List[float] = None        # per tie point, in x100 px
    rms_px: float = 0.0
    # --- piecewise model extras ------------------------------------------
    # P10/Q100: tie-point arrays (x10 / x100, full-res px). tri: scipy
    # Delaunay triangulation of P10 (None when unavailable). tri_points:
    # (i, j, k) vertex indices per simplex, kept separately so the object
    # stays picklable for the HTML report / session state.
    P10: Optional[np.ndarray] = None
    Q100: Optional[np.ndarray] = None
    tri_indices: Optional[np.ndarray] = None
    tri_finder: Optional[object] = None     # callable (x, y) -> simplex id

    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        if self.model == "piecewise" and self.tri_finder is not None \
                and self.P10 is not None and self.Q100 is not None:
            try:
                # find_simplex expects an (N, 2) array of points and
                # returns the per-point simplex index.
                si = int(self.tri_finder(
                    np.array([[float(x), float(y)]], dtype=np.float64))[0])
            except Exception:
                si = -1
            if si is not None and si >= 0 \
                    and self.tri_indices is not None \
                    and 0 <= si < len(self.tri_indices):
                i, j, k = (int(v) for v in self.tri_indices[si])
                # Exact affine map of the triangle containing the point
                # (barycentric): solves the 2x3 system from the 3 vertices.
                P = np.array([self.P10[i], self.P10[j], self.P10[k]],
                             dtype=np.float64)
                Q = np.array([self.Q100[i], self.Q100[j], self.Q100[k]],
                             dtype=np.float64)
                A = np.column_stack([P, np.ones(3)])          # 3x3
                try:
                    # Solve A @ X = Q: X is (3, 2); column 0 = (m00, m01, m02),
                    # column 1 = (m10, m11, m12) of the triangle's affine map.
                    X = np.linalg.solve(A, Q)
                    qx = float(X[0, 0] * x + X[1, 0] * y + X[2, 0])
                    qy = float(X[0, 1] * x + X[1, 1] * y + X[2, 1])
                    return (qx, qy)
                except np.linalg.LinAlgError:
                    pass                                      # fall through
        qx = self.m00 * x + self.m01 * y + self.m02
        qy = self.m10 * x + self.m11 * y + self.m12
        return (qx, qy)


def fit_similarity(
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Optional[Registration]:
    """Least-squares similarity from >= 2 tie points (x10 -> x100, full px).

    With exactly 2 points the fit is exact (4 DOF, zero redundancy); with
    more points the residuals give a real error estimate.
    """
    if len(pairs) < 2:
        return None
    P = np.array([p for p, _ in pairs], dtype=np.float64)
    Q = np.array([q for _, q in pairs], dtype=np.float64)
    pm, qm = P.mean(axis=0), Q.mean(axis=0)
    W = (P[:, 0] - pm[0]) + 1j * (P[:, 1] - pm[1])
    Z = (Q[:, 0] - qm[0]) + 1j * (Q[:, 1] - qm[1])
    denom = float(np.sum(np.abs(W) ** 2))
    if denom <= 0:
        return None
    a = complex(np.sum(Z * np.conj(W)) / denom)   # similarity, no reflection
    reg = Registration()
    reg.model = "similarity"
    reg.a_re, reg.a_im = a.real, a.imag
    reg.s = abs(a)
    reg.sx = reg.sy = reg.s
    reg.theta_deg = math.degrees(math.atan2(a.imag, a.real))
    reg.tx = float(qm[0] - (a.real * pm[0] - a.imag * pm[1]))
    reg.ty = float(qm[1] - (a.imag * pm[0] + a.real * pm[1]))
    reg.m00, reg.m01, reg.m02 = reg.a_re, -reg.a_im, reg.tx
    reg.m10, reg.m11, reg.m12 = reg.a_im, reg.a_re, reg.ty
    qx = a.real * P[:, 0] - a.imag * P[:, 1] + reg.tx
    qy = a.imag * P[:, 0] + a.real * P[:, 1] + reg.ty
    res = np.hypot(Q[:, 0] - qx, Q[:, 1] - qy)
    reg.residuals_px = [float(r) for r in res]
    reg.rms_px = float(np.sqrt(np.mean(res ** 2)))
    return reg


def fit_affine(
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Optional[Registration]:
    """Least-squares AFFINE transform from >= 3 tie points (x10 -> x100).

    Adds anisotropic scale + shear on top of the similarity model, which
    absorbs the tilt anisotropy of the x10 overview (shot through the
    tilted sacrificial glass) — the systematic part of the mid-panorama
    mapping error. Returns None for fewer than 3 points or degenerate
    (collinear) geometry.
    """
    if len(pairs) < 3:
        return None
    P = np.array([p for p, _ in pairs], dtype=np.float64)
    Q = np.array([q for _, q in pairs], dtype=np.float64)
    A = np.zeros((2 * len(pairs), 6), dtype=np.float64)
    b = np.zeros(2 * len(pairs), dtype=np.float64)
    A[0::2, 0] = P[:, 0]
    A[0::2, 1] = P[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 3] = P[:, 0]
    A[1::2, 4] = P[:, 1]
    A[1::2, 5] = 1.0
    b[0::2] = Q[:, 0]
    b[1::2] = Q[:, 1]
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    m00, m01, m02, m10, m11, m12 = (float(v) for v in sol)
    det = m00 * m11 - m01 * m10
    if abs(det) < 1e-9 or not np.isfinite(det):
        return None
    # Degeneracy guard: collinear (or nearly collinear) tie points make
    # the affine system rank-deficient. np.linalg.lstsq still returns a
    # (minimum-norm) solution with a finite determinant, so the det check
    # above does NOT catch it. Test the geometry directly: after centering,
    # the design [x, y, 1] must have FULL RANK 3 — scale-independent.
    try:
        Pc = P - P.mean(axis=0)
        B = np.column_stack([Pc[:, 0], Pc[:, 1], np.ones(len(Pc))])
        sv = np.linalg.svd(B, compute_uv=False)
        if not (sv[0] > 0 and sv[-1] / sv[0] > 1e-5):
            return None                              # collinear / degenerate
    except Exception:
        return None
    reg = Registration()
    reg.model = "affine"
    reg.m00, reg.m01, reg.m02 = m00, m01, m02
    reg.m10, reg.m11, reg.m12 = m10, m11, m12
    reg.sx = math.hypot(m00, m10)           # x-axis scale
    reg.sy = math.hypot(m01, m11)           # y-axis scale
    reg.s = math.sqrt(abs(det))             # area scale (for σ estimates)
    reg.a_re, reg.a_im, reg.tx, reg.ty = m00, m10, m02, m12
    qx = m00 * P[:, 0] + m01 * P[:, 1] + m02
    qy = m10 * P[:, 0] + m11 * P[:, 1] + m12
    res = np.hypot(Q[:, 0] - qx, Q[:, 1] - qy)
    reg.residuals_px = [float(r) for r in res]
    reg.rms_px = float(np.sqrt(np.mean(res ** 2)))
    return reg


def fit_piecewise(
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Optional[Registration]:
    """Piecewise-affine registration from >= 3 NON-collinear tie points.

    The x10 plane is triangulated (Delaunay) over the tie points; every
    query point is mapped with the EXACT affine transform of the triangle
    it falls into (barycentric interpolation between the three nearest
    tie points). Points outside the tie-point hull fall back to a global
    affine fit (minimum-norm least squares).

    Why this exists: the tie points sit on the disk EDGES, but the real
    error the operator fights is in the MIDDLE of the panorama (glass
    parallax + independent stitching drift are NON-linear). A global
    similarity/affine model is exactly pinned by 2-3 points and cannot
    bend; the piecewise model bends between the tie points — adding one
    tie point off the edge-to-edge line (a distinctive pore/grain in the
    middle strip) directly corrects the centre of the panorama.

    The model is EXACT at every tie point (residual 0 there), so the RMS
    residual is not an independent error estimate for it — the sigma
    estimate then reflects the picking precision only.

    Returns None for fewer than 3 points, collinear geometry, or when
    scipy/Qhull is unavailable.
    """
    if len(pairs) < 3:
        return None
    P = np.array([p for p, _ in pairs], dtype=np.float64)
    Q = np.array([q for _, q in pairs], dtype=np.float64)
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return None
    # Collinearity guard (same full-rank test as the affine fit).
    try:
        Pc = P - P.mean(axis=0)
        B = np.column_stack([Pc[:, 0], Pc[:, 1], np.ones(len(Pc))])
        sv = np.linalg.svd(B, compute_uv=False)
        if not (sv[0] > 0 and sv[-1] / sv[0] > 1e-5):
            return None
    except Exception:
        return None
    try:
        tri = Delaunay(P)
    except Exception:
        return None
    reg = Registration()
    reg.model = "piecewise"
    reg.P10 = P
    reg.Q100 = Q
    reg.tri_indices = np.asarray(tri.simplices, dtype=np.int64)
    reg.tri_finder = tri.find_simplex
    # Global affine fallback for points outside the tie-point hull —
    # minimum-norm lstsq over ALL pairs (finite even for degenerate sets).
    A = np.zeros((2 * len(pairs), 6), dtype=np.float64)
    b = np.zeros(2 * len(pairs), dtype=np.float64)
    A[0::2, 0] = P[:, 0]
    A[0::2, 1] = P[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 3] = P[:, 0]
    A[1::2, 4] = P[:, 1]
    A[1::2, 5] = 1.0
    b[0::2] = Q[:, 0]
    b[1::2] = Q[:, 1]
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        reg.m00, reg.m01, reg.m02 = (float(v) for v in sol[:3])
        reg.m10, reg.m11, reg.m12 = (float(v) for v in sol[3:])
    except np.linalg.LinAlgError:
        sim = fit_similarity(pairs)
        if sim is None:
            return None
        reg.m00, reg.m01, reg.m02 = sim.m00, sim.m01, sim.m02
        reg.m10, reg.m11, reg.m12 = sim.m10, sim.m11, sim.m12
    det = reg.m00 * reg.m11 - reg.m01 * reg.m10
    reg.s = math.sqrt(abs(det)) if det else 1.0
    reg.sx = math.hypot(reg.m00, reg.m10)
    reg.sy = math.hypot(reg.m01, reg.m11)
    # Residuals: exact at the tie points by construction.
    reg.residuals_px = [0.0] * len(pairs)
    reg.rms_px = 0.0
    return reg


def estimate_sigma_um(reg: Optional[Registration],
                      cal10: Calib,
                      cal100: Calib,
                      pick_precision_x10_px: float) -> Optional[float]:
    """Registration uncertainty (µm, x100 frame).

    Combines the fit residual RMS (only meaningful with > 2 points) with
    the operator's tie-point picking precision (given in x10 px, converted
    through the transform scale). RSS combination.
    """
    if reg is None or not cal100.ok:
        return None
    s_pick = pick_precision_x10_px * reg.s            # x100 px
    s_rms = reg.rms_px if len(reg.residuals_px or []) > 2 else 0.0
    sigma_x100_px = math.hypot(s_pick, s_rms)
    return sigma_x100_px / cal100.px_per_um


def consistency_report(
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        cal10: Calib,
        cal100: Calib) -> List[dict]:
    """Physical distance between every pair of tie points on BOTH panoramas.

    The distances must match (same disk!) — a mismatch means one of the
    scale-bar calibrations is wrong.
    """
    rows: List[dict] = []
    if not (cal10.ok and cal100.ok):
        return rows
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            pi = np.array(pairs[i][0], dtype=float)
            pj = np.array(pairs[j][0], dtype=float)
            qi = np.array(pairs[i][1], dtype=float)
            qj = np.array(pairs[j][1], dtype=float)
            d10 = float(np.linalg.norm(pj - pi)) / cal10.px_per_um
            d100 = float(np.linalg.norm(qj - qi)) / cal100.px_per_um
            dev = abs(d10 / d100 - 1.0) if d100 > 0 else float("inf")
            rows.append({
                "points": f"{i + 1}–{j + 1}",
                "d_x10_um": d10,
                "d_x100_um": d100,
                "dev_pct": dev * 100.0,
                "ok": dev <= CONSISTENCY_WARN,
            })
    return rows


# ---------------------------------------------------------------------------
# Geometry / rendering helpers
# ---------------------------------------------------------------------------

def downscale(img_rgb: np.ndarray,
              max_side: float) -> Tuple[np.ndarray, float]:
    """Return (preview, scale) with full = preview / scale."""
    h, w = img_rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return np.ascontiguousarray(img_rgb), 1.0
    s = float(max_side) / float(m)
    prev = cv2.resize(img_rgb, (max(1, int(round(w * s))),
                                max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(prev), s


def _crop_window(H: int, W: int, cx: float, cy: float,
                 half_px: float) -> Tuple[int, int, int, int, bool]:
    """Geometry of the clamped crop window around (cx, cy).

    Returns (x0, y0, win_x, win_y, was_clamped) — the SAME math as
    ``render_crop``; used by callers that need the window ORIGIN (e.g. the
    two-click µm measurement, which converts fragment clicks back into
    full-res panorama pixels) in addition to the crop itself.
    """
    half = max(1, int(round(half_px)))
    win_x = min(2 * half, W)
    win_y = min(2 * half, H)
    x0 = int(round(cx)) - win_x // 2
    y0 = int(round(cy)) - win_y // 2
    clamped = (win_x < 2 * half) or (win_y < 2 * half) \
        or x0 < 0 or y0 < 0 or x0 + win_x > W or y0 + win_y > H
    x0 = max(0, min(W - win_x, x0))
    y0 = max(0, min(H - win_y, y0))
    return x0, y0, win_x, win_y, bool(clamped)


def render_crop(img_rgb: np.ndarray,
                cx: float,
                cy: float,
                half_px: float) -> Tuple[np.ndarray, Tuple[float, float], bool]:
    """Crop a (2*half)^2 window around (cx, cy), clamped to the image.

    Returns (crop, point_inside_crop, was_clamped).
    """
    H, W = img_rgb.shape[:2]
    x0, y0, win_x, win_y, clamped = _crop_window(H, W, cx, cy, half_px)
    crop = img_rgb[y0:y0 + win_y, x0:x0 + win_x].copy()
    return crop, (cx - x0, cy - y0), bool(clamped)


# Two-click µm measurement (operator request: "click two pixels on the x100
# fragment and get the distance between them in µm"). Shared drawing style:
# cyan markers/line — the SAME colour the HTML report's measurement uses.
_MEAS_COLOR_RGB = (0, 188, 212)          # cyan


def _draw_measure_overlay(img_rgb: np.ndarray,
                          pts_crop: List[Tuple[float, float]],
                          px_per_um: float,
                          ) -> Tuple[np.ndarray, Optional[float]]:
    """Draw the two-click distance measurement ON the fragment.

    ``pts_crop`` — 0/1/2 clicked points in FRAGMENT (crop) coordinates;
    points outside the fragment are simply not drawn. Returns
    (image, distance_um_or_None) — the distance is computed from the
    PASSED points (full-res geometry), independent of what is visible.
    """
    out = img_rgb.copy()
    dist = None
    if len(pts_crop) >= 2 and px_per_um and px_per_um > 0:
        a, b = pts_crop[0], pts_crop[1]
        dist = math.hypot(b[0] - a[0], b[1] - a[1]) / float(px_per_um)
    H, W = out.shape[:2]
    vis = []
    for p in pts_crop:
        px, py = float(p[0]), float(p[1])
        if -0.5 <= px < W and -0.5 <= py < H:
            vis.append((px, py))
            cv2.circle(out, (int(round(px)), int(round(py))), 8,
                       _MEAS_COLOR_RGB, 2, cv2.LINE_AA)
            cv2.drawMarker(out, (int(round(px)), int(round(py))),
                           _MEAS_COLOR_RGB, cv2.MARKER_CROSS, 17, 2,
                           cv2.LINE_AA)
    if len(pts_crop) >= 2 and dist is not None and len(vis) == 2:
        (ax, ay), (bx, by) = vis
        cv2.line(out, (int(round(ax)), int(round(ay))),
                 (int(round(bx)), int(round(by))),
                 _MEAS_COLOR_RGB, 2, cv2.LINE_AA)
        label = f"{dist:.2f} \u00b5m"
        try:
            from PIL import Image, ImageDraw
            fs = max(13, min(24, int(W / 60)))
            img = Image.fromarray(out)
            dr = ImageDraw.Draw(img)
            font = _ui_font(fs)
            bbox = font.getbbox(label)
            tw_, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
            mx = int(round((ax + bx) / 2))
            my = int(round((ay + by) / 2))
            # plate above the midpoint, clamped into the fragment
            px0 = max(2, min(W - tw_ - 12, mx - tw_ // 2 - 6))
            py0 = max(2, min(H - th_ - 12, my - th_ - 14))
            dr.rectangle([px0 - 4, py0 - 4, px0 + tw_ + 8, py0 + th_ + 6],
                         fill=(0, 0, 0))
            dr.text((px0, py0), label, fill=_MEAS_COLOR_RGB, font=font)
            out = np.array(img)
        except Exception:
            cv2.putText(out, label.replace("\u00b5", "u"),
                        (int((ax + bx) / 2), int((ay + by) / 2) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        _MEAS_COLOR_RGB, 2, cv2.LINE_AA)
    return out, dist


def _center_fragment_preview(img_rgb: np.ndarray,
                             glass_top: bool,
                             px_per_um: Optional[float] = None,
                             display_w: int = 880) -> np.ndarray:
    """Fragment from the MIDDLE of the x100 panorama with the GLASS / IC
    side plates (operator request for the glass-side radio: "provide a
    fragment of the x100 panorama from its middle — choosing glass vs IC
    becomes a visual decision, not a guess").

    The window is centred, spans the full specimen thickness when the
    panorama is a wide strip (win_y clamps to H) and up to ~1800 px of
    its length; it is downscaled to ``display_w`` CSS px for display.
    An optional true-µm scale bar (bottom-left) plus the side plates
    (top/bottom edges, centred) are drawn on top.
    """
    H, W = img_rgb.shape[:2]
    half = min(H, W // 2, 900)
    win_x = min(2 * half, W)
    win_y = min(2 * half, H)
    x0 = max(0, (W - win_x) // 2)
    y0 = max(0, (H - win_y) // 2)
    crop = img_rgb[y0:y0 + win_y, x0:x0 + win_x].copy()
    s = min(1.0, float(display_w) / max(1, win_x))
    if s < 1.0:
        crop = cv2.resize(crop, (max(1, int(round(win_x * s))),
                                 max(1, int(round(win_y * s)))),
                          interpolation=cv2.INTER_AREA)
    if px_per_um and px_per_um > 0:
        crop = _draw_scale_bar(crop, float(px_per_um))
    return _draw_side_labels(crop, glass_top=glass_top)


_SCALE_BAR_CANDIDATES_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def _pick_scale_bar_um(px_per_um: float, crop_w_px: int,
                       max_fraction: float = 0.35,
                       min_px: float = 24.0) -> Tuple[float, float, bool]:
    """Pick a 'nice' scale-bar length (µm) and its pixel length for a crop.

    Chooses the LARGEST round value (1/2/5/10/... µm) that fits within
    ``max_fraction`` of the crop width and is at least ``min_px`` long;
    falls back to the round value closest to the target when nothing
    fits (very small or very large crops). Returns (bar_um, bar_px,
    degenerate) — ``degenerate`` is True when the fallback branch fired,
    which almost always means the µm/px calibration itself is implausible
    for this crop (e.g. a metadata/DPI value off by orders of magnitude).

    Units: ``px_per_um`` is px PER µm, so a length in µm converts to px
    by MULTIPLYING (µm · px/µm = px). The previous version DIVIDED,
    which inverted every bar: a correct x100 calibration then drew a
    "1000 µm" bar inside a 200 µm field of view (overestimated ~50x).
    """
    target = max(1.0, float(crop_w_px) * max_fraction)
    for v in reversed(_SCALE_BAR_CANDIDATES_UM):
        px = v * px_per_um
        if px <= target and px >= min_px:
            return float(v), px, False
    # Nothing fits (tiny FOV or a wrong calibration): round value whose
    # pixel length is closest to the target, clamped into the crop.
    best = min(_SCALE_BAR_CANDIDATES_UM,
               key=lambda v: abs(v * px_per_um - target))
    px = max(8.0, min(best * px_per_um, float(crop_w_px) * 0.9))
    return float(best), px, True


def _ui_font(size_px: int):
    """Small truetype font with the µ glyph (Pillow ships with Streamlit)."""
    from PIL import ImageFont
    size = max(11, int(size_px))
    for cand in (
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:                                     # Pillow >= 10.1 scalable default
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_scale_bar(crop_rgb: np.ndarray, px_per_um: float,
                    warn_bad_calib: bool = False) -> np.ndarray:
    """Draw a labelled scale bar in the BOTTOM-LEFT corner of the crop.

    A white bar with end ticks and a black backing plate so it stays
    readable on any background; the label above it states the real
    length ('50 µm'). Also used by the HTML report conceptually — there
    the same bar is drawn by the report's JavaScript.

    ``warn_bad_calib`` marks the label with a warning sign when the bar
    had to be clamped (no round value fits) — a strong hint that the
    µm/px calibration is implausible (see issue "scale value overes-
    timated by orders of magnitude").
    """
    if not px_per_um or px_per_um <= 0:
        return crop_rgb
    H, W = crop_rgb.shape[:2]
    if W < 80 or H < 60:
        return crop_rgb
    bar_um, bar_px, degenerate = _pick_scale_bar_um(float(px_per_um), W)
    label = f"{bar_um:g} µm"
    if warn_bad_calib and degenerate:
        label = "⚠ " + label + " (check calibration!)"
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(crop_rgb)
        dr = ImageDraw.Draw(img)
        m = max(10, int(W * 0.03))
        y = H - max(16, int(H * 0.08))
        x0, x1 = m, m + int(round(bar_px))
        lw = max(2, int(round(H / 350)))
        tick = max(4, int(H * 0.02))
        dr.rectangle([x0 - 4, y - tick - 4, x1 + 4, y + tick + 4],
                     fill=(0, 0, 0))
        dr.line([x0, y, x1, y], fill=(255, 255, 255), width=lw)
        dr.line([x0, y - tick, x0, y + tick], fill=(255, 255, 255), width=lw)
        dr.line([x1, y - tick, x1, y + tick], fill=(255, 255, 255), width=lw)
        font = _ui_font(max(12, int(H * 0.045)))
        try:
            bbox = font.getbbox(label)
            th = bbox[3] - bbox[1]
        except Exception:
            th = int(H * 0.045)
        dr.text((x0, y - tick - 6 - th), label,
                fill=(255, 255, 255),
                stroke_width=2, stroke_fill=(0, 0, 0), font=font)
        # np.asarray(PIL) is READ-ONLY; downstream cv2 drawing needs a
        # writable array, hence the copy.
        return np.array(img)
    except Exception:
        # Pillow missing (never expected — it ships with Streamlit):
        # ASCII fallback without the proper µ glyph.
        out = crop_rgb.copy()
        m = max(10, int(W * 0.03))
        y = H - max(16, int(H * 0.08))
        x1 = m + int(round(bar_px))
        cv2.rectangle(out, (m - 4, y - 10), (x1 + 4, y + 10), (0, 0, 0), -1)
        cv2.line(out, (m, y), (x1, y), (255, 255, 255), 2)
        cv2.putText(out, label.replace("µ", "u"), (m, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, H / 900),
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out


def annotate_crop(crop_rgb: np.ndarray,
                  pt: Tuple[float, float],
                  px_per_um: float,
                  fov_um: float,
                  k: float,
                  sigma_um: Optional[float]) -> np.ndarray:
    """Crosshair + uncertainty circle + scale bar (bottom-left) + caption
    (bottom-right) on the x100 crop."""
    out = crop_rgb.copy()
    px, py = pt
    H, W = out.shape[:2]
    ix, iy = int(round(px)), int(round(py))
    cv2.line(out, (max(0, ix - 16), iy), (min(W - 1, ix + 16), iy),
             (255, 40, 40), 1)
    cv2.line(out, (ix, max(0, iy - 16)), (ix, min(H - 1, iy + 16)),
             (255, 40, 40), 1)
    if sigma_um:
        r_unc = int(round(k * sigma_um * px_per_um))
        if 2 < r_unc < max(H, W):
            cv2.circle(out, (ix, iy), r_unc, (255, 150, 0), 2)
    out = _draw_scale_bar(out, px_per_um, warn_bad_calib=True)
    # Degenerate bar pick (nothing round fits) -> the calibration itself
    # is almost surely implausible; say so on the fragment caption too.
    if (px_per_um and px_per_um > 0):
        _bu, _bp, _deg = _pick_scale_bar_um(float(px_per_um), W)
    else:
        _deg = False
    label = (f"FOV {fov_um:.0f} µm · "
             f"{(1.0 / px_per_um if px_per_um > 0 else 0):.4g} µm/px"
             + (f" · σ≈{sigma_um:.1f} µm" if sigma_um else "")
             + (" · ⚠ CHECK CALIBRATION" if _deg else ""))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ox, oy = max(8, W - tw - 8), H - 10        # caption bottom-RIGHT — the
    cv2.rectangle(out, (ox - 2, oy - th - 4), (ox + tw + 2, oy + 4),  # scale
                  (0, 0, 0), -1)                      # bar owns bottom-left
    cv2.putText(out, label, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def annotate_x10_preview(prev_rgb: np.ndarray,
                         scale: float,
                         pt_full: Tuple[float, float],
                         unc_radius_full: Optional[float] = None,
                         label: str = "") -> np.ndarray:
    """Crosshair (+ uncertainty circle) drawn on the downscaled x10 preview."""
    out = prev_rgb.copy()
    px = int(round(pt_full[0] * scale))
    py = int(round(pt_full[1] * scale))
    H, W = out.shape[:2]
    cv2.line(out, (max(0, px - 12), py), (min(W - 1, px + 12), py),
             (255, 40, 40), 1)
    cv2.line(out, (px, max(0, py - 12)), (px, min(H - 1, py + 12)),
             (255, 40, 40), 1)
    if unc_radius_full:
        r = int(round(unc_radius_full * scale))
        if 2 < r < max(H, W):
            cv2.circle(out, (px, py), r, (255, 150, 0), 2)
    if label:
        cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 40, 40), 1, cv2.LINE_AA)
    return out


GLASS_COLOR_RGB = (31, 111, 235)     # blue  — sacrificial glass side
IC_COLOR_RGB = (46, 139, 87)         # green — IC (device) side


def _draw_side_labels(img_rgb: np.ndarray,
                      glass_top: bool = True) -> np.ndarray:
    """Plates 'GLASS' / 'IC' on the TOP / BOTTOM edges of an x100
    fragment (the operator selects which side of the x100 panorama is
    the glass; the choice is marked on the live preview AND in the HTML
    report). Each plate is centred on its edge of the fragment so the
    TOP/BOTTOM semantics are unambiguous."""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    fs = max(0.42, min(0.9, W / 900.0))
    for top, text, color in (
            (True, "GLASS" if glass_top else "IC", GLASS_COLOR_RGB
             if glass_top else IC_COLOR_RGB),
            (False, "IC" if glass_top else "GLASS", IC_COLOR_RGB
             if glass_top else GLASS_COLOR_RGB)):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      fs, 2)
        pad = max(4, int(th * 0.6))
        bw_, bh_ = tw + 2 * pad, th + 2 * pad
        x0 = max(0, (W - bw_) // 2)          # centred on the edge
        y0 = 6 if top else max(0, H - 6 - bh_)
        cv2.rectangle(out, (x0, y0), (x0 + bw_, y0 + bh_), color, -1)
        cv2.putText(out, text, (x0 + pad, y0 + pad + th),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

# Hard limit of the JPEG format / libjpeg: an image side may not exceed
# 65500 pixels. cv2.imencode(".jpg", ...) returns ok=False beyond it —
# the direct cause of the operator's "Report generation failed: JPEG
# encoding failed" (the x100 panorama is ~99 000 px wide and the report
# tried to embed it at FULL resolution). Any embed is therefore capped at
# this side; a wider source is downscaled to it automatically. PNG has no
# such limit (see _encode_png_rgb).
_JPEG_MAX_SIDE = 65500


def _encode_png_rgb(img_rgb: np.ndarray) -> Optional[bytes]:
    """PNG-encode an RGB uint8 image of ANY size; None on failure.

    Tries Pillow first (ships with Streamlit): it encodes the RGB array
    directly, WITHOUT the extra full-size BGR copy that
    ``cv2.cvtColor(RGB2BGR) + cv2.imencode`` makes — on a ~100 000 px wide
    x100 panorama that copy alone is ~0.5 GB, which on a small-RAM machine
    contributed to the PNG downloads silently failing. Falls back to
    OpenCV (with the copy) when Pillow is unavailable.
    """
    if img_rgb is None or img_rgb.size == 0:
        return None
    try:
        import io as _io
        from PIL import Image
        bio = _io.BytesIO()
        Image.fromarray(img_rgb).save(bio, format="PNG", compress_level=6)
        data = bio.getvalue()
        if data:
            return data
    except Exception:
        pass
    try:
        ok, buf = cv2.imencode(
            ".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return buf.tobytes() if ok else None
    except Exception:
        return None


def _embed_jpeg(img_rgb: np.ndarray, max_side: float,
                quality: int) -> Tuple[str, float]:
    """Encode a downscaled JPEG -> (base64, full->embed scale factor).

    ``max_side`` is capped at ``_JPEG_MAX_SIDE`` (65500 px — the hard
    limit of the JPEG format): a wider source is downscaled to the cap
    instead of failing the whole report with "JPEG encoding failed".
    """
    if img_rgb is None or img_rgb.size == 0 or img_rgb.ndim != 3:
        raise RuntimeError(
            "the panorama image is empty or has an unexpected format")
    h, w = img_rgb.shape[:2]
    max_side = min(float(max_side), float(_JPEG_MAX_SIDE))
    scale = 1.0
    img = img_rgb
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        img = cv2.resize(img_rgb, (max(1, int(round(w * scale))),
                                   max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii"), scale


def _fmt(x: float) -> str:
    return f"{float(x):.10g}"


def _meta_tables(cal10: Calib, cal100: Calib,
                 pairs, reg: Optional[Registration],
                 cons: List[dict], fov_um: float, k: float,
                 sigma_um: Optional[float], crop_half_um: float,
                 glass_side: str = "") -> str:
    def table(headers: List[str], rows: List[List[str]]) -> str:
        th = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
        trs = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in rows)
        return f"<table><tr>{th}</tr>{trs}</table>"

    parts: List[str] = []
    gs = str(glass_side or "").lower()
    if gs == "top":
        parts.append("<p><b>x100 side labels:</b> TOP — <b style="
                     "'color:#1f6feb'>GLASS</b> (sacrificial glass) · "
                     "BOTTOM — <b style='color:#2e8b57'>IC</b> (the "
                     "device). The plates are drawn on the top and "
                     "bottom edges of every x100 fragment shown in "
                     "this report.</p>")
    elif gs == "bottom":
        parts.append("<p><b>x100 side labels:</b> TOP — <b style="
                     "'color:#2e8b57'>IC</b> (the device) · BOTTOM — <b "
                     "style='color:#1f6feb'>GLASS</b> (sacrificial "
                     "glass). The plates are drawn on the top and "
                     "bottom edges of every x100 fragment shown in "
                     "this report.</p>")
    parts.append(table(
        ["Panorama", "Calibration", "Bar (px)", "Bar value (µm)", "Source"],
        [
            ["x10 overview", _html.escape(cal10.summary()),
             f"{cal10.bar_len_px:.0f}" if cal10.bar_len_px else "—",
             f"{cal10.bar_value_um:g}" if cal10.bar_value_um else "—",
             "OCR: " + _html.escape(cal10.ocr_text or "—")
             if cal10.method == "auto" else _html.escape(cal10.method)],
            ["x100 detail", _html.escape(cal100.summary()),
             f"{cal100.bar_len_px:.0f}" if cal100.bar_len_px else "—",
             f"{cal100.bar_value_um:g}" if cal100.bar_value_um else "—",
             "OCR: " + _html.escape(cal100.ocr_text or "—")
             if cal100.method == "auto" else _html.escape(cal100.method)],
        ]))
    if pairs:
        rows = []
        for i, (p, q) in enumerate(pairs):
            res = ""
            if reg and reg.residuals_px:
                res = f"{reg.residuals_px[i]:.1f} px"
            rows.append([str(i + 1),
                         f"({p[0]:.0f}, {p[1]:.0f})",
                         f"({q[0]:.0f}, {q[1]:.0f})", res])
        parts.append("<p><b>Tie points (disk edges)</b></p>" +
                     table(["#", "x10 (px)", "x100 (px)", "residual"],
                           rows))
    if reg:
        if reg.model == "affine":
            parts.append(
                f"<p>Affine transform: scale x <b>{reg.sx:.4f}</b>, "
                f"scale y <b>{reg.sy:.4f}</b> (x100 px per x10 px), "
                f"RMS residual <b>{reg.rms_px:.1f} px</b>"
                + (f" ({reg.rms_px / cal100.px_per_um:.1f} µm)"
                   if cal100.ok else "") +
                ". The affine model absorbs the anisotropic scale/shear "
                "of the tilted x10 view.")
        elif reg.model == "piecewise":
            parts.append(
                f"<p>Piecewise-affine transform: <b>{len(reg.tri_indices)}"
                "</b> Delaunay triangles over the tie points; every click "
                "is mapped with the exact affine transform of its triangle. "
                "Exact at the tie points; outside their hull the global "
                "affine fallback is used (matrix shown below). The model "
                "bends between tie points and therefore corrects part of "
                "the mid-panorama drift/parallax.</p>")
        else:
            parts.append(
                f"<p>Similarity transform: scale <b>{reg.s:.4f}</b> "
                f"(x100 px per x10 px), rotation <b>{reg.theta_deg:.2f}°</b>, "
                f"RMS residual <b>{reg.rms_px:.1f} px</b>"
                + (f" ({reg.rms_px / cal100.px_per_um:.1f} µm)"
                   if cal100.ok else "") + ".</p>")
        parts.append(
            "<p style='color:#666'>Note: the mapping is exact at the tie "
            "points; between them it is an interpolation and beyond them "
            "an extrapolation. Glass parallax and stitching drift of the "
            "two independently stitched panoramas can make the local "
            "error in the middle of the panorama much larger than the "
            "global σ estimate — see the live note under the report "
            "fragment for the distance to the nearest tie point.</p>")
    if cons:
        rows = [[c["points"], f"{c['d_x10_um']:.1f}", f"{c['d_x100_um']:.1f}",
                 f"{c['dev_pct']:.1f}%",
                 "OK" if c["ok"] else "<b style='color:#b00'>MISMATCH</b>"]
                for c in cons]
        parts.append(
            "<p><b>Calibration consistency</b> (physical distances between "
            "tie points must match on both panoramas):</p>" +
            table(["Points", "x10 (µm)", "x100 (µm)", "deviation", ""],
                  rows))
    if sigma_um:
        parts.append(
            f"<p><b>Report parameters:</b> FOV {fov_um:g} µm around the "
            f"clicked point · safety factor k = {k:g} · σ ≈ "
            f"{sigma_um:.1f} µm.</p>")
    else:
        parts.append(
            f"<p><b>Report parameters:</b> FOV {fov_um:g} µm around the "
            f"clicked point · safety factor k = {k:g} · σ unknown.</p>")
    parts.append(
        f"<p><b>Crop window:</b> {crop_half_um * 2:g} µm across "
        f"({crop_half_um * 2:g}/2 µm FOV + {k:g}·σ margin).</p>")
    return "".join(parts)


_HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$TITLE</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
         margin: 24px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 22px; margin: 0 0 4px 0; }
  h2 { font-size: 16px; margin: 24px 0 8px 0;
       border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 18px; }
  .wrap { position: relative; margin: 0 0 8px 0; }
  #c10 { display: block; border: 1px solid #999; cursor: crosshair;
         max-width: 100%; }
  #o10 { position: absolute; left: 0; top: 0; pointer-events: none;
         max-width: 100%; }
  #crop { display: block; border: 1px solid #999; background: #333;
          max-width: 100%; height: auto; box-sizing: border-box; }
  /* The fragment canvas shows the FOV window (plus the k·sigma margin)
     from the app, CLAMPED to the x100 panorama borders exactly like the
     app's live preview (render_crop): the window never sticks out of the
     panorama, so the canvas is NOT padded into a square — it has the
     panorama's true shape (a wide, short strip for a wide, short
     panorama). Its DISPLAYED size is capped by the page width
     (max-width:100%) so a wide canvas can never overflow the screen —
     it is scaled down to fit and stays fully visible and
     human-readable. When the window is SMALLER than the page it is
     shown 1:1, so a small FOV visibly shrinks the fragment on screen.
     Either way the CONTENT of the fragment is always the full clamped
     FOV window: a bigger FOV shows more of the panorama (a smaller one
     zooms in), and the scale bar inside the fragment always reads
     true µm. */
  #cropscroll { overflow-x: auto; margin: 0 0 8px 0; }
  #measure { font-size: 14px; margin: 8px 0; min-height: 20px;
             color: #006064; }
  #measure b { color: #00707d; }
  #fnav button.mclr { background: #6b7280; }
  #fnav button.mclr:hover { background: #575b63; }
  #fnav { margin: 6px 0; font-size: 14px; }
  #fnav button { padding: 6px 12px; margin-right: 6px; background: #1f6feb;
                 color: #fff; border: 0; border-radius: 6px; cursor: pointer;
                 font-size: 13px; }
  #fnav button:hover { background: #1a5cd8; }
  #fnav input { width: 80px; padding: 4px 6px; border: 1px solid #bbb;
                border-radius: 4px; }
  #fnav label { margin-left: 8px; color: #444; }
  /* Download buttons for the CURRENT x100 fragment (the operator's
     request): PNG / JPEG with the GLASS / IC labels and the scale bar,
     without the crosshair. Green = "export", so they read differently
     from the blue navigation buttons above. */
  #fdl { margin: 6px 0; font-size: 14px; }
  #fdl button { padding: 6px 12px; margin-right: 6px; background: #2e8b57;
                color: #fff; border: 0; border-radius: 6px; cursor: pointer;
                font-size: 13px; }
  #fdl button:hover { background: #256f45; }
  #posinfo { color: #555; margin-left: 10px; font-size: 13px; }
  #readout { font-size: 14px; margin: 10px 0; min-height: 20px; }
  #warn { color: #b00; font-size: 14px; min-height: 18px; font-weight: 600; }
  table { border-collapse: collapse; font-size: 12.5px; margin-top: 6px; }
  th, td { border: 1px solid #ccc; padding: 3px 8px; text-align: left; }
  th { background: #eee; }
  a.btn { display: inline-block; padding: 8px 14px; margin: 4px 8px 4px 0;
          background: #1f6feb; color: #fff; border-radius: 6px;
          text-decoration: none; font-size: 14px; }
  .hint { font-size: 13.5px; color: #444; margin: 8px 0; }
</style>
</head>
<body>
<h1>$TITLE</h1>
<div class="sub">Generated $DATE · TEM Analysis Platform · Optical Stitcher ·
Cross-section Map</div>

<h2>1 · Interactive map — click any pixel of the x10 panorama</h2>
<div class="hint">The x10 panorama below spans the FULL page width; the x100
fragment underneath shows the $FOV µm field of view around the mapped point
plus a $K× safety margin for the registration uncertainty
(σ ≈ $SIGMA µm). The orange circle on the x10 panorama marks the
$KSIGMA µm uncertainty radius. IMPORTANT: the mapping is exact at the
tie points; BETWEEN them it is an interpolation and OUTSIDE them an
extrapolation — glass parallax and stitching drift of the two
independently stitched panoramas can make the local error much larger
than σ. Click close to a tie point, or add more tie points
(≥ 3 also enable the Affine model in the app) for an accurate map.</div>
<div class="wrap"><canvas id="c10"></canvas><canvas id="o10"></canvas></div>
<div id="readout">Click on the x10 panorama…</div>
<div id="warn"></div>
<div id="note" class="hint"></div>

<h2>2 · x100 fragment — top view of the IC cross-section</h2>
$SIDENOTE
<div id="fnav">Move the fragment along the disk-edge line:
  <button id="mvL" title="One step towards tie point 1">&#9664; Left</button>
  <button id="mvR" title="One step towards tie point 2">Right &#9654;</button>
  <button id="mvHome" title="Back to the last clicked point">&#10226; Clicked point</button>
  <button id="mClear" class="mclr" title="Remove the A/B markers and the distance readout">&#10005; Clear measure</button>
  <label>Step (µm) <input id="stepUm" type="number" value="50" min="0.1" step="10"></label>
  <span id="posinfo"></span>
</div>
<div id="cropscroll"><canvas id="crop"></canvas></div>
<div id="measure">Click twice on the fragment to measure the distance between
the two pixels in µm (uses the x100 calibration).</div>
<div id="fdl">Download the fragment CURRENTLY shown above (with the GLASS /
IC labels and the scale bar, WITHOUT the crosshair — the file name carries
the fragment centre coordinates):
  <button id="dlPng" title="Save the current x100 fragment as a PNG image">&#11015; PNG</button>
  <button id="dlJpg" title="Save the current x100 fragment as a JPEG image">&#11015; JPEG</button>
</div>
<div class="hint" id="cropinfo">The fragment is EXACTLY the field of view
entered in the app ("Field of view around the point (µm)", plus the
k·σ safety margin) CLAMPED to the x100 panorama borders — the same
window, in the same shape as the app's live preview (a bigger FOV shows
more of the x100 panorama, a smaller one zooms in; the fragment is never
padded into a square). The fragment ALWAYS fits the page width — a wide
window is scaled down to the page instead of overflowing it, so the
whole picture stays visible and readable at any FOV. The fragment opens at
the MIDDLE of the segment between the two disk-edge tie points (or at
the panorama centre when fewer than two tie points exist) — clicking
the x10 panorama above recentres it. The Left / Right buttons shift the
fragment along the edge-to-edge line by the chosen step in µm.</div>

<h2>3 · Downloads</h2>
<a class="btn" id="dl10" download="$DL10" href="data:image/jpeg;base64,$DIMG10">Download
x10 panorama (JPEG, full resolution)</a>
<a class="btn" id="dl100" download="$DL100" href="data:image/jpeg;base64,$DIMG100">Download
x100 panorama (JPEG, full resolution)</a>
<div class="hint">$DLNOTE</div>

<h2>4 · Registration metadata</h2>
$METADATA

<script>
(function () {
  "use strict";
  var M00 = $M00, M01 = $M01, M02 = $M02;
  var M10 = $M10, M11 = $M11, M12 = $M12;
  var TIES10 = $TIES10;              // x10 tie points (full-res px)
  var F10 = $F10, F100 = $F100;               // full-res -> embedded scale
  var PXUM10 = $PXUM10, PXUM100 = $PXUM100;   // µm per pixel
  var HALF = $HALF;                           // crop half size, full-res x100 px
  var SIGMAUM = $SIGMAUM, KF = $KF;
  var SIDETOP = "$SIDETOP", SIDEBOT = "$SIDEBOT";  // side labels: GLASS / IC
  var SIDETOPCOL = "$SIDETOPCOL", SIDEBOTCOL = "$SIDEBOTCOL";
  var img10 = new Image(), img100 = new Image();
  var loaded = 0;
  function onLoad() { loaded++; if (loaded === 2) init(); }
  img10.onload = onLoad;
  img100.onload = onLoad;
  img10.src = "data:image/jpeg;base64,$IMG10";
  img100.src = "data:image/jpeg;base64,$IMG100";

  function init() {
    // Download links whose embedded copy was omitted (identical to the
    // interactive copy when the source exceeds the JPEG size limit — the
    // base64 block is stored once): fill them from the interactive image.
    var dl10 = document.getElementById("dl10");
    var dl100 = document.getElementById("dl100");
    if (dl10 && dl10.getAttribute("href").length <= 26) dl10.href = img10.src;
    if (dl100 && dl100.getAttribute("href").length <= 26) dl100.href = img100.src;
    var c10 = document.getElementById("c10");
    var o10 = document.getElementById("o10");
    c10.width = img10.naturalWidth; c10.height = img10.naturalHeight;
    o10.width = c10.width; o10.height = c10.height;
    c10.getContext("2d").drawImage(img10, 0, 0);
    fit10();
    window.addEventListener("resize", fit10);
    c10.addEventListener("click", onClick);
    document.getElementById("mvL").addEventListener(
      "click", function () { moveFragment(-1); });
    document.getElementById("mvR").addEventListener(
      "click", function () { moveFragment(1); });
    document.getElementById("mvHome").addEventListener(
      "click", function () {
        if (clickedQ) {
          curQ = [clickedQ[0], clickedQ[1]];
          showFragment(curQ[0], curQ[1]);
        }
      });
    // Fragment download buttons (the operator's request): save the
    // CURRENTLY shown x100 fragment as PNG / JPEG — with the GLASS / IC
    // side labels and the scale bar, but WITHOUT the crosshair.
    document.getElementById("dlPng").addEventListener(
      "click", function () { downloadFragment("png"); });
    document.getElementById("dlJpg").addEventListener(
      "click", function () { downloadFragment("jpeg"); });
    // Two-click µm measurement: every click on the fragment toggles the
    // A → B pair (full-res coordinates are stored, so the markers and the
    // line survive moving the fragment with the Left / Right buttons).
    var ccrop = document.getElementById("crop");
    ccrop.addEventListener("click", function (e) {
      if (!CUR) return;
      var rect = ccrop.getBoundingClientRect();
      var ex = (e.clientX - rect.left) * (ccrop.width / rect.width);
      var ey = (e.clientY - rect.top) * (ccrop.height / rect.height);
      var fpx = CUR.wx0 + ex / CUR.cscale;      // canvas px -> full-res x100
      var fpy = CUR.wy0 + ey / CUR.cscale;
      if (measPts.length >= 2) measPts = [];    // third click: new pair
      measPts.push([fpx, fpy]);
      showFragment(curQ[0], curQ[1]);
    });
    document.getElementById("mClear").addEventListener(
      "click", function () {
        measPts = [];
        showFragment(curQ[0], curQ[1]);
      });
    // The report opens with the fragment at the MIDDLE of the segment
    // between the two disk-edge tie points (or at the centre of the x100
    // panorama when fewer than two tie points exist).
    if (TIES10.length >= 2) {
      curQ = mapFwd((TIES10[0][0] + TIES10[1][0]) / 2,
                    (TIES10[0][1] + TIES10[1][1]) / 2);
    } else {
      curQ = [img100.naturalWidth / (2 * F100),
              img100.naturalHeight / (2 * F100)];
    }
    showFragment(curQ[0], curQ[1]);
  }

  // Span the x10 panorama across the full page width (item: "x10 во всю
  // ширину листа"), but never upscale it beyond its embedded size.
  function fit10() {
    var c10 = document.getElementById("c10");
    var o10 = document.getElementById("o10");
    var avail = c10.parentElement.clientWidth;
    var w = Math.min(c10.width, avail);
    var h = Math.round(c10.height * w / c10.width);
    c10.style.width = w + "px"; c10.style.height = h + "px";
    o10.style.width = w + "px"; o10.style.height = h + "px";
  }

  // ---- mapping helpers -------------------------------------------------
  function mapFwd(x, y) {              // x10 full-res -> x100 full-res
    return [M00 * x + M01 * y + M02, M10 * x + M11 * y + M12];
  }
  // Inverse map (x100 -> x10): moves the crosshair on the x10 panorama
  // when the fragment is shifted with the Left / Right buttons.
  var DET = M00 * M11 - M01 * M10;
  var HASINV = Math.abs(DET) > 1e-12;
  var I00 = M11 / DET, I01 = -M01 / DET, I10 = -M10 / DET, I11 = M00 / DET;
  function mapBack(qx, qy) {
    if (!HASINV) return [0, 0];
    var dx = qx - M02, dy = qy - M12;
    return [I00 * dx + I01 * dy, I10 * dx + I11 * dy];
  }

  // Unit direction of the disk-edge line (tie point 1 -> tie point 2),
  // mapped to the x100 frame and normalized. Fallback: +x. The Left /
  // Right buttons move the fragment along THIS line; "Right" always
  // steps towards tie point 2.
  var DIRX = 1, DIRY = 0, DIRFROMTIES = false;
  if (TIES10.length >= 2) {
    var dq1 = mapFwd(TIES10[0][0], TIES10[0][1]);
    var dq2 = mapFwd(TIES10[1][0], TIES10[1][1]);
    var ddx = dq2[0] - dq1[0], ddy = dq2[1] - dq1[1];
    var dl = Math.sqrt(ddx * ddx + ddy * ddy);
    if (dl > 1e-6) { DIRX = ddx / dl; DIRY = ddy / dl; DIRFROMTIES = true; }
  }

  var curQ = null;          // current fragment centre (full-res x100 px)
  var clickedQ = null;      // last operator-clicked point
  var CUR = null;           // drawing parameters of the CURRENT fragment
                            // (window, scale, bar, plates) — reused by the
                            // download buttons so the saved image is the
                            // fragment EXACTLY as shown on screen

  // ---- two-click µm measurement on the fragment (operator request: click
  // two pixels and read the distance between them in µm) -------------------
  var measPts = [];         // 0/1/2 clicked points, FULL-RES x100 px
  var MEASCOL = "#00bcd4";  // cyan — the same colour the app's live
                            // preview draws the measurement with
  function drawMeasure(ctx, P) {
    var mi2 = document.getElementById("measure");
    if (!measPts.length) {
      mi2.textContent = "Click twice on the fragment to measure the " +
        "distance between the two pixels in µm (uses the x100 " +
        "calibration). The markers and the line survive moving the " +
        "fragment; a third click starts a new measurement.";
      return;
    }
    var cvPts = [], distUm = null, pxLen = 0;
    if (measPts.length >= 2) {
      var ddx = measPts[1][0] - measPts[0][0];
      var ddy = measPts[1][1] - measPts[0][1];
      pxLen = Math.sqrt(ddx * ddx + ddy * ddy);
      distUm = pxLen / PXUM100;
    }
    ctx.font = "bold 12px sans-serif";
    for (var mi = 0; mi < measPts.length; mi++) {
      var mx = (measPts[mi][0] - P.wx0) * P.cscale;
      var my = (measPts[mi][1] - P.wy0) * P.cscale;
      cvPts.push([mx, my]);
      if (mx < -20 || my < -20 || mx > P.cw + 20 || my > P.ch + 20) {
        continue;                        // outside the shown fragment
      }
      ctx.strokeStyle = MEASCOL; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(mx, my, 7, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(mx - 12, my); ctx.lineTo(mx + 12, my);
      ctx.moveTo(mx, my - 12); ctx.lineTo(mx, my + 12); ctx.stroke();
      ctx.fillStyle = MEASCOL;
      ctx.fillText(mi === 0 ? "A" : "B", mx + 10, my - 10);
    }
    if (measPts.length >= 2) {
      var ax = cvPts[0][0], ay = cvPts[0][1];
      var bx = cvPts[1][0], by = cvPts[1][1];
      var inA = !(ax < -20 || ay < -20 || ax > P.cw + 20 || ay > P.ch + 20);
      var inB = !(bx < -20 || by < -20 || bx > P.cw + 20 || by > P.ch + 20);
      if (inA && inB) {
        ctx.strokeStyle = MEASCOL; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        var mx2 = (ax + bx) / 2, my2 = (ay + by) / 2;
        var lbl = distUm.toFixed(2) + " µm";
        ctx.font = "bold 13px sans-serif";
        var tw2 = ctx.measureText(lbl).width;
        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.fillRect(mx2 - tw2 / 2 - 5, my2 - 24, tw2 + 10, 19);
        ctx.fillStyle = MEASCOL;
        ctx.fillText(lbl, mx2 - tw2 / 2, my2 - 10);
      }
      var off = 0;
      for (var mo = 0; mo < 2; mo++) {
        if (measPts[mo][0] < P.wx0 || measPts[mo][0] > P.wx0 + P.winW ||
            measPts[mo][1] < P.wy0 || measPts[mo][1] > P.wy0 + P.winH) off++;
      }
      mi2.innerHTML = "<b>Distance A → B: " + distUm.toFixed(2) + " µm</b> (" +
        Math.round(pxLen) + " px at the x100 panorama's full resolution) · " +
        "A(" + Math.round(measPts[0][0]) + ", " + Math.round(measPts[0][1]) +
        ") · B(" + Math.round(measPts[1][0]) + ", " +
        Math.round(measPts[1][1]) + ")" +
        (off ? " · " + off + " point(s) OUTSIDE the currently shown " +
          "fragment (the view moved) — the distance stays valid" : "") +
        " — click again to start a new measurement.";
    } else {
      mi2.innerHTML = "A(" + Math.round(measPts[0][0]) + ", " +
        Math.round(measPts[0][1]) + ") px placed — click once more for " +
        "point B (the distance is shown in µm).";
    }
  }

  // ---- shared fragment drawing helpers ----------------------------------
  // The on-screen fragment and the downloaded PNG / JPEG are built with
  // the SAME three helpers (panorama pixels + scale bar + side plates),
  // so the downloaded file is pixel-identical to what the operator sees —
  // minus the crosshair, which is drawn only on the screen canvas.

  // Panorama pixels of the clamped FOV window (canvas px = full-res px ·
  // cscale; the window is pinned inside the panorama exactly like the
  // app's live preview).
  function drawFragmentImage(ctx, P) {
    var We = img100.naturalWidth, He = img100.naturalHeight;
    ctx.fillStyle = "#333"; ctx.fillRect(0, 0, P.cw, P.ch);
    var winEx = P.winW * F100, winEy = P.winH * F100;
    var ex0 = P.wx0 * F100, ey0 = P.wy0 * F100;
    var x0 = Math.max(0, ex0), y0 = Math.max(0, ey0);
    var x1 = Math.min(We, ex0 + winEx), y1 = Math.min(He, ey0 + winEy);
    if (x1 > x0 && y1 > y0) {
      ctx.drawImage(img100, x0, y0, x1 - x0, y1 - y0,
        (x0 - ex0) / F100 * P.cscale, (y0 - ey0) / F100 * P.cscale,
        (x1 - x0) / F100 * P.cscale, (y1 - y0) / F100 * P.cscale);
    }
  }

  // Scale bar, BOTTOM-LEFT corner: white bar with end ticks on a black
  // backing plate, labelled with its real length (largest round
  // 1/2/5/10... µm value that fits in ~1/3 of the width).
  function drawScaleBar(ctx, P) {
    var cands = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
    var bar = 0, barpx = 0, bardeg = false;
    for (var ci = cands.length - 1; ci >= 0; ci--) {
      var cpx = cands[ci] * PXUM100 * P.cscale;
      if (cpx <= P.cw * 0.35 && cpx >= 24) { bar = cands[ci]; barpx = cpx; break; }
    }
    if (bar === 0) {
      // No round value fits the window: almost always an implausible
      // calibration (e.g. a default-DPI metadata value). Draw the closest
      // candidate but MARK the label so the operator re-checks it.
      bar = cands[0];
      barpx = cands[0] * PXUM100 * P.cscale;
      barpx = Math.max(8, Math.min(barpx, P.cw * 0.9));
      bardeg = true;
    }
    var bm = Math.max(10, Math.round(P.cw * 0.03));
    var by = P.ch - Math.max(14, Math.round(P.ch * 0.06));
    var bx1 = bm + barpx;
    // Font capped by the canvas HEIGHT too: on a wide, short fragment
    // (cw >> ch) the width-based size alone would fill half the strip.
    var fsBar = Math.max(11, Math.min(Math.round(P.cw * 0.026),
                                      Math.round(P.ch * 0.18)));
    ctx.fillStyle = "#000";
    ctx.fillRect(bm - 4, by - 11, barpx + 8, 22);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bm, by); ctx.lineTo(bx1, by);
    ctx.moveTo(bm, by - 6); ctx.lineTo(bm, by + 6);
    ctx.moveTo(bx1, by - 6); ctx.lineTo(bx1, by + 6); ctx.stroke();
    ctx.font = fsBar + "px sans-serif";
    var lbl = (bardeg ? "⚠ " : "") + bar + " µm" + (bardeg ? " — check calibration" : "");
    ctx.lineWidth = 3; ctx.strokeStyle = "#000";
    ctx.strokeText(lbl, bm, by - 9);
    ctx.fillStyle = "#fff"; ctx.fillText(lbl, bm, by - 9);
  }

  // Side plates (GLASS / IC) centred on the TOP / BOTTOM edges of the
  // fragment — COMPACT size (the operator asked to shrink them). The
  // scale bar stays in the bottom-LEFT corner, the bottom plate is
  // centred — no overlap.
  function drawSidePlates(ctx, P) {
    if (SIDETOP && SIDEBOT) {
      var fs2 = Math.max(9, Math.min(Math.round(P.cw * 0.016),
                                     Math.round(P.ch * 0.10)));
      var bh2 = Math.round(fs2 * 1.5);
      ctx.font = "bold " + fs2 + "px sans-serif";
      ctx.textBaseline = "middle";
      var wT = ctx.measureText(SIDETOP).width + 8;
      var wB = ctx.measureText(SIDEBOT).width + 8;
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = SIDETOPCOL;
      ctx.fillRect(P.cw / 2 - wT / 2, 4, wT, bh2);
      ctx.fillStyle = SIDEBOTCOL;
      ctx.fillRect(P.cw / 2 - wB / 2, P.ch - 4 - bh2, wB, bh2);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = "#fff";
      ctx.fillText(SIDETOP, P.cw / 2 - wT / 2 + 4, 4 + bh2 / 2);
      ctx.fillText(SIDEBOT, P.cw / 2 - wB / 2 + 4, P.ch - 4 - bh2 / 2);
      ctx.textBaseline = "alphabetic";
    }
  }

  // Build a fresh canvas with the CURRENT fragment — everything the
  // operator sees, EXCEPT the crosshair (the download must be clean).
  function buildCleanCanvas() {
    if (!CUR) return null;
    var cv = document.createElement("canvas");
    cv.width = CUR.cw; cv.height = CUR.ch;
    var x = cv.getContext("2d");
    drawFragmentImage(x, CUR);
    drawScaleBar(x, CUR);
    drawSidePlates(x, CUR);
    return cv;
  }

  // Export the clean fragment as a file download (works offline —
  // everything comes from the data-URL images embedded in this file).
  function downloadFragment(fmt) {
    var cv = buildCleanCanvas();
    if (!cv) return;
    var isJpg = (fmt === "jpeg");
    var name = "x100_fragment_x" + Math.round(CUR.qx) + "_y" +
               Math.round(CUR.qy) + (isJpg ? ".jpg" : ".png");
    var mime = isJpg ? "image/jpeg" : "image/png";
    if (cv.toBlob) {
      cv.toBlob(function (blob) {
        if (!blob) return;
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url; a.download = name;
        document.body.appendChild(a); a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
      }, mime, 0.95);
    } else {
      // Very old browsers: data-URL fallback.
      var a = document.createElement("a");
      a.href = cv.toDataURL(mime, 0.95); a.download = name;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a);
    }
  }

  function showFragment(qx, qy) {
    var c10 = document.getElementById("c10");
    var o10 = document.getElementById("o10");
    var cc = document.getElementById("crop");
    var W100full = img100.naturalWidth / F100;
    var H100full = img100.naturalHeight / F100;
    var outside = (qx < 0 || qy < 0 || qx > W100full || qy > H100full);

    var We = img100.naturalWidth, He = img100.naturalHeight;

    // Draw the fragment so that its CONTENT is always the whole FOV
    // window ("Field of view around the point (µm)" + k·σ margin) —
    // CLAMPED to the x100 panorama borders exactly like the app's live
    // preview (render_crop): the square window shrinks to what the
    // panorama actually covers (winW × winH) and is pinned inside it, so
    // the canvas is NOT a padded square — it has the panorama's true
    // shape (e.g. a wide 4096×241 strip for a wide, short panorama).
    // The DISPLAYED size can never overflow the page: the CSS rule
    // #crop { max-width:100% } scales a wide canvas down to the page
    // (the whole fragment stays visible and readable) and leaves a
    // small window at its true pixel size. cscale = canvas px per
    // full-res x100 px, uniform in BOTH axes (a CMAX=4096 cap on the
    // larger side only downscale-scales the whole canvas uniformly —
    // a memory-safe bound, 4096 px = 67 MB RGBA).
    var winTrue = 2 * HALF;              // requested square window, full-res x100 px
    var Wfull = W100full, Hfull = H100full;
    var winW = Math.max(1, Math.min(winTrue, Wfull));
    var winH = Math.max(1, Math.min(winTrue, Hfull));
    // Pin the window INSIDE the panorama (same as the app): the centre
    // shifts at the borders instead of drawing dark filler around the
    // image.
    var wx0 = Math.min(Math.max(qx - winW / 2, 0), Math.max(0, Wfull - winW));
    var wy0 = Math.min(Math.max(qy - winH / 2, 0), Math.max(0, Hfull - winH));
    var CMAX = 4096;                     // canvas internal cap
    var cscale = Math.min(CMAX / winW, CMAX / winH, 1);
    var cw = Math.max(2, Math.round(winW * cscale));
    var ch = Math.max(2, Math.round(winH * cscale));
    if (cc.width !== cw || cc.height !== ch) { cc.width = cw; cc.height = ch; }
    var ctx = cc.getContext("2d");
    // All drawing parameters of THIS fragment — stored in CUR so the
    // download buttons re-draw the SAME fragment (without the crosshair)
    // at the SAME size, with the SAME scale bar and side plates.
    var P = { qx: qx, qy: qy, wx0: wx0, wy0: wy0,
              winW: winW, winH: winH, cscale: cscale, cw: cw, ch: ch };
    CUR = P;
    drawFragmentImage(ctx, P);
    var px = (qx - wx0) * cscale;   // crosshair, canvas px
    var py = (qy - wy0) * cscale;
    ctx.strokeStyle = "#ff3b30"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px - 14, py); ctx.lineTo(px + 14, py);
    ctx.moveTo(px, py - 14); ctx.lineTo(px, py + 14); ctx.stroke();
    ctx.strokeStyle = "#ff9500";
    ctx.strokeRect(px - 9, py - 9, 18, 18);
    drawScaleBar(ctx, P);
    drawSidePlates(ctx, P);
    drawMeasure(ctx, P);

    // Crosshair on the x10 panorama: back-map the (possibly button-moved)
    // fragment centre so the marker follows the navigation too.
    var back = mapBack(qx, qy);
    var fx = back[0], fy = back[1];
    var ex = fx * F10, ey = fy * F10;                    // embedded x10 px
    var ctx2 = o10.getContext("2d");
    ctx2.clearRect(0, 0, o10.width, o10.height);
    ctx2.strokeStyle = "#ff3b30"; ctx2.lineWidth = 2;
    ctx2.beginPath();
    ctx2.moveTo(ex - 12, ey); ctx2.lineTo(ex + 12, ey);
    ctx2.moveTo(ex, ey - 12); ctx2.lineTo(ex, ey + 12); ctx2.stroke();
    if (SIGMAUM > 0 && KF > 0) {
      // sigma in µm -> px: MULTIPLY by px/µm (the old division inverted it).
      var r = SIGMAUM * KF * PXUM10 * F10;
      ctx2.strokeStyle = "#ff9500";
      ctx2.beginPath(); ctx2.arc(ex, ey, r, 0, Math.PI * 2); ctx2.stroke();
    }

    document.getElementById("readout").innerHTML =
      "x10: (" + Math.round(fx) + ", " + Math.round(fy) + ") px · " +
      "x100: (" + Math.round(qx) + ", " + Math.round(qy) + ") px · " +
      "≈ (" + Math.round(fx / PXUM10) + ", " + Math.round(fy / PXUM10) +
      ") µm";
    var warn = "";
    if (outside) {
      var dx = Math.max(0, -qx, qx - W100full);
      var dy = Math.max(0, -qy, qy - H100full);
      var dist = Math.sqrt(dx * dx + dy * dy) / PXUM100;
      warn = "The mapped point lies OUTSIDE the x100 panorama — showing " +
        "the nearest available area (≈" + Math.round(dist) + " µm away).";
    }
    document.getElementById("warn").textContent = warn;
    // Distance to the nearest tie point (on x10) — the mapping is only an
    // interpolation between tie points and an extrapolation beyond them,
    // so the local error can exceed σ by a large factor (glass parallax,
    // stitching drift of the two independently stitched panoramas).
    var note = "";
    if (TIES10.length > 0) {
      var nd2 = Infinity;
      for (var ti = 0; ti < TIES10.length; ti++) {
        var tdx = fx - TIES10[ti][0], tdy = fy - TIES10[ti][1];
        var d2 = tdx * tdx + tdy * tdy;
        if (d2 < nd2) nd2 = d2;
      }
      var nd_um = Math.sqrt(nd2) / PXUM10;
      note = "Nearest tie point: ≈" + Math.round(nd_um) + " µm (on the x10 " +
        "panorama). " + (nd_um > 1 ?
        "The mapping here is interpolated/extrapolated between tie points — " +
        "the true local error can be MUCH larger than σ. Click near a tie " +
        "point or add more tie points for an accurate map." :
        "Clicking right at a tie point — the mapping is at its best here.");
    }
    document.getElementById("note").textContent = note;
    // Position of the fragment centre along / off the disk-edge line —
    // a large "off the line" value in the MIDDLE of the segment is the
    // direct measure of the mid-panorama registration disagreement.
    var pos = "";
    if (DIRFROMTIES) {
      var q1 = mapFwd(TIES10[0][0], TIES10[0][1]);
      var along = (qx - q1[0]) * DIRX + (qy - q1[1]) * DIRY;
      var off = (qx - q1[0]) * DIRY - (qy - q1[1]) * DIRX;
      pos = "centre ≈ " + Math.round(along / PXUM100) + " µm from edge " +
        "point 1 · " + Math.round(off / PXUM100) + " µm off the edge line";
    }
    document.getElementById("posinfo").textContent = pos;
    var disp = document.getElementById("crop").clientWidth;
    document.getElementById("cropinfo").innerHTML =
      "x100 fragment · window " + Math.round(2 * HALF / PXUM100) +
      " µm across (" + Math.round(2 * HALF) + " px at the x100 panorama's " +
      "full resolution)" +
      (winW !== winTrue || winH !== winTrue ?
        " — clamped to the x100 panorama (" +
        Math.round(Wfull) + "×" + Math.round(Hfull) +
        " px full-res), same shape as in the app" : "") +
      " · shown " + cw + "×" + ch + " px canvas" +
      (disp > 0 && Math.abs(disp - cw) > 1 ?
        ", displayed " + disp + " px wide (scaled down to fit the page)" :
        ", displayed 1:1 (fits the page)");
  }

  function onClick(e) {
    var c10 = document.getElementById("c10");
    var rect = c10.getBoundingClientRect();
    var ex = (e.clientX - rect.left) * (c10.width / rect.width);
    var ey = (e.clientY - rect.top) * (c10.height / rect.height);
    var fx = ex / F10, fy = ey / F10;                    // full-res x10
    var q = mapFwd(fx, fy);                              // full-res x100
    curQ = [q[0], q[1]];
    clickedQ = [q[0], q[1]];
    showFragment(q[0], q[1]);
  }

  // Shift the fragment along the disk-edge line by ± the step (µm).
  function moveFragment(sign) {
    if (!curQ) return;
    var stepUm = parseFloat(document.getElementById("stepUm").value);
    if (!(stepUm > 0)) return;
    var stepPx = stepUm * PXUM100;        // µm · px/µm = full-res x100 px
    curQ = [curQ[0] + sign * DIRX * stepPx,
            curQ[1] + sign * DIRY * stepPx];
    showFragment(curQ[0], curQ[1]);
  }
})();
</script>
</body>
</html>
""")


def generate_html_report(img10_rgb: np.ndarray,
                         img100_rgb: np.ndarray,
                         cal10: Calib,
                         cal100: Calib,
                         pairs,
                         reg: Optional[Registration],
                         fov_um: float,
                         k: float,
                         sigma_um: Optional[float],
                         embed_max_side: float = DEFAULT_EMBED_MAX_SIDE,
                         jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                         embed_max_side_x100: float = 0.0,
                         dl_full_resolution: bool = True,
                         dl_jpeg_quality: int = 95,
                         glass_side: str = "",
                         title: str = "TEM Disk — Cross-section Map") -> str:
    """Build the standalone interactive HTML report (one file, offline).

    ``embed_max_side``      — max side of the INTERACTIVE x10 image
                              (displayed across the full page width);
    ``embed_max_side_x100`` — max side of the INTERACTIVE x100 image;
                              0 embeds it at FULL resolution. The
                              fragment shown on a click is cut from THIS
                              copy and drawn without upscaling, so its
                              quality is bounded by this resolution —
                              full resolution = pixel-perfect fragment.
                              BOTH sides are capped at 65500 px (the hard
                              JPEG format limit): a wider x100 panorama
                              is embedded at 65500 px automatically
                              (fragments stay near-native) instead of
                              failing the whole report; the report notes
                              the downscaling in the downloads section;
    ``dl_full_resolution``  — embed ADDITIONAL full-resolution JPEG
                              copies (quality ``dl_jpeg_quality``) used
                              by the two Download buttons, so the
                              downloaded images are no longer the
                              downscaled report-resolution copies;
                              OFF falls back to the interactive copies;
    ``glass_side``          — which side of the x100 panorama is the
                              sacrificial GLASS: "top" or "bottom" (the
                              other side is the IC). Drawn as GLASS / IC
                              plates on the top / bottom edges of every
                              shown x100 fragment and written into the
                              registration metadata; empty string = no
                              side labels.
    """
    b64_10, f10 = _embed_jpeg(img10_rgb, embed_max_side, jpeg_quality)
    x100_side = (embed_max_side_x100
                 if (embed_max_side_x100 and embed_max_side_x100 > 0)
                 else 10 ** 9)
    b64_100, f100 = _embed_jpeg(img100_rgb, x100_side, jpeg_quality)
    # "Full-resolution" download copies: capped by the JPEG format limit
    # (65500 px per side) inside _embed_jpeg. When the interactive copy of
    # an image ALREADY sits at that cap (a panorama wider than 65500 px,
    # e.g. the operator's ~99 000 px x100), the two copies are identical —
    # reuse it instead of embedding a second multi-megabyte copy.
    if dl_full_resolution:
        dl_side = float(_JPEG_MAX_SIDE)
        int10 = max(img10_rgb.shape[:2]) / max(f10, 1e-9) if f10 else 10 ** 9
        int100 = (max(img100_rgb.shape[:2]) / max(f100, 1e-9)
                  if f100 else 10 ** 9)
        # An empty sentinel string = "identical to the interactive copy":
        # the template emits an empty href and the report's JavaScript
        # fills it from the interactive image at load time — the huge
        # base64 block is stored ONCE instead of twice.
        b64_10d = ("" if int10 > dl_side
                   else _embed_jpeg(img10_rgb, 10 ** 9, dl_jpeg_quality)[0])
        b64_100d = ("" if int100 > dl_side
                    else _embed_jpeg(img100_rgb, 10 ** 9, dl_jpeg_quality)[0])
        dl_note = ("These links download the FULL-resolution copies "
                   "embedded in this file (JPEG, quality "
                   f"{dl_jpeg_quality:g}) — the interactive images above "
                   "are separate, compact report-resolution copies that "
                   "keep this file manageable.")
    else:
        b64_10d, b64_100d = b64_10, b64_100
        dl_note = ("These links download the report-resolution JPEG "
                   "copies embedded in this file (enable 'Embed "
                   "full-resolution copies' in the app for full-quality "
                   "downloads). Full-resolution panoramas are also "
                   "available in the application (Cross-section Map tab "
                   "→ section 5).")
    # Transparent note when a panorama exceeded the JPEG format limit and
    # was downscaled for the embeds (the operator's "JPEG encoding failed"
    # report — the x100 panorama is ~99 000 px wide).
    _over = []
    if max(img10_rgb.shape[:2]) > _JPEG_MAX_SIDE:
        _over.append(f"x10 ({img10_rgb.shape[1]}×{img10_rgb.shape[0]} px)")
    if max(img100_rgb.shape[:2]) > _JPEG_MAX_SIDE:
        _over.append(
            f"x100 ({img100_rgb.shape[1]}×{img100_rgb.shape[0]} px)")
    if _over:
        dl_note += (
            " NOTE: the " + " and the ".join(_over) +
            " exceed the JPEG format limit of 65500 px per side, so the "
            "embedded interactive copy and the download link of that "
            "panorama share ONE downscaled-to-65500-px copy (stored "
            "once, linked by the report's script — no duplicate is "
            "embedded). The UNTOUCHED full-resolution panorama is "
            "available in the application: Cross-section Map → section "
            "5 → 'Prepare PNG downloads' (PNG has no size limit).")
    if not cal100.ok:
        raise ValueError("x100 calibration missing")
    crop_half_um = fov_um / 2.0 + (k * sigma_um if sigma_um else 0.0)
    half_full = crop_half_um * cal100.px_per_um
    cons = consistency_report(pairs, cal10, cal100)
    metadata = _meta_tables(cal10, cal100, pairs, reg, cons,
                            fov_um, k, sigma_um, crop_half_um,
                            glass_side=glass_side)
    gs = str(glass_side or "").lower()
    if gs == "top":
        side_t, side_b = "GLASS", "IC"
        side_tc, side_bc = "#1f6feb", "#2e8b57"
    elif gs == "bottom":
        side_t, side_b = "IC", "GLASS"
        side_tc, side_bc = "#2e8b57", "#1f6feb"
    else:
        side_t = side_b = ""
        side_tc = side_bc = "#1f6feb"
    sidenote = ""
    if side_t:
        sidenote = (
            '<div class="hint"><b>Side labels on the x100 fragment '
            '(top / bottom edges):</b> TOP = '
            + ("GLASS — sacrificial glass side" if gs == "top"
               else "IC — the device itself")
            + " · BOTTOM = "
            + ("IC — the device itself" if gs == "top"
               else "GLASS — sacrificial glass side")
            + ". The same plates are drawn on every fragment view as "
              "you move it along the disk-edge line.</div>")
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return _HTML_TEMPLATE.substitute(
        TITLE=_html.escape(title),
        DATE=now,
        FOV=_fmt(fov_um),
        K=_fmt(k),
        SIGMA=(_fmt(sigma_um) if sigma_um else "—"),
        KSIGMA=(_fmt(k * sigma_um) if sigma_um else "—"),
        METADATA=metadata,
        DL10="panorama_x10.jpg",
        DL100="panorama_x100.jpg",
        DLNOTE=dl_note,
        IMG10=b64_10,
        IMG100=b64_100,
        DIMG10=b64_10d,
        DIMG100=b64_100d,
        SIDENOTE=sidenote,
        SIDETOP=side_t,
        SIDEBOT=side_b,
        SIDETOPCOL=side_tc,
        SIDEBOTCOL=side_bc,
        M00=_fmt(reg.m00) if reg else "1",
        M01=_fmt(reg.m01) if reg else "0",
        M02=_fmt(reg.m02) if reg else "0",
        M10=_fmt(reg.m10) if reg else "0",
        M11=_fmt(reg.m11) if reg else "1",
        M12=_fmt(reg.m12) if reg else "0",
        TIES10=_json.dumps([[float(p[0]), float(p[1])] for p, _ in pairs]),
        F10=_fmt(f10),
        F100=_fmt(f100),
        PXUM10=_fmt(cal10.px_per_um or 1.0),
        PXUM100=_fmt(cal100.px_per_um),
        HALF=_fmt(half_full),
        SIGMAUM=_fmt(sigma_um) if sigma_um else "0",
        KF=_fmt(k),
    )


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------

def _state() -> dict:
    st.session_state.setdefault("xmap_state", {})
    return st.session_state["xmap_state"]


def _cal_display_width() -> int:
    """Display width (CSS px) chosen by the operator for the calibration
    snapshots (the "Snapshot display size" selector in Step 3). Larger =
    every image pixel spans more screen pixels = easier to place the
    cursor ON a scale-bar pixel (the operator's request). Falls back to
    CAL_WIDTH_DEFAULT when the selector has not been rendered yet."""
    try:
        v = int(_state().get("cal_display_width", CAL_WIDTH_DEFAULT))
    except (TypeError, ValueError):
        v = CAL_WIDTH_DEFAULT
    return v if v in CAL_WIDTH_OPTIONS else CAL_WIDTH_DEFAULT


def _img_content_sig(img_rgb: np.ndarray) -> str:
    """Short hash of the preview's content (shape + pixel bytes).

    The clickable widget (streamlit-image-coordinates) is known to keep
    serving a STALE/blank render when the same component key is reused
    with a changed image (new annotations, a different ROI crop, the
    other tab's tile after a Stitcher <-> Pair Review switch). Hashing
    the content into the widget key forces a clean remount whenever the
    image actually changes — the picture can no longer stick.
    """
    try:
        h = _hl.md5()
        h.update(repr(img_rgb.shape).encode("utf-8"))
        h.update(np.ascontiguousarray(img_rgb).tobytes())
        return h.hexdigest()[:10]
    except Exception:
        return "x"


def _click_preview_scaled(img_rgb: np.ndarray,
                          key: str,
                          scale: float,
                          display_width: Optional[int] = None,
                          allow_upscale: bool = False
                          ) -> Optional[Tuple[float, float]]:
    """Clickable preview (ALREADY downscaled) -> one deduplicated click,
    converted to full-res coordinates with the given ``scale``
    (full = displayed / scale). Returns None when the widget is
    unavailable or no NEW click happened.

    ``display_width`` — when set, the widget renders at EXACTLY this
    width (CSS px, aspect preserved) via explicit NUMERIC width/height
    arguments instead of the CSS "stretch to column" mode. Why this
    matters: the stretch mode sizes the component's IFRAME from the
    browser layout at mount time — when the column width later changes
    (sidebar collapse, window resize), the image inside re-fits by CSS
    but the iframe height stays STALE, and the preview is cut off (the
    operator's "the first-open Calibration-mode snapshot is cropped and
    the scale bar is in the hidden part" — the first mount happened at
    one width, the column then widened, the bottom of the tile with the
    scale bar fell outside the stale frame). Numeric sizing removes the
    dependency completely: the frame height is a CONSTANT sent from
    Python, identical on every mount and every rerun, so the preview can
    never be cut off — first render included. The default width is the
    operator-adjustable _cal_display_width() (larger = easier to hit a
    thin scale bar), falling back to PREVIEW_WIDTH_CAL.

    ``allow_upscale`` — when True, a preview whose NATURAL width is
    smaller than ``display_width`` is still rendered at the requested
    width (the browser enlarges it): each image pixel then spans MORE
    screen pixels, which is exactly what the operator needs to hit a
    thin scale bar or a tie-point pixel. The click -> full-res mapping
    stays exact in both sizing modes (the widget reports the displayed
    size together with the click). Default False keeps the old "never
    upscale a small preview" behaviour for other callers.

    IMMEDIATE RESIZE (the operator's follow-up request: "changing the
    'Snapshot display size' must resize the snapshot at once, without
    switching to another mode and back"). The numeric width/height are
    widget ARGS, but a custom component only re-applies its args when
    its iframe (re)receives a render event — and Streamlit does NOT
    resend the args to an already-mounted component whose element did
    not change. A widget whose ONLY change is the numeric size would
    therefore keep the old on-screen size until something else remounts
    it (the operator's old workaround: switch the mode and back, which
    unmounts/remounts the component). The fix: the display width is
    part of the WIDGET KEY, so a size change remounts the component as
    a NEW element — the new iframe receives its args and the snapshot
    is rendered at the new size on the very same rerun.

    Without ``display_width`` (x10 tie-point previews) the widget shows
    the preview STRETCHED to the container width (``width="stretch"`` on
    widget >= 0.4.1 AND ``use_column_width="always"`` on EVERY version,
    see the import block above). The widget reports the click in
    DISPLAYED pixels together with the displayed size, so the click is
    rescaled back to the natural preview pixels before the full-res
    conversion — the mapping is exact for BOTH sizing modes. The same
    click is never delivered twice (dedup on the widget's unix_time
    token, with a coordinate fallback for old widget versions without
    it).
    """
    if not _HAS_IMG_COORDS or img_rgb.shape[0] < 4 or img_rgb.shape[1] < 4:
        if img_rgb.shape[0] >= 4 and img_rgb.shape[1] >= 4:
            st.image(img_rgb, use_container_width=True)
            if not _HAS_IMG_COORDS:
                st.caption("streamlit-image-coordinates is not installed — "
                           "type the coordinates into the fields below "
                           "instead of clicking.")
        return None
    if display_width:
        # Deterministic numeric sizing (see the docstring): the frontend
        # takes its plain-attribute branch — img.width = width,
        # img.height = height, Streamlit.setFrameHeight(height) — which
        # EVERY published widget version (0.1.0 ... 0.4.1) implements
        # identically. No CSS class, no layout measurement, no stale
        # frame height. With allow_upscale=False a small preview is never
        # enlarged beyond its natural size; the calibration previews opt
        # IN to upscaling (bigger on-screen pixels = easier to hit).
        nat_w, nat_h = img_rgb.shape[1], img_rgb.shape[0]
        dw = int(max(2, int(display_width)))
        if not allow_upscale:
            dw = int(min(dw, nat_w))
        dh = int(max(2, round(dw * nat_h / nat_w)))
        kw = {"width": dw, "height": dh}
        # The size is part of the widget key (see the docstring above):
        # changing the "Snapshot display size" remounts the component, so
        # the new width/height are applied IMMEDIATELY — no mode switch
        # needed any more.
        key = f"{key}:w{dw}"
    else:
        kw = {}
        if _IMG_COORDS_HAS_WIDTH:
            kw["width"] = "stretch"
        if _IMG_COORDS_HAS_UCW:
            # use_column_width="always" is the UNIVERSAL stretch switch: the
            # frontends of widget 0.1.7 ... 0.4.0 only implement THIS argument
            # (their python accepts a `width` parameter but silently ignores
            # the "stretch" STRING, which made the first-open preview render
            # at natural size and get cut off). On widget 0.4.1 it takes
            # precedence over width="stretch" and means the same thing.
            kw["use_column_width"] = "always"
    # Content-dependent key: a changed image (new markers / another
    # tile / another ROI) remounts the component instead of reusing a
    # possibly stale frontend render (fix of the "ROI snapshot does not
    # appear after a tab switch" report).
    key = f"{key}:{_img_content_sig(img_rgb)}"
    val = streamlit_image_coordinates(img_rgb, key=key, **kw)
    if val is None:
        return None
    try:
        cx = int(val["x"]) if isinstance(val, dict) else int(val[0])
        cy = int(val["y"]) if isinstance(val, dict) else int(val[1])
        vw = int(val.get("width") or 0) if isinstance(val, dict) else 0
        vh = int(val.get("height") or 0) if isinstance(val, dict) else 0
        ut = val.get("unix_time") if isinstance(val, dict) else None
    except Exception:
        return None
    # Dedup FIRST (on the click identity, not on rescaled coords): the
    # widget replays its stored value on every rerun, and a stale value
    # must never be re-applied after a window resize changed the ratio.
    last_key = key + "_last"
    token = (("t", ut) if ut is not None
             else ("p", cx, cy, vw, vh))
    if st.session_state.get(last_key) == token:
        return None                             # same click re-delivered
    st.session_state[last_key] = token
    # Displayed pixels -> natural preview pixels (the widget may display
    # the preview scaled to the container width).
    nat_w, nat_h = img_rgb.shape[1], img_rgb.shape[0]
    fx = (nat_w / vw) if (vw > 0 and abs(vw - nat_w) > 1) else 1.0
    fy = (nat_h / vh) if (vh > 0 and abs(vh - nat_h) > 1) else 1.0
    return (cx * fx / scale, cy * fy / scale)   # full-res coordinates


def _apply_click_to_inputs(click: Optional[Tuple[float, float]],
                           x_key: str, y_key: str) -> None:
    """Write a picked click into the numeric input widget states."""
    if click is None:
        return
    st.session_state[x_key] = float(click[0])
    st.session_state[y_key] = float(click[1])


def _marked_preview(img_rgb: np.ndarray, max_side: float,
                    points_full: List[Tuple[float, float]],
                    cur_full: Optional[Tuple[float, float]] = None,
                    unc_radius_full: Optional[float] = None
                    ) -> Tuple[np.ndarray, float]:
    """Downscaled preview with tie-point markers + current point.

    Returns (preview, scale) with full = preview / scale.
    """
    prev, s = downscale(img_rgb, max_side)
    out = prev.copy()
    for i, p in enumerate(points_full):
        px, py = int(round(p[0] * s)), int(round(p[1] * s))
        cv2.circle(out, (px, py), 5, (255, 40, 40), -1)
        cv2.putText(out, str(i + 1), (px + 7, py - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 40, 40), 1,
                    cv2.LINE_AA)
    if cur_full is not None:
        out = annotate_x10_preview(out, s, cur_full, unc_radius_full)
    return out, s


def _tie_zoom_pick_block(img_full: np.ndarray,
                         centre_full: Tuple[float, float],
                         key: str) -> Optional[Tuple[float, float]]:
    """Magnified fragment around a pending tie-point click (two-step pick,
    the operator's request for section '3 · Tie points (disk edges)').

    The FIRST click on the whole-panorama preview only CENTRES this
    magnifier (nothing is fixed yet); the operator then clicks INSIDE the
    enlarged fragment and THAT click is returned as the exact full-res
    panorama pixel. Returns None while the refining click has not
    happened yet.

    The crop is taken at FULL panorama resolution, enlarged with
    nearest-neighbour interpolation (real pixels stay crisp, not smeared)
    by an ADAPTIVE factor so the enlarged fragment is rendered at about
    ``TIE_ZOOM_DISPLAY_WIDTH`` CSS px (f = display width / window side,
    clamped to 1.0…4.0 — a big 760 px window gets a moderate ~1.7x zoom,
    a small window clamped at the panorama border gets a higher one),
    drawn with a red crosshair + orange box at the first-click position.
    Click -> full-res mapping stays exact even when the widget displays
    the fragment at a different size than its natural pixels:
    _click_preview_scaled rescales the click by the reported displayed
    size, and the crop origin is added here.
    """
    ih, iw = img_full.shape[:2]
    cx = float(centre_full[0])
    cy = float(centre_full[1])
    half = int(TIE_ZOOM_HALF_PX)
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = int(round(cx)) + half, int(round(cy)) + half
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(iw, x1), min(ih, y1)
    if x1c - x0c < 8 or y1c - y0c < 8:
        return None                              # degenerate panorama
    crop = img_full[y0c:y1c, x0c:x1c].copy()
    ch_, cw_ = crop.shape[:2]
    # Crosshair + orange pick box at the first-clicked pixel (in crop
    # coordinates — NOT necessarily centred when the window was clamped
    # at the panorama border).
    px = int(round(cx)) - x0c
    py = int(round(cy)) - y0c
    cv2.line(crop, (max(0, px - 20), py), (min(cw_ - 1, px + 20), py),
             (255, 40, 40), 1)
    cv2.line(crop, (px, max(0, py - 20)), (px, min(ch_ - 1, py + 20)),
             (255, 40, 40), 1)
    cv2.rectangle(crop, (px - 10, py - 10), (px + 10, py + 10),
                  (255, 150, 0), 1)
    # Adaptive nearest-neighbour zoom: the enlarged fragment should be
    # about TIE_ZOOM_DISPLAY_WIDTH CSS px wide (see the constants block).
    f = max(1.0, min(4.0, float(TIE_ZOOM_DISPLAY_WIDTH)
                     / float(max(cw_, ch_))))
    if f > 1.01:
        crop = cv2.resize(crop, None, fx=f, fy=f,
                          interpolation=cv2.INTER_NEAREST)
    clamped = (x0 < 0 or y0 < 0 or x1 > iw or y1 > ih)
    cap = ("Click INSIDE this magnified fragment — the clicked pixel "
           "becomes the tie point (the red crosshair is your first "
           "click; the orange box marks it). "
           + ("The window was clamped at the panorama border. " if clamped
              else ""))
    st.caption(cap)
    # scale = the ZOOM FACTOR f: _click_preview_scaled converts the click
    # by "full = displayed / scale" and for the enlarged fragment
    # full-res offset = zoom px / f. The crop origin is added below, so
    # the returned pair is the exact full-res panorama pixel.
    click = _click_preview_scaled(
        crop, key, f,
        display_width=TIE_ZOOM_DISPLAY_WIDTH, allow_upscale=True)
    if click is None:
        return None
    return (float(click[0]) + x0c, float(click[1]) + y0c)



# ---------------------------------------------------------------------------
# Metadata-based pixel-size extraction (TIFF XResolution, ImageJ
# ImageDescription, OME-XML, FEI/SEM vendor tags) + the new top-of-tab
# calibration flow used by the Stitcher / Pair Review tabs.
# ---------------------------------------------------------------------------

# TIFF ResolutionUnit -> µm per "1 inch / 1 cm" unit
#   1 = none (no unit), 2 = inch, 3 = centimeter
_TIFF_RESUNIT_UM = {1: None, 2: 25_400.0, 3: 10_000.0}

_IMAGEJ_UNIT_RE = re.compile(r"unit\s*=\s*([A-Za-zµμ]+)", re.IGNORECASE)
_IMAGEJ_PW_RE = re.compile(r"pixel[_ ]?width\s*=\s*([0-9eE.+-]+)",
                           re.IGNORECASE)
_IMAGEJ_PH_RE = re.compile(r"pixel[_ ]?height\s*=\s*([0-9eE.+-]+)",
                           re.IGNORECASE)
_OME_PHYS_RE = re.compile(
    r"PhysicalSize(?:X|Y)\s*=\s*\"([0-9eE.+-]+)\"", re.IGNORECASE)
_FEI_PIX_RE = re.compile(
    r"<\s*PixelSize[XY]?[^>]*>\s*([0-9eE.+-]+)", re.IGNORECASE)
_FEI_PW_RE = re.compile(
    r"<\s*pixel[_ ]?width[^>]*>\s*([0-9eE.+-]+)", re.IGNORECASE)


def _tiff_resolution_to_um_per_px(xres, res_unit) -> Optional[float]:
    """Convert TIFF XResolution (a (num, denom) rational or a float) +
    ResolutionUnit (1/2/3) into µm per pixel. Returns None if no unit."""
    if res_unit is None:
        return None
    try:
        ru = int(res_unit)
    except (TypeError, ValueError):
        return None
    um_per_unit = _TIFF_RESUNIT_UM.get(ru)
    if um_per_unit is None:
        return None
    try:
        # xres may be a tuple (numerator, denominator) -> px per unit
        if isinstance(xres, (tuple, list)) and len(xres) == 2:
            num, denom = float(xres[0]), float(xres[1])
            if denom == 0:
                return None
            px_per_unit = num / denom
        else:
            px_per_unit = float(xres)
    except (TypeError, ValueError):
        return None
    if px_per_unit <= 0:
        return None
    return um_per_unit / px_per_unit


def _parse_imagej_description(desc: str) -> Optional[float]:
    """Parse ImageJ-style ImageDescription for µm/px.

    ImageJ calibration is usually written as::

        ImageJ=1.53c
        ...
        unit=micron
        pixel_width=0.45
        pixel_height=0.45
        ...

    Some versions use 'µm' / 'μm' / 'um' as the unit name. A
    ``pixel_width`` without a recognised unit is interpreted as µm (the
    most common case for microscope captures).
    """
    if not desc:
        return None
    um_factor: Optional[float] = None
    m = _IMAGEJ_UNIT_RE.search(desc)
    if m:
        unit = m.group(1).strip().lower()
        if unit in ("micron", "µm", "μm", "um", "u"):
            um_factor = 1.0
        elif unit == "nm":
            um_factor = 1e-3
        elif unit == "mm":
            um_factor = 1e3
    pw = _IMAGEJ_PW_RE.search(desc)
    ph = _IMAGEJ_PH_RE.search(desc)
    for match in (pw, ph):
        if match and um_factor is not None:
            try:
                v = float(match.group(1))
                if v > 0:
                    return v * um_factor
            except ValueError:
                pass
    # Fallback: pixel_width without a recognised unit -> assume µm
    if pw and um_factor is None:
        try:
            return float(pw.group(1))
        except ValueError:
            pass
    return None


def _parse_ome_xml_pixel_size(desc: str) -> Optional[float]:
    """Parse OME-XML ``PhysicalSizeX`` / ``PhysicalSizeY`` from an
    ImageDescription. The OME schema defines the default unit of
    ``PhysicalSize*`` as µm, so the value is returned as-is."""
    if not desc or "PhysicalSize" not in desc:
        return None
    for s in _OME_PHYS_RE.findall(desc):
        try:
            v = float(s)
            if v > 0:
                return v
        except ValueError:
            continue
    return None


def _parse_fei_xml_pixel_size(xml_str: str) -> Optional[float]:
    """Best-effort pixel-size extraction from FEI / SEM XML metadata."""
    if not xml_str:
        return None
    for pat in (_FEI_PIX_RE, _FEI_PW_RE):
        m = pat.search(xml_str)
        if m:
            try:
                v = float(m.group(1))
                if v > 0:
                    return v
            except ValueError:
                pass
    return None


def extract_pixel_size_metadata(path) -> List[dict]:
    """Extract candidate pixel-size readings from one image file.

    Returns a list of dicts ``{"um_per_px": float, "source": str,
    "raw": str}`` ordered by reliability (TIFF resolution tags first,
    then ImageJ description, then OME-XML, then vendor tags). Empty
    list if nothing was found or the file cannot be opened. Non-TIFF
    formats are not scanned (microscope cameras almost always write
    TIFF; JPEG EXIF has no reliable pixel-size information).
    """
    out: List[dict] = []
    if path is None:
        return out
    try:
        p = Path(path)
    except Exception:
        return out
    if not p.exists():
        return out
    ext = p.suffix.lower()
    if ext not in (".tif", ".tiff"):
        return out
    try:
        with tifffile.TiffFile(str(p)) as tf:
            if not tf.pages:
                return out
            page = tf.pages[0]
            tags = {}
            try:
                for tag in page.tags.values():
                    tags[tag.name] = tag.value
            except Exception:
                pass
            # 1) TIFF resolution tags
            xres = tags.get("XResolution")
            yres = tags.get("YResolution")
            res_unit = tags.get("ResolutionUnit")
            if xres and res_unit:
                um = _tiff_resolution_to_um_per_px(xres, res_unit)
                if um and um > 0:
                    out.append({
                        "um_per_px": float(um),
                        "source": "TIFF XResolution + ResolutionUnit",
                        "raw": (f"XResolution={xres}, "
                                f"YResolution={yres}, "
                                f"ResolutionUnit={res_unit}"),
                    })
            # 2) ImageDescription (ImageJ or OME-XML)
            desc = tags.get("ImageDescription")
            if isinstance(desc, bytes):
                try:
                    desc = desc.decode("utf-8", errors="replace")
                except Exception:
                    desc = None
            if desc and isinstance(desc, str):
                if "ImageJ" in desc or "unit=" in desc:
                    um_ij = _parse_imagej_description(desc)
                    if um_ij and um_ij > 0:
                        out.append({
                            "um_per_px": float(um_ij),
                            "source": "ImageDescription (ImageJ)",
                            "raw": desc[:300],
                        })
                if "PhysicalSize" in desc:
                    um_ome = _parse_ome_xml_pixel_size(desc)
                    if um_ome and um_ome > 0:
                        out.append({
                            "um_per_px": float(um_ome),
                            "source": "ImageDescription (OME-XML)",
                            "raw": desc[:300],
                        })
            # 3) Vendor TIFF tags (FEI/SEM XML)
            for vname in ("FEI_HELIOS", "FEI", "XmlMicroscopeDescription",
                          "FEI_XML", "FEI_TITAN"):
                v = tags.get(vname)
                if v:
                    s = (v if isinstance(v, str)
                         else (v.decode("utf-8", errors="replace")
                               if isinstance(v, bytes) else str(v)))
                    pixsize = _parse_fei_xml_pixel_size(s)
                    if pixsize and pixsize > 0:
                        out.append({
                            "um_per_px": float(pixsize),
                            "source": f"TIFF vendor metadata ({vname})",
                            "raw": s[:300],
                        })
                        break
    except Exception:
        pass
    return out


def scan_metadata_for_tiles(tiles) -> dict:
    """Scan the metadata of every tile in the series.

    ``tiles`` is a list of TileInfo-like objects (anything with a
    ``.name`` and a ``.path`` attribute, or a dict with the same keys).

    Returns a dict::

        {
          "per_tile": [(name, [candidate_dict, ...]), ...],
          "per_tile_found": int,             # tiles with >=1 candidate
          "common_um_per_px": Optional[float],  # median of all found values
          "consistent": bool,                # all values agree within 1%
          "sources": List[str],              # distinct source labels
        }
    """
    per_tile: List[Tuple[str, List[dict]]] = []
    found = 0
    values: List[float] = []
    sources: List[str] = []
    for t in tiles:
        name = _tile_name(t)
        path = _tile_path(t)
        cands = extract_pixel_size_metadata(path) if path is not None else []
        per_tile.append((name, cands))
        if cands:
            found += 1
            for c in cands:
                values.append(c["um_per_px"])
                if c["source"] not in sources:
                    sources.append(c["source"])
    common: Optional[float] = None
    consistent = True
    if values:
        vs = sorted(values)
        common = vs[len(vs) // 2]
        if len(set(values)) > 1:
            vmin, vmax = min(values), max(values)
            consistent = (vmax - vmin) / max(vmin, 1e-12) < 0.01
    return {
        "per_tile": per_tile,
        "per_tile_found": found,
        "common_um_per_px": (float(common) if common is not None else None),
        "consistent": bool(consistent),
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Tile-info shims — accept either a TileInfo (from io_utils) or a dict
# pseudo-tile (for the Cross-section Map on-demand fallback where we only
# have the panorama image and no source path).
# ---------------------------------------------------------------------------

def _tile_image_rgb(tile) -> np.ndarray:
    """Get an RGB uint8 image from a TileInfo or pseudo-tile dict."""
    if isinstance(tile, dict):
        img = tile.get("image")
    else:
        img = getattr(tile, "image", None)
    if img is None:
        return np.zeros((4, 4, 3), dtype=np.uint8)
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # TileInfo convention is BGR; pseudo-tiles built from saved panorama
    # entries are already RGB. Detect by nothing — caller should pass
    # the right thing. For pseudo-tiles from the store we store RGB, so
    # we treat dict tiles as RGB; TileInfo as BGR.
    if isinstance(tile, dict):
        return np.ascontiguousarray(img.astype(np.uint8))
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _tile_path(tile):
    if isinstance(tile, dict):
        return tile.get("path")
    return getattr(tile, "path", None)


def _tile_name(tile) -> str:
    if isinstance(tile, dict):
        return str(tile.get("name", "?"))
    return str(getattr(tile, "name", "?"))


def _sig_key(sig) -> str:
    """Stable short hash of an upload signature for state keying."""
    if sig is None:
        return "none"
    try:
        return _hl.md5(repr(sig).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "sig"


# ---------------------------------------------------------------------------
# ROI picker + per-mode calibration blocks (the two manual-flow modes:
# mode 1 = one click on the bar, mode 2 = ROI + two clicks on the bar
# ends — the old auto-search modes and the old 3 / 4 numbering were
# removed/renumbered at the operator's request; mode 1 is the default).
# All operate on a single tile's RGB image.
# ---------------------------------------------------------------------------

def _pick_roi_block(img_rgb: np.ndarray, key_prefix: str,
                    label: str = "ROI") -> Optional[Tuple[int, int, int, int]]:
    """Two-click ROI selector. Returns (x0, x1, y0, y1) in full-res px.

    Replaces the old four-slider picker: the user clicks the UPPER-LEFT
    corner and then the LOWER-RIGHT corner of the rectangle directly on
    the (annotated) clickable preview — what you click is what you get.
    Corner order does not matter (the extremes are used, so dragging is
    not required). A third click starts a NEW rectangle. The rectangle
    is drawn on the preview immediately, and the numeric bounds are
    shown in the caption for reference.
    """
    h, w = img_rgb.shape[:2]
    if w < 8 or h < 8:
        return None
    state = _state()
    clicks_key = f"{key_prefix}_clicks"
    clicks: List[Tuple[float, float]] = list(state.get(clicks_key) or [])
    prev, s = downscale(img_rgb, PREVIEW_SIDE_AUX)
    disp = prev.copy()
    for i, c in enumerate(clicks[:2]):
        px, py = int(round(c[0] * s)), int(round(c[1] * s))
        cv2.circle(disp, (px, py), 6, (255, 40, 40), -1)
        cv2.putText(disp, str(i + 1), (px + 8, max(12, py - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 40, 40), 1,
                    cv2.LINE_AA)
    roi: Optional[Tuple[int, int, int, int]] = None
    if len(clicks) >= 2:
        (ax, ay), (bx, by) = clicks[0], clicks[1]
        x0, x1 = sorted((int(round(ax)), int(round(bx))))
        y0, y1 = sorted((int(round(ay)), int(round(by))))
        x0 = max(0, min(w - 4, x0))
        x1 = max(x0 + 4, min(w - 1, x1))
        y0 = max(0, min(h - 4, y0))
        y1 = max(y0 + 4, min(h - 1, y1))
        roi = (x0, x1, y0, y1)
        cv2.rectangle(disp, (int(x0 * s), int(y0 * s)),
                      (int(x1 * s), int(y1 * s)), (40, 200, 80), 2)
    # The clickable widget shows the ANNOTATED preview: placed corners and
    # the rectangle are visible in the very image the user clicks on.
    # Deterministic display width (operator-adjustable, _cal_display_width):
    # the snapshot is never cut off — not on the first render, not after a
    # window or sidebar resize (see _click_preview_scaled) — and is LARGE
    # enough to hit the scale bar precisely.
    click = _click_preview_scaled(disp, f"{key_prefix}_ck", s,
                                  display_width=_cal_display_width(),
                                  allow_upscale=True)
    if click is not None:
        if len(clicks) >= 2:
            clicks = []                    # third click -> start a new ROI
        clicks.append(click)
        state[clicks_key] = clicks
        st.rerun()

    def _reset_roi() -> None:
        """on_click callback (runs before the widgets of the next rerun)."""
        state[clicks_key] = []

    st.caption((f"{label}: ({roi[0]},{roi[2]}) → ({roi[1]},{roi[3]}) · "
                f"{roi[1] - roi[0]}×{roi[3] - roi[2]} px of {w}×{h}")
               if roi is not None else
               (f"{label}: click 1 = upper-left corner, click 2 = "
                "lower-right corner (a 3rd click starts a new rectangle)"))
    if st.button(f"Reset {label}", key=f"{key_prefix}_reset",
                 on_click=_reset_roi):
        st.rerun()
    return roi


def _roi_crop(img_rgb: np.ndarray,
              roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = roi
    return np.ascontiguousarray(img_rgb[y0:y1, x0:x1, :].copy())


def _roi_manual_two_clicks_block(img_rgb: np.ndarray, key_prefix: str,
                                 roi: Tuple[int, int, int, int]
                                 ) -> Optional[Calib]:
    """Mode 2 — ROI + two clicks on the bar ends + length input.

    The user clicks the two ends of the scale bar inside the ROI
    preview; the picked line is drawn IN RED ON THE CLICKABLE PREVIEW
    ITSELF and updates LIVE: the annotated image passed to the click
    widget is rebuilt from the CURRENT numeric values (read from
    session_state BEFORE the widgets are instantiated), so every change
    of end-1/end-2 coordinates — typed by hand or placed by a click —
    is visible on the image on the very next rerun.

    The scale bar is a strictly horizontal line, so while
    **Force horizontal line** is ON (default) y2 is kept EXACTLY equal
    to y1: the clicked y of point 2 is replaced by the y of point 1,
    the length is |x2 − x1| and the y2 field is locked. The old
    "Snap y2 → y1" button is gone — the checkbox makes it unnecessary.
    Turning the checkbox OFF restores two free points.

    Click coordinates are pushed into the numeric input widget states
    ONCE (at the moment of the click, BEFORE those widgets are
    instantiated in this run); they are never re-applied on later
    reruns, so values typed by hand persist.
    """
    state = _state()
    crop = _roi_crop(img_rgb, roi)
    mkey = f"{key_prefix}_mline"
    mline = state.setdefault(mkey, [])
    horiz_key = f"{key_prefix}_horiz"
    xs_key, ys_key = f"{key_prefix}_mlp1x", f"{key_prefix}_mlp1y"
    xe_key, ye_key = f"{key_prefix}_mlp2x", f"{key_prefix}_mlp2y"

    def _num(key: str) -> float:
        """Current widget value (read BEFORE the widget is instantiated —
        reading is always legal, so the preview below is always LIVE)."""
        try:
            v = st.session_state.get(key, 0.0)
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    horiz_now = bool(st.session_state.get(horiz_key, True))
    p1_now = (_num(xs_key), _num(ys_key))
    p2_now = (_num(xe_key), p1_now[1] if horiz_now else _num(ye_key))
    st.markdown("**Mode 2 · Manual** — click the two ends of the scale "
                "bar on the ROI preview (or type the coordinates), then "
                "enter its length. The red line on the preview updates "
                "LIVE with every coordinate change. The bar is "
                "horizontal: with **Force horizontal line** ON the two "
                "y coordinates are kept equal automatically.")
    prev, s = downscale(crop, PREVIEW_SIDE_AUX)
    annotated = prev.copy()
    if (p1_now[0] or p1_now[1]) and (p2_now[0] or p2_now[1]):
        pa = (int((p1_now[0] - roi[0]) * s), int((p1_now[1] - roi[2]) * s))
        pb = (int((p2_now[0] - roi[0]) * s), int((p2_now[1] - roi[2]) * s))
        cv2.line(annotated, pa, pb, (255, 40, 40), 2)
        for p, tag in ((pa, "1"), (pb, "2")):
            cv2.circle(annotated, p, 5, (255, 40, 40), -1)
            cv2.putText(annotated, tag, (p[0] + 7, max(12, p[1] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 40, 40), 1,
                        cv2.LINE_AA)
    click = _click_preview_scaled(annotated, f"{key_prefix}_ck", s,
                                  display_width=_cal_display_width(),
                                  allow_upscale=True)
    if click is not None and len(mline) < 2:
        # click is in CROP coordinates; convert to full-res.
        full_x = click[0] + roi[0]
        full_y = click[1] + roi[2]
        first_point = len(mline) == 0
        if (not first_point and horiz_now
                and (p1_now[0] or p1_now[1])):
            # Scale bars are horizontal: keep y2 EXACTLY equal to the
            # current y1 (the clicked y of point 2 is vertical jitter).
            full_y = float(p1_now[1])
        mline.append((full_x, full_y))
        # Push the picked point into the number-input states ONCE, here —
        # the number inputs are NOT yet instantiated in this run, so the
        # mutation is legal. Re-applying on every rerun would silently
        # overwrite whatever the operator typed by hand.
        tgt_x, tgt_y = ((xs_key, ys_key) if first_point
                        else (xe_key, ye_key))
        st.session_state[tgt_x] = float(full_x)
        st.session_state[tgt_y] = float(full_y)
        st.rerun()
    horiz = st.checkbox(
        "Force horizontal line (scale bar) — keep y2 = y1",
        value=True, key=horiz_key,
        help=("ON: the second y is forced equal to the first y — the "
              "picked line is strictly horizontal and its length is "
              "|x2 − x1|, as a scale bar should be. OFF: the two points "
              "are used exactly as picked/typed."))
    c1, c2 = st.columns(2)
    p1x = c1.number_input("end 1 · x (full-res px)", min_value=0.0,
                          step=1.0, key=xs_key)
    p1y = c1.number_input("end 1 · y (full-res px)", min_value=0.0,
                          step=1.0, key=ys_key)
    p2x = c2.number_input("end 2 · x (full-res px)", min_value=0.0,
                          step=1.0, key=xe_key)
    p2y = c2.number_input("end 2 · y (full-res px)", min_value=0.0,
                          step=1.0, key=ye_key, disabled=horiz,
                          help=("Locked while 'Force horizontal line' is "
                                "ON — y2 is kept equal to y1."))
    lc, uc = st.columns(2)
    length = lc.number_input("Scale bar length", min_value=1e-9,
                             value=100.0, step=1.0,
                             key=f"{key_prefix}_len", format="%.6g")
    unit = uc.selectbox("Unit", ["µm", "nm", "mm"],
                        key=f"{key_prefix}_unit")
    b1, b2 = st.columns(2)
    apply = b1.button("Apply manual calibration",
                      key=f"{key_prefix}_apply")

    def _clear_line() -> None:
        """on_click callback (runs BEFORE the widgets of the next rerun,
        so mutating the input states here is legal)."""
        state[mkey] = []
        for k in (xs_key, ys_key, xe_key, ye_key):
            st.session_state[k] = 0.0

    if b2.button("Clear picked line", key=f"{key_prefix}_clearml",
                 on_click=_clear_line):
        st.rerun()
    p1 = (float(p1x), float(p1y))
    # While the line is forced horizontal, everything below uses
    # p2y := p1y — what you see (a horizontal line) is what you get.
    p2 = (float(p2x), float(p1y) if horiz else float(p2y))
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    dy_txt = " · dy = 0 (forced horizontal)" if horiz else ""
    st.caption(f"Line: ({p1[0]:.0f},{p1[1]:.0f}) → ({p2[0]:.0f},{p2[1]:.0f})"
               f" = {dist:.0f} px{dy_txt} — the preview above updates "
               "live with every change.")
    if apply:
        cand = calib_from_manual(p1, p2, float(length), unit)
        if cand is None:
            st.error("Invalid line or length — pick two DISTINCT x "
                     "positions and enter a positive length.")
        else:
            state[mkey] = [p1, p2]
            st.success(f"Applied: {(1.0 / cand.px_per_um):.4f} µm/px")
            return cand
    return None


def _click_trace_bar_block(img_rgb: np.ndarray, key_prefix: str,
                           ) -> Optional[Calib]:
    """Mode 1 — ONE click on the scale bar; the app traces the whole bar.

    The operator clicks any pixel that belongs to the bar (whatever its
    colour); ``trace_scale_bar_from_click`` grows the trace left/right
    along the clicked row by colour similarity, so the full bar length is
    measured without the operator having to hit both ends precisely (the
    failure mode of mode 2 on thin bars). The traced line is drawn on the
    preview; the value (µm) is then read by OCR or typed in and applied —
    nothing is applied automatically.
    """
    state = _state()
    tkey = f"{key_prefix}_trace"
    st.markdown(
        "**Mode 1 · Click on the bar** — click anywhere ON the scale-bar "
        "line (the bar may be red, blue, white, black — any colour). The "
        "app measures the colour of the clicked pixel and traces the "
        "whole horizontal line through it, so you do NOT need to hit "
        "both ends precisely. If the trace stops early (noisy tips) or "
        "runs into same-coloured text, adjust the tolerance below and "
        "click again.")
    sig_now = _img_content_sig(img_rgb)
    tol = st.slider(
        "Colour tolerance (RGB distance, 0–255)", 10, 120, 40,
        key=f"{key_prefix}_tol",
        help=("How different a pixel may be from the clicked pixel's "
              "colour and still count as part of the bar. Raise it when "
              "the trace stops in the middle of the bar (noisy JPEG "
              "artefacts); lower it when it swallows the label text or "
              "runs across the whole image."))
    trace = state.get(tkey)
    if trace is not None and trace.get("sig") != sig_now:
        trace = None                            # another tile -> stale
    prev, s = downscale(img_rgb, PREVIEW_SIDE_AUX)
    annotated = prev.copy()
    if trace is not None:
        pa = (int(trace["x0"] * s), int(trace["y"] * s))
        pb = (int(trace["x1"] * s), int(trace["y"] * s))
        cv2.line(annotated, pa, pb, (255, 40, 40), 2)
        for p, tag in ((pa, "1"), (pb, "2")):
            cv2.circle(annotated, p, 5, (40, 200, 80), -1)
            cv2.putText(annotated, tag, (p[0] + 7, max(12, p[1] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 80), 1,
                        cv2.LINE_AA)
        cv2.putText(annotated, f"{trace['len_px']:.0f} px",
                    (pa[0], max(14, pa[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 40, 40), 2,
                    cv2.LINE_AA)
    # Deterministic display width (operator-adjustable, _cal_display_width):
    # the FULL tile — scale bar included — is visible from the very FIRST
    # render of the Calibration mode (the operator's request: the first
    # click must be able to land on the scale-bar pixel), stays fully
    # visible when the sidebar or the window is resized afterwards, and is
    # BIGGER than before (larger default + higher snapshot resolution) so
    # the thin bar is easy to hit with the cursor.
    click = _click_preview_scaled(annotated, f"{key_prefix}_ck", s,
                                  display_width=_cal_display_width(),
                                  allow_upscale=True)
    if click is not None:
        full_x, full_y = float(click[0]), float(click[1])
        result = trace_scale_bar_from_click(img_rgb, int(full_x),
                                            int(full_y), tol=float(tol))
        if result is None:
            st.warning("Nothing similar to the clicked pixel on that row "
                       "— make sure you clicked ON the bar itself, or "
                       "raise the colour tolerance.")
            state.pop(tkey, None)
        else:
            result["sig"] = sig_now
            state[tkey] = result
            st.rerun()

    def _clear_trace() -> None:
        state.pop(tkey, None)

    if trace is not None:
        st.success(f"Traced bar: {trace['len_px']:.0f} px "
                   f"({trace['x0']},{trace['y']}) → "
                   f"({trace['x1']},{trace['y']}) · reference colour "
                   f"RGB {trace['ref_color']}. Check the ZOOMED view "
                   "below — the green endpoints must sit exactly on "
                   "the bar tips.")
        # ---- zoomed verification view (operator request) ------------------
        # A fragment of the tile around the found bar, ENLARGED, with the
        # traced line and endpoints drawn — the operator must be able to
        # SEE the found bar at magnification and confirm it is correct
        # (endpoints on the tips, label legible above/below) before
        # applying the calibration.
        _zx0 = int(trace["x0"])
        _zx1 = int(trace["x1"])
        _zy = int(trace["y"])
        _ln = max(1, _zx1 - _zx0)
        _pad_y = min(300, max(40, int(_ln * 0.25)))    # label room above/below
        _pad_x = max(20, int(_ln * 0.08))
        _ih, _iw = img_rgb.shape[:2]
        _cy0, _cy1 = max(0, _zy - _pad_y), min(_ih, _zy + _pad_y + 1)
        _cx0, _cx1 = max(0, _zx0 - _pad_x), min(_iw, _zx1 + _pad_x)
        zoom = img_rgb[_cy0:_cy1, _cx0:_cx1].copy()
        _zyz = _zy - _cy0
        cv2.line(zoom, (_zx0 - _cx0, _zyz), (_zx1 - _cx0, _zyz),
                 (255, 40, 40), 2)
        cv2.circle(zoom, (_zx0 - _cx0, _zyz), 5, (40, 200, 80), -1)
        cv2.circle(zoom, (_zx1 - _cx0, _zyz), 5, (40, 200, 80), -1)
        _zh, _zw = zoom.shape[:2]
        _target = 900.0
        _f = max(0.05, min(6.0, _target / max(1, _zw)))
        if abs(_f - 1.0) > 0.02:
            _interp = (cv2.INTER_NEAREST if _f > 1.0 else cv2.INTER_AREA)
            zoom = cv2.resize(zoom, None, fx=_f, fy=_f,
                              interpolation=_interp)
        _mag = (_f if _f >= 1.0 else 1.0)
        st.image(zoom, use_container_width=True,
                 caption=(f"Zoomed check (×{_mag:.1f} magnification of "
                          "this fragment): the red traced line must cover "
                          "the FULL bar — green endpoints exactly on the "
                          "bar tips, no bar colour left outside them, the "
                          "value label legible above or below the bar."))
        # ---- value: OCR or manual, then Apply -----------------------------
        full_bar = ScaleBar(x0=trace["x0"], y0=trace["y"],
                            x1=trace["x1"], y1=trace["y"] + 1, score=1.0)
        bkey = f"{key_prefix}_val"
        vkey = f"{key_prefix}_value"
        if not _HAS_PYTESSERACT:
            st.caption("OCR is unavailable on this machine — type the "
                       "bar value below (the traced line still works).")
        if st.button("Read the value near the traced line (OCR — above AND "
                     "below the bar)",
                     key=f"{key_prefix}_ocr", disabled=not _HAS_PYTESSERACT,
                     help=("Runs tesseract on the label windows ABOVE and "
                           "BELOW the traced bar (they grow TALLER and "
                           "WIDER automatically while no trustworthy "
                           "value is found, so a label far from the bar "
                           "is captured too). The label pixels are "
                           "ISOLATED BY THE BAR'S COLOUR first (the label "
                           "is drawn in the same colour as the bar), every "
                           "reading is ranked, and only a trustworthy value "
                           "(an explicit unit or a plausible scale-bar "
                           "magnitude) pre-fills the field — noise digits "
                           "like a bare '3' can no longer win. Check the "
                           "reading, correct it if needed, then press "
                           "Apply.")):
            with st.spinner("OCR of the scale-bar label…"):
                try:
                    raw, ocr_status, ocr_crop = ocr_scale_label(
                        img_rgb, full_bar,
                        ref_color=trace.get("ref_color"))
                except Exception as e:
                    raw, ocr_status, ocr_crop = None, f"error:{e}", None
            # ONLY a "parsed@" status carries a TRUSTWORTHY value (an
            # explicit unit or a plausible magnitude). The raw text is
            # still shown for transparency, but it no longer pre-fills
            # the value field — that was the "3 instead of 200" bug.
            trusted = bool(ocr_status and ocr_status.startswith("parsed@"))
            parsed = parse_scale_text(raw) if trusted else None
            state[bkey] = {"raw": raw, "parsed": parsed,
                           "status": ocr_status, "crop": ocr_crop}
            if parsed is not None:
                st.session_state[vkey] = float(parsed[0])       # pre-fill
        vcache = state.get(bkey)
        if vcache is not None:
            ocr_crop = vcache.get("crop")
            _status_txt = str(vcache.get("status") or "")
            _where = ""
            if "@" in _status_txt:
                _where = _status_txt.split("@", 1)[1]
            if ocr_crop is not None and ocr_crop.size:
                cprev, _ = downscale(ocr_crop, 360)
                st.image(cprev, use_container_width=True,
                         caption=("The label window the OCR was run on"
                                  + (f" ({_where})." if _where else ".")))
            raw = vcache.get("raw")
            parsed = vcache.get("parsed")
            status = _status_txt
            if parsed is not None:
                st.success(f"OCR read **{parsed[1]}** → {parsed[0]:g} µm "
                           + (f"(window {_where})" if _where else "")
                           + " — check it and correct below if wrong.")
            elif status.startswith("error:"):
                st.error("OCR failed: " + status[len("error:"):] +
                         " — type the value manually below.")
            elif raw:
                _wtxt = f" (window {_where})" if _where else ""
                st.info(f"OCR text: {raw!r}{_wtxt} — no trustworthy scale "
                        "value in it (a bare noise digit is NOT applied "
                        "automatically any more). Type the value manually "
                        "below.")
        value = st.number_input(
            "Scale bar value (µm)", min_value=1e-9, value=100.0,
            step=1.0, key=vkey, format="%.6g",
            help=("Real length of the traced bar in micrometres (e.g. "
                  "'100 µm' printed next to it)."))
        if st.button("Apply calibration", key=f"{key_prefix}_apply"):
            cand = calib_from_manual((float(trace["x0"]), float(trace["y"])),
                                     (float(trace["x1"]), float(trace["y"])),
                                     float(value), "µm")
            if cand is None:
                st.error("Invalid trace — click on the bar again.")
            else:
                raw = str(vcache.get("raw") or "") if vcache else ""
                cand.ocr_text = raw
                st.success(f"Applied: traced {trace['len_px']:.0f} px = "
                           f"{float(value):g} µm → "
                           f"**{(1.0 / cand.px_per_um):.4f} µm/px**.")
                return cand
        if st.button("Clear trace", key=f"{key_prefix}_cleartr",
                     on_click=_clear_trace):
            st.rerun()
    else:
        st.caption("Click on the preview above — one click anywhere on "
                   "the bar is enough (a third click replaces the "
                   "current trace).")
        if st.button("Clear trace", key=f"{key_prefix}_cleartr0",
                     on_click=_clear_trace):
            st.rerun()
    return None


# ---------------------------------------------------------------------------
# Top-of-tab calibration flow (Stitcher / Pair Review) + helpers
# ---------------------------------------------------------------------------

def render_calibration_section(tiles, role: str,
                               sig=None) -> Calib:
    """Top-of-tab calibration block. Implements the agreed flow:

    **Step 1 — Auto-search the metadata** of every tile in the series
    (TIFF XResolution + ResolutionUnit, ImageJ ``ImageDescription``,
    OME-XML ``PhysicalSize*``, FEI/SEM vendor tags).

    **Step 2 — Action.** Accept the metadata calibration (when one was
    found) OR enter µm per pixel manually (a direct numeric entry — the
    operator types the length of ONE pixel in µm, no scale-bar clicking
    needed) OR skip to the manual scale-bar flow below (always
    available, whether metadata was found or not).

    **Step 3 — Manual flow (after Skip).** The user picks the tile
    of the series where the scale bar is visible (the bar is common to
    the whole series), then one of TWO modes (the old auto-search modes
    were removed and the two click-based modes were renumbered 1 / 2 at
    the operator's request — mode 1 is the FIRST option and therefore
    the DEFAULT, mode 2 is the precise manual fallback):

    1. **One click on the bar (default, universal)** — click ANY pixel
       that belongs to the bar (whatever its colour: red, blue, white,
       black); the app traces the whole horizontal line through that
       pixel by colour similarity, shows a ZOOMED verification view of
       the found bar, so thin bars no longer require precise
       end-to-end clicking. The value is read by OCR (windows above
       AND below the bar) or typed in, then applied explicitly.
    2. **ROI + manual two clicks** — draw a ROI, click the two ends of
       the bar to build a strictly horizontal line (y2 is kept equal
       to y1), enter its length in µm / nm / mm → µm per pixel is
       computed.

    The resulting ``Calib`` is keyed by ``sig`` (the upload signature)
    in ``st.session_state["xmap_state"]["calibs_by_sig"]`` so the
    Stitcher / Pair Review tabs share ONE calibration per upload. The
    "Save to Cross-section Map" button attaches it to the saved entry
    so the Cross-section Map tab can read it without re-calibrating.

    Returns the current ``Calib``.
    """
    state = _state()
    calib_store = state.setdefault("calibs_by_sig", {})
    key = f"{role}:{_sig_key(sig)}"
    calib = calib_store.get(key, Calib())

    # ---- Widget-choice memory --------------------------------------------
    # Streamlit deletes the state of widgets that are not rendered on two
    # consecutive runs — and the whole calibration block exists only in
    # the tab that shows it. After a Stitcher <-> Pair Review switch the
    # "Calibration mode" radio and the tile selectbox silently RESET,
    # which left the block in a half-restored state (the operator saw
    # mode 1 with no image where mode 2 had been). Mirror both choices
    # into the plain (never-cleaned) xmap_state dict via on_change
    # callbacks and re-seed the widget keys whenever Streamlit dropped
    # them. (The example in the original comment referred to the old
    # auto-search modes; with the current modes 1 / 2 the same reset
    # would show one mode with no image where the other had been.)
    mode_key = f"cal_mode_{key}"
    tile_key = f"cal_tile_{key}"
    mode_mem_key = f"{key}::mode_mem"
    tile_mem_key = f"{key}::tile_mem"

    def _remember_mode() -> None:
        state[mode_mem_key] = st.session_state.get(mode_key)

    def _remember_tile() -> None:
        state[tile_mem_key] = st.session_state.get(tile_key)

    if tile_key not in st.session_state and state.get(tile_mem_key) is not None:
        st.session_state[tile_key] = state[tile_mem_key]
    # Sanitize a STALE mode memory: any stored value that is not among
    # the radio options (the removed auto-search modes, or the old
    # "3 ·" / "4 ·" numbering) would make Streamlit raise a
    # StreamlitAPIException — reset both the widget state and the memory
    # instead; the radio then opens on its DEFAULT (mode 1).
    if mode_key in st.session_state and \
            st.session_state[mode_key] not in CAL_MODE_OPTIONS:
        st.session_state.pop(mode_key, None)
        state.pop(mode_mem_key, None)
    if mode_key not in st.session_state and state.get(mode_mem_key) \
            in CAL_MODE_OPTIONS:
        st.session_state[mode_key] = state[mode_mem_key]

    st.subheader("Calibration (µm per pixel)")
    st.caption("Current: " + calib.summary())

    # ---- Step 1 · metadata scan -----------------------------------------
    md_cache = state.setdefault("md_scan", {})
    if key not in md_cache:
        md_cache[key] = scan_metadata_for_tiles(tiles)
    md = md_cache[key]

    with st.expander(
            f"Step 1 · Auto-search in image metadata "
            f"({len(tiles)} tile(s) in the series)",
            expanded=not calib.ok):
        if md["per_tile_found"] == 0:
            st.info("No pixel-size metadata was found in any of the "
                    f"{len(tiles)} tile(s). This is normal for most "
                    "microscope TIFFs — enter the µm per pixel manually "
                    "below, or pick **Skip** to calibrate from the "
                    "visible scale bar.")
        else:
            rows = []
            for tname, cands in md["per_tile"]:
                if cands:
                    c = cands[0]
                    rows.append({
                        "tile": tname,
                        "µm/px": f"{c['um_per_px']:.6g}",
                        "source": c["source"],
                    })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if md["common_um_per_px"] is not None:
                if md["consistent"]:
                    st.success(
                        f"Suggested: **{md['common_um_per_px']:.6g} µm/px** "
                        f"(consistent across all {md['per_tile_found']} "
                        "tile(s) with metadata).")
                else:
                    st.warning(
                        f"Pixel-size values disagree between tiles — "
                        f"using the median **{md['common_um_per_px']:.6g} "
                        "µm/px**. Check the series or calibrate manually.")
                _plaus = calib_plausibility_warning(
                    1.0 / md["common_um_per_px"],
                    ", ".join(md["sources"]))
                if _plaus:
                    st.error("**Implausible metadata value — verify before "
                             "accepting.** " + _plaus)

    # ---- Step 2 · action ------------------------------------------------
    accept_label = "Accept metadata calibration"
    manual_label = ("Enter µm per pixel manually (type the pixel size — "
                    "no scale bar needed)")
    decline_label = ("Skip — continue with Step 3 · Manual calibration "
                     "(click on the scale bar)")
    if md["common_um_per_px"] is not None:
        options = [accept_label, manual_label, decline_label]
        if calib.method == "metadata":
            default_idx = 0
        else:
            default_idx = 1
    else:
        options = [manual_label, decline_label]
        default_idx = 0
    # The option list depends on the metadata scan, which can CHANGE
    # between runs (tiles re-uploaded, another tab's series): a stored
    # radio value that is no longer offered would make Streamlit raise a
    # StreamlitAPIException — drop the stale state first.
    if st.session_state.get(f"cal_action_{key}") not in options:
        st.session_state.pop(f"cal_action_{key}", None)
    action = st.radio("Step 2 · Action", options, horizontal=True,
                      key=f"cal_action_{key}", index=default_idx)

    if action == accept_label and md["common_um_per_px"] is not None:
        raw_first = ""
        for _tname, _cands in md["per_tile"]:
            if _cands:
                raw_first = str(_cands[0].get("raw") or "")
                break
        _plaus = calib_plausibility_warning(
            1.0 / md["common_um_per_px"],
            ", ".join(md["sources"]), raw_first)
        if _plaus:
            st.error("**This metadata calibration looks WRONG — it was "
                     "applied, but re-check it.** " + _plaus)
        new_calib = calib_from_metadata(
            md["common_um_per_px"],
            ", ".join(md["sources"]) if md["sources"] else "metadata")
        calib_store[key] = new_calib
        st.success(f"Applied metadata calibration: "
                   f"{(1.0 / new_calib.px_per_um):.6g} µm/px.")
        return new_calib
    elif action == accept_label:
        # No metadata found but the user picked Accept -> fall through
        st.info("No pixel-size metadata was found. Enter the µm per pixel "
                "manually below, or switch to **Skip** to calibrate from "
                "a scale bar.")
        return calib

    if action == manual_label:
        # ---- Manual numeric entry (the operator's request: "let me
        # enter the length of ONE pixel in µm myself, or skip the step
        # and continue with Step 3 · Manual calibration") — the value is
        # typed directly, no scale-bar clicking needed. -----------------
        st.markdown(
            "**Manual pixel size** — type the length of ONE pixel in µm "
            "(from the microscope settings / acquisition software) and "
            "press **Apply**. The value applies to the WHOLE series.")
        um_ppx = st.number_input(
            "Length of one pixel (µm)", min_value=1e-9, max_value=1e6,
            value=float(state.get(f"{key}::man_um") or 0.1),
            step=0.01, format="%.6g", key=f"{key}_man_um",
            help="Typical magnitudes: x10 ≈ 0.5–5 µm/px, x100 ≈ "
                 "0.05–0.5 µm/px. This is the value the microscope "
                 "software prints as the pixel size.")
        state[f"{key}::man_um"] = float(um_ppx)
        _plaus = calib_plausibility_warning(1.0 / float(um_ppx),
                                            "manual entry")
        if _plaus:
            st.warning("**This pixel size looks implausible.** " + _plaus)
        if st.button("Apply this calibration", key=f"{key}_man_apply"):
            if float(um_ppx) <= 0:
                st.error("The pixel size must be positive.")
            else:
                new_calib = Calib(
                    px_per_um=1.0 / float(um_ppx), method="manual",
                    metadata_source="manual entry",
                    ocr_text=f"typed by operator: {um_ppx:.6g} µm/px")
                calib_store[key] = new_calib
                st.success(f"Applied manual calibration: "
                           f"{float(um_ppx):.6g} µm/px.")
                return new_calib
        st.caption(
            "Not sure about the value? Switch the 'Step 2 · Action' radio "
            "above to **Skip — continue with Step 3 · Manual calibration**: "
            "the scale bar is then measured directly on the image "
            "(mode 1: one click on the bar, mode 2: ROI + the two bar "
            "ends).")
        return calib

    # ---- Step 3 · manual flow -------------------------------------------
    st.markdown(f"**Step 3 · Manual calibration** — pick the tile that "
                f"shows the scale bar (the bar is common to the whole "
                f"{len(tiles)}-tile series):")
    tile_names = [_tile_name(t) for t in tiles]
    sel_idx = st.selectbox(
        "Choose a tile from the series",
        list(range(len(tiles))),
        format_func=lambda i: tile_names[i],
        key=tile_key, index=0, on_change=_remember_tile)
    sel_tile = tiles[int(sel_idx)]
    img_rgb = _tile_image_rgb(sel_tile)

    # ---- Snapshot display size (operator request) -------------------------
    # The default snapshot was too small to place the cursor ON a
    # scale-bar pixel. The width of ALL calibration-mode clickable
    # snapshots is now adjustable and REMEMBERED for the session (the
    # widget state itself is cleaned by Streamlit after a tab switch, so
    # the choice is mirrored into the plain xmap_state dict and re-seeds
    # the widget — the same pattern as the mode / tile memory above).
    dsize_mem = state.setdefault("cal_display_width", CAL_WIDTH_DEFAULT)
    if dsize_mem not in CAL_WIDTH_OPTIONS:
        dsize_mem = CAL_WIDTH_DEFAULT
        state["cal_display_width"] = dsize_mem
    if "xmap_cal_dsize" not in st.session_state:
        st.session_state["xmap_cal_dsize"] = dsize_mem
    state["cal_display_width"] = st.selectbox(
        "Snapshot display size (px width) — larger = easier to hit the "
        "scale-bar pixel with the cursor",
        CAL_WIDTH_OPTIONS,
        key="xmap_cal_dsize",
        help=("On-screen width of the clickable calibration snapshots "
              "(modes 1 and 2). The FULL snapshot always stays visible "
              "(it is never cropped at any width); a larger value just "
              "enlarges it, so a thin scale bar spans more screen pixels "
              "and is easier to hit. If the widest option overflows your "
              "window, pick a smaller one — the choice is remembered."))

    mode = st.radio(
        "Calibration mode", CAL_MODE_OPTIONS,
        horizontal=True, key=mode_key, on_change=_remember_mode)

    new_calib: Optional[Calib] = None
    if mode.startswith("1"):
        # Mode 1 — one click on the bar, full auto-trace of the line
        # (the DEFAULT — first radio option).
        new_calib = _click_trace_bar_block(img_rgb, key + "_ct1")
    elif mode.startswith("2"):
        # Mode 2 — ROI + two clicks on the bar ends (manual fallback).
        roi = _pick_roi_block(img_rgb, key + "_roi2",
                              label="ROI containing the scale bar")
        if roi is not None:
            new_calib = _roi_manual_two_clicks_block(
                img_rgb, key + "_rm2", roi)

    if new_calib is not None and new_calib.ok:
        calib_store[key] = new_calib
        return new_calib
    return calib


def get_calibration_for_upload(sig) -> Calib:
    """Return the Calib last produced by ``render_calibration_section``
    for the given upload signature (across the Stitcher / Pair Review
    tabs — they share ONE calibration per upload). Returns an empty
    Calib when nothing has been calibrated yet."""
    state = _state()
    calib_store = state.get("calibs_by_sig", {})
    sk = _sig_key(sig)
    # Search both role-prefixed keys for this signature; return the
    # most recently set (any) one.
    for k, calib in calib_store.items():
        if k.endswith(f":{sk}") and calib is not None and calib.ok:
            return calib
    return Calib()


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

def render_map_tab() -> None:
    st.title("Cross-section Map — x10 overview → x100 detail")
    st.caption(
        "Register the stitched x10 panorama (whole disk, shot through the "
        "glass) with the stitched x100 panorama (top view of the IC "
        "cross-section). Calibrate each panorama from its microscope scale "
        "bar (at the top of the Stitcher / Pair Review tabs), pick tie "
        "points on the disk edges, check the calibration consistency, "
        "then click any point of the x10 panorama to get the "
        "corresponding x100 field of view — as a live preview and as a "
        "standalone interactive HTML report.")
    _clear_store_if_requested()
    state = _state()
    store = get_store()

    # ---- 1 · Sources -----------------------------------------------------
    st.subheader("1 · Panorama sources")
    names = [e["name"] for e in store]
    store_sig = _auto_assign_sources(store)   # pre-select overview/detail by
    if not names:                        # resolution (see the helper)
        st.info(
            "No saved panoramas yet. Run a stitch in the **Stitcher** or "
            "**Pair Review** tab and press **Save to Cross-section Map** "
            "next to the Download buttons — once for the x10 series and "
            "once for the x100 series. You can also load panorama image "
            "files below.")
    c10, c100 = st.columns(2)
    with c10:
        st.markdown("**Overview — x10 (whole disk)**")
        sel10 = _pick_entry(names, "xmap_sel_x10")
    with c100:
        st.markdown("**Detail — x100 (cross-section top view)**")
        sel100 = _pick_entry(names, "xmap_sel_x100")
    if names:
        st.caption("Overview/Detail are pre-selected automatically by "
                   "resolution (Overview = the smaller panorama, Detail = "
                   "the larger one) every time the store changes — you can "
                   "still pick different entries manually.")

    up1, up2 = st.columns(2)
    with up1:
        _upload_to_store("Load x10 panorama image", "xmap_up_x10")
    with up2:
        _upload_to_store("Load x100 panorama image", "xmap_up_x100")

    if store:
        st.caption(f"{len(store)} panorama(s) in the session store "
                   f"(max {STORE_LIMIT}).")
        st.button("Clear saved panoramas", key="xmap_clear_store")

    item10 = store[sel10] if sel10 is not None else None
    item100 = store[sel100] if sel100 is not None else None
    # Persist the CURRENT selection against the store signature so it
    # survives the tab switch (Streamlit deletes the selectbox state
    # while this tab is hidden — see _auto_assign_sources).
    if names:
        _remember_selection(store_sig, sel10, sel100)

    # ---- 2 · Scale calibration (read-only) -------------------------------
    # The calibration is done at the TOP of the Stitcher / Pair Review
    # tabs and attached to the saved panorama entry by the "Save to
    # Cross-section Map" button. Per the operator's request there is NO
    # calibration UI here any more — this section only SHOWS what is set
    # and explains where to calibrate when a panorama is not calibrated.
    cal10 = (item10.get("calib") if item10 else None) or Calib()
    cal100 = (item100.get("calib") if item100 else None) or Calib()
    cal10 = cal10 if cal10.ok else Calib()
    cal100 = cal100 if cal100.ok else Calib()
    st.subheader("2 · Scale calibration (µm per pixel)")
    if item10 is None or item100 is None:
        st.caption("Select both panoramas above to see their "
                   "calibrations.")
    elif cal10.ok and cal100.ok:
        st.caption(
            f"x10: **{cal10.summary()}** · x100: **{cal100.summary()}** "
            "— calibrated at the top of the **Stitcher** / **Pair "
            "Review** tabs and attached to the saved panoramas. To "
            "re-calibrate, return to those tabs and redo the flow "
            "there, then press **Save to Cross-section Map** again.")
        # Cross-calibration sanity: the SAME physical disk is seen by both
        # panoramas. Internal unit px/µm: the x100 calibration must be
        # FINER (MORE px/µm) than the x10 one — roughly the magnification
        # ratio (~10x for a x10/x100 pair, 2–30x with adapters/zoom).
        # A ratio outside the band means one of the two calibrations is
        # very likely wrong (default-DPI metadata, a misread scale-bar
        # value, or an INVERTED value from an older export).
        ratio = cal10.px_per_um / max(1e-12, cal100.px_per_um)
        if not (0.005 <= ratio <= 2.0):
            st.error(
                f"**Calibration mismatch between the two panoramas:** "
                f"x10 = {(1.0 / cal10.px_per_um):.4g} µm/px vs x100 = "
                f"{(1.0 / cal100.px_per_um):.4g} µm/px — their ratio is "
                f"{ratio:.3f} px/µm (x10 / x100), outside the plausible "
                "band [0.005 … 2]. For the same disk the x100 panorama "
                "must be FINER (more px per µm) than the x10 overview, "
                "typically by ~2–30x. One of the two calibrations is very "
                "likely WRONG (for example a default-DPI metadata value "
                "was accepted, the scale-bar value was misread, or an "
                "older export re-imported an inverted µm/px). The fragment "
                "scale bar below is drawn with the wrong µm/px until this "
                "is fixed — re-calibrate at the top of the Stitcher / Pair "
                "Review tab and press **Save to Cross-section Map** "
                "again.")
        # Implausibility of either value alone (even with a plausible ratio).
        _w10 = calib_plausibility_warning(cal10.px_per_um, "x10 panorama")
        _w100 = calib_plausibility_warning(cal100.px_per_um, "x100 panorama")
        if _w10:
            st.warning("**x10 calibration looks implausible.** " + _w10)
        if _w100:
            st.warning("**x100 calibration looks implausible.** " + _w100)
    else:
        uncal = ([] +
                 (["x10"] if not cal10.ok else []) +
                 (["x100"] if not cal100.ok else []))
        st.warning(
            f"The {(' / '.join(uncal))} panorama is not calibrated yet. "
            "There is deliberately NO calibration UI here — calibrate "
            "at the top of the **Stitcher** / **Pair Review** tab "
            "(Step 3, modes 3–4) and press **Save to Cross-section "
            "Map** again so the new entry carries the calibration. "
            "Sections 3–5 below need BOTH calibrations.")

    if item10 is None or item100 is None:
        st.warning("Select both panoramas to continue.")
        return

    # ---- 3 · Tie points ---------------------------------------------------
    st.subheader("3 · Tie points (disk edges)")
    st.caption(
        "Pick the SAME physical point on both panoramas — the disk edges "
        "are visible in both. Two points (left edge + right edge) give an "
        "exact mapping; 3–6 points additionally give a real error estimate "
        "from the residuals. TWO-STEP picking: the first click on a "
        "preview does NOT fix the pixel — it opens a MAGNIFIED fragment "
        "around the clicked pixel, and the SECOND click inside that "
        "fragment fixes the exact point (or type the coordinates).")
    pk = f"{item10['name']}||{item100['name']}"
    pairs = state.setdefault("pairs_by_key", {}).setdefault(pk, [])

    # Pending magnifier centres of the two-step pick (the FIRST-click
    # positions), kept per panorama pair — switching to another pair of
    # panoramas starts clean. Nothing reaches the numeric coordinate
    # inputs until the SECOND click inside the magnified fragment (the
    # operator's request: "first show an enlarged fragment around the
    # pixel, and the second click inside it gives the needed pixel").
    tie_zoom = state.setdefault("tie_zoom", {})
    if pk not in tie_zoom:
        tie_zoom.clear()                    # another pair of panoramas
    zs = tie_zoom.setdefault(pk, {})

    # STRUCTURAL EPOCH of the two-step pick (the operator's bug report:
    # "on the first and the second click the SECOND panorama disappears
    # and I cannot pick the corresponding point on it"). Root cause: when
    # the magnifier block APPEARS or DISAPPEARS (first click opens it,
    # second click closes it), Streamlit's frontend re-mounts the OTHER
    # panorama's click widget because elements were inserted/removed
    # before it in the page — but it does NOT re-send the element data to
    # the freshly created component iframe when the element itself is
    # unchanged (same key, same image). The new iframe then renders
    # forever EMPTY: the second panorama "disappears" (the space is
    # reserved, the picture is gone) and no point can be picked on it.
    # Fix: the open/closed state of BOTH magnifiers (and the pair count)
    # is part of BOTH previews' widget keys. Whenever the structure
    # changes, every tie-point widget gets a NEW key -> Streamlit treats
    # it as a NEW element -> the new iframe receives its image data and
    # the other panorama stays fully visible after EVERY click.
    tie_epoch = (f"~z10{int(zs.get('10') is not None)}"
                 f"~z100{int(zs.get('100') is not None)}"
                 f"~np{len(pairs)}"
                 f"~rp{int(st.session_state.get('xmap_tie_restore', 0))}")

    def _tie_pick_flow(img_full: np.ndarray,
                       marker_points: List[Tuple[float, float]],
                       title: str,
                       ck_key: str, zoom_key: str,
                       x_key: str, y_key: str, side: str) -> None:
        """One panorama's two-step tie-point picker: whole-panorama
        preview -> first click centres the magnifier -> magnified
        fragment -> second click fixes the exact point -> coordinates
        are pushed into the numeric input states ONCE (before those
        widgets are instantiated). The pending crosshair is drawn on the
        preview while the magnifier is open; 'Cancel magnifier' closes
        it without fixing anything.

        ``ck_key`` / ``zoom_key`` arrive with the structural epoch
        already appended (see tie_epoch above) so the OTHER panorama's
        widget remounts WITH fresh data whenever the magnifier of THIS
        panorama opens or closes.

        The whole-panorama preview uses the EXPLICIT NUMERIC display
        sizing (TIE_PREVIEW_DISPLAY_WIDTH, allow_upscale=True) instead of
        the CSS stretch mode — a constant frame height is sent from
        Python on every mount/rerun, so a preview re-mounted while the
        magnifier structure changes can never render cut-off or blank
        (part of the fix of the operator's 'the panoramas disappear on
        clicking' report)."""
        st.markdown(title)
        marked, s_marked = _marked_preview(img_full, PREVIEW_SIDE_MAIN,
                                           marker_points, zs.get(side))
        click = _click_preview_scaled(
            marked, ck_key, s_marked,
            display_width=TIE_PREVIEW_DISPLAY_WIDTH, allow_upscale=True)
        if click is not None:
            zs[side] = (float(click[0]), float(click[1]))
            st.rerun()      # rebuild the preview with the pending marker
        if zs.get(side) is None:
            st.caption("Click the preview — a magnified fragment will "
                       "open around the clicked pixel; the exact point "
                       "is fixed by a click inside that fragment.")
            return
        picked = _tie_zoom_pick_block(img_full, zs[side], zoom_key)
        cz1, cz2, _padz = st.columns([1, 1, 3])
        if cz1.button("Cancel magnifier (pick another spot)",
                      key=f"{zoom_key}_cancel",
                      help="Closes the magnified fragment without "
                           "fixing a point — click the preview again "
                           "to centre the magnifier elsewhere."):
            zs.pop(side, None)
            st.rerun()
        if picked is not None:
            _apply_click_to_inputs(picked, x_key, y_key)
            zs.pop(side, None)
            st.rerun()

    # Stacked layout (operator request: each panorama one under another):
    # every preview uses the FULL page width; the magnifier (second step)
    # provides the pixel-precise refinement.
    _tie_pick_flow(item10["image"], [p for p, _ in pairs],
                   "**x10 (overview)** — click where the tie point sits "
                   "(1st click opens the magnifier · 2nd click inside it "
                   "fixes the point)",
                   f"xmap_ck_tie10{tie_epoch}", f"xmap_tiezoom10{tie_epoch}",
                   "xmap_tie10_x", "xmap_tie10_y", "10")
    cc1, cc2, _pad1 = st.columns([1, 1, 3])
    x10 = cc1.number_input("x10 · x", min_value=0.0, step=1.0,
                           key="xmap_tie10_x")
    y10 = cc2.number_input("x10 · y", min_value=0.0, step=1.0,
                           key="xmap_tie10_y")

    _tie_pick_flow(item100["image"], [q for _, q in pairs],
                   "**x100 (detail)** — click the SAME physical point "
                   "(1st click opens the magnifier · 2nd click inside it "
                   "fixes the point)",
                   f"xmap_ck_tie100{tie_epoch}",
                   f"xmap_tiezoom100{tie_epoch}",
                   "xmap_tie100_x", "xmap_tie100_y", "100")
    cc3, cc4, _pad2 = st.columns([1, 1, 3])
    qx = cc3.number_input("x100 · x", min_value=0.0, step=1.0,
                          key="xmap_tie100_x")
    qy = cc4.number_input("x100 · y", min_value=0.0, step=1.0,
                          key="xmap_tie100_y")

    # Belt-and-braces escape hatch for the 'the panoramas disappear on
    # clicking' report (extends the structural epoch above): ANY blank /
    # unresponsive preview can be restored with one click. The button
    # bumps the epoch, BOTH preview widget keys change, Streamlit mounts
    # them as NEW elements and delivers their image data — a stale or
    # empty component iframe gets replaced by a fresh one no matter what
    # caused it (a browser-discarded iframe under memory pressure, a
    # remount race, ...). Also useful after the browser tab was restored.
    rb1, rb2, _rb3 = st.columns([2, 1, 3])
    if rb2.button("Previews blank / not clickable? — Restore",
                  key="xmap_tie_restore_btn",
                  help="Re-mounts both panorama previews with fresh "
                       "widget keys. Use it if a preview ever shows up "
                       "blank or ignores clicks (e.g. after switching "
                       "browser tabs on a low-memory machine). Nothing "
                       "else changes — picked points are kept."):
        st.session_state["xmap_tie_restore"] = (
            int(st.session_state.get("xmap_tie_restore", 0)) + 1)
        st.rerun()

    abtn, dbtn, _ = st.columns([1, 1, 2])
    can_add = (st.session_state.get("xmap_tie10_x") and
               st.session_state.get("xmap_tie100_x") is not None and
               len(pairs) < MAX_TIE_POINTS)
    if abtn.button("Add tie point pair", key="xmap_add_pair",
                   disabled=not can_add):
        p = (float(x10), float(y10))
        q = (float(qx), float(qy))
        pairs.append((p, q))
        st.rerun()
    if pairs and dbtn.button("Remove last pair", key="xmap_remove_last"):
        pairs.pop()
        st.rerun()
    if len(pairs) >= MAX_TIE_POINTS:
        st.caption(f"Maximum of {MAX_TIE_POINTS} pairs reached.")

    # ---- Automatic midpoint (operator request, "distortion in the middle")
    # With exactly the two disk-edge pairs selected, the midpoint of each
    # segment (on x10 AND on x100) can be added as a third pair with one
    # click — no picking in the middle needed.
    if len(pairs) == 2 and len(pairs) < MAX_TIE_POINTS:
        st.caption(
            "**About a third point in the middle:** the automatic midpoint "
            "(button below) becomes a REAL third pair — it is counted, "
            "shown on both previews and stored. Honest caveat: because it "
            "lies exactly on the edge-to-edge line, it adds no new "
            "information to the linear similarity/affine models (the "
            "mapping through it is identical) and it does NOT capture the "
            "non-linear mid-panorama distortion by itself. What ACTUALLY "
            "corrects the middle: pick 1–2 additional tie points OFF that "
            "line (a distinctive pore / scratch / grain visible in both "
            "panoramas) and switch the mapping model to **Piecewise** — "
            "then the mapping bends between the points and the centre of "
            "the panorama is corrected directly.")
        if st.button("Add midpoint of points 1–2 as point 3 (auto)",
                     key="xmap_add_midpoint"):
            (p1, q1), (p2, q2) = pairs[0], pairs[1]
            p3 = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            q3 = ((q1[0] + q2[0]) / 2.0, (q1[1] + q2[1]) / 2.0)
            pairs.append((p3, q3))
            auto_set = state.setdefault("auto_midpair", set())
            auto_set.add((pk, len(pairs) - 1))
            st.success("Added the automatic midpoint as point 3 (it is the "
                       "midpoint of the segment between the two edge "
                       "points on EACH panorama independently).")
            st.rerun()
    auto_mid = state.setdefault("auto_midpair", set())
    if (pk, len(pairs) - 1) in auto_mid and len(pairs) >= 3:
        st.caption("Point 3 = automatic midpoint of points 1–2 "
                   "(middle of the segment, computed on both panoramas).")

    # Mapping model: similarity with 2 points; from 3 points an AFFINE fit
    # becomes available (absorbs the anisotropic scale/shear of the tilted
    # x10 view) and — when the points are NOT collinear — a PIECEWISE model
    # that actually corrects mid-panorama distortion.
    reg_model = "similarity"
    if len(pairs) >= 3:
        options = ["Similarity (scale + rotation + shift)",
                   "Affine (adds anisotropic scale + shear — absorbs tilt "
                   "distortion of the x10 view)"]
        piecewise_available = fit_piecewise(pairs) is not None
        if piecewise_available:
            options.append(
                "Piecewise (local triangles between tie points — bends "
                "the mapping to correct mid-panorama distortion)")
        # The option list is dynamic: if a stored choice (e.g. "Piecewise…")
        # is no longer offered (points removed / became collinear), drop it
        # BEFORE the radio is instantiated, otherwise Streamlit raises.
        if st.session_state.get("xmap_reg_model") not in options:
            st.session_state.pop("xmap_reg_model", None)
        choice = st.radio(
            "Mapping model", options,
            key="xmap_reg_model", horizontal=True,
            help=("Affine needs ≥ 3 tie points spread across the panorama "
                  "(2 collinear points are not enough). Piecewise goes "
                  "further: every click is mapped with the exact affine "
                  "transform of the surrounding triangle of tie points, so "
                  "local stitching drift / glass parallax in the MIDDLE of "
                  "the panorama is corrected — pick at least one tie point "
                  "off the edge-to-edge line (a pore or grain visible in "
                  "both panoramas) for that. Outside the tie-point hull it "
                  "falls back to the global affine fit."))
        if choice.startswith("Piecewise"):
            reg_model = "piecewise"
        elif choice.startswith("Affine"):
            reg_model = "affine"
        else:
            reg_model = "similarity"
    reg = None
    if len(pairs) >= 2:
        reg = fit_similarity(pairs)
        if reg_model == "affine":
            aff = fit_affine(pairs)
            if aff is not None:
                reg = aff
            else:
                st.warning("Affine fit is degenerate (tie points nearly "
                           "collinear) — using the similarity model.")
        elif reg_model == "piecewise":
            pw = fit_piecewise(pairs)
            if pw is not None:
                reg = pw
            else:
                st.warning("Piecewise fit unavailable (need ≥ 3 NON-collinear "
                           "tie points) — using the similarity model.")
    if len(pairs) == 1:
        st.info("One pair picked — add at least one more "
                "(the opposite disk edge).")
    if reg is not None:
        m1, m2, m3 = st.columns(3)
        if reg.model == "affine":
            m1.metric("Scale x (x100 px / x10 px)", f"{reg.sx:.4f}")
            m2.metric("Scale y (x100 px / x10 px)", f"{reg.sy:.4f}")
        elif reg.model == "piecewise":
            m1.metric("Model", "piecewise (local triangles)")
            m2.metric("Tie-point triangles", f"{len(reg.tri_indices)}")
        else:
            m1.metric("Scale (x100 px / x10 px)", f"{reg.s:.4f}")
            m2.metric("Rotation", f"{reg.theta_deg:.2f}°")
        rms_um = (reg.rms_px / cal100.px_per_um
                  if cal100.ok else float("nan"))
        m3.metric("RMS residual",
                  f"{reg.rms_px:.1f} px"
                  + (f" · {rms_um:.1f} µm" if cal100.ok else ""))
        if reg.model == "piecewise":
            st.caption("Piecewise model: exact at every tie point (residual "
                       "0 there), so the RMS value does not measure the "
                       "true mid-panorama error — the uncertainty estimate "
                       "below uses the picking precision only.")
        if len(pairs) == 2:
            st.caption(
                "Why the edges are exact but the MIDDLE disagrees: with "
                "two pairs the similarity model is fitted THROUGH them, so "
                "the residual is 0 at both edges by construction — the "
                "model is pinned there, not 'verified' there. Between the "
                "edges the model can only interpolate a UNIFORM scale "
                "along a straight segment, while the real mapping carries "
                "non-uniform components the 4-DOF model cannot express: "
                "(1) the x10 overview is shot THROUGH the tilted "
                "sacrificial glass — parallax + anisotropic scale shift "
                "the projected cross-section position by a position-"
                "dependent amount; (2) the two panoramas are stitched "
                "INDEPENDENTLY, and each pairwise 0-1 px residual "
                "accumulates along the row (drift) — the x100 panorama "
                "(many more tiles) drifts more than the x10 one. An "
                "accumulated drift term grows roughly quadratically "
                "along the segment, and a linear model fitted at the two "
                "ends leaves zero error at the ends and the MAXIMUM "
                "error in the middle — exactly the observed pattern. "
                "Both lines are straight, but their lengths differ "
                "(tilt anisotropy), and the middle of the x10 line sits "
                "where the accumulated drift is largest. What actually "
                "helps: 1-2 tie points OFF the edge-to-edge line (a pore "
                "/ scratch visible in both panoramas) + the Piecewise "
                "model — the mapping then bends between the points and "
                "corrects the centre directly. The midpoint button adds "
                "a point ON the line only: it is stored and drawn, but "
                "it adds no new information to similarity/affine and "
                "does not enable Piecewise (3 collinear points are "
                "degenerate). The uncertainty below is estimated from "
                "the picking precision (section 5).")

    # ---- Calibration consistency check -----------------------------------
    st.markdown("**Calibration consistency check** — the physical distance "
                "between tie points must be the same on both panoramas:")
    cons = consistency_report(pairs, cal10, cal100)
    if not cons:
        st.caption("Needs both calibrations and ≥ 2 tie points.")
    else:
        rows = [{
            "Points": c["points"],
            "x10 (µm)": f"{c['d_x10_um']:.1f}",
            "x100 (µm)": f"{c['d_x100_um']:.1f}",
            "Deviation": f"{c['dev_pct']:.1f}%",
            "Status": "OK" if c["ok"] else "MISMATCH",
        } for c in cons]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        worst = max(c["dev_pct"] for c in cons)
        if worst > CONSISTENCY_WARN * 100:
            st.error(
                f"Max distance deviation {worst:.1f}% exceeds the "
                f"{CONSISTENCY_WARN * 100:.0f}% tolerance — re-check the "
                "scale-bar calibration of both panoramas (section 2).")
        else:
            st.success(f"All distances consistent (max deviation "
                       f"{worst:.1f}%).")

    # ---- 4 · Live preview -------------------------------------------------
    st.subheader("4 · Live preview — click the x10 panorama")
    glass_choice = st.radio(
        "Which side of the x100 panorama is the GLASS? (the other side is "
        "the IC) — drawn as plates on the TOP / BOTTOM edges of the x100 "
        "fragment below and in the HTML report",
        ["Top — GLASS · Bottom — IC",
         "Top — IC · Bottom — GLASS"],
        index=0, key="xmap_glass_side_tb", horizontal=True)
    glass_top = glass_choice.startswith("Top — GLASS")
    # Visual aid (operator request: "provide a fragment of the x100
    # panorama from its MIDDLE, so choosing glass vs IC is easier"): the
    # coloured plates show the CURRENT choice and flip instantly with the
    # radio. The fragment spans the full specimen thickness (top and
    # bottom edges of the panorama) plus a true-µm scale bar.
    try:
        _cf = _center_fragment_preview(
            item100["image"], glass_top=glass_top,
            px_per_um=(cal100.px_per_um if cal100.ok else None))
        st.image(_cf, use_container_width=True)
        st.caption(
            "A fragment from the MIDDLE of the x100 panorama — the full "
            "thickness is visible, so the glass side (even, glassy "
            "appearance) and the IC side (the device structure) can be "
            "told apart directly. The coloured plates are your CURRENT "
            "choice above — switch the radio and they flip immediately. "
            "The same plates are drawn on every x100 fragment in the live "
            "preview and in the HTML report.")
    except Exception as e:
        st.caption(f"(Center-fragment preview unavailable: {e})")
    fov_um = st.number_input("Field of view around the point (µm)",
                             min_value=1.0, value=DEFAULT_FOV_UM,
                             step=10.0, key="xmap_fov")
    k = st.number_input("Safety factor k (crop margin = k·σ)",
                        min_value=0.0, value=DEFAULT_SAFETY_K, step=0.5,
                        key="xmap_k")
    pick_prec = st.number_input(
        "Tie-point picking precision on x10 (px)", min_value=0.5,
        value=DEFAULT_PICK_PRECISION_X10_PX, step=0.5, key="xmap_pick_prec")
    sigma_um = estimate_sigma_um(reg, cal10, cal100, float(pick_prec))
    if sigma_um:
        st.caption(f"Estimated registration uncertainty: "
                   f"**σ ≈ {sigma_um:.1f} µm** → crop margin k·σ ≈ "
                   f"{k * sigma_um:.1f} µm.")
    if not (cal100.ok and reg is not None):
        st.info("Live preview needs: both calibrations (section 2) and "
                "≥ 2 tie points (section 3).")
        return
    unc_full_live = (k * sigma_um * cal10.px_per_um) \
        if (sigma_um and cal10.ok) else None
    marked_live, s_live = _marked_preview(
        item10["image"], PREVIEW_SIDE_MAIN, [p for p, _ in pairs],
        state.get("live_pt"), unc_full_live)
    click = _click_preview_scaled(marked_live, "xmap_ck_live", s_live)
    if click is not None:
        state["live_pt"] = click
        # The clickable preview above was BUILT from the PREVIOUS point
        # (it existed before this click happened) — rerun immediately so
        # the crosshair on the x10 panorama marks the CURRENT point, not
        # the previous one.
        st.rerun()
    pt = state.get("live_pt")
    if pt is None:
        st.caption("Click the preview above to map a point.")
        return
    qx, qy = reg.map_point(pt[0], pt[1])
    h100, w100 = item100["image"].shape[:2]
    half_full = (fov_um / 2.0 + (k * sigma_um if sigma_um else 0.0)) \
        * cal100.px_per_um
    crop, pt_in, clamped = render_crop(item100["image"], qx, qy, half_full)
    annotated = annotate_crop(crop, pt_in, cal100.px_per_um, fov_um, k,
                              sigma_um)
    # Side plates (GLASS / IC) on the fragment — the operator's choice
    # from the radio above travels into the HTML report too.
    annotated = _draw_side_labels(annotated, glass_top=glass_top)
    # --- Two-click µm measurement on the x100 fragment (operator request:
    # click two pixels and read the distance between them in µm, using the
    # x100 calibration). The clicks are stored in FULL-RES x100 coordinates
    # (so the measurement survives moving the view) and drawn as cyan
    # markers A/B + line + label — the SAME style the HTML report uses.
    _meas_entry = state.setdefault("live_measure", {})
    if _meas_entry.get("pan") != item100["name"]:
        state["live_measure"] = {"pan": item100["name"], "pts": []}
        _meas_entry = state["live_measure"]
    meas = _meas_entry["pts"]
    _x0c, _y0c, _wxc, _wyc, _clc = _crop_window(
        *item100["image"].shape[:2], qx, qy, half_full)
    annotated, meas_dist = _draw_measure_overlay(
        annotated,
        [(p[0] - _x0c, p[1] - _y0c) for p in meas],
        cal100.px_per_um)
    # Only the x100 fragment is shown here — the x10 panorama with the
    # crosshair is ALREADY displayed above (the clickable preview); the
    # old second copy next to the fragment was a duplicate and is removed.
    # The fragment is CLICKABLE (two-click µm measurement).
    meas_click = _click_preview_scaled(annotated, "xmap_ck_meas", 1.0)
    if meas_click is not None:
        if len(meas) >= 2:
            meas.clear()                 # third click starts a new pair
        meas.append((float(meas_click[0] + _x0c),
                     float(meas_click[1] + _y0c)))
        st.rerun()
    mc1, mc2, _mc3 = st.columns([1, 1, 3])
    if mc1.button("Clear measurement", key="xmap_meas_clear",
                  help="Removes the A/B markers and the distance readout."):
        meas.clear()
        st.rerun()
    if meas_dist is not None:
        px_len = math.hypot(meas[1][0] - meas[0][0],
                            meas[1][1] - meas[0][1])
        offcrop = sum(1 for p in meas
                      if not (_x0c <= p[0] < _x0c + _wxc
                              and _y0c <= p[1] < _y0c + _wyc))
        st.success(
            f"Distance A → B: **{meas_dist:.2f} µm** "
            f"({px_len:.1f} px on the x100 panorama). Click again to "
            "start a new measurement — the previous one is replaced.")
        if offcrop:
            st.caption(f"{offcrop} of the 2 measured points lie OUTSIDE "
                       "the currently shown fragment (the view moved) — "
                       "the distance above stays valid.")
    elif len(meas) == 1:
        st.caption("Point A placed — click the fragment once more to "
                   "measure the distance to point B (µm).")
    else:
        st.caption("Click TWICE on the fragment to measure the distance "
                   "between the two pixels in µm (uses the x100 "
                   "calibration; the third click starts a new "
                   "measurement).")
    st.caption(f"Clicked x10: ({pt[0]:.0f}, {pt[1]:.0f}) px → "
               f"mapped x100: ({qx:.0f}, {qy:.0f}) px")
    tie_dist_um = _nearest_tie_dist_um(pt, pairs, cal10)
    if tie_dist_um is not None and tie_dist_um > 1.0:
        st.caption(f"Nearest tie point: ≈{tie_dist_um:.0f} µm away (on x10). "
                   "The mapping here is interpolated/extrapolated between "
                   "tie points — the true local error can be much larger "
                   "than σ (glass parallax, stitching drift of the two "
                   "independently stitched panoramas). Click near a tie "
                   "point or add more tie points for an accurate map.")
    outside = (qx < 0 or qy < 0 or qx > w100 or qy > h100)
    if outside:
        dx = max(0.0, -qx, qx - w100)
        dy = max(0.0, -qy, qy - h100)
        dist_um = math.hypot(dx, dy) / cal100.px_per_um
        st.warning(f"The mapped point lies OUTSIDE the x100 panorama — "
                   f"the crop shows the nearest available area "
                   f"(≈{dist_um:.0f} µm away).")
    elif clamped:
        st.caption("The crop window was clamped to the x100 panorama "
                   "borders.")

    # ---- 5 · Report + downloads ------------------------------------------
    st.subheader("5 · HTML report + downloads")
    ec1, ec2, ec3 = st.columns(3)
    embed_side = ec1.number_input(
        "Max embedded x10 side (px)", min_value=800,
        value=DEFAULT_EMBED_MAX_SIDE, step=200, key="xmap_embed",
        help=("Resolution of the interactive x10 panorama inside the "
              "report (it is displayed across the full page width)."))
    jpeg_q = ec2.slider("Embedded JPEG quality", 50, 100,
                        DEFAULT_JPEG_QUALITY, key="xmap_q")
    embed_side_x100 = ec3.number_input(
        "Max embedded x100 side (px, 0 = full)", min_value=0, value=0,
        step=400, key="xmap_embed100",
        help=("Resolution of the interactive x100 panorama inside the "
              "report — the fragment that appears on a click is cut "
              "from THIS copy and drawn WITHOUT upscaling, so 0 (full "
              "resolution) gives the best fragment quality."))
    st.caption(
        "0 = full resolution, capped at **65500 px per side — the hard "
        "limit of the JPEG format**. A wider x100 panorama (e.g. "
        "~99 000 px) is embedded at 65500 px automatically — this is "
        "what previously failed as 'Report generation failed: JPEG "
        "encoding failed'. Lower the value for a smaller report file "
        "(the fragments get slightly softer).")
    dl_full = st.checkbox(
        "Embed full-resolution copies for the two download buttons "
        "(larger HTML file)", value=True, key="xmap_dl_full",
        help=("ON: the download buttons inside the report give the "
              "FULL-resolution panoramas (JPEG, quality 95) instead of "
              "the report-resolution copies — much better quality, "
              "bigger file."))
    d1, d2, d3 = st.columns(3)
    if d1.button("Generate HTML report", key="xmap_gen_report"):
        missing = []
        if not cal10.ok:
            missing.append("x10 calibration")
        if not cal100.ok:
            missing.append("x100 calibration")
        if reg is None:
            missing.append("≥ 2 tie points")
        if missing:
            st.error("Cannot generate the report: missing " +
                     ", ".join(missing) + ".")
        else:
            with st.spinner("Building report…"):
                try:
                    html_str = generate_html_report(
                        item10["image"], item100["image"],
                        cal10, cal100, pairs, reg,
                        float(fov_um), float(k), sigma_um,
                        float(embed_side), int(jpeg_q),
                        embed_max_side_x100=float(embed_side_x100),
                        dl_full_resolution=bool(dl_full),
                        glass_side=("top" if glass_top else "bottom"))
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
                else:
                    # Persist the report in session_state: the download
                    # button below then survives EVERY rerun (the old
                    # code rendered it only inside the run that pressed
                    # Generate, so the first download click removed it).
                    state["xmap_html_report"] = {
                        "x10": item10["name"],
                        "x100": item100["name"],
                        "html": html_str,
                    }
                    st.success("Report ready — download it below. It is "
                               "fully standalone (offline interactive).")
    html_entry = state.get("xmap_html_report")
    if html_entry and html_entry.get("html") \
            and html_entry.get("x10") == item10["name"] \
            and html_entry.get("x100") == item100["name"]:
        st.download_button(
            "Download HTML report",
            data=html_entry["html"].encode("utf-8"),
            file_name="cross_section_map_report.html",
            mime="text/html",
            key="xmap_dl_html")
        if st.button("Discard generated report", key="xmap_drop_html"):
            state.pop("xmap_html_report", None)
            st.rerun()
    with st.expander("Full-resolution downloads (PNG)"):
        st.caption("Encoding large panoramas takes a moment — prepare "
                   "them on demand. PNG has NO size limit (unlike JPEG, "
                   "which stops at 65500 px per side), so a ~100 000 px "
                   "wide x100 panorama downloads here at its full "
                   "resolution; encoding may take a while and needs "
                   "some free RAM.")
        if st.button("Prepare PNG downloads",
                     key="xmap_prep_png"):
            with st.spinner("Encoding PNG…"):
                buf10b = _encode_png_rgb(item10["image"])
                buf100b = _encode_png_rgb(item100["image"])
                st.session_state["xmap_png10"] = (
                    item10["name"], buf10b)
                st.session_state["xmap_png100"] = (
                    item100["name"], buf100b)
                if buf100b is not None:
                    st.caption(f"x100 PNG ready: "
                               f"{len(buf100b) / 1e6:.1f} MB.")
        png10 = st.session_state.get("xmap_png10")
        png100 = st.session_state.get("xmap_png100")
        if png10 and png10[0] == item10["name"] and not png10[1]:
            st.warning("x10 PNG encoding failed — press 'Prepare PNG "
                       "downloads' again.")
        if png100 and png100[0] == item100["name"] and not png100[1]:
            st.error(
                "x100 PNG encoding failed — most likely the machine ran "
                "out of free memory while compressing the very wide "
                "image. Close other programs / tabs and try again; as a "
                "fallback, the HTML report's 'Download x100 panorama' "
                "link gives a JPEG at the maximum encodable size "
                "(65500 px per side).")
        d1, d2 = st.columns(2)
        if png10 and png10[0] == item10["name"] and png10[1]:
            d1.download_button("Download x10 panorama (PNG)",
                               data=png10[1],
                               file_name="panorama_x10.png",
                               mime="image/png", key="xmap_dl_10")
        if png100 and png100[0] == item100["name"] and png100[1]:
            d2.download_button("Download x100 panorama (PNG)",
                               data=png100[1],
                               file_name="panorama_x100.png",
                               mime="image/png", key="xmap_dl_100")


def _pick_entry(names: List[str], key: str) -> Optional[int]:
    """Selectbox over the saved panoramas -> index or None."""
    if not names:
        return None
    idx = st.selectbox("Saved panorama", range(len(names)),
                       format_func=lambda i: names[i], key=key,
                       index=0)
    return int(idx)


def _remember_selection(store_sig: tuple, sel10: Optional[int],
                        sel100: Optional[int]) -> None:
    """Mirror the CURRENT panorama selection into plain session state.

    Streamlit deletes the state of widgets that are not rendered on two
    consecutive runs (the two selectboxes exist only while the
    Cross-section Map tab is open). The plain ``xmap_state`` dict is
    never cleaned, so it is used to bring the selection back when the
    operator returns to the tab — see ``_auto_assign_sources``.
    """
    state = _state()
    state["xmap_last_sel"] = {
        "sig": tuple(store_sig),
        "x10": (int(sel10) if sel10 is not None else None),
        "x100": (int(sel100) if sel100 is not None else None),
    }


def _auto_assign_sources(store: List[dict]) -> tuple:
    """Pre-select the Overview / Detail panorama by RESOLUTION.

    Every time the panorama store CHANGES (a new save, an upload, a
    clear, pruning beyond STORE_LIMIT) the two selectboxes are re-assigned
    automatically: the SMALLEST-resolution entry becomes the Overview
    (x10 — the whole disk, lower magnification) and the LARGEST-resolution
    entry becomes the Detail (x100 — the cross-section top view). While
    (the store is unchanged, so the manual choice is kept). Returns the
    store signature (an empty tuple when the store is empty) so the
    caller can persist the operator's selection against the same key.
    """
    if not store:
        state = _state()
        state["xmap_store_sig"] = ()
        return ()
    sig = tuple(
        (str(e.get("name", "")), int(e.get("image").shape[1]),
         int(e.get("image").shape[0]))
        for e in store
    )
    state = _state()
    if state.get("xmap_store_sig") == sig:
        # Store unchanged -> keep the operator's choice. BUT Streamlit has
        # silently DELETED the two selectbox states while this tab was
        # hidden (widget missing from two consecutive runs) — without a
        # restore the boxes fell back to index 0, i.e. the NEWEST entry,
        # which put the x100 detail panorama into the Overview box
        # (reported as "returning to Cross-section Map selects 25672×678
        # instead of 6824×567"). Restore the last selection from the
        # plain-state mirror; fall back to the resolution auto-assign.
        missing = ("xmap_sel_x10" not in st.session_state
                   or "xmap_sel_x100" not in st.session_state)
        if not missing:
            return sig
        last = state.get("xmap_last_sel")
        if (isinstance(last, dict) and last.get("sig") == sig
                and isinstance(last.get("x10"), int)
                and isinstance(last.get("x100"), int)
                and 0 <= last["x10"] < len(store)
                and 0 <= last["x100"] < len(store)):
            st.session_state["xmap_sel_x10"] = int(last["x10"])
            st.session_state["xmap_sel_x100"] = int(last["x100"])
            return sig
        # No valid mirror (first visit / state wiped) -> auto-assign below.
    state["xmap_store_sig"] = sig
    areas = [max(1, int(e["image"].shape[0]) * int(e["image"].shape[1]))
             for e in store]
    i_small = min(range(len(store)), key=lambda i: areas[i])
    i_large = max(range(len(store)), key=lambda i: areas[i])
    # Assign BEFORE the selectboxes are instantiated in this run — legal
    # in Streamlit and picked up by the widgets as their current value.
    st.session_state["xmap_sel_x10"] = i_small
    st.session_state["xmap_sel_x100"] = i_large
    return sig


def _nearest_tie_dist_um(pt: Tuple[float, float],
                         pairs,
                         cal10: Calib) -> Optional[float]:
    """Distance (µm, on the x10 panorama) from a picked point to the
    nearest tie point — an honest indicator of how far the mapping is
    extrapolating (glass parallax + stitching drift grow with it)."""
    if not cal10.ok or not pairs or pt is None:
        return None
    try:
        d = min(math.hypot(pt[0] - p[0], pt[1] - p[1]) for p, _ in pairs)
        return d / cal10.px_per_um
    except Exception:
        return None


def _upload_to_store(label: str, key: str) -> None:
    """Load a panorama image file into the store (no first tile)."""
    up = st.file_uploader(label, type=["png", "jpg", "jpeg", "tif", "tiff"],
                          key=key)
    if up is None:
        return
    data = np.frombuffer(up.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error(f"Cannot decode {up.name}")
        return
    if st.button(f"Add {up.name} to store", key=key + "_add"):
        name = save_panorama(img_bgr, None, f"uploaded: {up.name}")
        st.success(f"Added: {name}")
        st.rerun()


# ---------------------------------------------------------------------------
# Operator tools shown with EVERY stitched result (Stitcher AND Pair
# Review): (1) calibration & stitch check by measuring a known distance
# on the panorama, (2) manual scan / repair of residual broken pixels.
# ---------------------------------------------------------------------------

def render_disk_distance_check(snap: dict, snap_key: str) -> None:
    """'Calibration & stitch check' — click two points on the STITCHED
    panorama (e.g. the opposite edges of the TEM disk) and compare the
    measured distance with the expected physical length.

    The check catches two failure classes at once:
    * a WRONG CALIBRATION (µm/px) — even a perfect stitch then reports a
      wrong physical distance;
    * a WRONG STITCH (a pair placed at an offset that stretches or
      shrinks part of the panorama) — the physical distance between two
      real features deviates even with a correct calibration.

    Works on the panorama of ANY stitching workflow because it is
    rendered from the persisted result snapshot by ``render_stitch_result``
    (i.e. after the automatic Stitcher run AND after the Pair Review
    stitch). All clicks are deduplicated by ``_click_preview_scaled``;
    a third click starts a new measurement.
    """
    key = f"diskchk_{snap_key}"
    pan_bgr = snap.get("panorama_bgr")
    if pan_bgr is None or pan_bgr.ndim != 3:
        return
    pan_rgb = cv2.cvtColor(pan_bgr, cv2.COLOR_BGR2RGB)
    calib = get_calibration_for_upload(snap.get("upload_sig"))
    st.subheader("Calibration & stitch check — measure a known distance")
    st.caption(
        "Click the two opposite EDGES of the TEM disk on the panorama "
        "below (or any other feature of KNOWN physical length — the "
        "cross-section length, a structure the operator measured "
        "before). The app measures the distance between the two clicks "
        "in pixels and, when the panorama is calibrated, in µm, and "
        "compares it with the expected value. The expected distance "
        "defaults to **2200 µm** (the disk diameter used in this "
        "workflow); standard TEM disks are 3.05 mm in diameter — set "
        "the value to YOUR actual disk. A large deviation means the "
        "stitching and/or the calibration is wrong — re-check the pair "
        "positions (Pair Review) and the scale bar (calibration block "
        "at the top of the tab).")
    # Clicks live INSIDE the result snapshot, NOT in a fixed session key:
    # a new stitch / another uploaded series replaces the snapshot, so
    # the clicks of the previous series are gone automatically (the
    # operator no longer has to clear them manually after switching to
    # another series).
    clicks = snap.setdefault("diskchk_clicks", [])
    exp_um = st.number_input(
        "Expected distance (µm)", min_value=1.0, value=2200.0, step=50.0,
        key=f"{key}_exp",
        help="The known physical length you are measuring. Default: "
             "2200 µm — the disk diameter used in this workflow; set it "
             "to the actual disk diameter (standard TEM disks are "
             "3.05 mm = 3050 µm).")
    tol_pct = st.number_input(
        "Allowed deviation (%)", min_value=0.5, value=10.0, step=1.0,
        key=f"{key}_tol",
        help="Measured within this band = OK. Up to 2x the band = suspect "
             "(check the stitch). Beyond that = wrong calibration and/or "
             "stitch. The disk diameter itself is not always exactly "
             "3050 µm (3.00 / 3.05 mm disks, chamfered edges) — set the "
             "band accordingly.")

    prev, s = downscale(pan_rgb, PREVIEW_SIDE_MAIN)
    annotated = prev.copy()
    line_um = None
    if len(clicks) >= 1:
        pa = (int(clicks[0][0] * s), int(clicks[0][1] * s))
        cv2.circle(annotated, pa, 6, (255, 40, 40), -1)
        cv2.putText(annotated, "1", (pa[0] + 8, max(12, pa[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 40, 40), 2,
                    cv2.LINE_AA)
    if len(clicks) >= 2:
        pb = (int(clicks[1][0] * s), int(clicks[1][1] * s))
        cv2.line(annotated, pa, pb, (255, 40, 40), 2)
        cv2.circle(annotated, pb, 6, (40, 200, 80), -1)
        cv2.putText(annotated, "2", (pb[0] + 8, max(12, pb[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 200, 80), 2,
                    cv2.LINE_AA)
        dist_px = math.hypot(clicks[1][0] - clicks[0][0],
                             clicks[1][1] - clicks[0][1])
        if calib.ok:
            line_um = dist_px / calib.px_per_um
        lbl = (f"{dist_px:.0f} px"
               + (f" = {line_um:.0f} µm" if line_um is not None else ""))
        cv2.putText(annotated, lbl,
                    ((pa[0] + pb[0]) // 2, max(14, (pa[1] + pb[1]) // 2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 40), 2,
                    cv2.LINE_AA)
    click = _click_preview_scaled(annotated, f"{key}_ck", s)
    if click is not None:
        if len(clicks) >= 2:
            clicks.clear()                     # third click -> new measure
        clicks.append((float(click[0]), float(click[1])))
        st.rerun()

    if len(clicks) < 2:
        st.caption("Click 1 = first edge, click 2 = the opposite edge "
                   "(a 3rd click starts a new measurement). The clicks "
                   "belong to THIS panorama — stitching a new panorama or "
                   "switching to another uploaded series starts with an "
                   "empty measurement.")
        return

    def _clear_clicks() -> None:
        snap["diskchk_clicks"] = []

    dist_px = math.hypot(clicks[1][0] - clicks[0][0],
                         clicks[1][1] - clicks[0][1])
    if not calib.ok:
        st.info(f"Measured distance: **{dist_px:.0f} px**. This panorama "
                "has NO calibration yet — set it at the top of the tab "
                "(calibration block) to convert the distance to µm and "
                "run the check.")
    else:
        dist_um = dist_px / calib.px_per_um
        dev_pct = abs(dist_um - exp_um) / max(exp_um, 1e-9) * 100.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Measured", f"{dist_um:.0f} µm",
                  f"{dist_px:.0f} px · {(1.0 / calib.px_per_um):.4g} µm/px")
        m2.metric("Expected", f"{exp_um:g} µm")
        m3.metric("Deviation", f"{dev_pct:.1f} %",
                  ("OK" if dev_pct <= tol_pct else
                   "check!" if dev_pct <= 2 * tol_pct else "WRONG"),
                  delta_color=("normal" if dev_pct <= tol_pct else "inverse"))
        verdict_color = ("normal" if dev_pct <= tol_pct else
                         "inverse" if dev_pct > 2 * tol_pct else "off")
        if dev_pct <= tol_pct:
            m4.metric("Verdict", "OK", "calibration + stitch consistent")
        elif dev_pct <= 2 * tol_pct:
            m4.metric("Verdict", "SUSPECT", "re-check stitch/calibration")
            st.warning(
                f"The measured {dist_um:.0f} µm deviates {dev_pct:.1f}% "
                f"from the expected {exp_um:g} µm — beyond the "
                f"{tol_pct:g}% band. Verify the scale-bar calibration "
                "first (a 2x-wrong bar value is the most common cause), "
                "then the pair positions (Pair Review diagnostics).")
        else:
            m4.metric("Verdict", "WRONG", "fix calibration / re-stitch")
            st.error(
                f"The measured {dist_um:.0f} µm deviates {dev_pct:.1f}% "
                f"from the expected {exp_um:g} µm — far beyond the "
                f"{tol_pct:g}% band. Either the calibration is wrong "
                "(misread scale-bar value, default-DPI metadata) or a "
                "pair was stitched at a wrong offset (stretched part of "
                "the panorama). Compare with the OTHER panorama: measure "
                "the SAME distance in the Cross-section Map tab.")
    rec1, rec2, rec3 = st.columns([1, 1, 2])
    if rec1.button("Record this measurement", key=f"{key}_rec",
                   help="Append the current measurement to the table "
                        "below (survives until a new panorama is "
                        "stitched)."):
        recs = snap.setdefault("disk_checks", [])
        recs.append({
            "from": f"({clicks[0][0]:.0f}, {clicks[0][1]:.0f})",
            "to": f"({clicks[1][0]:.0f}, {clicks[1][1]:.0f})",
            "px": round(dist_px, 1),
            "µm": (round(dist_px / calib.px_per_um, 1) if calib.ok else None),
            "expected µm": exp_um,
            "dev %": (round(abs(dist_px / calib.px_per_um - exp_um)
                            / exp_um * 100.0, 1) if calib.ok else None),
        })
        clicks.clear()
        st.rerun()
    if rec2.button("Clear clicks", key=f"{key}_clr", on_click=_clear_clicks):
        st.rerun()
    recs = snap.get("disk_checks") or []
    if recs:
        if rec3.button("Clear recorded list", key=f"{key}_clrl"):
            snap.pop("disk_checks", None)
            st.rerun()
        st.dataframe(recs, use_container_width=True, hide_index=True)


def render_broken_pixel_repair(snap: dict, snap_key: str) -> None:
    """Manual scan / repair of residual 'broken and grey pixels'.

    TWO detectors run on the FINISHED panorama (no re-stitch needed):

    1. **3×3-median outlier scan** — every pixel deviating from its
       3×3 median by more than the threshold is a speck; real texture
       (grain boundaries, dust) is never an isolated 1-px spike and
       survives. Pixels adjacent to pure-black areas (no-overlap gap
       borders, canvas corners) are excluded — crisp black borders
       must not be softened.
    2. **Smooth-patch scan (window 9×9 … 31×31, slider)** — the fix of
       the "the scan finds nothing although grey patches are visible"
       report: a grey SEAM BAND or patch a few pixels wide is locally
       FLAT, so its 3×3 median equals itself and detector 1 is blind
       to it. Every locally-smooth pixel (small 3×3 deviation) is
       compared with the median of the WIDE window instead; a large
       difference there flags the band while real texture (large 3×3
       deviation) stays protected. A band is detected while it is
       narrower than roughly half of the window width (the median of
       the window must be dominated by the surrounding content).

    Repair replaces flagged pixels by the corresponding window median.
    """
    key = f"bpr_{snap_key}"
    pan = snap.get("panorama_bgr")
    if pan is None or pan.ndim != 3:
        return
    st.subheader("Broken pixel repair")
    st.caption(
        "The blend pass already repairs seam slivers and isolated broken "
        "pixels automatically. If the downloaded panorama still shows "
        "grey / broken pixels, re-scan here and repair them in place — "
        "the preview and ALL downloads (including the 'Save to "
        "Cross-section Map' copy) then use the repaired panorama. "
        "'Restore original' brings the untouched stitch back. TWO "
        "detectors run at once: (a) the 3×3-median outlier scan for "
        "salt/pepper specks, and (b) the SMOOTH-PATCH scan — grey "
        "seam bands / blocks a few pixels wide are locally flat, so a "
        "3×3 median equals them and sees nothing; comparing every "
        "locally-smooth pixel with the median of a WIDER window "
        "catches them (the earlier 'the scan finds nothing although "
        "grey patches are visible' report).")
    thr = st.slider(
        "Outlier threshold vs median (grey levels)", 10, 100, 40,
        key=f"{key}_thr",
        help=("Lower = more pixels counted/repaired (softer texture); "
              "higher = only extreme specks. 40 catches the typical "
              "grey specks; 60 was the automatic pass."))
    pw = st.slider(
        "Smooth-patch scan width (odd px, 0 = off)", 0, 31, 15, step=2,
        key=f"{key}_pw",
        help=("Window size of the second detector for grey SEAM BANDS "
              "and patches wider than 3 px. A band is found while it is "
              "narrower than about HALF of this window, so set it to "
              "roughly 2x the width of the grey bands you see (9 = up "
              "to ~4 px bands, 15 = up to ~7 px, 21 = up to ~10 px; "
              "larger windows are slower). Only locally SMOOTH pixels "
              "are flagged (their 3×3 deviation is small), so real "
              "texture — grain boundaries, dust, sharp edges — is "
              "never counted or repaired. 0 disables this scan."))
    orig = snap.get("panorama_bgr_orig")
    c1, c2, c3 = st.columns(3)
    do_scan = c1.button("Scan for broken pixels", key=f"{key}_scan")
    do_repair = c2.button("Scan + repair", key=f"{key}_fix",
                          help="Replace every detected outlier pixel by "
                               "its median (3×3 median for specks, wide-"
                               "window median for smooth seam bands; the "
                               "original panorama is kept once, so the "
                               "repair is reversible).")
    do_restore = c3.button("Restore original panorama", key=f"{key}_undo",
                           disabled=orig is None,
                           help="Undo all repairs of this session (the "
                                "pristine stitch is kept in the result "
                                "snapshot).")
    if do_scan or do_repair:
        black_mask = (pan.max(axis=2) <= 8)
        # Protect crisp black borders (gap separators, canvas corners):
        # pixels ADJACENT to pure black are excluded from the SPECK scan
        # only — grey bands hugging a black gap border are exactly what
        # the smooth-patch scan must still see.
        black_dil = cv2.dilate(black_mask.astype(np.uint8),
                               np.ones((3, 3), np.uint8)).astype(bool)
        med3 = cv2.medianBlur(pan, 3)
        diff3 = np.abs(pan.astype(np.int16) - med3.astype(np.int16)
                       ).max(axis=2)
        bad3 = (diff3 > int(thr)) & (~black_mask) & (~black_dil)
        n3 = int(bad3.sum())
        bad_band = None
        med_pw = None
        n_band = 0
        pw_int = int(pw)
        if pw_int >= 3:
            k = pw_int if pw_int % 2 == 1 else pw_int + 1
            med_pw = cv2.medianBlur(pan, k)
            diff_pw = np.abs(pan.astype(np.int16)
                             - med_pw.astype(np.int16)).max(axis=2)
            # Only locally smooth pixels (3x3 deviation small) can be
            # band pixels — real texture never qualifies.
            smooth = diff3 <= max(6, int(thr) // 2)
            bad_band = ((diff_pw > int(thr)) & smooth & (~black_mask)
                        & (~black_dil))
            n_band = int(bad_band.sum())
        bad = bad3 if bad_band is None else (bad3 | bad_band)
        n_bad = int(bad.sum())
        if n_bad == 0:
            st.success("No outlier pixels found at this threshold — the "
                       "panorama is clean.")
            return
        msg = f"{n_bad} suspicious pixel(s) found: {n3} speck(s)"
        if bad_band is not None:
            msg += f" + {n_band} smooth seam-band pixel(s)"
        st.warning(msg + ".")
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            bad.astype(np.uint8), connectivity=8)
        clusters = sorted(
            (int(s[4]), int(s[0]), int(s[1]), int(s[2]), int(s[3]))
            for s in stats[1:])[-8:][::-1]
        if clusters:
            rows = [{"area (px)": a, "x": x, "y": y, "w": w, "h": h}
                    for (a, x, y, w, h) in clusters]
            st.caption("Largest affected areas (full-res panorama px):")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        if do_repair:
            if "panorama_bgr_orig" not in snap:
                snap["panorama_bgr_orig"] = pan.copy()
            if n3:
                pan[bad3] = med3[bad3]
            if bad_band is not None and n_band and med_pw is not None:
                k = pw_int if pw_int % 2 == 1 else pw_int + 1
                pan[bad_band] = med_pw[bad_band]
            snap.pop("enc_cache", None)          # re-encode downloads
            done = f"Repaired {n3} speck pixel(s) by their 3×3 median"
            if bad_band is not None and n_band:
                done += (f" and {n_band} seam-band pixel(s) by the "
                         f"{k}×{k} window median")
            st.success(done + ". The preview and downloads are "
                       "refreshed — re-download the panorama.")
            st.rerun()
    elif do_restore and orig is not None:
        snap["panorama_bgr"] = orig.copy()
        snap.pop("panorama_bgr_orig", None)
        snap.pop("enc_cache", None)
        st.rerun()
