"""
pair_review.py — Interactive pair review and manual alignment support.

Implements the "Pair Review" workflow, in which the user visually verifies
how every adjacent pair of tiles is positioned relative to each other,
approves correct pairs, manually nudges incorrect ones in X and Y, and then
stitches the panorama using exactly these positions.

Additionally supports "No overlap" pairs: when the operator accidentally
captured two neighbouring shots WITHOUT any overlap, the user can mark such
a pair and it is placed side by side with a visible black gap on the final
panorama instead of a (meaningless) forced overlap.

Functions:
- downscale_tiles(): reduced-resolution copies for fast interactive review
- estimate_pairs(): fast automatic estimate of all adjacent-pair shifts
    (full stitching.stitch_pair logic on downscaled images; translations
    are reported in FULL-RESOLUTION pixels)
- render_pair_preview(): alignment overlay renderings
    ("blend" 50/50, "checker" board, "difference" map) with tile outlines
- no_overlap_position() / apply_no_overlap_flag(): "No overlap" pair logic
    (place tiles side by side with a black gap)
- to_json() / apply_json(): save / restore a review session (positions,
    approvals, no-overlap marks) so manual work survives a browser refresh
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import stitching

DEFAULT_REVIEW_MAX_SIDE = 1200
DEFAULT_GAP_PX = 60          # black gap width for "No overlap" pairs, px

_PREVIEW_MODES = ("blend", "checker", "difference")


# --------------------------------------------------------------------------
# Downscaling
# --------------------------------------------------------------------------

def downscale_tiles(
    images: List[np.ndarray],
    max_side: int = DEFAULT_REVIEW_MAX_SIDE,
) -> Tuple[List[np.ndarray], float]:
    """Return downscaled copies of the tiles and the scale factor used.

    The scale factor maps review-scale pixels to full-resolution pixels:
        full_px = review_px / scale
    """
    if not images:
        return [], 1.0
    h, w = images[0].shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    imgs: List[np.ndarray] = []
    for img in images:
        if scale < 1.0:
            imgs.append(cv2.resize(img, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA))
        else:
            imgs.append(img.copy())
    return imgs, scale


# --------------------------------------------------------------------------
# Fast pairwise estimation at review resolution
# --------------------------------------------------------------------------

def estimate_pairs(
    imgs: List[np.ndarray],
    scale: float,
    n_features: int = 5000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
    min_inliers_sift: int = 8,
    enable_phase_fallback: bool = True,
    enable_ncc_fallback: bool = True,
    verify_with_ncc: bool = False,
    ncc_verify_gap: float = 0.07,
    suppress_fixed_pattern: bool = True,
    expected_dx_full: Optional[float] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> List[Dict]:
    """Estimate the translation of every adjacent pair on downscaled images.

    The full stitch_pair pipeline (SIFT + optional NCC verification +
    phase/NCC fallbacks) runs on the review-resolution images, which is
    several times faster than full resolution. All returned translations
    are converted back to FULL-RESOLUTION pixels.

    verify_with_ncc defaults to False here: the review workflow relies on
    the user's visual judgement instead of an automatic cross-check, so the
    per-pair NCC verification scan is skipped unless explicitly requested.

    Returns a list of dicts, one per pair i (tiles[i] -> tiles[i+1]):
        {pair_index, dx, dy, dx0, dy0, method, success, score, inliers,
         message}
    with dx/dy in full-resolution pixels. dx0/dy0 keep the ORIGINAL
    automatic estimate (dx/dy may later be changed by the user; the pair
    review UI uses dx0/dy0 for its reset action and manual-edit flags).
    """
    records: List[Dict] = []
    n_pairs = len(imgs) - 1
    expected_dx_s = (expected_dx_full * scale
                     if expected_dx_full is not None else None)
    dx_history_s: List[float] = []   # solved dx values, review-scale

    for i in range(n_pairs):
        if progress_cb is not None:
            progress_cb(f"Estimating pair {i + 1}/{n_pairs}...",
                        i / max(1, n_pairs))
        row_prior_s = (float(np.median(dx_history_s))
                       if len(dx_history_s) >= 2 else None)
        res = stitching.stitch_pair(
            imgs[i], imgs[i + 1],
            n_features=n_features,
            ratio_thresh=ratio_thresh,
            ransac_thresh=ransac_thresh,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            min_inliers_sift=min_inliers_sift,
            enable_phase_fallback=enable_phase_fallback,
            enable_ncc_fallback=enable_ncc_fallback,
            verify_with_ncc=verify_with_ncc,
            ncc_verify_gap=ncc_verify_gap,
            suppress_fixed_pattern=suppress_fixed_pattern,
            row_prior_dx=row_prior_s,
            expected_dx=expected_dx_s,
        )
        if res.success:
            dx_history_s.append(res.translation[0])
        dx_full = res.translation[0] / scale   # back to full-resolution px
        dy_full = res.translation[1] / scale
        records.append({
            "pair_index": i,
            "dx": dx_full,
            "dy": dy_full,
            "dx0": dx_full,   # original automatic estimate (never modified)
            "dy0": dy_full,
            "method": res.method,
            "success": res.success,
            "score": res.score,
            "inliers": res.inliers,
            "message": res.message,
        })
    return records


# --------------------------------------------------------------------------
# Alignment preview rendering
# --------------------------------------------------------------------------

def render_pair_preview(
    img_a: np.ndarray,
    img_b: np.ndarray,
    dx: float,
    dy: float,
    mode: str = "blend",
    out_width: int = 480,
    checker: int = 48,
) -> np.ndarray:
    """Render an alignment preview of a pair (both images at review scale).

    dx, dy: shift of tile b relative to tile a, in the SAME pixel units as
        the input images (review-scale px; positive dx = b to the right).

    mode:
      - "blend":      tiles overlaid, overlap shown as a 50/50 mix
      - "checker":    overlap shown as an alternating checkerboard of the
                      two tiles (classic alignment inspector)
      - "difference": |A - B| inside the overlap (dark = aligned),
                      non-overlapping parts dimmed
    Tile outlines are always drawn: A in yellow, B in cyan.

    Returns an RGB uint8 image with width <= out_width.
    """
    mode = mode if mode in _PREVIEW_MODES else "blend"
    ia, ib = int(round(dx)), int(round(dy))
    h, w = img_a.shape[:2]
    hb, wb = img_b.shape[:2]

    x0 = min(0, ia)
    y0 = min(0, ib)
    x1 = max(w, ia + wb)
    y1 = max(h, ib + hb)
    cw, ch = x1 - x0, y1 - y0

    ca = np.zeros((ch, cw, 3), np.uint8)
    cb = np.zeros((ch, cw, 3), np.uint8)
    ma = np.zeros((ch, cw), dtype=bool)
    mb = np.zeros((ch, cw), dtype=bool)

    ax0, ay0 = -x0, -y0
    ca[ay0:ay0 + h, ax0:ax0 + w] = img_a
    ma[ay0:ay0 + h, ax0:ax0 + w] = True
    bx0, by0 = ia - x0, ib - y0
    cb[by0:by0 + hb, bx0:bx0 + wb] = img_b
    mb[by0:by0 + hb, bx0:bx0 + wb] = True

    ov = ma & mb
    only_a = ma & ~ov
    only_b = mb & ~ov
    out = np.zeros((ch, cw, 3), np.uint8)

    if mode == "checker":
        yy, xx = np.mgrid[0:ch, 0:cw]
        blk_a = ((xx // checker) + (yy // checker)) % 2 == 0
        sel_a = ov & blk_a
        sel_b = ov & ~blk_a
        out[only_a] = ca[only_a]
        out[only_b] = cb[only_b]
        out[sel_a] = ca[sel_a]
        out[sel_b] = cb[sel_b]
    elif mode == "difference":
        ga = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY)
        diff = np.abs(ga.astype(np.int16) - gb.astype(np.int16))
        d3 = cv2.cvtColor(np.clip(diff.astype(np.float32) * 3.0, 0, 255)
                          .astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out[ov] = d3[ov]
        out[only_a] = (ca[only_a] * 0.35).astype(np.uint8)
        out[only_b] = (cb[only_b] * 0.35).astype(np.uint8)
    else:  # blend
        out[only_a] = ca[only_a]
        out[only_b] = cb[only_b]
        if ov.any():
            mix = (0.5 * ca[ov].astype(np.float32)
                   + 0.5 * cb[ov].astype(np.float32))
            out[ov] = mix.astype(np.uint8)

    # Tile outlines + labels (A = yellow, B = cyan)
    cv2.rectangle(out, (ax0, ay0), (ax0 + w - 1, ay0 + h - 1), (0, 255, 255), 1)
    cv2.rectangle(out, (bx0, by0), (bx0 + wb - 1, by0 + hb - 1), (255, 255, 0), 1)
    _put_label(out, (ax0 + 8, ay0 + 40), "A", (0, 255, 255))
    _put_label(out, (bx0 + 8, by0 + 40), "B", (255, 255, 0))

    if out_width and cw > out_width:
        s = out_width / cw
        out = cv2.resize(out, (out_width, max(1, int(round(ch * s)))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _put_label(img: np.ndarray, org: Tuple[int, int], text: str,
               color: Tuple[int, int, int]) -> None:
    """Draw a readable text label (dark outline + colored fill)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                color, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# "No overlap" pairs (black gap between two tiles)
# --------------------------------------------------------------------------

def no_overlap_position(tile_width: int, gap_px: int = DEFAULT_GAP_PX) -> Tuple[int, int]:
    """(dx, dy) that places tile b directly beside tile a with a black gap.

    dx = tile_width + gap_px puts the left edge of tile b exactly `gap_px`
    pixels to the right of the right edge of tile a, so the final panorama
    shows a vertical black band of width `gap_px` between the two tiles.
    dy = 0 keeps the row level (vertical alignment is unknown when there is
    no overlap; it can still be fine-tuned manually afterwards).
    """
    return (int(tile_width) + int(gap_px), 0)


def apply_no_overlap_flag(
    state: Dict,
    pair_index: int,
    rec: Dict,
    tile_width: int,
    gap_px: int = DEFAULT_GAP_PX,
) -> None:
    """Toggle handler for the per-pair "No overlap (black gap)" checkbox.

    `state` is any dict-like store with the rvp_* keys (Streamlit session
    state in the UI, a plain dict in tests). `rec` is the pair's record in
    review_state["pairs"] — the NON-widget source of truth for positions
    (Streamlit deletes widget-backed keys when their widget is no longer
    rendered, so positions must live here to survive approval). The value
    of the checkbox f"rvp_no_{pair_index}" decides the action:

    True  — the pair is marked as "no overlap": the current position is
            remembered under f"rvp_prev_{pair_index}" (for undo), the pair
            is placed side by side with a black gap (dx = tile_width + gap,
            dy = 0) and auto-approved ("Position is correct" = True).
    False — the mark is removed: the remembered position is restored and
            the pair is un-approved again (falling back to the original
            automatic estimate rec["dx0"]/rec["dy0"] when nothing was
            remembered).

    Runs inside an st.checkbox on_change callback, i.e. BEFORE any rvp_*
    widget is instantiated in the rerun — mutating widget keys here is safe.
    """
    key_dx = f"rvp_dx_{pair_index}"
    key_dy = f"rvp_dy_{pair_index}"
    key_ok = f"rvp_ok_{pair_index}"
    key_prev = f"rvp_prev_{pair_index}"

    if state.get(f"rvp_no_{pair_index}", False):
        # Remember the current position so the mark can be undone.
        state[key_prev] = (float(state.get(key_dx, rec["dx"])),
                           float(state.get(key_dy, rec["dy"])))
        dx, dy = no_overlap_position(tile_width, gap_px)
    else:
        prev = state.get(key_prev, None)
        if key_prev in state:
            del state[key_prev]
        if prev is not None:
            dx = int(round(float(prev[0])))
            dy = int(round(float(prev[1])))
        else:
            dx = int(round(float(rec.get("dx0", rec["dx"]))))
            dy = int(round(float(rec.get("dy0", rec["dy"]))))

    state[key_dx] = int(dx)
    state[key_dy] = int(dy)
    rec["dx"] = float(dx)
    rec["dy"] = float(dy)
    state[key_ok] = bool(state.get(f"rvp_no_{pair_index}", False))


# --------------------------------------------------------------------------
# Review session export / import (JSON)
# --------------------------------------------------------------------------

def to_json(
    tile_names: List[str],
    pairs: List[Dict],
    approved: List[bool],
    no_overlap_flags: Optional[List[bool]] = None,
) -> bytes:
    """Serialize the current review state (positions + approvals) to JSON.

    pairs: list of dicts with at least {"dx", "dy"} in full-resolution px.
    approved: per-pair boolean ("Position is correct" checkbox).
    no_overlap_flags: per-pair boolean ("No overlap (black gap)" checkbox);
        optional for backward compatibility with older sessions.
    """
    payload = {
        "app": "tem-analysis-platform / optical stitcher — pair review",
        "version": 2,
        "tiles": list(tile_names),
        "pairs": [
            {
                "pair_index": i,
                "tile_a": tile_names[i],
                "tile_b": tile_names[i + 1] if i + 1 < len(tile_names) else None,
                "dx": round(float(p["dx"]), 2),
                "dy": round(float(p["dy"]), 2),
                "approved": bool(approved[i]),
                "no_overlap": bool(no_overlap_flags[i])
                    if no_overlap_flags is not None
                    and i < len(no_overlap_flags) else False,
                "source_method": p.get("method", ""),
            }
            for i, p in enumerate(pairs)
        ],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def apply_json(
    data: bytes,
    tile_names: List[str],
) -> List[Dict]:
    """Parse an exported review JSON and return per-pair position records.

    Returns a list of dicts: {dx, dy, approved, method, note}.
    Raises ValueError on malformed input. Pair order is matched by index;
    tile names are checked and mismatches are reported in "note".
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Not a valid JSON file: {e}")

    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError("JSON does not contain a 'pairs' list — "
                         "is this a pair-review export file?")

    out: List[Dict] = []
    for j, p in enumerate(payload["pairs"]):
        try:
            dx = float(p["dx"])
            dy = float(p["dy"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Pair {j + 1}: missing or invalid dx/dy ({e})")

        note = ""
        name_a = p.get("tile_a")
        name_b = p.get("tile_b")
        if name_a is not None and j < len(tile_names) and name_a != tile_names[j]:
            note = f"tile_a mismatch: file has '{name_a}', current row has '{tile_names[j]}'"
        if name_b is not None and j + 1 < len(tile_names) and name_b != tile_names[j + 1]:
            note = (note + "; " if note else "") + \
                f"tile_b mismatch: file has '{name_b}', current row has '{tile_names[j + 1]}'"

        out.append({
            "dx": dx,
            "dy": dy,
            "approved": bool(p.get("approved", False)),
            "no_overlap": bool(p.get("no_overlap", False)),
            "method": "imported",
            "note": note,
        })
    return out
