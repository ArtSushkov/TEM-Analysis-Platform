"""
stitching.py — Pairwise alignment between two microscope images.

Strategy:
1. CLAHE preprocessing -> boost contrast of dark microscope images
2. SIFT detector -> robust to low contrast
3. FLANN matcher -> fast approximate nearest neighbor
4. Lowe's ratio test -> filter out bad matches
   + fixed-pattern suppression (dust on optics matches itself at ~zero shift)
5. RANSAC + estimateAffinePartial2D -> robust transform estimation
6. Verification: SIFT result is cross-checked against a global NCC scan.
   On samples with long horizontal scratches / sparse dots (dust, pores),
   RANSAC can lock onto a self-consistent but WRONG consensus. The NCC
   scan is immune to this and catches such blunders.
7. Fallback to phase correlation if SIFT did not yield enough inliers
8. Fallback to NCC scan if phase correlation also fails.
   NCC uses a two-stage scan (coarse -> fine) with sub-pixel refinement,
   so vertical shifts are accurate to ~0.1 px (no step quantization).
   On samples dominated by long horizontal scratches the dx surface is a
   broad sloped hill, and scoring the FULL strip makes small overlaps
   artificially easy once dy is free — so the global re-scan scores every
   candidate dx on a FIXED-WIDTH central window of the overlap, with dy
   aligned per-dx on the column-averaged band structure.
   The interface-line excursions (pores / dirt / bumps) provide a
   prior-gated correction: the "booking" check (every excursion must find
   a counterpart; fixed-pattern/optics excursions removed) can correct dx
   only when cross-validated by a prior (row median dx / user's expected
   overlap), because single specks — dust on the optics, dirt that moved
   between frames — otherwise outvote the truth.
   A final ECC polish refines the translation to true sub-pixel accuracy.

IMPORTANT — translation convention used everywhere in this module:
`(dx, dy)` is the SHIFT of tile b relative to tile a in a's coordinate
frame (dx > 0 when b is to the RIGHT of a, dy > 0 when b is LOWER).
All NCC strip comparisons therefore use overlap strips of width
`w - dx` (a's columns [dx, w) vs b's columns [0, w - dx)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import preprocessing


@dataclass
class PairResult:
    """Result of stitching a pair (a -> b, where b is the next tile)."""
    success: bool
    method: str              # "sift" | "sift+ncc" | "phase" | "ncc" | "failed"
    translation: Tuple[float, float]   # (dx, dy) — shift of tile b relative to tile a
    rotation_deg: float
    scale: float
    matches: int = 0         # total matches after ratio test
    inliers: int = 0         # RANSAC inliers
    affine: Optional[np.ndarray] = None  # 2x3 affine matrix (b -> a)
    message: str = ""
    # NCC evidence (filled when a global NCC scan was executed for this pair):
    score: float = 0.0                                # NCC score at the final translation
    ncc_best: Optional[Tuple[float, float]] = None    # global NCC scan optimum (dx, dy)
    ncc_best_score: float = 0.0                       # NCC score at the global optimum
    # Row-structure (interface line) evidence. Set when the profile check
    # CORRECTED the intensity scan's dx — trustworthy on scratched samples
    # where the intensity NCC ridge is misleading.
    profile_translation: Optional[Tuple[float, float]] = None


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def _prepare_gray(
    img_a: np.ndarray,
    img_b: np.ndarray,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """CLAHE + grayscale conversion for both images (float32)."""
    if use_clahe:
        a_proc = preprocessing.apply_clahe(img_a, clip_limit=clahe_clip)
        b_proc = preprocessing.apply_clahe(img_b, clip_limit=clahe_clip)
    else:
        a_proc, b_proc = img_a, img_b
    ga = preprocessing.to_gray(a_proc).astype(np.float32)
    gb = preprocessing.to_gray(b_proc).astype(np.float32)
    return ga, gb


def _ncc_score(
    ga: np.ndarray,
    gb: np.ndarray,
    dx: int,
    dy: int,
    min_size: int = 50,
    win: Optional[int] = None,
) -> float:
    """NCC between the overlapping parts of a and b at shift (dx, dy).

    dx is the SHIFT of tile b relative to tile a (dx > 0: b is to the
    right of a, dx < 0: b is to the LEFT of a). Vertically shifted by dy.
    Translation-only score in [-1, 1].

    win: optional fixed evaluation width — only the central `win` columns
    of the overlap are scored. This matters when comparing DIFFERENT dx:
    with the full strip a small overlap is easier to fit (once dy aligns
    the horizontal bands, a narrow strip is mostly band structure), which
    makes the score rise towards degenerate slivers. A fixed window gives
    every dx the same amount of evidence.
    """
    h, w = ga.shape
    if dx >= 0:
        ov = w - dx
        if ov < min_size:
            return -1.0
        a_strip = ga[:, dx:]      # a's right `ov` columns
        b_strip = gb[:, :ov]      # b's left  `ov` columns
    else:
        ov = w + dx
        if ov < min_size:
            return -1.0
        a_strip = ga[:, :ov]      # a's left  `ov` columns
        b_strip = gb[:, -dx:]     # b's right `ov` columns
    if win is not None and win < ov:
        c0 = (ov - win) // 2
        a_strip = a_strip[:, c0:c0 + win]
        b_strip = b_strip[:, c0:c0 + win]
    if dy >= 0:
        ap = a_strip[dy:, :]
        bp = b_strip[:h - dy, :]
    else:
        ap = a_strip[:h + dy, :]
        bp = b_strip[-dy:, :]
    if ap.shape[0] < min_size or ap.shape[1] < min_size:
        return -1.0
    an = (ap - ap.mean()) / (ap.std() + 1e-6)
    bn = (bp - bp.mean()) / (bp.std() + 1e-6)
    return float((an * bn).sum() / an.size)


def _parabolic_offset(fm1: float, f0: float, fp1: float) -> float:
    """Sub-pixel peak offset from three samples (-1, 0, +1). Result in [-1, 1]."""
    denom = fm1 - 2.0 * f0 + fp1
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (fm1 - fp1) / denom, -1.0, 1.0))


def _line_excursions(
    p: np.ndarray,
    thresh: float = 40.0,
    med_size: int = 301,
    min_len: int = 3,
) -> List[Tuple[int, int, float]]:
    """Prominent excursions of the line profile from its local level.

    Returns a list of (x0, x1, prominence) where prominence is how far the
    darkest-structure position jumps up (px). Excursions are pores / dust
    specks / wide bumps on the interface line — the only dx-discriminating
    features on scratched samples.
    """
    from scipy.ndimage import median_filter
    med = median_filter(p, size=med_size, mode="nearest")
    exc = p < med - thresh
    regions: List[Tuple[int, int, float]] = []
    n = len(p)
    i = 0
    while i < n:
        if exc[i]:
            j = i
            # bridge single-column gaps
            while j + 1 < n and (exc[j + 1] or (j + 2 < n and exc[j + 2])):
                j += 1
            if j - i + 1 >= min_len:
                regions.append((int(i), int(j),
                                float(med[(i + j) // 2] - p[i:j + 1].min())))
            i = j + 1
        else:
            i += 1
    return regions


def _remove_fixed_pattern_excursions(
    exc_a: List[Tuple[int, int, float]],
    exc_b: List[Tuple[int, int, float]],
    tol: int = 10,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Drop excursions present in BOTH tiles at the SAME x.

    Dust on the optics appears at the same pixel position in every frame,
    so its excursion shows up in both profiles at the same columns. If kept,
    it creates fake cross-matches (b's dust spike aligns with any of a's
    real spikes) that corrupt every 1D profile correlation.
    """
    def _overlaps(r, lst):
        x0, x1, _ = r
        return any(not (y1 < x0 - tol or y0 > x1 + tol) for (y0, y1, _) in lst)

    keep_a = [r for r in exc_a if not _overlaps(r, exc_b)]
    keep_b = [r for r in exc_b if not _overlaps(r, exc_a)]
    return keep_a, keep_b


def _booking_score(
    exc_a: List[Tuple[int, int, float]],
    exc_b: List[Tuple[int, int, float]],
    dx: int,
    w: int,
    tol: int = 20,
) -> float:
    """Total unmatched excursion prominence at shift dx.

    For shift dx (b placed relative to a) every excursion of a inside the
    overlap [dx, w) must have a counterpart in b at x - dx, and every
    excursion of b inside the overlap [0, w - dx) must have one in a at
    x + dx. Unmatched prominence (px of unexplained line jump) is summed
    from BOTH sides. The true shift minimizes it; wrong shifts leave whole
    structures unexplained. Unlike plain correlation this cannot be fooled
    by ONE coincidental match (e.g. dust) — every structure must book.
    """
    miss = 0.0
    for (x0, x1, pr) in exc_a:
        if x1 < dx or x0 >= w:
            continue                      # exclusive to a (or invalid)
        t0, t1 = x0 - dx, x1 - dx
        if not any(not (y1 < t0 - tol or y0 > t1 + tol)
                   for (y0, y1, _) in exc_b):
            miss += pr
    for (y0, y1, pr) in exc_b:
        if y0 > w - dx - 1 or y1 < 0:
            continue                      # exclusive to b (or invalid)
        t0, t1 = y0 + dx, y1 + dx
        if not any(not (x1 < t0 - tol or x0 > t1 + tol)
                   for (x0, x1, _) in exc_a):
            miss += pr
    return miss


def _decisive_booking(
    fa: np.ndarray,
    fb: np.ndarray,
    min_ov: int = 300,
) -> Optional[Tuple[int, float, float, float]]:
    """Scan shifts by unmatched excursion prominence; return if decisive.

    Returns (best_dx, best_miss, runner_miss, matched_prominence) or None.
    Decisive requires ALL of:
      * the best shift is >= 150 px of prominence better than the runner-up
        (outside +-100 px) and at most 60% of it;
      * at least 100 px of prominence ACTUALLY MATCHES at the best shift.
    The match requirement matters: with only frame-specific dirt excursions
    a large shift can "explain" everything by declaring it exclusive —
    zero misses with zero matches is degenerate, not evidence.
    Flat booking (few/ambiguous excursions) returns None.
    """
    w = len(fa)
    exc_a = _line_excursions(fa)
    exc_b = _line_excursions(fb)
    if not exc_a and not exc_b:
        return None
    exc_a, exc_b = _remove_fixed_pattern_excursions(exc_a, exc_b)
    if not exc_a and not exc_b:
        return None
    lo, hi = -(w - min_ov), w - min_ov
    if hi <= lo:
        return None
    dxs = np.arange(lo, hi + 1, 4)
    scores = np.array([_booking_score(exc_a, exc_b, int(dx), w) for dx in dxs])
    i = int(np.argmin(scores))
    best, bx = float(scores[i]), int(dxs[i])
    mask = np.abs(dxs - bx) > 100
    runner = float(scores[mask].min()) if mask.any() else float("inf")
    matched = _matched_prominence(exc_a, exc_b, bx, w)
    if ((runner - best) >= 150.0 and best <= 0.6 * runner
            and matched >= 100.0):
        return bx, best, runner, matched
    return None


def _matched_prominence(
    exc_a: List[Tuple[int, int, float]],
    exc_b: List[Tuple[int, int, float]],
    dx: int,
    w: int,
    tol: int = 20,
) -> float:
    """Total prominence of excursion pairs that actually match at dx.

    Both directions count, each excursion is credited once.
    """
    total = 0.0
    used_b = set()
    for (x0, x1, pr) in exc_a:
        if x1 < dx or x0 >= w:
            continue
        t0, t1 = x0 - dx, x1 - dx
        hit = None
        for k, (y0, y1, _) in enumerate(exc_b):
            if not (y1 < t0 - tol or y0 > t1 + tol):
                hit = k
                break
        if hit is not None:
            total += pr
            used_b.add(hit)
    for k, (y0, y1, pr) in enumerate(exc_b):
        if k in used_b:
            continue
        if y0 > w - dx - 1 or y1 < 0:
            continue
        t0, t1 = y0 + dx, y1 + dx
        if any(not (x1 < t0 - tol or x0 > t1 + tol)
               for (x0, x1, _) in exc_a):
            total += pr
    return total


def _column_profile_dy(
    ga: np.ndarray,
    gb: np.ndarray,
    dx: int,
    dy_lo: int,
    dy_hi: int,
) -> Tuple[Optional[float], float]:
    """Best constant dy from column-averaged intensity profiles.

    Averaging the overlap columns washes out the repetitive scratch texture
    and keeps the global band structure (dark region / interface line /
    bright region), so the peak matches what a human eye judges. dx is the
    shift of b relative to a (either sign). Returns (best_dy, best_score).
    """
    h, w = ga.shape
    if dx >= 0:
        ov = w - dx
        if ov < 50:
            return None, -1.0
        ra = ga[:, dx:].mean(axis=1)
        rb = gb[:, :ov].mean(axis=1)
    else:
        ov = w + dx
        if ov < 50:
            return None, -1.0
        ra = ga[:, :ov].mean(axis=1)
        rb = gb[:, -dx:].mean(axis=1)
    best_s, best_dy = -1.0, None
    lo = max(int(dy_lo), -(h - 100))
    hi = min(int(dy_hi), h - 100)
    for dy in range(lo, hi + 1):
        if dy >= 0:
            va, vb = ra[dy:], rb[:h - dy]
        else:
            va, vb = ra[:h + dy], rb[-dy:]
        if va.shape[0] < 100:
            continue
        va_ = (va - va.mean()) / (va.std() + 1e-9)
        vb_ = (vb - vb.mean()) / (vb.std() + 1e-9)
        s = float((va_ * vb_).mean())
        if s > best_s:
            best_s, best_dy = s, dy
    if best_dy is None:
        return None, best_s
    # parabolic sub-pixel refinement
    off = 0.0
    if lo < best_dy < hi:
        def _cp(dy):
            if dy >= 0:
                va, vb = ra[dy:], rb[:h - dy]
            else:
                va, vb = ra[:h + dy], rb[-dy:]
            va_ = (va - va.mean()) / (va.std() + 1e-9)
            vb_ = (vb - vb.mean()) / (vb.std() + 1e-9)
            return float((va_ * vb_).mean())
        off = _parabolic_offset(_cp(best_dy - 1), _cp(best_dy), _cp(best_dy + 1))
    return float(best_dy + off), best_s


# --------------------------------------------------------------------------
# Row-structure (interface line) profile — dx-ridge disambiguation
# --------------------------------------------------------------------------

def _line_profile(g: np.ndarray, dark_k: int = 15) -> np.ndarray:
    """Per-column y of the darkest `dark_k`-row window.

    On layered samples (polished cross-sections) the darkest horizontal
    structure in every column is the layer/interface line. Its y-position
    as a function of x — bumps, steps, dark dots near the line — is a
    distinctive 1D signature of the sample that is NOT ambiguous under
    the repetitive horizontal scratch texture.
    """
    h, w = g.shape
    k = min(dark_k, h - 1)
    cs = np.cumsum(g, axis=0)
    cs = np.vstack([np.zeros((1, w), dtype=cs.dtype), cs])
    win = cs[k:, :] - cs[:-k, :]          # sum over each k-row window
    return np.argmin(win, axis=0).astype(np.float32)


def _clean_line_profile(p: np.ndarray) -> np.ndarray:
    """Reject narrow dot-dip outliers, keep genuine wide interface bumps.

    Isolated dark dots above the interface make the raw profile jump for a
    few columns; such short excursions (< 8 columns) are replaced by the
    local median, then the profile is median-smoothed.
    """
    from scipy.ndimage import median_filter
    q = p.astype(np.float32).copy()
    n = len(q)
    for _ in range(3):
        med = median_filter(q, size=101, mode="nearest")
        bad = np.abs(q - med) > 35
        i = 0
        while i < n:
            if bad[i]:
                j = i
                while j < n and bad[j]:
                    j += 1
                if j - i < 8:
                    q[i:j] = med[i:j]
                i = j
            else:
                i += 1
    return median_filter(q, size=15, mode="nearest")


def _profile_ncc_curve(
    fa: np.ndarray,
    fb: np.ndarray,
    min_ov: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized correlation of two line profiles for ALL integer shifts.

    ox = placement offset of tile b relative to tile a (b's profile
    sampled at x - ox). Computed for every ox with |overlap| >= min_ov
    via FFT cross-correlation + prefix sums — O(n log n) for the whole
    curve.
    Returns (oxs, ncc_values), both 1D arrays.
    """
    from scipy.signal import fftconvolve
    va0 = fa.astype(np.float64) - float(fa.mean())
    vb0 = fb.astype(np.float64) - float(fb.mean())
    w = len(va0)
    corr = fftconvolve(va0, vb0[::-1], mode="full")     # length 2w - 1
    # conv[k] = sum_x va0[x] * vb0[x - ox] with ox = k - (w - 1)
    pa = np.concatenate([[0.0], np.cumsum(va0)])
    qa = np.concatenate([[0.0], np.cumsum(va0 ** 2)])
    pb = np.concatenate([[0.0], np.cumsum(vb0)])
    qb = np.concatenate([[0.0], np.cumsum(vb0 ** 2)])
    oxs = np.arange(-(w - min_ov), w - min_ov + 1)
    x0 = np.maximum(0, oxs)
    x1 = np.minimum(w, w + oxs)
    L = (x1 - x0).astype(np.float64)
    La = pa[x1] - pa[x0]
    Qa = qa[x1] - qa[x0]
    # b's window is a's window shifted by -ox: [x0 - ox, x1 - ox)
    x0b, x1b = x0 - oxs, x1 - oxs
    Lb = pb[x1b] - pb[x0b]
    Qb = qb[x1b] - qb[x0b]
    C = corr[(w - 1) + oxs]
    ma, mb = La / L, Lb / L
    num = C - L * ma * mb
    den = np.sqrt(np.maximum(Qa - L * ma * ma, 1e-9)
                  * np.maximum(Qb - L * mb * mb, 1e-9))
    return oxs, num / den


# --------------------------------------------------------------------------
# SIFT
# --------------------------------------------------------------------------

def _estimate_affine(
    pts_src: np.ndarray,   # points in b
    pts_dst: np.ndarray,   # points in a
    ransac_thresh: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate partial affine (translation + rotation + uniform scale) via RANSAC."""
    if len(pts_src) < 4:
        return None, None
    H, mask = cv2.estimateAffinePartial2D(
        pts_src, pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.995,
    )
    return H, mask


def _filter_fixed_pattern_matches(
    matches: List[cv2.DMatch],
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    zero_radius: float = 25.0,
    min_cluster: int = 3,
) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray, int]:
    """Drop matches whose translation is ~zero while typical translation is large.

    Dust and other fixed-pattern artifacts on the optics appear at the SAME
    pixel position in every frame. Feature points on such artifacts match
    each other with translation ~ (0, 0). If enough of them survive the
    ratio test, they can steal the RANSAC consensus and collapse the shift.
    They are dropped when the median match displacement is far from zero.

    Returns: (matches, pts_a, pts_b, n_dropped) — points are filtered too.
    """
    if len(matches) < 12:
        return matches, pts_a, pts_b, 0
    tx = pts_b[:, 0] - pts_a[:, 0]
    ty = pts_b[:, 1] - pts_a[:, 1]
    disp = np.sqrt(tx ** 2 + ty ** 2)
    near_zero = (np.abs(tx) < zero_radius) & (np.abs(ty) < zero_radius)
    if near_zero.sum() < min_cluster:
        return matches, pts_a, pts_b, 0
    if float(np.median(disp)) < 100.0:
        return matches, pts_a, pts_b, 0  # row barely moves — may be genuine
    kept_idx = [i for i, nz in enumerate(near_zero) if not nz]
    if len(kept_idx) < 8:
        return matches, pts_a, pts_b, 0  # too risky to filter
    kept = [matches[i] for i in kept_idx]
    return kept, pts_a[kept_idx], pts_b[kept_idx], int(near_zero.sum())


def stitch_pair_sift(
    img_a: np.ndarray,
    img_b: np.ndarray,
    n_features: int = 5000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
    suppress_fixed_pattern: bool = True,
    prior_translation: Optional[Tuple[float, float]] = None,
    prior_window: float = 60.0,
) -> PairResult:
    """Stitch a pair via SIFT + RANSAC.

    img_a, img_b: BGR images.
    Returns the transform b -> a. The translation (dx, dy) is the shift of
    tile b relative to tile a. If b is to the right of a, dx > 0.

    suppress_fixed_pattern: drop matches with ~zero translation when the
        row clearly moves (protects against dust on the optics).
    prior_translation: if given (e.g. from the NCC scan), only matches whose
        translation is within `prior_window` px of the prior are used for
        RANSAC. This rescues SIFT on samples where ambiguous features
        (scratches, dots) otherwise produce a wrong consensus.
    """
    if use_clahe:
        a_proc = preprocessing.apply_clahe(img_a, clip_limit=clahe_clip)
        b_proc = preprocessing.apply_clahe(img_b, clip_limit=clahe_clip)
    else:
        a_proc, b_proc = img_a, img_b

    ga = preprocessing.to_gray(a_proc)
    gb = preprocessing.to_gray(b_proc)

    sift = cv2.SIFT_create(nfeatures=n_features)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)

    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return PairResult(
            success=False, method="sift",
            translation=(0.0, 0.0), rotation_deg=0.0, scale=1.0,
            message="Not enough keypoints",
        )

    # FLANN for SIFT (KD-tree)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < 4:
        return PairResult(
            success=False, method="sift",
            translation=(0.0, 0.0), rotation_deg=0.0, scale=1.0,
            matches=len(good),
            message=f"Too few good matches after ratio test: {len(good)}",
        )

    # Points: src in b, dst in a (transform b -> a)
    pts_b = np.float32([kb[m.trainIdx].pt for m in good])
    pts_a = np.float32([ka[m.queryIdx].pt for m in good])

    notes = []
    if suppress_fixed_pattern:
        good_f, pts_a_f, pts_b_f, n_dropped = _filter_fixed_pattern_matches(
            good, pts_a, pts_b)
        if n_dropped > 0:
            good, pts_a, pts_b = good_f, pts_a_f, pts_b_f
            notes.append(f"dropped {n_dropped} fixed-pattern (dust) matches")

    if prior_translation is not None:
        pdx, pdy = prior_translation
        mdx = pts_b[:, 0] - pts_a[:, 0]
        mdy = pts_b[:, 1] - pts_a[:, 1]
        near = (np.abs(mdx - pdx) <= prior_window) & (np.abs(mdy - pdy) <= prior_window)
        if near.sum() < 6:
            return PairResult(
                success=False, method="sift",
                translation=(0.0, 0.0), rotation_deg=0.0, scale=1.0,
                matches=len(good), inliers=0,
                message=f"Only {int(near.sum())} matches near NCC prior "
                        f"({pdx:.0f}, {pdy:.0f}) — prior refinement not possible",
            )
        pts_b = pts_b[near]
        pts_a = pts_a[near]
        notes.append(f"prior-filtered matches: {int(near.sum())}/{len(good)}")

    H, mask = _estimate_affine(pts_b, pts_a, ransac_thresh=ransac_thresh)
    if H is None:
        return PairResult(
            success=False, method="sift",
            translation=(0.0, 0.0), rotation_deg=0.0, scale=1.0,
            matches=len(good), inliers=0,
            message="RANSAC did not converge",
        )

    inliers = int(mask.sum()) if mask is not None else 0
    dx, dy = float(H[0, 2]), float(H[1, 2])
    rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    scale = float(np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2))

    msg = f"SIFT: {len(good)} matches, {inliers} inliers"
    if notes:
        msg += " (" + ", ".join(notes) + ")"

    return PairResult(
        success=True,
        method="sift",
        translation=(dx, dy),
        rotation_deg=rot,
        scale=scale,
        matches=len(good),
        inliers=inliers,
        affine=H,
        message=msg,
    )


# --------------------------------------------------------------------------
# Phase correlation
# --------------------------------------------------------------------------

def stitch_pair_phase(
    img_a: np.ndarray,
    img_b: np.ndarray,
    use_clahe: bool = True,
    gray: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> PairResult:
    """Fallback: phase correlation for pure translation.

    Works well only when there is no rotation and brightness is similar.
    In our case (CLAHE equalizes brightness), it usually succeeds.
    """
    if gray is not None:
        ga, gb = gray
    else:
        ga, gb = _prepare_gray(img_a, img_b, use_clahe=use_clahe)

    (shift_x, shift_y), response = cv2.phaseCorrelate(ga, gb)
    # phaseCorrelate returns the shift to apply to b to get a.
    # This matches our convention: translation = (dx, dy) — shift of b relative to a.

    success = response > 0.15  # empirical threshold
    return PairResult(
        success=success,
        method="phase",
        translation=(float(shift_x), float(shift_y)),
        rotation_deg=0.0,
        scale=1.0,
        matches=0,
        inliers=0,
        score=float(response),
        message=f"Phase correlation: response={response:.4f}",
    )


# --------------------------------------------------------------------------
# NCC scan (two-stage, sub-pixel)
# --------------------------------------------------------------------------

def _scan_ncc(
    ga: np.ndarray,
    gb: np.ndarray,
    dx_range: Tuple[int, int],
    dy_range: Tuple[int, int],
    step: int,
    rows_step: int = 1,
) -> Tuple[float, int, int]:
    """Brute-force NCC scan over shifts. Returns (best_score, best_dx, best_dy).

    dx is the SHIFT of b relative to a; overlap strips have width `w - dx`.
    """
    h, w = ga.shape
    gas = ga[::rows_step, :]
    gbs = gb[::rows_step, :]
    hs = gas.shape[0]

    best_score = -1.0
    best_dx, best_dy = dx_range[0], dy_range[0]

    dx_hi = min(dx_range[1], w - 50)
    dy_lo = max(dy_range[0], -(hs - 50))
    dy_hi = min(dy_range[1], hs - 50)

    for dx in range(dx_range[0], dx_hi + 1, step):
        ov = w - dx
        a_strip = gas[:, dx:]      # a's right (w - dx) columns
        b_strip = gbs[:, :ov]      # b's left  (w - dx) columns
        for dy in range(dy_lo, dy_hi + 1, step):
            # dy is in full-resolution units; convert for the subsampled rows
            dy_s = int(round(dy / rows_step))
            if dy_s >= 0:
                ap = a_strip[dy_s:, :]
                bp = b_strip[:hs - dy_s, :]
            else:
                ap = a_strip[:hs + dy_s, :]
                bp = b_strip[-dy_s:, :]
            if ap.shape[0] < 50 or ap.shape[1] < 50:
                continue
            an = (ap - ap.mean()) / (ap.std() + 1e-6)
            bn = (bp - bp.mean()) / (bp.std() + 1e-6)
            score = float((an * bn).sum() / an.size)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_score, best_dx, best_dy


def stitch_pair_ncc(
    img_a: np.ndarray,
    img_b: np.ndarray,
    dx_range: Tuple[int, int] = (100, 2600),
    dy_range: Tuple[int, int] = (-600, 600),
    step: int = 25,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
    fine_step: int = 2,
    fine_window: int = 24,
    gray: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    row_prior_dx: Optional[float] = None,
    expected_dx: Optional[float] = None,
) -> PairResult:
    """Robust fallback: brute-force shift scan using NCC.

    dx_range / dy_range are SHIFT ranges (dx = shift of tile b relative to
    tile a; internally clamped so the overlap stays >= eval_win px).

    Coarse-to-fine search:
      1. Global scan on a downscaled pyramid (max side ~600 px) covering the
         FULL dx/dy search ranges. Fast enough to allow a wide vertical
         range (±600 px) — microscope stages often drift hundreds of px
         between shots, which the old ±100 px scan silently missed.
      2. Fine scan ±`fine_window` px around the coarse optimum at full
         resolution with `fine_step` px.
      3. Global dx re-scan: every candidate dx is vertically aligned on the
         column-averaged band structure and then scored on a FIXED-WIDTH
         central window of the overlap, so scores of different dx carry the
         same evidence (a small overlap must not win by being easy).
      4. Row-structure booking: interface-line excursions must find
         counterparts in the other tile; prior-gated dx correction.
      5. dx-ridge resolution with the row prior / expected overlap when the
         correlation surface is flat along dx.
      6. Parabolic sub-pixel refinement.
      7. ECC polish: full 2D intensity-based refinement of the translation
         (sub-pixel, robust to smooth brightness gradients).
    `step` is kept for backward compatibility but is no longer used:
    the pyramid grid step is derived from the scale factor.

    row_prior_dx: median dx of the pairs already solved in this row (robust
        prior for the expected overlap of the current pair).
    expected_dx: dx implied by the user's expected capture overlap.
    """
    del step  # legacy parameter, unused
    if gray is not None:
        ga, gb = gray
    else:
        ga, gb = _prepare_gray(img_a, img_b, use_clahe=use_clahe,
                               clahe_clip=clahe_clip)
    h, w = ga.shape

    # Fixed evaluation window for dx-comparable full-resolution scoring.
    eval_win = max(250, min(400, w // 5))

    # Stage 1: global coarse scan on a downscaled copy
    scale = min(1.0, 600.0 / max(h, w))
    if scale < 1.0:
        gs_a = cv2.resize(ga, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gs_b = cv2.resize(gb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gs_a, gs_b = ga, gb

    dx_hi_cap = min(dx_range[1], w - eval_win - 10)
    c_dx = (max(10, int(dx_range[0] * scale)), int(dx_hi_cap * scale))
    c_dy = (int(dy_range[0] * scale), int(dy_range[1] * scale))
    # ~8 px grid on the small image; peak localization error ~±4 small px
    # (~±14 full px) — well inside the ±30 px fine window below.
    c_step = 8
    coarse_score, cdx_s, cdy_s = _scan_ncc(gs_a, gs_b, c_dx, c_dy, c_step, rows_step=2)
    cdx, cdy = int(round(cdx_s / scale)), int(round(cdy_s / scale))

    # Stage 2: fine scan around the coarse optimum at full resolution
    dx_lo = max(50, cdx - fine_window)
    dx_hi = min(cdx + fine_window, w - eval_win)
    dy_lo = max(dy_range[0], cdy - fine_window)
    dy_hi = min(dy_range[1], cdy + fine_window)
    fine_score, fdx, fdy = _scan_ncc(ga, gb, (dx_lo, dx_hi), (dy_lo, dy_hi),
                                     fine_step, rows_step=1)
    if fine_score < 0:  # degenerate window — keep coarse result
        fine_score, fdx, fdy = coarse_score, cdx, cdy

    # Stage 2b: global dx re-scan, scored consistently at full resolution.
    # On samples with long horizontal scratches the dx surface is a broad
    # sloped hill with several near-equal bumps, and the coarse pyramid
    # (downscaled -> smoothed) can land on a neighbour bump. Every candidate
    # dx is aligned vertically on the column-averaged band structure (cheap
    # 1D estimate) and then scored on a FIXED-WIDTH central window of the
    # overlap, so different dx carry the same amount of evidence and the
    # comparison is apples-to-apples. (The previous versions either compared
    # a rows_step=2 coarse winner with the stage-2 score, or used the full
    # strip — whose score rises towards degenerate small overlaps once dy is
    # free per dx — and silently missed the hill top.)
    dx_lo2 = max(50, dx_range[0])
    dx_hi2 = min(dx_hi_cap, w - eval_win)
    if dx_hi2 - dx_lo2 > 2 * fine_window:
        best = (fine_score, int(round(fdx)), int(round(fdy)))
        for dx in range(dx_lo2, dx_hi2 + 1, 8):
            dy_c, _ = _column_profile_dy(ga, gb, dx,
                                         int(fdy) - 70, int(fdy) + 70)
            if dy_c is None:
                continue
            for dy in range(int(dy_c) - 6, int(dy_c) + 7, 2):
                s = _ncc_score(ga, gb, dx, dy, win=eval_win)
                if s > best[0]:
                    best = (s, dx, dy)
        # local step-1 refinement of the re-scan winner (full rows)
        s2, x2, y2 = best
        for dx in range(max(dx_lo2, best[1] - 8), min(dx_hi2, best[1] + 8) + 1):
            for dy in range(best[2] - 8, best[2] + 9):
                s = _ncc_score(ga, gb, dx, dy, win=eval_win)
                if s > s2:
                    s2, x2, y2 = s, dx, dy
        if s2 > fine_score:
            fine_score, fdx, fdy = s2, x2, y2

    # Stage 2c (always computed, prior-gated): row-structure booking of dx.
    # The interface-line excursions (pores / dirt specks / wide bumps) are
    # the only dx-discriminating features on scratched samples, but single
    # specks are treacherous: dust on the OPTICS sits at the same x in every
    # frame and fake-matches every real spike of the other tile, and loose
    # dirt ON the sample may move between frames (leaving "unexplained"
    # spikes in one tile). So the booking (every excursion must find a
    # counterpart; fixed-pattern excursions removed first) is used only when
    # it is decisive AND cross-validated by a prior (the median dx of the
    # pairs already solved in this row, or the user's expected overlap).
    # Without a prior the intensity answer is kept and the booking candidate
    # is reported in the message for transparency.
    profile_translation: Optional[Tuple[float, float]] = None
    profile_note = ""
    booking_rejected = False
    try:
        fa_p = _clean_line_profile(_line_profile(ga))
        fb_p = _clean_line_profile(_line_profile(gb))
        booking = _decisive_booking(fa_p, fb_p)
    except Exception:
        booking = None
    prior = row_prior_dx if row_prior_dx is not None else expected_dx
    if booking is not None:
        bx, best_miss, runner_miss, matched_prom = booking
        if abs(bx - int(round(fdx))) <= 60:
            pass  # booking agrees with the intensity answer — no action
        elif prior is not None and abs(bx - prior) <= max(150.0, 0.15 * prior):
            x0p, x1p = max(0, bx), min(w, w + bx)
            if x1p - x0p >= 300:
                dy_prof = float(np.median(
                    fa_p[x0p:x1p] - fb_p[x0p - bx:x1p - bx]))
                if abs(dy_prof - fdy) <= 90:
                    dy_n, dy_s = _column_profile_dy(
                        ga, gb, bx, int(dy_prof) - 45, int(dy_prof) + 45)
                    if dy_n is not None:
                        old_dx, old_dy = fdx, fdy
                        fdx, fdy = float(bx), float(dy_n)
                        fine_score = _ncc_score(ga, gb, int(round(fdx)),
                                                int(round(fdy)), win=eval_win)
                        profile_translation = (float(fdx), float(fdy))
                        profile_note = (
                            f" | dx corrected by row-structure booking "
                            f"(cross-validated by prior dx~{prior:.0f}): "
                            f"intensity scan said dx={old_dx:.0f}, booking "
                            f"leaves {best_miss:.0f}px unexplained at "
                            f"dx={bx:.0f} vs {runner_miss:.0f}px at the "
                            f"runner-up ({matched_prom:.0f}px matched); "
                            f"dy={fdy:.1f} from band structure "
                            f"(NCC {dy_s:.3f})")
        elif prior is not None:
            # booking far from the row prior: the row's consistency is
            # stronger evidence than a single pair's excursion booking
            booking_rejected = True
            profile_note = (
                f" | NOTE row-structure booking prefers dx={bx} "
                f"({best_miss:.0f}px unexplained vs {runner_miss:.0f}px, "
                f"{matched_prom:.0f}px matched) but it is far from the row "
                f"prior dx~{prior:.0f} — kept the intensity answer "
                f"dx={fdx:.0f}")
        else:
            profile_note = (
                f" | NOTE row-structure booking prefers dx={bx} "
                f"({best_miss:.0f}px unexplained vs {runner_miss:.0f}px, "
                f"{matched_prom:.0f}px matched); kept the intensity answer "
                f"dx={fdx:.0f} unless the dx ridge turns out flat")
    fdx, fdy = int(round(fdx)), int(round(fdy))

    # Stage 2d: flat-ridge handling. When the fixed-window score barely
    # varies across the neighbourhood, the intensity evidence alone cannot
    # sharpen dx (this is the normal case on scratched samples — the score
    # plateau IS the best the texture can offer). Two honest disambiguators:
    #   * a prior — the median dx of the pairs already solved in this row,
    #     or the user's expected overlap: anchor to it;
    #   * the row-structure booking (fixed-pattern excursions removed, real
    #     matches required): adopt when decisive and no prior exists.
    # Note: plain 1D profile CORRELATION is deliberately NOT used here — a
    # single dust speck fixed to the optics (same x in both frames) fake-
    # matches every real spike of the other tile and can outvote the truth.
    ridge_note = ""
    if profile_translation is None:
        probe_dxs = [int(round(fdx)) + d for d in range(-150, 151, 10)
                     if 50 <= int(round(fdx)) + d <= w - eval_win]
        probe_scores = [_ncc_score(ga, gb, d, int(round(fdy)), win=eval_win)
                        for d in probe_dxs]
        if probe_scores and (max(probe_scores) - min(probe_scores)) < 0.02:
            prior = row_prior_dx if row_prior_dx is not None else expected_dx
            adopted = False
            if prior is not None:
                px = int(round(prior))
                if 50 <= px <= w - eval_win:
                    s_p, x_p, y_p = -1.0, px, int(round(fdy))
                    for dx in range(max(50, px - 30),
                                    min(w - eval_win, px + 30) + 1):
                        for dy in range(int(fdy) - 25, int(fdy) + 26):
                            s = _ncc_score(ga, gb, dx, dy, win=eval_win)
                            if s > s_p:
                                s_p, x_p, y_p = s, dx, dy
                    if s_p >= fine_score - 0.10:
                        fdx, fdy = float(x_p), float(y_p)
                        fine_score = s_p
                        which = ("row median" if row_prior_dx is not None
                                 else "expected overlap")
                        ridge_note = (
                            f" | flat dx ridge — anchored to {which} prior "
                            f"dx~{prior:.0f} (NCC {s_p:.4f})")
                        adopted = True
            if not adopted and booking is not None and not booking_rejected:
                bx, best_miss, runner_miss, matched_prom = booking
                x0p, x1p = max(0, bx), min(w, w + bx)
                if x1p - x0p >= 300:
                    dy_prof = float(np.median(
                        fa_p[x0p:x1p] - fb_p[x0p - bx:x1p - bx]))
                    if abs(dy_prof - fdy) <= 90:
                        dy_n, dy_s = _column_profile_dy(
                            ga, gb, bx, int(dy_prof) - 45, int(dy_prof) + 45)
                        if dy_n is not None:
                            fdx, fdy = float(bx), float(dy_n)
                            fine_score = _ncc_score(ga, gb, int(round(fdx)),
                                                    int(round(fdy)),
                                                    win=eval_win)
                            profile_translation = (float(fdx), float(fdy))
                            ridge_note = (
                                f" | flat dx ridge — adopted decisive "
                                f"row-structure booking dx={bx:.0f} "
                                f"({best_miss:.0f}px unexplained vs "
                                f"{runner_miss:.0f}px runner-up, "
                                f"{matched_prom:.0f}px matched)")
                            adopted = True
            if not adopted:
                ridge_note = (
                    " | WARNING: dx weakly constrained (flat correlation "
                    "ridge — repetitive horizontal texture); verify this "
                    "seam visually, or set Expected overlap in the sidebar "
                    "to lock it")
    fdx, fdy = int(round(fdx)), int(round(fdy))

    # Stage 3: parabolic sub-pixel refinement (score is smooth near the peak)
    if profile_translation is not None:
        # dx/dy already sub-pixel refined from the row structure; the
        # intensity surface here is the misleading sloped ridge, so its
        # parabola would drag dx back towards the false maximum.
        final = (float(profile_translation[0]), float(profile_translation[1]))
        final_score = fine_score
    else:
        s0 = fine_score
        sx_m = _ncc_score(ga, gb, fdx - 1, fdy, win=eval_win)
        sx_p = _ncc_score(ga, gb, fdx + 1, fdy, win=eval_win)
        sy_m = _ncc_score(ga, gb, fdx, fdy - 1, win=eval_win)
        sy_p = _ncc_score(ga, gb, fdx, fdy + 1, win=eval_win)
        off_x = _parabolic_offset(sx_m, s0, sx_p)
        off_y = _parabolic_offset(sy_m, s0, sy_p)
        final = (float(fdx + off_x), float(fdy + off_y))
        final_score = _ncc_score(ga, gb, int(round(final[0])),
                                 int(round(final[1])), win=eval_win)

    # Stage 4: ECC polish — full 2D intensity-based refinement of the
    # translation on the overlap region (sub-pixel, robust to smooth
    # brightness gradients that a strip-NCC handles poorly).
    # SKIPPED when the row-structure profile corrected the translation:
    # the profile + column-averaged dy already carry the sub-pixel info,
    # and ECC here would drift back towards the misleading texture ridge
    # (CLAHE-ECC moves dy by ~8-14 px on scratched samples).
    ecc_note = ""
    if profile_translation is None:
        if use_clahe:
            try:
                a8 = preprocessing.to_gray(preprocessing.apply_clahe(
                    img_a, clip_limit=clahe_clip)).astype(np.float32)
                b8 = preprocessing.to_gray(preprocessing.apply_clahe(
                    img_b, clip_limit=clahe_clip)).astype(np.float32)
            except Exception:
                a8, b8 = ga.astype(np.float32), gb.astype(np.float32)
        else:
            a8, b8 = ga.astype(np.float32), gb.astype(np.float32)
        ix0, iy0 = max(0, int(final[0])), max(0, int(final[1]))
        ix1, iy1 = min(w, int(final[0]) + w), min(h, int(final[1]) + h)
        if ix1 - ix0 >= 100 and iy1 - iy0 >= 100:
            tpl = a8[iy0:iy1, ix0:ix1]
            inp = b8[iy0 - int(final[1]):iy1 - int(final[1]),
                     ix0 - int(final[0]):ix1 - int(final[0])]
            if tpl.size and inp.size and tpl.shape == inp.shape:
                warp = np.eye(2, 3, dtype=np.float32)
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        200, 1e-6)
                try:
                    cc, warp = cv2.findTransformECC(tpl, inp, warp,
                                                    cv2.MOTION_TRANSLATION,
                                                    crit, None, 5)
                    # warp maps input(b-crop) coords -> template(a-crop) coords
                    dx_e = final[0] - float(warp[0, 2])
                    dy_e = final[1] - float(warp[1, 2])
                    # accept only a small correction (guard against ECC drift)
                    if (cc > 0.3 and abs(dx_e - final[0]) < 30
                            and abs(dy_e - final[1]) < 30):
                        ecc_note = f", ECC polish ({cc:.4f})"
                        final = (float(dx_e), float(dy_e))
                except cv2.error:
                    pass  # ECC failed to converge — keep the NCC result

    # Diagnostics: how sharply is the peak constrained along dx?
    # (After Stage 2c/2d this only fires when no decisive interface line was
    # available — e.g. a featureless sample.)
    notes = ""
    if profile_translation is None:
        dx_probe = [int(round(final[0]) + d) for d in range(-150, 151, 10)
                    if 50 <= int(round(final[0]) + d) <= w - eval_win]
        dx_scores = [_ncc_score(ga, gb, d, int(round(final[1])), win=eval_win)
                     for d in dx_probe]
        if dx_scores and (max(dx_scores) - min(dx_scores)) < 0.02:
            notes = (" | WARNING: dx weakly constrained (flat correlation ridge — "
                     "repetitive horizontal texture); verify this seam visually")

    success = max(coarse_score, fine_score) > 0.3
    return PairResult(
        success=success,
        method="ncc",
        translation=final,
        rotation_deg=0.0,
        scale=1.0,
        matches=0,
        inliers=0,
        score=final_score,
        ncc_best=final,
        ncc_best_score=final_score,
        profile_translation=profile_translation,
        message=f"NCC scan: best_score={final_score:.4f} at "
                f"dx={final[0]:.1f}, dy={final[1]:.1f}{ecc_note}{profile_note}"
                f"{ridge_note}{notes}",
    )


# --------------------------------------------------------------------------
# Orchestrator with NCC verification
# --------------------------------------------------------------------------

def stitch_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    n_features: int = 5000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
    min_inliers_sift: int = 8,
    enable_phase_fallback: bool = True,
    enable_ncc_fallback: bool = True,
    verify_with_ncc: bool = True,
    ncc_verify_gap: float = 0.07,
    suppress_fixed_pattern: bool = True,
    row_prior_dx: Optional[float] = None,
    expected_dx: Optional[float] = None,
) -> PairResult:
    """Full pair stitching pipeline with verification and fallbacks.

    Order:
    1. SIFT + RANSAC (with fixed-pattern suppression)
    2. Global NCC scan with row-structure (interface line) verification
       (if verification enabled or SIFT failed)
    3. If SIFT succeeded: cross-check against the NCC scan. If the NCC scan
       finds a significantly better alignment (score gap > `ncc_verify_gap`)
       OR its row-structure profile corrected dx away from the SIFT answer,
       SIFT likely locked onto a wrong consensus (scratches / dust dots).
       Then:
       a. re-run SIFT with the corrected translation as a prior — if enough
          matches agree, use this refined SIFT (keeps sub-pixel precision +
          rotation);
       b. otherwise use the corrected NCC translation.
    4. If SIFT failed: phase correlation -> NCC scan (classic fallbacks).

    row_prior_dx / expected_dx: optional horizontal-shift priors used only
    when a pair's dx is ambiguous (flat correlation ridge); see
    `stitch_pair_ncc`.
    """
    ga, gb = _prepare_gray(img_a, img_b, use_clahe=use_clahe, clahe_clip=clahe_clip)

    # 1. SIFT
    result = stitch_pair_sift(
        img_a, img_b,
        n_features=n_features,
        ratio_thresh=ratio_thresh,
        ransac_thresh=ransac_thresh,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        suppress_fixed_pattern=suppress_fixed_pattern,
    )
    sift_ok = result.success and result.inliers >= min_inliers_sift

    # 2. Global NCC scan — as verification evidence and/or fallback
    ncc_result: Optional[PairResult] = None
    need_ncc = verify_with_ncc or (not sift_ok and enable_ncc_fallback)
    if need_ncc:
        ncc_result = stitch_pair_ncc(
            img_a, img_b, use_clahe=use_clahe, gray=(ga, gb),
            row_prior_dx=row_prior_dx, expected_dx=expected_dx,
        )

    # 3. Verification of a successful SIFT result
    if sift_ok and verify_with_ncc and ncc_result is not None and ncc_result.success:
        sdx, sdy = result.translation
        eval_win = max(250, min(400, ga.shape[1] // 5))
        s_at = _ncc_score(ga, gb, int(round(sdx)), int(round(sdy)),
                          win=eval_win)
        result.score = s_at
        result.ncc_best = ncc_result.ncc_best
        result.ncc_best_score = ncc_result.ncc_best_score

        # The row-structure profile corrected dx away from SIFT's answer:
        # on scratched samples the SIFT consensus can sit on the same
        # misleading intensity ridge as the NCC scan, with a DECENT score,
        # so the plain score-gap check would not catch it.
        prof_t = ncc_result.profile_translation
        prof_disagrees = (
            prof_t is not None
            and (abs(prof_t[0] - sdx) > 60 or abs(prof_t[1] - sdy) > 60)
        )
        if prof_disagrees or ncc_result.ncc_best_score > s_at + ncc_verify_gap:
            # SIFT disagrees with the verified alignment — suspect.
            # 3a. try SIFT restricted to the corrected neighborhood
            reason = ("row-structure profile "
                      f"({prof_t[0]:.0f}, {prof_t[1]:.0f})"
                      if prof_disagrees else
                      f"NCC score {s_at:.3f} vs {ncc_result.ncc_best_score:.3f}")
            refined = stitch_pair_sift(
                img_a, img_b,
                n_features=n_features,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
                use_clahe=use_clahe,
                clahe_clip=clahe_clip,
                suppress_fixed_pattern=suppress_fixed_pattern,
                prior_translation=(ncc_result.ncc_best[0], ncc_result.ncc_best[1]),
                prior_window=60.0,
            )
            if refined.success and refined.inliers >= min_inliers_sift:
                rdx, rdy = refined.translation
                near_profile = (prof_t is None
                                or (abs(rdx - prof_t[0]) <= 60
                                    and abs(rdy - prof_t[1]) <= 60))
                s_ref = _ncc_score(ga, gb, int(round(rdx)), int(round(rdy)),
                                   win=eval_win)
                if near_profile and s_ref >= ncc_result.ncc_best_score - 0.05:
                    refined.method = "sift+ncc"
                    refined.score = s_ref
                    refined.ncc_best = ncc_result.ncc_best
                    refined.ncc_best_score = ncc_result.ncc_best_score
                    refined.profile_translation = prof_t
                    refined.message += (
                        f" | verified: SIFT re-anchored to corrected prior "
                        f"({ncc_result.ncc_best[0]:.0f}, {ncc_result.ncc_best[1]:.0f}); "
                        f"first SIFT answer ({sdx:.0f}, {sdy:.0f}) rejected "
                        f"({reason})"
                    )
                    return refined

            # 3b. fall back to the corrected NCC translation
            ncc_result.method = "ncc"
            ncc_result.message = (
                f"NCC scan (SIFT answer ({sdx:.0f}, {sdy:.0f}) rejected by "
                f"verification: {reason}): "
                f"best_score={ncc_result.ncc_best_score:.4f} at "
                f"dx={ncc_result.translation[0]:.1f}, dy={ncc_result.translation[1]:.1f}"
            )
            return ncc_result

        # verification passed
        result.message += f" | NCC verification passed ({s_at:.3f} vs best " \
                          f"{ncc_result.ncc_best_score:.3f})"
        return result

    if sift_ok:
        # verification disabled — classic behavior
        return result

    # 4. Fallbacks when SIFT failed
    if enable_phase_fallback:
        phase_result = stitch_pair_phase(img_a, img_b, use_clahe=use_clahe,
                                         gray=(ga, gb))
        if phase_result.success:
            if result.inliers > 0:
                sift_dx = result.translation[0]
                if abs(phase_result.translation[0] - sift_dx) <= 300:
                    return phase_result
            else:
                return phase_result

    if enable_ncc_fallback and ncc_result is not None and ncc_result.success:
        return ncc_result

    # Nothing worked — return SIFT result even if poor
    if result.inliers > 0:
        result.message = f"Low quality: {result.message}"
        return result

    return PairResult(
        success=False,
        method="failed",
        translation=(0.0, 0.0),
        rotation_deg=0.0,
        scale=1.0,
        message="All stitching methods failed",
    )
