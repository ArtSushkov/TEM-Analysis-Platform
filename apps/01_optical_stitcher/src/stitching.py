"""
stitching.py — Pairwise alignment between two microscope images.

Strategy:
1. CLAHE preprocessing -> boost contrast of dark microscope images
2. SIFT detector -> robust to low contrast
3. FLANN matcher -> fast approximate nearest neighbor
4. Lowe's ratio test -> filter out bad matches
5. RANSAC + estimateAffinePartial2D -> robust transform estimation
6. Fallback to phase correlation if SIFT did not yield enough inliers
7. Fallback to template matching (NCC scan) if phase correlation also fails
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import preprocessing


@dataclass
class PairResult:
    """Result of stitching a pair (a -> b, where b is the next tile)."""
    success: bool
    method: str              # "sift" | "phase" | "ncc" | "failed"
    translation: Tuple[float, float]   # (dx, dy) — shift of b relative to a in a's coordinates
    rotation_deg: float
    scale: float
    matches: int = 0         # total matches after ratio test
    inliers: int = 0         # RANSAC inliers
    affine: Optional[np.ndarray] = None  # 2x3 affine matrix (b -> a)
    message: str = ""


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


def stitch_pair_sift(
    img_a: np.ndarray,
    img_b: np.ndarray,
    n_features: int = 5000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0,
    use_clahe: bool = True,
    clahe_clip: float = 3.0,
) -> PairResult:
    """Stitch a pair via SIFT + RANSAC.

    img_a, img_b: BGR images.
    Returns the transform b -> a (i.e. applying affine to b yields a).
    The translation in the result is the dx, dy by which b must be shifted
    to align with a. If b is to the right of a (typical case), dx > 0.
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

    return PairResult(
        success=True,
        method="sift",
        translation=(dx, dy),
        rotation_deg=rot,
        scale=scale,
        matches=len(good),
        inliers=inliers,
        affine=H,
        message=f"SIFT: {len(good)} matches, {inliers} inliers",
    )


def stitch_pair_phase(
    img_a: np.ndarray,
    img_b: np.ndarray,
    use_clahe: bool = True,
) -> PairResult:
    """Fallback: phase correlation for pure translation.

    Works well only when there is no rotation and brightness is similar.
    In our case (CLAHE equalizes brightness), it usually succeeds.
    """
    if use_clahe:
        a_proc = preprocessing.apply_clahe(img_a)
        b_proc = preprocessing.apply_clahe(img_b)
    else:
        a_proc, b_proc = img_a, img_b

    ga = preprocessing.to_gray(a_proc).astype(np.float32)
    gb = preprocessing.to_gray(b_proc).astype(np.float32)

    (shift_x, shift_y), response = cv2.phaseCorrelate(ga, gb)
    # phaseCorrelate returns the shift to apply to b to get a.
    # This matches our convention: translation = (dx, dy) — shift of b relative to a.
    # If shift_x > 0, b is to the right of a — our typical case.

    success = response > 0.15  # empirical threshold
    return PairResult(
        success=success,
        method="phase",
        translation=(float(shift_x), float(shift_y)),
        rotation_deg=0.0,
        scale=1.0,
        matches=0,
        inliers=0,
        message=f"Phase correlation: response={response:.4f}",
    )


def stitch_pair_ncc(
    img_a: np.ndarray,
    img_b: np.ndarray,
    dx_range: Tuple[int, int] = (200, 2500),
    dy_range: Tuple[int, int] = (-100, 100),
    step: int = 25,
    use_clahe: bool = True,
) -> PairResult:
    """Last-resort fallback: brute-force shift scan using NCC.

    Used when both SIFT and phase correlation fail.
    Slow, but robust for pairs with a known overlap range.
    """
    if use_clahe:
        a_proc = preprocessing.apply_clahe(img_a)
        b_proc = preprocessing.apply_clahe(img_b)
    else:
        a_proc, b_proc = img_a, img_b

    ga = preprocessing.to_gray(a_proc).astype(np.float32)
    gb = preprocessing.to_gray(b_proc).astype(np.float32)

    h, w = ga.shape
    best_score = -1.0
    best_dx, best_dy = 0, 0

    for dx in range(dx_range[0], dx_range[1] + 1, step):
        if dx >= w:
            continue
        for dy in range(dy_range[0], dy_range[1] + 1, step):
            if abs(dy) >= h:
                continue
            # Take the right edge of a (width dx) and the left edge of b (width dx)
            a_strip = ga[:, -dx:]
            b_strip = gb[:, :dx]
            # Y shift: take the intersection
            if dy >= 0:
                a_part = a_strip[dy:, :]
                b_part = b_strip[:h - dy, :]
            else:
                a_part = a_strip[:h + dy, :]
                b_part = b_strip[-dy:, :]
            if a_part.shape[0] < 50 or a_part.shape[1] < 50:
                continue
            # NCC
            a_n = (a_part - a_part.mean()) / (a_part.std() + 1e-6)
            b_n = (b_part - b_part.mean()) / (b_part.std() + 1e-6)
            score = float((a_n * b_n).sum() / a_n.size)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    success = best_score > 0.3
    return PairResult(
        success=success,
        method="ncc",
        translation=(float(best_dx), float(best_dy)),
        rotation_deg=0.0,
        scale=1.0,
        matches=0,
        inliers=0,
        message=f"NCC scan: best_score={best_score:.4f} at dx={best_dx}, dy={best_dy}",
    )


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
) -> PairResult:
    """Full pair stitching pipeline with fallbacks.

    Order:
    1. SIFT + RANSAC (primary)
    2. Phase correlation (if SIFT yielded too few inliers)
    3. NCC scan (if phase correlation failed)
    """
    # 1. SIFT
    result = stitch_pair_sift(
        img_a, img_b,
        n_features=n_features,
        ratio_thresh=ratio_thresh,
        ransac_thresh=ransac_thresh,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
    )
    if result.success and result.inliers >= min_inliers_sift:
        return result

    # 2. Phase correlation fallback
    if enable_phase_fallback:
        phase_result = stitch_pair_phase(img_a, img_b, use_clahe=use_clahe)
        if phase_result.success:
            # Verify that phase correlation result does not contradict SIFT
            # (if SIFT gave some answer, it should be close)
            if result.inliers > 0:
                sift_dx = result.translation[0]
                if abs(phase_result.translation[0] - sift_dx) > 300:
                    # Contradiction — do not trust phase
                    pass
                else:
                    return phase_result
            else:
                return phase_result

    # 3. NCC fallback
    if enable_ncc_fallback:
        ncc_result = stitch_pair_ncc(img_a, img_b, use_clahe=use_clahe)
        if ncc_result.success:
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
