"""
structure_tensor_segmentation.py — Grain segmentation via local lattice-fringe
orientation, using the structure tensor.

**Why this works for TEM**
-------------------------
On TEM cross-sections of polycrystalline layers, each grain shows **lattice
fringes** — parallel bright/dark stripes from atomic planes. The fringes run
in different directions in adjacent grains (different crystallographic
orientations). The structure tensor computes the **local dominant direction**
of these stripes at every pixel; pixels with the same direction belong to the
same grain.

Unlike the global-FFT approach (which averages over all grains and cannot
localise boundaries), the structure tensor is **inherently local** — it
operates in a window of size ~sigma around each pixel.

Pipeline
--------
1. **CLAHE** — local contrast enhancement so lattice fringes are visible.
2. **Structure tensor** — `skimage.feature.structure_tensor(sigma)` computes
   the 2×2 symmetric matrix ``(Axx, Axy, Ayy)`` at every pixel.
3. **Orientation** — ``orientation = 0.5 * atan2(2*Axy, Ayy−Axx)`` gives the
   dominant gradient direction; the fringes are perpendicular to this, but
   for grain segmentation we only care that pixels in the same grain share
   the same orientation.
4. **Coherence** — ``coherence = (λ1−λ2)/(λ1+λ2) = sqrt((Axx−Ayy)² + 4Axy²) /
   (Axx+Ayy)`` where λ1 ≥ λ2 are the tensor eigenvalues. This is the classic
   anisotropy measure: coherence ≈ 1 = locally oriented texture (clear lattice
   fringes, straight boundary — crystalline); coherence ≈ 0 = isotropic
   gradients (amorphous speckle, noise) or no texture at all (uniform area,
   vacuum). NOTE: an earlier version used ``sqrt(det)/trace = sqrt(λ1λ2)/(λ1+λ2)``,
   which is an ISOTROPY measure (≈0.5 for noise, 0 for fringes) — the opposite
   semantics. The current formula is the correct one.
5. **Threshold** — pixels with coherence below a threshold are set to
   background.
6. **Orientation quantisation** — the continuous orientation angle is binned
   into N discrete bins (e.g. 36 bins of 5° each). Pixels in the same bin
   share the same crystallographic orientation.
7. **Connected-component labeling** — within each orientation bin, separate
   connected regions = individual grains.
8. **Post-processing** — remove small regions, fill holes, exclude border
   grains.

Parameters
----------
* **sigma** — structure-tensor window size. Must be ~1.5× the lattice-fringe
  period. Too small = noisy; too large = blurs across grain boundaries.
  Default 5.
* **coherence_threshold** — pixels with coherence < this are background.
  Default 0.3. Higher = stricter (fewer grains, more background).
* **n_orientation_bins** — number of discrete orientation bins. Default 36
  (5° per bin). More bins = finer orientation discrimination but more
  fragmented grains.
* **min_grain_area_px** — remove grains smaller than this.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure
from skimage import color as skcolor


# === Parameters ===

@dataclass
class StructureTensorParams:
    """Parameters for the structure-tensor grain segmentation pipeline."""
    # --- Preprocessing ---
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8

    # --- Structure tensor ---
    # Window size (sigma in pixels). Should be ~1.5× the lattice-fringe period.
    # For TEM at 20–35 pm/px with fringes every ~5–10 px, sigma 5 is good.
    sigma: float = 8.0

    # --- Coherence threshold ---
    # Pixels with coherence < this are background (amorphous/noise).
    # Range [0, 1]. Higher = stricter (fewer grains, more background).
    coherence_threshold: float = 0.35

    # --- Orientation smoothing ---
    # Median-filter the orientation map before quantisation to reduce noise.
    # 0 = no smoothing. 3–7 typical.
    orientation_median_filter_px: int = 11

    # --- Orientation quantisation ---
    # Number of discrete orientation bins (over [-90°, 90°)).
    # 72 bins = 2.5° per bin; 36 bins = 5° per bin.
    n_orientation_bins: int = 36

    # --- Watershed refinement ---
    # After initial labeling, use distance-transform + watershed to split
    # touching grains that share the same orientation but are separated by
    # a boundary visible in the image.
    use_watershed_refinement: bool = False
    watershed_min_distance_px: int = 5

    # --- Image downscale ---
    max_image_dim: int = 1024

    # --- Post-processing ---
    min_grain_area_px: int = 500
    fill_holes_max_px: int = 100
    exclude_left_right_borders: bool = True
    exclude_top_bottom_borders: bool = False
    border_margin_px: int = 1

    # --- Merge similar-orientation grains ---
    merge_adjacent_orientations: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# === Result ===

@dataclass
class StructureTensorResult:
    """Output of the structure-tensor segmentation pipeline."""
    labels: np.ndarray              # int32 label image (0 = background)
    n_grains: int
    orientation_map: np.ndarray     # (H, W) float64 — orientation angle in degrees [-90, 90)
    coherence_map: np.ndarray       # (H, W) float64 — coherence in [0, 1]
    orientation_bins: np.ndarray    # (H, W) int32 — quantised orientation bin id (0 = bg)
    method: str = "structure_tensor"


# === Pipeline ===

def _preprocess(image: np.ndarray, p: StructureTensorParams) -> np.ndarray:
    """CLAHE for contrast enhancement."""
    img = image.astype(np.float64)
    if p.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip_limit,
                                tileGridSize=(p.clahe_tile_grid, p.clahe_tile_grid))
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        img = clahe.apply(img_u8).astype(np.float64)
    return img


def _compute_structure_tensor(image: np.ndarray, sigma: float
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the structure tensor and derive orientation + coherence.

    Returns
    -------
    (orientation_deg, coherence, energy)
        orientation_deg: (H, W) float64 — dominant gradient direction in
            degrees [-90, 90). The fringes themselves run perpendicular to it.
        coherence: (H, W) float64 — anisotropy in [0, 1].
            coherence = (λ1−λ2)/(λ1+λ2); 1 = locally oriented (crystalline
            fringes / straight edge), 0 = isotropic (amorphous/noise) or
            textureless.
        energy: (H, W) float64 — trace = λ1+λ2 = total gradient energy
            ("is there any texture at all").
    """
    # structure_tensor returns (Axx, Axy, Ayy) for 2D.
    Axx, Axy, Ayy = feature.structure_tensor(image, sigma=sigma, order="xy")

    # Orientation: angle of the dominant gradient direction.
    # atan2(2*Axy, Ayy - Axx) gives the angle in [-pi, pi].
    # We convert to [-pi/2, pi/2) (axial data, mod 180°).
    orientation = 0.5 * np.arctan2(2.0 * Axy, Ayy - Axx)
    orientation_deg = np.degrees(orientation)

    # Coherence (anisotropy) = (λ1−λ2)/(λ1+λ2).
    # For a 2×2 symmetric tensor: (λ1−λ2)² = (Axx−Ayy)² + 4·Axy².
    # 1.0 = all gradient energy in one direction (oriented texture);
    # 0.0 = isotropic gradients (amorphous speckle / noise) or no texture.
    trace = Axx + Ayy
    diff = np.sqrt((Axx - Ayy) ** 2 + 4.0 * Axy ** 2)
    coherence = np.where(trace > 1e-12, diff / (trace + 1e-12), 0.0)
    coherence = np.clip(coherence, 0.0, 1.0)

    return orientation_deg, coherence, trace


def cleanup_grain_mask_with_structure_tensor(
    image: np.ndarray,
    grain_mask: np.ndarray,
    sigma: float = 3.0,
    coherence_threshold: float = 0.35,
    orientation_targets_deg: list | None = None,
    orientation_tolerance_deg: float = 25.0,
    min_remove_area_px: int = 25,
) -> tuple[np.ndarray, dict]:
    """Remove amorphous / wrongly-oriented pixels from a grain mask.

    Designed as a post-filter for click-to-segment (FFT): region growing may
    include amorphous inclusions or flat areas that sit between the crystal
    and the region-growing threshold. The structure tensor judges every pixel
    of the mask independently:

    1. **Coherence gate** — pixels whose local texture is isotropic or
       textureless (coherence < ``coherence_threshold``) are amorphous /
       vacuum → removal candidates.
    2. **Orientation gate** (optional, needs ``orientation_targets_deg``) —
       pixels with enough gradient energy whose local fringe direction does
       not match ANY of the target orientations (within tolerance) belong to
       another grain or an edge artifact → removal candidates. The gradient
       direction of a fringe pattern equals the scattering-vector direction
       of the corresponding FFT reflection (mod 180°), so the FFT angles of
       the selected reflections can be used as targets directly.
    3. **Hysteresis** — only *connected components* of the candidate mask
       larger than ``min_remove_area_px`` are actually removed. Small
       candidate specks (fringe gaps, single noisy pixels) survive.

    Parameters
    ----------
    image : (H, W) grayscale array (uint8 or float).
    grain_mask : (H, W) bool — the grain mask to clean.
    sigma : structure tensor window (px). ~1.5× the lattice-fringe period;
        3.0 is a good default for typical d-spacings of 2–8 px.
    coherence_threshold : anisotropy below this = amorphous/textureless.
    orientation_targets_deg : FFT angles (0–180°, mod 180) of the reflections
        selected for this grain (the same convention as the app's FFT θ:
        ``atan2(dr, dc) mod 180``). They are converted to the tensor frame
        internally. Empty list = orientation gate disabled.
    orientation_tolerance_deg : ± tolerance for the orientation gate.
    min_remove_area_px : smallest connected candidate region that is removed.

    Returns
    -------
    (cleaned_mask, diag)
        cleaned_mask: (H, W) bool — grain minus removed regions.
        diag: dict — removed_px, removed_frac, n_components_removed,
        coherence_gate_px, orientation_gate_px, params echo.
    """
    diag: dict[str, Any] = {
        "removed_px": 0,
        "removed_frac": 0.0,
        "n_components_removed": 0,
        "coherence_gate_px": 0,
        "orientation_gate_px": 0,
        "sigma": float(sigma),
        "coherence_threshold": float(coherence_threshold),
        "orientation_tolerance_deg": float(orientation_tolerance_deg),
        "min_remove_area_px": int(min_remove_area_px),
        "orientation_gate_applied": False,
    }
    mask = np.asarray(grain_mask, dtype=bool)
    if not mask.any():
        return mask, diag

    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float64)

    orientation_deg, coherence, energy = _compute_structure_tensor(img, sigma)

    inside = mask
    # 1. Coherence gate: isotropic or textureless → amorphous/vacuum.
    cand_coherence = inside & (coherence < coherence_threshold)

    # 2. Orientation gate: textured pixels whose fringe direction matches
    #    none of the selected reflections. Only applied where the local
    #    texture is strong enough to give a meaningful orientation.
    cand_orientation = np.zeros_like(mask)
    targets = [float(t) % 180.0 for t in (orientation_targets_deg or [])
               if np.isfinite(t)]
    if targets and orientation_tolerance_deg > 0:
        # Energy guard: skip the orientation test where the gradient energy is
        # negligible (textureless), because the orientation estimate there is
        # noise. The floor is RELATIVE to the strongest texture inside the
        # mask — a quantile of the mask's energy would fail when the mask
        # contains only strongly-textured pixels (the quantile would then sit
        # at full texture energy and wrongly exempt a quarter of the pixels).
        energy_floor = 0.05 * float(np.max(energy[inside]))
        textured = inside & (energy >= max(energy_floor, 1e-9))
        if textured.any():
            # Convert the tensor orientation into the FFT angle frame.
            # Convention check (empirical + analytic): for a fringe pattern
            # whose FFT reflection sits at angle theta = atan2(dr, dc), the
            # tensor orientation (skimage order='xy', x = columns) is
            # (90° − theta) mod 180. Verified for theta = 0/30/60/90°.
            ang_fft = np.mod(90.0 - orientation_deg, 180.0)
            min_diff = np.full(ang_fft.shape, 180.0)
            for t in targets:
                d = np.abs(ang_fft - t)
                d = np.minimum(d, 180.0 - d)  # axial distance mod 180°
                min_diff = np.minimum(min_diff, d)
            cand_orientation = textured & (min_diff > orientation_tolerance_deg)
            diag["orientation_gate_applied"] = True

    candidates = cand_coherence | cand_orientation
    diag["coherence_gate_px"] = int(cand_coherence.sum())
    diag["orientation_gate_px"] = int((cand_orientation & ~cand_coherence).sum())

    # 3. Hysteresis: remove only sufficiently large connected components.
    cleaned = mask.copy()
    if candidates.any():
        lbl, n = ndi.label(candidates.astype(np.int32))
        if n > 0:
            areas = np.bincount(lbl.ravel())
            areas[0] = 0  # background
            to_remove = np.where(areas >= min_remove_area_px)[0]
            if len(to_remove) > 0:
                remove_mask = np.isin(lbl, to_remove) & inside
                cleaned = mask & ~remove_mask
                diag["n_components_removed"] = int(len(to_remove))
                diag["removed_px"] = int(remove_mask.sum())
                diag["removed_frac"] = float(
                    remove_mask.sum() / max(mask.sum(), 1))

    diag["final_area_px"] = int(cleaned.sum())
    # Safety net: if the cleanup removed (almost) everything, the FFT seed
    # found *something* — it is far more likely that the cleanup thresholds
    # are too aggressive for this image than that the whole grain is vacuum.
    # A few stray survivors below 10% of the original area are treated the
    # same way (they would produce a meaningless speck of a grain).
    if mask.any() and cleaned.sum() < 0.10 * mask.sum():
        diag["note"] = (f"cleanup would remove {100 * (1 - cleaned.sum() / mask.sum()):.0f}% "
                        f"of the grain — thresholds likely too aggressive; "
                        f"original mask kept")
        return mask, diag
    return cleaned, diag


def _quantise_orientation(orientation_deg: np.ndarray, n_bins: int
                           ) -> np.ndarray:
    """Quantise the continuous orientation angle into discrete bins.

    Orientation is in [-90, 90). Bins are 0..n_bins-1.

    Returns
    -------
    np.ndarray (H, W) int32 — bin id per pixel.
    """
    # Map [-90, 90) → [0, n_bins)
    bin_width = 180.0 / n_bins
    bins = ((orientation_deg + 90.0) / bin_width).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def _watershed_refinement(labels: np.ndarray, image: np.ndarray,
                           p: StructureTensorParams) -> np.ndarray:
    """Split touching grains using distance-transform watershed.

    Only applies to labels that have **more than one connected component after
    distance-transform peak finding** — i.e. only splits grains that appear to
    contain multiple seeds. This prevents over-segmentation.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32 — initial labels.
    image : np.ndarray (H, W) float64 — preprocessed image.
    p : StructureTensorParams

    Returns
    -------
    np.ndarray (H, W) int32 — refined labels.
    """
    if labels.max() == 0:
        return labels

    from skimage import feature as skfeature
    from skimage import segmentation as skseg

    h, w = labels.shape
    refined = labels.copy()
    next_label = int(labels.max()) + 1

    # For each existing label, check if it should be split.
    # Use distance transform inside each grain's mask; if there are multiple
    # local maxima (peaks), the grain likely contains multiple sub-grains.
    for label_id in range(1, int(labels.max()) + 1):
        mask = (labels == label_id)
        if not mask.any():
            continue
        area = mask.sum()
        if area < p.watershed_min_distance_px ** 2 * 4:
            continue  # too small to split

        # Distance transform inside this grain.
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

        # Find local maxima (potential seeds for splitting).
        coords = skfeature.peak_local_max(
            dist,
            min_distance=p.watershed_min_distance_px,
            threshold_abs=dist.max() * 0.3,
            exclude_border=False,
        )

        if len(coords) <= 1:
            continue  # only one peak — no split needed

        # Create markers for this grain.
        markers = np.zeros((h, w), dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        # Compute gradient ONLY within this grain's mask.
        grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        # Set gradient to 0 outside the grain (so watershed doesn't leak).
        gradient[~mask] = 0

        # Run watershed on this grain only.
        ws_result = skseg.watershed(gradient, markers=markers, mask=mask)

        # Assign new labels.
        for i in range(1, len(coords) + 1):
            sub_mask = (ws_result == i)
            if sub_mask.any():
                refined[sub_mask] = next_label
                next_label += 1

    return refined


def _label_grains(orientation_bins: np.ndarray, coherence: np.ndarray,
                   p: StructureTensorParams) -> np.ndarray:
    """Label grains: connected components within each orientation bin.

    Pixels with coherence < threshold are background (label 0).
    Each orientation bin is labeled separately, so pixels with the same
    orientation but in different spatial locations become separate grains.

    Vectorised via compound encoding: for each pixel, compute
    (bin_id + 1) * (n_cc + 1) + cc_label, then relabel compactly.
    """
    # Background mask: low coherence.
    valid = coherence >= p.coherence_threshold
    if not valid.any():
        return np.zeros(orientation_bins.shape, dtype=np.int32)

    # Encode: each valid pixel gets a compound key = (bin_id + 1).
    # Invalid pixels get 0 (background).
    # Then ndi.label on each bin separately is equivalent to:
    # 1. Create a combined image where pixel value = bin_id + 1 (0 = bg).
    # 2. ndi.label on the "same value" regions.
    # But ndi.label doesn't do "same value" — it does "non-zero".
    # So we need a different approach:
    # Use watershed or region labeling on the bin image.

    # Fast approach: label connected components of the valid mask, then
    # split each CC by orientation bin.
    # Step 1: label all valid pixels together.
    cc_labels, n_cc = ndi.label(valid)
    if n_cc == 0:
        return np.zeros(orientation_bins.shape, dtype=np.int32)

    # Step 2: For each CC, check if it spans multiple orientation bins.
    # If it does, split it by bin.
    # Compound key: cc_label * n_bins + (bin_id + 1).
    # Two pixels with the same compound key are in the same grain
    # (same CC + same bin). But pixels in the same CC + same bin that are
    # NOT connected to each other (separated by a different bin) should be
    # separate grains. So we need to re-label within each compound.

    # Build compound image: 0 = bg, else (cc_label * n_bins + bin_id + 1).
    compound = np.where(valid,
                         cc_labels.astype(np.int64) * (p.n_orientation_bins + 1)
                         + (orientation_bins.astype(np.int64) + 1),
                         0)

    # Step 3: Re-label connected components of the compound image.
    # Two adjacent pixels with the same compound value are in the same grain.
    # Two adjacent pixels with different compound values are in different grains
    # (either different CC, or same CC but different bin = grain boundary).
    # This correctly splits grains at orientation boundaries!

    # Use ndi.label on "compound > 0" but that would merge different compounds.
    # Instead, use a watershed-like approach: label each unique compound value
    # as a separate region, then check connectivity.

    # Efficient approach: use the compound values directly as labels,
    # but split non-connected regions with the same compound.
    # ndi.label doesn't support "label by value" — but we can use:
    # For each unique compound, find its connected components.

    # However, iterating over all unique compounds is slow if there are many.
    # Alternative: use skimage.measure.label on each compound mask.
    # But the FAST way: assign each pixel a unique grain id based on
    # (compound, connected_component_within_compound).

    # The fastest vectorised approach:
    # 1. Create a boundary mask where adjacent pixels have different compounds.
    # 2. Label connected regions of "same compound" using a custom labeling.

    # Use skimage.measure.label with the compound as the input:
    # It labels connected regions of non-zero values, treating different
    # values as different regions? No — skimage.measure.label treats all
    # non-zero as foreground and gives them sequential labels regardless of value.

    # The correct approach: for each unique compound value, label its CCs.
    # But this is O(n_compounds) iterations of ndi.label.

    # Optimisation: use the fact that compound = cc_label * (n_bins+1) + bin+1.
    # Within a single CC (cc_label), pixels with different bins are different grains.
    # We can use a "boundary detection" approach:
    # boundaries = where adjacent pixels have different compounds.
    # Then label connected regions of "non-boundary, non-background".

    # Simple and fast: dilate each compound, compare with neighbors.
    # Actually, the simplest correct approach that's also fast:
    # Use the compound values themselves as labels, then split same-value
    # non-connected regions with a single pass of ndi.label on each.

    # Let's try: use a lookup approach.
    # 1. Get all unique compound values (excluding 0).
    unique_compounds = np.unique(compound[compound > 0])
    if len(unique_compounds) == 0:
        return np.zeros(orientation_bins.shape, dtype=np.int32)

    # 2. For each unique compound, label its connected components.
    # Vectorised: use cc labels as offset from next_label.
    labels = np.zeros(compound.shape, dtype=np.int32)
    next_label = 1
    for uc in unique_compounds:
        mask = (compound == uc)
        if not mask.any():
            continue
        cc, n = ndi.label(mask)
        if n > 0:
            # Assign labels: next_label, next_label+1, ..., next_label+n-1.
            labels[mask] = next_label + cc[mask] - 1
            next_label += n

    return labels


def _merge_adjacent_orientations(labels: np.ndarray, orientation_bins: np.ndarray,
                                  coherence: np.ndarray, p: StructureTensorParams
                                  ) -> np.ndarray:
    """Merge grains whose orientation bins differ by 1 (adjacent bins).

    This handles cases where the same grain has a slight orientation gradient
    that spans two bins. Two grains are merged if:
    - They touch (share a boundary).
    - Their orientation bins differ by at most 1.
    """
    if labels.max() == 0:
        return labels

    # Build a mapping: label → orientation bin (the most common bin in that grain).
    label_bins = {}
    for region in measure.regionprops(labels):
        label_id = region.label
        coords = region.coords
        bins_in_region = orientation_bins[coords[:, 0], coords[:, 1]]
        # Most common bin.
        bin_counts = np.bincount(bins_in_region)
        label_bins[label_id] = int(np.argmax(bin_counts))

    # Find adjacent labels (dilation-based adjacency).
    # Dilate each label by 1px and check overlaps.
    unique_labels = sorted(label_bins.keys())
    if len(unique_labels) < 2:
        return labels

    # Build adjacency graph via region boundary contact.
    # Use a quick approach: dilate the label image and check neighbors.
    dilated = ndi.grey_dilation(labels.astype(np.int32), size=3)
    adjacency = set()
    h, w = labels.shape
    # For each pixel, check if its dilated neighbor has a different label.
    # Vectorised: compare labels with dilated labels where they differ.
    diff_mask = (dilated != labels) & (labels > 0) & (dilated > 0)
    if diff_mask.any():
        pairs = np.stack([labels[diff_mask], dilated[diff_mask]], axis=1)
        for a, b in pairs:
            if a != b:
                adjacency.add((min(a, b), max(a, b)))

    # Union-Find for merging.
    parent = list(range(max(unique_labels) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for a, b in adjacency:
        bin_a = label_bins.get(a, -1)
        bin_b = label_bins.get(b, -1)
        if bin_a < 0 or bin_b < 0:
            continue
        # Check if bins are adjacent (differ by at most 1, considering wrap-around).
        diff = abs(bin_a - bin_b)
        diff = min(diff, p.n_orientation_bins - diff)  # wrap-around
        if diff <= 1:
            union(a, b)

    # Apply merges.
    merged = np.zeros_like(labels)
    new_id = 1
    root_to_new = {}
    for old_label in unique_labels:
        root = find(old_label)
        if root not in root_to_new:
            root_to_new[root] = new_id
            new_id += 1
        merged[labels == old_label] = root_to_new[root]

    return merged


def _postprocess_labels(labels: np.ndarray, p: StructureTensorParams) -> np.ndarray:
    """Remove small regions, fill holes, exclude border grains (vectorised)."""
    if labels.max() == 0:
        return labels
    h, w = labels.shape

    # 1. Remove small regions.
    counts = np.bincount(labels.ravel())
    small_labels = np.where(counts < p.min_grain_area_px)[0]
    small_labels = small_labels[small_labels > 0]
    if len(small_labels) > 0:
        labels = labels.copy()
        labels[np.isin(labels, small_labels)] = 0

    if labels.max() == 0:
        return labels

    # 2. Fill holes (dilation-based, fast).
    grain_mask = labels > 0
    filled_mask = ndi.binary_fill_holes(grain_mask)
    new_pixels = filled_mask & ~grain_mask
    if new_pixels.any():
        labels_u16 = labels.astype(np.uint16)
        for _ in range(3):
            dilated = cv2.dilate(labels_u16, np.ones((3, 3), np.uint8), iterations=1)
            fill_mask = new_pixels & (labels == 0)
            labels[fill_mask] = dilated[fill_mask].astype(labels.dtype)
            new_pixels = filled_mask & (labels == 0)
            if not new_pixels.any():
                break

    # 3. Exclude border-touching grains.
    margin = p.border_margin_px
    border_labels = set()
    if p.exclude_left_right_borders:
        border_labels.update(labels[:, :margin].ravel().tolist())
        border_labels.update(labels[:, w - margin:].ravel().tolist())
    if p.exclude_top_bottom_borders:
        border_labels.update(labels[:margin, :].ravel().tolist())
        border_labels.update(labels[h - margin:, :].ravel().tolist())
    border_labels.discard(0)
    if border_labels:
        labels[np.isin(labels, list(border_labels))] = 0

    # 4. Relabel compactly.
    unique = np.unique(labels)
    unique = unique[unique > 0]
    relabel_lut = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    for new, old in enumerate(unique, start=1):
        relabel_lut[old] = new
    labels = relabel_lut[labels]

    return labels


# === Main entry point ===

def segment_grains_structure_tensor(
    image: np.ndarray,
    pixel_size_nm: float,
    params: StructureTensorParams | None = None,
) -> StructureTensorResult:
    """Run the structure-tensor grain segmentation pipeline.

    Parameters
    ----------
    image : np.ndarray (H, W) uint8
        Grayscale image of the layer to analyse.
    pixel_size_nm : float
        Pixel size in nm/px (for documentation only).
    params : StructureTensorParams, optional

    Returns
    -------
    StructureTensorResult
    """
    if params is None:
        params = StructureTensorParams()
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h_orig, w_orig = image.shape

    # 0. Downscale for speed.
    scale = min(1.0, float(params.max_image_dim) / max(h_orig, w_orig))
    if scale < 1.0:
        img_small = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)),
                                interpolation=cv2.INTER_AREA)
    else:
        img_small = image
        scale = 1.0

    # 1. Preprocess.
    img = _preprocess(img_small, params)

    # 2. Structure tensor.
    orientation_deg, coherence, _ = _compute_structure_tensor(img, params.sigma)

    # 2b. Smooth orientation map (median filter) to reduce noise.
    if params.orientation_median_filter_px > 0:
        ksize = params.orientation_median_filter_px
        if ksize % 2 == 0:
            ksize += 1
        # Median filter on orientation (handle wrap-around at ±90°).
        # Use cos/sin trick to avoid discontinuities at the ±90° boundary.
        orient_rad = np.deg2rad(orientation_deg)
        cos_orient = ndi.median_filter(np.cos(2 * orient_rad), size=ksize)
        sin_orient = ndi.median_filter(np.sin(2 * orient_rad), size=ksize)
        orientation_deg = np.degrees(0.5 * np.arctan2(sin_orient, cos_orient))
        # Also smooth coherence.
        coherence = ndi.median_filter(coherence, size=ksize)

    # 3. Quantise orientation.
    orientation_bins = _quantise_orientation(orientation_deg, params.n_orientation_bins)

    # 4. Label grains.
    labels_small = _label_grains(orientation_bins, coherence, params)

    # 4b. Watershed refinement — split touching grains with same orientation.
    if params.use_watershed_refinement and labels_small.max() > 0:
        labels_small = _watershed_refinement(labels_small, img, params)

    # 5. Merge adjacent-orientation grains.
    if params.merge_adjacent_orientations and labels_small.max() > 1:
        labels_small = _merge_adjacent_orientations(labels_small, orientation_bins, coherence, params)

    # 6. Post-process on small image (fast).
    small_params = StructureTensorParams(**{**asdict(params),
        "min_grain_area_px": max(1, int(params.min_grain_area_px * scale * scale)),
        "fill_holes_max_px": max(1, int(params.fill_holes_max_px * scale * scale)),
    })
    labels_small = _postprocess_labels(labels_small, small_params)

    # 7. Upscale to original resolution.
    if scale < 1.0:
        labels_full = cv2.resize(
            labels_small.astype(np.uint16), (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        orientation_full = cv2.resize(
            orientation_deg, (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR,
        )
        coherence_full = cv2.resize(
            coherence, (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR,
        )
        orient_bins_full = cv2.resize(
            orientation_bins.astype(np.uint16), (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    else:
        labels_full = labels_small
        orientation_full = orientation_deg
        coherence_full = coherence
        orient_bins_full = orientation_bins

    # 8. Final cleanup on full-res: remove tiny upscale artifacts.
    if labels_full.max() > 0 and scale < 1.0:
        counts = np.bincount(labels_full.ravel())
        small_labels = np.where(counts < params.min_grain_area_px)[0]
        small_labels = small_labels[small_labels > 0]
        if len(small_labels) > 0:
            labels_full[np.isin(labels_full, small_labels)] = 0
        unique = np.unique(labels_full)
        unique = unique[unique > 0]
        relabel_lut = np.zeros(int(labels_full.max()) + 1, dtype=labels_full.dtype)
        for new, old in enumerate(unique, start=1):
            relabel_lut[old] = new
        labels_full = relabel_lut[labels_full]

    n_grains = int(labels_full.max()) if labels_full.max() > 0 else 0

    return StructureTensorResult(
        labels=labels_full.astype(np.int32),
        n_grains=n_grains,
        orientation_map=orientation_full,
        coherence_map=coherence_full,
        orientation_bins=orient_bins_full.astype(np.int32) + 1,  # 0 = bg
    )


# === Visualization ===

def colorize_orientation_map(orientation_deg: np.ndarray, coherence: np.ndarray,
                              coherence_threshold: float = 0.3) -> np.ndarray:
    """Render the orientation map as an RGB image.

    Hue = orientation angle, saturation = coherence (high = vivid, low = gray).

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    # Normalise orientation to [0, 1] for hue.
    hue = (orientation_deg + 90.0) / 180.0  # [0, 1]
    # Saturation = coherence (clipped).
    sat = np.clip(coherence, 0.0, 1.0)
    # Value = 1.0 where coherent, 0.3 where not (background dimmer).
    val = np.where(coherence >= coherence_threshold, 1.0, 0.3)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = skcolor.hsv2rgb(hsv)
    return (rgb * 255).astype(np.uint8)


def colorize_coherence_map(coherence: np.ndarray) -> np.ndarray:
    """Render the coherence map as a grayscale image (bright = high coherence)."""
    return (coherence * 255).astype(np.uint8)
