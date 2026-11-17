"""
sam_segmenter.py — Layer segmentation via Segment Anything Model (SAM).

This module is a thin wrapper around Meta's SAM (ViT-B by default) that:
  * Downloads the model checkpoint on first use (~350 MB, cached locally).
  * Generates automatic masks for a TEM cross-section image.
  * Filters masks to candidate layers (by area, eccentricity, intensity uniformity).
  * Provides helpers to render candidates for the UI.

The torch / segment_anything imports are **lazy**: the module can be imported
without torch installed, and only fails when ``Segmenter.detect_layers`` is
actually called. This keeps the rest of the app usable on machines without
GPU/torch (e.g. when the user only wants the polygon / manual-row fallback).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# === Model checkpoint ===

# === Model registry ===
#
# Two model families are supported:
#
# 1. **Vanilla SAM** — Meta's original Segment Anything Model, trained on the
#    SA-1B dataset of natural images. Works but is not specialised for
#    microscopy; may give poor masks on TEM cross-sections.
#
# 2. **Micro-SAM** — Fine-tuned SAM variants from EMBL / computational-cell-
#    analytics, trained on microscopy data (electron microscopy, light
#    microscopy, histopathology). For TEM cross-sections, the EM-organelles
#    and LM (general light microscopy) variants tend to give better masks
#    than vanilla SAM.
#
# Checkpoint URLs are sourced from micro-sam's util.py:
# https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/util.py

MODEL_REGISTRY: dict[str, dict] = {
    # --- Vanilla SAM (Meta, natural images) ---
    "sam_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "description": "Vanilla SAM ViT-B (Meta, natural images). ~350 MB.",
        "family": "vanilla",
    },
    "sam_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "model_type": "vit_l",
        "description": "Vanilla SAM ViT-L (Meta, natural images). ~1.2 GB. More accurate but slower.",
        "family": "vanilla",
    },
    # --- Micro-SAM (EMBL, microscopy fine-tuned) ---
    # EM organelles: fine-tuned on electron microscopy (mitochondria, etc.)
    "microsam_vit_b_em": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b.pt",
        "filename": "microsam_vit_b_em_organelles.pt",
        "model_type": "vit_b",
        "description": "Micro-SAM ViT-B EM-organelles (fine-tuned on electron microscopy). ~350 MB. Recommended for TEM.",
        "family": "microsam",
    },
    "microsam_vit_l_em": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1.2/files/vit_l.pt",
        "filename": "microsam_vit_l_em_organelles.pt",
        "model_type": "vit_l",
        "description": "Micro-SAM ViT-L EM-organelles (fine-tuned on electron microscopy). ~1.2 GB. More accurate but slower.",
        "family": "microsam",
    },
    # LM (general light microscopy): fine-tuned on diverse light microscopy data.
    "microsam_vit_b_lm": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt",
        "filename": "microsam_vit_b_lm.pt",
        "model_type": "vit_b",
        "description": "Micro-SAM ViT-B LM (fine-tuned on light microscopy). ~350 MB. Alternative for TEM.",
        "family": "microsam",
    },
    "microsam_vit_l_lm": {
        "url": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1.2/files/vit_l.pt",
        "filename": "microsam_vit_l_lm.pt",
        "model_type": "vit_l",
        "description": "Micro-SAM ViT-L LM (fine-tuned on light microscopy). ~1.2 GB. More accurate but slower.",
        "family": "microsam",
    },
}

# Default model: Micro-SAM ViT-B LM (fine-tuned on microscopy — better
# TEM-layer masks than vanilla SAM, same ~350 MB size class as sam_vit_b).
DEFAULT_MODEL_KEY = "microsam_vit_b_lm"

# Default cache directory for checkpoints.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tem_grain_analyzer"


def get_model_info(model_key: str) -> dict:
    """Return the model registry entry for ``model_key``.

    Raises ``KeyError`` if the key is not in ``MODEL_REGISTRY``.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key: {model_key!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key]


def list_models() -> list[str]:
    """Return the list of available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_checkpoint_path(model_key: str, cache_dir: Path | None = None) -> Path:
    """Return the local cache path for a model's checkpoint (without downloading)."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    info = get_model_info(model_key)
    return cache_dir / info["filename"]


def is_checkpoint_cached(model_key: str, cache_dir: Path | None = None) -> bool:
    """Check whether a model's checkpoint is already downloaded and non-empty.

    Use this to warn the user before triggering a download (e.g. when switching
    to a new model in the UI).
    """
    path = get_checkpoint_path(model_key, cache_dir)
    return path.exists() and path.stat().st_size > 0


def get_model_size_mb(model_key: str) -> float:
    """Return the approximate checkpoint size in MB (for display purposes)."""
    # ViT-B ~ 350 MB, ViT-L ~ 1200 MB, ViT-H ~ 2500 MB.
    info = get_model_info(model_key)
    sizes = {"vit_b": 350.0, "vit_l": 1200.0, "vit_h": 2500.0, "vit_t": 150.0}
    return sizes.get(info["model_type"], 350.0)


# Legacy aliases (kept for backward compatibility with older code).
DEFAULT_CHECKPOINT_URL = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["url"]
DEFAULT_CHECKPOINT_FILENAME = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["filename"]


def ensure_checkpoint(url: str = DEFAULT_CHECKPOINT_URL,
                       cache_dir: Path | None = None,
                       filename: str | None = None,
                       progress_callback=None) -> Path:
    """Download a checkpoint if not already cached. Returns its path.

    Parameters
    ----------
    url : str
        Download URL.
    cache_dir : Path, optional
        Cache directory. Defaults to ``~/.cache/tem_grain_analyzer``.
    filename : str, optional
        Local filename. Defaults to the basename of ``url``.
    progress_callback : callable, optional
        Called with ``(bytes_downloaded, total_bytes)`` during download.

    Returns
    -------
    Path
        Path to the cached checkpoint file.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or os.path.basename(url)
    checkpoint_path = cache_dir / filename

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        return checkpoint_path

    # Download with progress reporting.
    print(f"[SAM] Downloading checkpoint from {url} ...")
    print(f"[SAM] Saving to {checkpoint_path}")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if progress_callback is not None:
            progress_callback(block_num * block_size, total_size)
        elif block_num % 100 == 0:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = downloaded * 100 / total_size
                print(f"\r[SAM] {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.1f}%)",
                      end="", flush=True)
            else:
                print(f"\r[SAM] {downloaded / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(checkpoint_path), reporthook=_reporthook)
    print()
    return checkpoint_path


# === Data classes ===

@dataclass
class LayerCandidate:
    """A candidate layer mask found by SAM."""
    mask: np.ndarray             # bool, shape (H, W)
    area_px: int
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    centroid: tuple[float, float]    # (row, col)
    mean_intensity: float
    std_intensity: float
    eccentricity: float
    # Approximate polygon (Nx2 array of (row, col) vertices) for display.
    polygon: Optional[np.ndarray] = None
    # SAM's predicted IoU (confidence) for this mask.
    predicted_iou: float = 0.0
    # Stability score (SAM's mask-quality metric).
    stability_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_px": int(self.area_px),
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "mean_intensity": float(self.mean_intensity),
            "std_intensity": float(self.std_intensity),
            "eccentricity": float(self.eccentricity),
            "predicted_iou": float(self.predicted_iou),
            "stability_score": float(self.stability_score),
        }


@dataclass
class SAMResult:
    """Result of SAM layer detection."""
    candidates: list[LayerCandidate] = field(default_factory=list)
    n_total_masks: int = 0
    n_candidates: int = 0
    checkpoint_path: str = ""
    model_type: str = "vit_b"
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# === Availability check ===

def is_available() -> bool:
    """Return True if torch + segment_anything are importable."""
    try:
        import torch  # noqa: F401
        import segment_anything  # noqa: F401
        return True
    except ImportError:
        return False


def missing_dependencies() -> list[str]:
    """Return the list of missing pip packages for SAM."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import segment_anything  # noqa: F401
    except ImportError:
        missing.append("git+https://github.com/facebookresearch/segment-anything.git")
    return missing


# === Segmenter ===

class Segmenter:
    """Lazy SAM segmenter.

    The model is loaded on first call to ``detect_layers``. Subsequent calls
    reuse the loaded model.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY,
                 checkpoint_path: str | Path | None = None,
                 model_type: str | None = None,
                 device: str | None = None):
        """
        Parameters
        ----------
        model_key : str
            Key into ``MODEL_REGISTRY`` (e.g. ``"sam_vit_b"``, ``"microsam_vit_b_em"``).
            Used to resolve the checkpoint URL and model type. Ignored if
            ``checkpoint_path`` is provided.
        checkpoint_path : str | Path | None
            Explicit path to a SAM checkpoint. If provided, overrides
            ``model_key``. If None, the checkpoint is downloaded from the URL
            in ``MODEL_REGISTRY[model_key]``.
        model_type : str, optional
            SAM model type: "vit_b", "vit_l", or "vit_h". Must match the
            checkpoint. If None, taken from ``MODEL_REGISTRY[model_key]``.
        device : str, optional
            Torch device ("cuda", "cpu", or None for auto). Defaults to "cuda"
            if available, else "cpu".
        """
        self.model_key = model_key
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        # Resolve model_type from registry if not explicitly provided.
        if model_type is None:
            model_info = get_model_info(model_key)
            model_type = model_info["model_type"]
        self.model_type = model_type
        self.device = device
        self._sam = None
        self._predictor = None

    def _ensure_model(self) -> None:
        """Download checkpoint (if needed) and load the SAM model."""
        if self._sam is not None:
            return
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise RuntimeError(
                "SAM dependencies are not installed. Install with:\n"
                "  pip install torch torchvision\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                f"Original error: {e}"
            ) from e

        # Resolve checkpoint.
        if self.checkpoint_path is not None:
            ckpt = Path(self.checkpoint_path)
            if not ckpt.exists():
                raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
        else:
            # Download from registry.
            model_info = get_model_info(self.model_key)
            ckpt = ensure_checkpoint(
                url=model_info["url"],
                filename=model_info["filename"],
            )
        self.checkpoint_path = str(ckpt)

        # Resolve device.
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM] Loading model {self.model_key} ({self.model_type}) "
              f"from {ckpt} on {device} ...")

        sam = sam_model_registry[self.model_type](checkpoint=ckpt)
        sam.to(device=device)
        self._sam = sam
        self._device = device

    def detect_layers(
        self,
        image: np.ndarray,
        min_area_frac: float = 0.01,
        max_area_frac: float = 0.95,
        min_eccentricity: float = 0.85,
        max_std_intensity: float = 60.0,
        points_per_side: int = 24,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        max_candidates: int = 10,
        max_image_dim: int = 1024,
    ) -> SAMResult:
        """Run SAM automatic mask generation and filter for layer candidates.

        **Memory management:** SAM's image encoder internally resizes to 1024×1024,
        so passing a larger image wastes memory without improving mask quality.
        This method automatically downscales the input to ``max_image_dim`` (default
        1024 px) before SAM, runs SAM on the small image, then upscales the masks
        back to the original resolution. This reduces peak memory by ~16× for a
        4000-px image (from ~7.6 GB to ~500 MB) with no loss in mask quality.

        Parameters
        ----------
        image : np.ndarray
            2-D grayscale uint8 TEM image.
        min_area_frac, max_area_frac : float
            Min/max candidate area as a fraction of the image area.
        min_eccentricity : float
            Minimum eccentricity (0=circle, 1=line). Layers are elongated →
            high eccentricity. Default 0.85.
        max_std_intensity : float
            Maximum within-mask intensity std. Layers tend to be relatively
            uniform in brightness; very textured masks are likely not layers.
        points_per_side : int
            SAM grid density. Default 24 (reduced from SAM's default 32 to save
            memory). More = more masks but slower and more memory.
        pred_iou_thresh : float
            SAM's predicted-IoU threshold for accepting a mask.
        stability_score_thresh : float
            SAM's stability-score threshold.
        max_candidates : int
            Maximum number of candidates to return (sorted by area descending).
        max_image_dim : int
            Maximum image dimension (px) passed to SAM. The image is downscaled
            to fit within this; masks are upscaled back to original resolution.
            Default 1024 (matches SAM's internal encoder size).

        Returns
        -------
        SAMResult
        """
        if image.ndim != 2:
            raise ValueError("Expected a 2-D grayscale image.")

        try:
            self._ensure_model()
        except Exception as e:
            return SAMResult(error=str(e))

        try:
            from segment_anything import SamAutomaticMaskGenerator
            from skimage import measure
        except ImportError as e:
            return SAMResult(error=f"Missing dependency: {e}")

        h_orig, w_orig = image.shape

        # === Downscale to max_image_dim to save memory ===
        # SAM's encoder resizes to 1024×1024 internally, so passing a larger
        # image wastes memory. We downscale, run SAM, then upscale masks back.
        scale = min(1.0, float(max_image_dim) / max(h_orig, w_orig))
        if scale < 1.0:
            h_small = int(h_orig * scale)
            w_small = int(w_orig * scale)
            img_small = cv2.resize(image, (w_small, h_small), interpolation=cv2.INTER_AREA)
            print(f"[SAM] Downscaled {w_orig}×{h_orig} → {w_small}×{h_small} "
                  f"(scale={scale:.3f}) to save memory")
        else:
            img_small = image
            scale = 1.0

        # SAM expects a 3-channel RGB image.
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)

        # Run SAM on the downscaled image.
        # min_mask_region_area is in the small-image's pixel units.
        small_area = float(img_small.shape[0] * img_small.shape[1])
        mask_generator = SamAutomaticMaskGenerator(
            self._sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=int(small_area * min_area_frac * 0.1),
        )
        print(f"[SAM] Generating masks on {img_small.shape[1]}×{img_small.shape[0]} "
              f"(points_per_side={points_per_side}, this can take 5–30 s) ...")
        masks_data = mask_generator.generate(img_rgb)
        n_total = len(masks_data)
        print(f"[SAM] Generated {n_total} raw masks")

        # Filter for layer candidates and upscale masks back to original resolution.
        total_area_orig = float(h_orig * w_orig)
        candidates: list[LayerCandidate] = []

        for md in masks_data:
            mask_small = md["segmentation"]
            area_small = int(md["area"])

            # Upscale mask to original resolution if we downscaled.
            if scale < 1.0:
                mask_full = cv2.resize(
                    mask_small.astype(np.uint8),
                    (w_orig, h_orig),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                # Recompute area in original pixels.
                area_full = int(mask_full.sum())
            else:
                mask_full = mask_small
                area_full = area_small

            area_frac = area_full / total_area_orig
            if area_frac < min_area_frac or area_frac > max_area_frac:
                continue

            # Compute region properties on the full-resolution mask.
            region_props = None
            for rp in measure.regionprops(mask_full.astype(np.int32)):
                if rp.label == 1:
                    region_props = rp
                    break
            if region_props is None:
                continue

            eccentricity = float(region_props.eccentricity)
            if eccentricity < min_eccentricity:
                continue

            # Intensity statistics within the mask (on original image).
            masked_pixels = image[mask_full]
            mean_i = float(masked_pixels.mean())
            std_i = float(masked_pixels.std())
            if std_i > max_std_intensity:
                continue

            bbox = tuple(int(x) for x in region_props.bbox)
            cy, cx = region_props.centroid
            centroid = (float(cy), float(cx))

            # Approximate polygon from the full-resolution mask.
            contours, _ = cv2.findContours(mask_full.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)
                polygon = approx.reshape(-1, 2)[:, [1, 0]].astype(np.float32)

            candidates.append(LayerCandidate(
                mask=mask_full,
                area_px=area_full,
                bbox=bbox,
                centroid=centroid,
                mean_intensity=mean_i,
                std_intensity=std_i,
                eccentricity=eccentricity,
                polygon=polygon,
                predicted_iou=float(md.get("predicted_iou", 0.0)),
                stability_score=float(md.get("stability_score", 0.0)),
            ))

        # Sort by area descending and take top N.
        candidates.sort(key=lambda c: c.area_px, reverse=True)
        candidates = candidates[:max_candidates]

        return SAMResult(
            candidates=candidates,
            n_total_masks=n_total,
            n_candidates=len(candidates),
            checkpoint_path=self.checkpoint_path,
            model_type=self.model_type,
        )

    def detect_layer_at_point(
        self,
        image: np.ndarray,
        point: tuple[int, int],
        max_image_dim: int = 1024,
        multimask_output: bool = True,
    ) -> SAMResult:
        """Segment the layer at a single point using SAM's predictor (prompt mode).

        This is the **interactive** alternative to ``detect_layers``: instead of
        generating masks for the whole image automatically, the user clicks on
        the layer they want, and SAM segments exactly that region. This is more
        reliable than automatic mode for TEM images where the "right" layer is
        one among many candidates.

        Parameters
        ----------
        image : np.ndarray
            2-D grayscale uint8 TEM image.
        point : (row, col)
            The click location in original-image coordinates.
        max_image_dim : int
            SAM's predictor works at the image's native resolution, but we still
            downscale to save memory on the image-encoding step. The click point
            is scaled accordingly. Default 1024.
        multimask_output : bool
            If True, SAM returns 3 candidate masks (for ambiguous prompts);
            we pick the one with the highest predicted IoU. If False, SAM
            returns a single mask.

        Returns
        -------
        SAMResult
            With a single candidate (the best mask at the click point), or
            ``error`` set if something went wrong.
        """
        if image.ndim != 2:
            raise ValueError("Expected a 2-D grayscale image.")
        row, col = point
        h_orig, w_orig = image.shape

        try:
            self._ensure_model()
        except Exception as e:
            return SAMResult(error=str(e))

        try:
            from segment_anything import SamPredictor
            from skimage import measure
            import torch
        except ImportError as e:
            return SAMResult(error=f"Missing dependency: {e}")

        # Downscale for the encoder (saves memory).
        scale = min(1.0, float(max_image_dim) / max(h_orig, w_orig))
        if scale < 1.0:
            h_small = int(h_orig * scale)
            w_small = int(w_orig * scale)
            img_small = cv2.resize(image, (w_small, h_small), interpolation=cv2.INTER_AREA)
            # Scale the click point to the small image.
            pt_row = row * scale
            pt_col = col * scale
            print(f"[SAM] Downscaled {w_orig}×{h_orig} → {w_small}×{h_small} "
                  f"(scale={scale:.3f}), click point → ({pt_col:.0f}, {pt_row:.0f})")
        else:
            img_small = image
            pt_row = float(row)
            pt_col = float(col)
            scale = 1.0

        # SAM expects a 3-channel RGB image.
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)

        # Use SamPredictor (prompt mode) instead of SamAutomaticMaskGenerator.
        if self._predictor is None:
            self._predictor = SamPredictor(self._sam)
        print(f"[SAM] Encoding image (first call takes ~5 s, cached after) ...")
        self._predictor.set_image(img_rgb)

        # Run prediction with the point prompt.
        # SamPredictor expects point_coords as (N, 2) of (col, row) = (x, y).
        point_coords = np.array([[pt_col, pt_row]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1 = foreground
        print(f"[SAM] Predicting mask at click point ...")
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

        # Pick the mask with the highest score.
        best_idx = int(np.argmax(scores))
        mask_small = masks[best_idx]
        score = float(scores[best_idx])
        print(f"[SAM] Got {len(masks)} masks, best score = {score:.3f}")

        # Upscale mask to original resolution.
        if scale < 1.0:
            mask_full = cv2.resize(
                mask_small.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_full = mask_small

        area_full = int(mask_full.sum())
        if area_full == 0:
            return SAMResult(error="SAM returned an empty mask at the click point.")

        # Compute region properties.
        region_props = None
        for rp in measure.regionprops(mask_full.astype(np.int32)):
            if rp.label == 1:
                region_props = rp
                break
        if region_props is None:
            return SAMResult(error="No region found in the predicted mask.")

        eccentricity = float(region_props.eccentricity)
        bbox = tuple(int(x) for x in region_props.bbox)
        cy, cx = region_props.centroid
        centroid = (float(cy), float(cx))

        # Intensity statistics.
        masked_pixels = image[mask_full]
        mean_i = float(masked_pixels.mean())
        std_i = float(masked_pixels.std())

        # Polygon.
        contours, _ = cv2.findContours(mask_full.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            polygon = approx.reshape(-1, 2)[:, [1, 0]].astype(np.float32)

        candidate = LayerCandidate(
            mask=mask_full,
            area_px=area_full,
            bbox=bbox,
            centroid=centroid,
            mean_intensity=mean_i,
            std_intensity=std_i,
            eccentricity=eccentricity,
            polygon=polygon,
            predicted_iou=score,
            stability_score=score,  # reuse score as a proxy
        )

        return SAMResult(
            candidates=[candidate],
            n_total_masks=1,
            n_candidates=1,
            checkpoint_path=self.checkpoint_path,
            model_type=self.model_type,
        )


# === Convenience functions ===

def detect_layers(image: np.ndarray, model_key: str = DEFAULT_MODEL_KEY,
                   **kwargs) -> SAMResult:
    """One-shot automatic layer detection. Creates a Segmenter, runs it, returns the result.

    Parameters
    ----------
    image : np.ndarray
    model_key : str
        Key into ``MODEL_REGISTRY`` (e.g. ``"sam_vit_b"``, ``"microsam_vit_b_em"``).
    **kwargs : See ``Segmenter.detect_layers`` for parameters.
    """
    seg = Segmenter(model_key=model_key)
    return seg.detect_layers(image, **kwargs)


def detect_layer_at_point(image: np.ndarray, point: tuple[int, int],
                            model_key: str = DEFAULT_MODEL_KEY,
                            **kwargs) -> SAMResult:
    """One-shot point-prompt layer detection.

    Parameters
    ----------
    image : np.ndarray
    point : (row, col)
    model_key : str
        Key into ``MODEL_REGISTRY``.
    **kwargs : See ``Segmenter.detect_layer_at_point`` for parameters.
    """
    seg = Segmenter(model_key=model_key)
    return seg.detect_layer_at_point(image, point, **kwargs)


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.005) -> np.ndarray | None:
    """Convert a boolean mask to a simplified polygon (Nx2 array of (row, col)).

    Returns None if no contour is found.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    # Convert (col, row) → (row, col).
    return approx.reshape(-1, 2)[:, [1, 0]].astype(np.float32)
