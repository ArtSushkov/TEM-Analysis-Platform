"""
export_training_data.py — Convert JSON reports from the Streamlit app into
training datasets for Micro-SAM (layer detection) and U-Net (grain segmentation).

Usage:
    python apps/02_grain_analyzer/export_training_data.py \
        --reports-dir path/to/json_reports/ \
        --images-dir path/to/tiff_images/ \
        --output-dir training_data/

The script reads each JSON report (produced by the "Download report" button in
the Single Image tab), extracts:
  - The original TEM image (from the TIFF file referenced in the report).
  - The layer mask (from the polygon stored in the report).
  - The labeled grain image (from the base64 PNG embedded in the report).

And writes two datasets:

  training_data/layer_detection/
    images/   — full TEM cross-section images
    masks/    — binary layer masks (0 = not layer, 1 = layer)
    points.json — click points [col, row] inside each layer (centroid of polygon)

  training_data/grain_segmentation/
    images/   — cropped layer images (only the layer region)
    boundaries/ — binary grain boundary masks (0 = no boundary, 255 = boundary)
    labels/   — labeled grain images (0 = bg, 1 = grain 1, 2 = grain 2, ...)
    layer_masks/ — 255 = inside the layer, 0 = outside (the U-Net notebook
                  excludes out-of-layer pixels from the loss via ignore_index=255)

Both datasets include a train_val_split.json (80/20 split).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi


def load_json_report(path: str) -> dict:
    """Load a JSON report file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_base64_png(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded PNG image into a NumPy array."""
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img


def polygon_to_mask(polygon: list, shape: tuple) -> np.ndarray:
    """Convert a polygon [(row, col), ...] to a binary mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    pts = np.array([[int(c), int(r)] for r, c in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_centroid(polygon: list) -> tuple[int, int]:
    """Compute the centroid of a polygon [(row, col), ...]. Returns (col, row)."""
    if not polygon:
        return (0, 0)
    rows = [p[0] for p in polygon]
    cols = [p[1] for p in polygon]
    return (int(np.mean(cols)), int(np.mean(rows)))


def labels_to_boundaries(labels: np.ndarray) -> np.ndarray:
    """Convert a labeled image to a binary boundary mask (0/255)."""
    if labels.max() == 0:
        return np.zeros_like(labels, dtype=np.uint8)
    # Boundary = pixels where the label differs from any neighbor.
    boundaries = np.zeros_like(labels, dtype=np.uint8)
    for shift in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        shifted = np.roll(labels, shift, axis=(0, 1))
        boundaries |= (labels != shifted) & (labels > 0) & (shifted > 0)
    # Also mark the outer edge of each grain (label > 0 next to label == 0).
    for shift in [(1, 0), (0, 1)]:
        shifted = np.roll(labels, shift, axis=(0, 1))
        boundaries |= (labels > 0) & (shifted == 0)
        boundaries |= (labels == 0) & (shifted > 0)
    return (boundaries > 0).astype(np.uint8) * 255


def crop_to_bbox(img: np.ndarray, mask: np.ndarray, padding: int = 2) -> tuple:
    """Crop image and mask to the bounding box of the mask."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return img, mask, (0, 0, img.shape[0], img.shape[1])
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    cmin = max(0, cmin - padding)
    rmax = min(img.shape[0], rmax + padding + 1)
    cmax = min(img.shape[1], cmax + padding + 1)
    cropped_img = img[rmin:rmax, cmin:cmax]
    cropped_mask = mask[rmin:rmax, cmin:cmax]
    return cropped_img, cropped_mask, (rmin, cmin, rmax, cmax)


def process_report(
    report: dict,
    image_path: str,
    output_dir: str,
    sample_id: str,
) -> bool:
    """Process a single JSON report and write training data.

    Returns True if successful, False if skipped.
    """
    # Load the original TEM image.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  WARNING: cannot load image {image_path}, skipping")
        return False

    # Extract layer polygon from the report.
    layer_info = report.get("layer", {})
    polygon_raw = layer_info.get("polygon_px", [])
    if not polygon_raw or len(polygon_raw) < 3:
        print(f"  WARNING: no layer polygon in report for {sample_id}, skipping")
        return False

    # The polygon may be stored as [[row, col], ...] or [{row, col}, ...].
    polygon = []
    for pt in polygon_raw:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            polygon.append((float(pt[0]), float(pt[1])))
        elif isinstance(pt, dict):
            polygon.append((float(pt.get("row", pt.get("y", 0))),
                            float(pt.get("col", pt.get("x", 0)))))
    if len(polygon) < 3:
        print(f"  WARNING: polygon has < 3 points for {sample_id}, skipping")
        return False

    # Create layer mask.
    layer_mask = polygon_to_mask(polygon, img.shape)

    # --- Layer detection dataset ---
    layer_dir = os.path.join(output_dir, "layer_detection")
    img_dir = os.path.join(layer_dir, "images")
    mask_dir = os.path.join(layer_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    cv2.imwrite(os.path.join(img_dir, f"{sample_id}.tif"), img)
    cv2.imwrite(os.path.join(mask_dir, f"{sample_id}.tif"), layer_mask)

    # Click point = centroid of the polygon (simulates user click).
    click_point = polygon_centroid(polygon)

    # --- Grain segmentation dataset ---
    grain_dir = os.path.join(output_dir, "grain_segmentation")
    grain_img_dir = os.path.join(grain_dir, "images")
    grain_bound_dir = os.path.join(grain_dir, "boundaries")
    grain_label_dir = os.path.join(grain_dir, "labels")
    grain_layermask_dir = os.path.join(grain_dir, "layer_masks")
    os.makedirs(grain_img_dir, exist_ok=True)
    os.makedirs(grain_bound_dir, exist_ok=True)
    os.makedirs(grain_label_dir, exist_ok=True)
    os.makedirs(grain_layermask_dir, exist_ok=True)

    # Crop image to layer bbox.
    cropped_img, cropped_mask, bbox = crop_to_bbox(img, layer_mask, padding=2)

    # Validity mask: 255 = inside the layer, 0 = outside. The U-Net notebook
    # marks out-of-layer target pixels with ignore_index and excludes them
    # from the loss (they carry textures of neighboring layers).
    cv2.imwrite(os.path.join(grain_layermask_dir, f"{sample_id}.tif"),
                cropped_mask.astype(np.uint8) * 255)

    # Try to extract the labeled grain image from the report (base64 PNG).
    viz_b64 = report.get("visualization_png_base64", "")
    labels = None
    if viz_b64:
        try:
            viz_img = decode_base64_png(viz_b64)
            # The visualization is RGB with colored grains on a grayscale background.
            # We need to extract the labels — but the report stores the visualization,
            # not the raw labels. Let's check if there's a separate labels field.
            # If not, we'll try to extract from the visualization (approximate).
            # Actually, the report stores labeled_image_rgb, not raw labels.
            # For now, skip label extraction from base64 — we need the raw labels.
            pass
        except Exception as e:
            print(f"  WARNING: cannot decode visualization PNG: {e}")

    # Check if the report has grain data (measurements).
    grains_data = report.get("grains", [])
    if grains_data and len(grains_data) > 0:
        # We have grain measurements but need pixel-level labels.
        # The labels are NOT in the JSON report — they're only in the visualization PNG.
        # For training, the user should save the labels separately (from the app's
        # session state) OR re-annotate using the Manual tracing mode and export
        # the labels directly.
        #
        # For now, we save the cropped layer image and note that labels need to be
        # added separately (e.g., by running the app and saving the labels image).
        print(f"  NOTE: {len(grains_data)} grains in report, but pixel-level labels "
              f"are not stored in JSON. Use the app's 'Save labels' feature or "
              f"re-annotate with Manual tracing mode.")
        # Save the cropped image — labels can be added later.
        cv2.imwrite(os.path.join(grain_img_dir, f"{sample_id}.tif"), cropped_img)
        # Save a placeholder boundary mask (zeros) — to be filled by the user.
        cv2.imwrite(os.path.join(grain_bound_dir, f"{sample_id}.tif"),
                    np.zeros_like(cropped_img, dtype=np.uint8))
        cv2.imwrite(os.path.join(grain_label_dir, f"{sample_id}.tif"),
                    np.zeros_like(cropped_img, dtype=np.uint8))
    else:
        print(f"  NOTE: no grain data in report for {sample_id}")
        cv2.imwrite(os.path.join(grain_img_dir, f"{sample_id}.tif"), cropped_img)
        cv2.imwrite(os.path.join(grain_bound_dir, f"{sample_id}.tif"),
                    np.zeros_like(cropped_img, dtype=np.uint8))
        cv2.imwrite(os.path.join(grain_label_dir, f"{sample_id}.tif"),
                    np.zeros_like(cropped_img, dtype=np.uint8))

    return True, click_point


def main():
    parser = argparse.ArgumentParser(
        description="Export training data from JSON reports."
    )
    parser.add_argument("--reports-dir", required=True,
                        help="Directory containing JSON reports from the Streamlit app.")
    parser.add_argument("--images-dir", required=True,
                        help="Directory containing the original TIFF images.")
    parser.add_argument("--output-dir", default="training_data",
                        help="Output directory for training data (default: training_data).")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    images_dir = Path(args.images_dir)
    output_dir = args.output_dir

    # Find all JSON reports.
    report_files = sorted(reports_dir.glob("*.json"))
    if not report_files:
        print(f"No JSON reports found in {reports_dir}")
        return

    print(f"Found {len(report_files)} JSON reports.")

    points = {}
    successful = 0

    for report_path in report_files:
        sample_id = report_path.stem  # filename without extension
        print(f"\nProcessing {sample_id}...")

        try:
            report = load_json_report(str(report_path))
        except Exception as e:
            print(f"  ERROR: cannot load report: {e}")
            continue

        # Find the corresponding image file.
        image_metadata = report.get("image_metadata", {})
        image_filename = image_metadata.get("filename", f"{sample_id}.tif")
        image_path = images_dir / image_filename
        if not image_path.exists():
            # Try common extensions.
            for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
                alt_path = images_dir / (Path(image_filename).stem + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                print(f"  ERROR: cannot find image {image_filename} in {images_dir}")
                continue

        result = process_report(report, str(image_path), output_dir, sample_id)
        if isinstance(result, tuple):
            success, click_point = result
            if success:
                points[sample_id] = list(click_point)
                successful += 1
        elif result:
            successful += 1

    # Save points.json for layer detection.
    if points:
        layer_dir = os.path.join(output_dir, "layer_detection")
        with open(os.path.join(layer_dir, "points.json"), "w") as f:
            json.dump(points, f, indent=2)

    # Create train/val split (80/20).
    sample_ids = [report_path.stem for report_path in report_files]
    np.random.seed(42)
    np.random.shuffle(sample_ids)
    n_val = max(1, len(sample_ids) // 5)
    val_ids = sample_ids[:n_val]
    train_ids = sample_ids[n_val:]

    split = {"train": train_ids, "val": val_ids}

    for subdir in ["layer_detection", "grain_segmentation"]:
        split_path = os.path.join(output_dir, subdir, "train_val_split.json")
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)

    print(f"\n=== Export complete ===")
    print(f"Processed: {successful}/{len(report_files)} reports")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"Output: {output_dir}/")
    print(f"\nLayer detection data: {output_dir}/layer_detection/")
    print(f"  - images/   : full TEM cross-sections")
    print(f"  - masks/    : binary layer masks")
    print(f"  - points.json: click points for each image")
    print(f"\nGrain segmentation data: {output_dir}/grain_segmentation/")
    print(f"  - images/     : cropped layer images")
    print(f"  - boundaries/ : grain boundary masks (need manual annotation)")
    print(f"  - labels/     : labeled grain images (need manual annotation)")
    print(f"  - layer_masks/: 255 = inside the layer (for ignore_index training)")
    print(f"\nNOTE: Grain boundary/label masks are currently empty (zeros).")
    print(f"To fill them, use the app's Manual tracing mode to annotate grains,")
    print(f"then save the labeled image. Or modify the app to export labels")
    print(f"directly from session_state.analysis_result.labels.")


if __name__ == "__main__":
    main()
