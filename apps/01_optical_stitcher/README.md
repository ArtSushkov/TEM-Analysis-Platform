# 01 — Optical Stitcher

A Streamlit app for stitching a series of optical microscope images into a single panorama.

> **Part of TEM Analysis Platform** — see the root [README](../../README.md) for context and setup.

## Purpose

Preparing cross-sections for TEM often requires inspecting the disk surface on an optical microscope. A single frame usually does not cover the entire region of interest, so a series of overlapping images is captured. This app automatically stitches them into a single panorama.

## Features

- **Automatic relative position estimation** between adjacent images (SIFT + RANSAC)
- **Pair Review tab** — verify every pair visually and adjust positions by hand
- **No-overlap pairs** — mark accidentally non-overlapping shots with a checkbox;
  they are placed side by side with a visible black gap on the panorama
- **Robust to dark images** via CLAHE preprocessing
- **Multi-band blending** — invisible seams even with blurred tile edges
- **Fallback strategies**: phase correlation and NCC scan in case SIFT fails
- **TIFF support (including LZW compression)**, PNG, JPEG
- **Full diagnostics**: method, match/inlier counts, estimated shifts, position map

## Installation

This app uses the **shared virtual environment** at the project root. There is no app-specific `requirements.txt` — see the root [README](../../README.md#quick-start) for setup instructions.

Quick recap:

```bash
# From the project root
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate               # Linux/macOS
pip install -r requirements.txt
```

## Running

From the project root, with the virtual environment activated:

```bash
streamlit run apps/01_optical_stitcher/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

### 1. Prepare Images

Filenames must end with a sequential index, separated by an underscore:

```
my_sample_1.tif
my_sample_2.tif
my_sample_3.tif
...
my_sample_5.tif
```

Supported formats: `.tif`, `.tiff` (including LZW), `.png`, `.jpg`, `.jpeg`, `.bmp`.

### 2. Scale Bar (Important!)

TIFF files from microscope cameras typically do **not** store scale information in metadata.
To preserve the scale bar in the final panorama, embed it visually into one of the images
via the microscope software:

- **First image of the series**: place the scale bar in the **bottom-left corner**.
  This part of the first tile is never overlapped by the second tile, so the scale bar
  remains intact in the final panorama.
- **OR last image of the series**: place the scale bar in the **bottom-right corner**.

Avoid placing the scale bar in the right part of intermediate tiles — it will be blended away.

### 3. Upload

In the sidebar, choose the upload method:
- **Multiple files** — select all files at once
- **ZIP archive** — upload the whole archive

### 4. Configure Settings (optional)

The sidebar exposes settings for feature detection, preprocessing, and blending.
See the in-app **Help** tab for a detailed explanation of each setting.

### 5. Pair Review (manual alignment, recommended for hand-captured rows)

Open the **Pair Review** tab for an interactive workflow that does not require
a precise `Expected overlap` prior:

1. **Estimate** — fast automatic estimate of every adjacent pair on downscaled copies.
2. **Review** — each pair is shown as an overlay (Blend / Checkerboard / Difference)
   with tile outlines A (yellow) and B (cyan). Tick **Position is correct** for good pairs.
   If two neighbouring shots were captured **without any overlap**, tick
   **No overlap (black gap)** instead — the pair is placed side by side with a visible
   black band on the panorama (gap width configurable in Step 2; unticking restores
   the previous position). Such a pair counts as approved.
3. **Adjust** — unticked pairs get nudge buttons (1–100 px steps) and numeric dx/dy
   inputs with a live overlay. Increasing `dx` moves the tiles apart (less overlap).
4. **Stitch** — the panorama is blended using exactly the reviewed positions;
   no algorithm can override what you approved.

Positions, approvals and no-overlap marks can be exported/imported as JSON, so a
review session survives browser refreshes.

### 6. Run

Click **Stitch panorama** (Stitcher tab) or **Stitch panorama with these positions**
(Pair Review tab). Progress is shown in real time.

### 7. Result

- Preview of the stitched panorama
- Diagnostics table for each pair (method, dx/dy, rotation, matches, inliers)
- Tile position map (absolute coordinates on the canvas)
- Download buttons:
  - **Panorama** in the chosen format (PNG / TIFF / JPEG)
  - **CSV** with all tile positions

## How It Works

```
[Load N images, sort by _N]
    |
[CLAHE preprocessing — boost contrast]
    |
[Pairwise stitching i <-> i+1]:
   1. SIFT detector + FLANN matcher
   2. Lowe ratio test -> good matches
   3. RANSAC + estimateAffinePartial2D
   4. Fallback: phase correlation
   5. Fallback: NCC scan (if all else fails)
    |
[Compute cumulative absolute positions]
    |
[Multi-band blending]:
   - Laplacian pyramid (5 levels)
   - Weighted average per level
   - Gain compensation between tiles
    |
[Panorama + diagnostics]
```

## Limitations

- **1D row only** — 2D grids are not supported yet
- **Up to ~100 tiles** — for larger sets, split into multiple panoramas
- **Translation + small rotation** — significant perspective distortion is not handled

## Project Structure

```
01_optical_stitcher/
├── app.py                  # Streamlit UI (Stitcher / Pair Review / Help tabs)
├── README.md
└── src/
    ├── __init__.py
    ├── io_utils.py         # Loading + _N parsing + sorting
    ├── preprocessing.py    # CLAHE + sharpness map
    ├── stitching.py        # SIFT+RANSAC + fallbacks
    ├── pair_review.py      # Review-scale estimation + overlay previews + JSON session
    ├── blending.py         # Multi-band blending (memory-optimized)
    └── pipeline.py         # Orchestration (+ reviewed-position overrides)
```

## Troubleshooting

See the in-app **Help** tab for a full troubleshooting guide. The most common issues:

- **"Unable to allocate ... GiB" (OOM)**: reduce `Laplacian pyramid levels` to 3-4
- **Dark zone at the panorama edge**: ensure `Use focus-weighted mask` is OFF (default)
- **Visible seam**: increase `Laplacian pyramid levels` to 6-7 and `Gain compensation` to 1.0
- **All methods failed**: increase `SIFT features` to 10000, lower `Lowe ratio threshold` to 0.7

## License

MIT — see the root [LICENSE](../../LICENSE).
