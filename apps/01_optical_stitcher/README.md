# 01 — Optical Stitcher

A Streamlit app for stitching a series of optical microscope images into a single panorama.

> **Part of TEM Analysis Platform** — see the root [README](../../README.md) for context and setup.

## Purpose

Preparing cross-sections for TEM often requires inspecting the disk surface on an optical microscope. A single frame usually does not cover the entire region of interest, so a series of overlapping images is captured. This app automatically stitches them into a single panorama.

## Features

- **Automatic relative position estimation** between adjacent images (SIFT + RANSAC)
- **Region of interest (ROI) crop** — auto-detect or manually correct a horizontal
  band on each tile (typically the ~1/4 of the frame that contains the visible
  cross-section of the TEM disk). The crop runs BEFORE SIFT/NCC/blending, giving
  ~4x speedup and ~4x smaller panorama on 20 MP frames. Per-tile offset
  correction in Pair Review when the operator shifted the disk vertically.
- **Pair Review tab** — verify every pair visually and adjust positions by hand
- **No-overlap pairs** — mark accidentally non-overlapping shots with a checkbox;
  they are placed side by side with a visible black gap on the panorama
- **Static side-by-side pair preview** — overlay views can hide a wrong dx/dy
  behind averaging; the new side-by-side mode shows both tiles in full next to
  each other with seam markers, so misalignment is immediately visible
- **Robust to dark images** via CLAHE preprocessing
- **Multi-band blending** — invisible seams even with blurred tile edges
- **Focus-priority blending (default)** — in the overlap zone, the tile with
  the highest Tenengrad sharpness score wins the high-frequency content; low
  frequencies are still averaged so brightness stays smooth. Use this when the
  operator focused each frame on a different depth and the goal is to see the
  sharpest detail rather than a smoothly-averaged picture
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

The sidebar exposes settings for feature detection, preprocessing, blending,
and region of interest (ROI). See the in-app **Help** tab for a detailed
explanation of each setting.

### 4b. Region of interest (ROI) — when only a horizontal strip of the frame matters

If the visible cross-section of the TEM disk occupies only a horizontal strip
of each frame (typically ~1/4 of the frame height — the rest is empty
microscope background), enable **Crop tiles to ROI band** in the sidebar's
**Region of interest** section:

- **Auto (reference tile 1) — same band on every tile** (default, recommended):
  the app detects the textured band on the FIRST tile (a window of the
  requested height, centred on the per-row Tenengrad energy centroid of the
  disk band — robust to dust specks) and applies the SAME band to every tile.
  You verify / correct it visually in Pair Review Step 3 (interactive vertical
  offset slider on tile 1; the band height is set by the sidebar
  "ROI height (fraction of tile height)" slider) or in the Stitcher's
  "ROI band preview" expander (same offset slider, synchronised). Because the
  band is identical
  on every tile, Pair Review positions transfer exactly — no vertical blur at
  the seams.
- **Auto-detect per tile (advanced)**: each tile gets its own band (use ONLY
  when the disk moved vertically a LOT between shots) and its own offset (px)
  slider — in the Stitcher's "ROI band preview" expander and in Pair Review
  Step 3 (the same sliders, synchronised). The band height always comes from
  the sidebar "ROI height (fraction of tile height)" slider. When the per-tile
  bands differ, the pipeline automatically converts pair dy between the
  full-tile and cropped-tile frames — but the safest workflow is still
  reference tile 1.
- **Also keep full panorama (debug)**: when ON AND a single ROI band is
  applied to every tile, the pipeline ALSO blends a full-resolution panorama
  without cropping, available as a separate download. Costs one EXTRA blend
  pass — keep OFF for production runs.

The crop happens BEFORE SIFT / NCC / blending, so on 20 MP frames with a
1/3-frame band the whole pipeline runs ~3x faster and produces a panorama
~3x smaller in height. The crop is vertical only — horizontal overlap
between adjacent tiles is preserved, so pairwise alignment is unaffected.

In the **Pair Review** tab, **Step 3** shows the reference tile with its
ROI band drawn on top. Tune the **Vertical offset** slider (the band height
is set by the sidebar **ROI height (fraction of tile height)** slider) until
the band brackets the disk the way you want — the SAME band is
then applied to every tile (a preview grid shows the band on tiles 2..N).

### 5. Pair Review (manual alignment, recommended for hand-captured rows)

Open the **Pair Review** tab for an interactive workflow that does not require
a precise `Expected overlap` prior:

1. **Estimate** — fast automatic estimate of every adjacent pair on downscaled copies.
2. **Review** — each pair is shown as an overlay (Blend / Checkerboard / Difference)
   with tile outlines A (yellow) and B (cyan). Tick **Position is correct** for good pairs.
   If two neighbouring shots were captured **without any overlap**, tick
   **No overlap (black gap)** instead — the pair is placed side by side with a visible
   black band on the panorama (gap width configurable in Step 2; unticking restores
   the previous position). Such a pair counts as approved. The black band is pure
   black with crisp image borders — multi-band blending cannot bleed darkness into
   the tiles next to the gap.
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
[Auto-detect / manually correct a horizontal ROI band on every tile
 (Tenengrad energy centroid +/- ~2 sigma; per-tile offset sliders
  in the Stitcher ROI preview / Pair Review Step 3 when the disk
  moved vertically between shots)]
    |
[Downscale tiles for estimation (max side 1600 px by default)]
    |
[Pairwise stitching i <-> i+1]  (on the downscaled copies — ~9-13x faster):
   1. SIFT detector + FLANN matcher
   2. Lowe ratio test -> good matches
   3. RANSAC + estimateAffinePartial2D
   4. Fallback: phase correlation
   5. Fallback: NCC scan (if all else fails)
    |
[Scale shifts back to full resolution + full-res ECC polish (sub-pixel)]
    |
[Compute cumulative absolute positions]
    |
[Multi-band blending]  (always at FULL resolution of the ROI-cropped tiles):
   - Laplacian pyramid (3 levels by default)
   - Focus-priority weight mode by default: in the overlap zone the tile
     with the highest Tenengrad sharpness score wins the high-frequency
     content; low frequencies are still averaged with a feather mask
   - Gain compensation between tiles
    |
[Optional debug full-resolution panorama (no ROI crop) — when
 "Also keep full panorama (debug)" is ON and a single ROI band
  was applied to every tile]
    |
[Panorama (ROI-cropped) + optional panorama_full (debug) + diagnostics]
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
    ├── io_utils.py         # Loading + _N parsing + sorting + ROI band helpers
    ├── preprocessing.py    # CLAHE + sharpness/Tenengrad maps + feather mask
    ├── stitching.py        # SIFT+RANSAC + fallbacks
    ├── pair_review.py      # Review-scale estimation + overlay/static previews
    │                       #   + ROI overlay rendering + JSON session
    ├── blending.py         # Multi-band blending (memory-optimized, focus-priority)
    └── pipeline.py         # Orchestration (+ reviewed-position overrides + ROI)
```

## Troubleshooting

See the in-app **Help** tab for a full troubleshooting guide. The most common issues:

- **"Unable to allocate ... GiB" (OOM)**: enable the ROI crop in the sidebar (Region of interest
  → Crop tiles to ROI band) — on 20 MP frames whose disk occupies ~1/4 of the frame height this
  shrinks the canvas ~4x in height; also try reducing `Laplacian pyramid levels` to 3–4
- **Dark zone at the panorama edge**: switch `Blending weight mode` to `Focus-priority (default)`
  or `Feather` — both are safe on dark images. The `Focus-weighted (per-tile)` mode is a legacy
  option that can fail in non-overlap zones on near-zero-sharpness dark frames
- **Visible seam**: increase `Laplacian pyramid levels` to 6-7 and `Gain compensation` to 1.0
- **ROI band missed the disk on some tile**: in Pair Review → Step 3, use the Vertical offset
  correction slider for that tile to shift the band up or down
- **Want the sharpest detail to win in the overlap zone** (operator focused each frame on a
  different depth): keep `Blending weight mode` on `Focus-priority (default)` (this is now the
  default) — the tile with the highest Tenengrad focus score wins the high-frequency content;
  low frequencies are still averaged so brightness stays smooth and residual alignment errors
  are not revealed
- **All methods failed**: increase `SIFT features` to 10000, lower `Lowe ratio threshold` to 0.7

## License

MIT — see the root [LICENSE](../../LICENSE).
