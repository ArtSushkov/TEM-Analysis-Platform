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
- **Cross-section Map tab** — register the stitched x10 overview panorama
  with the stitched x100 cross-section panorama of the SAME TEM disk:
  automatic scale-bar calibration (detection + OCR, manually correctable),
  tie-point registration on the disk edges, calibration consistency check,
  click-to-crop preview (200 µm field of view + uncertainty margin) and a
  standalone interactive HTML report
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

TIFF files from microscope cameras typically do **not** store scale information in metadata,
but some software (ImageJ, OME-TIFF writers, FEI/SEM capture tools) does. The calibration
block at the top of the **Stitcher** and **Pair Review** tabs handles both cases — and if
the metadata is missing (or you decline it), you can still calibrate from the visible scale
bar embedded in one of the tiles:

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

### 5b. Calibration (µm per pixel) — top of the Stitcher / Pair Review tabs

The calibration block is the FIRST thing you see in the **Stitcher** and
**Pair Review** tabs after uploading a series. The same flow runs in both
tabs and the calibration is keyed by the upload — calibrate once per
series and it sticks (the calibration is attached automatically to the
saved panorama when you press **Save to Cross-section Map**).

**Step 1 — Auto-search in image metadata.** Every tile of the series is
scanned for pixel-size metadata. Sources checked (in this order):

- **TIFF XResolution / YResolution + ResolutionUnit** — the standard
  TIFF resolution tags (inch / centimeter). Returns µm/px.
- **ImageDescription (ImageJ)** — ImageJ writes `unit=micron` /
  `pixel_width=...` / `pixel_height=...` into the TIFF
  ImageDescription tag.
- **ImageDescription (OME-XML)** — OME-TIFF writers embed
  `PhysicalSizeX="..."` / `PhysicalSizeY="..."` (default unit µm).
- **Vendor TIFF tags (FEI / SEM XML)** — best-effort extraction of
  `<PixelSizeX>` / `<pixel_width>` from FEI/SEM XML metadata.

The found values are shown per tile; a consistent value across all tiles
with metadata is highlighted as the suggested calibration. Disagreements
between tiles fall back to the median with a warning.

**Step 2 — Action.** Pick one of:

1. **Accept metadata calibration** — apply the suggested µm/px value
   (only available when metadata was found).
2. **Decline — calibrate manually from a scale bar** — always
   available, whether metadata was found or not.

**Step 3 — Manual flow (after Decline).** Pick the tile of the series
where the scale bar is visible (the bar is common to the whole series),
choose the **Snapshot display size** (640 / 800 / 960 / 1120 / 1280 px,
default 960 — the operator's request: a BIGGER snapshot so the cursor can
land ON a scale-bar pixel; the choice is remembered for the session, the
full snapshot is never cropped at any width), then one of TWO modes (the
two auto-search modes "1 · Auto search the entire selected tile" and
"2 · Draw ROI on the tile, then auto search within it" were REMOVED at
the operator's request — the click-based modes proved faster and more
reliable; at the operator's request the two click-based modes were
RENUMBERED 1 / 2, and mode 1 is the FIRST option and therefore the
DEFAULT):

1. **Click once ON the bar — the app traces its full length (mode 1,
   default, universal fallback)** — click ANY pixel that belongs to the bar,
   whatever its colour (red, blue, white, black — any microscope or
   operator colour scheme; no neural network is trained, so nothing
   breaks when the next microscope draws the bar in a different
   colour). The app takes the colour of the clicked pixel (3x3 median)
   and grows the trace left/right along the clicked row by colour
   similarity, following slightly tilted bars across adjacent rows,
   extending through anti-aliased tips, and rejecting traces that do
   not stand out against the surrounding background (so a click on a
   flat background can never produce an image-wide "bar"). The traced
   line with its endpoints is drawn on the preview AND, immediately
   after the click, a **ZOOMED verification view** of the found bar is
   shown — an enlarged fragment with the red traced line and the green
   endpoints on it, with enough room above and below to read the label
   — so the operator can confirm at magnification that the trace is
   correct before applying anything. A **colour tolerance** slider
   handles noisy JPEG artefacts. The value is then read by OCR (tall
   label windows ABOVE **and** BELOW the bar are both tried; the
   window that produced the reading is reported). The windows
   grow PROGRESSIVELY — three levels (base, tall, extra-tall,
   each also wider) — while no trustworthy value is found, so a
   label sitting far above/below the bar is captured too (the
   earlier fixed-height window sometimes could not contain the
   number at all). The value can also be typed in and applied
   explicitly. This is the recommended mode when the bar is
   thin and precise end-to-end clicking (mode 2) is fiddly.
2. **Draw ROI on the tile, then click two ends of the bar** — draw the
   ROI with two clicks (click 1 = upper-left corner, click 2 =
   lower-right corner; a 3rd click starts a new rectangle), then click
   the two extreme points of the scale bar inside the ROI preview to
   build a red line. A scale bar is a strictly horizontal line, so
   while **Force horizontal line** is ON (default) the second y is
   kept EXACTLY equal to the first one: the clicked y of point 2 is
   replaced by the y of point 1, the length is |x2 − x1| and the y2
   field is locked. The line is drawn ON the clickable preview and
   updates LIVE: every change of the end-1/end-2 coordinates (a new
   click or a typed value) is re-drawn on the image immediately.
   Enter the bar's real length in µm / nm / mm; the pixel length is
   measured in full-res coordinates. Click coordinates fill the
   numeric fields ONCE (at the moment of the click) and never
   overwrite values you typed by hand. Always works and is the most
   accurate mode (the OCR mode needs pytesseract + the tesseract
   binary).

The resulting calibration is shown as `X.XXXX µm/px (source · extra)`,
where source is `metadata`, `auto (OCR)` or `manual`.

### 6. Run

Click **Stitch panorama** (Stitcher tab) or **Stitch panorama with these positions**
(Pair Review tab). Progress is shown in real time.

### 7. Result

The result block is PERSISTED in the session: it stays visible after any
click, so you can download the panorama, download its calibration sidecar,
copy the positions CSV and save to the Cross-section Map — in any order,
repeatedly. Previously the whole block vanished after the first
download/save click. **Hide result (clear downloads)** removes it from the
session (a new upload also clears it automatically).

- Preview of the stitched panorama
- Diagnostics table for each pair (method, dx/dy, rotation, matches, inliers)
- Tile position map (absolute coordinates on the canvas)
- Download buttons (each label shows the resulting file size in MB, so
  an interrupted download is spotted instantly):
  - **Panorama** in the chosen format — **PNG is the DEFAULT** (JPEG is
    also available). The TIFF export was REMOVED at the operator's
    request: the x100 panorama in TIFF repeatedly opened black-grey or
    with grey "broken" pixels in third-party viewers, no matter which
    exporter variant or metadata layout was used, while PNG stores
    exactly the same pixels losslessly with no codec layer that any
    viewer could reinterpret. The panorama bytes are cached per format
    and calibration, so repeated downloads do not re-encode the image.
  - **Very wide panoramas**: the JPEG format itself cannot store a side
    longer than 65500 px — a ~99 000 px wide panorama is downscaled to
    65500 px for the JPEG download automatically (a caption says so);
    PNG has no such limit and downloads at the FULL resolution. If the
    encoder fails on a low-RAM machine, a second encoder + (as a last
    resort) a downscaled PNG fallback are tried, and a persistent
    failure shows a red error message instead of a silently empty
    download (a failed encode is never cached — pressing the button
    again retries).
  - **PNG / JPEG** cannot store a reliable µm/px value (the PNG `pHYs`
    chunk holds only INTEGER pixels-per-metre and most viewers ignore
    it) — an additional **Download calibration (.txt)** button offers a
    plain-text sidecar (`um_per_pixel`, `px_per_um`, `method`, the bar
    line and value, OCR text) to keep next to the image.
  - **CSV** with all tile positions
  - **Save to Cross-section Map** — keeps the stitched panorama (+ its
    first tile) in the session memory for the Cross-section Map tab,
    TOGETHER WITH the calibration produced at the top of the tab (the
    caption above the button shows what will be attached). Do this once
    for the x10 series and once for the x100 series — the registration
    of the two panoramas needs BOTH calibrations. The saved entry keeps
    its calibration across reruns, so downloading a file and saving to
    the map can be combined freely.

The generated panorama bytes are cached in the session (keyed by format
and calibration), so repeated downloads do not re-encode the image every
click.

### 8. Cross-section Map (x10 → x100 registration + HTML report)

The third navigation tab connects the two panoramas of ONE TEM disk: the x10
overview (whole disk, shot slightly tilted through the sacrificial glass) and
the x100 detail (top view of the IC cross-section). Workflow:

1. **Panorama sources** — pick the x10 and x100 panoramas from the in-session
   store (what you saved with **Save to Cross-section Map** in the Stitcher /
   Pair Review tabs) or load panorama image files directly. Overview and
   Detail are PRE-SELECTED automatically by resolution every time the store
   changes (Overview = the smallest-resolution panorama, Detail = the
   largest one) — the manual selection is still available and is kept while
   the store stays unchanged.
2. **Calibrations (from Stitcher / Pair Review)** — both panoramas arrive
   already calibrated (the calibration produced at the top of the Stitcher
   / Pair Review tabs is attached to the saved entry). This section is
   READ-ONLY: it only shows the two calibrations. There is deliberately NO
   calibration UI here any more — calibrate at the top of the Stitcher /
   Pair Review tabs and press **Save to Cross-section Map** again. If a
   panorama is not calibrated, the section says so and points to those
   tabs (sections 3–5 need BOTH calibrations).
3. **Tie points (disk edges)** — pick the SAME physical point on both
   panoramas (the disk edges are visible in both): 2 points give an exact
   similarity mapping (scale + rotation + translation), 3–6 points give a real
   error estimate from the residuals. The two previews are STACKED one under
   another (each spans the full page width). Picking is TWO-STEP (the
   operator's request for pixel-precise agreement between the x10 and x100
   points): the FIRST click on a preview does NOT fix the pixel — it opens a
   MAGNIFIED fragment (a 760 px window of the full-resolution panorama —
   about three times the area of the previous 440 px window — rendered
   ~1280 px wide with an adaptive nearest-neighbour zoom, red crosshair +
   orange box at the first click); the SECOND click INSIDE the magnified
   fragment fixes the exact point and prefills the coordinate fields.
   "Cancel magnifier" closes it without fixing anything. If a preview ever
   shows up blank / not clickable (e.g. the browser discarded its iframe on
   a low-memory machine), press **'Previews blank / not clickable? —
   Restore'** under the x100 coordinate fields: both previews are re-mounted
   with fresh widget keys, all picked points are kept. From 3 tie points an
   optional **Affine** mapping model becomes available — it adds anisotropic
   scale + shear and absorbs the tilt distortion of the x10 view (part of
   the large mid-panorama errors).
4. **Consistency check** — the physical distance between every pair of tie
   points is computed on BOTH panoramas via their calibrations; the distances
   must match (same disk!). A deviation above 5 % means one of the
   calibrations is wrong.
5. **Live preview** — click any pixel of the x10 panorama: the mapped point
   (with its uncertainty radius) is drawn on x10 and the corresponding crop is
   taken from the x100 panorama — 200 µm field of view (configurable) plus a
   safety margin k·σ. The crosshair always marks the CURRENT point (the
   preview is rebuilt right after every click). The x100 fragment carries a
   labelled SCALE BAR in its bottom-left corner (the largest round
   1/2/5/10... µm value that fits) and **GLASS / IC side plates** centred
   on its TOP / BOTTOM edges — a radio above the preview asks which side
   of the x100
   panorama is the sacrificial glass (the other side is the IC); the
   same plates appear on every fragment in the HTML report. Points mapped
   outside the
   x100 coverage are clamped to the nearest area with an explicit warning.
   The preview also reports the distance to the nearest tie point — between
   tie points the mapping is an interpolation and beyond them an
   extrapolation, so the true local error (glass parallax, stitching drift
   of the two independently stitched panoramas) can be much larger than σ.
   All clickable previews (tie points, live preview, ROI pickers) now
   STRETCH to the container width, so a wide panorama is fully visible and
   every point of it is clickable — clicks are rescaled from the displayed
   size back to the full-resolution coordinates automatically.
6. **HTML report** — one standalone offline file with the same interactive
   click-to-crop behaviour (pure inline JS) and the registration metadata
   (calibrations, tie points, residuals, consistency, glass/IC side
   labels). Layout: the x10
   panorama spans the FULL page width; the x100 fragment is placed BELOW
   it and shows EXACTLY the field of view from the app ("Field of view
   around the point (µm)" plus the k·σ margin) — a bigger FOV shows more
   of the panorama, a smaller one zooms in. The fragment ALWAYS fits the
   page width: a window wider than the page is scaled down to it (the
   whole picture stays visible and readable), a smaller window is shown
   1:1. The fragment carries the same labelled scale
   bar in its bottom-left corner and the same **GLASS / IC plates
   centred on its top / bottom edges** (they travel with the fragment when it is moved
   along the disk-edge line; the side assignment is also written into
   the registration metadata section). Every click also reports the distance
   to the nearest tie point (interpolation/extrapolation warning). The
   two download buttons give
   FULL-resolution JPEG copies (quality 95) embedded separately from the
   compact interactive images (this can be switched off with the
   'Embed full-resolution copies' checkbox). Full-resolution PNG downloads
   are in the same tab. The generated report is kept in the session, so
   the download button stays available after clicks (**Discard generated
   report** removes it).

The mapping is deliberately approximate; with ≥ 3 tie points the optional
Affine model absorbs the anisotropic part (tilt), but glass parallax and
the stitching drift between the two independently stitched panoramas stay
unmodelled — the uncertainty estimate, the crop margin and the
nearest-tie-point warning make the residual error visible instead.

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
├── app.py                  # Streamlit UI (Stitcher / Pair Review /
│                           #   Cross-section Map / Help tabs)
├── README.md
└── src/
    ├── __init__.py
    ├── io_utils.py         # Loading + _N parsing + sorting + ROI band helpers
    ├── preprocessing.py    # CLAHE + sharpness/Tenengrad maps + feather mask
    ├── stitching.py        # SIFT+RANSAC + fallbacks
    ├── pair_review.py      # Review-scale estimation + overlay/static previews
    │                       #   + ROI overlay rendering + JSON session
    ├── blending.py         # Multi-band blending (memory-optimized, focus-priority)
    ├── pipeline.py         # Orchestration (+ reviewed-position overrides + ROI)
    └── cross_section_map.py  # x10/x100 registration + HTML report
                              #   + calibration flow (metadata auto-search,
                              #     manual auto/ROI-auto/ROI-manual-two-clicks)
                              #   used at the TOP of the Stitcher / Pair Review
                              #   tabs and as an on-demand fallback in the
                              #   Cross-section Map tab
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
