# 02 — TEM Grain Analyzer

A Streamlit app for characterising the grain-size distribution of a selected polycrystalline layer on a TEM cross-section image.

## What it does

1. Loads a TIFF and reads the pixel size automatically from the private `OlympusSIS` TIFF tag (33560) — no manual scale-bar calibration needed.
2. Lets the user select a layer using one of four modes: SAM (click/auto), Polygon (click vertices), or Manual rows.
3. Segments grains using one of five methods (see below).
4. Measures each grain (area, equivalent diameter, perimeter, circularity, eccentricity, aspect ratio, orientation) and fits a lognormal to the equivalent-diameter distribution.
5. Produces a self-contained JSON report (measurements + metadata + parameters + colour-labeled grain map embedded as base64 PNG).
6. In the **Batch Statistics** tab, aggregates several JSON reports into pooled grain-size statistics.

## Run

From the project root (after activating the shared `.venv`):

```bash
streamlit run apps/02_grain_analyzer/app.py
```

## Segmentation methods

1. **Manual tracing (polygon)** — draw grain outlines by hand. ISO 13067 compliant, most reliable.
2. **Structure tensor (auto-assist)** — auto-detect grains by local lattice-fringe orientation change.
3. **Click-to-segment grains (FFT)** — click on each grain, local FFT + sliding window amplitude maps with orientation selectivity.
4. **FFT (Grachev)** — global FFT + radial profile detection + Wiener filter + reference d-spacings mode.
5. **Classical watershed** — bilateral filter → CLAHE → Scharr gradient → Otsu threshold → watershed.

## Tabs

### Single Image

- Upload one TIFF
- Auto-detected pixel size (with manual override)
- Layer selection: SAM (click/auto), Polygon (click vertices), Manual rows
- Five segmentation methods with per-method parameters
- Grain review: select, highlight, and delete individual grains
- Outputs: colour-labeled grain map, boundary overlay, summary statistics, histograms with lognormal fit
- Download: self-contained JSON report

### Batch Statistics

- Upload multiple JSON reports (or a ZIP)
- Per-image summary table
- Pooled statistics + pooled lognormal fit
- Pooled histograms + per-image box plot
- Download: combined CSV (one row per grain) + combined JSON

### Help

Quick start, method description, settings reference, troubleshooting, limitations, JSON schema.

## Files

```
02_grain_analyzer/
├── app.py                          # Streamlit UI (3 tabs, 5 segmentation methods)
├── README.md                       # this file
└── src/
    ├── __init__.py
    ├── io_utils.py                 # TIFF load + OlympusSIS tag parsing
    ├── layer_detect.py             # row-mean profile, info-strip detection, band suggestion
    ├── segmentation.py             # classical watershed pipeline
    ├── fft_segmentation.py         # FFT-based segmentation (Grachev method): Wiener filter,
    │                               # radial profile detection, reference d-spacings, Gaussian
    │                               # quality assessment, Scherrer coherent-size estimate
    ├── click_segmentation.py       # Click-to-segment: local FFT, sliding window amplitude
    │                               # maps, orientation-specific masks, region growing
    ├── structure_tensor_segmentation.py  # Structure tensor: local lattice-fringe orientation
    ├── sam_segmenter.py            # SAM (Segment Anything Model) wrapper with lazy torch import
    ├── polygon.py                  # LayerMask, polygon/row mask utilities
    ├── measurements.py             # regionprops + lognormal fit + summary stats
    ├── visualization.py            # color-labeled image, boundary overlay, histograms
    ├── report.py                   # self-contained JSON report builder
    ├── batch.py                    # batch aggregation, pooled stats
    └── orientation.py              # layer angle detection (Radon/row-mean variance) — utility
```

## Dependencies

Uses the shared project `requirements.txt`. Key dependencies:

- **Streamlit** — web UI framework
- **OpenCV (opencv-python)** — image processing, morphological operations
- **scikit-image** — regionprops, label2rgb, structure tensor
- **scikit-learn** — DBScan for FFT reflection clustering
- **scipy** — FFT, signal processing, optimization
- **tifffile** + **imagecodecs** — TIFF loading with OlympusSIS metadata
- **NumPy** — array operations
- **Matplotlib** — histograms, radial spectrum plots
- **Plotly** — interactive 1D radial power spectrum
- **Pandas** — editable table for reference d-spacings
- **streamlit-image-coordinates** — click-on-image interaction
- Optional: **torch** + **segment-anything** — SAM layer selection (falls back gracefully)
