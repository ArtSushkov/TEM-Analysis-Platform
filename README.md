# TEM Analysis Platform

A collection of Streamlit web apps for routine processing of data from transmission electron microscopy (TEM) and related specimen-preparation tasks.

## Concept

The platform follows a **modular monorepo** principle: each task gets its own standalone Streamlit app. All apps share a single virtual environment and a single `requirements.txt` at the project root.

## Contents

### Apps

| # | Folder | Description | Status |
|---|---|---|---|
| 01 | [`apps/01_optical_stitcher`](apps/01_optical_stitcher) | Stitching a series of optical microscope images into a panorama (for TEM cross-section prep) | Ready |
| 02 | `apps/02_*` | *(reserved)* | Planned |

### Repository Structure

```
tem-analysis-platform/
├── requirements.txt          # Shared dependencies for all apps
├── apps/                     # Streamlit apps (each is standalone in code)
│   └── 01_optical_stitcher/
├── docs/                     # Platform-level documentation
└── shared/                   # Shared utilities (future)
```

## Quick Start

### 1. Create a virtual environment at the project root

```bash
cd tem-analysis-platform

# Create the venv (do this once)
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\activate
# Windows cmd:
.venv\Scripts\activate.bat
```

After activation, your prompt should show `(.venv)`.

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run any app

```bash
streamlit run apps/01_optical_stitcher/app.py
```

The app opens in your browser at `http://localhost:8501`.

### 4. Switching between apps

No need to deactivate — just run another app from the same activated environment:

```bash
streamlit run apps/02_spectrum_viewer/app.py     # (when available)
```

### 5. Deactivate when done

```bash
deactivate
```

## Architecture: Single Shared Environment

This project uses **one virtual environment at the root** for all apps, with **one shared `requirements.txt`**.

**Why this approach:**
- Simpler setup — one `pip install -r requirements.txt` covers all apps
- Lower disk usage (one `.venv` instead of one per app)
- Faster switching between apps (no re-activation needed)
- Most apps share the same core stack (Streamlit + NumPy + OpenCV + scikit-image)

**When to reconsider:** if a future app requires conflicting package versions (e.g. `numpy<2.0` vs `numpy>=2.0`), move that single app to its own `.venv` inside its folder. Until then, the shared environment is the simplest and best choice.

See [docs/platform_overview.md](docs/platform_overview.md) for the full architectural rationale.

## Technologies

- **Python 3.10+**
- **Streamlit** — web UI
- **OpenCV / scikit-image** — image processing
- **tifffile / imagecodecs** — TIFF support (including LZW)
- **NumPy / SciPy** — numerical computing
- **Matplotlib / Plotly** — plotting

## License

[MIT](LICENSE) — free use, modification, and distribution.
