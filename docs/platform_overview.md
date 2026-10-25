# Platform Overview

## Purpose

TEM Analysis Platform is a collection of modular web apps for routine processing of data related to transmission electron microscopy (TEM):

- Optical microscope images (cross-section preparation)
- TEM images (Bright Field, Dark Field, HRTEM)
- Diffraction patterns (SAED)
- EDS spectra
- Metadata (scale, calibration, acquisition conditions)

## Architectural Principles

### 1. Modularity

Each app is fully self-contained in code:
- Its own folder in `apps/`
- Its own `README.md`
- Its own `src/` package with business logic

### 2. Single Shared Virtual Environment

All apps share **one virtual environment at the project root** with **one `requirements.txt`**.

Rationale:
- Most apps use the same core stack (Streamlit + NumPy + OpenCV + scikit-image)
- A single `pip install -r requirements.txt` covers all apps
- Lower disk usage (one `.venv` instead of one per app)
- Faster switching between apps (no re-activation needed)

When to reconsider: if a future app requires conflicting package versions
(e.g. `numpy<2.0` vs `numpy>=2.0`), move that single app to its own `.venv`
inside its folder. Until then, the shared environment is the simplest choice.

### 3. Incremental Development

Start with isolated apps for specific tasks.
Then extract common utilities into `shared/` and integrate into a unified platform.

### 4. Reproducibility

- All dependencies are pinned in the root `requirements.txt`
- Common data formats are described in [data_formats.md](data_formats.md)
- Test datasets are kept per-app (not in the repo)

## Conventions

### App Naming

`apps/NN_short_name/`, where:
- `NN` is a two-digit number (01, 02, ...)
- `short_name` is a snake_case name reflecting the task (e.g. `optical_stitcher`, `spectrum_viewer`)

### Versioning

While apps evolve independently, no global version is needed.
After integration into the platform, use SemVer (`v1.0.0`).

### Branching

- `main` — stable releases
- `dev/<feature>` — development
- `fix/<issue>` — bug fixes

## Integration (Future)

When 3-5 apps are ready, the plan is:
1. A common launcher with an app-selection menu
2. A shared cache of loaded data
3. Passing results between apps
4. A shared UI style (theme, fonts, icons)
