"""
app.py — Streamlit UI for stitching optical microscopy images.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

# Add path to src
sys.path.insert(0, str(Path(__file__).parent))

from src import io_utils, pipeline, pair_review, preprocessing, stitching


# === Page configuration ===
st.set_page_config(
    page_title="Optical Microscopy Stitcher",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _param(key: str, default):
    """Read a sidebar widget value from session state (shared across tabs)."""
    return st.session_state.get(key, default)


def _current_upload_signature():
    """Stable signature of the current upload (used to cache loaded tiles)."""
    mode = st.session_state.get("upload_mode", "Multiple files")
    if mode == "ZIP archive":
        z = st.session_state.get("uploaded_zip")
        if z is None:
            return None
        return ("zip", z.name, z.size)
    files = st.session_state.get("uploaded_files") or []
    if not files:
        return None
    return ("files", tuple((f.name, f.size) for f in files))


def _get_tiles_cached():
    """Load tiles once per upload, cached in session state.

    Shared by the Stitcher and Pair Review tabs. Returns (tiles, error)
    where exactly one of the two is None.
    """
    sig = _current_upload_signature()
    if sig is None:
        return None, None
    if st.session_state.get("tiles_sig") == sig:
        cached = st.session_state.get("tiles_data")
        if isinstance(cached, dict) and cached.get("error"):
            return None, cached["error"]
        return cached, None

    mode = st.session_state.get("upload_mode", "Multiple files")
    try:
        if mode == "ZIP archive":
            zf = st.session_state["uploaded_zip"]
            data_dir = Path(tempfile.mkdtemp(prefix="micro_zip_"))
            with zipfile.ZipFile(zf) as z:
                z.extractall(data_dir)
        else:
            data_dir = Path(tempfile.mkdtemp(prefix="micro_files_"))
            for f in st.session_state["uploaded_files"]:
                (data_dir / f.name).write_bytes(f.getbuffer())
        tiles = io_utils.load_tiles(data_dir)
        st.session_state["tiles_sig"] = sig
        st.session_state["tiles_data"] = tiles
        return tiles, None
    except Exception as e:
        st.session_state["tiles_sig"] = sig
        st.session_state["tiles_data"] = {"error": str(e)}
        return None, str(e)


def _wipe_pair_widgets():
    """Remove per-pair widget state (positions / approvals) of the review."""
    for k in list(st.session_state.keys()):
        if k.startswith("rvp_"):
            del st.session_state[k]


def render_stitch_result(result, tiles):
    """Render panorama result: diagnostics, warnings, preview, downloads."""
    st.success(f"Panorama assembled: {result.canvas_size[1]}x{result.canvas_size[0]} px")

    # Pairwise diagnostics
    st.subheader("Stitching Diagnostics")
    diag_data = []
    for d in result.diagnostics:
        diag_data.append({
            "Pair": f"{d.pair_index+1} <-> {d.pair_index+2}",
            "Method": d.method,
            "Success": "yes" if d.success else "no",
            "dx": f"{d.translation[0]:.1f}",
            "dy": f"{d.translation[1]:.1f}",
            "rot_deg": f"{d.rotation_deg:.3f}",
            "scale": f"{d.scale:.4f}",
            "matches": d.matches,
            "inliers": d.inliers,
            "ncc": f"{d.score:.3f}" if d.score > 0 else "-",
            "Note": d.message,
        })
    st.dataframe(diag_data, use_container_width=True)
    if result.drift_trend is not None:
        slope, intercept = result.drift_trend
        st.caption(
            f"Vertical drift trend: {slope*1000:+.1f} px per 1000 px of horizontal "
            f"travel (intercept {intercept:+.1f} px). A smooth trend is normal — "
            "microscope stages drift vertically while moving along the row. "
            "Single pairs jumping off the trend are flagged in the warnings below.")

    if result.warnings:
        st.warning("Warnings:")
        for w in result.warnings:
            st.write(f"- {w}")

    # Panorama preview
    st.subheader("Result")
    panorama_rgb = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB)
    h, w = panorama_rgb.shape[:2]
    if max(h, w) > 4000:
        st.info(f"Large image ({w}x{h}), showing downscaled preview. "
                f"Download full size below.")
        s = 4000 / max(h, w)
        preview = cv2.resize(panorama_rgb, None, fx=s, fy=s)
    else:
        preview = panorama_rgb
    st.image(preview, use_container_width=True)

    # Tile positions
    st.subheader("Tile Position Map")
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p[0] for p in result.tile_positions],
            y=[p[1] for p in result.tile_positions],
            mode="lines+markers+text",
            text=[t.name for t in tiles],
            textposition="top center",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            hovertemplate="%{text}<br>x=%{x}, y=%{y}<extra></extra>",
        ))
        fig.update_layout(
            title="Tile offsets on the canvas (px) — a smooth curve is expected",
            xaxis_title="offset_x",
            yaxis_title="offset_y",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_yaxes(autorange="reversed")  # image coordinates
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # plotly not available — text output below still works
    positions_text = "\n".join(
        f"  {t.name}: ({pos[0]}, {pos[1]})"
        for t, pos in zip(tiles, result.tile_positions)
    )
    st.code(positions_text)

    # Download buttons
    st.subheader("Download")
    out_format = _param("out_format", "PNG")
    ext_map = {"PNG": "png", "TIFF (LZW)": "tif", "JPEG": "jpg"}
    ext = ext_map.get(out_format, "png")

    if ext == "tif":
        import tifffile
        rgb = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        tifffile.imwrite(buf, rgb, compression="lzw")
        data = buf.getvalue()
        mime = "image/tiff"
    else:
        if ext == "png":
            ok, buf = cv2.imencode(".png", result.panorama)
            mime = "image/png"
        else:
            ok, buf = cv2.imencode(".jpg", result.panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mime = "image/jpeg"
        data = buf.tobytes()

    fname = f"panorama_{len(tiles)}_tiles.{ext}"
    st.download_button(
        label=f"Download panorama ({out_format})",
        data=data,
        file_name=fname,
        mime=mime,
    )

    # CSV with tile positions
    import csv
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["tile_name", "tile_index", "offset_x", "offset_y", "width", "height"])
    for t, pos in zip(tiles, result.tile_positions):
        h, w = t.image.shape[:2]
        writer.writerow([t.name, t.index, pos[0], pos[1], w, h])
    st.download_button(
        label="Download tile positions (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="tile_positions.csv",
        mime="text/csv",
    )


def render_help_tab():
    """Render the Help & Usage Guide tab."""
    st.markdown("# Help & Usage Guide")

    st.markdown("## Quick Start")
    st.markdown(
        "1. **Prepare images**: name them `<prefix>_N.<ext>` where `N` is the sequential index "
        "(e.g. `sample_1.tif`, `sample_2.tif`, ..., `sample_5.tif`)."
        "\n"
        "2. **Upload**: choose multiple files or a ZIP archive in the sidebar."
        "\n"
        "3. **Run**: click **Stitch panorama** (Stitcher tab) — or use the "
        "**Pair Review** tab to check and correct every pair manually first."
        "\n"
        "4. **Download**: panorama (PNG/TIFF/JPEG) and tile positions CSV."
    )

    st.markdown("---")
    st.markdown("## Pair Review workflow (manual alignment)")
    st.markdown(
        "When a row was captured by hand, automatic alignment can be slightly "
        "off for some pairs (and a precise `Expected overlap` prior is hard to "
        "give). The **Pair Review** tab solves this interactively:"
    )
    st.markdown(
        "1. **Step 1 — Estimate**: the app estimates the shift of every adjacent "
        "pair on downscaled copies (fast, full SIFT + fallback pipeline)."
        "\n"
        "2. **Step 2 — Review**: every pair is rendered as an overlay "
        "(**Blend** 50/50, **Checkerboard**, or **Difference** map) with tile "
        "outlines A (yellow) and B (cyan). Tick **Position is correct** for "
        "pairs that look right. If two neighbouring shots were captured "
        "**without any overlap** (a common operator mistake), tick **No "
        "overlap (black gap)** instead — the pair is placed side by side "
        "with a visible black band on the panorama, and its checkbox counts "
        "as approved."
        "\n"
        "3. **Step 3 — Adjust**: every unticked pair gets manual controls — "
        "nudge buttons (1–100 px steps) and numeric dx/dy inputs, with a live "
        "overlay that updates on every click."
        "\n"
        "4. **Step 4 — Stitch**: the panorama is blended using **exactly** the "
        "reviewed positions — approved pairs are taken as-is, manually adjusted "
        "pairs are placed exactly where you put them."
    )
    st.markdown(
        "- `dx` is the horizontal offset of tile N+1 relative to tile N: "
        "**increasing dx moves the tiles apart** (less overlap), decreasing dx "
        "pushes them closer together."
        "\n"
        "- **No overlap (black gap)** pairs: when two neighbouring shots share "
        "no common content, no algorithm can position them — any automatic "
        "estimate is a guess. Ticking **No overlap (black gap)** skips the "
        "estimate for that pair entirely: dx = tile width + gap, dy = 0, so "
        "the panorama shows a clear black separator band (gap width is "
        "configurable in Step 2) instead of a wrong forced overlap. Unticking "
        "restores the previous position and the pair goes back to the normal "
        "review flow."
        "\n"
        "- The automatic estimate is skipped entirely in the final stitch — no "
        "algorithm can override what you approved."
        "\n"
        "- **Export positions (JSON)** in Step 4 saves your work (positions, "
        "approvals, no-overlap marks); load it later in Step 1 to continue "
        "where you left off — useful because browser refreshes reset the "
        "interactive session."
        "\n"
        "- Optional *Refine approved pairs at full resolution* re-runs SIFT in "
        "a small window around approved positions for sub-pixel polish — it "
        "never touches manually adjusted pairs."
    )

    st.markdown("---")
    st.markdown("## Best Practices for Capturing Images")

    st.markdown("### Overlap")
    st.markdown(
        "Adjacent images must overlap by **20–40%** (30% is optimal). "
        "Too little overlap → SIFT cannot find enough matching keypoints. "
        "Too much overlap → unnecessary data and slower stitching."
    )

    st.markdown("### Scale Bar (Important!)")
    st.markdown(
        "TIFF files from microscope cameras typically **do not store scale information** in metadata "
        "(`XResolution`, `YResolution` are usually empty). The only way to preserve scale is to **embed "
        "the scale bar visually into one of the images** via the microscope software."
    )
    st.markdown(
        "**Where to place the scale bar** so that it survives stitching:"
    )
    st.markdown(
        "- **First image of the series**: scale bar in the **bottom-left corner**. "
        "This part of the first tile is never overlapped by the second tile, so the scale bar "
        "remains intact in the final panorama."
    )
    st.markdown(
        "- **OR last image of the series**: scale bar in the **bottom-right corner**. "
        "Same reasoning — this part of the last tile is never overlapped."
    )
    st.markdown(
        "**Avoid** placing the scale bar in the right part of intermediate tiles "
        "or the left part of any tile except the first — it will be blended away or "
        "distorted at the seam."
    )

    st.markdown("### Illumination")
    st.markdown(
        "Try to keep illumination **uniform** between shots. "
        "The `Gain compensation` setting corrects for small brightness differences, "
        "but large variations (e.g. lamp warming up) may still cause visible seams."
    )

    st.markdown("### Focus")
    st.markdown(
        "If the specimen is uneven, some frames may be partially out of focus. "
        "This is acceptable — the multi-band blending will handle it. "
        "If you see blur at tile edges, enable `Use focus-weighted mask` (but be aware "
        "it may cause dark zones on very dark images — see Troubleshooting)."
    )

    st.markdown("### Number of Images")
    st.markdown(
        "Up to ~100 images per row are supported. "
        "For more, split the series into multiple panoramas. "
        "Memory consumption scales linearly with the number of tiles."
    )

    st.markdown("---")
    st.markdown("## Settings Reference")

    st.markdown("### Feature Detection")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **SIFT features (per tile)** | 5000 | Number of keypoints detected per tile. "
        "Increase to 8000–10000 for low-contrast images. Decrease to 2000 for speed. |"
        "\n"
        "| **Lowe ratio threshold** | 0.75 | Strictness of match filtering. "
        "Lower (0.7) = stricter, fewer but more reliable matches. "
        "Higher (0.85) = more matches, but more false positives. |"
        "\n"
        "| **RANSAC reprojection threshold (px)** | 5.0 | Allowed geometric error for inliers. "
        "Lower = stricter. Higher = more tolerant to noise. |"
        "\n"
        "| **Min inliers for SIFT** | 8 | If SIFT produces fewer inliers than this, "
        "the app falls back to phase correlation or NCC scan. |"
    )

    st.markdown("### Preprocessing")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Apply CLAHE** | ON | Contrast-Limited Adaptive Histogram Equalization. "
        "**Critical for dark microscope images** — boosts SIFT keypoint count by 5–10x. |"
        "\n"
        "| **CLAHE clip limit** | 3.0 | Contrast amplification strength. "
        "Increase to 4–5 for very dark images. |"
    )

    st.markdown("### Alignment Verification")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Verify SIFT with NCC scan** | ON | Every SIFT result is cross-checked "
        "against a global normalized cross-correlation scan. On samples with long "
        "horizontal scratches or sparse dots, SIFT+RANSAC can lock onto a "
        "self-consistent but **wrong** consensus (e.g. matching one scratch to the "
        "neighboring scratch). The NCC scan is immune to this and catches such "
        "blunders. Adds a few seconds per pair. |"
        "\n"
        "| **Rejection score gap** | 0.07 | SIFT is rejected when the NCC scan finds "
        "an alignment better by this margin. Increase (0.10–0.12) if SIFT answers "
        "are wrongly rejected on low-texture samples; decrease (0.04–0.05) for "
        "stricter control. |"
        "\n"
        "| **Suppress dust / fixed-pattern matches** | ON | Dust and other artifacts "
        "on the optics appear at the **same pixel position in every frame** (they do "
        "not move with the stage). Keypoints on such artifacts match each other with "
        "a ~zero shift and can steal the RANSAC consensus. They are dropped "
        "automatically when the row clearly moves. |"
        "\n"
        "| **Vertical drift guard** | ON | Microscope stages drift vertically while "
        "moving along the row — that is normal and preserved. But a single pair whose "
        "vertical shift deviates sharply from the row-wide trend (e.g. +350 px among "
        "+25 px neighbors) is usually a matching blunder. Such pairs are re-verified "
        "with a constrained correlation scan and corrected when it confirms the trend. |"
    )

    st.markdown("### Blending")
    st.markdown(
        "| Setting | Default | Effect |"
        "\n"
        "|---|---|---|"
        "\n"
        "| **Laplacian pyramid levels** | 5 | Number of frequency bands for multi-band blending. "
        "More = smoother seams, but slower and uses more memory. "
        "Reduce to 3–4 if you get OOM errors. |"
        "\n"
        "| **Feather zone width (px)** | 64 | Smooth mask falloff at tile edges. "
        "Larger = softer seam transition. |"
        "\n"
        "| **Use focus-weighted mask** | OFF | When ON, pixels from sharper zones get higher weight. "
        "Useful when tile edges are blurred. May cause dark zones on very dark images — "
        "leave OFF for typical dark-field optical microscopy. |"
        "\n"
        "| **Gamma for focus weights** | 1.5 | Contrast of sharpness weights. "
        "Only used when focus-weighted mask is ON. |"
        "\n"
        "| **Gain compensation** | 0.8 | Equalizes average brightness between tiles. "
        "0 = off, 1 = full correction. Useful when illumination varies between shots. |"
    )

    st.markdown("### Backup Methods")
    st.markdown(
        "When SIFT fails to find enough inliers, the app falls back to alternative methods "
        "(in order):"
    )
    st.markdown(
        "1. **Phase correlation** — fast Fourier-based method. Works well for pure translation, "
        "fails if there is rotation or significant brightness difference."
        "\n"
        "2. **NCC scan** — brute-force search over a range of shifts using normalized "
        "cross-correlation. Slow but very robust. Used as last resort."
    )
    st.markdown(
        "Disable these if you want to enforce SIFT-only stitching (faster but less robust)."
    )

    st.markdown("---")
    st.markdown("## Troubleshooting")

    with st.expander("Some frames are shifted up or down (vertical misalignment)"):
        st.markdown(
            "**Possible causes and what the app does about them:**"
            "\n\n"
            "1. **Wrong SIFT consensus on repetitive texture** (most common). "
            "Polished cross-sections are full of long horizontal scratches; resin "
            "areas contain sparse look-alike dots (dust, pores). SIFT matches "
            "scratch-to-scratch or dot-to-dot, and RANSAC can lock onto a "
            "self-consistent but wrong consensus — one frame lands one scratch "
            "period higher/lower. **Fix**: keep `Verify SIFT with NCC scan` ON "
            "(default) — the wrong consensus is detected and replaced."
            "\n\n"
            "2. **Dust on the optics**. It does not move with the specimen, so "
            "dust-to-dust matches vote for a ~zero shift and can corrupt the "
            "estimate. **Fix**: `Suppress dust / fixed-pattern matches` ON (default). "
            "Also clean the objective lens — it is the cheapest and most effective fix."
            "\n\n"
            "3. **Vertical stage drift**. If the whole panorama is a smooth diagonal "
            "band, the stage drifted vertically while you moved it along the row — "
            "this is **genuine** and preserved by the app (the panorama simply "
            "becomes taller). Only *single pairs* jumping off the trend are treated "
            "as errors. Check the **Tile Position Map**: a smooth curve is normal, "
            "a lone jump indicates a problem pair."
            "\n\n"
            "4. **Coarse fallback quantization** (versions before the drift guard): "
            "the NCC fallback scanned shifts with a 25 px step, so dy values like "
            "25, 50, 75 were step artifacts that accumulated along the row. The "
            "current NCC scan uses a two-stage search with sub-pixel refinement "
            "plus an ECC polish."
            "\n\n"
            "5. **Flat correlation ridge on striped textures**. When the sample is "
            "dominated by long horizontal scratches / layers, the correlation "
            "surface stays almost flat along dx: any horizontal shift in a wide "
            "range fits almost equally well, so dx is only weakly constrained "
            "by the texture. The app scores every candidate shift on a "
            "fixed-width window of the overlap (so a small overlap cannot win "
            "by being easy), aligns each candidate vertically on the band "
            "structure first, then resolves the remaining ambiguity honestly: "
            "(a) if you set the `Expected overlap` in the **Capture** sidebar "
            "section — or the row already has solved pairs — the answer is "
            "anchored to that prior (`flat dx ridge — anchored to ... prior`); "
            "(b) otherwise, when the interface-line excursions (pores / dirt "
            "specks / bumps — the only non-repetitive features of such a "
            "sample) form a decisive \"booking\" in which every excursion "
            "finds a counterpart in the other tile while dust fixed to the "
            "optics is excluded, that booking corrects dx; (c) if neither "
            "works, the best texture agreement is kept and the diagnostics "
            "show `WARNING: dx weakly constrained` — verify that seam "
            "visually."
        )

    with st.expander("Dark zone appears at the edge of the panorama"):
        st.markdown(
            "**Cause**: `Use focus-weighted mask` is enabled, and the dark microscope images "
            "produce near-zero sharpness values, causing blending to fail in non-overlap zones."
            "\n\n"
            "**Fix**: Disable `Use focus-weighted mask` (default). The simple feather mask "
            "is stable for dark images."
        )

    with st.expander("\"Unable to allocate ... GiB for an array\" (Out of Memory)"):
        st.markdown(
            "**Cause**: The canvas is too large for available RAM."
            "\n\n"
            "**Fixes**:"
            "\n"
            "1. Reduce `Laplacian pyramid levels` to 3–4."
            "\n"
            "2. Ensure `Use focus-weighted mask` is OFF (default)."
            "\n"
            "3. For very large sets (>30 tiles), split into multiple panoramas."
        )

    with st.expander("Visible seam between tiles"):
        st.markdown(
            "**Fixes**:"
            "\n"
            "1. Increase `Laplacian pyramid levels` to 6–7."
            "\n"
            "2. Increase `Feather zone width` to 100–150."
            "\n"
            "3. Increase `Gain compensation` to 1.0 if brightness differs between tiles."
        )

    with st.expander("All stitching methods failed for a pair"):
        st.markdown(
            "**Fixes**:"
            "\n"
            "1. Verify the two images actually overlap (at least 20%)."
            "\n"
            "2. Increase `SIFT features` to 10000."
            "\n"
            "3. Lower `Lowe ratio threshold` to 0.7."
            "\n"
            "4. Increase `CLAHE clip limit` to 4–5 for very dark images."
        )

    with st.expander("Stitching is slow (>30 seconds)"):
        st.markdown(
            "**Fixes**:"
            "\n"
            "1. Reduce `SIFT features` to 2000–3000."
            "\n"
            "2. Reduce `Laplacian pyramid levels` to 4."
            "\n"
            "3. Disable `NCC scan fallback` (it is the slowest backup method)."
        )

    with st.expander("Cannot extract index from filename"):
        st.markdown(
            "**Cause**: Filenames must end with `_N` where N is an integer."
            "\n\n"
            "**Examples**:"
            "\n"
            "- ✅ `sample_1.tif`, `sample_2.tif`, ..., `sample_5.tif`"
            "\n"
            "- ✅ `x10_001.tif`, `x10_002.tif`, ..., `x10_015.tif`"
            "\n"
            "- ❌ `sample.tif` (no index)"
            "\n"
            "- ❌ `sample_1a.tif` (non-numeric suffix)"
        )

    st.markdown("---")
    st.markdown("## How It Works")
    st.markdown(
        "1. **Load** N images sorted by `_N` index."
        "\n"
        "2. **CLAHE preprocessing** — boost contrast for keypoint detection."
        "\n"
        "3. **Pairwise stitching** for each `i ↔ i+1` pair:"
        "\n"
        "   - SIFT detector + FLANN matcher"
        "\n"
        "   - Lowe ratio test → good matches (fixed-pattern / dust matches dropped)"
        "\n"
        "   - RANSAC + `estimateAffinePartial2D` → translation + small rotation"
        "\n"
        "   - **NCC verification**: the SIFT answer is cross-checked against a global "
        "correlation scan; a wrong consensus on repetitive textures is rejected"
        "\n"
        "   - Fallbacks: phase correlation → NCC scan (two-stage, sub-pixel + ECC polish)"
        "\n"
        "4. **Vertical drift guard** — pairwise shifts that deviate sharply from the "
        "row-wide drift trend are re-verified and corrected."
        "\n"
        "5. **Cumulative position calculation** — chain pairwise translations."
        "\n"
        "6. **Multi-band blending** with focus-aware feather masks:"
        "\n"
        "   - Laplacian pyramid (5 levels by default)"
        "\n"
        "   - Weighted average per level"
        "\n"
        "   - Gain compensation between tiles"
        "\n"
        "7. **Output**: panorama + diagnostics table + tile position map + CSV."
    )

    st.markdown("---")
    st.markdown("## Limitations")
    st.markdown(
        "- **1D row only** — 2D grids are not supported yet."
        "\n"
        "- **Up to ~100 tiles** — for larger sets, split into multiple panoramas."
        "\n"
        "- **Translation + small rotation** — significant perspective distortion is not handled."
        "\n"
        "- **Scale bar must be embedded visually** — TIFF metadata is not used for scale."
    )


def render_stitcher_tab():
    """Render the main Stitcher tab."""
    st.title("Microscope Image Stitcher")
    st.caption(
        "Upload a series of images with `_N` suffix (N = sequential index). "
        "The app automatically determines their relative positions and stitches a panorama."
    )

    # === Sidebar: settings ===
    st.sidebar.header("Method Settings")

    with st.sidebar.expander("Upload", expanded=True):
        upload_mode = st.radio(
            "Upload method",
            ["Multiple files", "ZIP archive"],
            index=0,
            key="upload_mode",
        )
        uploaded_files = st.file_uploader(
            "Select images",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="uploaded_files",
        ) if upload_mode == "Multiple files" else None

        uploaded_zip = st.file_uploader(
            "Select ZIP archive",
            type=["zip"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="uploaded_zip",
        ) if upload_mode == "ZIP archive" else None

    with st.sidebar.expander("Capture (optional)", expanded=False):
        expected_overlap = st.slider(
            "Expected overlap between adjacent frames (%)", 0, 80, 0, step=5,
            key="expected_overlap",
            help="Approximate overlap you aim for when capturing a row of "
                 "frames. Used ONLY when a pair's horizontal shift is "
                 "ambiguous (flat correlation ridge on layered/scratched "
                 "samples): the app then prefers the alignment that matches "
                 "this overlap. 0 = unknown — fully automatic.")

    with st.sidebar.expander("Feature Detection", expanded=True):
        n_features = st.slider("SIFT features (per tile)", 500, 10000, 5000, step=500,
                                key="n_features",
                                help="More = more robust but slower. 5000 is a good balance.")
        ratio_thresh = st.slider("Lowe ratio threshold", 0.5, 0.9, 0.75, step=0.05,
                                  key="ratio_thresh",
                                  help="Lower = stricter match filtering. 0.75 is standard.")
        ransac_thresh = st.slider("RANSAC reprojection threshold (px)", 1.0, 15.0, 5.0, step=0.5,
                                   key="ransac_thresh",
                                   help="Allowed inlier error. 5 px is standard.")
        min_inliers = st.slider("Min inliers for SIFT", 4, 50, 8, step=1,
                                 key="min_inliers",
                                 help="Below this — fall back to alternative methods.")

    with st.sidebar.expander("Preprocessing", expanded=False):
        use_clahe = st.checkbox("Apply CLAHE (recommended)", value=True,
                                 key="use_clahe",
                                 help="Adaptive histogram equalization. Critical for dark microscope images.")
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 8.0, 3.0, step=0.5, key="clahe_clip")

    with st.sidebar.expander("Blending", expanded=False):
        blend_levels = st.slider("Laplacian pyramid levels", 2, 8, 5, step=1,
                                  key="blend_levels",
                                  help="More = smoother seams but slower.")
        blend_feather = st.slider("Feather zone width (px)", 0, 200, 64, step=8,
                                   key="blend_feather",
                                   help="Smooth mask falloff toward tile edges.")
        use_focus_weights = st.checkbox("Use focus-weighted mask", value=False,
                                         key="use_focus_weights",
                                         help="Weighting by sharpness map. "
                                              "OFF (default) — simple feather mask, more stable on dark images. "
                                              "ON — better for tiles with blurred edges.")
        blend_gamma = st.slider("Gamma for focus weights", 0.5, 3.0, 1.5, step=0.1,
                                 key="blend_gamma",
                                 help=">1 — sharp zones dominate more. "
                                      "Only used when focus-weighted mask is enabled.")
        gain_strength = st.slider("Gain compensation", 0.0, 1.0, 0.8, step=0.1,
                                   key="gain_strength",
                                   help="Brightness equalization between tiles. "
                                        "0 = off, 1 = full correction.")

    with st.sidebar.expander("Backup Methods", expanded=False):
        enable_phase = st.checkbox("Phase correlation fallback", value=True, key="enable_phase")
        enable_ncc = st.checkbox("NCC scan fallback (slow)", value=True, key="enable_ncc")

    with st.sidebar.expander("Alignment Verification", expanded=False):
        verify_with_ncc = st.checkbox(
            "Verify SIFT with NCC scan", value=True, key="verify_with_ncc",
            help="Cross-checks every SIFT result against a global correlation scan. "
                 "Protects against wrong SIFT consensus on samples with repetitive "
                 "scratches or dust dots. Recommended ON. Adds a few seconds per pair.")
        ncc_verify_gap = st.slider(
            "Rejection score gap", 0.02, 0.15, 0.07, step=0.01, key="ncc_verify_gap",
            help="SIFT is rejected when the NCC scan finds an alignment better "
                 "by this margin. 0.07 is a balanced value.")
        suppress_fixed_pattern = st.checkbox(
            "Suppress dust / fixed-pattern matches", value=True,
            key="suppress_fixed_pattern",
            help="Dust on the optics stays at the same pixel position in every "
                 "frame. Matches between such artifacts vote for a ~zero shift "
                 "and can corrupt RANSAC — they are dropped automatically.")
        enable_drift_guard = st.checkbox(
            "Vertical drift guard", value=True, key="enable_drift_guard",
            help="Detects pairs whose vertical shift deviates sharply from the "
                 "row-wide drift trend and re-verifies them with a constrained "
                 "correlation scan. Fixes single frames shifted up or down.")

    with st.sidebar.expander("Output Format", expanded=False):
        out_format = st.selectbox("Output format", ["PNG", "TIFF (LZW)", "JPEG"], index=0,
                                  key="out_format")

    # === Main content ===
    tiles, load_error = _get_tiles_cached()

    if tiles is None:
        if load_error:
            st.error(f"Loading error: {load_error}")
            st.error("Verify that filenames contain `_N` (e.g. `sample_1.tif`).")
            return
        st.info("Upload images or a ZIP archive in the sidebar to begin.")
        st.markdown("### Expected filename format")
        st.markdown(
            "Files must be named `<name>_N.<ext>`, where `N` is the sequential index."
            "\n\n"
            "Example: `x10_1.tif`, `x10_2.tif`, ..., `x10_5.tif`."
        )
        st.markdown("### How it works")
        st.markdown(
            "1. **CLAHE** — adaptive brightness equalization (important for dark microscope images)"
            "\n"
            "2. **SIFT + RANSAC** — keypoint detection and affine transform estimation"
            "\n"
            "3. **Multi-band blending** — invisible seams between tiles"
            "\n"
            "4. **Gain compensation** — brightness equalization across tiles"
        )
        st.markdown("See the **Help** tab for detailed usage guide and best practices.")
        return

    st.success(f"Loaded {len(tiles)} images")
    st.write(f"Stitching order: {' -> '.join(t.name for t in tiles)}")

    # Preview uploaded tiles
    with st.expander("Preview uploaded tiles", expanded=False):
        cols = st.columns(min(len(tiles), 5))
        for i, (col, tile) in enumerate(zip(cols, tiles)):
            with col:
                st.caption(f"#{tile.index}: {tile.name}")
                preview = tile.image
                h, w = preview.shape[:2]
                if max(h, w) > 600:
                    s = 600 / max(h, w)
                    preview = cv2.resize(preview, None, fx=s, fy=s)
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

    # Run button
    if not st.button("Stitch panorama", type="primary"):
        return

    # Progress
    progress = st.progress(0.0, "Initializing...")

    def update_progress(msg, p):
        progress.progress(min(max(p, 0.0), 1.0), msg)

    # Run stitching
    try:
        result = pipeline.stitch_row(
            tiles,
            n_features=n_features,
            ratio_thresh=ratio_thresh,
            ransac_thresh=ransac_thresh,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            min_inliers_sift=min_inliers,
            enable_phase_fallback=enable_phase,
            enable_ncc_fallback=enable_ncc,
            verify_with_ncc=verify_with_ncc,
            ncc_verify_gap=ncc_verify_gap,
            suppress_fixed_pattern=suppress_fixed_pattern,
            enable_drift_guard=enable_drift_guard,
            expected_overlap_pct=(expected_overlap if expected_overlap > 0
                                  else None),
            blend_levels=blend_levels,
            blend_feather=blend_feather,
            blend_gamma=blend_gamma,
            blend_gain_strength=gain_strength,
            blend_use_focus_weights=use_focus_weights,
            progress_callback=update_progress,
        )
    except Exception as e:
        st.error(f"Stitching error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    render_stitch_result(result, tiles)


def render_pair_review_tab():
    """Render the interactive Pair Review tab (manual alignment workflow)."""
    st.title("Pair Review — Verify & Adjust Pairs Manually")
    st.caption(
        "Interactive workflow: get a fast automatic estimate for every adjacent "
        "pair, visually check how each pair will be positioned, approve the good "
        "pairs, nudge the bad ones by hand in X and Y, then stitch the panorama "
        "using exactly these positions. "
        "The `Expected overlap` prior is NOT needed here — your eyes decide.")

    tiles, load_error = _get_tiles_cached()
    if tiles is None:
        if load_error:
            st.error(f"Loading error: {load_error}")
            st.error("Verify that filenames contain `_N` (e.g. `sample_1.tif`).")
        else:
            st.info("Upload images or a ZIP archive in the sidebar to begin.")
        return

    st.success(f"Loaded {len(tiles)} images")
    st.write(f"Review order: {' -> '.join(t.name for t in tiles)}")

    n = len(tiles)
    if n < 2:
        st.warning("At least 2 images are required.")
        return
    n_pairs = n - 1
    tile_names = [t.name for t in tiles]
    sig = _current_upload_signature()

    # =================== Step 1 — estimate ===================
    st.subheader("Step 1 — Automatic position estimate")
    st.caption(
        "Fast estimate on downscaled copies (the full SIFT + fallback pipeline, "
        "a few seconds for the whole row). Positions are reported in "
        "full-resolution pixels. You verify and correct them visually below.")

    c1, _ = st.columns([1, 2])
    with c1:
        max_side = st.selectbox(
            "Review resolution (max side, px)",
            [800, 1200, 1600, 2200], index=1, key="rv_max_side",
            help="Lower = faster estimation and previews. 1600-2200 gives a "
                 "more precise initial estimate at the cost of speed.")

    estimate_clicked = st.button("Estimate pair positions", type="primary",
                                 key="rv_estimate")

    if estimate_clicked:
        progress = st.progress(0.0, "Estimating...")

        def est_cb(msg, p):
            progress.progress(min(max(p, 0.0), 1.0), msg)

        try:
            imgs, scale = pair_review.downscale_tiles(
                [t.image for t in tiles], max_side)
            _wipe_pair_widgets()
            eo = _param("expected_overlap", 0)
            exp_dx_full = (tiles[0].image.shape[1] * (1.0 - eo / 100.0)
                           if eo > 0 else None)
            records = pair_review.estimate_pairs(
                imgs, scale,
                n_features=_param("n_features", 5000),
                ratio_thresh=_param("ratio_thresh", 0.75),
                ransac_thresh=_param("ransac_thresh", 5.0),
                use_clahe=_param("use_clahe", True),
                clahe_clip=_param("clahe_clip", 3.0),
                min_inliers_sift=_param("min_inliers", 8),
                enable_phase_fallback=_param("enable_phase", True),
                enable_ncc_fallback=_param("enable_ncc", True),
                verify_with_ncc=False,   # the user is the verifier here
                ncc_verify_gap=_param("ncc_verify_gap", 0.07),
                suppress_fixed_pattern=_param("suppress_fixed_pattern", True),
                expected_dx_full=exp_dx_full,
                progress_cb=est_cb,
            )
            st.session_state["review_state"] = {
                "sig": sig, "max_side": max_side, "scale": scale,
                "imgs": imgs, "pairs": records,
            }
            progress.progress(1.0, "Estimation complete")
        except Exception as e:
            st.error(f"Estimation error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Optional import of a previously saved review session
    with st.expander("Load positions from a saved review JSON (optional)"):
        imp = st.file_uploader("Review JSON file", type=["json"], key="rv_import")
        if imp is not None:
            try:
                imported = pair_review.apply_json(imp.getvalue(), tile_names)
            except ValueError as e:
                st.error(str(e))
            else:
                review_now = st.session_state.get("review_state")
                if (review_now is None or review_now.get("sig") != sig):
                    imgs, scale = pair_review.downscale_tiles(
                        [t.image for t in tiles], max_side)
                    records = []
                    for i, p in enumerate(imported):
                        records.append({
                            "pair_index": i, "dx": p["dx"], "dy": p["dy"],
                            "dx0": p["dx"], "dy0": p["dy"],
                            "method": "imported", "success": True,
                            "score": 0.0, "inliers": 0,
                            "message": p.get("note") or "imported from JSON",
                        })
                    _wipe_pair_widgets()
                    st.session_state["review_state"] = {
                        "sig": sig, "max_side": max_side, "scale": scale,
                        "imgs": imgs, "pairs": records,
                    }
                else:
                    _wipe_pair_widgets()
                    for i, p in enumerate(imported):
                        if i < len(review_now["pairs"]):
                            review_now["pairs"][i].update({
                                "dx": p["dx"], "dy": p["dy"],
                                "dx0": p["dx"], "dy0": p["dy"],
                                "method": "imported",
                                "message": p.get("note") or "imported from JSON",
                            })
                for i, p in enumerate(imported):
                    if i >= n_pairs:
                        break
                    st.session_state[f"rvp_ok_{i}"] = p["approved"]
                    st.session_state[f"rvp_no_{i}"] = bool(
                        p.get("no_overlap", False))
                    st.session_state[f"rvp_dx_{i}"] = int(round(p["dx"]))
                    st.session_state[f"rvp_dy_{i}"] = int(round(p["dy"]))
                st.success(f"Imported {len(imported)} pair positions from JSON.")

    review = st.session_state.get("review_state")
    if review is None:
        st.info("Run Step 1 to estimate all pair positions — or import a "
                "previously saved review JSON above.")
        return
    if review.get("sig") != sig:
        st.warning("The uploaded images changed since the last estimate — "
                   "please run Step 1 again.")
        return

    imgs = review["imgs"]
    scale = review["scale"]
    pairs = review["pairs"]
    w_full = int(tiles[0].image.shape[1])
    h_full = int(tiles[0].image.shape[0])

    mode_label = st.radio(
        "Overlay mode",
        ["Blend", "Checkerboard", "Difference"],
        horizontal=True, key="rv_mode",
        help="Blend: 50/50 mix in the overlap. Checkerboard: alternating "
             "squares of both tiles — the classic alignment inspector. "
             "Difference: |A-B| map, dark = perfectly aligned.")
    mode = {"Blend": "blend", "Checkerboard": "checker",
            "Difference": "difference"}[mode_label]

    # =================== Step 2 — review grid ===================
    st.subheader("Step 2 — Review every pair")
    qa1, qa2, qa3, _ = st.columns([1, 1, 1, 2])
    if qa1.button("Approve all pairs", key="rv_approve_all"):
        for i in range(n_pairs):
            st.session_state[f"rvp_ok_{i}"] = True
    if qa2.button("Clear all approvals", key="rv_clear_all"):
        for i in range(n_pairs):
            st.session_state[f"rvp_ok_{i}"] = False
    with qa3:
        st.number_input(
            "Gap width (px)",
            min_value=5, max_value=1000, value=pair_review.DEFAULT_GAP_PX,
            step=5, key="rv_gap_px",
            help="Width of the black band inserted between the tiles of a "
                 "pair marked 'No overlap (black gap)'. Applied at the "
                 "moment the checkbox is ticked; the pair's dx can still be "
                 "fine-tuned in Step 3 afterwards.")

    st.caption(
        "Tick **Position is correct** under each pair that is properly aligned. "
        "If two neighbouring shots were captured **without any overlap** (the "
        "automatic estimate is meaningless for them), tick **No overlap (black "
        "gap)** instead — the pair is placed side by side with a visible black "
        "band on the panorama. Unticked pairs are offered for manual adjustment "
        "in Step 3.")

    def _on_no_overlap(i_: int, rec_: dict) -> None:
        """on_change callback for the 'No overlap (black gap)' checkbox.

        Runs at the START of the rerun, before the rvp_* widgets are
        instantiated, so mutating session_state here is legal."""
        pair_review.apply_no_overlap_flag(
            st.session_state, i_, rec_,
            tile_width=w_full,
            gap_px=int(st.session_state.get("rv_gap_px",
                                            pair_review.DEFAULT_GAP_PX)),
        )

    def _on_pos_change(i_: int) -> None:
        """on_change callback for the dx/dy number inputs: sync the widget
        value into review_state['pairs'] — the non-widget source of truth
        (Streamlit deletes widget keys whose widgets stop rendering, e.g.
        after a pair is approved, so positions must live in the record)."""
        review_now = st.session_state.get("review_state")
        if review_now is None or i_ >= len(review_now["pairs"]):
            return
        rec_ = review_now["pairs"][i_]
        if f"rvp_dx_{i_}" in st.session_state:
            rec_["dx"] = float(st.session_state[f"rvp_dx_{i_}"])
        if f"rvp_dy_{i_}" in st.session_state:
            rec_["dy"] = float(st.session_state[f"rvp_dy_{i_}"])

    grid_cols = st.columns(2)
    for i in range(n_pairs):
        with grid_cols[i % 2]:
            rec = pairs[i]
            dx_cur = float(rec.get("dx", rec.get("dx0", 0.0)))
            dy_cur = float(rec.get("dy", rec.get("dy0", 0.0)))
            no_ovl_i = bool(st.session_state.get(f"rvp_no_{i}", False))
            prev = pair_review.render_pair_preview(
                imgs[i], imgs[i + 1], dx_cur * scale, dy_cur * scale,
                mode=mode, out_width=470)
            st.image(prev, use_container_width=True)
            st.markdown(f"**Pair {i+1} ↔ {i+2}** — `{tile_names[i]}` → "
                        f"`{tile_names[i+1]}`")
            ovl = max(0.0, 100.0 * (w_full - abs(dx_cur)) / w_full)
            caps = [f"dx = {dx_cur:.1f} px", f"dy = {dy_cur:.1f} px",
                    f"x-overlap ≈ {ovl:.0f} %", f"method: {rec['method']}"]
            if rec.get("inliers"):
                caps.append(f"{rec['inliers']} inliers")
            if not rec.get("success", True):
                caps.append("AUTOMATIC ESTIMATE FAILED")
            if no_ovl_i:
                caps.insert(0, "NO OVERLAP — black gap")
            st.caption(" · ".join(caps))
            okc, noc = st.columns(2)
            okc.checkbox("Position is correct", key=f"rvp_ok_{i}")
            noc.checkbox(
                "No overlap (black gap)", key=f"rvp_no_{i}",
                on_change=_on_no_overlap, args=(i, rec),
                help="Tick if these two shots were captured WITHOUT any "
                     "overlap: the tiles are placed side by side with a "
                     "black band between them on the panorama (dx = tile "
                     "width + gap, dy = 0). Unticking restores the previous "
                     "position.")

    # =================== Step 3 — manual adjustment ===================
    st.subheader("Step 3 — Manually adjust flagged pairs")
    unchecked = [i for i in range(n_pairs)
                 if not st.session_state.get(f"rvp_ok_{i}", False)]
    if not unchecked:
        st.success("All pairs approved — continue to Step 4.")
    else:
        st.caption(
            f"{len(unchecked)} pair(s) not approved yet. Nudge dx/dy below — "
            "the overlay updates live. **Increasing dx moves the right tile "
            "further right (tiles separate, overlap decreases)**; increasing "
            "dy moves it down.")

        for i in unchecked:
            rec = pairs[i]
            no_ovl_i = bool(st.session_state.get(f"rvp_no_{i}", False))

            def _reset_pair(i_=i, rec_=rec):
                """on_click callback: runs BEFORE the dx/dy widgets of the
                next rerun are instantiated, so mutating session_state is
                legal here (direct mutation after widget creation is not)."""
                st.session_state[f"rvp_dx_{i_}"] = int(round(
                    rec_.get("dx0", rec_["dx"])))
                st.session_state[f"rvp_dy_{i_}"] = int(round(
                    rec_.get("dy0", rec_["dy"])))
                rec_["dx"] = float(st.session_state[f"rvp_dx_{i_}"])
                rec_["dy"] = float(st.session_state[f"rvp_dy_{i_}"])

            title = (f"Pair {i+1} ↔ {i+2} — {tile_names[i]} → {tile_names[i+1]}")
            if no_ovl_i:
                title += " — NO OVERLAP (black gap)"
            elif not rec.get("success", True):
                title += " — automatic estimate FAILED, set the position manually"
            with st.expander(title, expanded=(len(unchecked) <= 3)):
                if no_ovl_i:
                    st.caption(
                        "This pair is marked **No overlap**: dx = tile width + "
                        "gap places the tiles side by side with a black vertical "
                        "band between them. Tweak dx (gap width) and dy below "
                        "if needed — the overlay updates live.")
                lc, rc = st.columns([3, 2])
                with rc:
                    step = st.selectbox("Nudge step (px)",
                                        [1, 2, 5, 10, 25, 50, 100],
                                        index=3, key=f"rvp_step_{i}")
                    bc = st.columns(4)
                    if bc[0].button("dx −", key=f"rvp_dxm_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dx_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dx_{i}", int(round(rec["dx"])))) - step
                        rec["dx"] = float(st.session_state[f"rvp_dx_{i}"])
                    if bc[1].button("dx +", key=f"rvp_dxp_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dx_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dx_{i}", int(round(rec["dx"])))) + step
                        rec["dx"] = float(st.session_state[f"rvp_dx_{i}"])
                    if bc[2].button("dy −", key=f"rvp_dym_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dy_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dy_{i}", int(round(rec["dy"])))) - step
                        rec["dy"] = float(st.session_state[f"rvp_dy_{i}"])
                    if bc[3].button("dy +", key=f"rvp_dyp_{i}",
                                    use_container_width=True):
                        st.session_state[f"rvp_dy_{i}"] = \
                            int(st.session_state.get(
                                f"rvp_dy_{i}", int(round(rec["dy"])))) + step
                        rec["dy"] = float(st.session_state[f"rvp_dy_{i}"])

                    if f"rvp_dx_{i}" not in st.session_state:
                        st.session_state[f"rvp_dx_{i}"] = int(round(rec["dx"]))
                    if f"rvp_dy_{i}" not in st.session_state:
                        st.session_state[f"rvp_dy_{i}"] = int(round(rec["dy"]))

                    dx = st.number_input(
                        "dx — horizontal offset of tile N+1 (px)",
                        min_value=-2 * w_full, max_value=3 * w_full, step=1,
                        key=f"rvp_dx_{i}", on_change=_on_pos_change, args=(i,),
                        help="Shift of the right tile relative to the left "
                             "one. Larger dx = tiles separate = less overlap.")
                    dy = st.number_input(
                        "dy — vertical offset of tile N+1 (px)",
                        min_value=-3 * h_full, max_value=3 * h_full, step=1,
                        key=f"rvp_dy_{i}", on_change=_on_pos_change, args=(i,),
                        help="Vertical shift of the right tile. Positive = "
                             "down.")
                    ovl = max(0.0, 100.0 * (w_full - abs(dx)) / w_full)
                    st.caption(f"x-overlap now ≈ {ovl:.1f} %")

                    st.button("Reset to automatic estimate",
                              key=f"rvp_reset_{i}", on_click=_reset_pair)
                    st.caption("When the overlay looks right, also tick "
                               "**Position is correct** for this pair in "
                               "Step 2 above.")
                with lc:
                    prev = pair_review.render_pair_preview(
                        imgs[i], imgs[i + 1], dx * scale, dy * scale,
                        mode=mode, out_width=760)
                    st.image(prev, use_container_width=True)

    # =================== Step 4 — stitch ===================
    st.subheader("Step 4 — Stitch the panorama with these positions")
    st.caption(
        "The panorama is blended using exactly the positions shown above "
        "(converted back to full-resolution pixels). Blending options come "
        "from the Method Settings in the sidebar.")

    with st.expander("Options"):
        refine = st.checkbox(
            "Refine approved pairs at full resolution (SIFT polish)",
            value=False, key="rv_refine",
            help="Re-runs SIFT at full resolution in a ±30 px window around "
                 "each approved position for sub-pixel accuracy. Manually "
                 "adjusted and no-overlap pairs are never touched. OFF "
                 "(default) = positions are used exactly as reviewed.")
        approved_flags = [bool(st.session_state.get(f"rvp_ok_{i}", False))
                          for i in range(n_pairs)]
        no_overlap_flags = [bool(st.session_state.get(f"rvp_no_{i}", False))
                            for i in range(n_pairs)]
        # Export the CURRENT positions (user-adjusted / no-overlap), not the
        # original automatic estimates.
        export_pairs = [{"dx": rec["dx"], "dy": rec["dy"],
                         "method": rec.get("method", "")}
                        for rec in pairs]
        st.download_button(
            "Export positions (JSON) — save your review work",
            data=pair_review.to_json(tile_names, export_pairs, approved_flags,
                                     no_overlap_flags),
            file_name="pair_review.json",
            mime="application/json",
            key="rv_export",
            help="Save all positions, approvals and no-overlap marks to a "
                 "JSON file. Load it later via 'Load positions from a saved "
                 "review JSON' above.")

    if st.button("Stitch panorama with these positions", type="primary",
                 key="rv_stitch"):
        final_translations: List[Tuple[float, float]] = []
        manual_flags: List[bool] = []
        no_flags: List[bool] = []
        for i in range(n_pairs):
            rec_i = pairs[i]
            # review_state['pairs'] is the source of truth: widget-backed
            # session keys are deleted by Streamlit once a pair is approved
            # (its number inputs stop rendering), but the record survives.
            dx_i = float(rec_i.get("dx",
                                   st.session_state.get(f"rvp_dx_{i}",
                                                        rec_i.get("dx0", 0.0))))
            dy_i = float(rec_i.get("dy",
                                   st.session_state.get(f"rvp_dy_{i}",
                                                        rec_i.get("dy0", 0.0))))
            dx0 = float(rec_i.get("dx0", dx_i))
            dy0 = float(rec_i.get("dy0", dy_i))
            is_manual = (abs(dx_i - dx0) > 0.5 or abs(dy_i - dy0) > 0.5)
            manual_flags.append(is_manual)
            no_flags.append(bool(st.session_state.get(f"rvp_no_{i}", False)))
            final_translations.append((dx_i, dy_i))

        refined_flags = [False] * n_pairs
        if refine:
            try:
                with st.spinner("Refining approved pairs at full resolution..."):
                    for i in range(n_pairs):
                        if manual_flags[i] or no_flags[i] \
                                or not pairs[i].get("success", True):
                            continue
                        r = stitching.stitch_pair_sift(
                            tiles[i].image, tiles[i + 1].image,
                            n_features=_param("n_features", 5000),
                            ratio_thresh=_param("ratio_thresh", 0.75),
                            ransac_thresh=_param("ransac_thresh", 5.0),
                            use_clahe=_param("use_clahe", True),
                            clahe_clip=_param("clahe_clip", 3.0),
                            suppress_fixed_pattern=True,
                            prior_translation=final_translations[i],
                            prior_window=30.0,
                        )
                        if r.success and r.inliers >= _param("min_inliers", 8):
                            final_translations[i] = r.translation
                            refined_flags[i] = True
            except Exception as e:
                st.warning(f"Refinement failed ({e}) — using reviewed positions as is.")

        methods = ["no-overlap" if no_flags[i]
                   else ("manual" if manual_flags[i]
                         else ("refined" if refined_flags[i] else "reviewed"))
                   for i in range(n_pairs)]

        progress = st.progress(0.0, "Blending panorama...")

        def stitch_cb(msg, p):
            progress.progress(min(max(p, 0.0), 1.0), msg)

        try:
            result = pipeline.stitch_row(
                tiles,
                blend_levels=_param("blend_levels", 5),
                blend_feather=_param("blend_feather", 64),
                blend_gamma=_param("blend_gamma", 1.5),
                blend_gain_strength=_param("gain_strength", 0.8),
                blend_use_focus_weights=_param("use_focus_weights", False),
                pair_translations=final_translations,
                pair_methods=methods,
                progress_callback=stitch_cb,
            )
        except Exception as e:
            st.error(f"Stitching error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

        render_stitch_result(result, tiles)


# === Render the tabs ===
tab_stitcher, tab_review, tab_help = st.tabs(["Stitcher", "Pair Review", "Help"])

with tab_stitcher:
    render_stitcher_tab()

with tab_review:
    render_pair_review_tab()

with tab_help:
    render_help_tab()
