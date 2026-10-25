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

import cv2
import numpy as np
import streamlit as st

# Add path to src
sys.path.insert(0, str(Path(__file__).parent))

from src import io_utils, pipeline, preprocessing, stitching


# === Page configuration ===
st.set_page_config(
    page_title="Optical Microscopy Stitcher",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "3. **Run**: click **Stitch panorama**."
        "\n"
        "4. **Download**: panorama (PNG/TIFF/JPEG) and tile positions CSV."
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
        "   - Lowe ratio test → good matches"
        "\n"
        "   - RANSAC + `estimateAffinePartial2D` → translation + small rotation"
        "\n"
        "   - Fallbacks: phase correlation → NCC scan"
        "\n"
        "4. **Cumulative position calculation** — chain pairwise translations."
        "\n"
        "5. **Multi-band blending** with focus-aware feather masks:"
        "\n"
        "   - Laplacian pyramid (5 levels by default)"
        "\n"
        "   - Weighted average per level"
        "\n"
        "   - Gain compensation between tiles"
        "\n"
        "6. **Output**: panorama + diagnostics table + tile positions CSV."
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
        )
        uploaded_files = st.file_uploader(
            "Select images",
            type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        ) if upload_mode == "Multiple files" else None

        uploaded_zip = st.file_uploader(
            "Select ZIP archive",
            type=["zip"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        ) if upload_mode == "ZIP archive" else None

    with st.sidebar.expander("Feature Detection", expanded=True):
        n_features = st.slider("SIFT features (per tile)", 500, 10000, 5000, step=500,
                                help="More = more robust but slower. 5000 is a good balance.")
        ratio_thresh = st.slider("Lowe ratio threshold", 0.5, 0.9, 0.75, step=0.05,
                                  help="Lower = stricter match filtering. 0.75 is standard.")
        ransac_thresh = st.slider("RANSAC reprojection threshold (px)", 1.0, 15.0, 5.0, step=0.5,
                                   help="Allowed inlier error. 5 px is standard.")
        min_inliers = st.slider("Min inliers for SIFT", 4, 50, 8, step=1,
                                 help="Below this — fall back to alternative methods.")

    with st.sidebar.expander("Preprocessing", expanded=False):
        use_clahe = st.checkbox("Apply CLAHE (recommended)", value=True,
                                 help="Adaptive histogram equalization. Critical for dark microscope images.")
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 8.0, 3.0, step=0.5)

    with st.sidebar.expander("Blending", expanded=False):
        blend_levels = st.slider("Laplacian pyramid levels", 2, 8, 5, step=1,
                                  help="More = smoother seams but slower.")
        blend_feather = st.slider("Feather zone width (px)", 0, 200, 64, step=8,
                                   help="Smooth mask falloff toward tile edges.")
        use_focus_weights = st.checkbox("Use focus-weighted mask", value=False,
                                         help="Weighting by sharpness map. "
                                              "OFF (default) — simple feather mask, more stable on dark images. "
                                              "ON — better for tiles with blurred edges.")
        blend_gamma = st.slider("Gamma for focus weights", 0.5, 3.0, 1.5, step=0.1,
                                 help=">1 — sharp zones dominate more. "
                                      "Only used when focus-weighted mask is enabled.")
        gain_strength = st.slider("Gain compensation", 0.0, 1.0, 0.8, step=0.1,
                                   help="Brightness equalization between tiles. "
                                        "0 = off, 1 = full correction.")

    with st.sidebar.expander("Backup Methods", expanded=False):
        enable_phase = st.checkbox("Phase correlation fallback", value=True)
        enable_ncc = st.checkbox("NCC scan fallback (slow)", value=True)

    with st.sidebar.expander("Output Format", expanded=False):
        out_format = st.selectbox("Output format", ["PNG", "TIFF (LZW)", "JPEG"], index=0)

    # === Main content ===
    has_files = (upload_mode == "Multiple files" and uploaded_files) or \
                (upload_mode == "ZIP archive" and uploaded_zip is not None)

    if not has_files:
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

    # Load files
    def save_uploaded_to_temp(files, source_name: str) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"micro_{source_name}_"))
        for f in files:
            out_path = tmp_dir / f.name
            out_path.write_bytes(f.getbuffer())
        return tmp_dir

    def extract_zip_to_temp(zip_file) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="micro_zip_"))
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(tmp_dir)
        return tmp_dir

    try:
        if upload_mode == "Multiple files":
            data_dir = save_uploaded_to_temp(uploaded_files, "files")
        else:
            data_dir = extract_zip_to_temp(uploaded_zip)
    except Exception as e:
        st.error(f"Error during unpacking: {e}")
        return

    # Load tiles
    with st.spinner("Loading images..."):
        try:
            tiles = io_utils.load_tiles(data_dir)
        except Exception as e:
            st.error(f"Loading error: {e}")
            st.error("Verify that filenames contain `_N` (e.g. `sample_1.tif`).")
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

    # === Result ===
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
            "Note": d.message,
        })
    st.dataframe(diag_data, use_container_width=True)

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
    positions_text = "\n".join(
        f"  {t.name}: ({pos[0]}, {pos[1]})"
        for t, pos in zip(tiles, result.tile_positions)
    )
    st.code(positions_text)

    # Download buttons
    st.subheader("Download")
    ext_map = {"PNG": "png", "TIFF (LZW)": "tif", "JPEG": "jpg"}
    ext = ext_map[out_format]

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


# === Render the two tabs ===
tab_stitcher, tab_help = st.tabs(["Stitcher", "Help"])

with tab_stitcher:
    render_stitcher_tab()

with tab_help:
    render_help_tab()
