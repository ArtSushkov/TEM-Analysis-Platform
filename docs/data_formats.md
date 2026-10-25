# Data Formats

## Supported Formats

### Optical Microscope

| Format | Description | Used in |
|---|---|---|
| TIFF (LZW) | Primary format, supports EXIF | 01_optical_stitcher |
| PNG | Simple lossless format | 01_optical_stitcher |
| JPEG | Lossy compression (not recommended) | 01_optical_stitcher |

## Filename Conventions

### Image series for stitching

```
<prefix>_<N>.<extension>
```

where `N` is the sequential index in the series.

Examples:
- `x10_1.tif`, `x10_2.tif`, ..., `x10_5.tif`
- `sample_001.tif`, `sample_002.tif`, ..., `sample_015.tif`

The `01_optical_stitcher` app automatically determines the order by `N`.

## Capture Recommendations

### For panorama stitching

- Overlap of adjacent frames: **20–40%** (30% optimal)
- Illumination: **uniform**, without abrupt changes between frames
- Focus: if the specimen is uneven, focus variation is acceptable — blending will compensate
- Number of frames: up to 100 for a single row (for more, split into series)
