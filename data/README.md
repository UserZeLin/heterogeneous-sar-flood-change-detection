# Dataset paths and types

This directory intentionally contains no imagery. Use local paths in
`configs/data.local.yaml`; do not commit that file if it reveals private or
machine-specific locations.

## Paper scenes

| Scene | Sensor | Product/type | Polarization | Acquisition | Expected local object |
| --- | --- | --- | --- | --- | --- |
| Aklavik 2022 | UAVSAR | GRD, `.grd` + `.ann` | L-band HHHH | 2022-08-22 | `aklavi_25702_22037_006_220822_L090HHHH_CX_01.grd` |
| Aklavik 2022 | Sentinel-1A | IW GRDH SAFE | C-band VV+VH | 2022-10-31 | `S1A_IW_GRDH_1SDV_20221031T022450_...FCA3.SAFE` |
| Aklavik 2024 | UAVSAR | GRD, `.grd` + `.ann` | L-band HHHH | 2024-08-18 | `aklavi_25702_24057_004_240818_L090HHHH_CX_01.grd` |
| Aklavik 2024 | Sentinel-1A | IW GRDH SAFE | C-band VV+VH | 2024-08-21 | `S1A_IW_GRDH_1SDV_20240821T022450_...DB44.SAFE` |

The exact identifiers are also provided in `manifest.example.csv`. Replace its
paths with local paths if you maintain a private provenance manifest.

## Data types used by the pipeline

1. **Raw Sentinel-1 SAFE product**: measurement TIFFs plus annotation,
   calibration, noise, manifest and orbit-related metadata.
2. **Raw UAVSAR GRD**: floating-point raster plus `.ann` metadata.
3. **Processed full-scene GeoTIFF**: geocoded/reprojected VV, VH, UAVSAR and
   merged rasters; Float32 with valid CRS/transform.
4. **Normalized full-scene GeoTIFF**: band-wise 1–99 percentile normalization.
5. **Paired translation chips**: 256×256 GeoTIFFs with identical A/B names.
6. **Reference change mask**: optional single-band binary GeoTIFF, positive
   pixels encoded by any value greater than zero.

## Required external preprocessing

The repository handles dB conversion, alignment of already readable rasters,
band stacking, normalization and tiling. Converting a UAVSAR `.grd` or a
Sentinel SAFE product to a calibrated/geocoded GeoTIFF may require provider
tools, ESA SNAP, GDAL, or the `uavsar_pytools` project. Keep those raw-product
steps documented in the private manifest.

Minimum requirements before `hsarfcd-prepare tile-pair`:

- both rasters have a CRS;
- both rasters cover the same spatial region;
- pixel size, transform, width and height match;
- nodata is known and masked or replaced consistently;
- channels and physical units are recorded.

## Recommended local layout

```text
hsarfcd-data/
├── raw/
│   ├── 2022/sentinel1/
│   ├── 2022/uavsar/
│   ├── 2024/sentinel1/
│   └── 2024/uavsar/
├── processed/
│   ├── 2022/
│   └── 2024/
├── translation/
│   ├── train/A and B
│   ├── val/A and B
│   └── test/A and B
└── labels/
```

Do not upload raw or processed imagery to GitHub. Publish download instructions,
checksums, split manifests and small licensed previews instead.

