# Mapping from the original workspace

No original dataset or checkpoint is copied into this GitHub package.

| Original location | Role | New location |
| --- | --- | --- |
| `data/flood/aklavi_*` | Raw UAVSAR/Sentinel scenes | `data/manifest.example.csv` paths only |
| `data/flood/2022`, `2024` | Full-scene preprocessing | `hsarfcd.preprocess`, `data/README.md` |
| `data/flood/2022_patch`, `2024_patch` | ROI/normalized/stacked products | configuration only |
| `PyTorch-GAN-master/data/patch*` | Training chips | excluded; expected layout documented |
| `dualgan/datasets.py` | GDAL chip loader | `hsarfcd.dataset` |
| `dualgan/models.py` | Baseline and commented paper architecture | `hsarfcd.models` |
| `dualgan/dualgan.py` | WGAN-GP/cycle training | `hsarfcd.train` |
| `dualgan/inference_tile.ipynb` | Tiled model inference | `hsarfcd.inference` |
| `dualgan/change_detection.ipynb` | Difference and morphology experiments | `hsarfcd.change_detection` |
| `data/flood/change_*.tif` | Experiment outputs | excluded; output names reproduced |
| `dualgan/saved_models` | Historical checkpoints | excluded; new checkpoint format documented |

The new code avoids absolute paths and repeated notebook cells, preserves
GeoTIFF metadata, provides deterministic configuration, and separates data,
model, inference, detection and evaluation responsibilities.

