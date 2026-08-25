# Real workspace sample demo

This record documents a functional run on real Aklavik SAR chips found in the
original workspace. The GeoTIFF inputs, temporary dataset, checkpoint and
GeoTIFF outputs are **not** included in this repository. Only reduced PNG
previews are committed under `docs/images/`.

## Reproducible selection

Fixed Python random seed: `42`.

| Pool | Candidates | Selected file | Use |
| --- | ---: | --- | --- |
| Filenames common to `2022_4_threebands/` and `2024_4_threebands/` | 70 | `3_0015.tif` | Real same-modality change detection |
| Four-band files in `patch3/train/` | 484 | `2_0104.tif` | A/B split, training and tiled inference smoke test |

The primary result is `3_0015.tif`. The supplementary translation chip is
needed because the change-detection pool contains Sentinel-1 pairs but no
UAVSAR domain band.

Original workspace-relative locations:

```text
PyTorch-GAN-master/data/change_detection/2022_4_threebands/3_0015.tif
PyTorch-GAN-master/data/change_detection/2024_4_threebands/3_0015.tif
PyTorch-GAN-master/data/patch3/train/2_0104.tif
```

All rasters are 256×256 Float32 GeoTIFF chips in EPSG:32608. The change pair
has three bands. The translation source has four bands and was split as A =
bands 1/2/3 and B = band 4.

## Operations exercised

1. `hsarfcd-prepare split-combined`: converted the legacy four-band chip to
   explicit paired A/B files while retaining georeferencing.
2. `hsarfcd-validate`: verified train/val/test directories, filenames, bands,
   dimensions, CRS and affine transforms.
3. `hsarfcd-train`: ran 25 CPU updates with a reduced 8-channel, one-residual-
   block model and wrote metrics, samples and checkpoints.
4. `hsarfcd-translate`: loaded the checkpoint and ran B→A overlap-tiled
   GeoTIFF inference.
5. `hsarfcd-detect`: aligned the 2024 chip to the 2022 grid, computed the
   three-band magnitude, fit a two-component GMM, applied an after-image
   low-backscatter water mask and performed morphology.
6. `scripts/make_demo_figures.py`: generated the committed previews.

The reduced translation run proves code integration only. One chip and 25
updates cannot estimate cross-scene model quality and should not be compared
with the paper's trained model.

## Observed change output

Configuration: GMM seed 42, full covariance, opening radius 1, closing radius
2, minimum object size 100, minimum hole size 60, and an Otsu low-backscatter
mask from after-image band 1.

| Quantity | Observed value |
| --- | ---: |
| GMM means | 0.338928 / 0.193632 |
| GMM weights | 0.451396 / 0.548604 |
| Water-mask Otsu threshold | 0.530433 |
| Water-constrained raw change | 6,889 pixels / 10.512% |
| Morphology-filtered change | 2,156 pixels / 3.290% |
| Filtered connected components | 11 |
| Approximate detected area | 0.2145 km² |

No manual reference mask was found for this chip. Consequently, these are
descriptive output statistics, not accuracy metrics or an authoritative flood
area estimate. `hsarfcd-evaluate` was not used to fabricate a score.

## Regenerating the panels

Install the optional plotting dependency and pass local paths to the actual
inputs and generated outputs:

```bash
pip install -e '.[demo]'

python scripts/make_demo_figures.py \
  --before /path/to/2022_4_threebands/3_0015.tif \
  --after /path/to/2024_4_threebands/3_0015.tif \
  --change-dir /path/to/change_outputs \
  --domain-a /path/to/test/A/2_0104.tif \
  --domain-b /path/to/test/B/2_0104.tif \
  --translated /path/to/2_0104_B_to_A.tif \
  --output-dir docs/images
```
