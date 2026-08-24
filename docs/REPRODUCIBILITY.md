# Reproducibility notes

## What is reproduced faithfully

- two-domain bidirectional translation;
- WGAN-GP adversarial objective;
- L1 cycle-consistency objective;
- multi-scale parallel encoding;
- parallel dilated convolutions at rates 1, 2 and 4;
- three decoding branches;
- spatial attention fusion;
- image differencing;
- Gaussian mixture classification of changed pixels;
- morphological refinement;
- geospatial raster output and standard binary metrics.

## Parameters recovered from the original workspace

| Parameter | Value |
| --- | ---: |
| Epochs | 300 |
| Batch size | 8 |
| Adam learning rate | 0.0002 |
| Adam beta1 / beta2 | 0.5 / 0.999 |
| Critic interval | 5 |
| Cycle weight | 10 |
| Gradient penalty weight | 10 |
| Input channels | 3 |
| Default chip size | 256×256 |
| Checkpoint interval | 10 epochs |

## Known uncertainty

The original experiment directory is not a valid Git repository and therefore
does not contain a usable commit history. The retained source has two important
inconsistencies:

1. the paper architecture is present as commented code while a baseline U-Net
   is active at the top of the file;
2. the paper abstract specifies GMM, while retained notebook cells mainly use
   Otsu thresholding.

This release resolves those inconsistencies in favor of the published method.
The generator was made internally consistent for `[-1,1]` training and the GMM
classifier was implemented explicitly. Old checkpoints should be treated as
historical artifacts unless their exact source revision is recovered.

## Checklist for reproducing reported numbers

- [ ] Record checksums for all four raw scenes.
- [ ] Record calibration, speckle filtering, terrain correction and resampling parameters.
- [ ] Freeze spatial train/val/test polygons and publish the split manifest.
- [ ] Confirm which sensor is domain A and which is domain B for each table/figure.
- [ ] Confirm normalization statistics are per-chip or dataset/global.
- [ ] Recover the exact paper checkpoint or retrain this implementation.
- [ ] Recover GMM covariance type, sample count and random seed.
- [ ] Recover morphology kernel sizes and minimum component area.
- [ ] Add the paper's exact reference masks and metric table.
- [ ] Run at least three seeds and report mean ± standard deviation.

Until these items are confirmed, describe results as a method reproduction, not
as an exact numerical replication of every published value.

