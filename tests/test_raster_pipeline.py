from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from hsarfcd.change_detection import detect
from hsarfcd.inference import translate_raster
from hsarfcd.models import MultiScaleGenerator
from hsarfcd.train import train


def _write(path: Path, data: np.ndarray) -> None:
    if data.ndim == 2:
        data = data[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:32608",
        transform=from_origin(500000, 7600000, 10, 10),
    ) as dst:
        dst.write(data.astype(np.float32))


def test_tiled_translation_writes_georeferenced_raster(tmp_path):
    input_path = tmp_path / "uavsar.tif"
    output_path = tmp_path / "translated.tif"
    checkpoint_path = tmp_path / "model.pt"
    image = np.linspace(0, 1, 40 * 40, dtype=np.float32).reshape(40, 40)
    _write(input_path, image)

    model = MultiScaleGenerator(base_channels=8, residual_blocks=3)
    torch.save(
        {
            "train_config": {
                "model": {"channels": 3, "base_channels": 8, "residual_blocks": 3}
            },
            "generator_ba": model.state_dict(),
        },
        checkpoint_path,
    )
    translate_raster(
        {
            "device": "cpu",
            "direction": "B_to_A",
            "checkpoint": str(checkpoint_path),
            "input": str(input_path),
            "output": str(output_path),
            "bands": [1],
            "repeat_single_band": True,
            "tile_size": 32,
            "overlap": 8,
            "batch_size": 2,
            "normalization": {"mode": "minmax", "output_range": [-1, 1]},
        }
    )
    with rasterio.open(output_path) as result:
        output = result.read()
        assert result.count == 3
        assert result.crs.to_epsg() == 32608
        assert output.shape == (3, 40, 40)
        assert np.isfinite(output).all()
        assert 0.0 <= output.min() <= output.max() <= 1.0


def test_full_change_detection_writes_all_products(tmp_path):
    before_path = tmp_path / "before.tif"
    after_path = tmp_path / "after.tif"
    before = np.zeros((3, 64, 64), dtype=np.float32)
    after = before.copy()
    after[:, 20:44, 20:44] = 1.0
    _write(before_path, before)
    _write(after_path, after)
    outputs = detect(
        {
            "before": str(before_path),
            "after": str(after_path),
            "output_dir": str(tmp_path / "changes"),
            "bands": [1, 2, 3],
            "gmm": {"max_samples": 10_000, "random_state": 4},
            "morphology": {
                "opening_radius": 0,
                "closing_radius": 0,
                "min_object_size": 10,
                "min_hole_size": 0,
            },
        }
    )
    assert set(outputs) == {"magnitude", "probability", "raw", "filtered", "overlay", "gmm"}
    assert all(path.exists() for path in outputs.values())
    with rasterio.open(outputs["filtered"]) as result:
        mask = result.read(1)
        assert mask[30, 30] == 1
        assert mask[0, 0] == 0
        assert result.crs.to_epsg() == 32608


def test_one_step_training_and_checkpoint(tmp_path):
    data_root = tmp_path / "dataset"
    for split in ("train", "val"):
        for domain in ("A", "B"):
            directory = data_root / split / domain
            directory.mkdir(parents=True)
            values = np.random.default_rng(7).random((3, 64, 64), dtype=np.float32)
            if domain == "B":
                values = 1.0 - values
            _write(directory / "tile_00001.tif", values)
    data_config = {
        "dataset": {
            "root": str(data_root),
            "layout": "paired_dirs",
            "train": {"domain_a": "train/A", "domain_b": "train/B"},
            "val": {"domain_a": "val/A", "domain_b": "val/B"},
            "bands_a": [1, 2, 3],
            "bands_b": [1, 2, 3],
            "repeat_single_band": False,
            "normalization": {"mode": "minmax", "output_range": [-1, 1]},
        }
    }
    output_dir = tmp_path / "run"
    train(
        data_config,
        {
            "seed": 1,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 0.0002,
            "n_critic": 1,
            "lambda_cycle": 10,
            "lambda_gp": 1,
            "sample_interval": 0,
            "checkpoint_interval": 1,
            "output_dir": str(output_dir),
            "model": {"channels": 3, "base_channels": 4, "residual_blocks": 1},
        },
    )
    checkpoint = torch.load(
        output_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 0
    assert checkpoint["step"] == 1
    assert (output_dir / "metrics.jsonl").exists()
