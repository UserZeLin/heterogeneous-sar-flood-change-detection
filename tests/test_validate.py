from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from hsarfcd.validate import validate_data_config


def _write_chip(path: Path, bands: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=bands,
        dtype="float32",
        crs="EPSG:32608",
        transform=from_origin(0, 16, 1, 1),
    ) as dst:
        dst.write(np.ones((bands, 16, 16), dtype=np.float32))


def test_validate_complete_dataset(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_chip(tmp_path / split / "A" / "tile_000.tif", 3)
        _write_chip(tmp_path / split / "B" / "tile_000.tif", 1)
    config = {
        "dataset": {
            "root": str(tmp_path),
            "layout": "paired_dirs",
            "file_glob": "*.tif",
            "bands_a": [1, 2, 3],
            "bands_b": [1],
            **{
                split: {"domain_a": f"{split}/A", "domain_b": f"{split}/B"}
                for split in ("train", "val", "test")
            },
        }
    }
    report = validate_data_config(config)
    assert report["valid"]
    assert report["splits"]["train"]["paired_count"] == 1


def test_validate_reports_unpaired_files(tmp_path: Path) -> None:
    _write_chip(tmp_path / "train" / "A" / "only_a.tif", 3)
    (tmp_path / "train" / "B").mkdir(parents=True)
    config = {
        "dataset": {
            "root": str(tmp_path),
            "train": {"domain_a": "train/A", "domain_b": "train/B"},
        }
    }
    report = validate_data_config(config)
    assert not report["valid"]
    assert any("filenames differ" in error for error in report["errors"])
