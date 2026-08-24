from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .raster import NormalizationSpec, normalize, repeat_to_channels


class PairedSARDataset(Dataset):
    """Paired GeoTIFF dataset with matching filenames in domain A and B."""

    def __init__(
        self,
        domain_a: str | Path,
        domain_b: str | Path,
        file_glob: str = "*.tif",
        bands_a: list[int] | None = None,
        bands_b: list[int] | None = None,
        repeat_single_band: bool = True,
        normalization: dict[str, Any] | None = None,
        augment: bool = False,
    ) -> None:
        self.domain_a = Path(domain_a)
        self.domain_b = Path(domain_b)
        self.files_a = sorted(self.domain_a.glob(file_glob))
        self.files_b = sorted(self.domain_b.glob(file_glob))
        names_a = [path.name for path in self.files_a]
        names_b = [path.name for path in self.files_b]
        if not self.files_a:
            raise ValueError(f"No files matching {file_glob!r} in {self.domain_a}")
        if names_a != names_b:
            only_a = sorted(set(names_a) - set(names_b))[:5]
            only_b = sorted(set(names_b) - set(names_a))[:5]
            raise ValueError(f"A/B filenames do not match; only A={only_a}, only B={only_b}")
        self.bands_a = bands_a
        self.bands_b = bands_b
        self.repeat_single_band = repeat_single_band
        self.spec = NormalizationSpec.from_mapping(normalization)
        self.augment = augment

    @staticmethod
    def _read(path: Path, bands: list[int] | None) -> np.ndarray:
        with rasterio.open(path) as src:
            indexes = bands or list(range(1, src.count + 1))
            return src.read(indexes=indexes, out_dtype="float32")

    def __len__(self) -> int:
        return len(self.files_a)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path_a = self.files_a[index]
        path_b = self.files_b[index]
        array_a = self._read(path_a, self.bands_a)
        array_b = self._read(path_b, self.bands_b)
        if self.repeat_single_band:
            array_a = repeat_to_channels(array_a)
            array_b = repeat_to_channels(array_b)
        if array_a.shape != array_b.shape:
            raise ValueError(f"Shape mismatch for {path_a.name}: {array_a.shape} vs {array_b.shape}")
        array_a, _ = normalize(array_a, self.spec)
        array_b, _ = normalize(array_b, self.spec)
        if self.augment and np.random.random() < 0.5:
            array_a = array_a[:, :, ::-1].copy()
            array_b = array_b[:, :, ::-1].copy()
        return {
            "A": torch.from_numpy(array_a),
            "B": torch.from_numpy(array_b),
            "name": path_a.stem,
        }


def dataset_from_config(data_config: dict[str, Any], split: str, augment: bool) -> PairedSARDataset:
    dataset = data_config["dataset"]
    if dataset.get("layout", "paired_dirs") != "paired_dirs":
        raise ValueError("This release supports the explicit paired_dirs layout")
    root = Path(dataset["root"]).expanduser()
    split_config = dataset[split]
    return PairedSARDataset(
        root / split_config["domain_a"],
        root / split_config["domain_b"],
        file_glob=dataset.get("file_glob", "*.tif"),
        bands_a=dataset.get("bands_a"),
        bands_b=dataset.get("bands_b"),
        repeat_single_band=bool(dataset.get("repeat_single_band", True)),
        normalization=dataset.get("normalization"),
        augment=augment,
    )

