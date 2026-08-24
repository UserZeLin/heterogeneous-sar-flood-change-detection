from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


@dataclass(frozen=True)
class NormalizationSpec:
    mode: str = "percentile"
    lower: float = 1.0
    upper: float = 99.0
    output_range: tuple[float, float] = (-1.0, 1.0)

    @classmethod
    def from_mapping(cls, value: dict | None) -> "NormalizationSpec":
        value = value or {}
        output_range = value.get("output_range", [-1.0, 1.0])
        return cls(
            mode=str(value.get("mode", "percentile")),
            lower=float(value.get("lower", 1.0)),
            upper=float(value.get("upper", 99.0)),
            output_range=(float(output_range[0]), float(output_range[1])),
        )


def read_raster(path: str | Path, bands: Sequence[int] | None = None) -> tuple[np.ndarray, dict]:
    """Read a raster as float32 in band-first order and return a writable profile."""
    with rasterio.open(path) as src:
        indexes = list(bands) if bands is not None else list(range(1, src.count + 1))
        data = src.read(indexes=indexes, out_dtype="float32")
        profile = src.profile.copy()
    profile.update(count=len(indexes), dtype="float32")
    return data, profile


def write_raster(path: str | Path, data: np.ndarray, profile: dict, nodata=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        compress="deflate",
        tiled=True,
        bigtiff="if_safer",
        nodata=nodata,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data)


def finite_percentiles(data: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for band in data:
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            lows.append(0.0)
            highs.append(1.0)
        else:
            lo, hi = np.percentile(valid, [lower, upper])
            if hi <= lo:
                hi = lo + 1.0
            lows.append(float(lo))
            highs.append(float(hi))
    return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)


def normalize(
    data: np.ndarray,
    spec: NormalizationSpec,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Normalize each band to a configured range and return the applied statistics."""
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if spec.mode == "none":
        zeros = np.zeros(data.shape[0], dtype=np.float32)
        ones = np.ones(data.shape[0], dtype=np.float32)
        return data, (zeros, ones)
    if spec.mode not in {"percentile", "minmax"}:
        raise ValueError(f"Unsupported normalization mode: {spec.mode}")
    if stats is None:
        if spec.mode == "percentile":
            lows, highs = finite_percentiles(data, spec.lower, spec.upper)
        else:
            lows = np.nanmin(data, axis=(1, 2)).astype(np.float32)
            highs = np.nanmax(data, axis=(1, 2)).astype(np.float32)
            highs = np.where(highs <= lows, lows + 1.0, highs)
    else:
        lows, highs = stats
    scaled = (data - lows[:, None, None]) / (highs - lows)[:, None, None]
    scaled = np.clip(scaled, 0.0, 1.0)
    out_min, out_max = spec.output_range
    scaled = scaled * (out_max - out_min) + out_min
    return scaled.astype(np.float32), (lows, highs)


def denormalize(data: np.ndarray, spec: NormalizationSpec, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if spec.mode == "none":
        return data.astype(np.float32)
    lows, highs = stats
    out_min, out_max = spec.output_range
    unit = (data - out_min) / (out_max - out_min)
    return (unit * (highs - lows)[:, None, None] + lows[:, None, None]).astype(np.float32)


def repeat_to_channels(data: np.ndarray, channels: int = 3) -> np.ndarray:
    if data.shape[0] == channels:
        return data
    if data.shape[0] != 1:
        raise ValueError(f"Cannot repeat {data.shape[0]} bands to {channels} channels")
    return np.repeat(data, channels, axis=0)


def align_to_profile(data: np.ndarray, source_profile: dict, reference_profile: dict) -> np.ndarray:
    """Reproject band-first data onto a reference grid."""
    aligned = np.empty(
        (data.shape[0], reference_profile["height"], reference_profile["width"]),
        dtype=np.float32,
    )
    for band_index in range(data.shape[0]):
        reproject(
            source=data[band_index],
            destination=aligned[band_index],
            src_transform=source_profile["transform"],
            src_crs=source_profile["crs"],
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return aligned


def stack_rasters(paths: Iterable[str | Path], bands: Iterable[int] | None = None) -> tuple[np.ndarray, dict]:
    selected_bands = list(bands) if bands is not None else None
    arrays: list[np.ndarray] = []
    reference_profile: dict | None = None
    for path in paths:
        data, profile = read_raster(path, selected_bands)
        if reference_profile is None:
            reference_profile = profile
        elif (
            profile["height"] != reference_profile["height"]
            or profile["width"] != reference_profile["width"]
            or profile["transform"] != reference_profile["transform"]
            or profile["crs"] != reference_profile["crs"]
        ):
            data = align_to_profile(data, profile, reference_profile)
        arrays.append(data)
    if reference_profile is None:
        raise ValueError("At least one raster path is required")
    return np.concatenate(arrays, axis=0), reference_profile

