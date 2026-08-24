from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm import tqdm

from .config import load_yaml
from .models import MultiScaleGenerator
from .raster import NormalizationSpec, finite_percentiles, repeat_to_channels
from .utils import select_device


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def blending_weight(tile_size: int) -> np.ndarray:
    axis = np.hanning(tile_size).astype(np.float32)
    axis = np.clip(axis, 0.05, None)
    return np.outer(axis, axis).astype(np.float32)


def _normalization_stats(src: rasterio.io.DatasetReader, bands: list[int], spec: NormalizationSpec):
    sample_height = min(src.height, 2048)
    sample_width = min(src.width, 2048)
    sample = src.read(
        indexes=bands,
        out_shape=(len(bands), sample_height, sample_width),
        out_dtype="float32",
    )
    if spec.mode == "percentile":
        return finite_percentiles(sample, spec.lower, spec.upper)
    lows = np.nanmin(sample, axis=(1, 2)).astype(np.float32)
    highs = np.nanmax(sample, axis=(1, 2)).astype(np.float32)
    return lows, np.where(highs <= lows, lows + 1.0, highs)


def _normalize_tile(tile: np.ndarray, spec: NormalizationSpec, stats) -> np.ndarray:
    if spec.mode == "none":
        return np.nan_to_num(tile, nan=0.0).astype(np.float32)
    lows, highs = stats
    unit = (np.nan_to_num(tile, nan=0.0) - lows[:, None, None]) / (
        highs - lows
    )[:, None, None]
    unit = np.clip(unit, 0.0, 1.0)
    low, high = spec.output_range
    return (unit * (high - low) + low).astype(np.float32)


def _batched(items: list[tuple[int, int]], size: int) -> Iterator[list[tuple[int, int]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_generator(checkpoint_path: str | Path, direction: str, device: torch.device) -> MultiScaleGenerator:
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_config = checkpoint_data.get("train_config", {})
    model_config = train_config.get("model", {})
    model = MultiScaleGenerator(
        channels=int(model_config.get("channels", 3)),
        base_channels=int(model_config.get("base_channels", 64)),
        residual_blocks=int(model_config.get("residual_blocks", 9)),
        dropout=bool(model_config.get("dropout", False)),
        checkpoint_blocks=False,
    )
    state_key = "generator_ab" if direction == "A_to_B" else "generator_ba"
    if state_key not in checkpoint_data:
        raise KeyError(f"Checkpoint does not contain {state_key}")
    model.load_state_dict(checkpoint_data[state_key])
    return model.to(device).eval()


@torch.no_grad()
def translate_raster(config: dict) -> Path:
    device = select_device(str(config.get("device", "cuda")))
    direction = str(config.get("direction", "B_to_A"))
    model = load_generator(config["checkpoint"], direction, device)
    input_path = Path(config["input"])
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bands = [int(value) for value in config.get("bands", [1])]
    repeat_single = bool(config.get("repeat_single_band", True))
    tile_size = int(config.get("tile_size", 256))
    overlap = int(config.get("overlap", 32))
    batch_size = int(config.get("batch_size", 4))
    spec = NormalizationSpec.from_mapping(config.get("normalization"))

    with rasterio.open(input_path) as src, tempfile.TemporaryDirectory(prefix="hsarfcd_") as tmp:
        stats = _normalization_stats(src, bands, spec)
        rows = tile_origins(src.height, tile_size, overlap)
        cols = tile_origins(src.width, tile_size, overlap)
        positions = [(row, col) for row in rows for col in cols]
        output_channels = model.channels
        accumulator = np.memmap(
            Path(tmp) / "sum.dat",
            mode="w+",
            dtype="float32",
            shape=(output_channels, src.height, src.width),
        )
        weights = np.memmap(
            Path(tmp) / "weights.dat",
            mode="w+",
            dtype="float32",
            shape=(src.height, src.width),
        )
        accumulator[:] = 0.0
        weights[:] = 0.0
        blend = blending_weight(tile_size)

        for batch_positions in tqdm(
            list(_batched(positions, batch_size)), desc="translate tiles"
        ):
            tiles = []
            shapes = []
            for row, col in batch_positions:
                height = min(tile_size, src.height - row)
                width = min(tile_size, src.width - col)
                window = Window(col, row, width, height)
                tile = src.read(indexes=bands, window=window, out_dtype="float32")
                tile = repeat_to_channels(tile) if repeat_single else tile
                tile = _normalize_tile(tile, spec, stats)
                if height < tile_size or width < tile_size:
                    tile = np.pad(
                        tile,
                        ((0, 0), (0, tile_size - height), (0, tile_size - width)),
                        mode="reflect",
                    )
                tiles.append(tile)
                shapes.append((height, width))
            tensor = torch.from_numpy(np.stack(tiles)).to(device)
            prediction = model(tensor).cpu().numpy()
            prediction = np.clip((prediction + 1.0) / 2.0, 0.0, 1.0)
            for (row, col), (height, width), tile in zip(
                batch_positions, shapes, prediction, strict=True
            ):
                local_weight = blend[:height, :width]
                accumulator[:, row : row + height, col : col + width] += (
                    tile[:, :height, :width] * local_weight
                )
                weights[row : row + height, col : col + width] += local_weight

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=output_channels,
            dtype="float32",
            compress="deflate",
            tiled=True,
            bigtiff="if_safer",
            nodata=None,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            block_height = min(512, src.height)
            for row in range(0, src.height, block_height):
                height = min(block_height, src.height - row)
                denominator = np.maximum(weights[row : row + height], 1e-6)
                block = accumulator[:, row : row + height] / denominator[None, ...]
                dst.write(block.astype(np.float32), window=Window(0, row, src.width, height))
            dst.update_tags(
                hsarfcd_direction=direction,
                hsarfcd_value_range="0,1",
                hsarfcd_checkpoint=str(config["checkpoint"]),
            )
        del accumulator, weights
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate a full SAR raster with overlap tiling")
    parser.add_argument("--config", default="configs/inference.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = translate_raster(load_yaml(args.config))
    print(output)


if __name__ == "__main__":
    main()
