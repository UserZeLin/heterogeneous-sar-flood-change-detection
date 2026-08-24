from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .raster import NormalizationSpec, normalize, read_raster, stack_rasters, write_raster


def power_to_db(data: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Convert positive SAR power values to decibels."""
    return (10.0 * np.log10(np.maximum(data, floor))).astype(np.float32)


def normalize_raster(input_path: str, output_path: str, spec: NormalizationSpec) -> None:
    data, profile = read_raster(input_path)
    result, _ = normalize(data, spec)
    write_raster(output_path, result, profile)


def stack_command(inputs: list[str], output: str) -> None:
    data, profile = stack_rasters(inputs)
    write_raster(output, data.astype(np.float32), profile)


def tile_raster(
    input_path: str | Path,
    output_dir: str | Path,
    tile_size: int = 256,
    stride: int | None = None,
    prefix: str | None = None,
) -> list[Path]:
    """Split a GeoTIFF into georeferenced, full-size tiles; edge tiles are skipped."""
    stride = stride or tile_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with rasterio.open(input_path) as src:
        base = prefix or Path(input_path).stem
        index = 1
        for row in range(0, src.height - tile_size + 1, stride):
            for col in range(0, src.width - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                tile = src.read(window=window)
                profile = src.profile.copy()
                profile.update(
                    height=tile_size,
                    width=tile_size,
                    transform=src.window_transform(window),
                    compress="deflate",
                    tiled=True,
                )
                path = output_dir / f"{base}_{index:05d}.tif"
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(tile)
                written.append(path)
                index += 1
    return written


def tile_pair(
    domain_a: str,
    domain_b: str,
    output_root: str,
    split: str,
    tile_size: int,
    stride: int,
) -> tuple[list[Path], list[Path]]:
    """Tile two already co-registered rasters into paired A/B directories."""
    with rasterio.open(domain_a) as a, rasterio.open(domain_b) as b:
        same_grid = (
            a.width == b.width
            and a.height == b.height
            and a.transform == b.transform
            and a.crs == b.crs
        )
    if not same_grid:
        raise ValueError("Paired rasters must share width, height, transform and CRS")
    root = Path(output_root) / split
    paths_a = tile_raster(domain_a, root / "A", tile_size, stride, prefix="tile")
    paths_b = tile_raster(domain_b, root / "B", tile_size, stride, prefix="tile")
    if [p.name for p in paths_a] != [p.name for p in paths_b]:
        raise RuntimeError("Paired tiling produced mismatched filenames")
    return paths_a, paths_b


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SAR rasters for HSAR-FCD")
    subparsers = parser.add_subparsers(dest="command", required=True)

    db = subparsers.add_parser("db", help="Convert SAR power raster to dB")
    db.add_argument("input")
    db.add_argument("output")
    db.add_argument("--floor", type=float, default=1e-6)

    norm = subparsers.add_parser("normalize", help="Normalize every raster band")
    norm.add_argument("input")
    norm.add_argument("output")
    norm.add_argument("--lower", type=float, default=1.0)
    norm.add_argument("--upper", type=float, default=99.0)

    stack = subparsers.add_parser("stack", help="Align and stack one or more rasters")
    stack.add_argument("output")
    stack.add_argument("inputs", nargs="+")

    tile = subparsers.add_parser("tile-pair", help="Tile co-registered A/B rasters")
    tile.add_argument("domain_a")
    tile.add_argument("domain_b")
    tile.add_argument("output_root")
    tile.add_argument("--split", default="train", choices=["train", "val", "test"])
    tile.add_argument("--tile-size", type=int, default=256)
    tile.add_argument("--stride", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "db":
        data, profile = read_raster(args.input)
        write_raster(args.output, power_to_db(data, args.floor), profile)
    elif args.command == "normalize":
        spec = NormalizationSpec(lower=args.lower, upper=args.upper)
        normalize_raster(args.input, args.output, spec)
    elif args.command == "stack":
        stack_command(args.inputs, args.output)
    elif args.command == "tile-pair":
        tile_pair(
            args.domain_a,
            args.domain_b,
            args.output_root,
            args.split,
            args.tile_size,
            args.stride,
        )


if __name__ == "__main__":
    main()
