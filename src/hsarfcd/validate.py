from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rasterio

from .config import load_yaml


def _inspect_raster(path: Path, bands: list[int] | None) -> dict[str, Any]:
    with rasterio.open(path) as src:
        requested = bands or list(range(1, src.count + 1))
        invalid = [band for band in requested if band < 1 or band > src.count]
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "requested_bands": requested,
            "invalid_bands": invalid,
            "dtype": list(src.dtypes),
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform),
        }


def validate_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate configured paired directories without loading imagery into memory."""
    errors: list[str] = []
    warnings: list[str] = []
    splits: dict[str, Any] = {}
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return {"valid": False, "errors": ["Missing 'dataset' mapping"], "warnings": [], "splits": {}}

    if dataset.get("layout", "paired_dirs") != "paired_dirs":
        errors.append("Only dataset.layout='paired_dirs' is supported")
    root_value = dataset.get("root")
    if not root_value:
        errors.append("Missing dataset.root")
        root = Path(".")
    else:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            errors.append(f"Dataset root is not a directory: {root}")

    file_glob = str(dataset.get("file_glob", "*.tif"))
    bands_a = dataset.get("bands_a")
    bands_b = dataset.get("bands_b")
    for split_name in ("train", "val", "test"):
        split = dataset.get(split_name)
        if not isinstance(split, dict):
            errors.append(f"Missing dataset.{split_name} mapping")
            continue
        if "domain_a" not in split or "domain_b" not in split:
            errors.append(f"dataset.{split_name} must define domain_a and domain_b")
            continue

        path_a = root / str(split["domain_a"])
        path_b = root / str(split["domain_b"])
        files_a = sorted(path_a.glob(file_glob)) if path_a.is_dir() else []
        files_b = sorted(path_b.glob(file_glob)) if path_b.is_dir() else []
        names_a = {path.name for path in files_a}
        names_b = {path.name for path in files_b}
        only_a = sorted(names_a - names_b)
        only_b = sorted(names_b - names_a)
        split_report: dict[str, Any] = {
            "domain_a": str(path_a),
            "domain_b": str(path_b),
            "count_a": len(files_a),
            "count_b": len(files_b),
            "paired_count": len(names_a & names_b),
            "only_a": only_a[:20],
            "only_b": only_b[:20],
        }
        splits[split_name] = split_report

        if not path_a.is_dir():
            errors.append(f"Missing directory: {path_a}")
        if not path_b.is_dir():
            errors.append(f"Missing directory: {path_b}")
        if path_a.is_dir() and not files_a:
            errors.append(f"No files matching {file_glob!r} in {path_a}")
        if path_b.is_dir() and not files_b:
            errors.append(f"No files matching {file_glob!r} in {path_b}")
        if only_a or only_b:
            errors.append(
                f"{split_name}: A/B filenames differ "
                f"({len(only_a)} only in A, {len(only_b)} only in B)"
            )

        common = sorted(names_a & names_b)
        if common:
            try:
                info_a = _inspect_raster(path_a / common[0], bands_a)
                info_b = _inspect_raster(path_b / common[0], bands_b)
                split_report["sample_a"] = info_a
                split_report["sample_b"] = info_b
                if info_a["invalid_bands"]:
                    errors.append(f"{split_name}: invalid A band indexes {info_a['invalid_bands']}")
                if info_b["invalid_bands"]:
                    errors.append(f"{split_name}: invalid B band indexes {info_b['invalid_bands']}")
                if (info_a["width"], info_a["height"]) != (info_b["width"], info_b["height"]):
                    errors.append(f"{split_name}: sample A/B dimensions differ")
                if info_a["crs"] != info_b["crs"]:
                    warnings.append(f"{split_name}: sample A/B CRS differ")
                if info_a["transform"] != info_b["transform"]:
                    warnings.append(f"{split_name}: sample A/B transforms differ")
            except Exception as exc:  # rasterio raises format-specific exceptions
                errors.append(f"{split_name}: cannot inspect sample {common[0]}: {exc}")

    scenes: dict[str, Any] = {}
    for name, value in (config.get("scenes") or {}).items():
        if value is None:
            scenes[name] = {"path": None, "exists": False, "optional": True}
            continue
        scene_path = Path(str(value)).expanduser().resolve()
        exists = scene_path.is_file()
        scenes[name] = {"path": str(scene_path), "exists": exists}
        if not exists:
            warnings.append(f"Scene path does not exist: {scene_path}")

    return {
        "valid": not errors,
        "dataset_root": str(root),
        "file_glob": file_glob,
        "errors": errors,
        "warnings": warnings,
        "splits": splits,
        "scenes": scenes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate HSAR-FCD dataset paths and paired rasters")
    parser.add_argument("--data-config", required=True, help="Path to configs/data.local.yaml")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_data_config(load_yaml(args.data_config))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
