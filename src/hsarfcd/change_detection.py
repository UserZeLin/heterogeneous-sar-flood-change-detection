from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)

from .config import load_yaml
from .raster import write_raster


def difference_magnitude(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if before.shape != after.shape:
        raise ValueError(f"Input shapes differ: {before.shape} vs {after.shape}")
    difference = np.nan_to_num(after - before, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sqrt(np.sum(difference**2, axis=0)).astype(np.float32)


def classify_gmm(
    magnitude: np.ndarray,
    components: int = 2,
    covariance_type: str = "full",
    max_samples: int = 500_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """Fit a GMM to difference magnitudes; the component with the largest mean is change."""
    values = magnitude[np.isfinite(magnitude)].reshape(-1, 1)
    if values.size < components or np.var(values) < 1e-12:
        raise ValueError("Difference magnitude is degenerate; GMM cannot be fitted")
    rng = np.random.default_rng(random_state)
    if values.shape[0] > max_samples:
        values = values[rng.choice(values.shape[0], max_samples, replace=False)]
    model = GaussianMixture(
        n_components=components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=3,
    ).fit(values)
    flat = magnitude.reshape(-1, 1)
    labels = model.predict(flat).reshape(magnitude.shape)
    probabilities = model.predict_proba(flat).reshape(*magnitude.shape, components)
    change_component = int(np.argmax(model.means_.reshape(-1)))
    mask = labels == change_component
    probability = probabilities[..., change_component]
    return mask, probability.astype(np.float32), model


def clean_mask(
    mask: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_object_size: int = 60,
    min_hole_size: int = 60,
) -> np.ndarray:
    result = mask.astype(bool)
    if opening_radius > 0:
        result = binary_opening(result, footprint=disk(opening_radius))
    if closing_radius > 0:
        result = binary_closing(result, footprint=disk(closing_radius))
    if min_object_size > 0:
        result = remove_small_objects(result, min_size=min_object_size)
    if min_hole_size > 0:
        result = remove_small_holes(result, area_threshold=min_hole_size)
    return result.astype(np.uint8)


def water_mask(image: np.ndarray, band: int = 1, keep_low_backscatter: bool = True) -> np.ndarray:
    values = np.nan_to_num(image[band - 1], nan=0.0)
    finite = values[np.isfinite(values)]
    threshold = threshold_otsu(finite) if finite.size else 0.0
    return (values < threshold) if keep_low_backscatter else (values > threshold)


def _read_aligned(before_path: str, after_path: str, bands: list[int]):
    with rasterio.open(before_path) as before_src:
        before = before_src.read(indexes=bands, out_dtype="float32")
        profile = before_src.profile.copy()
        with rasterio.open(after_path) as after_src:
            with WarpedVRT(
                after_src,
                crs=before_src.crs,
                transform=before_src.transform,
                width=before_src.width,
                height=before_src.height,
                resampling=Resampling.bilinear,
            ) as aligned:
                after = aligned.read(indexes=bands, out_dtype="float32")
    profile.update(count=len(bands), dtype="float32")
    return before, after, profile


def make_change_overlay(reference: np.ndarray, change: np.ndarray) -> np.ndarray:
    gray = reference[0].astype(np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        gray = np.clip((gray - low) / max(high - low, 1e-6), 0.0, 1.0)
    else:
        gray = np.zeros_like(gray)
    rgb = np.stack([gray, gray, gray])
    selected = change.astype(bool)
    rgb[0, selected] = 1.0
    rgb[1, selected] = 0.0
    rgb[2, selected] = 0.0
    return rgb.astype(np.float32)


def detect(config: dict) -> dict[str, Path]:
    bands = [int(value) for value in config.get("bands", [1, 2, 3])]
    before, after, profile = _read_aligned(config["before"], config["after"], bands)
    magnitude = difference_magnitude(before, after)
    gmm_config = config.get("gmm", {})
    raw, probability, model = classify_gmm(
        magnitude,
        components=int(gmm_config.get("components", 2)),
        covariance_type=str(gmm_config.get("covariance_type", "full")),
        max_samples=int(gmm_config.get("max_samples", 500_000)),
        random_state=int(gmm_config.get("random_state", 42)),
    )

    water_config = config.get("water_mask", {})
    if bool(water_config.get("enabled", False)):
        source = before if water_config.get("source", "before") == "before" else after
        raw &= water_mask(
            source,
            band=int(water_config.get("band", 1)),
            keep_low_backscatter=bool(water_config.get("keep_low_backscatter", True)),
        )

    morphology = config.get("morphology", {})
    filtered = clean_mask(
        raw,
        opening_radius=int(morphology.get("opening_radius", 1)),
        closing_radius=int(morphology.get("closing_radius", 2)),
        min_object_size=int(morphology.get("min_object_size", 60)),
        min_hole_size=int(morphology.get("min_hole_size", 60)),
    )
    output_dir = Path(config.get("output_dir", "outputs/change_detection"))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "magnitude": output_dir / "change_magnitude.tif",
        "probability": output_dir / "change_probability.tif",
        "raw": output_dir / "change_raw_binary.tif",
        "filtered": output_dir / "change_filtered_binary.tif",
        "overlay": output_dir / "change_map.tif",
        "gmm": output_dir / "gmm_parameters.npz",
    }
    single_profile = profile.copy()
    single_profile.update(count=1)
    write_raster(outputs["magnitude"], magnitude, single_profile)
    write_raster(outputs["probability"], probability, single_profile)
    write_raster(outputs["raw"], raw.astype(np.uint8), single_profile, nodata=0)
    write_raster(outputs["filtered"], filtered, single_profile, nodata=0)
    write_raster(outputs["overlay"], make_change_overlay(before, filtered), profile)
    np.savez(
        outputs["gmm"],
        means=model.means_,
        covariances=model.covariances_,
        weights=model.weights_,
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GMM flood change detection and morphology")
    parser.add_argument("--config", default="configs/change_detection.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, path in detect(load_yaml(args.config)).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

