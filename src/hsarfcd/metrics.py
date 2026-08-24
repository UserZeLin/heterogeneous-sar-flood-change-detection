from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def binary_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    reference = reference.astype(bool).reshape(-1)
    prediction = prediction.astype(bool).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(reference, prediction, labels=[False, True]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    kappa = cohen_kappa_score(reference, prediction)
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "overall_accuracy": float(accuracy),
        "kappa": float(kappa),
    }


def evaluate_rasters(reference_path: str, prediction_path: str) -> dict[str, float | int]:
    with rasterio.open(reference_path) as reference_src:
        reference = reference_src.read(1) > 0
        with rasterio.open(prediction_path) as prediction_src:
            with WarpedVRT(
                prediction_src,
                crs=reference_src.crs,
                transform=reference_src.transform,
                width=reference_src.width,
                height=reference_src.height,
                resampling=Resampling.nearest,
            ) as aligned:
                prediction = aligned.read(1) > 0
    return binary_metrics(reference, prediction)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a binary change map")
    parser.add_argument("reference")
    parser.add_argument("prediction")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_rasters(args.reference, args.prediction)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")


if __name__ == "__main__":
    main()
