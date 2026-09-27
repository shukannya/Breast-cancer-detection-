"""Command-line interface for training and using the educational classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import load_model, predict_image, train_model
from .visualization import save_prediction_figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breast-cancer-detector",
        description="Educational H&E breast histopathology image classifier (not for clinical diagnosis).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an SVM using benign/malignant image folders")
    train.add_argument("--data", required=True, help="dataset folder (BreaKHis layout or benign/ and malignant/)")
    train.add_argument("--model", default="models/breast_cancer_svm.joblib", help="checkpoint output path")
    train.add_argument("--test-size", type=float, default=0.2, help="held-out evaluation fraction (0.05–0.50)")
    train.add_argument("--seed", type=int, default=42, help="reproducible split/estimator seed")

    predict = sub.add_parser("predict", help="classify one image or all images in a folder")
    predict.add_argument("--model", default="models/breast_cancer_svm.joblib", help="trained checkpoint")
    predict.add_argument("--image", required=True, help="image path or folder containing images")
    predict.add_argument("--output", default="outputs", help="prediction visualization output folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_model(args.data, args.model, args.test_size, args.seed)
            print(json.dumps(result, indent=2))
            return 0

        bundle = load_model(args.model)
        input_path = Path(args.image).expanduser()
        if input_path.is_file():
            images = [input_path]
        elif input_path.is_dir():
            from .features import SUPPORTED_EXTENSIONS
            images = sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        else:
            raise FileNotFoundError(f"Image/folder not found: {input_path}")
        if not images:
            raise ValueError(f"No supported images found in {input_path}")

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for image in images:
            result = {"image": str(image.resolve()), **predict_image(bundle, image)}
            safe_name = image.stem.replace(" ", "_") + "_prediction.png"
            result["visualization"] = str(save_prediction_figure(image, result, output_dir / safe_name))
            results.append(result)
        json_path = output_dir / "predictions.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({"predictions": results, "saved_json": str(json_path.resolve())}, indent=2))
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
