"""Software smoke tests. Fixture images are not experimental or clinical data."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from breast_cancer_detector.cli import main
from breast_cancer_detector.features import extract_features, load_rgb
from breast_cancer_detector.model import load_model, predict_image, train_model


def _fixture_image(path: Path, kind: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    if kind == "benign":
        base[:] = (210, 150, 180)
        for y in range(10, 120, 24):
            base[y:y + 5, :, :] = (245, 210, 225)
    else:
        base[:] = (115, 45, 95)
        for _ in range(100):
            x, y = rng.integers(0, 128, 2)
            base[max(0, y-2):y+3, max(0, x-2):x+3] = (225, 80, 120)
    base = np.clip(base.astype(np.int16) + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(base).save(path)


def test_features_training_checkpoint_and_cli_prediction(tmp_path):
    root = tmp_path / "dataset"
    for class_index, name in enumerate(("benign", "malignant")):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(5):
            _fixture_image(folder / f"sample_{i}.png", name, class_index * 100 + i)

    sample = root / "benign" / "sample_0.png"
    assert load_rgb(sample).shape == (128, 128, 3)
    assert extract_features(sample).ndim == 1

    checkpoint = tmp_path / "model.joblib"
    result = train_model(root, checkpoint, test_size=0.25)
    assert checkpoint.is_file()
    assert Path(result["metrics_path"]).is_file()
    loaded = load_model(checkpoint)
    prediction = predict_image(loaded, sample)
    assert prediction["prediction"] in {"benign", "malignant"}
    assert 0.0 <= prediction["confidence"] <= 1.0

    output_dir = tmp_path / "cli_output"
    status = main(["predict", "--model", str(checkpoint), "--image", str(sample), "--output", str(output_dir)])
    assert status == 0
    cli_results = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    assert len(cli_results) == 1
    assert Path(cli_results[0]["visualization"]).is_file()
