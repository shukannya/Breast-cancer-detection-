"""Dataset discovery, patient-aware evaluation, and SVM model training."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import SUPPORTED_EXTENSIONS, extract_features

CLASS_NAMES = ("benign", "malignant")
BREAKHIS_PATIENT_RE = re.compile(r"SOB_[BM]_[A-Z]+-(\d+-\d+)-", re.IGNORECASE)


def infer_label(path: Path) -> str | None:
    """Find a binary class from a directory component, supporting BreaKHis layout."""
    for part in reversed(path.parts[:-1]):
        name = part.lower().replace("_", "-").strip()
        if name in {"benign", "benign-tumor", "benign-tumors", "b"}:
            return "benign"
        if name in {"malignant", "malignant-tumor", "malignant-tumors", "m", "cancer", "cancerous"}:
            return "malignant"
    return None


def discover_images(data_dir: str | Path) -> tuple[list[Path], list[str]]:
    """Recursively find labeled images below a data root; skip unlabeled files."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    paths, labels = [], []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            label = infer_label(path)
            if label:
                paths.append(path)
                labels.append(label)
    if not paths:
        raise ValueError(
            f"No labeled images found under {root}. Put images in benign/ and malignant/ folders "
            "or use the original BreaKHis folder layout."
        )
    return paths, labels


def patient_group(path: Path) -> str:
    """Extract BreaKHis patient identity; otherwise use a per-image group."""
    match = BREAKHIS_PATIENT_RE.search(path.name)
    if match:
        return "breakhis-patient-" + match.group(1)
    return "image-" + str(path.resolve())


def _make_split(paths: list[Path], labels: list[str], test_size: float, random_state: int):
    y = np.asarray(labels)
    groups = np.asarray([patient_group(path) for path in paths])
    # Group-level validation prevents near-duplicate crops from one BreaKHis patient
    # appearing in both train and test. Generic folders fall back to stratified split.
    has_patient_groups = any(group.startswith("breakhis-patient-") for group in groups)
    if has_patient_groups and len(np.unique(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=40, test_size=test_size, random_state=random_state)
        candidates = list(splitter.split(np.zeros(len(y)), y, groups))
        for train_idx, test_idx in candidates:
            if set(y[train_idx]) == set(CLASS_NAMES) and set(y[test_idx]) == set(CLASS_NAMES):
                return train_idx, test_idx, "patient-grouped"
        # Keep patient separation even if a small dataset cannot yield both classes in each fold.
        train_idx, test_idx = candidates[0]
        return train_idx, test_idx, "patient-grouped (small sample; class balance may vary)"

    counts = np.unique(y, return_counts=True)[1]
    stratify = y if len(counts) == 2 and counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train_idx, test_idx, "stratified image split" if stratify is not None else "image split"


def train_model(
    data_dir: str | Path,
    model_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train an RBF-SVM and save model plus reproducible split/metric metadata."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.50")
    paths, labels = discover_images(data_dir)
    if len(paths) < 4 or set(labels) != set(CLASS_NAMES):
        raise ValueError("At least four labeled images and both benign/ and malignant/ classes are required.")

    print(f"Extracting DIP features from {len(paths)} images...", flush=True)
    X = np.vstack([extract_features(path) for path in paths])
    y = np.asarray(labels)
    train_idx, test_idx, split_method = _make_split(paths, labels, test_size, random_state)
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError("The training split has only one class. Add more patients/images or adjust the split.")

    train_class_counts = np.unique(y[train_idx], return_counts=True)[1]
    if train_class_counts.min() < 2:
        raise ValueError("At least two training images per class are needed for probability calibration.")
    classifier = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=random_state),
            method="sigmoid",
            cv=min(3, int(train_class_counts.min())),
        ),
    )
    classifier.fit(X[train_idx], y[train_idx])
    prediction = classifier.predict(X[test_idx])
    metrics = {
        "accuracy": float(accuracy_score(y[test_idx], prediction)),
        "precision_malignant": float(precision_score(y[test_idx], prediction, pos_label="malignant", zero_division=0)),
        "recall_malignant": float(recall_score(y[test_idx], prediction, pos_label="malignant", zero_division=0)),
        "f1_malignant": float(f1_score(y[test_idx], prediction, pos_label="malignant", zero_division=0)),
        "confusion_matrix_labels": ["benign", "malignant"],
        "confusion_matrix": confusion_matrix(y[test_idx], prediction, labels=list(CLASS_NAMES)).tolist(),
        "classification_report": classification_report(y[test_idx], prediction, labels=list(CLASS_NAMES), output_dict=True, zero_division=0),
        "split_method": split_method,
        "train_images": int(len(train_idx)),
        "test_images": int(len(test_idx)),
        "total_images": int(len(paths)),
        "class_counts": {name: int(labels.count(name)) for name in CLASS_NAMES},
        "random_state": int(random_state),
        "test_size": float(test_size),
    }
    output_path = Path(model_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": classifier,
        "classes": list(CLASS_NAMES),
        "feature_description": "CLAHE grayscale + HOG + uniform LBP + RGB/intensity histograms",
        "image_size": [128, 128],
        "metrics": metrics,
    }
    joblib.dump(bundle, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_path": str(output_path.resolve()), "metrics_path": str(metrics_path.resolve()), **metrics}


def load_model(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}. Train a model first.")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Could not load model checkpoint '{path}': {exc}") from exc


def predict_image(model_bundle: dict, image_path: str | Path) -> dict:
    """Predict benign/malignant class and calibrated SVC probability for an image."""
    vector = extract_features(image_path).reshape(1, -1)
    model = model_bundle["model"]
    probabilities = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    index = int(np.argmax(probabilities))
    probability_by_class = {name: float(probabilities[i]) for i, name in enumerate(classes)}
    return {
        "prediction": str(classes[index]),
        "confidence": float(probabilities[index]),
        "probability_benign": probability_by_class.get("benign", 0.0),
        "probability_malignant": probability_by_class.get("malignant", 0.0),
    }
