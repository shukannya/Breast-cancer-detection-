"""Save easy-to-inspect original, enhanced, and prediction panels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .features import load_rgb, preprocess_image


def save_prediction_figure(image_path, result: dict, output_path: str | Path) -> str:
    rgb = load_rgb(image_path)
    _, enhanced = preprocess_image(rgb)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Input histopathology image")
    axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("CLAHE contrast enhancement")
    for axis in axes:
        axis.axis("off")
    label = result["prediction"].upper()
    title = f"Educational model output: {label} | estimated confidence {result['confidence']:.1%}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.text(
        0.5, -0.02,
        "Research/teaching demonstration only — not a medical diagnosis or a substitute for pathology review.",
        ha="center", fontsize=8, color="#7a1f1f",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())
