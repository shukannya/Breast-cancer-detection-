"""Classical digital-image-processing features for H&E microscopy images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import color, exposure, feature, transform

IMAGE_SIZE = (128, 128)  # width, height
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_rgb(path: str | Path, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load any supported image, apply EXIF orientation, composite alpha, and resize."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not read image '{path}': {exc}") from exc


def preprocess_image(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB image and CLAHE-enhanced grayscale image, both float in [0, 1]."""
    rgb_float = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = color.rgb2gray(rgb_float)
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.025)
    return rgb_float, enhanced.astype(np.float32)


def extract_features_from_array(rgb: np.ndarray) -> np.ndarray:
    """Build a fixed-length feature vector from color, texture, and edge structure."""
    rgb_float, enhanced = preprocess_image(rgb)

    # HOG encodes local edge orientations; LBP describes local micro-texture.
    hog = feature.hog(
        enhanced,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    gray_u8 = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 12), range=(0, 11), density=True)

    # Per-channel histograms preserve coarse H&E stain/color information.
    color_hist = []
    for channel in range(3):
        hist, _ = np.histogram(rgb_float[:, :, channel], bins=16, range=(0, 1), density=True)
        color_hist.extend(hist.astype(np.float32))

    # A compact intensity histogram adds global appearance information.
    intensity_hist, _ = np.histogram(enhanced, bins=16, range=(0, 1), density=True)
    return np.concatenate(
        [hog.astype(np.float32), lbp_hist.astype(np.float32), np.asarray(color_hist), intensity_hist.astype(np.float32)]
    )


def extract_features(path: str | Path) -> np.ndarray:
    """Read an image and extract its fixed-length feature vector."""
    return extract_features_from_array(load_rgb(path))
