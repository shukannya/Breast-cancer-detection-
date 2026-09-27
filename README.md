# Breast Cancer Detection System (Digital Image Processing Lab)

**Student:** Noushin Ulfat Shukannya · **Exam Roll:** 201746 · **Class Roll:** 2025  
**Model:** H&E breast-tumor histopathology image classifier (benign vs. malignant)

> This is an educational image-processing project, not a medical device. Outputs must not be used to diagnose, screen, or guide treatment. A qualified pathologist must interpret tissue images. It accepts histopathology microscopy images, not mammograms.

## What is included

- Python package `breast_cancer_detector/` with image loading, preprocessing, feature extraction, SVM training, inference, and visual result export.
- `app.py`: optional Streamlit interface for training and image upload/prediction.
- `tests/`: a software smoke test using generated fixtures; those fixtures are **not** scientific data and produce no reportable model-performance claim.
- `project_report.pdf`: lab report matching the supplied reference's cover and numbered section structure.

## Dataset

This project is designed for the **BreaKHis** breast histopathology image dataset. The detailed table on the official database page lists 7,909 PNG microscopy images (2,480 benign; 5,429 malignant) from 82 patients, acquired at 40×, 100×, 200×, and 400× magnifications. Note: the same page's opening summary says 9,109 images, which conflicts with the class totals in its detailed table; check the current source/archive metadata when citing counts. The official download page offers a tar archive for non-commercial research and asks users to acknowledge/cite its source. Review the current terms and complete the dataset authors' requested registration/acknowledgement before using it:

- Official information/download: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- Citation: F. Spanhol, L. S. Oliveira, C. Petitjean, and L. Heutte, “A Dataset for Breast Cancer Histopathological Image Classification,” *IEEE Transactions on Biomedical Engineering*, 63(7), 1455–1462, 2016. DOI: https://doi.org/10.1109/TBME.2015.2496264

The BreaKHis dataset is **not bundled** in this ZIP. Download and extract it locally; the code recursively finds images whose directory path includes `benign` or `malignant` (the original BreaKHis layout works).

For a small custom dataset, use this structure:

```text
data/
├── benign/
│   ├── image_001.png
│   └── ...
└── malignant/
    ├── image_101.png
    └── ...
```

Accepted image formats: PNG, JPG/JPEG, TIFF, BMP. Include both classes, and use multiple independent patients. For a credible evaluation, split by patient—not by image crop. When a BreaKHis filename follows the original convention, this project groups its patient ID during holdout evaluation. For other data, patient-aware splitting requires patient identifiers in the filenames or an added group manifest; the generic fallback is image-level and may overestimate performance if images from one patient are correlated.

## Install

Python 3.10+ recommended. In a terminal from the extracted project folder:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m breast_cancer_detector.cli train --data /path/to/BreaKHis_v1 --model models/breast_cancer_svm.joblib --test-size 0.2 --seed 42
```

The training command extracts the image features, trains an RBF-kernel SVM, evaluates a held-out split, and creates:

- `models/breast_cancer_svm.joblib` — model checkpoint;
- `models/breast_cancer_svm.metrics.json` — metrics, class counts, split method, and confusion matrix.

For a quick install/editable run, set `PYTHONPATH=.` if invoking the package from another directory.

## Predict an image

```bash
python -m breast_cancer_detector.cli predict --model models/breast_cancer_svm.joblib --image /path/to/tissue.png --output outputs
```

The command writes `outputs/predictions.json` and a PNG panel with the original image and CLAHE-enhanced grayscale view. Folder input is also supported:

```bash
python -m breast_cancer_detector.cli predict --model models/breast_cancer_svm.joblib --image /path/to/images --output outputs
```

## Optional browser app

```bash
streamlit run app.py
```

Use the sidebar to train, then upload a histopathology image and select the trained checkpoint. The app runs locally; no images are sent to a hosted service by this project.

## Image-processing / classification pipeline

1. Read RGB tissue images, apply EXIF orientation, and resize to 128×128.
2. Convert RGB to grayscale and apply CLAHE for local contrast normalization.
3. Extract HOG edge-orientation descriptors, uniform LBP texture histogram, RGB stain-color histograms, and enhanced-gray intensity histogram.
4. Standardize features and train an RBF-kernel Support Vector Machine with balanced class weights.
5. Evaluate on a held-out split and save accuracy, malignant precision/recall/F1, confusion matrix, and class-wise report.
6. For inference, emit predicted label and SVM probability estimates plus an inspection panel.

## Tests

```bash
python -m pytest -q
```

Smoke-test fixtures only check that loading, feature extraction, training, checkpoint serialization, and prediction execute. They do not represent breast tissue and are not used as project results.

## Limitations and reproducibility

- No training or validation on real BreaKHis images is included in the supplied report: download the real dataset and run the command above to generate dataset-specific metrics.
- Patient-level separation works for original BreaKHis patient IDs found in its standard filename format. For renamed/custom files, supply patient groups before claiming patient-independent test performance.
- A single random holdout can vary; use repeated patient-wise cross-validation, external data, and expert review for research claims.
- SVC probabilities are model estimates, not calibrated clinical risk values.
- Stain, scanner, magnification, tissue preparation, patient mix, and dataset shift can strongly affect predictions. No external clinical validation has been conducted.
