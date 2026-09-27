# Project Report

**Course Title:** Digital Image Processing Lab  
**Course Code:** ICT-4202

## Submitted to

**Dr. Fahima Tabassum**  
Professor  
Institute of Information Technology, Jahangirnagar University

## Submitted by

| Field | Student information |
|---|---|
| Name | Noushin Ulfat Shukannya |
| Class Roll | 2025 |
| Exam Roll | 201746 |

---

# Project Name: Breast Cancer Detection from Histopathology Images Using DIP Features and SVM

## 1. Project Overview

This project implements an educational binary image-classification pipeline for H&E-stained breast-tissue histopathology microscopy images. A user supplies a tissue image; the program normalizes its size, enhances local grayscale contrast, extracts color, texture, and edge descriptors, and uses a Support Vector Machine (SVM) trained on benign and malignant examples to produce an estimated class and probability. A command-line interface supports model training, evaluation, and inference, and an optional Streamlit page provides a local graphical interface.

The project is designed for the BreaKHis image dataset, which contains benign and malignant breast-tumor microscopy images at multiple magnifications. The dataset is not included in the project ZIP; users must obtain it from its official source and follow its access, attribution, and non-commercial research conditions. This project classifies histopathology images, not mammograms.

> This is a lab/educational prototype and is not a medical device. It has not been clinically validated and must not be used to diagnose, screen, or guide treatment.

## 2. Project Objectives

- Apply digital image processing to prepare histopathology images for classification.
- Extract interpretable color, texture, and edge features from breast-tissue microscopy images.
- Train a binary benign/malignant classifier using a balanced RBF-kernel SVM.
- Evaluate the model on held-out data and report accuracy, malignant precision/recall/F1, and a confusion matrix.
- Save a reusable trained checkpoint and support prediction on a new image or image folder.
- Visualize the input and contrast-enhanced image alongside the prediction.

## 3. Algorithm Used: CLAHE, HOG, LBP, Histograms, and SVM

### Image preprocessing and features

Images are loaded as RGB, corrected for EXIF orientation, and resized to 128 × 128 pixels. RGB-to-grayscale conversion provides a luminance channel, and Contrast Limited Adaptive Histogram Equalization (CLAHE) improves local contrast. The feature vector combines:

- **HOG (Histogram of Oriented Gradients):** summarizes local edge orientations and structural patterns.
- **Uniform LBP (Local Binary Patterns):** summarizes local micro-texture using neighboring pixel intensity comparisons.
- **RGB histograms:** capture coarse stain/color distribution.
- **Enhanced grayscale histogram:** represents global intensity distribution after CLAHE.

The combined feature vector is standardized, then classified using an RBF-kernel SVM with balanced class weights. The checkpoint stores the fitted preprocessing/classifier pipeline, class labels, and experiment metadata.

### Why these methods fit a DIP lab

- Uses classical image-processing and feature-engineering concepts that can be inspected and explained.
- Combines complementary appearance cues: color, local texture, and edges.
- SVMs are effective for fixed-length descriptors and modest training datasets.
- CPU training and inference are possible; a GPU is not required.
- The outputs include both machine-readable JSON and a visual inspection panel.

## 4. System Workflow and Methodology

The code follows: **image dataset → resize and CLAHE → HOG/LBP/color descriptors → feature scaling → SVM training → held-out evaluation → saved model → inference and visualization.** See the accompanying `workflow.png` for the diagram.

### Workflow explanation

1. **Dataset preparation:** Place data under benign and malignant class directories, or use the official BreaKHis directory structure. Only image files with recognized class folder names are loaded.
2. **Image preprocessing:** Apply EXIF orientation, convert to RGB, fit to 128 × 128, convert to grayscale, and perform CLAHE contrast enhancement.
3. **Feature extraction:** Calculate HOG, uniform LBP, RGB-channel histograms, and an enhanced-gray histogram.
4. **Split design:** For filenames matching the original BreaKHis convention, group patient IDs to keep each patient’s images together during holdout evaluation. Other directory layouts use a stratified image split; this can leak correlated examples if patients share images across the split.
5. **Training:** Standardize feature dimensions and fit a balanced RBF SVM with probability estimates enabled.
6. **Evaluation and saving:** Compute holdout metrics and save a joblib checkpoint plus a JSON metrics file.
7. **Inference:** Load the checkpoint, process a new image with the same pipeline, return the predicted label and estimated class probabilities, and save an original/enhanced image panel.

## 5. Main Training Configuration

| Parameter | Value |
|---|---|
| Task | Binary image classification (benign / malignant) |
| Input | H&E breast-tissue histopathology image (RGB) |
| Input size | 128 × 128 pixels |
| Contrast processing | Grayscale conversion + CLAHE (clip limit 0.025) |
| Descriptors | HOG, uniform LBP, RGB and intensity histograms |
| Classifier | RBF-kernel SVM; C = 10; balanced class weights |
| Feature scaling | StandardScaler |
| Evaluation | 80/20 holdout by default; patient-grouped for original BreaKHis filenames |
| Random seed | 42 by default |
| Hardware | CPU; GPU optional/not used |
| Dataset | BreaKHis; obtain separately from the official database page |
| Model output | `models/breast_cancer_svm.joblib` |
| Metrics output | `models/breast_cancer_svm.metrics.json` |

## 6. Evaluation and Results

The report-generation environment did not contain the real BreaKHis image dataset, so no real-data model-training experiment was run and **no scientific accuracy or clinical-performance result is claimed**. The project includes a synthetic-fixture software smoke test only; those fixtures verify that loading, feature extraction, training/checkpoint saving, and inference execute, but they are not tissue images and are not used as model results.

Run training with real, appropriately obtained images to produce dataset-specific metrics:

```bash
python -m breast_cancer_detector train --data /path/to/BreaKHis_v1 --model models/breast_cancer_svm.joblib
```

The command writes the confusion matrix, accuracy, malignant precision/recall/F1, class counts, split method, and class-wise report to the companion `.metrics.json` file. Report the results from that file only, and state the dataset split and patient-grouping method. A single holdout result is exploratory; independent external validation is required for broader claims.

## 7. Inference Procedure

- Train the classifier and retain the resulting checkpoint.
- Run `python -m breast_cancer_detector predict --model models/breast_cancer_svm.joblib --image /path/to/image.png --output outputs`.
- The program reuses the identical resizing, grayscale/CLAHE, and feature-extraction pipeline.
- Review the benign/malignant prediction and estimated probabilities cautiously; the score is not a clinical risk estimate.
- Inspect the generated input/enhancement panel and JSON output under `outputs/`.
- Alternatively, run `streamlit run app.py` for the optional local browser interface.

## 8. Tools and Technologies

| Component | Role |
|---|---|
| Python | Project programming language |
| Pillow / NumPy | Image loading, RGB conversion, numerical arrays |
| scikit-image | CLAHE, grayscale conversion, HOG and LBP extraction |
| scikit-learn | Feature scaling, SVM training, holdout metrics |
| joblib | Save and load the trained model bundle |
| Matplotlib | Prediction visualization export |
| Streamlit | Optional local user interface |
| pytest | Software smoke testing |
| BreaKHis | External image dataset (download separately; cite/acknowledge source) |

## 9. Conclusion and Scope for Improvement

The project provides a complete, runnable educational workflow for classifying breast histopathology images using classical DIP descriptors and an SVM. It includes dataset discovery, preprocessing, feature extraction, training, held-out evaluation, model serialization, command-line inference, a prediction visualization, and an optional browser interface. The software smoke test verifies the main code paths, but real-data performance remains to be measured after the user obtains and trains on BreaKHis.

Future work should include repeated patient-wise cross-validation, a strictly independent external test set, magnification-aware analysis, stain normalization, richer interpretable overlays, comparison with other classical and deep-learning methods, and review by qualified pathology experts. The system must not be used for medical decisions without rigorous clinical validation and regulatory review.

### Dataset reference

F. Spanhol, L. S. Oliveira, C. Petitjean, and L. Heutte, “A Dataset for Breast Cancer Histopathological Image Classification,” *IEEE Transactions on Biomedical Engineering*, vol. 63, no. 7, pp. 1455–1462, 2016. DOI: [10.1109/TBME.2015.2496264](https://doi.org/10.1109/TBME.2015.2496264). Dataset details and terms: [Official BreaKHis database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).
