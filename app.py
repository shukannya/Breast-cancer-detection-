"""Optional browser UI: streamlit run app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from breast_cancer_detector.model import load_model, predict_image, train_model
from breast_cancer_detector.visualization import save_prediction_figure

st.set_page_config(page_title="Breast Histopathology Classifier", page_icon="🔬", layout="centered")
st.title("Breast Histopathology Image Classifier")
st.warning("Educational/research tool only. It is not validated for clinical diagnosis or treatment decisions.")
st.caption("Input scope: H&E-stained breast tissue microscopy images (not mammograms).")

with st.sidebar:
    st.header("Train a model")
    data_dir = st.text_input("Dataset folder", "data")
    model_path = st.text_input("Checkpoint path", "models/breast_cancer_svm.joblib")
    if st.button("Train and evaluate", type="primary"):
        with st.spinner("Extracting image features and training SVM…"):
            try:
                result = train_model(data_dir, model_path)
                st.success(f"Saved model: {result['model_path']}")
                st.json({k: result[k] for k in ("accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "split_method", "train_images", "test_images")})
            except Exception as exc:
                st.error(str(exc))

model_path_ui = st.text_input("Trained model checkpoint", "models/breast_cancer_svm.joblib")
upload = st.file_uploader("Upload an H&E microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
if upload is not None:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption=upload.name, use_container_width=True)
    if st.button("Classify image"):
        temp_path = None
        try:
            suffix = Path(upload.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(upload.getvalue())
                temp_path = Path(temp.name)
            bundle = load_model(model_path_ui)
            result = predict_image(bundle, temp_path)
            color = "#b22222" if result["prediction"] == "malignant" else "#146b3a"
            st.markdown(f"### <span style='color:{color}'>{result['prediction'].upper()}</span>", unsafe_allow_html=True)
            st.metric("Model estimated confidence", f"{result['confidence']:.1%}")
            st.json({"estimated_probability_benign": result["probability_benign"], "estimated_probability_malignant": result["probability_malignant"]})
            with tempfile.TemporaryDirectory() as out:
                figure = save_prediction_figure(temp_path, result, Path(out) / "result.png")
                st.image(figure, caption="Original and contrast-enhanced image")
        except Exception as exc:
            st.error(str(exc))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
