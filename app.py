"""
Streamlit web application for the EcoVision project.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.inference import load_model, run_inference
from src.report import build_report, detections_to_table
from src.utils import load_image
from src.visualization import draw_detections


DEFAULT_MODEL = Path("models/best_onelabel.pt")


@st.cache_resource
def cached_model(model_path: str):
    return load_model(model_path)


def main() -> None:
    st.set_page_config(page_title="EcoVision", layout="wide")
    st.title("EcoVision")

    model_path = st.sidebar.text_input("Model", str(DEFAULT_MODEL))
    confidence = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.sidebar.select_slider("Image size", options=[320, 480, 640, 960], value=640)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload an image to run waste detection.")
        return

    try:
        image = load_image(uploaded)
        model = cached_model(model_path)
        detections = run_inference(model, image, confidence_threshold=confidence, imgsz=imgsz)
        annotated = draw_detections(image, detections)
        report = build_report(detections)
    except Exception as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    left.image(image, caption="Input image", use_container_width=True)
    right.image(annotated, caption="Detected waste", use_container_width=True)

    st.subheader("Detected objects")
    table = detections_to_table(detections)
    if table.empty:
        st.write("No objects passed the confidence threshold.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Report")
    st.write(f"Pollution level: **{report['pollution_level']}**")
    st.write(report["recommendation"])
    st.json(report["summary"])


if __name__ == "__main__":
    main()
