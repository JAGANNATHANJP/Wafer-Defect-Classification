"""
app.py
======================================================
Main Streamlit application entry point for the Wafer
Defect Classification & Explainable AI system.

Run with:
    streamlit run app.py

This module is intentionally kept focused on UI
orchestration. All heavy lifting (model loading,
preprocessing, inference, Grad-CAM, and file/report
utilities) is delegated to predictor.py, gradcam.py,
and utils.py respectively.

Author: AI Engineering Team
======================================================
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Wedge

import config
import gradcam
import predictor
import utils

# ------------------------------------------------------
# PAGE CONFIGURATION (must be the first Streamlit call)
# ------------------------------------------------------

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)


# ------------------------------------------------------
# CUSTOM CSS (dark-mode friendly, card-based UI)
# ------------------------------------------------------

def inject_custom_css() -> None:
    """
    Inject custom CSS to give the application a polished,
    industrial-dashboard look. Uses Streamlit's theme-aware
    CSS variables (e.g. var(--background-color)) wherever
    possible so the UI remains legible in both light and
    dark themes.
    """
    st.markdown(
        """
        <style>
        .app-card {
            padding: 1.25rem 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background-color: rgba(128, 128, 128, 0.06);
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            text-align: center;
        }
        .prediction-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
            color: white;
        }
        .section-header {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
            border-left: 5px solid #4C9AFF;
            padding-left: 0.6rem;
        }
        .app-footer {
            text-align: center;
            padding-top: 2rem;
            padding-bottom: 1rem;
            opacity: 0.7;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------

def init_session_state() -> None:
    """Initialize all Streamlit session-state keys used across pages."""
    defaults = {
        "uploaded_image": None,
        "preprocessed_input": None,
        "prediction_result": None,
        "gradcam_visuals": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------

NAV_ICONS: Dict[str, str] = {
    "Home": "🏠",
    "Image Upload & Prediction": "🖼️",
    "Grad-CAM Explainability": "🔥",
    "Model Information": "🧠",
    "Performance": "📊",
    "About": "📘",
}


def render_sidebar() -> str:
    """
    Render the sidebar navigation menu and developer info panel.

    Returns:
        The name of the currently selected navigation page.
    """
    with st.sidebar:
        st.markdown(f"## {config.APP_ICON} {config.APP_TITLE}")
        st.caption(config.APP_TAGLINE)
        st.divider()

        labels = [f"{NAV_ICONS.get(item, '')} {item}" for item in config.NAV_ITEMS]
        selected_label = st.radio("Navigation", labels, label_visibility="collapsed")
        selected_page = config.NAV_ITEMS[labels.index(selected_label)]

        st.divider()
        with st.expander("ℹ️ Developer Info", expanded=False):
            for key, value in config.DEVELOPER_INFO.items():
                st.markdown(f"**{key}:** {value}")

        st.divider()
        st.caption("Built with TensorFlow, Streamlit & Grad-CAM")

    return selected_page


# ------------------------------------------------------
# CHART HELPERS
# ------------------------------------------------------

def plot_probabilities_bar_chart(probabilities: Dict[str, float]) -> plt.Figure:
    """
    Build a horizontal bar chart of class probabilities.

    Args:
        probabilities: Mapping of class name to predicted probability.

    Returns:
        A Matplotlib Figure object.
    """
    sorted_items = sorted(probabilities.items(), key=lambda item: item[1])
    class_names = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]
    colors = [utils.get_class_color(name) for name in class_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(class_names, values, color=colors)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Class Probability Distribution")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.tight_layout()
    return fig


def plot_confidence_gauge(confidence: float, class_color: str) -> plt.Figure:
    """
    Build a semi-circular gauge chart representing prediction confidence.

    Args:
        confidence: Confidence score between 0 and 1.
        class_color: Hex color code associated with the predicted class.

    Returns:
        A Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"aspect": "equal"})

    background_wedge = Wedge(
        center=(0, 0), r=1.0, theta1=0, theta2=180, width=0.35, facecolor="#3A3F44", alpha=0.25
    )
    value_wedge = Wedge(
        center=(0, 0),
        r=1.0,
        theta1=180 - (confidence * 180),
        theta2=180,
        width=0.35,
        facecolor=class_color,
    )
    ax.add_patch(background_wedge)
    ax.add_patch(value_wedge)

    ax.text(0, -0.15, f"{confidence * 100:.1f}%", ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.42, "Confidence", ha="center", va="center", fontsize=10, alpha=0.7)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig


# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------

def render_home_page(dataset_df: Optional[pd.DataFrame]) -> None:
    """
    Render the Home page: project title, description, model info,
    dataset info, and developer info.

    Args:
        dataset_df: The loaded dataset DataFrame, or None if unavailable.
    """
    st.markdown(f"# {config.APP_ICON} {config.APP_TITLE}")
    st.markdown(f"##### {config.APP_TAGLINE}")
    st.divider()

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### 📌 Project Overview")
    st.write(
        "This application performs automated visual inspection of semiconductor "
        "wafer maps to detect and classify manufacturing defect patterns. It "
        "leverages a ResNet50 convolutional neural network trained via transfer "
        "learning, and provides Grad-CAM based explainability so that engineers "
        "can visually verify *why* the model made a given prediction — a critical "
        "requirement for trust and adoption in industrial semiconductor fabs."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Architecture", "ResNet50")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Defect Classes", str(config.NUM_CLASSES))
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_images = len(dataset_df) if dataset_df is not None else 0
        st.metric("Dataset Images", f"{total_images:,}" if total_images else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Model Information")
        for key, value in list(config.MODEL_METADATA.items())[:5]:
            st.markdown(f"- **{key}:** {value}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🗃️ Dataset Information")
        summary = utils.get_dataset_summary(dataset_df)
        if summary:
            st.markdown(f"- **Total Images:** {summary.get('total_images', 'N/A')}")
            st.markdown(f"- **Train Images:** {summary.get('train_count', 'N/A')}")
            st.markdown(f"- **Validation Images:** {summary.get('valid_count', 'N/A')}")
            st.markdown(f"- **Test Images:** {summary.get('test_count', 'N/A')}")
        else:
            st.info("Dataset CSV not found. Displaying placeholder information.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 👨‍💻 Developer Information")
    dev_cols = st.columns(len(config.DEVELOPER_INFO))
    for col, (key, value) in zip(dev_cols, config.DEVELOPER_INFO.items()):
        with col:
            st.markdown(f"**{key}**")
            st.caption(value)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# IMAGE UPLOAD & PREDICTION PAGE
# ------------------------------------------------------

def render_prediction_page(model) -> None:
    """
    Render the Image Upload & Prediction page: upload widget,
    preprocessing, inference, prediction display, confidence
    gauge, and class probability chart.

    Args:
        model: The loaded Keras classification model.
    """
    st.markdown('<div class="section-header">🖼️ Image Upload & Prediction</div>', unsafe_allow_html=True)

    if model is None:
        st.error(
            f"⚠️ Model file not found at `{config.MODEL_PATH}`. "
            "Please place `best_resnet50.keras` in the project root."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a wafer map image",
        type=config.ALLOWED_IMAGE_EXTENSIONS,
        help=f"Accepted formats: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS)}. "
             f"Max recommended size: {config.MAX_UPLOAD_SIZE_MB} MB.",
    )

    if uploaded_file is not None:
        image = utils.load_image(uploaded_file)
        if image is None:
            st.error("❌ Could not read the uploaded file as a valid image.")
            return

        st.session_state["uploaded_image"] = image
        st.session_state["prediction_result"] = None
        st.session_state["gradcam_visuals"] = None

        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded Wafer Image", use_container_width=True)
        with col_info:
            st.markdown("**Image Details**")
            st.markdown(f"- Original size: `{image.size[0]} x {image.size[1]}`")
            st.markdown(f"- Model input size: `{config.IMAGE_SIZE[0]} x {config.IMAGE_SIZE[1]}`")
            st.markdown(f"- Format: `{uploaded_file.type}`")

            run_prediction = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

        if run_prediction:
            with st.spinner("🔬 Analyzing wafer pattern and running inference..."):
                try:
                    result = predictor.predict(model, image)
                    st.session_state["prediction_result"] = result
                    st.session_state["preprocessed_input"] = result["preprocessed_input"]

                    history_record = utils.build_history_record(
                        predicted_class=result["predicted_class"],
                        confidence=result["confidence"],
                        inference_time=result["inference_time"],
                        probabilities=result["probabilities"],
                    )
                    utils.append_prediction_history(history_record)
                except ValueError as exc:
                    st.error(f"❌ Prediction failed: {exc}")

    result = st.session_state.get("prediction_result")
    if result is not None:
        _render_prediction_results(result)


def _render_prediction_results(result: Dict[str, object]) -> None:
    """
    Render the results block (metrics, gauge, bar chart, download
    buttons) for a completed prediction.

    Args:
        result: The prediction result dictionary returned by
            ``predictor.predict``.
    """
    st.divider()
    predicted_class = str(result["predicted_class"])
    confidence = float(result["confidence"])
    inference_time = float(result["inference_time"])
    probabilities = result["probabilities"]
    class_color = utils.get_class_color(predicted_class)
    confidence_level = utils.get_confidence_level_label(confidence)

    st.markdown(
        f'<span class="prediction-badge" style="background-color:{class_color};">'
        f'Predicted Defect: {predicted_class}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted Class", predicted_class)
    metric_col2.metric("Confidence", utils.format_confidence(confidence), confidence_level)
    metric_col3.metric("Inference Time", utils.format_inference_time(inference_time))

    st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {utils.format_confidence(confidence)}")

    gauge_col, chart_col = st.columns([1, 1.4])
    with gauge_col:
        st.pyplot(plot_confidence_gauge(confidence, class_color), use_container_width=True)
    with chart_col:
        st.pyplot(plot_probabilities_bar_chart(probabilities), use_container_width=True)

    with st.expander("📄 Raw Class Probabilities"):
        prob_df = pd.DataFrame(
            {"Class": list(probabilities.keys()), "Probability (%)": [v * 100 for v in probabilities.values()]}
        ).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    report_text = utils.generate_prediction_report_text(
        predicted_class=predicted_class,
        confidence=confidence,
        inference_time=inference_time,
        probabilities=probabilities,
    )
    st.download_button(
        "📥 Download Prediction Report",
        data=report_text,
        file_name=f"prediction_report_{predicted_class}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("🕘 Recent Prediction History"):
        history = utils.load_prediction_history()
        if history:
            display_columns = ["timestamp", "predicted_class", "confidence", "inference_time_seconds"]
            history_df = pd.DataFrame(history[-10:][::-1])
            history_df = history_df[[col for col in display_columns if col in history_df.columns]]
            history_df = history_df.rename(
                columns={
                    "timestamp": "Timestamp",
                    "predicted_class": "Predicted Class",
                    "confidence": "Confidence",
                    "inference_time_seconds": "Inference Time (s)",
                }
            )
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No prediction history recorded yet.")

    st.info("➡️ Go to the **Grad-CAM Explainability** page to visualize why the model made this prediction.")


# ------------------------------------------------------
# GRAD-CAM EXPLAINABILITY PAGE
# ------------------------------------------------------

def render_gradcam_page(model) -> None:
    """
    Render the Grad-CAM Explainability page: generates and displays
    the original image, heatmap, and overlay for the most recent
    prediction, with a download option for the overlay image.

    Args:
        model: The loaded Keras classification model.
    """
    st.markdown('<div class="section-header">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)

    image = st.session_state.get("uploaded_image")
    preprocessed_input = st.session_state.get("preprocessed_input")
    result = st.session_state.get("prediction_result")

    if model is None:
        st.error(f"⚠️ Model file not found at `{config.MODEL_PATH}`.")
        return

    if image is None or preprocessed_input is None or result is None:
        st.warning(
            "⚠️ Please upload an image and run a prediction on the "
            "**Image Upload & Prediction** page first."
        )
        return

    predicted_class_id = int(result["predicted_class_id"])
    predicted_class = str(result["predicted_class"])

    generate_clicked = st.button("🔥 Generate Grad-CAM Visualization", type="primary")

    if generate_clicked or st.session_state.get("gradcam_visuals") is not None:
        if generate_clicked:
            with st.spinner("🎨 Computing gradient activations and rendering heatmap..."):
                try:
                    visuals = gradcam.generate_gradcam_visuals(
                        model=model,
                        original_image=image,
                        preprocessed_input=preprocessed_input,
                        class_index=predicted_class_id,
                    )
                    st.session_state["gradcam_visuals"] = visuals
                except ValueError as exc:
                    st.error(f"❌ Grad-CAM generation failed: {exc}")
                    return

        visuals = st.session_state["gradcam_visuals"]
        if visuals is None:
            return

        st.success(f"Explaining prediction: **{predicted_class}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(visuals["original_image"], caption="Original Image", use_container_width=True)
        with col2:
            st.image(visuals["heatmap_image"], caption="Grad-CAM Heatmap", use_container_width=True)
        with col3:
            st.image(visuals["overlay_image"], caption="Overlay", use_container_width=True)

        overlay_bytes = utils.format_bytes_as_image(visuals["overlay_image"], fmt="PNG")
        st.download_button(
            "📥 Download Grad-CAM Overlay",
            data=overlay_bytes,
            file_name=f"gradcam_overlay_{predicted_class}.png",
            mime="image/png",
            use_container_width=True,
        )

        with st.expander("ℹ️ How to interpret this visualization"):
            st.write(
                "Warmer colors (red / yellow) indicate image regions that "
                "contributed most strongly to the model's predicted class, "
                "while cooler colors (blue) indicate regions with minimal "
                "influence. This helps verify that the model is focusing on "
                "the actual defect pattern rather than irrelevant artifacts."
            )


# ------------------------------------------------------
# MODEL INFORMATION PAGE
# ------------------------------------------------------

def render_model_info_page() -> None:
    """Render the Model Information page with architecture and training metadata."""
    st.markdown('<div class="section-header">🧠 Model Information</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### Architecture Summary")
    st.code(config.MODEL_ARCHITECTURE, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2)
    metadata_items = list(config.MODEL_METADATA.items())
    midpoint = (len(metadata_items) + 1) // 2

    with info_col1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        for key, value in metadata_items[:midpoint]:
            st.markdown(f"**{key}:** {value}")
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        for key, value in metadata_items[midpoint:]:
            st.markdown(f"**{key}:** {value}")
        st.markdown(f"**Last Conv Layer (Grad-CAM):** `{config.LAST_CONV_LAYER_NAME}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### Class Labels")
    label_cols = st.columns(config.NUM_CLASSES)
    for col, class_name in zip(label_cols, config.CLASS_LABELS):
        with col:
            color = utils.get_class_color(class_name)
            st.markdown(
                f'<div style="background-color:{color};padding:0.5rem;border-radius:0.5rem;'
                f'text-align:center;color:white;font-size:0.8rem;">{class_name}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# PERFORMANCE PAGE
# ------------------------------------------------------

def render_performance_page() -> None:
    """
    Render the Performance page, displaying accuracy/loss metrics,
    the confusion matrix, and the classification report. Falls back
    to placeholders when the corresponding artifacts are unavailable.
    """
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Validation Accuracy", config.MODEL_METADATA.get("Validation Accuracy", "N/A"))
    with metric_col2:
        st.metric("Validation Loss", config.MODEL_METADATA.get("Validation Loss", "N/A"))

    st.write("")
    col_cm, col_report = st.columns(2)

    with col_cm:
        st.markdown("#### 🔢 Confusion Matrix")
        confusion_matrix_path = config.PERFORMANCE_DIR / "confusion_matrix.png"
        if utils.file_exists(confusion_matrix_path):
            st.image(str(confusion_matrix_path), use_container_width=True)
        else:
            st.info(
                "Confusion matrix image not found.\n\n"
                f"Place a `confusion_matrix.png` file inside "
                f"`{config.PERFORMANCE_DIR}` to display it here."
            )

    with col_report:
        st.markdown("#### 📋 Classification Report")
        report_path = config.PERFORMANCE_DIR / "classification_report.csv"
        if utils.file_exists(report_path):
            try:
                report_df = pd.read_csv(report_path, index_col=0)
                st.dataframe(report_df, use_container_width=True)
            except Exception:  # noqa: BLE001
                st.warning("Found classification_report.csv but could not parse it.")
        else:
            st.info(
                "Classification report not found.\n\n"
                f"Place a `classification_report.csv` file inside "
                f"`{config.PERFORMANCE_DIR}` to display it here."
            )

    st.write("")
    st.markdown("#### 📈 Training Curves")
    training_curve_path = config.PERFORMANCE_DIR / "training_history.png"
    if utils.file_exists(training_curve_path):
        st.image(str(training_curve_path), use_container_width=True)
    else:
        st.info(
            "Training curve image not found.\n\n"
            f"Place a `training_history.png` file inside "
            f"`{config.PERFORMANCE_DIR}` to display it here."
        )


# ------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------

def render_about_page() -> None:
    """Render the About page with educational explanations of the core concepts."""
    st.markdown('<div class="section-header">📘 About This Project</div>', unsafe_allow_html=True)

    topics = {
        "🧩 Convolutional Neural Networks (CNNs)": (
            "CNNs are deep learning architectures designed to process grid-like data "
            "such as images. They use learnable convolutional filters to automatically "
            "extract spatial features — edges, textures, and shapes — at increasing "
            "levels of abstraction through successive layers."
        ),
        "🔄 Transfer Learning": (
            "Transfer learning reuses a model previously trained on a large, general "
            "dataset (such as ImageNet) as the starting point for a new, related task. "
            "This drastically reduces training time and data requirements while "
            "improving performance, since the model already understands general "
            "visual patterns."
        ),
        "🏗️ ResNet50": (
            "ResNet50 is a 50-layer deep residual convolutional neural network. It "
            "introduces 'skip connections' that allow gradients to flow directly "
            "across layers, solving the vanishing-gradient problem and enabling "
            "very deep networks to be trained effectively."
        ),
        "🔥 Grad-CAM": (
            "Gradient-weighted Class Activation Mapping (Grad-CAM) is an "
            "explainability technique that uses the gradients of a target class "
            "flowing into the final convolutional layer to produce a coarse "
            "localization heatmap, highlighting the regions of the image most "
            "influential to the prediction."
        ),
        "🏭 Semiconductor Wafer Inspection": (
            "During semiconductor fabrication, wafers are inspected for defect "
            "patterns that can indicate specific process issues (e.g. edge ring "
            "defects from etching problems, or scratch defects from handling "
            "damage). Automated visual inspection helps catch these issues early "
            "in the manufacturing pipeline."
        ),
        "⚙️ Industrial Applications": (
            "AI-driven defect classification systems like this one support yield "
            "engineers and process teams by accelerating root-cause analysis, "
            "reducing manual inspection workload, and enabling real-time quality "
            "control decisions on the fab floor."
        ),
    }

    for title, description in topics.items():
        with st.expander(title, expanded=False):
            st.write(description)


# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------

def render_footer() -> None:
    """Render a professional footer shown at the bottom of every page."""
    st.markdown(
        f"""
        <div class="app-footer">
            {config.APP_ICON} <strong>{config.APP_TITLE}</strong> &nbsp;|&nbsp;
            Powered by TensorFlow, ResNet50 & Grad-CAM &nbsp;|&nbsp;
            Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------
# MAIN APPLICATION ENTRY POINT
# ------------------------------------------------------

def main() -> None:
    """Application entry point: initializes state, renders navigation, and dispatches pages."""
    inject_custom_css()
    init_session_state()

    model = predictor.load_wafer_model()
    dataset_df = utils.load_dataset_csv()

    selected_page = render_sidebar()

    if selected_page == "Home":
        render_home_page(dataset_df)
    elif selected_page == "Image Upload & Prediction":
        render_prediction_page(model)
    elif selected_page == "Grad-CAM Explainability":
        render_gradcam_page(model)
    elif selected_page == "Model Information":
        render_model_info_page()
    elif selected_page == "Performance":
        render_performance_page()
    elif selected_page == "About":
        render_about_page()

    render_footer()


if __name__ == "__main__":
    main()
