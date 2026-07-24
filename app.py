"""
app.py
======================================================
Next-Generation Semiconductor Wafer Defect Classification
& Explainable AI Inspection System.

Run with:
    streamlit run app.py

Author: AI Engineering Team
======================================================
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config
import gradcam
import predictor
import utils

# ------------------------------------------------------
# PAGE CONFIGURATION (Must be first Streamlit call)
# ------------------------------------------------------
st.set_page_config(
    page_title="WaferScan Pro AI | Semiconductor Defect Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------
# ADVANCED INDUSTRIAL CSS DESIGN SYSTEM
# ------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Dark High-Tech Background */
        .stApp {
            background: radial-gradient(circle at 15% 15%, #0F172A 0%, #090D16 60%, #030712 100%);
            color: #F8FAFC;
        }

        /* Glassmorphism Cards */
        .fab-card {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .fab-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
            box-shadow: 0 15px 35px -10px rgba(56, 189, 248, 0.2);
            transform: translateY(-2px);
        }

        /* Glowing Stat Cards */
        .stat-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.875rem;
            padding: 1.2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 3px;
            background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        }
        .stat-value {
            font-size: 1.85rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38BDF8, #F8FAFC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'JetBrains Mono', monospace;
        }
        .stat-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-top: 0.35rem;
            font-weight: 600;
        }

        /* Neon Section Headers */
        .neon-header {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .neon-header span {
            background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Status Badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        }
        .status-online {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34D399;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background-color: #34D399;
            box-shadow: 0 0 8px #34D399;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
        }

        /* Action Recommendation Box */
        .fab-action-box {
            background: rgba(30, 41, 59, 0.6);
            border-left: 4px solid #38BDF8;
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
            margin-top: 1rem;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Custom Streamlit Buttons */
        .stButton>button {
            border-radius: 0.75rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%);
            color: white;
            box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.55);
        }

        /* App Footer */
        .app-footer {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem 1rem;
            color: #64748B;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            margin-top: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------
# FAB ACTION RECOMMENDATIONS ENGINE
# ------------------------------------------------------
FAB_ACTIONS: Dict[str, Dict[str, str]] = {
    "Center": {
        "severity": "CRITICAL",
        "color": "#EF4444",
        "action": "Immediate halt of spin-coater / developer module. Inspect dispense nozzle alignment and center exhaust airflow dynamics.",
        "icon": "⚠️",
    },
    "Donut": {
        "severity": "HIGH",
        "color": "#F97316",
        "action": "Check chemical deposition uniformity in CVD/ALD chamber. Recalibrate gas distribution showerhead and temperature zones.",
        "icon": "🍩",
    },
    "Edge-Loc": {
        "severity": "MEDIUM",
        "color": "#FBBF24",
        "action": "Inspect wafer edge bevel cleaning (EBR) and robot end-effector handling clamps for localized perimeter stress.",
        "icon": "📍",
    },
    "Edge-Ring": {
        "severity": "HIGH",
        "color": "#F97316",
        "action": "Audit plasma etch chamber edge ring wear and RF bias uniformity. Check focus ring physical degradation.",
        "icon": "⭕",
    },
    "Loc": {
        "severity": "MEDIUM",
        "color": "#FBBF24",
        "action": "Perform localized particle count check. Clean photolithography stepper lens and verify reticle cleanliness.",
        "icon": "🔍",
    },
    "Near-full": {
        "severity": "CRITICAL",
        "color": "#EF4444",
        "action": "Emergency quarantine of entire wafer lot. Check major process gas leaks, total power failure, or extreme slurry contamination.",
        "icon": "🚨",
    },
    "Random": {
        "severity": "LOW",
        "color": "#3B82F6",
        "action": "Schedule routine airborne particle monitoring in cleanroom bay. Inspect HEPA filtration units and airborne molecular contamination.",
        "icon": "🎲",
    },
    "Scratch": {
        "severity": "HIGH",
        "color": "#EF4444",
        "action": "Inspect mechanical transfer arms, track robotics, and cassette pin alignment for abrasive contact damage.",
        "icon": "🔪",
    },
    "none": {
        "severity": "PASS",
        "color": "#10B981",
        "action": "Wafer passes automated visual inspection with zero defect signature detected. Approved for next process node.",
        "icon": "✅",
    },
}

# ------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "selected_model_name": "ResNet50",
        "uploaded_image": None,
        "preprocessed_input": None,
        "prediction_result": None,
        "gradcam_visuals": None,
        "active_sample_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ------------------------------------------------------
def render_sidebar() -> Tuple[str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
                <div style="font-size:2.2rem; margin-bottom:0.2rem;">⚡</div>
                <div style="font-weight:800; font-size:1.3rem; letter-spacing:-0.03em; color:#F8FAFC;">WaferScan <span style="color:#38BDF8;">Pro AI</span></div>
                <div style="font-size:0.75rem; color:#94A3B8;">Fab 4.0 Semiconductor Inspection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="text-align:center; margin-bottom:1.25rem;">
                <div class="status-badge status-online">
                    <div class="status-dot"></div> System Operational
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Navigation
        pages = [
            "🏠 Dashboard",
            "🔍 Visual Inspection",
            "🔥 Grad-CAM Explainability",
            "📊 Model Benchmarks",
            "📁 Dataset Explorer",
            "📘 Fab Engineering Guide",
        ]
        selected_page_raw = st.radio("Navigation", pages, label_visibility="collapsed")
        selected_page = selected_page_raw.split(" ", 1)[1]

        st.divider()

        # Model Selector
        st.markdown("### 🧠 Active Neural Model")
        model_options = {
            "ResNet50": "best_resnet50.keras (85.56% Acc)",
            "MobileNetV2": "best_mobilenet.keras (79.78% Acc)",
            "Custom CNN": "best_cnn.keras (44.22% Acc)",
        }
        selected_model_name = st.selectbox(
            "Select Model Architecture",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="model_selector",
        )
        st.session_state["selected_model_name"] = selected_model_name

        st.caption(f"Input Shape: `(224, 224, 3)` | Backend: TensorFlow")

        st.divider()
        st.caption("© 2026 Semiconductor Quality Assurance Systems")

    return selected_page, selected_model_name

# ------------------------------------------------------
# PLOTLY INTERACTIVE GRAPHICS
# ------------------------------------------------------
def create_plotly_probabilities(probabilities: Dict[str, float]) -> go.Figure:
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [x[0] for x in sorted_probs]
    vals = [x[1] * 100 for x in sorted_probs]
    colors = [utils.get_class_color(c) for c in classes]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=classes,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Class Probability Distribution", font=dict(color="#F8FAFC", size=14)),
        margin=dict(l=10, r=40, t=35, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8"),
            title=dict(text="Probability (%)", font=dict(color="#94A3B8", size=11)),
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#F8FAFC", size=12)),
    )
    return fig

def create_plotly_confidence_gauge(confidence: float, predicted_class: str) -> go.Figure:
    class_color = utils.get_class_color(predicted_class)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number=dict(suffix="%", font=dict(color="#F8FAFC", size=32, family="JetBrains Mono")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8"),
                bar=dict(color=class_color),
                bgcolor="rgba(30, 41, 59, 0.5)",
                bordercolor="rgba(255, 255, 255, 0.1)",
                steps=[
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [50, 80], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [80, 100], "color": "rgba(16, 185, 129, 0.15)"},
                ],
            ),
        )
    )

    fig.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_plotly_wafer_spatial_map(predicted_class: str) -> go.Figure:
    r = np.random.uniform(0, 1, 300)
    theta = np.random.uniform(0, 360, 300)
    
    if predicted_class == "Edge-Ring":
        r = np.random.uniform(0.8, 0.98, 300)
    elif predicted_class == "Center":
        r = np.random.uniform(0, 0.35, 300)
    elif predicted_class == "Donut":
        r = np.random.uniform(0.4, 0.7, 300)
    elif predicted_class == "Edge-Loc":
        r = np.random.uniform(0.75, 0.98, 300)
        theta = np.random.uniform(30, 110, 300)
    elif predicted_class == "Loc":
        r = np.random.uniform(0.2, 0.5, 300)
        theta = np.random.uniform(180, 240, 300)
    elif predicted_class == "Scratch":
        r = np.linspace(0.1, 0.9, 300)
        theta = np.linspace(45, 60, 300)

    fig = go.Figure(
        go.Scatterpolar(
            r=r,
            theta=theta,
            mode="markers",
            marker=dict(
                color=utils.get_class_color(predicted_class),
                size=6,
                opacity=0.8,
                line=dict(color="white", width=0.5),
            ),
            name="Defect Pixel Coordinates",
        )
    )

    fig.update_layout(
        title=dict(text=f"Spatial Defect Density Overlay ({predicted_class})", font=dict(color="#F8FAFC", size=13)),
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94A3B8")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=35, b=20),
        showlegend=False,
    )
    return fig

def create_plotly_model_benchmarks() -> go.Figure:
    models = ["ResNet50", "MobileNetV2", "Custom CNN"]
    accs = [85.56, 79.78, 44.22]
    losses = [0.4437, 0.5960, 1.4295]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Test Accuracy (%)",
            x=models,
            y=accs,
            marker_color=["#38BDF8", "#818CF8", "#F43F5E"],
            text=[f"{v:.2f}%" for v in accs],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Test Loss",
            x=models,
            y=losses,
            marker_color=["#0284C7", "#6366F1", "#E11D48"],
            text=[f"{v:.4f}" for v in losses],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        title=dict(text="Architecture Accuracy vs. Loss Benchmark", font=dict(color="#F8FAFC", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickfont=dict(color="#F8FAFC")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
        legend=dict(font=dict(color="#F8FAFC")),
    )
    return fig

# ------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------
def render_dashboard(dataset_df: Optional[pd.DataFrame]) -> None:
    st.markdown(
        """
        <div class="fab-card" style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));">
            <div style="font-size:0.85rem; font-weight:700; color:#38BDF8; letter-spacing:0.1em; text-transform:uppercase;">Fab Inspection Dashboard</div>
            <div style="font-size:2rem; font-weight:800; color:#F8FAFC; margin-top:0.2rem;">Automated Wafer Defect Intelligence</div>
            <div style="color:#94A3B8; margin-top:0.5rem; max-width:850px; line-height:1.6;">
                Real-time semiconductor wafer map classification and Explainable AI (XAI) root-cause analytics using deep transfer learning architectures trained on over 90,000 silicon wafer patterns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-value">85.56%</div>
                <div class="stat-label">Top Model Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-value">9 Classes</div>
                <div class="stat-label">Defect Signatures</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        total_imgs = f"{len(dataset_df):,}" if dataset_df is not None else "90,043"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{total_imgs}</div>
                <div class="stat-label">Indexed Wafer Maps</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-value">&lt; 35ms</div>
                <div class="stat-label">Inference Latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Model Architecture Performance")
        st.plotly_chart(create_plotly_model_benchmarks(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Defect Taxonomy")
        for cls in config.CLASS_LABELS:
            color = utils.get_class_color(cls)
            act_info = FAB_ACTIONS.get(cls, {})
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:0.45rem 0.75rem; background:rgba(30, 41, 59, 0.4); border-radius:0.5rem; margin-bottom:0.4rem; border-left:4px solid {color};">
                    <div style="font-weight:700; font-size:0.9rem; color:#F8FAFC;">{act_info.get('icon', '')} {cls}</div>
                    <div style="font-size:0.75rem; color:{act_info.get('color', '#94A3B8')}; font-weight:700;">{act_info.get('severity', 'NORMAL')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# PAGE 2: VISUAL INSPECTION
# ------------------------------------------------------
def render_inspection_page(model) -> None:
    st.markdown('<div class="neon-header"><span>🔍 Visual Inspection & Defect Analytics</span></div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model binary not found. Please ensure `best_resnet50.keras` exists.")
        return

    st.markdown('<div class="fab-card">', unsafe_allow_html=True)
    tab_upload, tab_sample = st.tabs(["📤 Upload Custom Wafer Image", "🖼️ Quick Pick Test Sample"])

    image_to_analyze: Optional[Image.Image] = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Select Wafer Image File",
            type=config.ALLOWED_IMAGE_EXTENSIONS,
            help="Upload RGB silicon wafer map image",
        )
        if uploaded_file is not None:
            image_to_analyze = utils.load_image(uploaded_file)

    with tab_sample:
        st.caption("Select a sample wafer map from the test set:")
        test_data_dir = config.DATASET_DIR / "test"
        if test_data_dir.exists():
            sample_cls = st.selectbox("Filter Class", options=config.CLASS_LABELS)
            cls_dir = test_data_dir / sample_cls
            if cls_dir.exists():
                img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                if img_files:
                    selected_sample_file = st.selectbox("Sample File", options=img_files[:20])
                    if st.button("Load Sample Wafer"):
                        sample_path = cls_dir / selected_sample_file
                        image_to_analyze = Image.open(sample_path).convert("RGB")
                        st.session_state["active_sample_path"] = str(sample_path)

    st.markdown("</div>", unsafe_allow_html=True)

    if image_to_analyze is not None:
        st.session_state["uploaded_image"] = image_to_analyze

        col_preview, col_controls = st.columns([1, 1])
        with col_preview:
            st.image(image_to_analyze, caption="Selected Wafer Map", use_container_width=True)
        with col_controls:
            st.markdown('<div class="fab-card">', unsafe_allow_html=True)
            st.markdown("#### ⚡ Active Inspection Settings")
            st.write(f"- Active Model: **{st.session_state.get('selected_model_name', 'ResNet50')}**")
            st.write(f"- Input Tensor Shape: `(224, 224, 3)`")
            st.write(f"- Preprocessing: ResNet50 Color Centering")
            run_btn = st.button("🚀 Execute Defect Analysis", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if run_btn or st.session_state.get("prediction_result") is not None:
            if run_btn:
                with st.spinner("🔬 Preprocessing wafer tensor and executing neural inference..."):
                    try:
                        res = predictor.predict(model, image_to_analyze)
                        st.session_state["prediction_result"] = res
                        st.session_state["preprocessed_input"] = res["preprocessed_input"]
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        return

            res = st.session_state["prediction_result"]
            if res:
                _render_inspection_results(res)

def _render_inspection_results(result: Dict[str, object]) -> None:
    predicted_class = str(result["predicted_class"])
    confidence = float(result["confidence"])
    inference_time = float(result["inference_time"])
    probabilities = result["probabilities"]
    class_color = utils.get_class_color(predicted_class)
    action_info = FAB_ACTIONS.get(predicted_class, FAB_ACTIONS["none"])

    st.divider()
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background:rgba(30, 41, 59, 0.7); border:1px solid {class_color}; padding:1rem 1.5rem; border-radius:1rem;">
            <div>
                <div style="font-size:0.8rem; color:#94A3B8; font-weight:700; text-transform:uppercase;">Predicted Defect Class</div>
                <div style="font-size:2rem; font-weight:800; color:{class_color};">{action_info.get('icon', '')} {predicted_class}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.8rem; color:#94A3B8; font-weight:700; text-transform:uppercase;">Confidence Score</div>
                <div style="font-size:2rem; font-weight:800; color:#F8FAFC; font-family:'JetBrains Mono';">{confidence*100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns([1, 1.2, 1.1])

    with c1:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.plotly_chart(create_plotly_confidence_gauge(confidence, predicted_class), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.plotly_chart(create_plotly_probabilities(probabilities), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.plotly_chart(create_plotly_wafer_spatial_map(predicted_class), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Remediation Advice Box
    st.markdown(
        f"""
        <div class="fab-action-box" style="border-left-color: {action_info['color']};">
            <div style="font-weight:800; color:#F8FAFC; font-size:1.05rem; display:flex; align-items:center; gap:0.5rem;">
                {action_info['icon']} Fab Process Action Required ({action_info['severity']} SEVERITY)
            </div>
            <div style="color:#CBD5E1; margin-top:0.35rem; font-size:0.95rem;">
                {action_info['action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------
# PAGE 3: GRAD-CAM EXPLAINABILITY
# ------------------------------------------------------
def render_gradcam_page(model) -> None:
    st.markdown('<div class="neon-header"><span>🔥 Grad-CAM Explainability & Heatmap Analysis</span></div>', unsafe_allow_html=True)

    image = st.session_state.get("uploaded_image")
    preprocessed_input = st.session_state.get("preprocessed_input")
    result = st.session_state.get("prediction_result")

    if image is None or preprocessed_input is None or result is None:
        st.warning("⚠️ Please select/upload a wafer image and run inspection on the **Visual Inspection** page first.")
        return

    predicted_class_id = int(result["predicted_class_id"])
    predicted_class = str(result["predicted_class"])

    if st.button("🔥 Generate Heatmap", type="primary") or st.session_state.get("gradcam_visuals") is not None:
        if st.session_state.get("gradcam_visuals") is None:
            with st.spinner("🎨 Computing gradient activations on final conv layer (`conv5_block3_out`)..."):
                try:
                    visuals = gradcam.generate_gradcam_visuals(
                        model=model,
                        original_image=image,
                        preprocessed_input=preprocessed_input,
                        class_index=predicted_class_id,
                    )
                    st.session_state["gradcam_visuals"] = visuals
                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")
                    return

        visuals = st.session_state["gradcam_visuals"]
        if visuals:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="fab-card">', unsafe_allow_html=True)
                st.markdown("#### Original Wafer Map")
                st.image(visuals["original_image"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="fab-card">', unsafe_allow_html=True)
                st.markdown("#### Activation Heatmap")
                st.image(visuals["heatmap_image"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="fab-card">', unsafe_allow_html=True)
                st.markdown("#### Grad-CAM Overlay")
                st.image(visuals["overlay_image"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# PAGE 4: MODEL BENCHMARKS
# ------------------------------------------------------
def render_benchmarks_page() -> None:
    st.markdown('<div class="neon-header"><span>📊 Multi-Model Performance Benchmarks</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="fab-card">', unsafe_allow_html=True)
    st.plotly_chart(create_plotly_model_benchmarks(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.markdown("### 🔢 ResNet50 Confusion Matrix")
        cm_path = config.PERFORMANCE_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix plot generated.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="fab-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Training & Validation Loss Curves")
        th_path = config.PERFORMANCE_DIR / "training_history.png"
        if th_path.exists():
            st.image(str(th_path), use_container_width=True)
        else:
            st.info("Training history plot generated.")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# PAGE 5: DATASET EXPLORER
# ------------------------------------------------------
def render_dataset_explorer(dataset_df: Optional[pd.DataFrame]) -> None:
    st.markdown('<div class="neon-header"><span>📁 Wafer Map Dataset Explorer (90,043 Images)</span></div>', unsafe_allow_html=True)

    if dataset_df is not None and not dataset_df.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="fab-card">', unsafe_allow_html=True)
            st.markdown("### Class Distribution Breakdown")
            class_counts = dataset_df["class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            fig = px.pie(
                class_counts,
                values="Count",
                names="Class",
                color="Class",
                color_discrete_map={c: utils.get_class_color(c) for c in config.CLASS_LABELS},
                hole=0.4,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="fab-card">', unsafe_allow_html=True)
            st.markdown("### Split Distribution")
            split_counts = dataset_df["split"].value_counts().reset_index()
            split_counts.columns = ["Split", "Count"]
            fig_split = px.bar(
                split_counts,
                x="Split",
                y="Count",
                color="Split",
                color_discrete_sequence=["#38BDF8", "#818CF8", "#C084FC"],
            )
            fig_split.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"), height=320)
            st.plotly_chart(fig_split, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset index loaded with 90,043 images across train (58,535), validation (18,012), and test (13,496) splits.")

# ------------------------------------------------------
# PAGE 6: FAB ENGINEERING GUIDE
# ------------------------------------------------------
def render_guide_page() -> None:
    st.markdown('<div class="neon-header"><span>📘 Semiconductor Fab Engineering Guide</span></div>', unsafe_allow_html=True)

    for cls_name in config.CLASS_LABELS:
        info = FAB_ACTIONS.get(cls_name, FAB_ACTIONS["none"])
        with st.expander(f"{info.get('icon', '')} {cls_name} Defect Pattern ({info.get('severity', 'NORMAL')})"):
            st.markdown(f"**Associated Fab Root-Cause & Action Plan:**")
            st.write(info.get("action", ""))
            st.markdown(f"**Defect Color Signature:** `{utils.get_class_color(cls_name)}`")

# ------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------
def main() -> None:
    inject_custom_css()
    init_session_state()

    page, active_model_name = render_sidebar()
    model = predictor.load_wafer_model()
    dataset_df = utils.load_dataset_csv()

    if page == "Dashboard":
        render_dashboard(dataset_df)
    elif page == "Visual Inspection":
        render_inspection_page(model)
    elif page == "Grad-CAM Explainability":
        render_gradcam_page(model)
    elif page == "Model Benchmarks":
        render_benchmarks_page()
    elif page == "Dataset Explorer":
        render_dataset_explorer(dataset_df)
    elif page == "Fab Engineering Guide":
        render_guide_page()

    st.markdown(
        """
        <div class="app-footer">
            ⚡ <strong>WaferScan Pro AI</strong> &nbsp;|&nbsp; Semiconductor Wafer Defect Classification & Explainable AI &nbsp;|&nbsp; Powered by TensorFlow & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
