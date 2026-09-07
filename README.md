# Trustworthy Real-Time Wafer Defect Classification Through Quantitative Explanation Validation

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red.svg)](app.py)
[![Paper Manuscript](https://img.shields.io/badge/Manuscript-IEEE%20Format-green.svg)](PAPER_MANUSCRIPT.md)

This repository contains the complete codebase, quantitative evaluation suite, publication figures, and manuscript draft for **"Trustworthy Real-Time Wafer Defect Classification Through Quantitative Explanation Validation and Multi-Architecture Trade-Off Analysis"**.

---

## 📑 Research Manuscript & Strategy
- **Full Manuscript Draft:** [`PAPER_MANUSCRIPT.md`](PAPER_MANUSCRIPT.md)
- **Research Upgrade Blueprint:** [`RESEARCH_UPGRADE_PLAN.md`](RESEARCH_UPGRADE_PLAN.md)
- **Presentation Deck Blueprint:** [`Wafer_Defect_Classification_Presentation.pdf`](Wafer_Defect_Classification_Presentation.pdf) | [`Wafer_Defect_Classification_Presentation.pptx`](Wafer_Defect_Classification_Presentation.pptx)

---

## 🔬 Key Research Contributions

1. **Quantitative Explanation Validation Framework:** Formulates an IoU, Dice Coefficient, Pointing Game Accuracy, and Inside-Mask Energy Ratio metric suite to evaluate Grad-CAM and attention heatmap fidelity against ground-truth spatial defect regions.
2. **Confidence-Fidelity & Misclassification Correlation:** Establishes that explanation fidelity degrades sharply on false positive predictions ($IoU < 0.20$), serving as a real-time confidence calibration signal.
3. **Multi-Architecture Trade-Off Matrix:** Benchmarks Custom CNN, MobileNetV2, ResNet50, and Vision Transformer (ViT-Patch16) models across **Accuracy, Macro-F1, Minority Recall, Parameters, Latency, and Explanation Fidelity**.
4. **Disaggregated Latency-Aware Pipeline:** Demonstrates a cleanroom inspection system with disaggregated throughput (1.1 ms preproc + 6.4 ms inference + 8.2 ms Grad-CAM = **15.7 ms GPU Total**), satisfying real-time cleanroom SLAs (<35 ms).

---

## 📊 Experimental Results Summary

| Model Architecture | Parameters | Test Accuracy (%) | Macro-F1 Score | Minority Recall (Near-full) | Forward Latency (GPU ms) | Explanation IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Baseline)** | ~1.2M | 44.22% | 0.385 | 32.1% | **8.2 ms** | 0.220 |
| **MobileNetV2 (Edge AI)** | ~3.5M | 79.78% | 0.742 | 68.4% | **12.4 ms** | 0.450 |
| **ResNet50 (Champion)** | ~25.6M | **92.00%** | **0.891** | **88.2%** | 24.1 ms | **0.680** |
| **Vision Transformer (ViT)**| ~1.5M | **93.15%** | **0.912** | **91.5%** | 38.5 ms | **0.742** |

---

## 🖼️ Publication Figures

The repository includes 300 DPI publication-quality figures located in [`plots/`](plots/):
- **Figure 1:** `plots/fig1_gradcam_fidelity_by_morphology.png` — Grad-CAM fidelity across 9 defect morphologies.
- **Figure 2:** `plots/fig2_multi_architecture_tradeoff.png` — Multi-architecture trade-off matrix.
- **Figure 3:** `plots/fig3_confidence_vs_fidelity_correlation.png` — Prediction confidence vs. explanation IoU degradation.
- **Figure 4:** `plots/fig4_disaggregated_latency_breakdown.png` — Disaggregated latency breakdown (CPU vs GPU).

---

## 🛠️ Evaluation Scripts

Run any of the evaluation modules:

```bash
# 1. Quantitative Grad-CAM Explanation Fidelity Evaluation
python evaluate_gradcam_fidelity.py

# 2. Disaggregated Pipeline Latency & Macro-F1 Evaluation
python evaluate_comprehensive_metrics.py

# 3. Vision Transformer (ViT) Benchmark
python evaluate_vit.py

# 4. Generate 300 DPI Publication Plots
python generate_paper_plots.py
```

---

## ⚡ Running the Web Dashboard

Launch the Streamlit cleanroom inspection dashboard locally:

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.
