<div align="center">

# 🧠 Explainable Artificial Intelligence (XAI)

**Transforming AI from opaque "black boxes" to transparent, human-centric solutions**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LIME](https://img.shields.io/badge/XAI-LIME-orange)](https://github.com/marcotcr/lime)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet)](https://github.com/shap/shap)

</div>

---

## 📋 Overview

This project implements a **unified Explainable AI platform** based on the thesis *"Explainable Artificial Intelligence"* by Philippos-Paraskevas Zygouris. It provides a comprehensive framework for making machine learning models transparent and interpretable using state-of-the-art XAI methods.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🍋 **LIME** | Local surrogate models for tabular & text explanations |
| 📊 **SHAP** | Game-theoretic feature attribution via Shapley values |
| 🔥 **Grad-CAM** | CNN attention heatmaps for image classification |
| 🔄 **Counterfactuals** | "What-if" analysis with L₀/L₁/L₂ proximity metrics |
| 🧪 **Simulatability Lab** | Interactive experiment measuring explanation effectiveness |
| 🖥️ **Streamlit Dashboard** | Premium interactive UI for exploring all methods |

---

## 🏗️ Architecture

The project follows the **Omni XAI** four-layer architecture:

```
Input Layer → AutoExplainer → Explainers → Visualization
(Data)        (Preprocessing)  (LIME/SHAP/   (Plots &
               & Selection)    Grad-CAM)      Dashboard)
```

### 📂 Project Structure

```
xai/
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt          # Python dependencies
├── config/
│   ├── model_config.yaml     # ML model parameters
│   └── xai_config.yaml       # Explainer configuration
├── src/
│   ├── core/
│   │   ├── input_layer.py    # Data loading (Iris, MNIST, 20-Newsgroups)
│   │   ├── auto_explainer.py # Auto method selection by data type
│   │   └── explainers.py     # Unified explainer wrappers
│   ├── models/
│   │   ├── train_model.py    # RandomForest, CNN, TF-IDF+LogReg
│   │   └── load_model.py     # Model persistence utilities
│   ├── methods/
│   │   ├── model_agnostic.py # LIME (tabular + text), SHAP (Tree + Kernel)
│   │   ├── model_specific.py # Grad-CAM with PyTorch hooks
│   │   └── counterfactual.py # Greedy counterfactual search
│   ├── visualization/
│   │   ├── plotting.py       # Plotly charts (LIME bars, SHAP, heatmaps)
│   │   └── dashboard.py      # Streamlit helpers & architecture diagram
│   └── experiments/
│       ├── simulations.py    # Forward & Counterfactual simulation engine
│       └── evaluation.py     # Simulatability, Fidelity, L₀/L₁/L₂ metrics
├── data/                     # Datasets (auto-downloaded)
├── docs/references/          # Thesis PDF & presentation
└── notebooks/                # Jupyter notebooks for analysis
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/FilippeZ/xai.git
cd xai
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

### 3. Explore

| Page | What it does |
|------|-------------|
| 🏠 **Home** | Overview of the architecture and XAI methods |
| 📊 **Tabular XAI** | LIME & SHAP on the Iris dataset (RandomForest) |
| 🖼️ **Image XAI** | Grad-CAM heatmaps on MNIST digits (CNN) |
| 🔄 **Counterfactuals** | Generate "what-if" scenarios with proximity metrics |
| 🧪 **Simulatability Lab** | Interactive experiment — can explanations help you? |

---

## 🔬 XAI Methods

### LIME (Local Interpretable Model-agnostic Explanations)
Creates perturbed samples around an instance, fits a linear model to approximate the complex model locally, and extracts feature importance weights.

### SHAP (SHapley Additive exPlanations)
Uses Shapley values from game theory to assign each feature a contribution score. Supports `TreeExplainer` for tree-based models and `KernelExplainer` as a universal fallback.

### Grad-CAM (Gradient-weighted Class Activation Mapping)
Computes gradients of the target class score w.r.t. the last convolutional layer, producing a spatial heatmap that highlights the regions a CNN relies on for its prediction.

### Counterfactual Explanations
Answers *"What minimal change would flip the prediction?"* using a greedy feature-perturbation search toward the target class's feature distribution. Evaluated with L₀ (sparsity), L₁ (Manhattan), and L₂ (Euclidean) proximity metrics.

---

## 🧪 Simulatability Experiments

The project includes a novel **Simulatability Lab** that replicates the thesis experiments:

- **Forward Simulation**: Can you predict the model's output for new inputs? Does seeing LIME explanations improve your accuracy?
- **Counterfactual Simulation**: Can you predict how the model reacts to input perturbations?

**Simulatability Score** = Accuracy<sub>post</sub> − Accuracy<sub>pre</sub>

A positive score indicates that explanations genuinely help users understand the model.

---

## 🛠️ Technologies

- **ML Frameworks**: scikit-learn, PyTorch
- **XAI Libraries**: LIME, SHAP
- **Visualization**: Plotly, Matplotlib, Streamlit
- **Data**: Iris, MNIST (OpenML), 20-Newsgroups

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Philippos-Paraskevas Zygouris**

---

<div align="center">
<i>Making AI transparent, one explanation at a time.</i>
</div>
