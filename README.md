<div align="center">

# ⚖️ XAI Governance — AI Compliance for Clinical Software

**Operationalizing Explainable AI to mitigate 'black box' risks in clinical systems**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GDPR](https://img.shields.io/badge/GDPR-Compliant-28a745)](https://gdpr.eu)
[![EU AI Act](https://img.shields.io/badge/EU_AI_Act-High_Risk-fd7e14)](https://artificialintelligenceact.eu)
[![LIME](https://img.shields.io/badge/XAI-LIME-orange)](https://github.com/marcotcr/lime)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet)](https://github.com/shap/shap)

</div>

---

## 📋 Overview

This project implements a **comprehensive AI governance framework** for clinical software, based on the thesis *"Explainable Artificial Intelligence"* by Philippos-Paraskevas Zygouris. It operationalizes **XAI controls** to ensure that all AI-driven decisions are **auditable** and satisfy **GDPR** and **EU AI Act** mandates for high-risk clinical systems.

### 🎯 The Problem

Modern AI in clinical software operates as a **"black box"**, creating critical risks:
- An algorithm recommending surgery **without clear justification** could endanger lives
- Models trained on specific demographics may **fail on others** without detection
- Without explainability, clinicians face a **decision dilemma**: accept blindly or reject without cause
- Non-compliance carries fines up to **€35 million** (EU AI Act) or **€20 million** (GDPR)

### ✅ The Solution

This platform transforms opaque **"black boxes"** into transparent **"glass boxes"** using XAI controls:

| Control | XAI Method | What It Audits | Regulation |
|---------|-----------|---------------|------------|
| 🔍 **Transparency** | LIME, SHAP | Feature-level decision justification | GDPR Art. 22 |
| 🔬 **Visual Inspection** | Grad-CAM | Spatial attribution for imaging AI | EU AI Act Art. 13 |
| 🔄 **Causal Understanding** | Counterfactual | Actionable "what-if" scenarios | EU AI Act Art. 9 |
| 🧪 **Simulatability** | Forward/CF Sim | Explanation effectiveness validation | EU AI Act Art. 14 |

---

## 🏗️ Architecture

The project combines the **Omni XAI** four-layer architecture with an **AI Governance Pipeline**:

```
Clinical AI System (High-Risk)
    │
    ├── EU AI Act Conformity Assessment
    │
    ├── XAI Controls Layer
    │   ├── LIME / SHAP (Transparency)
    │   ├── Grad-CAM (Visual Inspection)
    │   ├── Counterfactuals (Causal Understanding)
    │   └── Simulatability (Effectiveness Testing)
    │
    ├── Audit Trail (Decision Records + Compliance Tags)
    │
    ├── Human-in-the-Loop (Clinician Review & Override)
    │
    └── Compliant Decision (GDPR Art.22 + EU AI Act)
```

### 📂 Project Structure

```
xai/
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt              # Python dependencies
├── config/
│   ├── model_config.yaml         # ML model parameters
│   └── xai_config.yaml           # Explainer + Governance configuration
├── src/
│   ├── governance/               # 🆕 AI Governance Framework
│   │   ├── framework.py          # Risk classification, controls, regulatory mappings
│   │   ├── compliance.py         # GDPR + EU AI Act compliance checker & scoring
│   │   └── audit_trail.py        # Audit record generation & report formatting
│   ├── core/
│   │   ├── input_layer.py        # Data loading (Iris, MNIST, 20-Newsgroups)
│   │   ├── auto_explainer.py     # Auto method selection by data type
│   │   └── explainers.py         # Unified explainer wrappers
│   ├── models/
│   │   ├── train_model.py        # RandomForest, CNN, TF-IDF+LogReg
│   │   └── load_model.py         # Model persistence utilities
│   ├── methods/
│   │   ├── model_agnostic.py     # LIME (tabular + text), SHAP (Tree + Kernel)
│   │   ├── model_specific.py     # Grad-CAM with PyTorch hooks
│   │   └── counterfactual.py     # Greedy counterfactual search
│   ├── visualization/
│   │   ├── plotting.py           # Plotly charts (LIME bars, SHAP, heatmaps)
│   │   └── dashboard.py          # Architecture diagrams & governance findings
│   └── experiments/
│       ├── simulations.py        # Forward & Counterfactual simulation engine
│       └── evaluation.py         # Simulatability, Fidelity, L₀/L₁/L₂ metrics
└── docs/references/              # Thesis PDF & presentation
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
| 🏠 **Home** | Governance overview, key findings, architecture, XAI methods |
| ⚖️ **AI Governance** | EU AI Act risk pyramid, governance controls, regulatory requirements |
| 📋 **Compliance Audit** | Interactive GDPR/EU AI Act compliance checker with scoring |
| 🏥 **Clinical XAI** | Black box vs glass box comparison, human-AI collaboration, audit records |
| 📊 **Tabular XAI** | LIME & SHAP on the Iris dataset (RandomForest) |
| 🖼️ **Image XAI** | Grad-CAM heatmaps on MNIST digits (CNN) |
| 🔄 **Counterfactuals** | Generate "what-if" scenarios with proximity metrics |
| 🧪 **Simulatability Lab** | Interactive experiment — do explanations actually help? |

---

## ⚖️ Regulatory Compliance

### GDPR (General Data Protection Regulation)

| Article | Requirement | XAI Solution |
|---------|------------|-------------|
| **Art. 22** | Right to Explanation | LIME/SHAP per-instance attribution |
| **Art. 13-14** | Transparency of Processing | SHAP global summaries + documentation |
| **Art. 22.3** | Right to Contest | Counterfactual "what-if" analysis |

### EU AI Act

| Article | Requirement | XAI Solution |
|---------|------------|-------------|
| **Art. 9** | Risk Management System | Counterfactual sensitivity + risk matrix |
| **Art. 13** | Transparency to Users | Grad-CAM + LIME visual/textual explanations |
| **Art. 14** | Human Oversight | Simulatability testing + override mechanisms |
| **Art. 15** | Accuracy & Robustness | L₀/L₁/L₂ metrics + model performance |
| **Art. 17** | Quality Management | Audit trail + compliance dashboard |

**Maximum Penalties:**
- GDPR: up to **€20 million** or **4% of global annual turnover**
- EU AI Act: up to **€35 million** or **7% of global annual turnover**

---

## 🏥 Clinical AI Governance

### Black Box Risks Mitigated

| Risk | Severity | XAI Mitigation |
|------|----------|---------------|
| Opaque Decision-Making | 🔴 Critical | LIME/SHAP feature attribution |
| Undetected Bias | 🟠 High | SHAP cohort analysis |
| Spurious Correlations | 🟠 High | Grad-CAM visual verification |
| Clinician Trust Gap | 🟡 Medium | Human-in-the-loop design |
| Regulatory Non-Compliance | 🔴 Critical | Full audit trail |
| Illusion of Understanding | 🟡 Medium | Simulatability testing |

### Audit Trail

Every AI decision generates an audit record containing:
- Unique Record ID and timestamp
- Model type and accuracy
- True and predicted labels
- XAI method used and feature attributions
- Compliance tags (GDPR Art. 22, EU AI Act Art. 13, etc.)
- Human reviewer sign-off

---

## 🔬 XAI Methods

### LIME (Local Interpretable Model-agnostic Explanations)
Creates perturbed samples around an instance, fits a linear model to approximate the complex model locally, and extracts feature importance weights.

### SHAP (SHapley Additive exPlanations)
Uses Shapley values from game theory to assign each feature a contribution score. Supports `TreeExplainer` for tree-based models and `KernelExplainer` as a universal fallback.

### Grad-CAM (Gradient-weighted Class Activation Mapping)
Computes gradients of the target class score w.r.t. the last convolutional layer, producing a spatial heatmap that highlights the regions a CNN relies on for its prediction.

### Counterfactual Explanations
Answers *"What minimal change would flip the prediction?"* using a greedy feature-perturbation search. Evaluated with L₀ (sparsity), L₁ (Manhattan), and L₂ (Euclidean) proximity metrics.

---

## 🧪 Simulatability Experiments

The **Simulatability Lab** replicates the thesis experiments:

- **Forward Simulation**: Can you predict the model's output? Does seeing LIME explanations improve your accuracy?
- **Counterfactual Simulation**: Can you predict how the model reacts to input perturbations?

**Simulatability Score** = Accuracy<sub>post</sub> − Accuracy<sub>pre</sub>

A positive score indicates that explanations genuinely help users understand the model. A negative score reveals the **"illusion of understanding"** effect.

---

## 🛠️ Technologies

- **ML Frameworks**: scikit-learn, PyTorch
- **XAI Libraries**: LIME, SHAP
- **Visualization**: Plotly, Matplotlib, Streamlit
- **Data**: Iris, MNIST (OpenML), 20-Newsgroups
- **Governance**: Custom compliance framework (GDPR + EU AI Act)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Philippos-Paraskevas Zygouris**

---

<div align="center">
<i>Making clinical AI transparent, auditable, and legally compliant — one explanation at a time.</i>
</div>
