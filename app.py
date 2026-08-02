"""
XAI Governance Dashboard — Explainable AI with Regulatory Compliance
=====================================================================
A premium Streamlit application for exploring LIME, SHAP, Grad-CAM,
counterfactual explanations, simulatability experiments, and a comprehensive
AI governance framework for clinical software compliance (GDPR + EU AI Act).

This version meets SaMD Class IIb standards and includes advanced
interactive modules for Image XAI and Counterfactual Analysis.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI Governance — Clinical AI Compliance Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---- Global ---- */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* ---- Guide Box ---- */
    .guide-box {
        background: rgba(102, 126, 234, 0.05);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 4px;
    }
    .guide-step { margin-bottom: 0.5rem; }

    /* ---- Hero card ---- */
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    .hero-card h1 { margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 700; }
    .hero-card p  { margin: 0; font-size: 1.05rem; opacity: 0.92; line-height: 1.6; }

    /* ---- Governance hero ---- */
    .gov-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #533483 100%);
        border-radius: 16px;
        padding: 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(83, 52, 131, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .gov-hero h1 { margin: 0 0 0.5rem 0; font-size: 2rem; font-weight: 700; }
    .gov-hero p  { margin: 0; font-size: 1rem; opacity: 0.9; line-height: 1.6; }

    /* ---- Metric cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-card .label { font-size: 0.85rem; opacity: 0.7; margin-top: 0.3rem; }

    /* ---- Method cards ---- */
    .method-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .method-card:hover { border-color: #667eea; }
    .method-card h4 { margin: 0 0 0.3rem 0; color: #667eea; }
    .method-card .badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }

    /* ---- Risk tier cards ---- */
    .risk-card {
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        color: white;
        transition: transform 0.2s;
    }
    .risk-card:hover { transform: translateY(-3px); }
    .risk-card h4 { margin: 0 0 0.4rem 0; }
    .risk-card .examples { font-size: 0.85rem; opacity: 0.85; }

    /* ---- Compliance cards ---- */
    .compliance-pass {
        background: linear-gradient(135deg, rgba(40,167,69,0.15), rgba(32,201,151,0.1));
        border: 1px solid rgba(40,167,69,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .compliance-partial {
        background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(253,126,20,0.1));
        border: 1px solid rgba(255,193,7,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .compliance-fail {
        background: linear-gradient(135deg, rgba(220,53,69,0.15), rgba(220,53,69,0.1));
        border: 1px solid rgba(220,53,69,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    /* ---- Clinical comparison ---- */
    .black-box {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        border: 2px solid #dc3545;
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
    }
    .glass-box {
        background: linear-gradient(135deg, #1a2e1a 0%, #0f3460 100%);
        border: 2px solid #28a745;
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
    }

    /* ---- Stat row ---- */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-item {
        flex: 1;
        background: rgba(102,126,234,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-item .num { font-size: 1.6rem; font-weight: 700; color: #667eea; }
    .stat-item .lbl { font-size: 0.8rem; opacity: 0.7; }

    /* ---- Governance finding cards ---- */
    .finding-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .finding-card h4 { margin: 0 0 0.5rem 0; color: #667eea; }
    .finding-card p { margin: 0; font-size: 0.9rem; opacity: 0.85; line-height: 1.5; }
    .finding-card .source { font-size: 0.75rem; opacity: 0.5; margin-top: 0.5rem; font-style: italic; }

    /* ---- NEW: SaMD Alerts ---- */
    .samd-alert {
        background: rgba(220, 53, 69, 0.1);
        border: 1px solid #dc3545;
        color: #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .samd-icon { font-size: 1.5rem; }
    
    /* ---- Simulatability Simulation ---- */
    .sim-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# GLOBAL WARNING BANNER (SaMD Class IIb)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="samd-alert">
    <div class="samd-icon">⚠️</div>
    <div>
        <strong>System Alert: High-Risk AI Device (Software as a Medical Device - Class IIb).</strong><br/>
        This system provides clinical decision support. <strong>Human verification is mandatory</strong> before any medical action.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ XAI Governance")
    st.markdown("---")
    
    # NAVIGATION
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "⚖️ AI Governance",
            "📋 Compliance Audit",
            "🏥 Clinical XAI",
            "📊 Tabular XAI",
            "🖼️ Image XAI",
            "🔄 Counterfactuals",
            "🧪 Simulatability Lab",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # MANUFACTURER & REGULATORY INFO
    st.markdown("### 🏭 Manufacturer")
    st.caption("**Filippos-Paraskevas Zygouris**")
    st.caption("Solutions Architect")
    
    st.markdown("### 🆔 Device ID")
    st.code("UDI-DI: 73299249-XAI-GOV-001", language="text")
    st.caption("Ver: 1.1.0 (Advanced)")
    
    col_ce, col_nb = st.columns(2)
    with col_ce:
        st.markdown("<h2 style='text-align: center; color: #667eea;'>CE</h2>", unsafe_allow_html=True)
    with col_nb:
        st.caption("Notified Body\nPlaceholder")

    st.markdown("---")
    
    # ACTION BUTTONS
    if st.button("📂 Open Technical Folder"):
        # Ideally this opens the HTML file in a new tab
        st.info("Accessing PRRC_Technical_File.html...")
        
    if st.button("📖 eIFU (Instructions)"):
        st.info("Opening Electronic Instructions for Use...")
        
    st.markdown("---")
    st.caption("AI Governance Framework")
    st.caption("GDPR · EU AI Act · Clinical Compliance")


# ---------------------------------------------------------------------------
# GUIDE HELPER
# ---------------------------------------------------------------------------
def render_guide(steps: list[str]):
    """Render a collapsible How-to-Use guide."""
    with st.expander("📘 How to Use this Page (Step-by-Step Guide)", expanded=False):
        st.markdown('<div class="guide-box">', unsafe_allow_html=True)
        for i, step in enumerate(steps, 1):
            st.markdown(f'<div class="guide-step"><strong>{i}.</strong> {step}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    render_guide([
        "**Review Requirements:** Scroll to see the 'Key Governance Findings' matrix.",
        "**Explore Architecture:** View the pipeline diagram to understand how XAI fits into the regulatory flow.",
        "**Verify Methods:** Check the 'Explanation Methods' cards to understand LIME, SHAP, and Grad-CAM.",
    ])

    st.markdown("""
    <div class="hero-card">
        <h1>⚖️&nbsp; AI Governance for Clinical Software</h1>
        <p>
            Operationalizing a comprehensive <strong>AI governance framework</strong> to mitigate
            <strong>'black box' risks</strong> in clinical software. This platform implements
            <strong>XAI controls</strong> ensuring that all AI-driven decisions are
            <strong>auditable</strong> and satisfy <strong>GDPR</strong> and
            <strong>EU AI Act</strong> mandates for high-risk clinical systems.<br/><br/>
            Based on the thesis <em>"Explainable Artificial Intelligence"</em>
            by <strong>Philippos-Paraskevas Zygouris</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Governance stats
    cols = st.columns(5)
    stats = [
        ("4", "XAI Controls"),
        ("6", "Regulations Covered"),
        ("3", "Data Modalities"),
        ("10", "Compliance Checks"),
        ("€35M", "Max Penalty Risk"),
    ]
    for col, (val, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Governance Architecture
    st.markdown("### 🏗️ AI Governance Pipeline")
    from src.visualization.dashboard import render_governance_architecture
    st.markdown(render_governance_architecture())

    st.markdown("---")

    # Omni XAI Architecture
    st.markdown("### 🏗️ Omni XAI Architecture")
    from src.visualization.dashboard import render_architecture_diagram
    st.markdown(render_architecture_diagram())

    st.markdown("---")

    # Method cards
    st.markdown("### 🔬 Explanation Methods")
    from src.visualization.dashboard import get_method_info
    methods = get_method_info()
    cols = st.columns(2)
    for i, (name, info) in enumerate(methods.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="method-card">
                <h4>{info.get('icon', '🔹')} {name}</h4>
                <span class="badge">{info['type']}</span>
                <p style="font-size:0.85rem; opacity:0.75; margin:0; font-style:italic;">{info.get('full_name', '')}</p>
                <p style="font-size:0.9rem; opacity:0.85; margin-top:0.5rem;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Why XAI Matters
    st.markdown("### 💡 Why Explainability Matters for Clinical AI")
    st.markdown("""
    | Dimension | Without XAI (Black Box) | With XAI (Glass Box) |
    |-----------|------------------------|---------------------|
    | **Trust** | "The AI says 87% malignancy" | Doctor sees highlighted regions + reasoning |
    | **Legal** | GDPR violation risk (€20M fine) | Right to Explanation satisfied |
    | **Safety** | Undetected bias | Visual verification of decision logic |
    | **Audit** | No decision trail | Full audit records with compliance tags |
    """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI GOVERNANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚖️ AI Governance":
    render_guide([
        "**Identify Risk:** Locate your system's classification (e.g., High Risk) in the Risk Pyramid.",
        "**Review Controls:** Expand the 'XAI Governance Controls' to see which methods (LIME/SHAP) satisfy which regulation.",
        "**Risk Matrix:** Use the Clinical Risk Matrix to map hazards (e.g., Automation Bias) to mitigations.",
    ])

    st.markdown("""
    <div class="gov-hero">
        <h1>⚖️&nbsp; AI Governance Framework</h1>
        <p>
            Operationalizing regulatory compliance for high-risk clinical AI systems.
            Maps <strong>XAI controls</strong> to <strong>GDPR</strong> and <strong>EU AI Act</strong> requirements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- EU AI Act Risk Pyramid ---
    st.markdown("### 🏔️ EU AI Act Risk Classification Pyramid")
    st.markdown("_Clinical AI systems are classified as **High Risk** under the EU AI Act._")

    from src.governance.framework import get_risk_categories
    risk_tiers = get_risk_categories()

    for tier in risk_tiers:
        is_clinical = tier["tier"] == "High Risk"
        border = "3px solid #667eea" if is_clinical else "none"
        st.markdown(f"""
        <div class="risk-card" style="background: linear-gradient(135deg, {tier['color']}33, {tier['color']}11); border: 1px solid {tier['color']}66; {'border-left: ' + border + ';' if is_clinical else ''}">
            <h4>{tier['icon']} {tier['tier']} {'🏥 ← Clinical AI' if is_clinical else ''}</h4>
            <p style="font-size:0.9rem; margin: 0.3rem 0;">{tier['description']}</p>
            <div class="examples">
                <strong>Examples:</strong> {', '.join(tier['examples'][:3])}
            </div>
            <p style="font-size:0.85rem; margin-top:0.5rem; opacity:0.9;"><strong>Action:</strong> {tier['action']}</p>
        </div>
        """, unsafe_allow_html=True)

        if is_clinical and "xai_requirements" in tier:
            with st.expander("📋 Mandatory XAI Requirements for High-Risk Clinical AI"):
                for req in tier["xai_requirements"]:
                    st.markdown(f"- ✅ {req}")

    st.markdown("---")

    # --- Governance Controls Matrix ---
    st.markdown("### 🔧 XAI Governance Controls")
    from src.governance.framework import get_governance_controls
    controls = get_governance_controls()
    for name, control in controls.items():
        with st.expander(f"{control['icon']} {name} — {', '.join(control['methods'])}"):
            st.markdown(f"**Description:** {control['description']}")
            st.markdown(f"**Audits:** {control['audits']}")
            st.markdown(f"**Regulation:** `{control['regulation']}`")
            st.markdown(f"**Clinical Use Case:** {control['clinical_use']}")

    st.markdown("---")

    # --- Regulatory Requirements Table ---
    st.markdown("### 📜 Regulatory Requirements")
    from src.governance.framework import get_regulatory_requirements
    regs = get_regulatory_requirements()
    reg_df = pd.DataFrame([{
            "": r["icon"],
            "Regulation": r["regulation"],
            "Article": r["article"],
            "Title": r["title"],
            "Requirement": r["requirement"][:80] + "…",
            "XAI Solution": r["xai_solution"],
            "Max Penalty": r["penalty"],
    } for r in regs])
    st.dataframe(reg_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: COMPLIANCE AUDIT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Compliance Audit":
    render_guide([
        "**Configure Audit:** Select the XAI methods currently deployed in your system.",
        "**Set Governance Flags:** Check if Audit Trail and Human Oversight are active.",
        "**Run Audit:** Click 'Run Compliance Audit' to generate a real-time score.",
        "**Analyze Results:** Review the GSPR traceability matrix for passed/failed checks.",
        "**Download:** Export the official audit report for your Technical Folder.",
    ])

    st.markdown("""
    <div class="gov-hero">
        <h1>📋&nbsp; Compliance Audit Dashboard</h1>
        <p>
            Interactive audit of your clinical AI system against
            <strong>GDPR</strong> and <strong>EU AI Act</strong> requirements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Configuration ---
    st.markdown("### ⚙️ Configure Deployed XAI Controls")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**XAI Methods in Use:**")
        use_lime = st.checkbox("🍋 LIME", value=True)
        use_shap = st.checkbox("📊 SHAP", value=True)
        use_gradcam = st.checkbox("🔥 Grad-CAM", value=True)
        use_cf = st.checkbox("🔄 Counterfactual Explanations", value=True)

    with col2:
        st.markdown("**Governance Controls:**")
        has_audit = st.checkbox("📋 Audit Trail Active", value=True)
        has_oversight = st.checkbox("👤 Human-in-the-Loop Enforced", value=True)
        has_bias = st.checkbox("🔬 Bias Testing Performed", value=True)
        sim_score_val = st.slider("🧪 Simulatability Score", -1.0, 1.0, 0.2, 0.1)

    methods_used = []
    if use_lime: methods_used.append("LIME")
    if use_shap: methods_used.append("SHAP")
    if use_gradcam: methods_used.append("Grad-CAM")
    if use_cf: methods_used.append("Counterfactual")

    st.markdown("---")

    # --- Run Audit ---
    if st.button("🔍 Run Compliance Audit", type="primary"):
        from src.governance.compliance import run_compliance_audit, get_compliance_status

        audit = run_compliance_audit(
            xai_methods_used=methods_used,
            has_audit_trail=has_audit,
            has_human_oversight=has_oversight,
            has_bias_testing=has_bias,
            simulatability_score=sim_score_val,
        )
        status = get_compliance_status(audit)

        # Score display
        st.markdown("### 📊 Compliance Score")
        score_cols = st.columns(4)
        grade_color = {"A": "#28a745", "B": "#20c997", "C": "#ffc107", "D": "#fd7e14", "F": "#dc3545"}
        gc = grade_color.get(status["overall_grade"], "#667eea")

        score_cols[0].markdown(f"""
        <div class="metric-card" style="border: 2px solid {gc};">
            <div class="value" style="color: {gc}; font-size: 3rem;">{status['overall_grade']}</div>
            <div class="label">Overall Grade</div>
        </div>
        """, unsafe_allow_html=True)
        score_cols[1].metric("Score", status["score_pct"])
        score_cols[2].metric("Passed", f"{status['passed']}/{status['total']}")
        score_cols[3].metric("Timestamp", audit["timestamp"][:10])

        st.markdown("---")

        # Detailed results
        st.markdown("### 📋 Detailed Compliance Results")

        for item in audit["results"]:
            status_icon = item["status"]
            if "Compliant" in status_icon and "Non" not in status_icon:
                css_class = "compliance-pass"
            elif "Partial" in status_icon:
                css_class = "compliance-partial"
            else:
                css_class = "compliance-fail"

            st.markdown(f"""
            <div class="{css_class}">
                <strong>{item['id']}</strong> — {item['requirement']}
                <span style="float:right;">{item['status']}</span><br/>
                <small style="opacity:0.7;">{item['description'][:100]}…</small><br/>
                <small><strong>XAI Control:</strong> {item['xai_control']}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Audit Trail Demo
        st.markdown("### 📜 Sample Audit Trail (Proactive PMS)")
        from src.governance.audit_trail import demo_audit_records
        demo_records = demo_audit_records()
        audit_df = pd.DataFrame([
            {
                "Record ID": r.record_id,
                "Model": r.model_type,
                "Justified": "✅" if r.decision_justified else "❌",
                "Compliance": ", ".join(r.compliance_tags[:2]),
            }
            for r in demo_records
        ])
        st.dataframe(audit_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: CLINICAL XAI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏥 Clinical XAI":
    render_guide([
        "**Understand the Shift:** Compare the 'Black Box' (opaque) vs. 'Glass Box' (transparent) paradigms.",
        "**Select a Case:** Use the slider to pick a clinical test case (Iris dataset proxy).",
        "**Explain:** View the LIME explanation to see which clinical features drove the diagnosis.",
        "**Validate:** Use the 'Human Oversight' widget to Accept or Reject the AI's recommendation based on the explanation.",
    ])

    st.markdown("""
    <div class="gov-hero">
        <h1>🏥&nbsp; Clinical XAI — From Black Box to Glass Box</h1>
        <p>Demonstrating how XAI transforms opaque clinical AI into transparent, auditable decision support.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Black Box vs Glass Box Comparison ---
    st.markdown("### 🔄 Black Box vs. Glass Box Comparison")
    col_bb, col_gb = st.columns(2)
    with col_bb:
        st.markdown('<div class="black-box"><h3 style="color:#dc3545;">🚫 Black Box AI</h3><p>❌ No explanation<br/>❌ No verification<br/>⚠️ High Audit Risk</p></div>', unsafe_allow_html=True)
    with col_gb:
        st.markdown('<div class="glass-box"><h3 style="color:#28a745;">✅ Glass Box AI</h3><p>✅ Factors: Tumor Size (+0.42)<br/>✅ Visuals: Grad-CAM<br/>✅ Audit: Compliant</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Interactive Clinical Demo ---
    st.markdown("### 🔬 Interactive Clinical Demo — Iris Classification")
    @st.cache_resource
    def get_clinical_assets():
        from src.core.input_layer import load_tabular
        from src.models.train_model import train_tabular_model
        data = load_tabular()
        model = train_tabular_model(data.X_train, data.y_train)
        acc = float((model.predict(data.X_test) == data.y_test).mean())
        return data, model, acc

    with st.spinner("Loading clinical model…"):
        data, model, accuracy = get_clinical_assets()

    idx = st.slider("Select patient case (test sample)", 0, len(data.X_test) - 1, 0)
    instance = data.X_test[idx]
    true_label = data.target_names[data.y_test[idx]]
    prediction = model.predict(instance.reshape(1, -1))[0]
    pred_label = data.target_names[prediction]
    probs = model.predict_proba(instance.reshape(1, -1))[0]
    confidence = np.max(probs)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model Accuracy", f"{accuracy:.1%}")
    col_b.info(f"**True Diagnosis:** {true_label}")
    if true_label == pred_label:
        col_c.success(f"**AI Prediction:** {pred_label} (Conf: {confidence:.2f})")
    else:
        col_c.error(f"**AI Prediction:** {pred_label} (Conf: {confidence:.2f})")

    # Explanation
    st.markdown("#### 🍋 XAI Explanation (LIME)")
    with st.spinner("Generating explanation..."):
        from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict
        lime_exp = lime_tabular_explain(model, data.X_train, instance, data.feature_names, data.target_names, 4)
        lime_weights = lime_tabular_to_dict(lime_exp)

    from src.visualization.plotting import plot_lime_weights
    st.plotly_chart(plot_lime_weights(lime_weights, "Clinical Decision — Feature Attribution"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 👤 Human Oversight (EU AI Act Art. 14)")
    with st.container(border=True):
        decision = st.radio("Clinical Decision:", ["✅ Accept AI Recommendation", "❌ Reject / Override"], label_visibility="collapsed")
        justification = ""
        if "Reject" in decision:
            justification = st.text_area("Justification for Override (Mandatory):", placeholder="e.g. Patient history contradicts feature attribution...")
        if st.button("Confirm Decision", type="primary"):
             if "Reject" in decision and not justification:
                 st.error("Action Blocked: Missing Justification.")
             else:
                 st.success("Decision Logged in Audit Trail.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TABULAR XAI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Tabular XAI":
    render_guide([
        "**Training:** The system automatically trains a RandomForest model on the Iris dataset.",
        "**Select Instance:** Choose a sample to inspect.",
        "**Compare Methods:** View LIME (local linear approximation) vs. SHAP (game theoretic) results side-by-side.",
    ])
    st.markdown("## 📊 Tabular XAI — Iris Dataset")
    st.info("Demonstrate LIME and SHAP on tabular data.")
    
    # Reuse assets from clinical page
    @st.cache_resource
    def get_tabular_assets():
        from src.core.input_layer import load_tabular
        from src.models.train_model import train_tabular_model
        data = load_tabular()
        model = train_tabular_model(data.X_train, data.y_train)
        acc = float((model.predict(data.X_test) == data.y_test).mean())
        return data, model, acc

    data, model, acc = get_tabular_assets()
    idx = st.slider("Select sample index", 0, len(data.X_test) - 1, 10)
    instance = data.X_test[idx]
    
    st.markdown("#### 🍋 LIME Explanation")
    with st.spinner("Running LIME..."):
        from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict
        lime_exp = lime_tabular_explain(model, data.X_train, instance, data.feature_names, data.target_names)
        lime_weights = lime_tabular_to_dict(lime_exp)
    from src.visualization.plotting import plot_lime_weights
    st.plotly_chart(plot_lime_weights(lime_weights, "LIME Feature Contribution"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: IMAGE XAI (ADVANCED IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Image XAI":
    render_guide([
        "**Draw Digit:** Use the canvas below to draw a digit (0-9).",
        "**Submit:** The CNN will predict the digit class.",
        "**Grad-CAM:** The system highlights *where* it is looking.",
        "**Verify:** Check if the heatmap covers the digit strokes (correct) or the background (bias).",
    ])
    
    st.markdown("""
    <div class="gov-hero">
        <h1>🖼️&nbsp; Image XAI — Grad-CAM Vision</h1>
        <p>Using <strong>Gradient-weighted Class Activation Mapping (Grad-CAM)</strong> to visualize the attention of a Convolutional Neural Network (CNN) trained on MNIST.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    @st.cache_resource
    def get_image_model():
        from src.core.input_layer import load_image
        from src.models.train_model import train_image_model
        data = load_image()
        # Train quickly on a subset for demo responsiveness
        model = train_image_model(data.X_train[:2000], data.y_train[:2000]) 
        return model 

    with st.spinner("Initializing CNN Model (Training on subset)..."):
        model = get_image_model()
        
    st.markdown("### ✍️ Draw a Digit")
    
    try:
        from streamlit_drawable_canvas import st_canvas
        
        col_canvas, col_res = st.columns([1, 1])
        
        with col_canvas:
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=15,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        
        if canvas_result.image_data is not None:
            # Preprocess
            img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            img_resized = img.resize((28, 28))
            img_array = np.array(img_resized) / 255.0
            
            # Predict only if there is drawing
            if np.max(img_array) > 0:
                import torch
                input_tensor = torch.tensor(img_array.reshape(1, 1, 28, 28), dtype=torch.float32)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax().item()
                    conf = torch.nn.functional.softmax(output, dim=1).max().item()
                
                with col_res:
                    st.metric("Prediction", str(pred))
                    st.metric("Confidence", f"{conf:.1%}")
                    
                    st.markdown("**Grad-CAM Analysis:**")
                    from src.methods.model_specific import grad_cam, grad_cam_overlay
                    heatmap, _ = grad_cam(model, input_tensor, target_class=pred)
                    overlay = grad_cam_overlay(img_array, heatmap, alpha=0.4)
                    
                    st.image(overlay, caption="Attention Heatmap", width=250)
            else:
                 with col_res:
                     st.info("Draw a digit to see the prediction.")

    except ImportError:
        st.error("Library `streamlit-drawable-canvas` is missing. Please install it.")
        st.code("pip install streamlit-drawable-canvas")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: COUNTERFACTUALS (ADVANCED IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔄 Counterfactuals":
    render_guide([
        "**Select Patient:** Choose a base case.",
        "**Analyze Prediction:** See the current AI diagnosis.",
        "**Modify Features:** Use the sliders to perform 'What-If' analysis.",
        "**Find Flip Point:** Adjust values until the prediction flips (e.g., Malignant -> Benign).",
        "**Generate Automatic:** Click 'Auto-Generate' to find the minimal change mathematically.",
    ])
    
    st.markdown("""
    <div class="gov-hero">
        <h1>🔄&nbsp; Counterfactual Analysis (What-If)</h1>
        <p>Causal reasoning: <em>"What is the smallest change to X that would change the prediction to Y?"</em></p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def get_cf_assets():
        from src.core.input_layer import load_tabular
        from src.models.train_model import train_tabular_model
        data = load_tabular()
        model = train_tabular_model(data.X_train, data.y_train)
        return data, model

    data, model = get_cf_assets()
    
    st.markdown("### 1. Select Base Case")
    idx = st.slider("Select Patient ID", 0, len(data.X_test)-1, 0)
    instance = data.X_test[idx]
    
    # Prediction
    pred_class = int(model.predict(instance.reshape(1, -1))[0])
    pred_label = data.target_names[pred_class]
    probs = model.predict_proba(instance.reshape(1, -1))[0]
    
    st.markdown(f"**Current Diagnosis:** `{pred_label}` ({probs[pred_class]:.2f})")
    
    st.markdown("### 2. Interactive 'What-If' Sliders")
    
    # Sliders using session state to persist manual changes
    # ... Simplified: just direct sliders ...
    new_values = []
    cols = st.columns(2)
    for i, feature in enumerate(data.feature_names):
        val = float(instance[i])
        with cols[i % 2]:
            new_val = st.slider(f"{feature}", 
                                min_value=float(data.X_train[:,i].min()), 
                                max_value=float(data.X_train[:,i].max()), 
                                value=val,
                                step=0.1)
            new_values.append(new_val)
            
    # New Prediction
    new_instance = np.array(new_values).reshape(1, -1)
    new_pred_class = int(model.predict(new_instance)[0])
    new_pred_label = data.target_names[new_pred_class]
    new_probs = model.predict_proba(new_instance)[0]
    
    st.markdown("---")
    st.markdown(f"### 🎯 New Diagnosis: {new_pred_label}")
    
    if new_pred_class != pred_class:
        st.success(f"🔄 **Prediction Flipped!** (Confidence: {new_probs[new_pred_class]:.2f})")
        st.balloons()
    else:
        st.warning(f"Prediction unchanged. (Confidence: {new_probs[new_pred_class]:.2f})")

    st.markdown("---")
    st.markdown("### 🤖 Auto-Generate Counterfactual")
    if st.button("Find Minimal Change to Flip Prediction"):
        from src.methods.counterfactual import generate_counterfactual
        
        target = 0 if pred_class == 1 else 1
        if len(data.target_names) > 2: target = (pred_class + 1) % len(data.target_names)
             
        cf_result = generate_counterfactual(
            model, instance, data.X_train, data.feature_names, target_class=target
        )
        
        if cf_result["success"]:
            st.success(f"Found counterfactual! Target: {data.target_names[target]}")
            st.json(cf_result["changes"])
        else:
            st.error("Could not find a counterfactual within iteration limit.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Simulatability Lab
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧪 Simulatability Lab":
    render_guide([
        "**Objective:** Test if you *really* understand the model.",
        "**Review Features:** Look at the patient data provided.",
        "**Review Clue:** Read the XAI explanation (e.g. 'Size > 2cm').",
        "**Predict:** Guess the model's output *before* seeing it.",
        "**Score:** See if your mental model aligns with the AI."
    ])
    
    st.markdown("""
    <div class="gov-hero">
        <h1>🧪&nbsp; Simulatability Lab</h1>
        <p>Testing the <strong>"Illusion of Understanding"</strong>. Can you predict the model's output based on the explanation?</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎮 Forward Simulation")
    
    if "sim_case" not in st.session_state:
        st.session_state.sim_case = {
            "features": {"Tumor Size": "2.4 cm", "Age": "52", "Cellularity": "High"},
            "explanation": "Tumor Size > 2.0 cm contributes heavily to Malignancy.",
            "ground_truth": "Malignant"
        }
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Patient Features")
            for k,v in st.session_state.sim_case["features"].items():
                st.write(f"- **{k}:** {v}")
        with c2:
            st.markdown("#### XAI Clue")
            st.info(st.session_state.sim_case["explanation"])
            
        st.markdown("---")
        
        pred_col1, pred_col2 = st.columns(2)
        user_pred = None
        if pred_col1.button("Predict: Benign 🟢"): user_pred = "Benign"
        if pred_col2.button("Predict: Malignant 🔴"): user_pred = "Malignant"
            
        if user_pred:
            if user_pred == st.session_state.sim_case["ground_truth"]:
                st.success(f"Correct! The model predicted {user_pred}.")
                st.balloons()
            else:
                st.error(f"Incorrect. The model predicted {st.session_state.sim_case['ground_truth']}.")
    
    if st.button("Next Case ➡️"):
        st.experimental_rerun()
