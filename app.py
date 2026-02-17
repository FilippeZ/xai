"""
XAI Governance Dashboard — Explainable AI with Regulatory Compliance
=====================================================================
A premium Streamlit application for exploring LIME, SHAP, Grad-CAM,
counterfactual explanations, simulatability experiments, and a comprehensive
AI governance framework for clinical software compliance (GDPR + EU AI Act).

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ XAI Governance")
    st.markdown("---")
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
    st.caption("AI Governance Framework")
    st.caption("GDPR · EU AI Act · Clinical Compliance")
    st.markdown("---")
    st.caption("Philippos-Paraskevas Zygouris")
    st.caption("XAI Thesis Project")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <h1>⚖️&nbsp; AI Governance for Clinical Software</h1>
        <p>
            Operationalizing a comprehensive <strong>AI governance framework</strong> to mitigate
            <strong>'black box' risks</strong> in clinical software. This platform implements
            <strong>XAI controls</strong> ensuring that all AI-driven decisions are
            <strong>auditable</strong> and satisfy <strong>GDPR</strong> and
            <strong>EU AI Act</strong> mandates for high-risk clinical systems.<br/><br/>
            Powered by <strong>LIME</strong>, <strong>SHAP</strong>, <strong>Grad-CAM</strong>,
            and <strong>Counterfactual</strong> explanations — transforming opaque
            <em>"black boxes"</em> into transparent <em>"glass boxes"</em>.<br/><br/>
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

    # Key Governance Findings from NotebookLM
    st.markdown("### 🔬 Key Governance Findings")
    st.caption("_Sourced from NotebookLM thesis analysis_")

    from src.visualization.dashboard import get_governance_findings
    findings = get_governance_findings()

    col_l, col_r = st.columns(2)
    finding_items = list(findings.values())
    for i, finding in enumerate(finding_items):
        with col_l if i % 2 == 0 else col_r:
            st.markdown(f"""
            <div class="finding-card">
                <h4>{finding['title']}</h4>
                <p>{finding['finding']}</p>
                <div class="source">📓 {finding['source']}</div>
            </div>
            """, unsafe_allow_html=True)

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

    # Key Thesis Findings
    st.markdown("### 🎓 Key Thesis Findings")
    from src.visualization.dashboard import get_thesis_findings
    th_findings = get_thesis_findings()

    col_l, col_r = st.columns(2)
    with col_l:
        st.warning("**⚠️ The 'Illusion of Understanding'**")
        st.markdown(th_findings["illusion_of_understanding"])

        st.info("**📊 Best for Tabular Data**")
        st.markdown(th_findings["best_tabular"])

        st.success("**🔄 Counterfactual Simulation**")
        st.markdown(th_findings["best_counterfactual"])

    with col_r:
        st.info("**🧬 From Correlation to Causality**")
        st.markdown(th_findings["causability"])

        st.error("**⚖️ GDPR & Legal Compliance**")
        st.markdown(th_findings["gdpr"])

        st.info("**🏥 Medical Applications**")
        st.markdown(th_findings["medical"])

    st.markdown("---")

    # Why XAI Matters
    st.markdown("### 💡 Why Explainability Matters for Clinical AI")
    st.markdown("""
    | Dimension | Without XAI (Black Box) | With XAI (Glass Box) |
    |-----------|------------------------|---------------------|
    | **Trust** | "The AI says 87% malignancy" | Doctor sees highlighted regions + reasoning |
    | **Legal** | GDPR violation risk (€20M fine) | Right to Explanation satisfied |
    | **Action** | User knows *what* but not *why* | User knows *why* and *what to change* |
    | **Science** | Correlation-based decisions | Causal understanding via counterfactuals |
    | **Safety** | Undetected bias and spurious correlations | Bias auditing + visual verification |
    | **Audit** | No decision trail | Full audit records with compliance tags |
    """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI GOVERNANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚖️ AI Governance":
    st.markdown("""
    <div class="gov-hero">
        <h1>⚖️&nbsp; AI Governance Framework</h1>
        <p>
            Operationalizing regulatory compliance for high-risk clinical AI systems.
            This framework maps <strong>XAI controls</strong> to
            <strong>GDPR</strong> and <strong>EU AI Act</strong> requirements,
            ensuring every AI decision is transparent, auditable, and legally defensible.
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

        # Show XAI requirements for high-risk
        if is_clinical and "xai_requirements" in tier:
            with st.expander("📋 Mandatory XAI Requirements for High-Risk Clinical AI"):
                for req in tier["xai_requirements"]:
                    st.markdown(f"- ✅ {req}")

    st.markdown("---")

    # --- Governance Controls Matrix ---
    st.markdown("### 🔧 XAI Governance Controls")
    st.markdown("_Each control maps to specific regulatory requirements._")

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

    reg_df = pd.DataFrame([
        {
            "": r["icon"],
            "Regulation": r["regulation"],
            "Article": r["article"],
            "Title": r["title"],
            "Requirement": r["requirement"][:80] + "…",
            "XAI Solution": r["xai_solution"],
            "Max Penalty": r["penalty"],
        }
        for r in regs
    ])
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Clinical Risk Matrix ---
    st.markdown("### 🏥 Clinical Software Risk Assessment")

    from src.governance.framework import get_clinical_risk_matrix
    risks = get_clinical_risk_matrix()

    risk_df = pd.DataFrame([
        {
            "Risk": r["risk"],
            "Severity": r["severity"],
            "Description": r["description"][:80] + "…",
            "Mitigation (XAI Control)": r["mitigation"],
            "Status": r["status"],
        }
        for r in risks
    ])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # Risk distribution chart
    severity_counts = pd.DataFrame(risks).groupby("severity").size().reset_index(name="count")
    color_map = {"Critical": "#dc3545", "High": "#fd7e14", "Medium": "#ffc107"}
    fig_risk = px.pie(
        severity_counts, values="count", names="severity",
        title="Risk Severity Distribution",
        color="severity",
        color_discrete_map=color_map,
    )
    fig_risk.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_risk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: COMPLIANCE AUDIT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Compliance Audit":
    st.markdown("""
    <div class="gov-hero">
        <h1>📋&nbsp; Compliance Audit Dashboard</h1>
        <p>
            Interactive audit of your clinical AI system against
            <strong>GDPR</strong> and <strong>EU AI Act</strong> requirements.
            Configure your deployed XAI controls and see your compliance score in real-time.
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

    # Build methods list
    methods_used = []
    if use_lime:
        methods_used.append("LIME")
    if use_shap:
        methods_used.append("SHAP")
    if use_gradcam:
        methods_used.append("Grad-CAM")
    if use_cf:
        methods_used.append("Counterfactual")

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
        st.markdown("### 📜 Sample Audit Trail")
        from src.governance.audit_trail import demo_audit_records, format_audit_report

        demo_records = demo_audit_records()
        audit_df = pd.DataFrame([
            {
                "Record ID": r.record_id,
                "Model": r.model_type,
                "Instance": f"#{r.instance_id}",
                "True": r.true_label,
                "Predicted": r.predicted_label,
                "XAI Method": r.xai_method,
                "Justified": "✅" if r.decision_justified else "❌",
                "Compliance": ", ".join(r.compliance_tags[:2]),
            }
            for r in demo_records
        ])
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

        # Downloadable report
        report_text = format_audit_report(demo_records)
        st.download_button(
            "📥 Download Full Audit Report",
            report_text,
            file_name="xai_audit_report.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: CLINICAL XAI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏥 Clinical XAI":
    st.markdown("""
    <div class="gov-hero">
        <h1>🏥&nbsp; Clinical XAI — From Black Box to Glass Box</h1>
        <p>
            Demonstrating how XAI transforms opaque clinical AI into
            transparent, auditable decision support. Based on the thesis
            finding that AI should <strong>not replace</strong> the physician
            but act as a <strong>collaborative tool</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Black Box vs Glass Box Comparison ---
    st.markdown("### 🔄 Black Box vs. Glass Box Comparison")

    col_bb, col_gb = st.columns(2)

    with col_bb:
        st.markdown("""
        <div class="black-box">
            <h3 style="color: #dc3545;">🚫 Black Box AI</h3>
            <p style="font-size: 2rem; text-align: center; margin: 1rem 0; color: #dc3545;">
                <strong>Malignancy: 87%</strong>
            </p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="font-size: 0.9rem; opacity: 0.8;">
                ❌ No explanation of which features drove the prediction<br/>
                ❌ No indication of which image regions were analysed<br/>
                ❌ No way to verify if the model is using relevant patterns<br/>
                ❌ Clinician cannot justify decision to patient<br/>
                ❌ GDPR Art. 22 violation — no right to explanation
            </p>
            <p style="font-size: 0.85rem; color: #dc3545; margin-top: 1rem;">
                ⚠️ Decision dilemma: Accept blindly or reject without cause?
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_gb:
        st.markdown("""
        <div class="glass-box">
            <h3 style="color: #28a745;">✅ Glass Box AI (with XAI)</h3>
            <p style="font-size: 2rem; text-align: center; margin: 1rem 0; color: #28a745;">
                <strong>Malignancy: 87%</strong>
            </p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="font-size: 0.9rem; opacity: 0.8;">
                ✅ <strong>LIME/SHAP:</strong> Tumor size (+0.42), margins (+0.35) drove prediction<br/>
                ✅ <strong>Grad-CAM:</strong> CNN focused on lesion area, not artifacts<br/>
                ✅ <strong>Counterfactual:</strong> "If margins were regular, probability drops to 23%"<br/>
                ✅ Clinician can validate reasoning and explain to patient<br/>
                ✅ Full audit trail with compliance tags
            </p>
            <p style="font-size: 0.85rem; color: #28a745; margin-top: 1rem;">
                ✅ Doctor makes informed final decision with AI as collaborative tool
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Human-AI Collaboration Flow ---
    st.markdown("### 👥 Human-AI Collaboration in Clinical Decision-Making")
    st.markdown("""
    | Stage | AI Role | Human Role |
    |-------|---------|------------|
    | **1. Data Processing** | Analyse patient data, imaging, lab results | Review data quality and completeness |
    | **2. Pattern Recognition** | Identify patterns across thousands of cases | Apply clinical intuition and experience |
    | **3. Prediction** | Generate probability scores with confidence | Evaluate prediction in patient context |
    | **4. Explanation** | Provide LIME/SHAP feature attribution | Verify explanations match medical knowledge |
    | **5. Visual Evidence** | Grad-CAM highlights regions of interest | Confirm AI focuses on relevant anatomy |
    | **6. What-If Analysis** | Generate counterfactual scenarios | Use scenarios for treatment planning |
    | **7. Final Decision** | — | **Clinician makes the final call** ✅ |
    | **8. Audit** | Log decision with XAI compliance tags | Sign off on decision record |
    """)

    st.markdown("---")

    # --- Interactive Clinical Demo ---
    st.markdown("### 🔬 Interactive Clinical Demo — Iris Classification")
    st.markdown("_Using Iris dataset as a proxy for clinical feature-based diagnosis._")

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
    pred_label = data.target_names[model.predict(instance.reshape(1, -1))[0]]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model Accuracy", f"{accuracy:.1%}")
    col_b.info(f"**True Diagnosis:** {true_label}")
    if true_label == pred_label:
        col_c.success(f"**AI Prediction:** {pred_label}")
    else:
        col_c.error(f"**AI Prediction:** {pred_label}")

    # Generate explanation
    st.markdown("#### 🍋 XAI Explanation (LIME)")
    with st.spinner("Generating explanation for audit…"):
        from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict
        lime_exp = lime_tabular_explain(model, data.X_train, instance,
                                        data.feature_names, data.target_names, 4)
        lime_weights = lime_tabular_to_dict(lime_exp)

    from src.visualization.plotting import plot_lime_weights
    st.plotly_chart(plot_lime_weights(lime_weights, "Clinical Decision — Feature Attribution"),
                    use_container_width=True)

    # Generate audit record
    st.markdown("#### 📋 Audit Record for This Decision")
    from src.governance.audit_trail import generate_audit_record

    record = generate_audit_record(
        model_type="RandomForest (Clinical Proxy)",
        model_accuracy=accuracy,
        instance_id=idx,
        true_label=true_label,
        predicted_label=pred_label,
        xai_method="LIME",
        feature_attributions=lime_weights,
        confidence=float(model.predict_proba(instance.reshape(1, -1)).max()),
    )

    audit_cols = st.columns(2)
    with audit_cols[0]:
        st.markdown(f"""
        - **Record ID:** `{record.record_id}`
        - **Timestamp:** {record.timestamp}
        - **Model:** {record.model_type}
        - **XAI Method:** {record.xai_method}
        - **Decision Justified:** {'✅ Yes' if record.decision_justified else '❌ No'}
        """)
    with audit_cols[1]:
        st.markdown(f"""
        - **Explanation:** {record.explanation_summary}
        - **Compliance Tags:** {', '.join(record.compliance_tags)}
        - **Reviewer:** {record.human_reviewer}
        """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TABULAR XAI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Tabular XAI":
    st.markdown("## 📊 Tabular XAI — Iris Dataset")
    st.markdown("Train a **RandomForest** on the Iris dataset, then explore **LIME** and **SHAP** explanations.")

    # --- Load & Train ---
    @st.cache_resource
    def get_tabular_assets():
        from src.core.input_layer import load_tabular
        from src.models.train_model import train_tabular_model
        data = load_tabular()
        model = train_tabular_model(data.X_train, data.y_train)
        acc = float((model.predict(data.X_test) == data.y_test).mean())
        return data, model, acc

    with st.spinner("Training RandomForest on Iris…"):
        data, model, accuracy = get_tabular_assets()

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{accuracy:.1%}")
    c2.metric("Training Samples", data.X_train.shape[0])
    c3.metric("Features", len(data.feature_names))

    st.markdown("---")

    # Instance selector
    st.markdown("### 🔍 Select an Instance to Explain")
    idx = st.slider("Test sample index", 0, len(data.X_test) - 1, 0)
    instance = data.X_test[idx]
    true_label = data.target_names[data.y_test[idx]]
    pred_label = data.target_names[model.predict(instance.reshape(1, -1))[0]]

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**True Label:** {true_label}")
    with col_b:
        st.success(f"**Predicted:** {pred_label}") if true_label == pred_label else st.error(f"**Predicted:** {pred_label}")

    # Feature values table
    df_instance = pd.DataFrame([instance], columns=data.feature_names)
    st.dataframe(df_instance.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")

    # LIME
    st.markdown("### 🍋 LIME Explanation")
    with st.spinner("Computing LIME…"):
        from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict
        lime_exp = lime_tabular_explain(model, data.X_train, instance, data.feature_names,
                                        data.target_names, num_features=4)
        lime_weights = lime_tabular_to_dict(lime_exp)

    from src.visualization.plotting import plot_lime_weights
    st.plotly_chart(plot_lime_weights(lime_weights, "LIME — Local Feature Importance"),
                    use_container_width=True)

    with st.expander("📋 Raw LIME weights"):
        st.json(lime_weights)

    st.markdown("---")

    # SHAP
    st.markdown("### 📊 SHAP Explanation")
    with st.spinner("Computing SHAP values…"):
        from src.methods.model_agnostic import shap_tabular_explain
        shap_values, shap_explainer = shap_tabular_explain(model, data.X_test, data.feature_names)

    from src.visualization.plotting import plot_shap_bar
    st.plotly_chart(plot_shap_bar(shap_values, data.feature_names, title="SHAP — Global Feature Importance"),
                    use_container_width=True)

    # SHAP bee-swarm via streamlit (matplotlib fallback)
    with st.expander("🐝 SHAP Beeswarm Plot"):
        import shap, matplotlib.pyplot as plt
        fig_bee, ax_bee = plt.subplots(figsize=(10, 4))
        shap.summary_plot(shap_values.values if shap_values.values.ndim == 2 else shap_values.values[:,:,0],
                          data.X_test, feature_names=data.feature_names, show=False)
        st.pyplot(fig_bee)
        plt.close(fig_bee)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: IMAGE XAI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Image XAI":
    st.markdown("## 🖼️ Image XAI — MNIST + Grad-CAM")
    st.markdown("Train a **CNN** on MNIST digits and visualize **Grad-CAM** attention heatmaps.")

    @st.cache_resource
    def get_image_assets():
        from src.core.input_layer import load_image
        from src.models.train_model import train_image_model
        data = load_image(n_samples=2000)
        model = train_image_model(data.X_train, data.y_train, epochs=3)
        return data, model

    with st.spinner("Loading MNIST and training CNN (this may take a moment)…"):
        img_data, cnn_model = get_image_assets()

    st.success("✅ CNN trained successfully!")

    # Accuracy
    import torch
    with torch.no_grad():
        X_t = torch.tensor(img_data.X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
        preds_t = cnn_model(X_t).argmax(dim=1).numpy()
    acc_img = float((preds_t == img_data.y_test).mean())
    st.metric("Test Accuracy", f"{acc_img:.1%}")

    st.markdown("---")
    st.markdown("### 🔍 Select a Digit to Explain")
    img_idx = st.slider("Test image index", 0, len(img_data.X_test) - 1, 0)
    img_flat = img_data.X_test[img_idx]
    img_2d = img_flat.reshape(28, 28)
    true_digit = img_data.y_test[img_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        import matplotlib.pyplot as plt
        fig_orig, ax_orig = plt.subplots(figsize=(3, 3))
        ax_orig.imshow(img_2d, cmap="gray")
        ax_orig.set_title(f"True Label: {true_digit}", fontsize=12)
        ax_orig.axis("off")
        st.pyplot(fig_orig)
        plt.close(fig_orig)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        with st.spinner("Computing Grad-CAM…"):
            from src.methods.model_specific import grad_cam, grad_cam_overlay
            input_tensor = torch.tensor(img_flat.reshape(1, 1, 28, 28), dtype=torch.float32)
            heatmap, pred_class = grad_cam(cnn_model, input_tensor)
            overlay = grad_cam_overlay(img_2d, heatmap, alpha=0.5)

        fig_cam, ax_cam = plt.subplots(figsize=(3, 3))
        ax_cam.imshow(overlay)
        ax_cam.set_title(f"Pred: {pred_class}", fontsize=12)
        ax_cam.axis("off")
        st.pyplot(fig_cam)
        plt.close(fig_cam)

    st.markdown("---")
    st.markdown("### 📖 How Grad-CAM Works")
    st.markdown("""
    1. **Forward pass** through the CNN
    2. **Compute gradients** of the target class score w.r.t. the last conv layer
    3. **Global average pool** the gradients → per-channel weights αₖ
    4. **Weighted combination** of feature maps, passed through ReLU
    5. **Result**: a heatmap highlighting which regions the CNN focuses on
    """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: COUNTERFACTUALS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔄 Counterfactuals":
    st.markdown("## 🔄 Counterfactual Explanations")
    st.markdown("Ask *\"What minimal change would flip the prediction?\"* for Iris samples.")

    @st.cache_resource
    def get_cf_assets():
        from src.core.input_layer import load_tabular
        from src.models.train_model import train_tabular_model
        data = load_tabular()
        model = train_tabular_model(data.X_train, data.y_train)
        return data, model

    data, model = get_cf_assets()

    idx = st.slider("Test sample", 0, len(data.X_test) - 1, 5)
    instance = data.X_test[idx]
    current_pred = data.target_names[model.predict(instance.reshape(1, -1))[0]]
    st.info(f"**Current prediction:** {current_pred}")

    target_cls = st.selectbox("Target class (flip to)", range(len(data.target_names)),
                              format_func=lambda i: data.target_names[i])

    if st.button("🔄 Generate Counterfactual", type="primary"):
        with st.spinner("Searching for counterfactual…"):
            from src.methods.counterfactual import generate_counterfactual, counterfactual_proximity
            result = generate_counterfactual(model, instance, data.X_train,
                                             data.feature_names, target_class=target_cls)
            result["proximity"] = counterfactual_proximity(result["original"], result["counterfactual"])

        if result["success"]:
            st.success(f"✅ Prediction flipped to **{data.target_names[result['new_pred']]}** in {result['n_steps']} steps!")
        else:
            st.warning(f"⚠️ Could not flip prediction in {result['n_steps']} steps. Current pred: {data.target_names[result['new_pred']]}")

        # Changes table
        st.markdown("### Feature Changes")
        if result["changes"]:
            changes_df = pd.DataFrame([
                {"Feature": f, "Original": v["from"], "Counterfactual": v["to"],
                 "Δ": round(v["to"] - v["from"], 4)}
                for f, v in result["changes"].items()
            ])
            st.dataframe(changes_df, use_container_width=True)

            from src.visualization.plotting import plot_counterfactual_changes
            st.plotly_chart(plot_counterfactual_changes(result["changes"]), use_container_width=True)

        # Proximity metrics
        st.markdown("### 📏 Proximity Metrics")
        pcols = st.columns(3)
        pcols[0].metric("L₀ (Sparsity)", result["proximity"]["L0"])
        pcols[1].metric("L₁ (Manhattan)", f"{result['proximity']['L1']:.4f}")
        pcols[2].metric("L₂ (Euclidean)", f"{result['proximity']['L2']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATABILITY LAB
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧪 Simulatability Lab":
    st.markdown("## 🧪 Simulatability Lab")
    st.markdown("""
    Test whether AI explanations actually help you predict the model's behaviour.
    This replicates the **Forward Simulation** experiment from the thesis.
    """)

    @st.cache_resource
    def get_sim_assets():
        from src.core.input_layer import load_tabular
        from src.models.train_model import train_tabular_model
        data = load_tabular()
        model = train_tabular_model(data.X_train, data.y_train)
        return data, model

    data, model = get_sim_assets()

    # Session state
    if "sim_trials" not in st.session_state:
        from src.experiments.simulations import create_forward_trials
        session = create_forward_trials(data.X_test, data.y_test, model, n_trials=5)
        st.session_state.sim_trials = session.trials
        st.session_state.sim_phase = "pre"  # 'pre' → 'post'
        st.session_state.sim_idx = 0

    trials = st.session_state.sim_trials
    phase = st.session_state.sim_phase
    trial_idx = st.session_state.sim_idx

    total = len(trials)
    progress = trial_idx / total if total > 0 else 0
    st.progress(progress, text=f"Trial {min(trial_idx + 1, total)} / {total}  —  Phase: {'Pre-Explanation' if phase == 'pre' else 'Post-Explanation'}")

    if trial_idx < total:
        trial = trials[trial_idx]
        instance = data.X_test[trial.instance_idx]

        st.markdown("---")
        st.markdown(f"### Trial {trial_idx + 1}")

        # Show instance
        st.markdown("**Instance features:**")
        df_show = pd.DataFrame([instance], columns=data.feature_names)
        st.dataframe(df_show.style.format("{:.2f}"), use_container_width=True)

        st.markdown(f"**True label:** {data.target_names[trial.true_label]}")

        # Explanation (only in post phase)
        if phase == "post":
            st.markdown("---")
            st.markdown("#### 💡 Explanation (LIME)")
            from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict
            lime_exp = lime_tabular_explain(model, data.X_train, instance,
                                            data.feature_names, data.target_names, 4)
            from src.visualization.plotting import plot_lime_weights
            st.plotly_chart(plot_lime_weights(lime_tabular_to_dict(lime_exp)),
                            use_container_width=True)

        # User prediction
        st.markdown("---")
        user_pred = st.radio(
            "What class will the **model** predict?",
            range(len(data.target_names)),
            format_func=lambda i: data.target_names[i],
            key=f"pred_{phase}_{trial_idx}",
            horizontal=True,
        )

        if st.button("Submit Prediction", type="primary", key=f"btn_{phase}_{trial_idx}"):
            if phase == "pre":
                trials[trial_idx].user_prediction_pre = user_pred
                st.session_state.sim_phase = "post"
            else:
                trials[trial_idx].user_prediction_post = user_pred
                st.session_state.sim_idx += 1
                st.session_state.sim_phase = "pre"
            st.rerun()
    else:
        # Results
        st.markdown("---")
        st.markdown("### 🏆 Experiment Complete!")

        from src.experiments.simulations import SimulationSession
        session = SimulationSession(trials=trials, session_type="forward")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (Pre)", f"{session.accuracy_pre:.0%}")
        c2.metric("Accuracy (Post)", f"{session.accuracy_post:.0%}")
        score = session.simulatability_score
        c3.metric("Simulatability Score", f"{score:+.0%}",
                  delta=f"{'Positive' if score > 0 else 'Negative'}")

        st.markdown("---")
        st.markdown("### 📊 Trial-Level Results")
        results_df = pd.DataFrame([
            {
                "Trial": i + 1,
                "True Label": data.target_names[t.true_label],
                "Model Pred": data.target_names[t.model_prediction],
                "Your Pre": data.target_names[t.user_prediction_pre] if t.user_prediction_pre is not None else "—",
                "Your Post": data.target_names[t.user_prediction_post] if t.user_prediction_post is not None else "—",
                "Pre ✓": "✅" if t.user_prediction_pre == t.model_prediction else "❌",
                "Post ✓": "✅" if t.user_prediction_post == t.model_prediction else "❌",
            }
            for i, t in enumerate(trials)
        ])
        st.dataframe(results_df, use_container_width=True)

        # Interpretation
        if score > 0:
            st.success("🎉 The LIME explanations **helped** you predict the model's behaviour!")
        elif score == 0:
            st.info("🤔 The explanations had **no measurable effect** on your prediction accuracy.")
        else:
            st.warning("⚠️ Interestingly, explanations **reduced** your accuracy — the 'illusion of understanding' effect noted in the thesis.")

        if st.button("🔁 Restart Experiment"):
            for k in ["sim_trials", "sim_phase", "sim_idx"]:
                del st.session_state[k]
            st.rerun()
