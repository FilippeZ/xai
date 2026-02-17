"""
XAI Dashboard — Explainable Artificial Intelligence Interactive Platform
=========================================================================
A premium Streamlit application for exploring LIME, SHAP, Grad-CAM,
counterfactual explanations, and simulatability experiments.

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
    page_title="XAI — Explainable AI Dashboard",
    page_icon="🧠",
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
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 XAI Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Tabular XAI", "🖼️ Image XAI", "🔄 Counterfactuals", "🧪 Simulatability Lab"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Explainable AI Thesis Project")
    st.caption("Philippos-Paraskevas Zygouris")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <h1>🧠&nbsp; Explainable Artificial Intelligence</h1>
        <p>
            Transforming AI from opaque <strong>"black boxes"</strong> to transparent,
            human-centric solutions.  This dashboard demonstrates <strong>LIME</strong>,
            <strong>SHAP</strong>, <strong>Grad-CAM</strong>, and <strong>Counterfactual</strong>
            explanations across tabular, image, and text data — powered by the
            <em>Omni XAI</em> architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("### 🏗️ System Architecture")
    from src.visualization.dashboard import render_architecture_diagram
    st.markdown(render_architecture_diagram())

    # Method cards
    st.markdown("### 🔬 Explanation Methods")
    from src.visualization.dashboard import get_method_info
    methods = get_method_info()
    cols = st.columns(2)
    for i, (name, info) in enumerate(methods.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="method-card">
                <h4>{name}</h4>
                <span class="badge">{info['type']}</span>
                <p style="font-size:0.9rem; opacity:0.85;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("### 📈 Project Statistics")
    cols = st.columns(4)
    stats = [("4", "XAI Methods"), ("3", "Data Modalities"), ("2", "Simulation Types"), ("6", "Evaluation Metrics")]
    for col, (val, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


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
