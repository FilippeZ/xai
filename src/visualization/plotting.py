"""
Plotting — Plotly-based visualization helpers for XAI explanations.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_lime_weights(weights: dict, title: str = "LIME Feature Importance") -> go.Figure:
    """Bar chart of LIME feature weights."""
    features = list(weights.keys())
    values = list(weights.values())
    colours = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colours,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Weight",
        yaxis_title="Feature",
        template="plotly_dark",
        height=max(300, len(features) * 40),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_shap_bar(shap_values, feature_names, class_idx: int = 0,
                  title: str = "SHAP Feature Importance") -> go.Figure:
    """Horizontal bar chart of mean |SHAP| values."""
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, class_idx]
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)

    fig = go.Figure(go.Bar(
        x=mean_abs[order],
        y=[feature_names[i] for i in order],
        orientation="h",
        marker_color="#3498db",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value|",
        template="plotly_dark",
        height=max(300, len(feature_names) * 40),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_shap_waterfall(shap_values, instance_idx: int = 0,
                        title: str = "SHAP Waterfall") -> go.Figure:
    """Waterfall chart for a single instance."""
    import shap
    fig = shap.plots.waterfall(shap_values[instance_idx], show=False)
    return fig


def plot_gradcam_heatmap(image: np.ndarray, heatmap: np.ndarray,
                         predicted_class: int) -> go.Figure:
    """Display original image alongside Grad-CAM heatmap."""
    from src.methods.model_specific import grad_cam_overlay

    overlay = grad_cam_overlay(image, heatmap, alpha=0.5)

    fig = go.Figure()
    fig.add_trace(go.Image(z=(overlay * 255).astype(np.uint8)))
    fig.update_layout(
        title=f"Grad-CAM — Predicted Class: {predicted_class}",
        template="plotly_dark",
        height=350,
        width=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_counterfactual_changes(changes: dict, title: str = "Counterfactual Changes") -> go.Figure:
    """Bar chart showing per-feature deltas for a counterfactual."""
    if not changes:
        fig = go.Figure()
        fig.update_layout(title="No feature changes required", template="plotly_dark")
        return fig

    features = list(changes.keys())
    deltas = [changes[f]["to"] - changes[f]["from"] for f in features]
    colours = ["#e67e22" if d > 0 else "#9b59b6" for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=features, orientation="h",
        marker_color=colours,
        text=[f"{d:+.4f}" for d in deltas],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Δ Value",
        template="plotly_dark",
        height=max(300, len(features) * 50),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
