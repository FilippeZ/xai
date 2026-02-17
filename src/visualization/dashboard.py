"""
Dashboard helpers — Streamlit integration utilities.
"""

from __future__ import annotations


def render_architecture_diagram() -> str:
    """Return a Mermaid diagram of the Omni XAI architecture."""
    return """
    graph TD
        A["Raw Data<br/>(Tabular / Image / Text)"] --> B["Input Layer<br/>omnixai.data"]
        B --> C["AutoExplainer<br/>Preprocessing + Method Selection"]
        C --> D["Explainers<br/>TabularExplainer / VisionExplainer / NLPExplainer"]
        D --> E["Explanations<br/>Feature weights, Heatmaps, Counterfactuals"]
        E --> F["Visualization<br/>Interactive Plots & Dashboard"]

        style A fill:#1a1a2e,stroke:#e94560,color:#fff
        style B fill:#16213e,stroke:#0f3460,color:#fff
        style C fill:#0f3460,stroke:#533483,color:#fff
        style D fill:#533483,stroke:#e94560,color:#fff
        style E fill:#e94560,stroke:#f5a623,color:#fff
        style F fill:#f5a623,stroke:#1a1a2e,color:#1a1a2e
    """


def get_method_info() -> dict:
    """Return descriptions of each XAI method for the dashboard."""
    return {
        "LIME": {
            "full_name": "Local Interpretable Model-agnostic Explanations",
            "type": "Model-Agnostic",
            "description": (
                "LIME approximates a complex model locally by creating perturbed "
                "samples around the instance, then fitting a simple interpretable "
                "model (linear) to those samples, weighted by proximity."
            ),
        },
        "SHAP": {
            "full_name": "SHapley Additive exPlanations",
            "type": "Model-Agnostic",
            "description": (
                "Based on cooperative game theory, SHAP assigns Shapley values "
                "to each feature representing their contribution to the "
                "prediction compared to the average prediction."
            ),
        },
        "Grad-CAM": {
            "full_name": "Gradient-weighted Class Activation Mapping",
            "type": "Model-Specific (CNN)",
            "description": (
                "Grad-CAM computes gradients of the target class score w.r.t. "
                "the last convolutional layer's feature maps, producing a "
                "heatmap of influential spatial regions."
            ),
        },
        "Counterfactual": {
            "full_name": "Counterfactual Explanations",
            "type": "What-If Analysis",
            "description": (
                "Counterfactuals answer: 'What minimal change to the input "
                "would flip the prediction?' They provide actionable insights "
                "by showing the nearest alternative scenario."
            ),
        },
    }
