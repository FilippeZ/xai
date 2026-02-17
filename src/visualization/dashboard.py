"""
Dashboard helpers — Streamlit integration utilities.
Includes architecture diagrams, method info, thesis findings,
and governance framework visualizations.
"""

from __future__ import annotations


def render_architecture_diagram() -> str:
    """Return a Mermaid diagram of the Omni XAI architecture (Streamlit-compatible)."""
    return """```mermaid
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
```"""


def render_governance_architecture() -> str:
    """Return a Mermaid diagram of the AI Governance pipeline."""
    return """```mermaid
graph TD
    A["🏥 Clinical AI System<br/>High-Risk Classification"] --> B["⚖️ EU AI Act<br/>Conformity Assessment"]
    B --> C["🔍 XAI Controls<br/>LIME / SHAP / Grad-CAM / Counterfactuals"]
    C --> D["📋 Audit Trail<br/>Decision Records + Compliance Tags"]
    D --> E["👤 Human-in-the-Loop<br/>Clinician Review & Override"]
    E --> F["✅ Compliant Decision<br/>GDPR Art.22 + EU AI Act"]
    
    C --> G["🧪 Simulatability Testing<br/>Explanation Effectiveness"]
    G --> E

    style A fill:#dc3545,stroke:#fff,color:#fff
    style B fill:#fd7e14,stroke:#fff,color:#fff
    style C fill:#667eea,stroke:#fff,color:#fff
    style D fill:#533483,stroke:#fff,color:#fff
    style E fill:#28a745,stroke:#fff,color:#fff
    style F fill:#20c997,stroke:#1a1a2e,color:#1a1a2e
    style G fill:#ffc107,stroke:#1a1a2e,color:#1a1a2e
```"""


def get_method_info() -> dict:
    """Return descriptions of each XAI method for the dashboard."""
    return {
        "LIME": {
            "full_name": "Local Interpretable Model-agnostic Explanations",
            "type": "Model-Agnostic · Local",
            "description": (
                "Creates perturbed samples around an instance, fits a locally-"
                "weighted linear model, and extracts feature importance weights. "
                "Most effective for tabular classification tasks."
            ),
            "icon": "🍋",
        },
        "SHAP": {
            "full_name": "SHapley Additive exPlanations",
            "type": "Model-Agnostic · Global + Local",
            "description": (
                "Uses Shapley values from cooperative game theory to assign each "
                "feature a contribution score. Supports TreeExplainer (fast, exact "
                "for trees) and KernelExplainer (universal fallback)."
            ),
            "icon": "📊",
        },
        "Grad-CAM": {
            "full_name": "Gradient-weighted Class Activation Mapping",
            "type": "Model-Specific · CNN",
            "description": (
                "Computes gradients of the target class w.r.t. the last conv layer, "
                "producing spatial heatmaps that reveal which image regions the CNN "
                "relies on for its prediction."
            ),
            "icon": "🔥",
        },
        "Counterfactual": {
            "full_name": "Counterfactual Explanations",
            "type": "What-If Analysis · Causal",
            "description": (
                "Answers: 'What minimal change flips the prediction?' Moves from "
                "correlation to causality — e.g. 'If income were €15K higher, "
                "the loan would be approved.' Evaluated with L₀/L₁/L₂ proximity."
            ),
            "icon": "🔄",
        },
    }


def get_thesis_findings() -> dict:
    """Return key findings from the thesis for the homepage."""
    return {
        "illusion_of_understanding": (
            "Subjective satisfaction does NOT equal objective understanding. "
            "Users often rated complex explanations positively but could not "
            "actually predict the model's behaviour — an 'illusion of understanding'."
        ),
        "best_tabular": (
            "LIME was the most effective method for classification tasks "
            "involving tabular data, helping users identify key features "
            "driving the model's decision."
        ),
        "best_counterfactual": (
            "The Prototype method proved most effective for Counterfactual "
            "Simulation across both text and tabular data."
        ),
        "text_challenge": (
            "For text data, no single method stood out consistently. LIME could "
            "identify keywords (e.g. 'funny' = positive) but failed when meaning "
            "changed due to subtle phrasing."
        ),
        "causability": (
            "The thesis argues AI must move from explaining correlations to "
            "explaining causality — 'Causability' — the ability to provide "
            "explanations that let humans understand the cause and how to alter the result."
        ),
        "gdpr": (
            "XAI is legally required under GDPR's 'Right to Explanation' for "
            "automated decisions. Non-compliance can lead to fines of up to "
            "€20 million or 4% of annual global turnover."
        ),
        "medical": (
            "In medicine, XAI transforms opaque predictions (e.g. 'Cancer: 87%') "
            "into justified diagnoses where a doctor can validate the reasoning "
            "and specific image regions highlighted."
        ),
    }


def get_governance_findings() -> dict:
    """
    Key governance findings from NotebookLM analysis of the XAI thesis.
    Sourced from: NotebookLM notebook 'xai' (55663590-a3c4-445c-9221-f48da1238f53)
    """
    return {
        "black_box_risk": {
            "title": "🚫 'Black Box' Risks in Clinical AI",
            "finding": (
                "Modern AI, particularly deep neural networks, provides results "
                "without revealing the logic taken to reach them. In clinical "
                "environments, an algorithm recommending a surgical procedure "
                "without clear justification could endanger lives."
            ),
            "source": "NotebookLM XAI Thesis Analysis",
        },
        "gdpr_mandate": {
            "title": "🛡️ GDPR Article 22 — Right to Explanation",
            "finding": (
                "The GDPR imposes strict transparency standards. Article 22 grants "
                "the 'Right to Explanation,' ensuring citizens cannot be subjected "
                "to decisions (medical diagnoses, insurance) solely based on "
                "automated processing without human intervention."
            ),
            "source": "NotebookLM XAI Thesis Analysis",
        },
        "eu_ai_act": {
            "title": "⚖️ EU AI Act — High-Risk Classification",
            "finding": (
                "The AI Act establishes a regulatory landscape where explainability "
                "is a prerequisite for any critical AI application. Clinical decision "
                "support systems are classified as HIGH RISK, requiring mandatory "
                "conformity assessment, XAI controls, and human oversight."
            ),
            "source": "NotebookLM XAI Thesis Analysis",
        },
        "glass_box": {
            "title": "🔍 From 'Black Box' to 'Glass Box'",
            "finding": (
                "The thesis proposes shifting AI from opaque 'black boxes' to "
                "transparent 'glass boxes' using XAI controls: LIME/SHAP for "
                "feature attribution, Grad-CAM for visual inspection, "
                "Counterfactuals for causal understanding, and Simulatability "
                "testing for explanation validation."
            ),
            "source": "NotebookLM XAI Thesis Analysis",
        },
        "human_in_loop": {
            "title": "👤 Human-in-the-Loop Necessity",
            "finding": (
                "AI should NOT replace the physician in critical life-or-death "
                "decisions. Instead, XAI acts as a collaborative tool where AI "
                "handles data processing and pattern recognition, while humans "
                "handle complex, unstructured problems requiring intuition and "
                "ethical judgment."
            ),
            "source": "NotebookLM XAI Thesis Analysis",
        },
        "penalties": {
            "title": "💰 Non-Compliance Penalties",
            "finding": (
                "GDPR: fines up to €20 million or 4% of global turnover. "
                "EU AI Act: fines up to €35 million or 7% of global annual "
                "turnover. This makes XAI controls essential for the legal "
                "viability of any clinical AI system in Europe."
            ),
            "source": "NotebookLM XAI Thesis Analysis",
        },
    }
