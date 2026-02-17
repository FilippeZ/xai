"""
AI Governance Framework — Risk classification, controls, and regulatory mappings.

Based on GDPR, EU AI Act, and thesis findings on XAI for clinical software.
"""

from __future__ import annotations
from typing import Dict, List


def get_risk_categories() -> List[Dict]:
    """
    EU AI Act risk classification pyramid.
    Returns tiers from highest to lowest risk.
    """
    return [
        {
            "tier": "Unacceptable Risk",
            "level": 4,
            "color": "#dc3545",
            "icon": "🚫",
            "description": "AI systems that pose a clear threat to safety, livelihoods, or rights.",
            "examples": [
                "Social scoring by governments",
                "Real-time biometric surveillance",
                "Subliminal manipulation systems",
            ],
            "action": "PROHIBITED — Cannot be deployed in the EU.",
        },
        {
            "tier": "High Risk",
            "level": 3,
            "color": "#fd7e14",
            "icon": "⚠️",
            "description": "AI in critical domains requiring strict governance and transparency.",
            "examples": [
                "Clinical decision support systems (CDSS)",
                "Medical diagnostic AI (e.g. radiology, pathology)",
                "AI-driven drug dosage optimization",
                "Insurance risk assessment algorithms",
            ],
            "action": "ALLOWED with mandatory conformity assessment, XAI controls, and human oversight.",
            "xai_requirements": [
                "Full explainability of every decision",
                "Auditable decision trails",
                "Human-in-the-loop for final decisions",
                "Bias detection and mitigation",
                "Continuous post-market monitoring",
            ],
        },
        {
            "tier": "Limited Risk",
            "level": 2,
            "color": "#ffc107",
            "icon": "ℹ️",
            "description": "AI with specific transparency obligations.",
            "examples": [
                "Chatbots (must disclose AI nature)",
                "Deepfake generators (must label content)",
                "Emotion recognition systems",
            ],
            "action": "ALLOWED with transparency obligations (user must know they interact with AI).",
        },
        {
            "tier": "Minimal Risk",
            "level": 1,
            "color": "#28a745",
            "icon": "✅",
            "description": "AI systems with negligible risk — no specific requirements.",
            "examples": [
                "Spam filters",
                "AI-powered video games",
                "Inventory management systems",
            ],
            "action": "ALLOWED — No additional regulatory burden.",
        },
    ]


def get_governance_controls() -> Dict[str, Dict]:
    """
    XAI governance controls mapped to regulatory requirements.
    Each control specifies the XAI method, what it audits, and which regulation it satisfies.
    """
    return {
        "Transparency Control": {
            "icon": "🔍",
            "methods": ["LIME", "SHAP"],
            "description": (
                "Model-agnostic methods that analyse input-output relationships "
                "to explain any algorithm's decisions without accessing internals."
            ),
            "audits": "Feature-level decision justification",
            "regulation": "GDPR Art. 22 — Right to Explanation",
            "clinical_use": (
                "Transforms 'Malignancy: 87%' into a documented justification "
                "showing which patient features drove the prediction."
            ),
        },
        "Visual Inspection Control": {
            "icon": "🔬",
            "methods": ["Grad-CAM", "Integrated Gradients"],
            "description": (
                "Model-specific methods that highlight which regions of medical "
                "images (X-rays, MRI, pathology slides) the CNN focuses on."
            ),
            "audits": "Spatial decision attribution for imaging AI",
            "regulation": "EU AI Act — High-risk system documentation",
            "clinical_use": (
                "Enables radiologists to verify that the AI focuses on the "
                "lesion, not on irrelevant artifacts (e.g. ruler marks)."
            ),
        },
        "Causal Understanding Control": {
            "icon": "🔄",
            "methods": ["Counterfactual Explanations (MACE, Polyjuice)"],
            "description": (
                "Generates 'what-if' scenarios: 'What minimal change to the "
                "patient's data would alter the diagnosis?' Moves from "
                "correlation to causality."
            ),
            "audits": "Actionable decision pathways and sensitivity analysis",
            "regulation": "EU AI Act — Risk management & robustness testing",
            "clinical_use": (
                "Answers: 'What symptoms would need to change for the diagnosis "
                "NOT to be cancer?' — enabling targeted intervention planning."
            ),
        },
        "Simulatability Control": {
            "icon": "🧪",
            "methods": ["Forward Simulation", "Counterfactual Simulation"],
            "description": (
                "Tests whether XAI explanations genuinely help clinicians "
                "predict the model's behaviour (not just feel confident)."
            ),
            "audits": "Explanation effectiveness and user comprehension",
            "regulation": "EU AI Act — Human oversight requirement",
            "clinical_use": (
                "Validates that explanations don't create an 'illusion of "
                "understanding' where doctors feel confident but cannot "
                "actually anticipate model errors."
            ),
        },
    }


def get_regulatory_requirements() -> List[Dict]:
    """
    Key regulatory requirements from GDPR and EU AI Act
    relevant to clinical AI systems.
    """
    return [
        {
            "regulation": "GDPR",
            "article": "Article 22",
            "title": "Right to Explanation",
            "requirement": (
                "Citizens cannot be subjected to decisions affecting them "
                "(medical diagnoses, insurance) solely based on automated "
                "processing without human intervention."
            ),
            "penalty": "Up to €20 million or 4% of global annual turnover",
            "xai_solution": "LIME/SHAP feature attribution + audit trail",
            "icon": "🛡️",
        },
        {
            "regulation": "GDPR",
            "article": "Articles 13-14",
            "title": "Transparency & Information",
            "requirement": (
                "Data subjects must be informed about the existence of "
                "automated decision-making, including meaningful information "
                "about the logic involved."
            ),
            "penalty": "Up to €20 million or 4% of global annual turnover",
            "xai_solution": "Global SHAP summaries + model documentation",
            "icon": "📋",
        },
        {
            "regulation": "EU AI Act",
            "article": "Article 9",
            "title": "Risk Management System",
            "requirement": (
                "High-risk AI must have a risk management system that identifies "
                "and mitigates risks throughout the AI lifecycle."
            ),
            "penalty": "Up to €35 million or 7% of global annual turnover",
            "xai_solution": "Counterfactual sensitivity analysis + bias detection",
            "icon": "⚖️",
        },
        {
            "regulation": "EU AI Act",
            "article": "Article 13",
            "title": "Transparency & Information",
            "requirement": (
                "High-risk AI must be designed to be sufficiently transparent "
                "that users can interpret and use the output appropriately."
            ),
            "penalty": "Up to €35 million or 7% of global annual turnover",
            "xai_solution": "Grad-CAM visual explanations + LIME local explanations",
            "icon": "🔍",
        },
        {
            "regulation": "EU AI Act",
            "article": "Article 14",
            "title": "Human Oversight",
            "requirement": (
                "High-risk AI must allow effective human oversight, including "
                "the ability to override or reverse automated decisions."
            ),
            "penalty": "Up to €35 million or 7% of global annual turnover",
            "xai_solution": "Simulatability testing + human-in-the-loop design",
            "icon": "👤",
        },
        {
            "regulation": "EU AI Act",
            "article": "Article 17",
            "title": "Quality Management System",
            "requirement": (
                "Providers of high-risk AI must implement a quality management "
                "system ensuring compliance throughout the AI lifecycle."
            ),
            "penalty": "Up to €35 million or 7% of global annual turnover",
            "xai_solution": "Audit trail + continuous monitoring dashboard",
            "icon": "📊",
        },
    ]


def get_clinical_risk_matrix() -> List[Dict]:
    """
    Clinical software risk assessment matrix based on
    the thesis findings about black-box risks.
    """
    return [
        {
            "risk": "Opaque Decision-Making",
            "severity": "Critical",
            "color": "#dc3545",
            "description": (
                "An algorithm recommending a surgical procedure without clear "
                "justification could endanger lives."
            ),
            "mitigation": "LIME/SHAP explainability for every clinical decision",
            "status": "Mitigated by XAI Controls",
        },
        {
            "risk": "Undetected Bias",
            "severity": "High",
            "color": "#fd7e14",
            "description": (
                "Models trained on specific racial demographics may fail on "
                "others. Without XAI, systematic bias stays hidden."
            ),
            "mitigation": "SHAP global analysis + fairness metrics across cohorts",
            "status": "Mitigated by XAI Controls",
        },
        {
            "risk": "Spurious Correlations",
            "severity": "High",
            "color": "#fd7e14",
            "description": (
                "Model relies on irrelevant patterns (e.g. ruler marks on X-ray "
                "rather than actual lesions)."
            ),
            "mitigation": "Grad-CAM visual verification by clinicians",
            "status": "Mitigated by XAI Controls",
        },
        {
            "risk": "Clinician Trust Gap",
            "severity": "Medium",
            "color": "#ffc107",
            "description": (
                "Without explanation, clinicians face a dilemma: accept algo "
                "blindly or reject without cause."
            ),
            "mitigation": "Human-in-the-loop design + simulatability testing",
            "status": "Mitigated by XAI Controls",
        },
        {
            "risk": "Regulatory Non-Compliance",
            "severity": "Critical",
            "color": "#dc3545",
            "description": (
                "GDPR fines up to €20M; EU AI Act fines up to €35M for "
                "non-compliant high-risk AI in clinical settings."
            ),
            "mitigation": "Full audit trail + compliance dashboard",
            "status": "Mitigated by XAI Controls",
        },
        {
            "risk": "Illusion of Understanding",
            "severity": "Medium",
            "color": "#ffc107",
            "description": (
                "Users may feel confident with explanations but cannot actually "
                "predict model behaviour — false sense of understanding."
            ),
            "mitigation": "Simulatability testing (Forward + Counterfactual)",
            "status": "Mitigated by XAI Controls",
        },
    ]
