"""
Compliance Checker — GDPR + EU AI Act compliance assessment for clinical AI.

Evaluates which XAI methods satisfy regulatory requirements
and generates compliance scoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Compliance checklist items
# ---------------------------------------------------------------------------

def generate_audit_checklist() -> List[Dict]:
    """
    Generate a structured compliance checklist aligned with
    GDPR and EU AI Act requirements for high-risk clinical AI.
    """
    return [
        # --- GDPR Requirements ---
        {
            "id": "GDPR-01",
            "category": "GDPR",
            "requirement": "Right to Explanation (Art. 22)",
            "description": (
                "Every automated decision must be accompanied by a meaningful "
                "explanation of the logic involved."
            ),
            "xai_control": "LIME / SHAP feature attribution",
            "evidence": "Per-instance feature importance exported with each prediction",
            "severity": "Critical",
        },
        {
            "id": "GDPR-02",
            "category": "GDPR",
            "requirement": "Transparency of Processing (Art. 13-14)",
            "description": (
                "Data subjects must be informed about automated decision-making "
                "existence and meaningful logic information."
            ),
            "xai_control": "SHAP global summary + model documentation",
            "evidence": "Global feature importance dashboard + model card",
            "severity": "Critical",
        },
        {
            "id": "GDPR-03",
            "category": "GDPR",
            "requirement": "Right to Contest (Art. 22.3)",
            "description": (
                "Data subject can contest the decision and request human "
                "intervention."
            ),
            "xai_control": "Counterfactual explanations + human-in-the-loop",
            "evidence": "Counterfactual 'what-if' analysis available per decision",
            "severity": "High",
        },
        # --- EU AI Act Requirements ---
        {
            "id": "EUAI-01",
            "category": "EU AI Act",
            "requirement": "Risk Management System (Art. 9)",
            "description": (
                "High-risk AI must have a continuous risk management system "
                "throughout the entire lifecycle."
            ),
            "xai_control": "Clinical risk matrix + counterfactual sensitivity testing",
            "evidence": "Documented risk assessment with mitigation strategies",
            "severity": "Critical",
        },
        {
            "id": "EUAI-02",
            "category": "EU AI Act",
            "requirement": "Data Governance (Art. 10)",
            "description": (
                "Training, validation, and testing datasets must be relevant, "
                "representative, and free from errors."
            ),
            "xai_control": "SHAP cohort analysis for bias detection",
            "evidence": "Bias audit across demographic subgroups",
            "severity": "High",
        },
        {
            "id": "EUAI-03",
            "category": "EU AI Act",
            "requirement": "Technical Documentation (Art. 11)",
            "description": (
                "Comprehensive documentation of the AI system before it is "
                "placed on the market."
            ),
            "xai_control": "Model card + XAI method documentation",
            "evidence": "Architecture diagram, method descriptions, performance metrics",
            "severity": "High",
        },
        {
            "id": "EUAI-04",
            "category": "EU AI Act",
            "requirement": "Transparency to Users (Art. 13)",
            "description": (
                "AI must be transparent enough for users to interpret output "
                "and use it appropriately."
            ),
            "xai_control": "Grad-CAM visual explanations + LIME local explanations",
            "evidence": "Visual and textual explanations accompany every output",
            "severity": "Critical",
        },
        {
            "id": "EUAI-05",
            "category": "EU AI Act",
            "requirement": "Human Oversight (Art. 14)",
            "description": (
                "Effective human oversight including ability to override or "
                "reverse automated decisions."
            ),
            "xai_control": "Simulatability testing + override mechanisms",
            "evidence": "Simulatability scores > 0 confirming explanation effectiveness",
            "severity": "Critical",
        },
        {
            "id": "EUAI-06",
            "category": "EU AI Act",
            "requirement": "Accuracy & Robustness (Art. 15)",
            "description": (
                "High-risk AI must achieve appropriate levels of accuracy, "
                "robustness, and cybersecurity."
            ),
            "xai_control": "Model performance metrics + counterfactual robustness",
            "evidence": "Accuracy, precision, recall metrics + L0/L1/L2 proximity",
            "severity": "High",
        },
        {
            "id": "EUAI-07",
            "category": "EU AI Act",
            "requirement": "Quality Management (Art. 17)",
            "description": (
                "Quality management system ensuring continuous compliance."
            ),
            "xai_control": "Audit trail + monitoring dashboard",
            "evidence": "Timestamped audit records for every AI decision",
            "severity": "High",
        },
    ]


def run_compliance_audit(
    xai_methods_used: List[str],
    has_audit_trail: bool = True,
    has_human_oversight: bool = True,
    has_bias_testing: bool = True,
    simulatability_score: Optional[float] = None,
) -> Dict:
    """
    Evaluate which GDPR / EU AI Act requirements are satisfied
    based on the XAI methods deployed and governance controls in place.

    Parameters
    ----------
    xai_methods_used : list of str
        e.g. ["LIME", "SHAP", "Grad-CAM", "Counterfactual"]
    has_audit_trail : bool
        Whether decision audit records are being generated
    has_human_oversight : bool
        Whether human-in-the-loop is enforced
    has_bias_testing : bool
        Whether bias detection has been performed
    simulatability_score : float or None
        Result of simulatability experiment (positive = good)

    Returns
    -------
    dict with overall score, per-item results, and summary
    """
    methods_upper = [m.upper() for m in xai_methods_used]
    checklist = generate_audit_checklist()
    results = []

    for item in checklist:
        status = _evaluate_item(
            item, methods_upper, has_audit_trail,
            has_human_oversight, has_bias_testing, simulatability_score,
        )
        results.append({**item, "status": status})

    passed = sum(1 for r in results if r["status"] == "✅ Compliant")
    partial = sum(1 for r in results if r["status"] == "⚠️ Partial")
    failed = sum(1 for r in results if r["status"] == "❌ Non-Compliant")
    total = len(results)

    score = (passed + 0.5 * partial) / total if total > 0 else 0

    return {
        "results": results,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "total": total,
        "score": score,
        "grade": _score_to_grade(score),
        "timestamp": datetime.now().isoformat(),
    }


def get_compliance_status(audit_result: Dict) -> Dict:
    """
    Generate a summary status from an audit result.
    """
    return {
        "overall_grade": audit_result["grade"],
        "score_pct": f"{audit_result['score']:.0%}",
        "passed": audit_result["passed"],
        "partial": audit_result["partial"],
        "failed": audit_result["failed"],
        "total": audit_result["total"],
        "timestamp": audit_result["timestamp"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_item(
    item: Dict,
    methods: List[str],
    has_audit: bool,
    has_oversight: bool,
    has_bias: bool,
    sim_score: Optional[float],
) -> str:
    """Determine compliance status for a single checklist item."""
    item_id = item["id"]

    if item_id == "GDPR-01":
        if "LIME" in methods or "SHAP" in methods:
            return "✅ Compliant"
        return "❌ Non-Compliant"

    if item_id == "GDPR-02":
        if "SHAP" in methods:
            return "✅ Compliant"
        if "LIME" in methods:
            return "⚠️ Partial"
        return "❌ Non-Compliant"

    if item_id == "GDPR-03":
        if "COUNTERFACTUAL" in methods and has_oversight:
            return "✅ Compliant"
        if "COUNTERFACTUAL" in methods or has_oversight:
            return "⚠️ Partial"
        return "❌ Non-Compliant"

    if item_id == "EUAI-01":
        if has_audit and ("COUNTERFACTUAL" in methods):
            return "✅ Compliant"
        if has_audit or ("COUNTERFACTUAL" in methods):
            return "⚠️ Partial"
        return "❌ Non-Compliant"

    if item_id == "EUAI-02":
        if has_bias and "SHAP" in methods:
            return "✅ Compliant"
        if has_bias or "SHAP" in methods:
            return "⚠️ Partial"
        return "❌ Non-Compliant"

    if item_id == "EUAI-03":
        # Documentation is always partially available (we have the dashboard)
        return "✅ Compliant" if has_audit else "⚠️ Partial"

    if item_id == "EUAI-04":
        if ("GRAD-CAM" in methods or "GRADCAM" in methods) and "LIME" in methods:
            return "✅ Compliant"
        if any(m in methods for m in ["LIME", "SHAP", "GRAD-CAM", "GRADCAM"]):
            return "⚠️ Partial"
        return "❌ Non-Compliant"

    if item_id == "EUAI-05":
        if has_oversight and sim_score is not None and sim_score > 0:
            return "✅ Compliant"
        if has_oversight:
            return "⚠️ Partial"
        return "❌ Non-Compliant"

    if item_id == "EUAI-06":
        if "COUNTERFACTUAL" in methods:
            return "✅ Compliant"
        return "⚠️ Partial"

    if item_id == "EUAI-07":
        return "✅ Compliant" if has_audit else "❌ Non-Compliant"

    return "⚠️ Partial"


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"
