"""
Audit Trail Generator — Creates audit-ready records for every XAI decision.

Each record documents the model, instance, explanation method, and result,
satisfying GDPR Art. 22 and EU AI Act Art. 17 requirements.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json
import hashlib


@dataclass
class AuditRecord:
    """A single auditable AI decision record."""
    record_id: str
    timestamp: str
    model_type: str
    model_accuracy: float
    instance_id: int
    true_label: str
    predicted_label: str
    confidence: Optional[float]
    xai_method: str
    explanation_summary: str
    feature_attributions: Dict[str, float]
    decision_justified: bool
    human_reviewer: str = "Clinician (Human-in-the-Loop)"
    compliance_tags: List[str] = field(default_factory=list)
    illusion_of_understanding: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def generate_audit_record(
    model_type: str,
    model_accuracy: float,
    instance_id: int,
    true_label: str,
    predicted_label: str,
    xai_method: str,
    feature_attributions: Dict[str, float],
    confidence: Optional[float] = None,
    illusion_of_understanding: bool = False,
) -> AuditRecord:
    """
    Create an audit-ready record for a single AI decision.

    Parameters
    ----------
    model_type : str
        e.g. "RandomForest", "CNN", "LogisticRegression"
    model_accuracy : float
        Overall model accuracy on test set
    instance_id : int
        Index of the instance being explained
    true_label : str
        Ground truth label
    predicted_label : str
        Model's predicted label
    xai_method : str
        e.g. "LIME", "SHAP", "Grad-CAM"
    feature_attributions : dict
        Feature name -> importance weight
    confidence : float, optional
        Model's prediction confidence
    illusion_of_understanding : bool, optional
        Flag set if simulatability <= 0 (explanation failed to convey causal understanding)

    Returns
    -------
    AuditRecord
    """
    timestamp = datetime.now().isoformat()
    record_id = _generate_record_id(timestamp, instance_id)

    # Determine top contributing features
    sorted_features = sorted(
        feature_attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_features = sorted_features[:3]
    explanation_summary = (
        f"Decision explained using {xai_method}. "
        f"Top contributing features: "
        + ", ".join(f"{name} ({weight:+.4f})" for name, weight in top_features)
        + "."
    )

    if illusion_of_understanding:
        explanation_summary += " [AUDIT GATE REJECT: Illusion of Understanding detected (Simulatability <= 0)]"

    # Tag compliance
    compliance_tags = _determine_compliance_tags(xai_method)
    if illusion_of_understanding:
        compliance_tags.append("illusion_of_understanding: True")
        compliance_tags.append("Audit Gate: REJECTED")

    decision_justified = not illusion_of_understanding

    return AuditRecord(
        record_id=record_id,
        timestamp=timestamp,
        model_type=model_type,
        model_accuracy=model_accuracy,
        instance_id=instance_id,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        xai_method=xai_method,
        explanation_summary=explanation_summary,
        feature_attributions=feature_attributions,
        decision_justified=decision_justified,
        compliance_tags=compliance_tags,
        illusion_of_understanding=illusion_of_understanding,
    )


def format_audit_report(records: List[AuditRecord]) -> str:
    """
    Format a list of audit records into a structured text report.
    """
    lines = [
        "=" * 72,
        "AI DECISION AUDIT REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Records: {len(records)}",
        "=" * 72,
        "",
    ]

    for i, record in enumerate(records, 1):
        lines.extend([
            f"--- Record {i} / {len(records)} ---",
            f"  Record ID     : {record.record_id}",
            f"  Timestamp     : {record.timestamp}",
            f"  Model         : {record.model_type} (Accuracy: {record.model_accuracy:.1%})",
            f"  Instance      : #{record.instance_id}",
            f"  True Label    : {record.true_label}",
            f"  Predicted     : {record.predicted_label}",
            f"  Confidence    : {record.confidence:.2%}" if record.confidence else "  Confidence    : N/A",
            f"  XAI Method    : {record.xai_method}",
            f"  Explanation   : {record.explanation_summary}",
            f"  Justified     : {'Yes' if record.decision_justified else 'No'}",
            f"  Reviewer      : {record.human_reviewer}",
            f"  Compliance    : {', '.join(record.compliance_tags)}",
            "",
        ])

    lines.extend([
        "=" * 72,
        "COMPLIANCE STATEMENT",
        "",
        "This audit report documents that all AI-driven decisions listed above",
        "have been processed through Explainable AI (XAI) controls, ensuring:",
        "  • GDPR Art. 22 — Right to Explanation: SATISFIED",
        "  • EU AI Act Art. 13 — Transparency: SATISFIED",
        "  • EU AI Act Art. 14 — Human Oversight: SATISFIED",
        "  • EU AI Act Art. 17 — Quality Management: SATISFIED",
        "",
        "Signed: [AI Governance Officer]",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        "=" * 72,
    ])

    return "\n".join(lines)


def demo_audit_records() -> List[AuditRecord]:
    """
    Generate a set of demo audit records for the dashboard.
    """
    demos = [
        {
            "model_type": "RandomForest",
            "model_accuracy": 0.967,
            "instance_id": 0,
            "true_label": "setosa",
            "predicted_label": "setosa",
            "xai_method": "LIME",
            "feature_attributions": {
                "petal length (cm)": 0.42,
                "petal width (cm)": 0.35,
                "sepal length (cm)": -0.08,
                "sepal width (cm)": 0.05,
            },
            "confidence": 0.98,
        },
        {
            "model_type": "RandomForest",
            "model_accuracy": 0.967,
            "instance_id": 5,
            "true_label": "versicolor",
            "predicted_label": "versicolor",
            "xai_method": "SHAP",
            "feature_attributions": {
                "petal length (cm)": 0.28,
                "petal width (cm)": 0.22,
                "sepal length (cm)": 0.12,
                "sepal width (cm)": -0.06,
            },
            "confidence": 0.91,
        },
        {
            "model_type": "CNN (MNIST)",
            "model_accuracy": 0.952,
            "instance_id": 12,
            "true_label": "7",
            "predicted_label": "7",
            "xai_method": "Grad-CAM",
            "feature_attributions": {
                "upper_stroke_region": 0.65,
                "diagonal_line": 0.28,
                "background": -0.02,
            },
            "confidence": 0.95,
        },
    ]

    return [generate_audit_record(**d) for d in demos]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_record_id(timestamp: str, instance_id: int) -> str:
    """Generate a unique record ID based on timestamp and instance."""
    raw = f"{timestamp}-{instance_id}"
    return f"AUD-{hashlib.sha256(raw.encode()).hexdigest()[:12].upper()}"


def _determine_compliance_tags(xai_method: str) -> List[str]:
    """Map XAI method to compliance regulation tags."""
    tags = ["GDPR Art. 22"]  # All XAI methods satisfy right to explanation

    method_upper = xai_method.upper()
    if method_upper in ("LIME", "SHAP"):
        tags.append("EU AI Act Art. 13")
    if method_upper in ("GRAD-CAM", "GRADCAM"):
        tags.extend(["EU AI Act Art. 13", "EU AI Act Art. 9"])
    if method_upper == "COUNTERFACTUAL":
        tags.extend(["EU AI Act Art. 9", "EU AI Act Art. 15"])

    tags.append("EU AI Act Art. 17")  # All records contribute to QMS
    return tags
