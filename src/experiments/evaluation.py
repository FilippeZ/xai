"""
Evaluation Metrics — Simulatability scores and counterfactual quality.
"""

from __future__ import annotations

import numpy as np
from typing import Dict


def simulatability_score(accuracy_pre: float, accuracy_post: float) -> float:
    """
    Effectiveness of an explanation.

    Score = Accuracy_post − Accuracy_pre

    A positive score means the explanation helped the user
    predict the model's behaviour more accurately.
    """
    return accuracy_post - accuracy_pre


def counterfactual_quality(original: np.ndarray, counterfactual: np.ndarray) -> Dict[str, float]:
    """
    Compute quality metrics for a counterfactual explanation.

    Returns
    -------
    dict with:
      - L0 : int   — number of changed features (sparsity)
      - L1 : float — Manhattan distance
      - L2 : float — Euclidean distance
    """
    diff = np.array(counterfactual) - np.array(original)
    return {
        "L0": int(np.count_nonzero(np.abs(diff) > 1e-4)),
        "L1": float(np.sum(np.abs(diff))),
        "L2": float(np.sqrt(np.sum(diff ** 2))),
    }


def explanation_fidelity(model, X_test: np.ndarray, explanations: list) -> float:
    """
    Measure fidelity: how well does the explanation's local model
    approximate the real model's predictions?

    Parameters
    ----------
    model        : the actual model
    X_test       : test instances
    explanations : list of LIME explanation objects with .local_pred

    Returns
    -------
    fidelity : float in [0, 1] — fraction of matching predictions
    """
    real_preds = model.predict(X_test)
    n_match = 0
    for i, exp in enumerate(explanations):
        local_pred = int(np.argmax(exp.local_pred))
        if local_pred == real_preds[i]:
            n_match += 1
    return n_match / len(explanations) if explanations else 0.0


def summary_metrics(session) -> dict:
    """
    Extract all key metrics from a SimulationSession.
    """
    return {
        "session_type": session.session_type,
        "n_trials": len(session.trials),
        "n_completed": session.n_completed,
        "accuracy_pre": round(session.accuracy_pre, 4),
        "accuracy_post": round(session.accuracy_post, 4),
        "simulatability_score": round(session.simulatability_score, 4),
    }
