"""
Counterfactual Explanations — Generate 'what-if' scenarios.

For tabular data we use a greedy feature-perturbation search:
iteratively modify the feature with the highest gradient of change
in predicted class probability until the prediction flips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


def generate_counterfactual(
    model,
    instance: np.ndarray,
    X_train: np.ndarray,
    feature_names: List[str],
    target_class: Optional[int] = None,
    immutable_features: Optional[List[str]] = None,
    max_iterations: int = 100,
    step_size: float = 0.1,
) -> dict:
    """
    Generate a counterfactual for a tabular instance subject to clinical constraints.

    Strategy: greedily perturb mutable features toward their training-set median
    for the target class until the model flips its prediction. Immutable features
    (e.g., age, sex, genetics) are held strictly constant.

    Parameters
    ----------
    model              : classifier with .predict() and .predict_proba()
    instance           : 1-D array of feature values
    X_train            : training data (for computing class medians)
    feature_names      : list of feature names
    target_class       : desired class (if None, any flip counts)
    immutable_features : list of feature names that CANNOT be changed
    max_iterations     : max perturbation steps
    step_size          : fraction of the distance to move each iteration

    Returns
    -------
    dict with keys: original, counterfactual, changes, original_pred,
                    new_pred, n_steps, success, metrics (L0, L1, L2)
    """
    original_pred = int(model.predict(instance.reshape(1, -1))[0])

    if target_class is None:
        # Pick the second-most-likely class
        probs = model.predict_proba(instance.reshape(1, -1))[0]
        sorted_classes = np.argsort(probs)[::-1]
        target_class = int(sorted_classes[1]) if len(sorted_classes) > 1 else int(sorted_classes[0])

    # Identify indices of immutable features
    immutable_indices = set()
    if immutable_features:
        for fname in immutable_features:
            if fname in feature_names:
                immutable_indices.add(feature_names.index(fname))

    # Compute the median feature values for the target class in training set
    y_train = model.predict(X_train)
    mask = y_train == target_class
    if mask.sum() == 0:
        target_medians = np.median(X_train, axis=0)
    else:
        target_medians = np.median(X_train[mask], axis=0)

    cf = instance.copy().astype(float)
    for step in range(1, max_iterations + 1):
        # Move each feature toward the target median unless it is immutable
        direction = target_medians - cf
        for idx in immutable_indices:
            direction[idx] = 0.0

        cf = cf + step_size * direction

        pred = int(model.predict(cf.reshape(1, -1))[0])
        if pred == target_class:
            break

    new_pred = int(model.predict(cf.reshape(1, -1))[0])
    changes = {
        fn: {"from": round(float(instance[i]), 4), "to": round(float(cf[i]), 4)}
        for i, fn in enumerate(feature_names)
        if not np.isclose(instance[i], cf[i], atol=1e-4)
    }

    # Proximity metrics
    diff = cf - instance
    metrics = {
        "L0": int(np.count_nonzero(np.abs(diff) > 1e-4)),
        "L1": float(np.sum(np.abs(diff))),
        "L2": float(np.sqrt(np.sum(diff ** 2))),
    }

    return {
        "original": instance.tolist(),
        "counterfactual": cf.tolist(),
        "changes": changes,
        "original_pred": original_pred,
        "new_pred": new_pred,
        "target_class": target_class,
        "n_steps": step,
        "success": new_pred == target_class,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Quality metrics (L0, L1, L2 proximity)
# ---------------------------------------------------------------------------

def counterfactual_proximity(original: np.ndarray, counterfactual: np.ndarray) -> dict:
    """Compute L0 (sparsity), L1, L2 distances between original and counterfactual."""
    diff = np.array(counterfactual) - np.array(original)
    return {
        "L0": int(np.count_nonzero(np.abs(diff) > 1e-4)),
        "L1": float(np.sum(np.abs(diff))),
        "L2": float(np.sqrt(np.sum(diff ** 2))),
    }
