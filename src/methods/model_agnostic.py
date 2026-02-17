"""
Model-Agnostic XAI Methods — LIME & SHAP implementations.

LIME creates local surrogate models by perturbing instances and fitting
a simple interpretable model to the perturbed neighbourhood.

SHAP uses game-theoretic Shapley values to attribute each feature's
contribution to the prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


# ---------------------------------------------------------------------------
# LIME — Tabular
# ---------------------------------------------------------------------------

def lime_tabular_explain(
    model,
    X_train: np.ndarray,
    instance: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    num_features: int = 4,
):
    """
    Generate a LIME explanation for a single tabular instance.

    Returns
    -------
    explanation : lime.explanation.Explanation
        LIME explanation object (call .as_list() for feature weights).
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features,
    )
    return explanation


def lime_tabular_to_dict(explanation) -> dict:
    """Convert a LIME explanation to a simple dict {feature_expr: weight}."""
    return dict(explanation.as_list())


# ---------------------------------------------------------------------------
# LIME — Text
# ---------------------------------------------------------------------------

def lime_text_explain(
    model_pipeline,
    instance: str,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
):
    """
    Generate a LIME explanation for a single text sample.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline with predict_proba
    instance : raw text string
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        instance,
        model_pipeline.predict_proba,
        num_features=num_features,
    )
    return explanation


# ---------------------------------------------------------------------------
# SHAP — Tabular (Tree-based)
# ---------------------------------------------------------------------------

def shap_tabular_explain(model, X: np.ndarray, feature_names: List[str]):
    """
    Compute SHAP values for a tree-based model (e.g. RandomForest).

    Returns
    -------
    shap_values : shap.Explanation
    explainer   : shap.TreeExplainer
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values, explainer


def shap_tabular_to_df(shap_values, feature_names: List[str], class_idx: int = 0) -> pd.DataFrame:
    """Convert SHAP values to a DataFrame for easy plotting."""
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, class_idx]
    return pd.DataFrame(vals, columns=feature_names)


# ---------------------------------------------------------------------------
# SHAP — Kernel (model-agnostic fallback)
# ---------------------------------------------------------------------------

def shap_kernel_explain(
    predict_fn,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
):
    """
    Compute SHAP values using KernelExplainer (works with any model).
    """
    import shap

    explainer = shap.KernelExplainer(predict_fn, X_background[:50])
    shap_values = explainer.shap_values(X_explain)
    return shap_values, explainer
