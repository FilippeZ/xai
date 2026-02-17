"""
Explainer Wrappers — Unified interface for Tabular, Vision, and Text explainers.

Each wrapper composes the raw methods from src.methods.* into a single
`.explain()` call that returns a standardised results dict.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


class TabularExplainer:
    """Unified explainer for tabular data (LIME + SHAP)."""

    def __init__(self, model, X_train: np.ndarray, feature_names: List[str],
                 class_names: Optional[List[str]] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names

    def explain_lime(self, instance: np.ndarray, num_features: int = 4):
        from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict
        explanation = lime_tabular_explain(
            self.model, self.X_train, instance,
            self.feature_names, self.class_names, num_features,
        )
        return {
            "method": "LIME",
            "raw": explanation,
            "weights": lime_tabular_to_dict(explanation),
            "predicted_class": int(self.model.predict(instance.reshape(1, -1))[0]),
        }

    def explain_shap(self, X: np.ndarray):
        from src.methods.model_agnostic import shap_tabular_explain, shap_tabular_to_df
        shap_values, explainer = shap_tabular_explain(self.model, X, self.feature_names)
        return {
            "method": "SHAP",
            "shap_values": shap_values,
            "explainer": explainer,
            "df": shap_tabular_to_df(shap_values, self.feature_names),
        }

    def explain_counterfactual(self, instance: np.ndarray, target_class: Optional[int] = None):
        from src.methods.counterfactual import generate_counterfactual, counterfactual_proximity
        result = generate_counterfactual(
            self.model, instance, self.X_train, self.feature_names, target_class,
        )
        result["proximity"] = counterfactual_proximity(
            result["original"], result["counterfactual"],
        )
        return result


class VisionExplainer:
    """Unified explainer for image data (Grad-CAM)."""

    def __init__(self, model):
        self.model = model

    def explain_gradcam(self, input_tensor, target_class: Optional[int] = None):
        from src.methods.model_specific import grad_cam
        heatmap, predicted = grad_cam(self.model, input_tensor, target_class)
        return {
            "method": "Grad-CAM",
            "heatmap": heatmap,
            "predicted_class": predicted,
        }


class TextExplainer:
    """Unified explainer for text data (LIME)."""

    def __init__(self, model_pipeline, class_names: Optional[List[str]] = None):
        self.model_pipeline = model_pipeline
        self.class_names = class_names

    def explain_lime(self, text: str, num_features: int = 10):
        from src.methods.model_agnostic import lime_text_explain
        explanation = lime_text_explain(
            self.model_pipeline, text, self.class_names, num_features,
        )
        return {
            "method": "LIME",
            "raw": explanation,
            "weights": dict(explanation.as_list()),
            "predicted_class": int(self.model_pipeline.predict([text])[0]),
        }
