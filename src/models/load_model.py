"""
Model Loading — Utilities to load persisted models.
"""

import joblib


def load_sklearn_model(path: str):
    """Load a scikit-learn model saved with joblib."""
    return joblib.load(path)


def load_pytorch_model(path: str):
    """Load a PyTorch model state dict."""
    import torch
    return torch.load(path, map_location="cpu")
