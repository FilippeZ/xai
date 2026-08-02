"""
Input Layer — Data loading and standardization for XAI experiments.

Supports three modalities: tabular (Iris), image (MNIST), and text (movie reviews).
Each loader returns a standardised dataclass ready for downstream explainers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TabularData:
    """Container for tabular datasets with constraint specification."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    name: str = "tabular"
    immutable_features: List[str] = field(default_factory=list)
    mutable_features: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.mutable_features:
            self.mutable_features = [f for f in self.feature_names if f not in self.immutable_features]


class ClinicalDataLayer:
    """
    Level 1: Input & Constraints Layer (Clinical Data Layer)
    Standardises clinical data and enforces plausibility by declaring immutable features.
    """
    def __init__(self, data: TabularData, immutable_features: Optional[List[str]] = None):
        self.data = data
        if immutable_features is not None:
            self.data.immutable_features = immutable_features
            self.data.mutable_features = [f for f in data.feature_names if f not in immutable_features]

    def validate_plausibility(self, original_instance: np.ndarray, modified_instance: np.ndarray) -> bool:
        """
        Check that no immutable feature (e.g., age, sex, genetics) has been changed.
        """
        for feat in self.data.immutable_features:
            if feat in self.data.feature_names:
                idx = self.data.feature_names.index(feat)
                if not np.isclose(original_instance[idx], modified_instance[idx], atol=1e-4):
                    return False
        return True



@dataclass
class ImageData:
    """Container for image datasets."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    image_shape: tuple
    target_names: List[str]
    name: str = "image"


@dataclass
class TextData:
    """Container for text datasets."""
    texts_train: List[str]
    texts_test: List[str]
    y_train: np.ndarray
    y_test: np.ndarray
    target_names: List[str]
    name: str = "text"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tabular(test_size: float = 0.2, random_state: int = 42) -> TabularData:
    """Load the Iris dataset and split into train / test."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state,
        stratify=iris.target,
    )
    return TabularData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(iris.feature_names),
        target_names=[str(n) for n in iris.target_names],
    )


def load_image(n_samples: int = 2000, random_state: int = 42) -> ImageData:
    """
    Load a subset of MNIST digits (0-9) using scikit-learn.
    Returns 28×28 grayscale images normalised to [0, 1].
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    # Subsample for speed
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y,
    )
    return ImageData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        image_shape=(28, 28, 1),
        target_names=[str(i) for i in range(10)],
    )


def load_text(random_state: int = 42) -> TextData:
    """
    Load a small movie-review-style sentiment dataset.
    Uses 20-newsgroups positive/negative categories as a proxy.
    """
    from sklearn.datasets import fetch_20newsgroups

    cats = ["rec.sport.baseball", "sci.space"]
    train_raw = fetch_20newsgroups(subset="train", categories=cats, random_state=random_state)
    test_raw = fetch_20newsgroups(subset="test", categories=cats, random_state=random_state)

    return TextData(
        texts_train=train_raw.data,
        texts_test=test_raw.data,
        y_train=np.array(train_raw.target),
        y_test=np.array(test_raw.target),
        target_names=[str(n) for n in train_raw.target_names],
    )
