"""
Model Training — Train ML models for each data modality.

* Tabular  → RandomForestClassifier (scikit-learn)
* Image    → Simple CNN (PyTorch)
* Text     → TF-IDF + LogisticRegression pipeline (scikit-learn)
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------

def train_tabular_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """Train a RandomForest on tabular data and return the fitted model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def train_text_model(texts_train, y_train, max_features: int = 5000):
    """Train a TF-IDF + LogisticRegression pipeline for text classification."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipeline.fit(texts_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Image — PyTorch CNN
# ---------------------------------------------------------------------------

def train_image_model(X_train, y_train, epochs: int = 5, lr: float = 0.001):
    """
    Train a small CNN on flattened image data (expects 784-dim vectors).
    Returns a trained PyTorch model in eval mode.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    device = torch.device("cpu")
    X_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_classes = len(np.unique(y_train))
    model = SimpleCNN(num_classes=num_classes).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str):
    """Save a scikit-learn model with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def evaluate_model(model, X_test, y_test) -> float:
    """Return accuracy on the test set."""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
