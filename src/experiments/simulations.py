"""
Simulatability Experiments — Forward and Counterfactual Simulation.

Forward Simulation: Test whether explanations help users predict model output.
Counterfactual Simulation: Test whether explanations help users predict output
on perturbed inputs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimulationTrial:
    """A single trial in the simulatability experiment."""
    instance_idx: int
    true_label: int
    model_prediction: int
    user_prediction_pre: Optional[int] = None
    user_prediction_post: Optional[int] = None
    explanation_shown: bool = False
    perturbation: Optional[np.ndarray] = None


@dataclass
class SimulationSession:
    """Tracks a full simulatability session across multiple trials."""
    trials: List[SimulationTrial] = field(default_factory=list)
    session_type: str = "forward"  # 'forward' or 'counterfactual'

    @property
    def accuracy_pre(self) -> float:
        """User accuracy before seeing explanations."""
        answered = [t for t in self.trials if t.user_prediction_pre is not None]
        if not answered:
            return 0.0
        correct = sum(1 for t in answered if t.user_prediction_pre == t.model_prediction)
        return correct / len(answered)

    @property
    def accuracy_post(self) -> float:
        """User accuracy after seeing explanations."""
        answered = [t for t in self.trials if t.user_prediction_post is not None]
        if not answered:
            return 0.0
        correct = sum(1 for t in answered if t.user_prediction_post == t.model_prediction)
        return correct / len(answered)

    @property
    def simulatability_score(self) -> float:
        """Effectiveness = Accuracy_post − Accuracy_pre."""
        return self.accuracy_post - self.accuracy_pre

    @property
    def n_completed(self) -> int:
        return sum(1 for t in self.trials if t.user_prediction_post is not None)


def create_forward_trials(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    n_trials: int = 10,
    random_state: int = 42,
) -> SimulationSession:
    """
    Prepare forward simulation trials.

    Selects n_trials random test instances, balanced across
    correct and incorrect model predictions.
    """
    rng = np.random.RandomState(random_state)
    preds = model.predict(X_test)
    correct_mask = preds == y_test
    incorrect_mask = ~correct_mask

    n_correct = min(n_trials // 2, correct_mask.sum())
    n_incorrect = min(n_trials - n_correct, incorrect_mask.sum())

    correct_idx = rng.choice(np.where(correct_mask)[0], size=n_correct, replace=False)
    incorrect_idx = rng.choice(np.where(incorrect_mask)[0], size=n_incorrect, replace=False) if n_incorrect > 0 else np.array([], dtype=int)

    all_idx = np.concatenate([correct_idx, incorrect_idx])
    rng.shuffle(all_idx)

    trials = []
    for idx in all_idx:
        trials.append(SimulationTrial(
            instance_idx=int(idx),
            true_label=int(y_test[idx]),
            model_prediction=int(preds[idx]),
        ))

    return SimulationSession(trials=trials, session_type="forward")


def create_counterfactual_trials(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    n_trials: int = 10,
    perturbation_scale: float = 0.3,
    random_state: int = 42,
) -> SimulationSession:
    """
    Prepare counterfactual simulation trials.

    For each trial, creates a perturbed version of the input
    and the user must predict the model's output on the perturbed input.
    """
    rng = np.random.RandomState(random_state)
    preds = model.predict(X_test)

    idx_pool = rng.choice(len(X_test), size=min(n_trials, len(X_test)), replace=False)

    trials = []
    for idx in idx_pool:
        noise = rng.normal(0, perturbation_scale, size=X_test[idx].shape)
        perturbed = X_test[idx] + noise
        perturbed_pred = int(model.predict(perturbed.reshape(1, -1))[0])

        trials.append(SimulationTrial(
            instance_idx=int(idx),
            true_label=int(y_test[idx]),
            model_prediction=perturbed_pred,  # The user must predict THIS
            perturbation=perturbed,
        ))

    return SimulationSession(trials=trials, session_type="counterfactual")
