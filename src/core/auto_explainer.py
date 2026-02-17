"""
AutoExplainer — Automatically select the right XAI method based on data type.

Maps data modalities to their best-suited explanation algorithms
following the Omni XAI architecture described in the thesis.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

EXPLAINER_REGISTRY = {
    "tabular": {
        "local": ["lime", "shap"],
        "global": ["shap", "pdp"],
        "counterfactual": ["mace"],
    },
    "image": {
        "local": ["gradcam", "integrated_gradients"],
        "global": [],
        "counterfactual": ["contrastive"],
    },
    "text": {
        "local": ["lime"],
        "global": [],
        "counterfactual": ["polyjuice"],
    },
}


def select_explainers(data_type: str, scope: str = "local") -> List[str]:
    """
    Return the list of applicable XAI methods for a given data type and scope.

    Parameters
    ----------
    data_type : 'tabular' | 'image' | 'text'
    scope     : 'local' | 'global' | 'counterfactual'
    """
    methods = EXPLAINER_REGISTRY.get(data_type, {}).get(scope, [])
    if not methods:
        raise ValueError(f"No explainers registered for data_type={data_type!r}, scope={scope!r}")
    return methods


def auto_select(data_type: str) -> dict:
    """
    Return a full configuration dict for the given data type.

    Returns
    -------
    dict with keys: data_type, local, global, counterfactual
    """
    registry = EXPLAINER_REGISTRY.get(data_type)
    if registry is None:
        raise ValueError(f"Unknown data type: {data_type!r}")
    return {"data_type": data_type, **registry}


def describe_methods() -> str:
    """Return a human-readable summary of all registered methods."""
    lines = []
    for dtype, scopes in EXPLAINER_REGISTRY.items():
        lines.append(f"\n### {dtype.title()}")
        for scope, methods in scopes.items():
            if methods:
                lines.append(f"  - **{scope}**: {', '.join(m.upper() for m in methods)}")
    return "\n".join(lines)
