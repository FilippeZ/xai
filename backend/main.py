"""
XAI Governance Middleware — Full Backend v3.0
All three modalities (tabular / image / text) fully implemented.
No placeholders.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.input_layer import load_tabular, load_image, load_text, ClinicalDataLayer
from src.methods.counterfactual import generate_counterfactual
from src.methods.model_agnostic import lime_tabular_explain, lime_tabular_to_dict, lime_text_explain
from src.experiments.evaluation import simulatability_score, detect_illusion_of_understanding
from src.governance.audit_trail import generate_audit_record, format_audit_report, AuditRecord
from src.governance.compliance import generate_audit_checklist
from src.governance.framework import get_risk_categories, get_governance_controls

# ===========================================================================
app = FastAPI(
    title="XAI Governance Middleware API",
    description="Enterprise REST API for Explainable AI — GDPR Art.22 & EU AI Act Art.9/13/14/17",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_cache: Dict[str, Any] = {}

# ===========================================================================
# Asset loaders (lazy, cached)
# ===========================================================================

def _tabular():
    if "tabular" not in _cache:
        from sklearn.ensemble import RandomForestClassifier
        data = load_tabular(random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(data.X_train, data.y_train)
        acc = float(accuracy_score(data.y_test, model.predict(data.X_test)))
        _cache["tabular"] = dict(data=data, model=model, accuracy=round(acc, 4),
                                  cdl=ClinicalDataLayer(data=data))
    return _cache["tabular"]


def _image():
    """Train CNN on 1 200-sample MNIST subset (≈30 s first call)."""
    if "image" not in _cache:
        import torch, torch.nn as nn
        from src.models.train_model import train_image_model
        data = load_image(n_samples=1200, random_state=42)
        model = train_image_model(data.X_train, data.y_train, epochs=3, lr=0.002)

        # Accuracy
        X_t = torch.tensor(data.X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            preds = torch.argmax(model(X_t), dim=1).numpy()
        acc = float(accuracy_score(data.y_test, preds))
        _cache["image"] = dict(data=data, model=model, accuracy=round(acc, 4))
    return _cache["image"]


def _text():
    """Train TF-IDF + LogReg on 20newsgroups (≈5 s first call)."""
    if "text" not in _cache:
        from src.models.train_model import train_text_model
        data = load_text(random_state=42)
        model = train_text_model(data.texts_train, data.y_train)
        acc = float(accuracy_score(data.y_test, model.predict(data.texts_test)))
        _cache["text"] = dict(data=data, model=model, accuracy=round(acc, 4))
    return _cache["text"]


# ===========================================================================
# Grad-CAM helper
# ===========================================================================

def _gradcam(model, img_tensor, target_class: int):
    """
    Compute Grad-CAM for the last Conv2d layer of SimpleCNN.
    Returns a 28x28 np.ndarray of activation values normalised to [0,1].
    """
    import torch, torch.nn.functional as F

    activations, gradients = [], []

    # Hook the LAST Conv2d layer (features[3])
    target_layer = model.features[3]
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))

    model.eval()
    inp = img_tensor.clone().requires_grad_(False)
    out = model(inp.unsqueeze(0))

    model.zero_grad()
    score = out[0, target_class]
    score.backward()

    fh.remove(); bh.remove()

    acts = activations[0][0]   # (32, 7, 7)
    grads = gradients[0][0]    # (32, 7, 7)

    weights = grads.mean(dim=(1, 2), keepdim=True)  # (32, 1, 1)
    cam = torch.relu((weights * acts).sum(dim=0))    # (7, 7)

    # Upsample to 28×28
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                        size=(28, 28), mode="bilinear", align_corners=False)
    cam = cam.squeeze().numpy()

    # Normalise
    mn, mx = cam.min(), cam.max()
    if mx > mn:
        cam = (cam - mn) / (mx - mn)
    return cam


# ===========================================================================
# Schemas
# ===========================================================================

class ExplainRequest(BaseModel):
    modality: str = Field("tabular", description="tabular | image | text")
    method: str = Field("SHAP", description="SHAP | LIME | COUNTERFACTUAL | GRAD-CAM")
    instance_index: int = Field(0, ge=0)
    immutable_features: List[str] = Field(default=["sepal length (cm)"])
    num_features: int = Field(4, ge=1, le=15)


class SimulatabilityRequest(BaseModel):
    accuracy_pre: float = Field(..., ge=0.0, le=1.0)
    accuracy_post: float = Field(..., ge=0.0, le=1.0)


class AuditRequest(BaseModel):
    model_type: str = "RandomForest"
    model_accuracy: float = Field(0.967, ge=0, le=1)
    instance_id: int = 0
    true_label: str = "setosa"
    predicted_label: str = "setosa"
    xai_method: str = "SHAP"
    feature_attributions: Dict[str, float]
    confidence: Optional[float] = None
    illusion_of_understanding: bool = False


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/api/health")
def health():
    t = _tabular()
    return {"status": "ok", "system": "XAI Governance Middleware v3.0",
            "model": "RandomForest (Iris)", "accuracy": t["accuracy"]}


@app.get("/api/datasets")
def datasets(modality: str = "tabular"):
    m = modality.lower()
    if m == "tabular":
        a = _tabular()
        d = a["data"]
        model = a["model"]
        samples = []
        for i in range(min(10, len(d.X_test))):
            inst = d.X_test[i]
            probs = model.predict_proba(inst.reshape(1, -1))[0]
            pi = int(np.argmax(probs))
            samples.append({"index": i,
                             "values": {fn: round(float(inst[j]), 4) for j, fn in enumerate(d.feature_names)},
                             "true_label": d.target_names[int(d.y_test[i])],
                             "predicted_label": d.target_names[pi],
                             "confidence": round(float(probs[pi]), 4)})
        return {"modality": "tabular", "dataset": "Iris Benchmark",
                "feature_names": d.feature_names, "target_names": d.target_names,
                "accuracy": a["accuracy"], "samples": samples}

    elif m == "image":
        a = _image()
        d = a["data"]
        return {"modality": "image", "dataset": "MNIST Digits (28×28)",
                "n_classes": 10, "target_names": d.target_names,
                "n_test": len(d.X_test), "accuracy": a["accuracy"]}

    elif m == "text":
        a = _text()
        d = a["data"]
        return {"modality": "text", "dataset": "20 Newsgroups (Baseball vs Space)",
                "target_names": d.target_names,
                "n_test": len(d.texts_test), "accuracy": a["accuracy"],
                "samples": [{"index": i, "text_preview": d.texts_test[i][:120] + "...",
                              "true_label": d.target_names[int(d.y_test[i])]}
                             for i in range(min(5, len(d.texts_test)))]}
    raise HTTPException(400, f"Unknown modality: {modality}")


@app.post("/api/explain")
def explain(req: ExplainRequest):
    modality = req.modality.lower()
    method = req.method.upper().replace("-", "")
    t0 = time.perf_counter()

    # ── TABULAR ────────────────────────────────────────────────────────────
    if modality == "tabular":
        a = _tabular()
        data, model = a["data"], a["model"]
        if req.instance_index >= len(data.X_test):
            raise HTTPException(400, "instance_index out of range")

        inst = data.X_test[req.instance_index]
        true_idx = int(data.y_test[req.instance_index])
        probs = model.predict_proba(inst.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))
        conf = round(float(probs[pred_idx]), 4)

        base = {
            "modality": "tabular",
            "instance_index": req.instance_index,
            "original_values": {fn: round(float(inst[i]), 4) for i, fn in enumerate(data.feature_names)},
            "true_label": data.target_names[true_idx],
            "predicted_label": data.target_names[pred_idx],
            "confidence": conf,
        }

        # ── SHAP ──
        if method == "SHAP":
            import shap
            expl = shap.TreeExplainer(model)
            sv = expl(inst.reshape(1, -1))
            raw = sv.values[0]
            if raw.ndim == 2:
                raw = raw[:, pred_idx]
            attrs = {fn: round(float(raw[i]), 6) for i, fn in enumerate(data.feature_names)}
            top = dict(sorted(attrs.items(), key=lambda x: abs(x[1]), reverse=True)[:req.num_features])
            bv = sv.base_values[0]
            base_val = float(bv[pred_idx]) if hasattr(bv, "__len__") else float(bv)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {**base, "method": "SHAP", "elapsed_ms": elapsed,
                    "feature_attributions": attrs, "top_features": top,
                    "base_value": round(base_val, 6),
                    "compliance_tags": ["GDPR Art. 22 – Right to Explanation",
                                        "EU AI Act Art. 13 – Transparency"],
                    "gdpr_art22": True, "eu_aiact_art13": True}

        # ── LIME ──
        elif method == "LIME":
            expl = lime_tabular_explain(model, data.X_train, inst,
                                         data.feature_names, data.target_names,
                                         req.num_features)
            raw = lime_tabular_to_dict(expl)
            attrs: Dict[str, float] = {}
            for cond, w in raw.items():
                matched = next((fn for fn in data.feature_names
                                if fn.lower() in cond.lower()), cond)
                attrs[matched] = round(float(w), 6)
            local_pred = {data.target_names[i]: round(float(p), 4)
                          for i, p in enumerate(expl.local_pred)}
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {**base, "method": "LIME", "elapsed_ms": elapsed,
                    "feature_attributions": attrs, "lime_conditions": raw,
                    "local_prediction": local_pred,
                    "compliance_tags": ["GDPR Art. 22 – Right to Explanation",
                                        "EU AI Act Art. 13 – Transparency"],
                    "gdpr_art22": True, "eu_aiact_art13": True}

        # ── COUNTERFACTUAL ──
        elif method == "COUNTERFACTUAL":
            cdl = ClinicalDataLayer(data=data, immutable_features=req.immutable_features)
            cf = generate_counterfactual(
                model, inst, data.X_train,
                data.feature_names,
                immutable_features=req.immutable_features,
                max_iterations=300, step_size=0.04)
            cf_arr = np.array(cf["counterfactual"])
            plaus = cdl.validate_plausibility(inst, cf_arr)
            imm_ok = all(
                np.isclose(inst[data.feature_names.index(f)],
                           cf_arr[data.feature_names.index(f)], atol=1e-4)
                for f in req.immutable_features if f in data.feature_names
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {**base, "method": "COUNTERFACTUAL", "elapsed_ms": elapsed,
                    "counterfactual_values": {fn: round(float(cf_arr[i]), 4)
                                               for i, fn in enumerate(data.feature_names)},
                    "changes": cf["changes"],
                    "original_pred": data.target_names[cf["original_pred"]],
                    "new_pred": data.target_names[cf["new_pred"]],
                    "success": cf["success"], "n_steps": cf["n_steps"],
                    "metrics": cf["metrics"],
                    "immutable_features": req.immutable_features,
                    "plausibility_check": plaus,
                    "immutable_constraints_satisfied": imm_ok,
                    "compliance_tags": [
                        "EU AI Act Art. 9 – Risk Management (Recourse)",
                        "EU AI Act Art. 15 – Robustness & Accuracy",
                        "GDPR Art. 22 – Right to Contest Automated Decision"],
                    "gdpr_art22": True, "eu_aiact_art9": True}

        raise HTTPException(400, f"Method {req.method} not supported for tabular modality")

    # ── IMAGE (GRAD-CAM) ────────────────────────────────────────────────────
    elif modality == "image":
        import torch
        a = _image()
        data, model = a["data"], a["model"]
        if req.instance_index >= len(data.X_test):
            raise HTTPException(400, "instance_index out of range")

        img_flat = data.X_test[req.instance_index]   # (784,)
        true_idx = int(data.y_test[req.instance_index])
        img_t = torch.tensor(img_flat.reshape(1, 28, 28), dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            logits = model(img_t.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_idx = int(np.argmax(probs))
        conf = round(float(probs[pred_idx]), 4)

        # Grad-CAM
        cam = _gradcam(model, img_t, pred_idx)   # (28, 28) normalised 0-1

        # Top activation regions (coarse 3x3 block analysis on 7×7 feature map)
        cam_low = cam.reshape(7, 4, 7, 4).mean(axis=(1, 3))  # 7x7 → 7x7
        block_names = {
            (0, 3): "Top-centre", (1, 3): "Upper-mid-centre",
            (3, 3): "Centre",     (5, 3): "Lower-mid-centre",
            (6, 3): "Bottom-centre",
            (3, 0): "Centre-left", (3, 6): "Centre-right",
        }
        regions = sorted(
            [{"region": block_names.get((r, c), f"Block ({r},{c})"),
              "importance": round(float(cam_low[r, c]), 4),
              "row": r, "col": c}
             for r in range(7) for c in range(7)],
            key=lambda x: x["importance"], reverse=True
        )[:6]

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "modality": "image",
            "method": "GRAD-CAM",
            "instance_index": req.instance_index,
            "elapsed_ms": elapsed,
            "true_label": data.target_names[true_idx],
            "predicted_label": data.target_names[pred_idx],
            "confidence": conf,
            "class_probabilities": {data.target_names[i]: round(float(p), 4)
                                     for i, p in enumerate(probs)},
            "heatmap": cam.flatten().tolist(),           # 784 floats 0-1
            "original_image": img_flat.tolist(),         # 784 floats 0-1
            "heatmap_regions": regions,
            "compliance_tags": [
                "EU AI Act Art. 13 – Transparency (Visual Inspection)",
                "EU AI Act Art. 9 – Risk Management"],
            "eu_aiact_art13": True,
        }

    # ── TEXT (LIME) ────────────────────────────────────────────────────────
    elif modality == "text":
        a = _text()
        data, model = a["data"], a["model"]
        if req.instance_index >= len(data.texts_test):
            raise HTTPException(400, "instance_index out of range")

        text = data.texts_test[req.instance_index]
        true_idx = int(data.y_test[req.instance_index])
        probs = model.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        conf = round(float(probs[pred_idx]), 4)

        expl = lime_text_explain(model, text,
                                  class_names=data.target_names,
                                  num_features=req.num_features)
        word_weights = dict(expl.as_list())    # {word: float}
        local_pred = {data.target_names[i]: round(float(p), 4)
                      for i, p in enumerate(expl.local_pred)}

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "modality": "text",
            "method": "LIME-TEXT",
            "instance_index": req.instance_index,
            "elapsed_ms": elapsed,
            "text_preview": text[:300],
            "true_label": data.target_names[true_idx],
            "predicted_label": data.target_names[pred_idx],
            "confidence": conf,
            "word_attributions": {w: round(float(v), 6) for w, v in word_weights.items()},
            "local_prediction": local_pred,
            "compliance_tags": [
                "GDPR Art. 22 – Right to Explanation",
                "EU AI Act Art. 13 – Transparency"],
            "gdpr_art22": True, "eu_aiact_art13": True,
        }

    raise HTTPException(400, f"Unknown modality: {req.modality}")


# ===========================================================================
# Simulatability
# ===========================================================================

@app.post("/api/simulatability")
def simulatability(req: SimulatabilityRequest):
    score = simulatability_score(req.accuracy_pre, req.accuracy_post)
    illusion = detect_illusion_of_understanding(score)
    return {
        "accuracy_pre": round(req.accuracy_pre, 4),
        "accuracy_post": round(req.accuracy_post, 4),
        "simulatability_score": round(score, 4),
        "simulatability_pct": round(score * 100, 2),
        "illusion_of_understanding": illusion,
        "audit_gate_status": "REJECT" if illusion else "PASS",
        "interpretation": (
            "⚠️ Illusion of Understanding — explanation did NOT improve human operator accuracy. "
            "Audit Gate REJECTED."
            if illusion else
            f"✅ Causal understanding verified — explanation improved human operator accuracy by "
            f"{round(score * 100, 1)}%."
        ),
        "regulation_note": (
            "EU AI Act Art. 14 — Human Oversight: "
            "Explanation must demonstrably aid correct prediction."
        ),
    }


# ===========================================================================
# Audit
# ===========================================================================

@app.post("/api/audit")
def audit(req: AuditRequest):
    record = generate_audit_record(
        model_type=req.model_type,
        model_accuracy=req.model_accuracy,
        instance_id=req.instance_id,
        true_label=req.true_label,
        predicted_label=req.predicted_label,
        xai_method=req.xai_method,
        feature_attributions=req.feature_attributions,
        confidence=req.confidence,
        illusion_of_understanding=req.illusion_of_understanding,
    )
    d = record.to_dict()
    for k, v in d.items():
        if isinstance(v, (np.integer,)): d[k] = int(v)
        elif isinstance(v, (np.floating,)): d[k] = float(v)
    return {
        "record": d,
        "decision_justified": record.decision_justified,
        "audit_gate_status": "PASS" if record.decision_justified else "REJECT",
        "formatted_text": format_audit_report([record]),
    }


@app.get("/api/compliance/checklist")
def checklist(): return generate_audit_checklist()

@app.get("/api/compliance/risk-categories")
def risk_categories(): return get_risk_categories()

@app.get("/api/compliance/governance-controls")
def gov_controls(): return get_governance_controls()


@app.get("/api/explain/shap-global")
def shap_global(n_samples: int = 30):
    import shap
    a = _tabular()
    data, model = a["data"], a["model"]
    n = min(n_samples, len(data.X_test))
    sv = shap.TreeExplainer(model)(data.X_test[:n])
    raw = sv.values
    mean_abs = np.mean(np.abs(raw), axis=(0, 2)) if raw.ndim == 3 else np.mean(np.abs(raw), axis=0)
    gi = {fn: round(float(mean_abs[i]), 6) for i, fn in enumerate(data.feature_names)}
    ranked = sorted(gi.items(), key=lambda x: x[1], reverse=True)
    return {"method": "SHAP Global (Mean |SHAP|)", "n_samples": n,
            "global_importance": gi,
            "ranked": [{"feature": f, "importance": v} for f, v in ranked],
            "compliance_tags": ["GDPR Art. 13-14 – Transparency of Processing"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8008, reload=True)
