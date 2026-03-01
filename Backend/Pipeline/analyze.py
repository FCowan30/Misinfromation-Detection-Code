# Backend/analyze.py
from __future__ import annotations

from typing import Optional, Dict, Any
import os

from Backend.Models.DistillBERT_Predict import predict_text

_explainer_import_error = None
try:
    from Backend.Explainers.Shap_DistilBERT import explain_text
except Exception as e:
    explain_text = None
    _explainer_import_error = f"{type(e).__name__}: {e}"

_fusion_import_error = None
try:
    from Backend.Models.Fusion import fuse_multimodal
except Exception as e:
    fuse_multimodal = None
    _fusion_import_error = f"{type(e).__name__}: {e}"

def _get_explanation(text: str) -> Dict[str, Any]:
    if explain_text is None:
        return {"error": f"Explainability import failed: {_explainer_import_error}"}
    try:
        return explain_text(text, top_n=10)
    except Exception as e:
        return {"error": f"SHAP explain failed: {type(e).__name__}: {e}"}

def _safe_path_exists(path: str) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


def analyze_post(text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Single entry point for your system.
    - text is required
    - image_path is optional
    Returns a JSON-serializable dict.

    Frontend/API can call this function directly later.
    """
    if not text or not text.strip():
        raise ValueError("Text input is required.")

    text = text.strip()
    image_path = image_path.strip() if image_path else None

    # ---- TEXT ONLY ----
    if not image_path:
        explanation = _get_explanation(text)

    # If SHAP explanation succeeded, it already contains prediction
        if isinstance(explanation, dict) and "prediction" in explanation:
            prediction = explanation["prediction"]
        else:
            prediction = predict_text(text)

        return {
            "mode": "text_only",
            "inputs": {"text": text, "image_path": None},
            "prediction": prediction,
            "explanation": explanation,
        }
    
    # ---- TEXT + IMAGE ----
    if not _safe_path_exists(image_path):
        # If an image was supplied but path is wrong, fallback to text-only (or raise).
        # For now: return a clear error + still provide text prediction.
        pred = predict_text(text)
        return {
            "mode": "text_only_fallback",
            "inputs": {"text": text, "image_path": image_path},
            "warning": "Image path provided but file does not exist. Fell back to text-only analysis.",
            "prediction": pred,
            "explanation": None,
        }

    if fuse_multimodal is None:
        raise RuntimeError(f"Fusion module not available: {_fusion_import_error}")

    fused = fuse_multimodal(text, image_path)
    explanation = _get_explanation(text)

    return {
        "mode": "multimodal",
        "inputs": {"text": text, "image_path": image_path},
        "fusion": fused,
        "explanation": explanation,
    }