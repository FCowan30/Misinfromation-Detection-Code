# Backend/analyze.py
from __future__ import annotations

from typing import Optional, Dict, Any
import os

from Backend.Models.DistillBERT_Predict import predict_text
from Backend.Models.Fusion import fuse_multimodal

# SHAP explainer
try:
    from Backend.Explainers.Shap_DistilBERT import explain_text
except Exception:
    explain_text = None


def _safe_path_exists(path: str) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


def _maybe_explain(text: str, explain: bool, top_n: int = 10) -> Optional[Dict[str, Any]]:
    """Return SHAP explanation only if explain=True and module is available."""
    if not explain:
        return None
    if explain_text is None:
        return {"error": "Explainability module not available (shap not installed or import failed)."}
    try:
        return explain_text(text, top_n=top_n)
    except Exception as e:
        return {"error": f"SHAP explain failed: {type(e).__name__}: {e}"}


def analyze_post(
    text: str,
    image_path: Optional[str] = None,
    explain: bool = False,
    top_n: int = 10,
) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Text input is required.")

    text = text.strip()
    image_path = image_path.strip() if image_path else None

    # -------------------------
    # TEXT ONLY MODE
    # -------------------------
    if not image_path:
        explanation = _maybe_explain(text, explain=explain, top_n=top_n)

        if isinstance(explanation, dict) and "prediction" in explanation:
            prediction = explanation["prediction"]
            explanation = {k: v for k, v in explanation.items() if k != "prediction"}
        else:
            prediction = predict_text(text)

        return {
            "mode": "text_only",
            "inputs": {"text": text, "image_path": None},
            "prediction": prediction,
            "explanation": explanation,
        }

    # -------------------------
    # IMAGE PATH INVALID
    # -------------------------
    if not _safe_path_exists(image_path):
        explanation = _maybe_explain(text, explain=explain, top_n=top_n)

        if isinstance(explanation, dict) and "prediction" in explanation:
            prediction = explanation["prediction"]
            explanation = {k: v for k, v in explanation.items() if k != "prediction"}
        else:
            prediction = predict_text(text)

        return {
            "mode": "text_only_fallback",
            "inputs": {"text": text, "image_path": image_path},
            "warning": "Image path does not exist. Fell back to text-only analysis.",
            "prediction": prediction,
            "explanation": explanation,
        }

    # -------------------------
    # MULTIMODAL MODE
    # -------------------------
    fused = fuse_multimodal(text, image_path)
    explanation = _maybe_explain(text, explain=explain, top_n=top_n)

    if isinstance(explanation, dict) and "prediction" in explanation:
        explanation = {k: v for k, v in explanation.items() if k != "prediction"}

    return {
        "mode": "multimodal",
        "inputs": {"text": text, "image_path": image_path},
        "prediction": fused.get("final"),   # recommended for frontend consistency
        "fusion": fused,
        "explanation": explanation,
    }