# Backend/Pipeline/analyze.py
from __future__ import annotations

from typing import Optional, Dict, Any
import os

from Backend.Models.DistillBERT_Predict import predict_text
from Backend.Models.Fusion import fuse_multimodal

# SHAP + NLG explainers
try:
    from Backend.Explainers.Shap_DistilBERT import explain_text
except Exception:
    explain_text = None

try:
    from Backend.Explainers.NLG_basic import generate_basic_explanation
except Exception:
    generate_basic_explanation = None

try:
    from Backend.Explainers.Grad_CAM import generate_gradcam
except Exception:
    generate_gradcam = None


def _safe_path_exists(path: str) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


def _maybe_explain(text: str, explain: bool, top_n: int = 10) -> Optional[Dict[str, Any]]:
    """Return raw SHAP explanation only if explain=True and module is available."""
    if not explain:
        return None
    if explain_text is None:
        return {"error": "Explainability module not available (SHAP import failed)."}
    try:
        return explain_text(text, top_n=top_n)
    except Exception as e:
        return {"error": f"SHAP explain failed: {type(e).__name__}: {e}"}


def _maybe_gradcam(image_path: str, text: str, explain: bool) -> Optional[Dict[str, Any]]:
    """Return raw Grad-CAM explanation only if explain=True and module is available."""
    if not explain:
        return None
    if generate_gradcam is None:
        return {"error": "Grad-CAM module not available (generate_gradcam import failed)."}
    try:
        return generate_gradcam(image_path=image_path, text=text)
    except Exception as e:
        return {"error": f"Grad-CAM failed: {type(e).__name__}: {e}"}


def _build_baseline_explanation(
    text: str,
    shap_result: Optional[Dict[str, Any]],
    fusion_result: Optional[Dict[str, Any]] = None,
    gradcam_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Rebuild the same explanation structure the original page expects:
    {
        "summary": ...,
        "shap": {...},
        "flags": [...],
        "input": {...},
        "gradcam": {...}   # new optional field
    }
    """
    if shap_result is None:
        return None

    if not isinstance(shap_result, dict):
        return shap_result

    if "error" in shap_result:
        return shap_result

    if generate_basic_explanation is None:
        return {
            "error": "NLG module not available (generate_basic_explanation import failed)."
        }

    nlg_result = generate_basic_explanation(
        shap_result=shap_result,
        gradcam_result=gradcam_result,
        fusion_result=fusion_result,
    )

    return {
        "summary": nlg_result.get("summary", ""),
        "shap": shap_result.get("shap", {}),
        "flags": shap_result.get("flags", []),
        "input": shap_result.get("input", {"text": text}),
        "gradcam": gradcam_result,  # optional, frontend can ignore for now
    }


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
        shap_result = _maybe_explain(text, explain=explain, top_n=top_n)

        if isinstance(shap_result, dict) and "prediction" in shap_result:
            prediction = shap_result["prediction"]
            explanation = _build_baseline_explanation(text, shap_result)
        else:
            prediction = predict_text(text)
            explanation = shap_result

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
        shap_result = _maybe_explain(text, explain=explain, top_n=top_n)

        if isinstance(shap_result, dict) and "prediction" in shap_result:
            prediction = shap_result["prediction"]
            explanation = _build_baseline_explanation(text, shap_result)
        else:
            prediction = predict_text(text)
            explanation = shap_result

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
    shap_result = _maybe_explain(text, explain=explain, top_n=top_n)
    gradcam_result = _maybe_gradcam(image_path=image_path, text=text, explain=explain)

    explanation = _build_baseline_explanation(
        text=text,
        shap_result=shap_result,
        fusion_result=fused,
        gradcam_result=gradcam_result,
    )

    return {
        "mode": "multimodal",
        "inputs": {"text": text, "image_path": image_path},
        "prediction": fused.get("final"),
        "fusion": fused,
        "explanation": explanation,
    }