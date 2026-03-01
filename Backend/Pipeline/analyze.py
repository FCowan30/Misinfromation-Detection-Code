# Backend/analyze.py
from __future__ import annotations

from typing import Optional, Dict, Any
import os

# Existing modules you already have
from Backend.Models.DistillBERT_Predict import predict_text

# If you already have a SHAP/NLG function, import it here.
# Rename these imports to match your actual file/function names.
try:
    from Backend.Explainers.Shap_DistilBERT import explain_text  # <-- adjust if needed
except Exception:
    explain_text = None

# Fusion is only needed when image is provided
try:
    from Backend.Models.Fusion import fuse_multimodal
except Exception:
    fuse_multimodal = None


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
        pred = predict_text(text)

        # Optional SHAP / NLG explanation
        explanation = None
        if explain_text is not None:
            try:
                explanation = explain_text(text)  # adjust signature if needed
            except Exception as e:
                explanation = {"error": f"SHAP explain failed: {e}"}

        return {
            "mode": "text_only",
            "inputs": {"text": text, "image_path": None},
            "prediction": pred,
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
        raise RuntimeError("Fusion module not available. Ensure Backend/Models/Fusion.py exists and imports correctly.")

    fused = fuse_multimodal(text, image_path)

    # Optional: if you want SHAP explanation even in multimodal mode, keep it:
    explanation = None
    if explain_text is not None:
        try:
            explanation = explain_text(text)  # text explanation still valid
        except Exception as e:
            explanation = {"error": f"SHAP explain failed: {e}"}

    return {
        "mode": "multimodal",
        "inputs": {"text": text, "image_path": image_path},
        "fusion": fused,              # includes text_model + clip_model + final decision
        "explanation": explanation,   # SHAP/NLG for text (later extend to fusion-aware NLG)
    }