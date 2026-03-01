from __future__ import annotations

from typing import Optional, Dict, Any
import os

#existing moduals
from Backend.Models.DistillBERT_Predict import predict_text

try:
    from Backend.Explainers.Shap_DistilBERT import explain_text
except Exception:
    print("[WARN] SHAP explainer not available. Explanation functionality will be disabled.")
    explain_text = None

try:
    from Backend.Models.Fusion import fuse_multimodal
except Exception:
    print("[WARN] Fusion model not available. Multimodal analysis will be disabled.")
    fuse_multimodal = None

def _safe_path_exists(path) -> bool:
    try:
        return bool(path) and os.path.existis(path)
    except Exception:
        return False

def analyze_post(text:str, image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    single entry point for the system
    - text is required
    -image_path is optional
    returns json-serializable dict
    
    frontend/API can call this function directly
    """
    if not text or not text.strip():
        raise ValueError("Text content is required for analysis.")
    
    text = text.strip()
    image_path = image_path.strip() if image_path else None

    #---- Text ONLY ----
    if not image_path: 
        pred = predict_text(text)

        #optional SHAP / NLG explanation
        explanation = None 
        if explain_text is not None:
            try:
                explanation = explain_text(text)
            except Exception as e:
                print(f"[WARN] Explanation failed: {e}")
                explanation = None
        return {
            "mode": "text-only",
            "inputs": {"text": text, "image_path": None},
            "predictions": pred,
            "explanation": explanation,
        }