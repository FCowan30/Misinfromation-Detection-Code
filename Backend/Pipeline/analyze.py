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