from __future__ import annotations

from typing import Optional, Dict, Any
import os
import sys

from Backend.Models.DistillBERT_Predict import predict_text
from Backend.Models.Fusion import fuse_multimodal
from Backend.Models.CLIP_predict import clip_similarity
from Backend.Models.claim_router import detect_claim_type, decide_descriptive_visual_result

# SHAP + NLG explainers
try:
    from Backend.Explainers.Shap_DistilBERT import explain_text
except Exception as e:
    print(f"[SHAP IMPORT ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    explain_text = None

try:
    from Backend.Explainers.NLG_basic import generate_basic_explanation
except Exception as e:
    print(f"[NLG BASIC IMPORT ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    generate_basic_explanation = None

try:
    from Backend.Explainers.NLG_detailed import generate_detailed_explanation
except Exception as e:
    print(f"[NLG DETAILED IMPORT ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    generate_detailed_explanation = None

try:
    from Backend.Explainers.Grad_CAM import generate_gradcam
except Exception as e:
    print(f"[GRADCAM IMPORT ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    generate_gradcam = None


def _safe_path_exists(path: str) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


def _clip_raw_to_01(raw_similarity: float) -> float:
    """
    Convert CLIP cosine similarity from approx [-1, 1] into [0, 1].
    """
    sim_01 = (float(raw_similarity) + 1.0) / 2.0
    return max(0.0, min(1.0, sim_01))


def _run_clip_descriptive_similarity(text: str, image_path: str) -> Dict[str, Any]:
    """
    Standalone CLIP route for short descriptive visual claims.

    Notes:
    - `clip_similarity()` returns a cosine similarity score.
    - We convert it to [0, 1] for easier interpretation.
    - `p_mismatch` here is a simple derived proxy: 1 - similarity_01.
      This is not a separately trained mismatch model.
    """
    raw_similarity = clip_similarity(image_path=image_path, text=text)
    similarity_01 = _clip_raw_to_01(raw_similarity)
    p_mismatch = 1.0 - similarity_01

    return {
        "raw_similarity": float(raw_similarity),
        "similarity_01": round(similarity_01, 4),
        "p_mismatch": round(p_mismatch, 4),
    }


def _maybe_explain(text: str, explain: bool, top_n: int = 10) -> Optional[Dict[str, Any]]:
    if not explain:
        return None
    if explain_text is None:
        print("[SHAP ERROR] explain_text import failed.", file=sys.stderr)
        return {"error": "Explainability module not available (SHAP import failed)."}
    try:
        return explain_text(text, top_n=top_n)
    except Exception as e:
        print(f"[SHAP ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return {"error": f"SHAP explain failed: {type(e).__name__}: {e}"}


def _maybe_gradcam(image_path: str, text: str, explain: bool) -> Optional[Dict[str, Any]]:
    if not explain:
        return None

    if generate_gradcam is None:
        print("[GRADCAM ERROR] generate_gradcam import failed.", file=sys.stderr)
        return {"error": "Grad-CAM module not available (generate_gradcam import failed)."}

    try:
        result = generate_gradcam(image_path=image_path, text=text)
        print(f"[GRADCAM SUCCESS] Result: {result}", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[GRADCAM ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return {"error": f"Grad-CAM failed: {type(e).__name__}: {e}"}


def _build_basic_explanation(
    text: str,
    shap_result: Optional[Dict[str, Any]],
    fusion_result: Optional[Dict[str, Any]] = None,
    gradcam_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if shap_result is None:
        return None

    if not isinstance(shap_result, dict):
        return shap_result

    if "error" in shap_result:
        return shap_result

    if generate_basic_explanation is None:
        return {"error": "Basic NLG module not available."}

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
        "gradcam": gradcam_result,
    }


def _build_detailed_explanation(
    prediction: Dict[str, Any],
    text: str,
    shap_result: Optional[Dict[str, Any]],
    fusion_result: Optional[Dict[str, Any]] = None,
    gradcam_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not shap_result and not fusion_result:
        return None

    if isinstance(shap_result, dict) and "error" in shap_result:
        return shap_result

    if generate_detailed_explanation is None:
        return {"error": "Detailed NLG module not available."}

    detailed = generate_detailed_explanation(
        prediction=prediction,
        shap_result=shap_result,
        fusion_result=fusion_result,
        gradcam_result=gradcam_result,
    )

    return {
        "summary": detailed.get("summary", ""),
        "prediction_text": detailed.get("prediction_text", ""),
        "risk_level": detailed.get("risk_level", ""),
        "analysis_route_text": detailed.get("analysis_route_text", ""),
        "evidence_for": detailed.get("evidence_for", ""),
        "evidence_against": detailed.get("evidence_against", ""),
        "top_reasons": detailed.get("top_reasons", []),
        "decision_pathway": detailed.get("decision_pathway", []),
        "confidence_meaning_text": detailed.get("confidence_meaning_text", ""),
        "uncertainty_text": detailed.get("uncertainty_text", ""),
        "limitations_text": detailed.get("limitations_text", ""),
        "flag_text": detailed.get("flag_text", ""),
        "similarity_text": detailed.get("similarity_text", ""),
        "mismatch_text": detailed.get("mismatch_text", ""),
        "driver_text": detailed.get("driver_text", ""),
        "visual_text": detailed.get("visual_text", ""),
        "scores": detailed.get("scores", {}),
        "shap": shap_result.get("shap", {}) if shap_result else {},
        "flags": shap_result.get("flags", []) if shap_result else [],
        "input": shap_result.get("input", {"text": text}) if shap_result else {"text": text},
        "gradcam": gradcam_result,
    }


def _build_descriptive_route_outputs(
    text: str,
    image_path: str,
    explain: bool,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Special route for short descriptive visual claims such as:
    - 'this is a dog'
    - 'the image shows a pizza'
    """
    clip_result = _run_clip_descriptive_similarity(text=text, image_path=image_path)

    prediction = decide_descriptive_visual_result(
        similarity_01=clip_result.get("similarity_01", 0.0),
        p_mismatch=clip_result.get("p_mismatch"),
    )

    # Add extra fields for downstream explanation / UI
    prediction["raw_similarity"] = clip_result.get("raw_similarity")
    prediction["text_confidence"] = None

    # Create a fusion-like structure so the existing NLG code can still read it
    fusion_result = {
        "clip_model": {
            "raw_similarity": clip_result.get("raw_similarity"),
            "similarity_01": clip_result.get("similarity_01"),
            "p_mismatch": clip_result.get("p_mismatch"),
        },
        "signals": {
            "driver": "image_mismatch" if (clip_result.get("p_mismatch") or 0.0) >= 0.50 else "both"
        },
        "analysis_route": prediction.get("analysis_route"),
        "route_reason": prediction.get("route_reason"),
        "used_text_model": False,
        "used_clip_model": True,
        "final": prediction,
    }

    gradcam_result = _maybe_gradcam(image_path=image_path, text=text, explain=explain)

    explanation = None
    explanation_detailed = None

    # No SHAP here because the text model is intentionally skipped
    shap_result = None

    if explain:
        explanation_detailed = _build_detailed_explanation(
            prediction=prediction,
            text=text,
            shap_result=shap_result,
            fusion_result=fusion_result,
            gradcam_result=gradcam_result,
        )

    return {
        "mode": "multimodal_descriptive",
        "inputs": {"text": text, "image_path": image_path},
        "prediction": prediction,
        "fusion": fusion_result,
        "routing": {
            "claim_type": "descriptive_visual",
            "route": "clip_descriptive",
            "reason": prediction.get("route_reason"),
        },
        "explanation": explanation,
        "explanation_detailed": explanation_detailed,
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

    # ----------------------------------------
    # TEXT ONLY
    # ----------------------------------------
    if not image_path:
        shap_result = _maybe_explain(text, explain=explain, top_n=top_n)

        if isinstance(shap_result, dict) and "prediction" in shap_result:
            prediction = shap_result["prediction"]
            prediction["analysis_route"] = "text_only"
            prediction["route_reason"] = "No image was provided, so text-only analysis was used."

            explanation = _build_basic_explanation(text, shap_result)
            explanation_detailed = _build_detailed_explanation(
                prediction=prediction,
                text=text,
                shap_result=shap_result,
            )
        else:
            prediction = predict_text(text)
            prediction["analysis_route"] = "text_only"
            prediction["route_reason"] = "No image was provided, so text-only analysis was used."
            explanation = shap_result
            explanation_detailed = None

        return {
            "mode": "text_only",
            "inputs": {"text": text, "image_path": None},
            "prediction": prediction,
            "explanation": explanation,
            "explanation_detailed": explanation_detailed,
        }

    # ----------------------------------------
    # INVALID IMAGE PATH
    # ----------------------------------------
    if not _safe_path_exists(image_path):
        shap_result = _maybe_explain(text, explain=explain, top_n=top_n)

        if isinstance(shap_result, dict) and "prediction" in shap_result:
            prediction = shap_result["prediction"]
            prediction["analysis_route"] = "text_only_fallback"
            prediction["route_reason"] = "The image path was invalid, so the system fell back to text-only analysis."

            explanation = _build_basic_explanation(text, shap_result)
            explanation_detailed = _build_detailed_explanation(
                prediction=prediction,
                text=text,
                shap_result=shap_result,
            )
        else:
            prediction = predict_text(text)
            prediction["analysis_route"] = "text_only_fallback"
            prediction["route_reason"] = "The image path was invalid, so the system fell back to text-only analysis."
            explanation = shap_result
            explanation_detailed = None

        return {
            "mode": "text_only_fallback",
            "inputs": {"text": text, "image_path": image_path},
            "warning": "Image path does not exist. Fell back to text-only analysis.",
            "prediction": prediction,
            "explanation": explanation,
            "explanation_detailed": explanation_detailed,
        }

    # ----------------------------------------
    # MULTIMODAL ROUTING
    # ----------------------------------------
    routing_info = detect_claim_type(text)

    # Route 1: descriptive visual claim -> skip DistilBERT / skip full fusion
    if routing_info.get("route") == "clip_descriptive":
        return _build_descriptive_route_outputs(
            text=text,
            image_path=image_path,
            explain=explain,
            top_n=top_n,
        )

    # ----------------------------------------
    # Route 2: full multimodal claim
    # ----------------------------------------
    fused = fuse_multimodal(text, image_path)
    shap_result = _maybe_explain(text, explain=explain, top_n=top_n)
    gradcam_result = _maybe_gradcam(image_path=image_path, text=text, explain=explain)

    final_prediction = dict(fused.get("final", {}) or {})
    final_prediction["analysis_route"] = "full_multimodal"
    final_prediction["route_reason"] = routing_info.get(
        "reason",
        "The statement appeared to be a broader multimodal claim, so full fusion was used."
    )
    final_prediction["used_text_model"] = True
    final_prediction["used_clip_model"] = True

    explanation = _build_basic_explanation(
        text=text,
        shap_result=shap_result,
        fusion_result=fused,
        gradcam_result=gradcam_result,
    )

    explanation_detailed = _build_detailed_explanation(
        prediction=final_prediction,
        text=text,
        shap_result=shap_result,
        fusion_result=fused,
        gradcam_result=gradcam_result,
    )

    return {
        "mode": "multimodal",
        "inputs": {"text": text, "image_path": image_path},
        "prediction": final_prediction,
        "fusion": fused,
        "routing": routing_info,
        "explanation": explanation,
        "explanation_detailed": explanation_detailed,
    }