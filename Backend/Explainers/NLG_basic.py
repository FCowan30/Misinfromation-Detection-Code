from __future__ import annotations

from typing import Dict, Any, List, Optional


def _confidence_band(conf: float) -> str:
    if conf >= 0.85:
        return "high"
    if conf >= 0.65:
        return "moderate"
    return "low"


def _describe_token_effect(token: str, impact: float, pred_label: str) -> str:
    token_clean = str(token).strip()

    if impact > 0:
        return (
            f"The token '{token_clean}' strongly pushed the prediction towards {pred_label}. "
            f"This suggests that this word was influential in the model's decision."
        )
    elif impact < 0:
        return (
            f"The token '{token_clean}' pushed the prediction away from {pred_label}. "
            f"This suggests that the word introduced evidence against the final decision."
        )
    else:
        return (
            f"The token '{token_clean}' had very little effect on the final prediction."
        )


def generate_basic_explanation(
    shap_result: Dict[str, Any],
    gradcam_result: Optional[Dict[str, Any]] = None,
    fusion_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a simple human-readable explanation from raw SHAP data.
    This is designed to preserve the original baseline page behaviour.
    """
    prediction = shap_result.get("prediction", {})
    pred_label = prediction.get("label", "UNKNOWN")
    confidence = float(prediction.get("confidence", 0.0))
    conf_band = _confidence_band(confidence)

    top_tokens: List[Dict[str, Any]] = shap_result.get("shap", {}).get("top_tokens", [])
    flags: List[Dict[str, Any]] = shap_result.get("flags", [])

    key_token = top_tokens[0] if top_tokens else None
    token_explanation = None
    if key_token:
        token_explanation = _describe_token_effect(
            key_token.get("token", ""),
            float(key_token.get("impact", 0.0)),
            pred_label,
        )

    if flags:
        main_flag = flags[0]
        rationale = str(main_flag.get("rationale", "")).strip()
        if rationale:
            flag_text = (
                f"A key language pattern detected was '{main_flag.get('name', 'unknown')}'. "
                f"{rationale}"
            )
        else:
            flag_text = (
                f"A key language pattern detected was '{main_flag.get('name', 'unknown')}'."
            )
    else:
        flag_text = "No strong language warning patterns were detected in the text."

    summary_parts = [
        f"The model predicted {pred_label} with {conf_band} confidence ({confidence:.3f})."
    ]

    if token_explanation:
        summary_parts.append(token_explanation)

    summary_parts.append(flag_text)

    # Optional fusion integration
    if fusion_result:
        driver = fusion_result.get("signals", {}).get("driver")
        if driver == "text":
            summary_parts.append(
                "The final multimodal decision was influenced mainly by the text rather than the image."
            )
        elif driver == "image_mismatch":
            summary_parts.append(
                "The final multimodal decision was influenced mainly by mismatch between the image and the text."
            )
        elif driver == "both":
            summary_parts.append(
                "Both the text and the image-text mismatch contributed to the final decision."
            )

    # Optional Grad-CAM integration
    if gradcam_result:
        region_summary = gradcam_result.get("top_region_summary")
        if region_summary:
            summary_parts.append(region_summary)

    summary = " ".join(summary_parts)

    return {
        "summary": summary,
        "token_explanation": token_explanation,
        "flag_text": flag_text,
        "top_tokens": top_tokens,
        "flags": flags,
    }


if __name__ == "__main__":
    sample = {
        "prediction": {"label": "FAKE", "confidence": 0.91},
        "shap": {"top_tokens": [{"token": "proof ", "impact": 0.21}]},
        "flags": [
            {
                "name": "Overconfident claim style",
                "severity": "medium",
                "rationale": "Absolute language can reduce nuance and is sometimes used to persuade rather than inform.",
                "examples": ["proof"],
            }
        ],
    }

    out = generate_basic_explanation(sample)
    print(out["summary"])