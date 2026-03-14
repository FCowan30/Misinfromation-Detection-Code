from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_token(token: str) -> str:
    return str(token).strip().replace("Ġ", "").replace("##", "")


def _join_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _to_percent(value: Optional[float], decimals: int = 0) -> Optional[float]:
    if value is None:
        return None
    return round(_safe_float(value) * 100, decimals)


def _format_percent(value: Optional[float], decimals: int = 0) -> str:
    if value is None:
        return "N/A"
    pct = _to_percent(value, decimals)
    if pct is None:
        return "N/A"
    if decimals == 0:
        return f"{int(pct)}%"
    return f"{pct:.{decimals}f}%"


# -----------------------------
# Confidence / risk banding
# -----------------------------
def get_confidence_band(conf: float) -> str:
    if conf >= 0.90:
        return "very high"
    elif conf >= 0.80:
        return "high"
    elif conf >= 0.65:
        return "moderate"
    elif conf >= 0.50:
        return "limited"
    return "low"


def get_risk_level(pred_label: str, confidence: float, p_mismatch: Optional[float]) -> str:
    pred = pred_label.upper()
    mismatch = _safe_float(p_mismatch) if p_mismatch is not None else None

    if pred == "FAKE":
        if confidence >= 0.85:
            return "High misinformation risk"
        elif confidence >= 0.65:
            return "Moderate misinformation risk"
        return "Possible misinformation risk"

    if pred == "REAL":
        if mismatch is not None and mismatch >= 0.60:
            return "Mixed evidence"
        elif confidence >= 0.80:
            return "Lower misinformation risk"
        return "Uncertain reliability"

    return "Uncertain result"


def describe_confidence(conf: float, pred_label: str) -> Tuple[str, str, str]:
    band = get_confidence_band(conf)
    conf_pct = _format_percent(conf)

    main = (
        f"The system predicts that this content is {pred_label} with {band} confidence "
        f"({conf_pct})."
    )

    meaning = (
        f"This {conf_pct} confidence score reflects how strongly the model supported the "
        f"{pred_label} label compared with the alternative label. It does not mean there is a "
        f"{conf_pct} real-world probability that the claim is {pred_label}."
    )

    if conf >= 0.90:
        caution = (
            "This is a strong model prediction, but it should still be treated as an automated assessment rather than proof."
        )
    elif conf >= 0.80:
        caution = (
            "The system shows a clear preference for this outcome, although some uncertainty remains."
        )
    elif conf >= 0.65:
        caution = (
            "The result is reasonably supported, but it is not fully decisive and should be interpreted with care."
        )
    elif conf >= 0.50:
        caution = (
            "The system only slightly favours this outcome, so the result should be treated cautiously."
        )
    else:
        caution = (
            "The system is uncertain about this outcome, so the result should be treated as weak evidence."
        )

    return main, meaning, caution


# -----------------------------
# Token handling
# -----------------------------
def split_token_impacts(top_tokens: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positive = []
    negative = []

    for item in top_tokens:
        token = _clean_token(item.get("token", ""))
        impact = _safe_float(item.get("impact", 0.0))
        if not token:
            continue

        enriched = {
            "token": token,
            "impact": impact,
            "impact_pct": round(abs(impact) * 100, 1)
        }

        if impact > 0:
            positive.append(enriched)
        elif impact < 0:
            negative.append(enriched)

    positive.sort(key=lambda x: x["impact"], reverse=True)
    negative.sort(key=lambda x: x["impact"])  # most negative first
    return positive, negative


def build_evidence_for(top_tokens: List[Dict[str, Any]], flags: List[Dict[str, Any]],
                       pred_label: str, similarity_01: Optional[float],
                       p_mismatch: Optional[float], driver: Optional[str]) -> str:
    positive, _ = split_token_impacts(top_tokens)
    reasons = []

    if positive:
        top_pos = [f"'{x['token']}'" for x in positive[:3]]
        reasons.append(
            f"Textual evidence supporting the {pred_label} prediction included {_join_list(top_pos)}."
        )

    if flags:
        top_flag = flags[0]
        flag_name = top_flag.get("name", "language pattern")
        reasons.append(
            f"The system also detected the language pattern '{flag_name}', which increased suspicion in the text analysis."
        )

    if similarity_01 is not None and p_mismatch is not None:
        sim = _safe_float(similarity_01)
        mismatch = _safe_float(p_mismatch)

        if pred_label.upper() == "FAKE":
            if mismatch >= 0.50:
                reasons.append(
                    f"The image-text mismatch score was {_format_percent(mismatch)}, which supported the misleading-content prediction."
                )
            elif sim < 0.35:
                reasons.append(
                    f"The image-text similarity score was only {_format_percent(sim)}, suggesting weak support between the image and the claim."
                )
        elif pred_label.upper() == "REAL":
            if sim >= 0.65 and mismatch < 0.35:
                reasons.append(
                    f"The image-text relationship appeared broadly consistent, with {_format_percent(sim)} similarity and {_format_percent(mismatch)} mismatch."
                )

    if driver == "text":
        reasons.append("The final decision was influenced mainly by the wording of the claim.")
    elif driver == "image_mismatch":
        reasons.append("The final decision was influenced mainly by inconsistency between the image and the text.")
    elif driver == "both":
        reasons.append("The final decision was supported by both textual and image-text evidence.")

    if not reasons:
        return "The system did not identify strong supporting evidence for the final prediction."

    return " ".join(reasons)


def build_evidence_against(top_tokens: List[Dict[str, Any]], pred_label: str,
                           similarity_01: Optional[float], p_mismatch: Optional[float]) -> str:
    _, negative = split_token_impacts(top_tokens)
    reasons = []

    if negative:
        top_neg = [f"'{x['token']}'" for x in negative[:2]]
        reasons.append(
            f"Some counter-evidence was also present in the text, particularly {_join_list(top_neg)}, which pushed against the {pred_label} prediction."
        )

    if similarity_01 is not None and p_mismatch is not None:
        sim = _safe_float(similarity_01)
        mismatch = _safe_float(p_mismatch)

        if pred_label.upper() == "FAKE":
            if sim >= 0.60 and mismatch < 0.40:
                reasons.append(
                    f"The image still showed {_format_percent(sim)} similarity with the text and only {_format_percent(mismatch)} mismatch, which weakens a strong contradiction-based explanation."
                )
        elif pred_label.upper() == "REAL":
            if mismatch >= 0.50:
                reasons.append(
                    f"The mismatch score was {_format_percent(mismatch)}, which introduces some concern about whether the image fully supports the claim."
                )
            elif sim < 0.45:
                reasons.append(
                    f"The image-text similarity score was only {_format_percent(sim)}, so the visual evidence did not strongly reinforce the claim."
                )

    if not reasons:
        return "The system found little clear counter-evidence against the final prediction."

    return " ".join(reasons)


# -----------------------------
# Flags
# -----------------------------
def describe_flags(flags: List[Dict[str, Any]]) -> str:
    if not flags:
        return "No strong language warning patterns were detected in the text."

    parts = []
    for flag in flags[:2]:
        name = str(flag.get("name", "unknown pattern")).strip()
        severity = str(flag.get("severity", "")).strip().lower()
        rationale = str(flag.get("rationale", "")).strip()

        prefix = f"The system detected the language pattern '{name}'"
        if severity:
            prefix += f" with {severity} severity"

        if rationale:
            parts.append(f"{prefix}. {rationale}")
        else:
            parts.append(f"{prefix}.")

    return " ".join(parts)


# -----------------------------
# Similarity / mismatch
# -----------------------------
def describe_similarity_and_mismatch(
    similarity_01: Optional[float],
    p_mismatch: Optional[float]
) -> Tuple[str, str]:
    if similarity_01 is None and p_mismatch is None:
        return (
            "The system could not evaluate the image-text relationship.",
            "No cross-modal evidence was available."
        )

    similarity_text = ""
    mismatch_text = ""

    if similarity_01 is not None:
        sim = _safe_float(similarity_01)
        sim_pct = _format_percent(sim)

        if sim >= 0.75:
            similarity_text = (
                f"The image and text show strong similarity ({sim_pct}), suggesting the visual content closely matches the written claim."
            )
        elif sim >= 0.50:
            similarity_text = (
                f"The image and text show moderate similarity ({sim_pct}), suggesting the image is broadly related to the claim."
            )
        elif sim >= 0.25:
            similarity_text = (
                f"The image and text show weak similarity ({sim_pct}), suggesting only limited support between the image and the claim."
            )
        else:
            similarity_text = (
                f"The image and text show very low similarity ({sim_pct}), suggesting the image may be unrelated or weakly connected to the claim."
            )

    if p_mismatch is not None:
        mismatch = _safe_float(p_mismatch)
        mismatch_pct = _format_percent(mismatch)

        if mismatch >= 0.75:
            mismatch_text = (
                f"The mismatch score is high ({mismatch_pct}), which suggests strong conflict between the image and the text."
            )
        elif mismatch >= 0.50:
            mismatch_text = (
                f"The mismatch score is moderate ({mismatch_pct}), suggesting the image may not fully support the text."
            )
        elif mismatch >= 0.25:
            mismatch_text = (
                f"The mismatch score is fairly low ({mismatch_pct}), so only limited image-text conflict was detected."
            )
        else:
            mismatch_text = (
                f"The mismatch score is very low ({mismatch_pct}), indicating little evidence of image-text conflict."
            )

    return similarity_text, mismatch_text


# -----------------------------
# Driver
# -----------------------------
def describe_driver(driver: Optional[str], pred_label: str) -> str:
    if not driver:
        return "The system could not determine which source of evidence most influenced the final decision."

    if driver == "text":
        return f"The final {pred_label} decision was influenced mainly by the wording of the text."
    elif driver == "image_mismatch":
        return f"The final {pred_label} decision was influenced mainly by mismatch between the image and the text."
    elif driver == "both":
        return f"The final {pred_label} decision was influenced by both the wording of the claim and the image-text relationship."
    else:
        return "The final decision was based on combined multimodal evidence."


# -----------------------------
# Visual explanation
# -----------------------------
def describe_visual_region(gradcam_result: Optional[Dict[str, Any]]) -> str:
    if not gradcam_result:
        return "No visual explanation was available for this example."

    if gradcam_result.get("error"):
        return "The system could not generate a visual region explanation for this example."

    summary = str(gradcam_result.get("top_region_summary", "")).strip()
    if summary:
        return summary

    return "A visual explanation was generated, but no region summary was available."


# -----------------------------
# Decision pathway
# -----------------------------
def build_decision_pathway(
    has_text: bool,
    has_fusion: bool,
    has_visual: bool,
    driver: Optional[str]
) -> List[str]:
    pathway = []

    if has_text:
        pathway.append("Step 1: The text was analysed to identify wording patterns and influential words or phrases.")
    else:
        pathway.append("Step 1: No detailed text explanation was available.")

    if has_fusion:
        pathway.append("Step 2: The image and text were compared to measure semantic similarity and possible mismatch.")
    else:
        pathway.append("Step 2: No image-text comparison was available.")

    if has_visual:
        pathway.append("Step 3: A visual explanation was generated to show which image regions were most relevant.")
    else:
        pathway.append("Step 3: No visual region explanation was available.")

    if driver == "text":
        pathway.append("Step 4: The final decision relied mostly on textual evidence.")
    elif driver == "image_mismatch":
        pathway.append("Step 4: The final decision relied mostly on inconsistency between the image and text.")
    elif driver == "both":
        pathway.append("Step 4: The final decision combined both textual and image-text evidence.")
    else:
        pathway.append("Step 4: The final decision was produced from the available model outputs.")

    return pathway


# -----------------------------
# Top reasons
# -----------------------------
def build_top_reasons(
    pred_label: str,
    top_tokens: List[Dict[str, Any]],
    flags: List[Dict[str, Any]],
    similarity_01: Optional[float],
    p_mismatch: Optional[float],
    driver: Optional[str]
) -> List[str]:
    reasons = []

    positive, _ = split_token_impacts(top_tokens)
    if positive:
        top_pos = [f"'{x['token']}'" for x in positive[:2]]
        reasons.append(
            f"Key wording such as {_join_list(top_pos)} strongly influenced the {pred_label} prediction."
        )

    if flags:
        reasons.append(
            f"The text triggered the warning pattern '{flags[0].get('name', 'language pattern')}'."
        )

    if similarity_01 is not None and p_mismatch is not None:
        sim = _safe_float(similarity_01)
        mismatch = _safe_float(p_mismatch)

        if mismatch >= 0.50:
            reasons.append(
                f"The image-text mismatch score was {_format_percent(mismatch)}, indicating noticeable conflict."
            )
        elif sim >= 0.60:
            reasons.append(
                f"The image-text similarity score was {_format_percent(sim)}, indicating moderate or stronger alignment."
            )
        else:
            reasons.append(
                f"The image provided only limited support for the claim, with {_format_percent(sim)} similarity."
            )

    if driver == "text":
        reasons.append("The final decision was mainly driven by textual evidence.")
    elif driver == "image_mismatch":
        reasons.append("The final decision was mainly driven by image-text mismatch.")
    elif driver == "both":
        reasons.append("The final decision was shaped by both text and image-text evidence.")

    return reasons[:3]


# -----------------------------
# Limitations
# -----------------------------
def build_limitations_statement(
    has_tokens: bool,
    has_fusion: bool,
    has_visual: bool
) -> str:
    parts = [
        "This explanation describes how the model reached its decision, not verified truth."
    ]

    parts.append(
        "A high confidence score means the model preferred one label over the other, but it does not guarantee that the prediction is correct."
    )

    if not has_tokens:
        parts.append("No strong token-level explanation was available, so the text reasoning may be incomplete.")
    if not has_fusion:
        parts.append("No image-text comparison was available, so the explanation relied less on multimodal evidence.")
    if not has_visual:
        parts.append("No visual region explanation was available, so the system could not show which parts of the image influenced the result.")

    return " ".join(parts)


# -----------------------------
# Score extraction
# -----------------------------
def extract_scores(
    prediction: Dict[str, Any],
    fusion_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    final_conf = _safe_float(prediction.get("confidence", 0.0))

    text_conf = None
    if "text_confidence" in prediction:
        text_conf = _safe_float(prediction.get("text_confidence"))
    elif fusion_result and "text_model" in fusion_result:
        text_conf = _safe_float(fusion_result["text_model"].get("confidence"))

    similarity = None
    mismatch = None
    if fusion_result:
        similarity = fusion_result.get("clip_model", {}).get("similarity_01")
        mismatch = fusion_result.get("clip_model", {}).get("p_mismatch")

    return {
        "text_confidence_raw": text_conf,
        "text_confidence_pct": _to_percent(text_conf) if text_conf is not None else None,
        "similarity_raw": _safe_float(similarity) if similarity is not None else None,
        "similarity_pct": _to_percent(similarity) if similarity is not None else None,
        "mismatch_raw": _safe_float(mismatch) if mismatch is not None else None,
        "mismatch_pct": _to_percent(mismatch) if mismatch is not None else None,
        "final_confidence_raw": final_conf,
        "final_confidence_pct": _to_percent(final_conf),
    }


# -----------------------------
# Main explanation builder
# -----------------------------
def generate_detailed_explanation(
    prediction: Dict[str, Any],
    shap_result: Optional[Dict[str, Any]] = None,
    fusion_result: Optional[Dict[str, Any]] = None,
    gradcam_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pred_label = str(prediction.get("label", "UNKNOWN"))
    confidence = _safe_float(prediction.get("confidence", 0.0))

    top_tokens = []
    flags = []
    if shap_result:
        top_tokens = shap_result.get("shap", {}).get("top_tokens", []) or []
        flags = shap_result.get("flags", []) or []

    similarity_01 = None
    p_mismatch = None
    driver = None
    if fusion_result:
        similarity_01 = fusion_result.get("clip_model", {}).get("similarity_01")
        p_mismatch = fusion_result.get("clip_model", {}).get("p_mismatch")
        driver = fusion_result.get("signals", {}).get("driver")

    prediction_text, confidence_meaning_text, uncertainty_text = describe_confidence(confidence, pred_label)
    risk_level = get_risk_level(pred_label, confidence, p_mismatch)

    evidence_for_text = build_evidence_for(
        top_tokens=top_tokens,
        flags=flags,
        pred_label=pred_label,
        similarity_01=similarity_01,
        p_mismatch=p_mismatch,
        driver=driver
    )

    evidence_against_text = build_evidence_against(
        top_tokens=top_tokens,
        pred_label=pred_label,
        similarity_01=similarity_01,
        p_mismatch=p_mismatch
    )

    flag_text = describe_flags(flags)

    similarity_text = None
    mismatch_text = None
    if fusion_result:
        similarity_text, mismatch_text = describe_similarity_and_mismatch(similarity_01, p_mismatch)

    driver_text = describe_driver(driver, pred_label) if fusion_result else None
    visual_text = describe_visual_region(gradcam_result) if gradcam_result else None

    decision_pathway = build_decision_pathway(
        has_text=bool(shap_result),
        has_fusion=bool(fusion_result),
        has_visual=bool(gradcam_result),
        driver=driver
    )

    top_reasons = build_top_reasons(
        pred_label=pred_label,
        top_tokens=top_tokens,
        flags=flags,
        similarity_01=similarity_01,
        p_mismatch=p_mismatch,
        driver=driver
    )
    limitations_text = build_limitations_statement(
        has_tokens=bool(top_tokens),
        has_fusion=bool(fusion_result),
        has_visual=bool(gradcam_result)
    )
    scores = extract_scores(prediction, fusion_result)
    summary_parts = [
        prediction_text,
        f"Risk level: {risk_level}.",
        evidence_for_text,
        evidence_against_text,
        flag_text,
        similarity_text,
        mismatch_text,
        driver_text,
        visual_text,
        confidence_meaning_text,
        uncertainty_text,
        limitations_text,
    ]
    final_summary = " ".join(part for part in summary_parts if part)
    return {
        "summary": final_summary,

        "prediction_text": prediction_text,
        "risk_level": risk_level,

        "evidence_for": evidence_for_text,
        "evidence_against": evidence_against_text,

        "top_reasons": top_reasons,
        "decision_pathway": decision_pathway,

        "confidence_meaning_text": confidence_meaning_text,
        "uncertainty_text": uncertainty_text,
        "limitations_text": limitations_text,

        "flag_text": flag_text,
        "similarity_text": similarity_text,
        "mismatch_text": mismatch_text,
        "driver_text": driver_text,
        "visual_text": visual_text,

        "scores": scores,

        "top_tokens": top_tokens,
        "flags": flags,
        "gradcam": gradcam_result,
    }
if __name__ == "__main__":
    sample_prediction = {
        "label": "FAKE",
        "confidence": 0.7123,
        "text_confidence": 0.7812
    }
    sample_shap = {
        "shap": {
            "top_tokens": [
                {"token": "proof", "impact": 0.22},
                {"token": "must", "impact": 0.18},
                {"token": "exposed", "impact": 0.16},
                {"token": "official", "impact": -0.09},
                {"token": "reported", "impact": -0.05},
            ]
        },
        "flags": [
            {
                "name": "Overconfident claim style",
                "severity": "medium",
                "rationale": "Absolute or forceful wording can reduce nuance and may be used to make a claim sound more certain than the available evidence supports."
            }
        ]
    }
    sample_fusion = {
        "clip_model": {
            "similarity_01": 0.6432,
            "p_mismatch": 0.3578
        },
        "signals": {
            "driver": "text"
        }
    }
    sample_gradcam = {
        "top_region_summary": "The model focused mainly on the central region of the image when comparing visual content with the written claim."
    }
    out = generate_detailed_explanation(
        prediction=sample_prediction,
        shap_result=sample_shap,
        fusion_result=sample_fusion,
        gradcam_result=sample_gradcam,
    )
    print("SUMMARY:\n")
    print(out["summary"])
    print("\nTOP REASONS:\n")
    print(out["top_reasons"])
    print("\nDECISION PATHWAY:\n")
    print(out["decision_pathway"])
    print("\nSCORES:\n")
    print(out["scores"])