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
# Confidence / risk
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


def get_risk_level(pred_label: str, confidence: float, p_mismatch: Optional[float], analysis_route: Optional[str]) -> str:
    pred = str(pred_label).upper()
    mismatch = _safe_float(p_mismatch) if p_mismatch is not None else None

    if analysis_route == "clip_descriptive":
        if pred == "TRUE":
            if confidence >= 0.80:
                return "Strong visual support"
            elif confidence >= 0.60:
                return "Moderate visual support"
            return "Weak visual support"

        if pred == "FAKE":
            if confidence >= 0.80:
                return "Strong visual conflict"
            elif confidence >= 0.60:
                return "Moderate visual conflict"
            return "Possible visual conflict"

        return "Uncertain visual support"

    if pred == "FAKE":
        if confidence >= 0.85:
            return "High misinformation risk"
        elif confidence >= 0.65:
            return "Moderate misinformation risk"
        return "Possible misinformation risk"

    if pred == "REAL" or pred == "TRUE":
        if mismatch is not None and mismatch >= 0.60:
            return "Mixed evidence"
        elif confidence >= 0.80:
            return "Lower misinformation risk"
        return "Uncertain reliability"

    return "Uncertain result"


def describe_confidence(conf: float, pred_label: str, analysis_route: Optional[str]) -> Tuple[str, str, str]:
    band = get_confidence_band(conf)
    conf_pct = _format_percent(conf)

    if analysis_route == "clip_descriptive":
        main = (
            f"The system predicts that this descriptive image claim is {pred_label} "
            f"with {band} confidence ({conf_pct})."
        )
        meaning = (
            f"This {conf_pct} confidence score reflects how strongly the visual similarity-based route "
            f"supported the {pred_label} outcome. It does not mean there is a {conf_pct} real-world "
            f"probability that the statement is objectively true or false in every context."
        )
    else:
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
# Claim route explanation
# -----------------------------
def describe_analysis_route(prediction: Dict[str, Any], fusion_result: Optional[Dict[str, Any]] = None) -> str:
    analysis_route = prediction.get("analysis_route")

    if analysis_route == "clip_descriptive":
        return (
            "This input was treated as a descriptive visual claim. "
            "Because the statement mainly describes visible image content, "
            "the system relied on image-text similarity rather than the text classifier."
        )

    if analysis_route == "full_multimodal":
        return (
            "This input was treated as a broader multimodal claim, "
            "so the system used the full analysis pipeline combining text and image evidence."
        )

    if analysis_route == "text_only":
        return "This input was analysed using the text-only route because no image was provided."

    if analysis_route == "text_only_fallback":
        return "This input was analysed using the text-only fallback route because the image could not be used."

    return "The system selected an analysis route based on the type of input provided."


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
    negative.sort(key=lambda x: x["impact"])
    return positive, negative


# -----------------------------
# Evidence for / against
# -----------------------------
def build_evidence_for(
    top_tokens: List[Dict[str, Any]],
    flags: List[Dict[str, Any]],
    pred_label: str,
    similarity_01: Optional[float],
    p_mismatch: Optional[float],
    driver: Optional[str],
    prediction: Optional[Dict[str, Any]] = None
) -> str:
    analysis_route = prediction.get("analysis_route") if prediction else None
    reasons: List[str] = []

    if analysis_route == "clip_descriptive":
        if similarity_01 is not None:
            sim_pct = _format_percent(similarity_01)
            if _safe_float(similarity_01) >= 0.75:
                reasons.append(
                    f"The main supporting evidence was the image-text similarity score of {sim_pct}, "
                    "which indicates strong alignment between the wording and the image."
                )
            elif _safe_float(similarity_01) >= 0.50:
                reasons.append(
                    f"The image-text similarity score was {sim_pct}, which indicates moderate visual support."
                )
            else:
                reasons.append(
                    f"The image-text similarity score was {sim_pct}, which provides only limited visual support."
                )

        if p_mismatch is not None:
            mismatch_pct = _format_percent(p_mismatch)
            if _safe_float(p_mismatch) < 0.35:
                reasons.append(
                    f"The derived mismatch score was low at {mismatch_pct}, suggesting little visual conflict."
                )
            elif _safe_float(p_mismatch) < 0.50:
                reasons.append(
                    f"The derived mismatch score was {mismatch_pct}, suggesting only limited visual conflict."
                )
            else:
                reasons.append(
                    f"The derived mismatch score was {mismatch_pct}, indicating noticeable visual conflict."
                )

        if not reasons:
            return "The system relied mainly on the image-text relationship for this descriptive claim."

        return " ".join(reasons)

    positive, _ = split_token_impacts(top_tokens)

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
        elif pred_label.upper() in {"REAL", "TRUE"}:
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


def build_evidence_against(
    top_tokens: List[Dict[str, Any]],
    pred_label: str,
    similarity_01: Optional[float],
    p_mismatch: Optional[float],
    prediction: Optional[Dict[str, Any]] = None
) -> str:
    analysis_route = prediction.get("analysis_route") if prediction else None
    reasons: List[str] = []

    if analysis_route == "clip_descriptive":
        if similarity_01 is not None and _safe_float(similarity_01) < 0.50:
            reasons.append(
                f"The similarity score was only {_format_percent(similarity_01)}, which weakens strong visual support."
            )
        if p_mismatch is not None and _safe_float(p_mismatch) >= 0.50:
            reasons.append(
                f"The derived mismatch score reached {_format_percent(p_mismatch)}, which suggests conflict between the image and the description."
            )

        if not reasons:
            return "The system found little clear counter-evidence against the descriptive visual prediction."

        return " ".join(reasons)

    _, negative = split_token_impacts(top_tokens)

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
        elif pred_label.upper() in {"REAL", "TRUE"}:
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
def describe_flags(flags: List[Dict[str, Any]], analysis_route: Optional[str]) -> str:
    if analysis_route == "clip_descriptive":
        return (
            "No text-classifier language pattern analysis was used for this result because the input "
            "was handled through the descriptive visual route."
        )

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
    p_mismatch: Optional[float],
    analysis_route: Optional[str]
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

        if analysis_route == "clip_descriptive":
            if sim >= 0.75:
                similarity_text = (
                    f"The descriptive image-text similarity score was strong ({sim_pct}), suggesting that the image closely matches the wording of the statement."
                )
            elif sim >= 0.50:
                similarity_text = (
                    f"The descriptive image-text similarity score was moderate ({sim_pct}), suggesting that the image is broadly consistent with the statement."
                )
            elif sim >= 0.25:
                similarity_text = (
                    f"The descriptive image-text similarity score was weak ({sim_pct}), suggesting that the image only loosely supports the statement."
                )
            else:
                similarity_text = (
                    f"The descriptive image-text similarity score was very low ({sim_pct}), suggesting that the image does not strongly match the statement."
                )
        else:
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

        if analysis_route == "clip_descriptive":
            if mismatch >= 0.75:
                mismatch_text = (
                    f"The derived mismatch score was high ({mismatch_pct}), which suggests strong conflict between the image and the descriptive statement."
                )
            elif mismatch >= 0.50:
                mismatch_text = (
                    f"The derived mismatch score was moderate ({mismatch_pct}), suggesting the image may not fully support the descriptive statement."
                )
            elif mismatch >= 0.25:
                mismatch_text = (
                    f"The derived mismatch score was fairly low ({mismatch_pct}), so only limited visual conflict was detected."
                )
            else:
                mismatch_text = (
                    f"The derived mismatch score was very low ({mismatch_pct}), indicating little visual conflict."
                )
        else:
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
def describe_driver(driver: Optional[str], pred_label: str, analysis_route: Optional[str]) -> str:
    if analysis_route == "clip_descriptive":
        return (
            f"The final {pred_label} decision was driven primarily by image-text similarity rather than the text classifier."
        )

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
    driver: Optional[str],
    analysis_route: Optional[str]
) -> List[str]:
    pathway = []

    if analysis_route == "clip_descriptive":
        pathway.append("Step 1: The system identified the input as a short descriptive visual claim.")
        pathway.append("Step 2: The text classifier was skipped because the statement mainly described visible image content.")
        pathway.append("Step 3: The image and text were compared using CLIP similarity to estimate visual support.")
        if has_visual:
            pathway.append("Step 4: A visual explanation was generated to show which image regions were most relevant.")
        else:
            pathway.append("Step 4: No visual region explanation was available.")
        pathway.append("Step 5: The final result was produced from the visual consistency route.")
        return pathway

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
    driver: Optional[str],
    analysis_route: Optional[str]
) -> List[str]:
    reasons: List[str] = []

    if analysis_route == "clip_descriptive":
        if similarity_01 is not None:
            sim = _safe_float(similarity_01)
            if sim >= 0.75:
                reasons.append(
                    f"The image-text similarity score was {_format_percent(sim)}, indicating strong visual alignment."
                )
            elif sim >= 0.50:
                reasons.append(
                    f"The image-text similarity score was {_format_percent(sim)}, indicating moderate visual alignment."
                )
            else:
                reasons.append(
                    f"The image-text similarity score was {_format_percent(sim)}, indicating weak visual alignment."
                )

        if p_mismatch is not None:
            mismatch = _safe_float(p_mismatch)
            if mismatch >= 0.50:
                reasons.append(
                    f"The derived mismatch score was {_format_percent(mismatch)}, indicating notable visual conflict."
                )
            else:
                reasons.append(
                    f"The derived mismatch score was {_format_percent(mismatch)}, indicating limited visual conflict."
                )

        reasons.append("The text classifier was intentionally skipped because the statement was treated as a descriptive visual claim.")
        return reasons[:3]

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
    has_visual: bool,
    prediction: Optional[Dict[str, Any]] = None
) -> str:
    analysis_route = prediction.get("analysis_route") if prediction else None

    parts = [
        "This explanation describes how the model reached its decision, not verified truth."
    ]

    parts.append(
        "A high confidence score means the model preferred one output over the alternatives, but it does not guarantee correctness."
    )

    if analysis_route == "clip_descriptive":
        parts.append(
            "For descriptive visual claims, the system mainly checks whether the image supports the wording, rather than whether a broader factual claim is true in the real world."
        )
        parts.append(
            "The mismatch value in this route is a derived proxy based on similarity rather than a separate calibrated classifier."
        )

    if not has_tokens and analysis_route not in {"clip_descriptive"}:
        parts.append("No strong token-level explanation was available, so the text reasoning may be incomplete.")
    if not has_fusion:
        parts.append("No full image-text fusion output was available, so the explanation relied on fewer multimodal signals.")
    if not has_visual:
        parts.append("No visual region explanation was available, so the system could not highlight which image areas influenced the result.")

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
        text_conf = prediction.get("text_confidence")
    elif fusion_result and "text_model" in fusion_result:
        text_conf = fusion_result["text_model"].get("confidence")

    similarity = None
    mismatch = None
    raw_similarity = None

    if fusion_result:
        similarity = fusion_result.get("clip_model", {}).get("similarity_01")
        mismatch = fusion_result.get("clip_model", {}).get("p_mismatch")
        raw_similarity = fusion_result.get("clip_model", {}).get("raw_similarity")

    return {
        "text_confidence_raw": _safe_float(text_conf) if text_conf is not None else None,
        "text_confidence_pct": _to_percent(text_conf) if text_conf is not None else None,
        "similarity_raw": _safe_float(similarity) if similarity is not None else None,
        "similarity_pct": _to_percent(similarity) if similarity is not None else None,
        "mismatch_raw": _safe_float(mismatch) if mismatch is not None else None,
        "mismatch_pct": _to_percent(mismatch) if mismatch is not None else None,
        "final_confidence_raw": final_conf,
        "final_confidence_pct": _to_percent(final_conf),
        "clip_raw_similarity": _safe_float(raw_similarity) if raw_similarity is not None else None,
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
    analysis_route = prediction.get("analysis_route")

    top_tokens: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []

    if shap_result and isinstance(shap_result, dict):
        top_tokens = shap_result.get("shap", {}).get("top_tokens", []) or []
        flags = shap_result.get("flags", []) or []

    similarity_01 = None
    p_mismatch = None
    driver = None

    if fusion_result:
        similarity_01 = fusion_result.get("clip_model", {}).get("similarity_01")
        p_mismatch = fusion_result.get("clip_model", {}).get("p_mismatch")
        driver = fusion_result.get("signals", {}).get("driver")

    prediction_text, confidence_meaning_text, uncertainty_text = describe_confidence(
        confidence, pred_label, analysis_route
    )

    analysis_route_text = describe_analysis_route(prediction, fusion_result)

    risk_level = get_risk_level(
        pred_label=pred_label,
        confidence=confidence,
        p_mismatch=p_mismatch,
        analysis_route=analysis_route
    )

    evidence_for_text = build_evidence_for(
        top_tokens=top_tokens,
        flags=flags,
        pred_label=pred_label,
        similarity_01=similarity_01,
        p_mismatch=p_mismatch,
        driver=driver,
        prediction=prediction
    )

    evidence_against_text = build_evidence_against(
        top_tokens=top_tokens,
        pred_label=pred_label,
        similarity_01=similarity_01,
        p_mismatch=p_mismatch,
        prediction=prediction
    )

    flag_text = describe_flags(flags, analysis_route)

    similarity_text = None
    mismatch_text = None
    if fusion_result:
        similarity_text, mismatch_text = describe_similarity_and_mismatch(
            similarity_01, p_mismatch, analysis_route
        )

    driver_text = describe_driver(driver, pred_label, analysis_route) if (fusion_result or analysis_route == "clip_descriptive") else None
    visual_text = describe_visual_region(gradcam_result) if gradcam_result else None

    decision_pathway = build_decision_pathway(
        has_text=bool(shap_result),
        has_fusion=bool(fusion_result),
        has_visual=bool(gradcam_result),
        driver=driver,
        analysis_route=analysis_route
    )

    top_reasons = build_top_reasons(
        pred_label=pred_label,
        top_tokens=top_tokens,
        flags=flags,
        similarity_01=similarity_01,
        p_mismatch=p_mismatch,
        driver=driver,
        analysis_route=analysis_route
    )

    limitations_text = build_limitations_statement(
        has_tokens=bool(top_tokens),
        has_fusion=bool(fusion_result),
        has_visual=bool(gradcam_result),
        prediction=prediction
    )

    scores = extract_scores(prediction, fusion_result)

    summary_parts = [
        prediction_text,
        analysis_route_text,
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
        "analysis_route_text": analysis_route_text,

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
        "label": "TRUE",
        "confidence": 0.91,
        "analysis_route": "clip_descriptive",
        "route_reason": "The statement mainly describes visible image content.",
        "used_text_model": False,
        "used_clip_model": True,
        "text_confidence": None,
    }
    sample_fusion = {
        "clip_model": {
            "raw_similarity": 0.82,
            "similarity_01": 0.91,
            "p_mismatch": 0.09
        },
        "signals": {
            "driver": "both"
        },
        "analysis_route": "clip_descriptive",
        "route_reason": "The statement mainly describes visible image content.",
        "used_text_model": False,
        "used_clip_model": True,
        "final": sample_prediction,
    }
    sample_gradcam = {
        "top_region_summary": "The model focused mainly on the central object in the image when checking whether the description matched the visual content."
    }
    out = generate_detailed_explanation(
        prediction=sample_prediction,
        shap_result=None,
        fusion_result=sample_fusion,
        gradcam_result=sample_gradcam,
    )
    print(out["summary"])
    print(out["scores"])