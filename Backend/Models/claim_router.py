from __future__ import annotations

from typing import Dict, Any


DESCRIPTIVE_PREFIXES = [
    "this is",
    "this shows",
    "the image shows",
    "this image shows",
    "a photo of",
    "an image of",
    "picture of",
    "this picture shows",
    "the photo shows",
]

FACTUAL_MARKERS = [
    "proves",
    "proof",
    "evidence",
    "according to",
    "reported",
    "happened",
    "happening",
    "fake",
    "real",
    "misleading",
    "false",
    "true",
    "claims that",
    "shows that",
    "in 20",
    "today",
    "yesterday",
    "last year",
    "this means",
]


def detect_claim_type(text: str) -> Dict[str, Any]:
    t = (text or "").strip().lower()
    tokens = t.split()

    if not t:
        return {
            "claim_type": "unknown",
            "route": "full_multimodal",
            "reason": "No text was provided."
        }

    if any(t.startswith(prefix) for prefix in DESCRIPTIVE_PREFIXES):
        return {
            "claim_type": "descriptive_visual",
            "route": "clip_descriptive",
            "reason": "The statement mainly describes visible image content."
        }

    if any(marker in t for marker in FACTUAL_MARKERS):
        return {
            "claim_type": "factual_claim",
            "route": "full_multimodal",
            "reason": "The statement contains broader factual or misinformation-style markers."
        }

    if len(tokens) <= 6:
        return {
            "claim_type": "descriptive_visual",
            "route": "clip_descriptive",
            "reason": "The statement is short and appears to describe visible image content."
        }

    return {
        "claim_type": "general_claim",
        "route": "full_multimodal",
        "reason": "The statement appears to be a broader claim, so full multimodal analysis is used."
    }


def decide_descriptive_visual_result(
    similarity_01: float,
    p_mismatch: float | None = None
) -> Dict[str, Any]:
    similarity_01 = float(similarity_01)
    mismatch = float(p_mismatch) if p_mismatch is not None else (1.0 - similarity_01)

    # Strong visual support
    if similarity_01 >= 0.75 and mismatch < 0.35:
        label = "TRUE"
        confidence = similarity_01
        explanation = (
            "The statement was treated as a descriptive visual claim, so the system relied on "
            "image-text similarity rather than the text classifier. The similarity was high, "
            "which suggests the description is visually supported by the image."
        )

    # Strong visual conflict
    elif similarity_01 < 0.45 or mismatch >= 0.60:
        label = "FAKE"
        confidence = max(1.0 - similarity_01, mismatch)
        explanation = (
            "The statement was treated as a descriptive visual claim, so the system relied on "
            "image-text similarity rather than the text classifier. The similarity was low "
            "or the derived mismatch score was high, which suggests the description is not well supported by the image."
        )

    # Middle zone = uncertain, but confidence should stay modest
    else:
        label = "UNCERTAIN"

        # confidence should be LOWER when the result is ambiguous
        # strongest uncertainty around the middle band
        distance_from_true = abs(similarity_01 - 0.75)
        distance_from_fake = abs(similarity_01 - 0.45)
        ambiguity = min(distance_from_true, distance_from_fake)

        # map ambiguity zone to a modest confidence range
        # UNCERTAIN should not look highly certain
        confidence = max(0.50, min(0.69, 0.65 - ambiguity))

        explanation = (
            "The statement was treated as a descriptive visual claim, so the system relied on "
            "image-text similarity rather than the text classifier. The similarity was in an "
            "intermediate range, so the image did not provide strong enough evidence to clearly "
            "support or reject the description."
        )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "analysis_route": "clip_descriptive",
        "route_reason": "The statement mainly describes visible image content.",
        "route_explanation": explanation,
        "used_text_model": False,
        "used_clip_model": True,
        "similarity_01": round(similarity_01, 4),
        "p_mismatch": round(mismatch, 4) if mismatch is not None else None,
    }