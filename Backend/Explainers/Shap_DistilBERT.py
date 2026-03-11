from __future__ import annotations

# -----------------------------
# Standard library imports
# -----------------------------
from dataclasses import dataclass
from typing import Dict, List, Any
import sys
import re

# -----------------------------
# Third-party imports (guarded)
# -----------------------------
try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "NumPy is required for SHAP explainability. "
        "Install it with: pip install numpy"
    ) from e

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for DistilBERT inference. "
        "Install it with: pip install torch"
    ) from e

try:
    import shap
except ImportError as e:
    raise ImportError(
        "SHAP is required for explainability. "
        "Install it with: pip install shap"
    ) from e

try:
    from scipy.special import softmax
except ImportError as e:
    raise ImportError(
        "SciPy is required for probability calculations. "
        "Install it with: pip install scipy"
    ) from e

# -----------------------------
# Local project imports (guarded)
# -----------------------------
try:
    from Backend.Models.DistillBERT_loader import load_distilbert
except ImportError as e:
    raise ImportError(
        "Failed to import DistilBERT loader. "
        "Ensure Backend/Models/DistillBERT_loader.py exists and "
        "Backend is a valid Python package."
    ) from e


LABELS = ["FAKE", "TRUE"]  # 0=fake, 1=true

# -----------------------------
# Determinism (helps reduce "samey" explanations)
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)

# --------------------------------------------
# Model + tokenizer (loaded once)
# --------------------------------------------
_tokenizer, _model = load_distilbert()

# SHAP is more stable on CPU
_model.to("cpu")
_model.eval()

# -----------------------------
# Small helpers
# -----------------------------
def _norm_token(t: str) -> str:
    """Normalize a token for matching/flags (lowercase, strip, trim punctuation)."""
    t = str(t).lower().strip()
    # Remove leading/trailing non-word characters
    t = re.sub(r"^\W+|\W+$", "", t)
    return t


# -----------------------------
# Prediction function for SHAP
# -----------------------------
def predict_proba(texts):
    """
    Returns probability for [FAKE, TRUE] for a batch of texts.
    SHAP may pass strings, lists, or numpy arrays.
    """
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    texts = [str(t) for t in texts]

    enc = _tokenizer(
        texts,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )

    # Ensure tensors on same device as model (cpu)
    enc = {k: v.to(next(_model.parameters()).device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits.detach().cpu().numpy()

    return softmax(logits, axis=1)


# -----------------------------
# SHAP Explainer
# -----------------------------
_masker = shap.maskers.Text(_tokenizer)
_explainer = shap.Explainer(predict_proba, _masker, output_names=LABELS)


# ------------------------------
# Simple adaptive "flags layer"
# ------------------------------
@dataclass
class Flag:
    name: str
    severity: str  # "low", "medium", "high"
    rationale: str
    examples: List[str]


def _contains_any(text_lc: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if p in text_lc]


def detect_flags(text: str, top_tokens: List[str]) -> List[Flag]:
    """
    Uses BOTH:
    - normalized SHAP top tokens (for model-driven signals)
    - raw lowercased text (for phrase hits)
    """
    text_lc = text.lower()
    top_set = set(_norm_token(t) for t in top_tokens if _norm_token(t))

    conspiracy_phrases = [
        "deep state",
        "false flag",
        "new world order",
        "hidden truth",
        "crisis actor",
        "wake up",
        "cover up",
        "covered up",  # fixed casing
    ]
    secrecy_words = {"secret", "exposed", "leaked", "leak", "hidden", "agenda", "coverup"}
    certainty_words = {"proof", "proven", "definitely", "undeniable", "guaranteed", "100"}
    emotion_words = {"shocking", "disgusting", "evil", "outrage", "terrifying", "scam"}
    vague_sources = [
        "experts say",
        "sources say",
        "many are saying",
        "it is said",
        "people are saying",
    ]
    viral_cta = ["share this", "spread this", "before it's deleted", "they will delete", "repost"]

    flags: List[Flag] = []

    # Secrecy / conspiracy framing
    phrase_hits = _contains_any(text_lc, conspiracy_phrases)
    secrecy_hits = sorted(list(secrecy_words.intersection(top_set)))
    if phrase_hits or secrecy_hits:
        flags.append(
            Flag(
                name="Secrecy / conspiracy framing",
                severity="high" if phrase_hits else "medium",
                rationale="Wording implies hidden motives or suppressed information, which can be common in misleading narratives.",
                examples=(phrase_hits[:3] + secrecy_hits[:3]),
            )
        )

    # Overconfident claim style
    certainty_hits = sorted(list(top_set.intersection(certainty_words)))
    # Also check raw text for "100%" (since tokenization can split it)
    if "100%" in text or "100 percent" in text_lc:
        if "100" not in certainty_hits:
            certainty_hits = ["100"] + certainty_hits
    if certainty_hits:
        flags.append(
            Flag(
                name="Overconfident claim style",
                severity="medium",
                rationale="Absolute language can reduce nuance and is sometimes used to persuade rather than inform.",
                examples=certainty_hits[:3],
            )
        )

    # Emotionally charged wording
    emotion_hits = sorted(list(top_set.intersection(emotion_words)))
    if emotion_hits:
        flags.append(
            Flag(
                name="Emotionally charged wording",
                severity="medium",
                rationale="Strong emotive terms can encourage reaction over evidence.",
                examples=emotion_hits[:3],
            )
        )

    # Vague sources / attribution
    vague_hits = _contains_any(text_lc, vague_sources)
    if vague_hits:
        flags.append(
            Flag(
                name="Vague sources / attribution",
                severity="low",
                rationale="Claims attributed to unnamed sources can be harder to verify.",
                examples=vague_hits[:2],
            )
        )

    # Virality / urgency prompt
    viral_hits = _contains_any(text_lc, viral_cta)
    if viral_hits:
        flags.append(
            Flag(
                name="Virality / urgency prompt",
                severity="high",
                rationale="Urgent calls to share or fear of deletion are common in manipulative posts.",
                examples=viral_hits[:2],
            )
        )

    # Authority / official framing (context)
    authority_words = {"confirmed", "official", "report", "government", "police", "nhs", "who", "cdc"}
    authority_hits = sorted(list(top_set.intersection(authority_words)))
    if authority_hits:
        flags.append(
            Flag(
                name="Authority / official framing (context)",
                severity="low",
                rationale="References to official sources can increase perceived credibility; check whether specific sources are cited.",
                examples=authority_hits[:3],
            )
        )

    return flags


# -----------------------------
# Main explain function
# -----------------------------
def explain_text(text: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Returns a JSON-friendly explanation:
    - top SHAP tokens for predicted class
    - adaptive flags
    - a more dynamic natural-language summary that changes with input
    """
    text = str(text).strip()
    if not text:
        raise ValueError("No text provided to explain_text().")

    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    # Run SHAP
    sv = _explainer([text])[0]
    tokens = list(sv.data)
    values = np.array(sv.values)

    # Select contributions for the predicted class
    contrib = values[:, pred_idx] if values.ndim == 2 else values

    ranked = sorted(
        [(t, float(v)) for t, v in zip(tokens, contrib) if str(t).strip()],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_contrib = ranked[:top_n]
    top_tokens = [t for t, _ in top_contrib]

    flags = detect_flags(text, top_tokens)

    # --- Dynamic NLG summary (changes with tokens + impacts) ---
    if confidence >= 0.85:
        conf_desc = "high"
    elif confidence >= 0.65:
        conf_desc = "moderate"
    else:
        conf_desc = "low"

    top_k = top_contrib[:3]
    token_bits = []
    for tok, val in top_k:
        tok_clean = str(tok).strip()
        direction = f"towards {pred_label}" if val >= 0 else f"away from {pred_label}"
        token_bits.append(f"{tok_clean!r} ({val:+.3f}, {direction})")
    tokens_str = ", ".join(token_bits) if token_bits else "N/A"

    if flags:
        main_flag = flags[0].name
        summary = (
            f"Prediction: {pred_label} (confidence {confidence:.3f}, {conf_desc}). "
            f"Top influential tokens: {tokens_str}. "
            f"Flag triggered: {main_flag}. "
            "This reflects learned patterns in the training data, not factual verification."
        )
    else:
        summary = (
            f"Prediction: {pred_label} (confidence {confidence:.3f}, {conf_desc}). "
            f"Top influential tokens: {tokens_str}. "
            "No heuristic language flags were triggered. "
            "This reflects learned patterns in the training data, not factual verification."
        )

    return {
        "prediction": {
            "label": pred_label,
            "confidence": confidence,
            "probs": {"FAKE": float(probs[0]), "TRUE": float(probs[1])},
        },
        "shap": {
            "top_tokens": [{"token": t, "impact": v} for t, v in top_contrib],
        },
        "flags": [f.__dict__ for f in flags],
        "summary": summary,
        "input": {"text": text},
    }


# -----------------------------
# CLI test
# -----------------------------
if __name__ == "__main__":
    user_text = input("Enter a sentence to explain: ").strip()
    out = explain_text(user_text, top_n=10)

    print("\n[EXPLANATION]")
    print("Summary:", out["summary"])
    print("Prediction:", out["prediction"])
    print("Top tokens:")
    for item in out["shap"]["top_tokens"]:
        print(f"  - {item['token']!r}: {item['impact']:+.4f}")

    if out["flags"]:
        print("Flags:")
        for f in out["flags"]:
            ex = ", ".join(f["examples"]) if f["examples"] else ""
            print(f"  - {f['name']} ({f['severity']}): {f['rationale']} {ex}")
    else:
        print("Flags: none")