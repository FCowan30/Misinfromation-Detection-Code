from __future__ import annotations

# -----------------------------
# Standard library imports
# -----------------------------
from dataclasses import dataclass
from typing import Dict, List, Any

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

#--------------------------------------------
#Model + tokenizer (loaded once)
#--------------------------------------------

_tokenizer, _model = load_distilbert()

# shap is more stable on cpu
_model.to("cpu")
_model.eval()

import re

LABELS = ["FAKE", "TRUE"]  # 0=fake, 1=true

# -----------------------------
# Prediction function for SHAP
# -----------------------------
def predict_proba(texts):
    #retruns probability for [Fake, True] for a batch of texts.
    #SHAP may pass strings,Lists or numPy arrays

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

    #ensure tensors on same device as model (cpu)
    enc = {k: v.to(next(_model.parameters()).device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits.detach().cpu().numpy()

    return softmax(logits, axis=1)

# -----------------------------
# SHAP Explainer Class  
# -----------------------------

_masker = shap.maskers.Text(_tokenizer)
_explainer = shap.Explainer(predict_proba, _masker, output_names=LABELS)

#------------------------------
#Simple adaptive "flags layer"
#------------------------------
@dataclass
class Flag:
    name: str
    severity: str # e.g., "low", "medium", "high"
    rationale: str
    example: List[str]

def _contains_any(text_lc: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if p in text_lc]

def detect_flags(text: str, top_tokens: List[str]) -> List[Flag]:
    text_lc = text.lower()
    top_set = set(t.lower() for t in top_tokens)

    conspiracy_phrases = [
        "deep state", 
        "false flag", 
        "new world order", 
        "hidden truth", 
        "crisis actor",
        "wake up",
        "cover up",
        "Covered up"
    ]
    secrecy_words = {"secret", "exposed", "leaked", "leak", "hidden", "agenda"}
    certainty_words = {"proof", "proven", "definitely", "undeniable", "guaranteed", "100%"}
    emotion_words = {"shocking", "disgusting", "evil", "outrage", "terrifying", "scam"}
    vague_sources = ["experts say", "sources say", "many are saying", "it is said", "people are saying"]
    viral_cta = ["share this", "spread this", "before it's deleted", "they will delete", "repost"]

    flags: List[Flag] = []

    phrase_hits = _contains_any(text_lc, conspiracy_phrases)
    secrecy_hits = secrecy_words.intersection(top_set)
    if phrase_hits or secrecy_hits:
        flags.append(
            Flag(
                name="Secrecy / conspiracy framing",
                severity="high" if phrase_hits else "medium",
                rationale="Wording implies hidden motives or suppressed information, which can be common in misleading narratives.",
                examples=(phrase_hits[:3] + secrecy_hits[:3]),
            )
        )
    certainty_hits = sorted(list(top_set.intersection(certainty_words)))
    if certainty_hits:
        flags.append(
            Flag(
                name="Overconfident claim style",
                severity="medium",
                rationale="Absolute language can reduce nuance and is sometimes used to persuade rather than inform.",
                examples=certainty_hits[:3],
            )
        )
    
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

    # Helpful context flag (not necessarily "bad")
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
    - prediction + probabilities
    - top SHAP tokens for predicted class
    - adaptive flags
    - a short natural-language summary
    """
    text = str(text).strip()
    if not text:
        raise ValueError("No text provided to explain_text().")

    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    sv = _explainer([text])[0]
    tokens = list(sv.data)
    values = np.array(sv.values)  # (tokens, outputs) usually

    contrib = values[:, pred_idx] if values.ndim == 2 else values

    ranked = sorted(
        [(t, float(v)) for t, v in zip(tokens, contrib) if str(t).strip()],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_contrib = ranked[:top_n]
    top_tokens = [t for t, _ in top_contrib]

    flags = detect_flags(text, top_tokens)

    # Simple NLG summary (you can expand later)
    if flags:
        main_flag = flags[0].name
        summary = (
            f"Prediction: {pred_label} (confidence {confidence:.3f}). "
            f"Key language signal: {main_flag}. "
            "This reflects learned patterns in the training data, not factual verification."
        )
    else:
        summary = (
            f"Prediction: {pred_label} (confidence {confidence:.3f}). "
            "No strong heuristic language flags were triggered. "
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