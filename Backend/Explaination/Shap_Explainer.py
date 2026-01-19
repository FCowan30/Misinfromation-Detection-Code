#Backend/Explainer/Bert_Explainer.py
from __future__ import annotations

from dataclassesfrom import dataclass
from typing import Dict,List,Any
import numpy as np
import torch
import shape 
from scipy.special import softmax

from backend.Models.DistillBERT_loader import load_distilbert

LABELS = ["FAKE", "TRUE"]  # 0=fake, 1=true

#--------------------------------------------
#Model + tokenizer (loaded once)
#--------------------------------------------

_tokenizer, __model = load_distilbert()

# shap is more stable on cpu
_model.to("cpu")
_model.eval()

# Backend/explainers/bert_shap_explainer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch
import shap
from scipy.special import softmax

# Adjust this import to match your exact casing/location
from Backend.Models.DistillBERT_loader import load_distilbert

LABELS = ["FAKE", "TRUE"]  # 0=fake, 1=true


# -----------------------------
# Model + tokenizer (loaded once)
# -----------------------------
_tokenizer, _model = load_distilbert()
# SHAP is more stable on CPU (especially on Windows)
_model.to("cpu")
_model.eval()


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

def detect_flags(text_lc: str, top_tokens: List[str]) -> List[Flag]:
    text_lc = text.lower()
    top_set = set([t.lower() for t in top_tokens])

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