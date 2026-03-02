# Backend/Models/DistillBERT_loader.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Backend.config import MODEL_DIR
import sys

_TOKENIZER = None
_MODEL = None

def load_distilbert(model_dir: str = MODEL_DIR):
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    print(f"[INFO] Loading DistilBERT model from: {model_dir}", file=sys.stderr)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _MODEL.eval()
    return _TOKENIZER, _MODEL
  