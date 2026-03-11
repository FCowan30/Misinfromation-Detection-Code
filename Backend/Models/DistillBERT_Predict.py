# Backend/Predict.py
import numpy as np
import torch
from scipy.special import softmax

from Backend.Models.DistillBERT_loader import load_distilbert

LABELS = ["FAKE", "TRUE"]  # 0=fake, 1=true

def predict_text(text: str) -> dict:
    tokenizer, model = load_distilbert()

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()[0]

    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))

    return {
        "label": LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {"FAKE": float(probs[0]), "TRUE": float(probs[1])},
    }


if __name__ == "__main__":
    user_text = input("Enter a sentence to classify: ").strip()
    if not user_text:
        print("[ERROR] No text provided.")
        raise SystemExit(1)

    result = predict_text(user_text)
    print("\n[RESULT]")
    print("Label:", result["label"])
    print("Confidence:", f"{result['confidence']:.4f}")
    print("Probabilities:", result["probs"])
