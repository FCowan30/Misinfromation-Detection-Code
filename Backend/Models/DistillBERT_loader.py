from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Backend.config import MODEL_DIR


def load_distilbert(model_dir: str = MODEL_DIR):
    print(f"[INFO] Loading DistilBERT model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    print("[INFO] DistilBERT model and tokenizer loaded successfully.")
    print(f"[INFO] Model device: {next(model.parameters()).device}")
    print(f"[INFO] Number of labels: {model.config.num_labels}")

    return tokenizer, model


if __name__ == "__main__":
    # Lightweight self-test (safe to run locally)
    tok, mdl = load_distilbert()

    test_text = "This is a simple test sentence."
    inputs = tok(test_text, return_tensors="pt")
    outputs = mdl(**inputs)

    print("[TEST] Loader self-test passed.")
    print("[TEST] Logits shape:", outputs.logits.shape)
  