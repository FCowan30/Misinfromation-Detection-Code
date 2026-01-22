from __future__ import annotations

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("Missing pandas. Install with: pip install pandas") from e

from Backend.config import ARTIFACTS_DIR

# Dataset loaders
from Backend.Datasets.kaggle import load_kaggle
from Backend.Datasets.fever import load_fever
from Backend.Datasets.pubhealth import load_pubhealth
from Backend.Datasets.social_covid import load_social


REQUIRED_COLS = ["text", "label", "domain", "source"]


def load_all_datasets(
    include_kaggle: bool = True,
    include_fever: bool = True,
    include_pubhealth: bool = True,
    include_social: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if include_kaggle:
        frames.append(load_kaggle())
    if include_fever:
        frames.append(load_fever())
    if include_pubhealth:
        frames.append(load_pubhealth())
    if include_social:
        frames.append(load_social())

    if not frames:
        raise ValueError("No datasets selected.")

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Combined dataset missing required columns: {missing}")

    print("[INFO] Combined dataset shape:", df.shape)
    print("[INFO] Label counts:\n", df["label"].value_counts(dropna=False))
    print("[INFO] Domain counts:\n", df["domain"].value_counts(dropna=False))
    print("[INFO] Source counts (top 10):\n", df["source"].value_counts().head(10))

    return df


def clean_combined(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure clean types
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].astype(bool)]

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])

    # If labels come through as strings, normalize here
    df["label"] = df["label"].astype(int)

    # De-dupe by text
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[INFO] Removed {before - len(df)} duplicate texts")

    print("[INFO] Cleaned dataset shape:", df.shape)
    return df


def tokenize_and_save(
    df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    test_size: float = 0.2,
    seed: int = 42,
    max_length: int = 256,
) -> None:
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("Missing datasets. Install with: pip install datasets") from e

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("Missing transformers. Install with: pip install transformers") from e

    # ✅ Keep only safe columns BEFORE Arrow conversion
    keep_cols = ["text", "label", "domain", "source"]
    df = df[keep_cols].copy()

    # ✅ Enforce clean types
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["domain"] = df["domain"].astype(str)
    df["source"] = df["source"].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    hf = Dataset.from_pandas(df, preserve_index=False)
    splits = hf.train_test_split(test_size=test_size, seed=seed)

    tokenized = splits.map(tok, batched=True)

    # Keep only Trainer-needed cols
    keep = {"input_ids", "attention_mask", "label"}
    remove_cols = [c for c in tokenized["train"].column_names if c not in keep]
    tokenized = tokenized.remove_columns(remove_cols)

    tokenized.set_format("torch")

    out_dir = ARTIFACTS_DIR / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenized["train"].save_to_disk(out_dir / "train")
    tokenized["test"].save_to_disk(out_dir / "eval")

    print("[INFO] Saved tokenized train/eval to:", out_dir)


if __name__ == "__main__":
    df_all = load_all_datasets(
        include_kaggle=True,
        include_fever=True,
        include_pubhealth=True,
        include_social=True,
    )
    df_all = clean_combined(df_all)
    tokenize_and_save(df_all)