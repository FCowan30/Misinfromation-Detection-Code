import pandas as pd
from pathlib import Path
from Backend.config import DATA_DIR


def load_fever() -> pd.DataFrame:
    """
    Loads the FEVER dataset (JSONL) and returns a standardised DataFrame:
        text | label | domain | source
    """

    fever_dir = DATA_DIR / "Fever"
    fever_path = fever_dir / "train.jsonl"

    if not fever_path.exists():
        raise FileNotFoundError(f"FEVER file not found: {fever_path}")

    # JSONL requires lines=True
    df = pd.read_json(fever_path, lines=True)

    # Standardise text column
    if "claim" not in df.columns:
        raise ValueError("FEVER dataset missing 'claim' column")

    df = df.rename(columns={"claim": "text"})

    # Map FEVER labels → binary
    label_map = {
        "SUPPORTS": 1,
        "REFUTES": 0
        # NOT ENOUGH INFO → drop
    }

    if "label" not in df.columns:
        raise ValueError("FEVER dataset missing 'label' column")

    df["label"] = df["label"].map(label_map)

    # Drop NEI and invalid rows
    df = df.dropna(subset=["label", "text"])

    # Add metadata
    df["label"] = df["label"].astype(int)
    df["domain"] = "news"
    df["source"] = "fever"

    # Minimal cleaning only
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].astype(bool)]
    df = df.drop_duplicates(subset=["text"])

    print("[INFO] FEVER dataset loaded")
    print("[INFO] Shape:", df.shape)
    print("[INFO] Label distribution:\n", df["label"].value_counts())

    return df


if __name__ == "__main__":
    df_fever = load_fever()
    print(df_fever.head())
