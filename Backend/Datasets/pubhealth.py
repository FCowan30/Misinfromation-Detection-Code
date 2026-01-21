import pandas as pd
from Backend.config import DATA_DIR

PUBHEALTH_DIR = DATA_DIR / "Health"  # adjust to your actual folder name


def load_pubhealth(split: str = "train") -> pd.DataFrame:
    """
    Loads PubHealth TSV (train/test) and returns standard schema:
      text | label | domain | source
    """
    path = PUBHEALTH_DIR / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"PubHealth file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    # --- adjust these after you inspect df.columns ---
    text_col_candidates = ["claim", "text", "statement", "headline"]
    label_col_candidates = ["label", "truth_label", "verdict", "class"]

    def pick_col(cands):
        lower = {c.lower(): c for c in df.columns}
        for cand in cands:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    text_col = pick_col(text_col_candidates)
    label_col = pick_col(label_col_candidates)

    if text_col is None:
        raise ValueError(f"Could not find text column. Columns are: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find label column. Columns are: {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str).str.strip()

    # ---- label mapping: EDIT once you see actual labels ----
    label_map = {
        "true": 1,
        "false": 0,
        "real": 1,
        "fake": 0,
        "supports": 1,
        "refutes": 0,
    }

    out["label"] = df[label_col].astype(str).str.strip().str.lower().map(label_map)

    # Drop rows with unknown labels + empty text
    out = out.dropna(subset=["text", "label"])
    out = out[out["text"].astype(bool)]

    out["label"] = out["label"].astype(int)
    out["domain"] = "health"
    out["source"] = f"pubhealth_{split}"

    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"[INFO] PubHealth {split} loaded:", out.shape)
    print(out["label"].value_counts())

    return out


if __name__ == "__main__":
    train_df = load_pubhealth("train")
    test_df = load_pubhealth("test")
    print(train_df.head())
