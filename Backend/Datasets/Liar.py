from __future__ import annotations

import os
import pandas as pd
from typing import Tuple


# ----------------------------------------
# LIAR COLUMN STRUCTURE
# ----------------------------------------
LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


# ----------------------------------------
# LABEL MAPPING (6 → 2 classes)
# ----------------------------------------
def map_label(label: str) -> int | None:
    label = str(label).strip().lower()

    if label in ["false", "pants-fire"]:
        return 1  # FAKE

    if label in ["true", "mostly-true"]:
        return 0  # TRUE

    # Drop ambiguous labels
    if label in ["barely-true", "half-true"]:
        return None

    return None


# ----------------------------------------
# LOAD SINGLE SPLIT
# ----------------------------------------
def load_liar_split(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        quoting=3,
    )

    # Keep only what you need
    df = df[["statement", "label"]].copy()

    # Map labels
    df["label"] = df["label"].apply(map_label)

    # Drop ambiguous rows
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Standardise column name for your pipeline
    df = df.rename(columns={"statement": "text"})

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    return df.reset_index(drop=True)


# ----------------------------------------
# LOAD FULL DATASET
# ----------------------------------------
def load_liar_dataset(data_dir: str = "Data/LIAR") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train.tsv")
    eval_path = os.path.join(data_dir, "eval.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    train_df = load_liar_split(train_path)
    eval_df = load_liar_split(eval_path)
    test_df = load_liar_split(test_path)

    print("\n[LIAR DATASET LOADED]")
    print(f"Train size: {len(train_df)}")
    print(f"Eval size: {len(eval_df)}")
    print(f"Test size: {len(test_df)}")

    return train_df, eval_df, test_df


# ----------------------------------------
# QUICK TEST
# ----------------------------------------
if __name__ == "__main__":
    train, eval_df, test = load_liar_dataset("Data/LIAR")

    print("\nSample:")
    print(train.head())