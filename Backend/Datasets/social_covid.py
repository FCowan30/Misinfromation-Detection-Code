from __future__ import annotations

from pathlib import Path

import pandas as pd

from Backend.config import DATA_DIR

SOCIAL_DIR = DATA_DIR / "Social"


def guess_text_column(columns) -> str | None:
    candidates = ["text", "content", "article", "body", "tweet", "post", "title"]
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def label_from_filename(filename: str) -> int | None:
    name = filename.lower()
    if "fake" in name:
        return 0
    if "real" in name or "true" in name:
        return 1
    return None


def load_social(folder: Path = SOCIAL_DIR) -> pd.DataFrame:
    """
    Loads Social CSV datasets from folder and returns a standard DataFrame:
      text, label, domain, source
    """
    if not folder.exists():
        raise FileNotFoundError(f"Social dataset folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for csv_path in csv_files:
        label = label_from_filename(csv_path.name)
        if label is None:
            skipped.append((csv_path.name, "no label in filename"))
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            skipped.append((csv_path.name, f"read error: {e}"))
            continue

        text_col = guess_text_column(df.columns)
        if text_col is None:
            skipped.append((csv_path.name, f"no text column found (cols={list(df.columns)[:10]})"))
            continue

        tmp = pd.DataFrame()
        tmp["text"] = df[text_col].astype(str).str.strip()
        tmp = tmp[tmp["text"].astype(bool)]  # drop empty strings
        tmp["label"] = label
        tmp["domain"] = "social"
        tmp["source"] = csv_path.stem

        if len(tmp) == 0:
            skipped.append((csv_path.name, "no rows after cleaning"))
            continue

        frames.append(tmp)

    if not frames:
        msg = "No usable social datasets found.\n"
        msg += f"Scanned files: {len(csv_files)}\n"
        msg += "Skipped reasons (first 10):\n"
        for s in skipped[:10]:
            msg += f" - {s[0]}: {s[1]}\n"
        raise ValueError(msg)

    df_all = pd.concat(frames, ignore_index=True)

    print(f"[INFO] Social CSVs scanned: {len(csv_files)} | usable: {len(frames)} | skipped: {len(skipped)}")
    print(f"[INFO] Social rows loaded: {len(df_all)}")
    if skipped:
        print("[INFO] Skipped files (first 5):")
        for s in skipped[:5]:
            print(" -", s)

    return df_all


if __name__ == "__main__":
    print("[TEST] Running social_covid loader...")
    df = load_social()
    print(df.head())
    print(df["label"].value_counts())
