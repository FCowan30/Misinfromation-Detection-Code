import pandas as pd
from pathlib import Path
from Backend.config import DATA_DIR

def Guess_text_column(columns):
    candidates =["text", "content", "article", "body", "tweet", "post", "title"]

    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None

def Label_From_Filename(filename):
    name = filename.lower()
    if "fake" in name:
        return 0
    if "real" in name or "true" in name:
        return 1
    return None

def load_Social() -> pd.DataFrame:

    """
    loads Social CSV folder datasets and returns a stardard DataFrame.
    """

    SOCIAL_DIR = DATA_DIR / "Social"

    # Add labels


    #add domain column


    #add source column

    # Combine datasets
    df = pd.concat([], ignore_index=True)

    #minimal cleaning 
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["text"])

    return df

if __name__ == "__main__":
    df = load_kaggle()
    print(df.head())
    print(df["label"].value_counts())