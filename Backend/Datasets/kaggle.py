import pandas as pd
from pathlib import Path
from Backend.config import DATA_DIR

def load_kaggle() -> pd.DataFrame:

    """
    loads Kaggle True/Fake news dataset and returns a stardard DataFrame.
    """

    kaggle_dir = DATA_DIR / "Kaggle"
    fake_path = kaggle_dir / "Fake.csv"
    true_path = kaggle_dir / "True.csv"

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Add labels
    df_fake["label"] = 0
    df_true["label"] = 1

    #add domain column
    df_fake["domain"] = "news"
    df_true["domain"] = "news"

    #add source column
    df_fake["source"] = "kaggle"
    df_true["source"] = "kaggle"

    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    #minimal cleaning 
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["text"])

    return df

if __name__ == "__main__":
    df = load_kaggle()
    print(df.head())
    print(df["label"].value_counts())
