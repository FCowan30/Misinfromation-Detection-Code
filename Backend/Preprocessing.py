    # import pandas with error handling.
try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Pandas is required to run this script (Preproecessing.py)." \
        "Please install it using 'pip install pandas'."
) from e 

def Load_training_data():
    # load dataset
    # ensure Files exist, are accessible, and not empty.
    try:
        df_fake= pd.read_csv("data/fake.csv")
        df_true = pd.read_csv("data/True.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Dataset files not found. Please ensure 'fake.csv' and 'True.csv' are present in the 'data' directory."
        ) from e 
    except pd.errors.EmptyDataError as e:
        raise ValueError(
            "One of the dataset files is empty. Please check the files in the 'data' directory."
        ) from e
    except Exception as e:
        raise Exception(
            f"An error occurred while loading the dataset files: {e}."
        ) from e
    
    #ensure Files are not empty
    if df_fake.empty:
        raise ValueError("The fake news dataset is empty.")
    
    if df_true.empty:
        raise ValueError("The true news dataset is empty.")

    print("Dataset loaded successfully.")

    return df_fake, df_true

def label_combine_data(df_fake, df_true):
    # Add labels
    df_fake["label"] = 0
    df_true["label"] = 1

    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    print("Data preprocessed successfully.")
    print("Dataset shape:", df.shape)

    return df

def data_cleaning(df):
    # Drop unnecessary columns
    columns_to_drop = ["title", "subject", "date"]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Check for missing values
    if df.isnull().values.any():
        df = df.dropna().reset_index(drop=True)
        print("Missing values found and removed.")
    else:
        print("No missing values found.")
    
    #check for duplicated rows
    if df.duplicated().any():
        df = df.drop_duplicates().reset_index(drop=True)
        print("Duplicate rows found and removed.")

    print("Data cleaning completed.")
    print("Cleaned dataset shape:", df.shape)

    return df

def build_tokenized_splits(
    df,
    model_name: str = "distilbert-base-uncased",
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    seed: int = 42,
    max_length: int = 256,
):
    """
    Converts a pandas DataFrame into Hugging Face train/eval tokenized datasets
    suitable for Trainer, using dynamic padding via DataCollatorWithPadding.
    """

    # Dependency checks (clear user-facing errors)
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "Missing dependency: datasets\n"
            "Install it with: pip install datasets"
        ) from e

    try:
        from transformers import AutoTokenizer, DataCollatorWithPadding
    except ImportError as e:
        raise ImportError(
            "Missing dependency: transformers\n"
            "Install it with: pip install transformers"
        ) from e

    # Basic input validation (prevents confusing runtime errors)
    if text_col not in df.columns:
        raise ValueError(f"DataFrame is missing required text column: '{text_col}'")
    if label_col not in df.columns:
        raise ValueError(f"DataFrame is missing required label column: '{label_col}'")

    # Load tokenizer (gives useful message if model can't be fetched)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{model_name}'.\n"
            "Check the model name and your internet connection (or cache)."
        ) from e

    def tokenize_function(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
        )

    # Convert to HF Dataset and split
    hf_dataset = Dataset.from_pandas(df)
    splits = hf_dataset.train_test_split(test_size=test_size, seed=seed)

    # Tokenize
    tokenized = splits.map(tokenize_function, batched=True)

    # Keep only columns Trainer needs
    keep_cols = {"input_ids", "attention_mask", label_col}
    remove_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)

    # Rename label column to exactly "label" if needed (Trainer expects "label")
    if label_col != "label":
        tokenized = tokenized.rename_column(label_col, "label")

    # Torch format
    tokenized.set_format("torch")

    # Dynamic padding per batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = tokenized["train"]
    eval_ds = tokenized["test"]

    print(train_ds[0].keys())
    print("Tokenized datasets created successfully.")


    return train_ds, eval_ds, tokenizer, data_collator

def CLIP_tokenize_text_Image(df):
    # Placeholder for CLIP tokenization logic
    
    return df

# Testing the function directly from file.
if __name__ == "__main__":
    df_Fake, df_True = Load_training_data()
    df = label_combine_data(df_Fake, df_True)
    df_cleaned = data_cleaning(df)
    Cleaned_Tokenised_df = build_tokenized_splits(df_cleaned)