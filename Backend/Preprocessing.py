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

def BERT_tokenize_text(df):
    # Placeholder for BERT tokenization logic
    
    return df

def CLIP_tokenize_text_Image(df):
    # Placeholder for CLIP tokenization logic
    
    return df

# Testing the function directly from file.
if __name__ == "__main__":
    df_Fake, df_True = Load_training_data()
    df = label_combine_data(df_Fake, df_True)
    df_cleaned = data_cleaning(df)