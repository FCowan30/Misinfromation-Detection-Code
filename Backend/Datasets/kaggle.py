try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Pandas is required to run this script (Preproecessing.py)." \
        "Please install it using 'pip install pandas'."
) from e 

from pathlib import Path
from Backend.config import DATA_DIR

def load_kaggle() -> pd.DataFrame:

    """
    loads Kaggle True/Fake news dataset and returns a stardard DataFrame.
    """


    return df
