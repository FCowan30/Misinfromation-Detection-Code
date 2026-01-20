from pathlib import Path

# -------------------------------------------------
# Project root (folder that contains Backend/)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Raw data directory (NOT committed to git)
# -------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------
# Generated artifacts (models, tokenized datasets)
# -------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TOKENIZED_DIR = ARTIFACTS_DIR / "tokenized"
MODEL_DIR = ARTIFACTS_DIR / "model"
CLIP_CACHE_DIR = ARTIFACTS_DIR / "clip_cache"

# -------------------------------------------------
# Dataset subdirectories (raw)
# -------------------------------------------------
KAGGLE_DIR = DATA_DIR / "kaggle"
FEVER_DIR = DATA_DIR / "fever"
PUBHEALTH_DIR = DATA_DIR / "pubhealth"
SOCIAL_DIR = DATA_DIR / "social"

# -------------------------------------------------
# Reproducibility / defaults
# -------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2

# -------------------------------------------------
# Model defaults
# -------------------------------------------------
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
