from __future__ import annotations 

#-----------------------------
#Standard library imports
#-----------------------------

from typing import Tuple

#-----------------------------
#Third-party imports (guarded)
#-----------------------------

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for CLIP inference. "
        "Install it with: pip install torch"
    ) from e

from Backend.config import ARTIFACTS_DIR

# You can keep this as OpenAI ClIP or swap later if you want to try a different CLIP model
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
