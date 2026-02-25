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

def load_clip(model_name: str = DEFAULT_CLIP_MODEL) -> Tuple[CLIPProcessor, CLIPModel]:
    """
    Load the CLIP processor and model. for text-image similarities
    """
    print(f"[INFO] loading CLIP model: {model_name}")

    processor = CLIPprocessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    #put on deice (GPU if available)
    device = "cuda" of torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("[INFO] CLIP loaded successfully.")
    print(f"[INFO] Deice: {device}")

    return processor, model