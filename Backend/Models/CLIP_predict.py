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

    print(f"[INFO] loading CLIP model: {model_name}")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    #put on deice (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("[INFO] CLIP loaded successfully.")
    print(f"[INFO] Deice: {device}")

    return processor, model

if __name__ == "__main__":
    # Test loading the model
    try:
        processor, model = load_clip()
        print("[TEST] CLIP loaded and ready for inference.")
    except Exception as e:
        print(f"[ERROR] Failed to load CLIP: {e}")