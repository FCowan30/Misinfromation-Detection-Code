from __future__ import annotations

# -----------------------------
# Standard library
# -----------------------------
from typing import Tuple

# -----------------------------
# Third-party imports (guarded)
# -----------------------------
try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as e:
    raise ImportError(
        "transformers is required for CLIP. Install with: pip install transformers"
    ) from e

from Backend.config import ARTIFACTS_DIR

# You can keep this as OpenAI CLIP, or swap later.
DEFAULT_CLIP_NAME = "openai/clip-vit-base-patch32"


def load_clip(model_name: str = DEFAULT_CLIP_NAME) -> Tuple[CLIPProcessor, CLIPModel]:
    """
    Loads CLIP processor + model for text-image similarity.
    Returns: (processor, model)
    """
    print(f"[INFO] Loading CLIP model: {model_name}")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    # put on device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("[INFO] CLIP loaded successfully.")
    print(f"[INFO] Device: {device}")

    return processor, model


if __name__ == "__main__":
    # Simple self-test
    processor, model = load_clip()
    print("[TEST] CLIP loader self-test passed.")