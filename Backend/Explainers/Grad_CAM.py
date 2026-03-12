from __future__ import annotations

import os
from typing import Dict, Any, Optional

try:
    import numpy as np
except ImportError as e:
    raise ImportError("NumPy is required for Grad-CAM. Install with: pip install numpy") from e

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("PyTorch is required for Grad-CAM. Install with: pip install torch") from e

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required for Grad-CAM. Install with: pip install pillow") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required for Grad-CAM. Install with: pip install matplotlib") from e

try:
    from Backend.Models.CLIP_loader import load_clip
except ImportError as e:
    raise ImportError(
        "Failed to import CLIP loader. Ensure Backend/Models/CLIP_loader.py exists."
    ) from e


# -----------------------------
# Helpers
# -----------------------------
def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    max_val = cam.max()
    if max_val > 0:
        cam = cam / max_val
    return cam


def _region_from_peak(row: int, col: int, h: int, w: int) -> str:
    """
    Convert heatmap peak location into a human-readable region.
    """
    row_third = h / 3
    col_third = w / 3

    if row < row_third:
        vertical = "upper"
    elif row < 2 * row_third:
        vertical = "central"
    else:
        vertical = "lower"

    if col < col_third:
        horizontal = "left"
    elif col < 2 * col_third:
        horizontal = "centre"
    else:
        horizontal = "right"

    if vertical == "central" and horizontal == "centre":
        return "the centre of the image"
    return f"the {vertical} {horizontal} region of the image"


def _save_overlay(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.45,
) -> None:
    """
    Save a simple heatmap overlay.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# -----------------------------
# Main Grad-CAM-style function
# -----------------------------
def generate_gradcam(
    image_path: str,
    text: str,
    output_dir: str = "FrontEnd/public/gradcam",
    output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a Grad-CAM-style heatmap for CLIP ViT image-text similarity.

    Since CLIP ViT does not have a final convolutional layer, this computes
    gradient-weighted relevance over the final patch embeddings of the visual transformer.
    """
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    if not text or not str(text).strip():
        raise ValueError("Text input is required for Grad-CAM generation.")

    clip_loaded = load_clip()

    if len(clip_loaded) == 3:
        processor, model, device = clip_loaded
    elif len(clip_loaded) == 2:
        processor, model = clip_loaded
        device = next(model.parameters()).device
    else:
        raise ValueError(f"Unexpected number of values returned by load_clip(): {len(clip_loaded)}")

    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.enable_grad():
        # Text branch
        text_features = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Vision branch with hidden states
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        # Support both dict-like and tuple-like outputs from different transformers versions
        if hasattr(vision_outputs, "last_hidden_state"):
            last_hidden_state = vision_outputs.last_hidden_state
            pooled = vision_outputs.pooler_output
        elif isinstance(vision_outputs, (tuple, list)):
            # Typical order: [last_hidden_state, pooled_output, hidden_states, ...]
            last_hidden_state = vision_outputs[0]
            pooled = vision_outputs[1]
        else:
            raise RuntimeError("Unexpected output type from CLIP vision model.")

        last_hidden_state.retain_grad()

        # Project vision features
        image_features = model.visual_projection(pooled)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Similarity objective
        similarity = (image_features * text_features).sum()

        model.zero_grad()
        similarity.backward()

        grads = last_hidden_state.grad   # [1, tokens, dim]
        acts = last_hidden_state         # [1, tokens, dim]

    if grads is None:
        raise RuntimeError("Gradients were not captured for CLIP vision hidden states.")

    # Remove CLS token, keep patch tokens only
    patch_grads = grads[:, 1:, :]   # [1, num_patches, dim]
    patch_acts = acts[:, 1:, :]     # [1, num_patches, dim]

    num_patches = patch_acts.shape[1]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"Could not reshape {num_patches} patch tokens into a square grid.")

    # Grad-CAM-style weighting
    weights = patch_grads.mean(dim=1, keepdim=True)   # [1, 1, dim]
    cam_tokens = (patch_acts * weights).sum(dim=-1)   # [1, num_patches]

    # For CLIP ViT, signed relevance can cancel out heavily.
    # Use absolute magnitude to avoid all-zero maps.
    cam_tokens = torch.abs(cam_tokens)

    cam = cam_tokens[0].detach().cpu().numpy().reshape(grid_size, grid_size)
    cam = _normalize_map(cam)

    if float(cam.max()) == 0.0:
        # fallback: use gradient magnitude only
        cam_tokens = patch_grads.abs().mean(dim=-1)   # [1, num_patches]
        cam = cam_tokens[0].detach().cpu().numpy().reshape(grid_size, grid_size)
        cam = _normalize_map(cam)

        if float(cam.max()) == 0.0:
            return {
                "heatmap_path": None,
                "top_region_summary": "The visual explanation could not identify a strong image region for this prediction.",
                "activation_strength": 0.0,
                "peak_location": {"row": 0, "col": 0},
                "grid_shape": {"height": int(grid_size), "width": int(grid_size)},
           }

    # Upsample to original image size
    cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(
        cam_t,
        size=(image_np.shape[0], image_np.shape[1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    cam_up = _normalize_map(cam_up)

    # Peak region summary
    peak_row, peak_col = np.unravel_index(np.argmax(cam_up), cam_up.shape)
    region_text = _region_from_peak(peak_row, peak_col, cam_up.shape[0], cam_up.shape[1])

    activation_strength = float(cam_up.max())

    if output_name is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        safe_base = base.replace(" ", "_")
        output_name = f"{safe_base}_gradcam.png"

    output_path = os.path.join(output_dir, output_name)
    _save_overlay(image_np, cam_up, output_path)

    return {
        "heatmap_path": output_path.replace("\\", "/"),
        "top_region_summary": (
            f"The visual explanation indicates that the model focused mainly on {region_text} "
            f"when comparing the image with the text."
        ),
        "activation_strength": activation_strength,
        "peak_location": {"row": int(peak_row), "col": int(peak_col)},
        "grid_shape": {"height": int(grid_size), "width": int(grid_size)},
    }


# -----------------------------
# CLI test
# -----------------------------
if __name__ == "__main__":
    img = input("Enter image path: ").strip()
    txt = input("Enter text/claim: ").strip()

    out = generate_gradcam(img, txt)
    print("\n[GRAD-CAM RESULT]")
    print(out)