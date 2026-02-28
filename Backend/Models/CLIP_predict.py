from __future__ import annotations

# -----------------------------
# Third-party imports (guarded)
# -----------------------------
try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required. Install with: pip install pillow") from e

from Backend.Models.CLIP_loader import load_clip


def clip_similarity(
    image_path: str,
    text: str,
) -> float:
    """
    Returns CLIP similarity score between image and text.
    Higher = more aligned.
    """
    processor, model = load_clip()
    device = next(model.parameters()).device

    # Load images
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image not found: {image_path}") from e

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        # embeddings (text/image)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # cosine similarity
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        sim = (image_embeds * text_embeds).sum(dim=-1).item()

    return float(sim)


def predict_clip_match(
    image_path: str,
    text: str,
    threshold: float = 0.25,
) -> dict:
    """
    Simple decision: if similarity >= threshold -> 'MATCH' else 'MISMATCH'
    """
    score = clip_similarity(image_path, text)
    verdict = "MATCH" if score >= threshold else "MISMATCH"

    return {
        "text": text,
        "image_path": image_path,
        "similarity": score,
        "threshold": threshold,
        "verdict": verdict,
    }


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    text = input("Enter text to compare: ").strip()

    out = predict_clip_match(image_path, text, threshold=0.25)

    print("\n[CLIP RESULT]")
    print("Verdict:", out["verdict"])
    print("Similarity:", round(out["similarity"], 4))
    print("Threshold:", out["threshold"])