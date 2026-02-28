# Backend/Models/Fusion.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

# Your existing modules
from Backend.Models.DistillBERT_Predict import predict_text
from Backend.Models.CLIP_predict import clip_similarity


@dataclass
class FusionConfig:
    # How much each modality contributes to the final "misinformation probability"
    w_text: float = 0.7
    w_clip: float = 0.3

    # Decision threshold for final label
    fake_threshold: float = 0.5

    # Optional: treat low similarity as "mismatch risk"
    # Similarity is cosine in [-1, 1], but usually sits ~[0, 1] for matched content.
    # We convert cosine -> [0, 1] using (sim + 1) / 2.
    # You can also set a "mismatch band" to flag uncertainty.
    mismatch_flag_threshold: float = 0.45  # below this, image-text looks mismatched


def _cosine_to_01(sim: float) -> float:
    # Map cosine similarity from [-1, 1] to [0, 1]
    x = (sim + 1.0) / 2.0
    # Clamp just in case
    return max(0.0, min(1.0, x))


def fuse_multimodal(
    text: str,
    image_path: str,
    cfg: Optional[FusionConfig] = None,
) -> Dict[str, Any]:
    """
    Multimodal fusion for misinformation:
    - Text side: DistilBERT probability of FAKE
    - Image-text side: CLIP mismatch probability derived from similarity
    Final: weighted average

    Returns a JSON-friendly dict suitable for an API response.
    """
    cfg = cfg or FusionConfig()

    # --- 1) Text prediction (DistilBERT) ---
    text_out = predict_text(text)
    p_fake_text = float(text_out["probs"]["FAKE"])
    p_true_text = float(text_out["probs"]["TRUE"])

    # --- 2) Image-text alignment (CLIP) ---
    sim = float(clip_similarity(image_path, text))
    sim_01 = _cosine_to_01(sim)

    # Interpret CLIP as "support" for the text being consistent with the image.
    # If image and text mismatch, that can indicate misleading multimodal content.
    p_mismatch = 1.0 - sim_01

    # --- 3) Fusion ---
    # "Multimodal fake probability" blends: text fake prob + mismatch prob
    p_fake_mm = (cfg.w_text * p_fake_text) + (cfg.w_clip * p_mismatch)

    final_label = "FAKE" if p_fake_mm >= cfg.fake_threshold else "TRUE"
    final_conf = p_fake_mm if final_label == "FAKE" else (1.0 - p_fake_mm)

    # --- 4) Helpful flags for the UI / evaluation ---
    text_label = text_out["label"]
    mismatch_flag = sim_01 < cfg.mismatch_flag_threshold

    # Simple “disagreement” indicators (useful for debugging + dashboard)
    disagreement = False
    if text_label == "TRUE" and mismatch_flag:
        disagreement = True  # text looks true, but image-text mismatch suggests misleading pairing
    if text_label == "FAKE" and sim_01 > 0.70:
        disagreement = True  # text looks fake, but image strongly matches claim (possible false positive)

    return {
        "final": {
            "label": final_label,
            "confidence": float(final_conf),
            "p_fake_multimodal": float(p_fake_mm),
            "threshold": cfg.fake_threshold,
        },
        "text_model": {
            "label": text_label,
            "probs": {"FAKE": p_fake_text, "TRUE": p_true_text},
            "confidence": float(text_out["confidence"]),
        },
        "clip_model": {
            "similarity_cosine": sim,
            "similarity_01": float(sim_01),
            "p_mismatch": float(p_mismatch),
            "mismatch_flag": bool(mismatch_flag),
            "mismatch_flag_threshold": cfg.mismatch_flag_threshold,
        },
        "signals": {
            "disagreement": bool(disagreement),
            "weights": {"w_text": cfg.w_text, "w_clip": cfg.w_clip},
        },
        "inputs": {"text": text, "image_path": image_path},
    }


if __name__ == "__main__":
    # Quick local test
    img = input("Enter image path: ").strip()
    txt = input("Enter claim/text: ").strip()

    out = fuse_multimodal(txt, img)
    print("\n[FUSION RESULT]")
    print(out["final"])
    print("Text:", out["text_model"])
    print("CLIP:", out["clip_model"])
    print("Signals:", out["signals"])