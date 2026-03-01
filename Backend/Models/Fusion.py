# Backend/Models/Fusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

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
    # Similarity is cosine in [-1, 1]. We map to [0,1] using (sim + 1)/2.
    mismatch_flag_threshold: float = 0.45  # below this, image-text looks mismatched

    # Driver tie band (if contributions are close, call it "both")
    driver_tie_band: float = 0.05


def _cosine_to_01(sim: float) -> float:
    """Map cosine similarity [-1,1] -> [0,1] and clamp."""
    x = (sim + 1.0) / 2.0
    return max(0.0, min(1.0, x))


def fuse_multimodal(
    text: str,
    image_path: str,
    cfg: Optional[FusionConfig] = None,
) -> Dict[str, Any]:
    """
    Multimodal fusion:
    - Text side: DistilBERT probability of FAKE
    - Image-text side: CLIP alignment; converted to mismatch probability
    Final: weighted average of (text fake prob) and (mismatch prob)

    Returns a JSON-friendly dict suitable for your API/dashboard.
    """
    cfg = cfg or FusionConfig()

    # --- 1) Text prediction (DistilBERT) ---
    text_out = predict_text(text)
    p_fake_text = float(text_out["probs"]["FAKE"])
    p_true_text = float(text_out["probs"]["TRUE"])
    text_label = str(text_out["label"])
    text_conf = float(text_out["confidence"])

    # --- 2) Image-text alignment (CLIP) ---
    sim = float(clip_similarity(image_path, text))
    sim_01 = _cosine_to_01(sim)

    # Interpret CLIP as "support" for consistency between image and text.
    # Low similarity => higher mismatch probability => possible misleading pairing.
    p_mismatch = 1.0 - sim_01
    mismatch_flag = sim_01 < cfg.mismatch_flag_threshold

    # --- 3) Fusion ---
    # Multimodal fake probability blends: text fake prob + mismatch prob
    p_fake_mm = (cfg.w_text * p_fake_text) + (cfg.w_clip * p_mismatch)

    final_label = "FAKE" if p_fake_mm >= cfg.fake_threshold else "TRUE"
    final_conf = p_fake_mm if final_label == "FAKE" else (1.0 - p_fake_mm)

    # --- 4) Driver / contribution breakdown (NEW) ---
    text_contrib = cfg.w_text * p_fake_text
    clip_contrib = cfg.w_clip * p_mismatch

    if abs(text_contrib - clip_contrib) < cfg.driver_tie_band:
        driver = "both"
    elif text_contrib > clip_contrib:
        driver = "text"
    else:
        driver = "image_mismatch"

    # --- 5) Disagreement signals (useful for debugging + NLG later) ---
    disagreement = False
    if text_label == "TRUE" and mismatch_flag:
        disagreement = True  # text says true, but image-text mismatch suggests misleading pairing
    if text_label == "FAKE" and sim_01 > 0.70:
        disagreement = True  # text says fake, but image strongly matches claim (possible false positive)

    return {
        "final": {
            "label": final_label,
            "confidence": float(final_conf),
            "p_fake_multimodal": float(p_fake_mm),
            "threshold": float(cfg.fake_threshold),
        },
        "text_model": {
            "label": text_label,
            "probs": {"FAKE": p_fake_text, "TRUE": p_true_text},
            "confidence": text_conf,
        },
        "clip_model": {
            "similarity_cosine": sim,
            "similarity_01": float(sim_01),
            "p_mismatch": float(p_mismatch),
            "mismatch_flag": bool(mismatch_flag),
            "mismatch_flag_threshold": float(cfg.mismatch_flag_threshold),
        },
        "signals": {
            "disagreement": bool(disagreement),
            "weights": {"w_text": float(cfg.w_text), "w_clip": float(cfg.w_clip)},
            "driver": driver,
            "contributions": {
                "text": float(text_contrib),
                "image_mismatch": float(clip_contrib),
            },
        },
        "inputs": {"text": text, "image_path": image_path},
    }


if __name__ == "__main__":
    img = input("Enter image path: ").strip()
    txt = input("Enter claim/text: ").strip()

    out = fuse_multimodal(txt, img)
    print("\n[FUSION RESULT]")
    print(out["final"])
    print("Text:", out["text_model"])
    print("CLIP:", out["clip_model"])
    print("Signals:", out["signals"])