# Backend/run.py
from __future__ import annotations
import json
from Backend.Pipeline.analyze import analyze_post

if __name__ == "__main__":
    text = input("Enter text (required): ").strip()
    image_path = input("Enter image path (optional - press Enter to skip): ").strip() or None
    explain = input("Explain? (y/n): ").strip().lower().startswith("y")

    out = analyze_post(text=text, image_path=image_path, explain=explain, top_n=10)
    print(json.dumps(out, indent=2))
