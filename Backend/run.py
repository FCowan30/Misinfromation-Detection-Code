# Backend/run.py
from __future__ import annotations

import json
import argparse
from Backend.Pipeline.analyze import analyze_post


def main():
    parser = argparse.ArgumentParser(description="Misinformation Detection - Unified Runner")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze (required unless using interactive mode)")
    parser.add_argument("--image", type=str, default=None, help="Optional image path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()

    # Interactive mode if --text not provided
    if not args.text:
        text = input("Enter text (required): ").strip()
        image_path = input("Enter image path (optional - press Enter to skip): ").strip()
        if image_path == "":
            image_path = None
    else:
        text = args.text.strip()
        image_path = args.image.strip() if args.image else None

    result = analyze_post(text=text, image_path=image_path)

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()