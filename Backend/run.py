# Backend/run.py
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from typing import Optional

from Backend.Pipeline.analyze import analyze_post


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the misinformation detection pipeline (text-only or multimodal)."
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text input (required for non-interactive mode).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path. If omitted, runs text-only mode.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="If set, include SHAP explanation in output.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of SHAP tokens to return (default: 10).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    return parser.parse_args()


def _interactive_prompt() -> tuple[str, Optional[str], bool]:
    text = input("Enter text (required): ").strip()
    image_path = input("Enter image path (optional - press Enter to skip): ").strip() or None
    explain = input("Explain? (y/n): ").strip().lower().startswith("y")
    return text, image_path, explain


def main() -> int:
    args = _parse_args()

    # -------------------------
    # INTERACTIVE MODE
    # -------------------------
    if args.text is None:
        text, image_path, explain = _interactive_prompt()
        out = analyze_post(text=text, image_path=image_path, explain=explain, top_n=args.top_n)
        print(json.dumps(out, indent=2))
        return 0

    # -------------------------
    # CLI / FRONTEND MODE
    # -------------------------
    text = (args.text or "").strip()
    image_path = args.image.strip() if args.image else None
    explain = bool(args.explain)

    if not text:
        # In CLI mode, print a machine-readable error to stderr and exit non-zero.
        err = {"error": "Text input is required. Provide --text or run without args for interactive mode."}
        print(json.dumps(err), file=sys.stderr)
        return 2

    # IMPORTANT:
    # Redirect stdout to stderr during model execution so any internal print()
    # statements (e.g., [INFO] model loading) do not pollute stdout.
    # We then print *only* the JSON result to stdout.
    try:
        with redirect_stdout(sys.stderr):
            out = analyze_post(text=text, image_path=image_path, explain=explain, top_n=args.top_n)
    except Exception as e:
        err = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(err), file=sys.stderr)
        return 1

    if args.pretty:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())