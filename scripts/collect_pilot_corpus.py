#!/usr/bin/env python3
"""Collect a small Hugging Face corpus for the pilot experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.corpus.loaders import load_hf_curated_models


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pilot corpus collection."""
    parser = argparse.ArgumentParser(
        description="Collect curated Hugging Face model cards for the pilot corpus."
    )
    parser.add_argument(
        "--after-date",
        default="2024-08-01",
        help="Only include model cards modified after this ISO date (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10_000,
        help="Maximum total tokens to collect (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="data/raw/pilot/hf_model_cards.json",
        help="Path where the collected corpus JSON will be written (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and report stats without writing the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    documents = load_hf_curated_models(
        after_date=args.after_date,
        max_tokens=args.max_tokens,
    )
    total_tokens = sum(doc["tokens"] for doc in documents)

    print("\nCollection summary")
    print("=" * 60)
    print(f"Documents collected : {len(documents)}")
    print(f"Total tokens        : {total_tokens:,}")
    print(f"After date filter   : {args.after_date}")
    print(f"Token target        : {args.max_tokens:,}")
    print("=" * 60)

    if args.dry_run:
        print("Dry run enabled; no file written.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\nSaved corpus to {output_path.resolve()}")


if __name__ == "__main__":
    main()
