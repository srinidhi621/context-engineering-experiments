#!/usr/bin/env python3
"""Collect the Experiment 1 model-card corpus (~700k tokens)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.corpus.loaders import load_hf_curated_models, load_hf_model_card, load_hf_model_cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the Experiment 1 Hugging Face model-card corpus."
    )
    parser.add_argument(
        "--after-date",
        default="2024-08-01",
        help="Minimum lastModified date for model cards.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700_000,
        help="Target token budget for the corpus (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="data/raw/exp1/hf_model_cards.json",
        help="Destination JSON file for the collected corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and report stats without writing the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting up to {args.max_tokens:,} tokens of model cards...")
    curated_docs = load_hf_curated_models(
        after_date=args.after_date,
        max_tokens=args.max_tokens,
    )

    documents = []
    seen = set()
    total_tokens = 0

    for doc in curated_docs:
        if doc["model_id"] in seen:
            continue
        documents.append(doc)
        seen.add(doc["model_id"])
        total_tokens += doc["tokens"]
        if total_tokens >= args.max_tokens:
            break

    if total_tokens < args.max_tokens:
        remaining = args.max_tokens - total_tokens
        print(
            f"\nCurated list reached {total_tokens:,} tokens."
            f" Fetching additional models (~{remaining:,} tokens)..."
        )
        extra_docs = load_hf_model_cards(
            after_date=args.after_date,
            max_tokens=remaining,
            tags=["text-generation"],
        )
        for doc in extra_docs:
            model_id = doc.get("model_id") or doc.get("dataset_id") or doc.get("url")
            if model_id in seen:
                continue
            documents.append(doc)
            seen.add(model_id)
            total_tokens += doc["tokens"]
            if total_tokens >= args.max_tokens:
                break

    print("\nExperiment 1 Corpus Summary")
    print("=" * 60)
    print(f"Documents collected : {len(documents)}")
    print(f"Total tokens        : {total_tokens:,}")
    print(f"After date filter   : {args.after_date}")
    print(f"Token target        : {args.max_tokens:,}")
    print("=" * 60)

    if args.dry_run:
        print("Dry run enabled; no file written.")
        return

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\nSaved Experiment 1 corpus to {output_path.resolve()}")


if __name__ == "__main__":
    main()
