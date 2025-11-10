#!/usr/bin/env python3
"""Collect Project Gutenberg padding corpus for experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.corpus.loaders import load_gutenberg_books

DEFAULT_BOOK_IDS = [
    1342,  # Pride and Prejudice
    84,    # Frankenstein
    98,    # A Tale of Two Cities
    1661,  # Sherlock Holmes
    11,    # Alice in Wonderland
    2701,  # Moby Dick
    174,   # The Picture of Dorian Gray
    1952,  # The Yellow Wallpaper
    345,   # Dracula
    74,    # Tom Sawyer
    645,   # Crime and Punishment
    1080,  # A Modest Proposal
    2600,  # War and Peace
    2852,  # The Hound of the Baskervilles
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect padding corpus from Project Gutenberg."
    )
    parser.add_argument(
        "--book-ids",
        nargs="+",
        type=int,
        default=DEFAULT_BOOK_IDS,
        help="Gutenberg book IDs to download (default: curated list).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2_000_000,
        help="Maximum total tokens to collect (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="data/raw/padding/gutenberg_corpus.json",
        help="Output JSON path for the padding corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect stats without writing the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    books = load_gutenberg_books(args.book_ids, max_tokens=args.max_tokens)
    total_tokens = sum(book["tokens"] for book in books)

    print("\nPadding Corpus Summary")
    print("=" * 60)
    print(f"Books retrieved  : {len(books)}")
    print(f"Total tokens     : {total_tokens:,}")
    print(f"Token target     : {args.max_tokens:,}")
    print(f"Output path      : {output_path}")
    print("=" * 60)

    if args.dry_run:
        print("Dry run enabled; no file written.")
        return

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(books, f, indent=2, ensure_ascii=False)

    print(f"\nSaved padding corpus to {output_path.resolve()}")


if __name__ == "__main__":
    main()
