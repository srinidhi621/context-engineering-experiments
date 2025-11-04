#!/usr/bin/env python3
"""Assemble pilot contexts with padding for predefined fill percentages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from src.context_engineering.naive import NaiveContextAssembler
from src.corpus.padding import PaddingGenerator
from src.utils.tokenizer import count_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble pilot contexts using naive concatenation and Gutenberg padding."
    )
    parser.add_argument(
        "--input",
        default="data/raw/pilot/hf_model_cards.json",
        help="Path to the collected pilot corpus JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/pilot/naive_contexts.json",
        help="Destination for the assembled contexts (default: %(default)s).",
    )
    parser.add_argument(
        "--fill-pcts",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Fill percentages to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=1_000_000,
        help="Maximum context window size used for padding (default: %(default)s).",
    )
    return parser.parse_args()


def load_documents(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input corpus must be a list of documents.")
    return data


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    documents = load_documents(input_path)
    if not documents:
        raise RuntimeError(f"No documents found in {input_path}")

    assembler = NaiveContextAssembler()
    padding = PaddingGenerator()

    base_context = assembler.assemble(documents, args.max_context_tokens)
    base_tokens = count_tokens(base_context)

    assembled = []
    for fill_pct in args.fill_pcts:
        padded_context = padding.pad_to_fill_percentage(
            base_context,
            fill_pct=fill_pct,
            max_context_tokens=args.max_context_tokens,
        )
        assembled.append(
            {
                "strategy": "naive",
                "fill_pct": fill_pct,
                "context": padded_context,
                "tokens": count_tokens(padded_context),
                "padding_tokens": max(count_tokens(padded_context) - base_tokens, 0),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(assembled, f, indent=2, ensure_ascii=False)

    print(f"Base context tokens: {base_tokens:,}")
    for item in assembled:
        print(
            f"Fill {item['fill_pct']:.0%}: "
            f"{item['tokens']:,} tokens (padding {item['padding_tokens']:,})"
        )
    print(f"\nSaved assembled contexts to {output_path.resolve()}")


if __name__ == "__main__":
    main()
