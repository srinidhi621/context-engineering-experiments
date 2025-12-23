#!/usr/bin/env python3
"""Run Exp2 generation on cache-ready levels only (no new embeddings).

Levels: 50k, 200k (all strategies).
Skips 500k/700k/950k to avoid accidental embedding rebuilds.
Optional --limit to cap runs for the day.
"""

from __future__ import annotations

import argparse

from src.experiments.exp2_pollution import PollutionExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp2 on cache-ready levels (50k/200k) without embedding builds."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of successful runs (for smoke/partial days).",
    )
    parser.add_argument(
        "--per-minute-token-limit",
        type=int,
        default=1_000_000,
        help="Per-minute token budget for generation calls (default: 1,000,000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = PollutionExperiment()
    exp.pollution_levels = [50_000, 200_000]
    exp.run(
        per_minute_token_limit=args.per_minute_token_limit,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
