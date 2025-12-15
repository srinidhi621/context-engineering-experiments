#!/usr/bin/env python3
"""Run Experiment 2 on cache-ready pollution levels only (no embedding builds).

- Pollution levels: 50k, 200k, 500k, 700k (all cached)
- Skips 950k to avoid embedding RPD consumption
- Optional --limit to cap successful runs for smoke testing (default: no cap)
- Uses per-minute token limit default of 1,000,000 (override with flag)
"""

from __future__ import annotations

import argparse

from src.experiments.exp2_pollution import PollutionExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp2 on cache-ready levels (50k/200k/500k/700k) without embedding builds."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap number of successful runs (use small value for smoke test)."
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
    exp.pollution_levels = [50_000, 200_000, 500_000, 700_000]  # cached levels only
    exp.run(
        per_minute_token_limit=args.per_minute_token_limit,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
