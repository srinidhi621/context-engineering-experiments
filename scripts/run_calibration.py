#!/usr/bin/env python3
"""Baseline calibration runner scaffold."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import config
from src.utils.throttle import PerMinuteTokenThrottle


def parse_args() -> argparse.Namespace:
    env_limit = int(os.getenv("PER_MINUTE_TOKEN_LIMIT", "240000"))
    parser = argparse.ArgumentParser(
        description="Calibrate fill levels and latency baselines before experiments."
    )
    parser.add_argument(
        "--output",
        default="results/calibration_plan.json",
        help="Path to store calibration metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--per-minute-token-limit",
        type=int,
        default=env_limit,
        help="Input token guardrail used for planning (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned calibration steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    throttle = PerMinuteTokenThrottle(args.per_minute_token_limit)
    summary = {
        "model": config.model_name,
        "fill_percentages": config.fill_percentages,
        "per_minute_token_limit": args.per_minute_token_limit,
        "repetitions": config.repetitions,
    }

    print("Calibration Summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")

    if args.dry_run:
        print("\nDry run mode: no calibration API calls executed.")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "Calibration dry run completed.\n", encoding="utf-8"
        )
        return

    raise NotImplementedError(
        "Calibration logic is pending implementation. "
        "Use --dry-run to verify configuration."
    )


if __name__ == "__main__":
    main()
