#!/usr/bin/env python3
"""Unified experiment runner scaffold."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from src.config import config

EXPERIMENT_CHOICES = ("pilot", "exp1", "exp2", "exp5")


def parse_args() -> argparse.Namespace:
    env_limit = int(os.getenv("PER_MINUTE_TOKEN_LIMIT", "240000"))
    parser = argparse.ArgumentParser(
        description="Run pilot or production experiments with shared options."
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_CHOICES,
        default="pilot",
        help="Experiment identifier to run (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override default output path for the selected experiment.",
    )
    parser.add_argument(
        "--fill-pcts",
        type=float,
        nargs="+",
        default=config.fill_percentages,
        help="Fill percentages to target (default: %(default)s).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=config.repetitions,
        help="Repetitions per condition (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.temperature,
        help="Generation temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--per-minute-token-limit",
        type=int,
        default=env_limit,
        help=(
            "Input token budget per rolling minute. "
            "Defaults to PER_MINUTE_TOKEN_LIMIT env or %(default)s."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the run without executing API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Experiment: {args.experiment}\n"
        f"Model: {config.model_name}\n"
        f"Fill %s: {', '.join(f'{p:.0%}' for p in args.fill_pcts)}\n"
        f"Per-minute token limit: {args.per_minute_token_limit:,}\n"
    )

    if args.experiment == "pilot":
        run_pilot_handler(args)
    else:
        print(
            f"Experiment '{args.experiment}' is not implemented yet. "
            "Use --dry-run to verify configuration or update the handler "
            "in scripts/run_experiment.py."
        )


def run_pilot_handler(args: argparse.Namespace) -> None:
    """
    Delegate pilot execution to scripts/run_minimal_pilot.py with shared flags.
    """
    script = Path(__file__).with_name("run_minimal_pilot.py")
    cmd: List[str] = [sys.executable, str(script)]

    cmd.extend(["--per-minute-token-limit", str(args.per_minute_token_limit)])
    cmd.extend(["--repetitions", str(args.repetitions)])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--max-output-tokens", str(args.max_output_tokens)])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.fill_pcts:
        cmd.append("--fill-pcts")
        cmd.extend(str(p) for p in args.fill_pcts)
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Executing pilot via {script} ...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
