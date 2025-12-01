"""
Main runner for Experiment 1: Needle in Multiple Haystacks.
Uses the robust NeedleExperiment class for execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Set

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.exp1_needle import NeedleExperiment
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_run_keys(runs_file: Optional[str]) -> Optional[Set[str]]:
    """Load run keys from a JSON file (list or {pending_runs:[...]}) if provided."""
    if not runs_file:
        return None

    path = Path(runs_file)
    if not path.exists():
        raise FileNotFoundError(f"Runs file does not exist: {runs_file}")

    with open(path, "r") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        if "pending_runs" in data:
            run_list = data["pending_runs"]
        elif "runs" in data:
            run_list = data["runs"]
        else:
            raise ValueError(
                f"Unexpected JSON format in {runs_file}. "
                "Expected a list or a dict with 'pending_runs'."
            )
    elif isinstance(data, list):
        run_list = data
    else:
        raise ValueError(f"Unsupported runs file format: {type(data).__name__}")

    return set(run_list)


def run_experiment_1(
    dry_run: bool = False,
    per_minute_token_limit: int = 240000,
    limit: Optional[int] = None,
    runs_file: Optional[str] = None,
) -> None:
    try:
        experiment = NeedleExperiment()
        run_keys = _load_run_keys(runs_file)
        experiment.run(
            dry_run=dry_run,
            limit=limit,
            per_minute_token_limit=per_minute_token_limit,
            target_run_keys=run_keys,
        )
    except Exception as exc:  # pragma: no cover - CLI wrapper
        logger.critical("Experiment 1 failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: Needle in Multiple Haystacks."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the run without making actual API calls.",
    )
    parser.add_argument(
        "--per-minute-token-limit", type=int, default=240000, help="Token limit per minute."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of successful runs (for testing)."
    )
    parser.add_argument(
        "--runs-file",
        default=None,
        help="Optional JSON file containing run keys to execute (list or {'pending_runs': [...]})",
    )
    cli_args = parser.parse_args()

    run_experiment_1(
        dry_run=cli_args.dry_run,
        per_minute_token_limit=cli_args.per_minute_token_limit,
        limit=cli_args.limit,
        runs_file=cli_args.runs_file,
    )
