#!/usr/bin/env python3
"""
Main runner for Experiment 1: Needle in Multiple Haystacks.
Uses the robust NeedleExperiment class for execution.
"""

import argparse
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments.exp1_needle import NeedleExperiment
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_experiment_1(dry_run: bool = False, per_minute_token_limit: int = 240000, limit: int = None):
    try:
        experiment = NeedleExperiment()
        experiment.run(
            dry_run=dry_run, 
            limit=limit, 
            per_minute_token_limit=per_minute_token_limit
        )
    except Exception as e:
        logger.critical(f"Experiment 1 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: Needle in Multiple Haystacks.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the run without making actual API calls.")
    parser.add_argument("--per-minute-token-limit", type=int, default=240000, help="Token limit per minute.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of successful runs (for testing).")
    args = parser.parse_args()
    
    run_experiment_1(
        dry_run=args.dry_run, 
        per_minute_token_limit=args.per_minute_token_limit, 
        limit=args.limit
    )
