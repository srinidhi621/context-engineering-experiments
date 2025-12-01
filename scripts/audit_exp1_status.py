#!/usr/bin/env python3
"""
Audit Experiment 1 status and regenerate pending run metadata.

Outputs:
- results/raw/exp1_failure_breakdown.json
- results/raw/exp1_pending_runs.json

Summary metrics are printed to stdout for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence, Set


def load_questions(path: Path) -> Sequence[Dict[str, str]]:
    with open(path, "r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise ValueError(f"Questions file must be a list. Got {type(data).__name__}")


def build_expected_run_keys(questions: Sequence[Dict[str, str]]) -> Set[str]:
    strategies = ["naive", "structured", "rag", "advanced_rag"]
    fill_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    repetitions = range(3)

    expected: Set[str] = set()
    for q in questions:
        qid = q["question_id"]
        for strat in strategies:
            for fill in fill_levels:
                for rep in repetitions:
                    expected.add(f"{qid}_{strat}_{fill}_{rep}")
    return expected


def load_status_completed(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with open(path, "r") as handle:
        data = json.load(handle)
    return set(data.get("completed_keys", []))


def load_results_completed(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = "{question_id}_{strategy}_{fill_pct}_{repetition}".format(**payload)
            keys.add(key)
    return keys


def parse_failure_breakdown(log_path: Path) -> Dict[str, Set[str]]:
    """
    Parse experiment1.log to categorize failures.
    """
    breakdown = {
        "token_limit": set(),
        "resource_exhausted": set(),
        "other": set(),
    }
    if not log_path.exists():
        return breakdown

    run_pattern = re.compile(r"Run \d+/\d+: (exp1_[^\s]+)")
    fail_pattern = re.compile(r"API Failed for (exp1_[^:]+)")

    current_run: str | None = None
    resource_flag = False
    with open(log_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            run_match = run_pattern.search(line)
            if run_match:
                current_run = run_match.group(1)

            if "Token limit exceeded" in line and current_run:
                breakdown["token_limit"].add(current_run)

            if "ResourceExhausted" in line:
                resource_flag = True

            fail_match = fail_pattern.search(line)
            if fail_match:
                failed_key = fail_match.group(1)
                if resource_flag:
                    breakdown["resource_exhausted"].add(failed_key)
                else:
                    breakdown["other"].add(failed_key)
                resource_flag = False

    return breakdown


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def summarize(
    expected: Set[str],
    completed_status: Set[str],
    completed_results: Set[str],
    breakdown: Dict[str, Set[str]],
) -> Dict[str, Any]:
    pending_status = sorted(expected - completed_status)

    reason_lookup = {}
    reason_counts = Counter()
    token_failures = breakdown.get("token_limit", set())
    resource_failures = breakdown.get("resource_exhausted", set())
    for key in pending_status:
        if key in token_failures:
            reason = "token_limit_estimate"
        elif key in resource_failures:
            reason = "resource_exhausted"
        else:
            reason = "not_attempted"
        reason_lookup[key] = reason
        reason_counts[reason] += 1

    return {
        "pending_runs": pending_status,
        "pending_reason_counts": reason_counts,
        "reason_lookup": reason_lookup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Experiment 1 run status.")
    parser.add_argument(
        "--questions",
        default="data/questions/exp1_questions.json",
        help="Path to exp1_questions.json",
    )
    parser.add_argument(
        "--results",
        default="results/raw/exp1_results.jsonl",
        help="Path to exp1 results JSONL",
    )
    parser.add_argument(
        "--status",
        default="results/raw/exp1_status.json",
        help="Path to exp1 status JSON",
    )
    parser.add_argument(
        "--log",
        default="experiment1.log",
        help="Path to experiment1.log",
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw",
        help="Directory for generated audit artifacts",
    )
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    expected = build_expected_run_keys(questions)
    completed_status = load_status_completed(Path(args.status))
    completed_results = load_results_completed(Path(args.results))
    breakdown = parse_failure_breakdown(Path(args.log))

    summary = summarize(expected, completed_status, completed_results, breakdown)

    output_dir = Path(args.output_dir)
    failure_out = output_dir / "exp1_failure_breakdown.json"
    write_json(
        failure_out,
        {
            "token_limit": sorted(breakdown.get("token_limit", [])),
            "resource_exhausted": sorted(breakdown.get("resource_exhausted", [])),
            "other": sorted(breakdown.get("other", [])),
        },
    )

    pending_out = output_dir / "exp1_pending_runs.json"
    write_json(
        pending_out,
        {
            "pending_runs": summary["pending_runs"],
            "reasons": summary["reason_lookup"],
        },
    )

    print("=== Experiment 1 Audit Summary ===")
    print(f"Expected run keys   : {len(expected)}")
    print(f"Status completions  : {len(completed_status)}")
    print(f"Results completions : {len(completed_results)} (duplicates allowed)")
    print(f"Pending (status)    : {len(summary['pending_runs'])}")
    print(f"Reason counts       : {dict(summary['pending_reason_counts'])}")
    print(f"Artfacts written    : {failure_out}, {pending_out}")


if __name__ == "__main__":
    main()
