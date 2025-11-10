#!/usr/bin/env python3
"""Score pilot run outputs against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pilot responses and emit scored JSONL + summary."
    )
    parser.add_argument(
        "--results",
        default="results/pilot_minimal_results.jsonl",
        help="Path to the pilot results JSONL file.",
    )
    parser.add_argument(
        "--question",
        default="data/questions/pilot_question_01.json",
        help="Path to the pilot question JSON file.",
    )
    parser.add_argument(
        "--output",
        default="results/pilot_minimal_results_scored.jsonl",
        help="Destination for the scored JSONL output.",
    )
    parser.add_argument(
        "--summary",
        default="results/pilot_minimal_summary.md",
        help="Destination for the markdown summary report.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise RuntimeError(f"No entries found in {path}")
    return entries


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def evaluate_entry(entry: Dict, ground_truth: str) -> Tuple[str, bool]:
    if entry.get("error"):
        return ("error", False)
    answer = entry.get("answer")
    if not answer:
        return ("missing", False)
    norm_answer = normalize(answer)
    norm_truth = normalize(ground_truth)
    is_correct = norm_truth in norm_answer or norm_answer in norm_truth
    return ("answered", is_correct)


def write_scored(entries: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def write_summary(stats: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pilot Evaluation Summary", ""]
    lines.append(f"Total entries: {stats['total']}")
    lines.append(f"Answered: {stats['answered']} (correct: {stats['correct']})")
    lines.append(f"Missing: {stats['missing']}")
    lines.append(f"Errors: {stats['error']}")
    lines.append("")
    lines.append("## Accuracy by Fill Percentage")
    for fill, data in sorted(stats["per_fill"].items()):
        denom = max(1, data["answered"])
        accuracy = data["correct"] / denom
        lines.append(
            f"- Fill {fill:.0%}: {data['correct']}/{denom} answered correct "
            f"(errors {data['error']}, missing {data['missing']}) - "
            f"{accuracy:.0%}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    question_path = Path(args.question)

    entries = load_results(results_path)
    question = json.loads(question_path.read_text(encoding="utf-8"))
    ground_truth = question["ground_truth"]

    stats = {
        "total": 0,
        "answered": 0,
        "correct": 0,
        "missing": 0,
        "error": 0,
        "per_fill": {},
    }

    scored_entries: List[Dict] = []
    for entry in entries:
        fill_pct = float(entry.get("fill_pct", 0))
        fill_stats = stats["per_fill"].setdefault(
            fill_pct, {"answered": 0, "correct": 0, "missing": 0, "error": 0}
        )

        status, is_correct = evaluate_entry(entry, ground_truth)

        stats["total"] += 1
        if status == "answered":
            stats["answered"] += 1
            fill_stats["answered"] += 1
            if is_correct:
                stats["correct"] += 1
                fill_stats["correct"] += 1
        elif status == "missing":
            stats["missing"] += 1
            fill_stats["missing"] += 1
        elif status == "error":
            stats["error"] += 1
            fill_stats["error"] += 1

        entry["evaluation_status"] = status
        entry["is_correct"] = is_correct if status == "answered" else False
        scored_entries.append(entry)

    write_scored(scored_entries, Path(args.output))
    write_summary(stats, Path(args.summary))

    print(
        f"Evaluation complete: "
        f"{stats['correct']}/{max(1, stats['answered'])} answered correct. "
        f"Summary written to {args.summary}"
    )


if __name__ == "__main__":
    main()
