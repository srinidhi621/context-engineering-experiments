#!/usr/bin/env python3
"""Validate experiment question sets for schema compliance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

REQUIRED_FIELDS = {
    "experiment",
    "question_id",
    "question",
    "ground_truth",
    "difficulty",
    "required_docs",
    "evaluation_criteria",
    "source_url",
    "source_model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate JSON question files for experiments."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to JSON files (arrays of question objects).",
    )
    parser.add_argument(
        "--require-experiment",
        help="Ensure every entry matches this experiment id.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    return data


def validate_questions(
    questions: List[Dict],
    *,
    require_experiment: str | None = None,
    path: Path,
) -> List[str]:
    errors: List[str] = []
    ids: Set[str] = set()

    for idx, q in enumerate(questions, start=1):
        location = f"{path}:{idx}"
        missing = REQUIRED_FIELDS - q.keys()
        if missing:
            errors.append(f"{location} missing fields: {', '.join(sorted(missing))}")

        qid = q.get("question_id")
        if not qid:
            errors.append(f"{location} has empty question_id")
        elif qid in ids:
            errors.append(f"{location} duplicates question_id {qid}")
        else:
            ids.add(qid)

        docs = q.get("required_docs", [])
        if not isinstance(docs, list) or not docs:
            errors.append(f"{location} must include at least one required_doc")

        if require_experiment and q.get("experiment") != require_experiment:
            errors.append(
                f"{location} experiment '{q.get('experiment')}' "
                f"!= '{require_experiment}'"
            )

    return errors


def main() -> None:
    args = parse_args()
    total_errors = 0
    for file_path in args.files:
        path = Path(file_path)
        try:
            questions = load_questions(path)
            errors = validate_questions(
                questions,
                require_experiment=args.require_experiment,
                path=path,
            )
            if errors:
                total_errors += len(errors)
                print(f"\n❌ {path} has {len(errors)} issue(s):")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"✅ {path} passed validation ({len(questions)} entries)")
        except Exception as exc:
            total_errors += 1
            print(f"\n❌ {path}: {exc}")

    if total_errors:
        print(f"\nValidation failed: {total_errors} issue(s) found.")
        raise SystemExit(1)

    print("\nAll question files passed validation.")


if __name__ == "__main__":
    main()
