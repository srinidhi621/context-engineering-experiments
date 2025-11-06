#!/usr/bin/env python3
"""Run the minimal pilot experiment with pre-built contexts.

This runner estimates prompt token usage and applies a local throttle to respect
Gemini's per-minute token quotas. If a single request would exceed the configured
per-minute limit it is skipped with a diagnostic message instead of triggering
a 429. Set ``PER_MINUTE_TOKEN_LIMIT`` in the environment (or use the CLI flag)
to match your project's quota. The default fallback is 240k tokens/minute,
which aligns with the historical free-tier ceiling.
"""

from __future__ import annotations

import argparse
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple

from collections import deque

from src.models.gemini_client import GeminiClient
from src.utils.tokenizer import count_tokens


PROMPT_TEMPLATE = """You are validating a context-engineering pipeline. Use only the supplied context to answer the question.

### Context
{context}

### Question
{question}

Provide a concise factual answer. Respond in plain text with no additional commentary."""


def parse_args() -> argparse.Namespace:
    env_limit = int(os.getenv("PER_MINUTE_TOKEN_LIMIT", "240000"))
    parser = argparse.ArgumentParser(
        description="Execute the pilot run using pre-assembled contexts."
    )
    parser.add_argument(
        "--contexts",
        default="data/processed/pilot/naive_contexts.json",
        help="Path to JSON containing assembled contexts (default: %(default)s).",
    )
    parser.add_argument(
        "--question",
        default="data/questions/pilot_question_01.json",
        help="Path to the pilot question JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="results/pilot_minimal_results.jsonl",
        help="Destination for JSONL results (default: %(default)s).",
    )
    parser.add_argument(
        "--fill-pcts",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Fill percentages to execute (default: %(default)s).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per fill percentage (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Maximum output tokens for each request (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended requests without calling the API.",
    )
    parser.add_argument(
        "--per-minute-token-limit",
        type=int,
        default=env_limit,
        help=(
            "Maximum input tokens allowed per rolling minute. "
            "Set slightly below the official quota to stay safe "
            "(default: %(default)s, override with PER_MINUTE_TOKEN_LIMIT)."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_contexts(contexts: List[Dict], fill_pcts: Iterable[float]) -> List[Dict]:
    targets = {round(p, 6) for p in fill_pcts}
    matched: List[Dict] = []
    for ctx in contexts:
        pct = round(float(ctx.get("fill_pct", 0.0)), 6)
        if pct in targets:
            matched.append(ctx)
    if not matched:
        raise RuntimeError("No contexts matched the requested fill percentages.")
    return matched


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def run_pilot(args: argparse.Namespace) -> None:
    contexts_data = load_json(Path(args.contexts))
    question_data = load_json(Path(args.question))
    contexts = filter_contexts(contexts_data, args.fill_pcts)

    question_text = question_data["question"]
    question_id = question_data["question_id"]

    if args.dry_run:
        print("Dry run mode; no API calls will be made.\n")

    client = None if args.dry_run else GeminiClient()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pilot question: {question_id} — {question_text}")
    print(f"Running fills: {', '.join(f'{p:.0%}' for p in args.fill_pcts)}")
    print(f"Repetitions: {args.repetitions}\n")
    print(
        f"Local throttle: ≤{args.per_minute_token_limit:,} input tokens per rolling minute.\n"
        "Calls exceeding this limit are skipped to avoid API 429 errors."
    )

    token_window: Deque[Tuple[float, int]] = deque()

    def wait_for_token_budget(estimated_tokens: int) -> float:
        if estimated_tokens > args.per_minute_token_limit:
            raise RuntimeError(
                f"Estimated prompt ({estimated_tokens:,} tokens) exceeds the configured "
                f"per-minute limit of {args.per_minute_token_limit:,} tokens. "
                "Reduce the fill percentage or increase the limit if your quota allows."
            )

        waited = 0.0
        now = time.monotonic()
        while token_window and now - token_window[0][0] >= 60:
            token_window.popleft()

        used = sum(tokens for (_, tokens) in token_window)
        while used + estimated_tokens > args.per_minute_token_limit:
            oldest_time, oldest_tokens = token_window[0]
            sleep_time = max(0.0, 60 - (now - oldest_time))
            print(
                f"Throttling for {sleep_time:.1f}s to stay under "
                f"{args.per_minute_token_limit:,} tokens/minute."
            )
            time.sleep(sleep_time)
            waited += sleep_time
            now = time.monotonic()
            while token_window and now - token_window[0][0] >= 60:
                token_window.popleft()
            used = sum(tokens for (_, tokens) in token_window)

        return waited

    with output_path.open("w", encoding="utf-8") as f_out:
        for ctx in contexts:
            fill_pct = float(ctx["fill_pct"])
            context_text = ctx["context"]
            prompt = build_prompt(context_text, question_text)
            prompt_tokens = count_tokens(prompt)

            for rep in range(1, args.repetitions + 1):
                metadata = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "experiment_id": "pilot",
                    "strategy": ctx.get("strategy", "naive"),
                    "fill_pct": fill_pct,
                    "repetition": rep,
                    "question_id": question_id,
                    "prompt_tokens_est": prompt_tokens,
                }

                if args.dry_run:
                    print(
                        f"[DRY RUN] Would call Gemini with fill {fill_pct:.0%}, "
                        f"rep {rep}, prompt tokens ≈ {prompt_tokens:,}"
                    )
                    record = {**metadata, "dry_run": True}
                    f_out.write(json.dumps(record) + "\n")
                    continue

                print(
                    f"Calling Gemini — fill {fill_pct:.0%}, rep {rep}, "
                    f"prompt tokens ≈ {prompt_tokens:,}"
                )

                try:
                    wait_seconds = wait_for_token_budget(prompt_tokens)
                except RuntimeError as exc:
                    print(f"Skipping call: {exc}")
                    record = {**metadata, "error": str(exc)}
                    f_out.write(json.dumps(record) + "\n")
                    continue

                response = client.generate_content(
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    experiment_id="pilot",
                    session_id=f"pilot_naive_fill{int(fill_pct*100)}_rep{rep}",
                )

                actual_input_tokens = response.get("tokens_input") or prompt_tokens
                now_monotonic = time.monotonic()
                while token_window and now_monotonic - token_window[0][0] >= 60:
                    token_window.popleft()
                token_window.append((now_monotonic, actual_input_tokens))

                record = {
                    **metadata,
                    "model": response.get("model"),
                    "tokens_input": actual_input_tokens,
                    "tokens_output": response.get("tokens_output"),
                    "latency": response.get("latency"),
                    "cost": response.get("cost"),
                    "answer": response.get("text"),
                    "throttle_wait_seconds": wait_seconds,
                }
                f_out.write(json.dumps(record) + "\n")

    print(f"\nResults written to {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
