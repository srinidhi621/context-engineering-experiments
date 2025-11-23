#!/usr/bin/env python3
"""Experiment 1 runner with context strategy loops and resumable logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.structured import StructuredContextAssembler
from src.corpus import padding as padding_module
from src.models.gemini_client import GeminiClient
from src.utils.logging import get_logger
from src.utils.throttle import PerMinuteTokenThrottle, TokenLimitExceeded
from src.utils.tokenizer import count_tokens, truncate_to_tokens

logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are validating a context-engineering pipeline. Use only the supplied context to answer the question.

### Context
{context}

### Question
{question}

Provide a concise factual answer. Respond in plain text with no additional commentary."""


@dataclass
class StrategyContext:
    """Container for assembled context metadata."""

    text: str
    tokens: int
    relevant_tokens: int
    padding_tokens: int
    doc_ids: Sequence[str]
    fill_pct: float


class BaseAssemblerStrategy:
    """Shared logic for Naive/Structured assemblers with optional caching."""

    def __init__(
        self,
        name: str,
        assembler,
        documents: Sequence[Mapping[str, str]],
        doc_index: Mapping[str, Mapping[str, str]],
        padding_generator: Optional[padding_module.PaddingGenerator],
        *,
        selection_mode: str = "all",
    ) -> None:
        self.name = name
        self.assembler = assembler
        self.documents = list(documents)
        self.doc_index = dict(doc_index)
        self.padding = padding_generator
        self.selection_mode = selection_mode
        self._shared_contexts: Dict[Tuple[float, int], StrategyContext] = {}
        self._all_doc_ids = self._collect_doc_ids(self.documents)

    def build_context(
        self,
        question: Mapping[str, object],
        *,
        fill_pct: float,
        max_context_tokens: int,
    ) -> StrategyContext:
        cache_key: Optional[Tuple[float, int]] = None

        if self.selection_mode == "all":
            cache_key = (round(fill_pct, 6), max_context_tokens)
            if cache_key in self._shared_contexts:
                return self._shared_contexts[cache_key]
            docs = self.documents
            doc_ids = self._all_doc_ids
        else:
            docs, doc_ids = self._select_documents(question)
            if not docs:
                logger.warning(
                    "No documents matched question %s; falling back to entire corpus.",
                    question.get("question_id"),
                )
                docs = self.documents
                doc_ids = self._all_doc_ids

        context = self._assemble(docs, doc_ids, fill_pct, max_context_tokens)
        if cache_key:
            self._shared_contexts[cache_key] = context
        return context

    def _assemble(
        self,
        docs: Sequence[Mapping[str, str]],
        doc_ids: Sequence[str],
        fill_pct: float,
        max_context_tokens: int,
    ) -> StrategyContext:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive.")
        if not docs:
            raise RuntimeError("No documents available for context assembly.")

        base_context = self.assembler.assemble(docs, max_context_tokens)
        if not base_context.strip():
            raise RuntimeError(f"{self.name} assembler produced empty context.")

        relevant_tokens = count_tokens(base_context)
        if self.padding:
            context_text = self.padding.pad_to_fill_percentage(
                base_context,
                fill_pct=fill_pct,
                max_context_tokens=max_context_tokens,
            )
        else:
            target_tokens = max(1, int(max_context_tokens * fill_pct))
            context_text = truncate_to_tokens(base_context, target_tokens)
        total_tokens = count_tokens(context_text)
        padding_tokens = max(total_tokens - min(total_tokens, relevant_tokens), 0)
        return StrategyContext(
            text=context_text,
            tokens=total_tokens,
            relevant_tokens=relevant_tokens,
            padding_tokens=padding_tokens,
            doc_ids=list(doc_ids),
            fill_pct=fill_pct,
        )

    def _select_documents(
        self, question: Mapping[str, object]
    ) -> Tuple[List[Mapping[str, str]], List[str]]:
        required = question.get("required_docs") or []
        selected: List[Mapping[str, str]] = []
        doc_ids: List[str] = []

        for doc_id in required:
            lookup_key = str(doc_id)
            doc = self.doc_index.get(lookup_key)
            if doc:
                selected.append(doc)
                doc_ids.append(lookup_key)

        return selected, doc_ids

    @staticmethod
    def _collect_doc_ids(docs: Sequence[Mapping[str, str]]) -> List[str]:
        collected: List[str] = []
        for doc in docs:
            for key in ("model_id", "dataset_id", "title", "url"):
                value = doc.get(key)
                if value:
                    collected.append(str(value))
                    break
        return collected


class NaiveStrategyHandler(BaseAssemblerStrategy):
    def __init__(
        self,
        documents: Sequence[Mapping[str, str]],
        doc_index: Mapping[str, Mapping[str, str]],
        padding_generator: Optional[padding_module.PaddingGenerator],
    ) -> None:
        super().__init__(
            name="naive",
            assembler=NaiveContextAssembler(),
            documents=documents,
            doc_index=doc_index,
            padding_generator=padding_generator,
            selection_mode="all",
        )


class StructuredStrategyHandler(BaseAssemblerStrategy):
    def __init__(
        self,
        documents: Sequence[Mapping[str, str]],
        doc_index: Mapping[str, Mapping[str, str]],
        padding_generator: Optional[padding_module.PaddingGenerator],
    ) -> None:
        super().__init__(
            name="structured",
            assembler=StructuredContextAssembler(),
            documents=documents,
            doc_index=doc_index,
            padding_generator=padding_generator,
            selection_mode="all",
        )


def parse_args() -> argparse.Namespace:
    env_limit = int(os.getenv("PER_MINUTE_TOKEN_LIMIT", "240000"))
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 (Needle in Multiple Haystacks).",
    )
    parser.add_argument(
        "--corpus",
        default="data/raw/exp1/hf_model_cards.json",
        help="Path to Experiment 1 corpus JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--questions",
        default="data/questions/exp1_questions.json",
        help="Path to Experiment 1 question set (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="results/exp1/experiment_1_results.jsonl",
        help="Destination JSONL file for run logs (default: %(default)s).",
    )
    parser.add_argument(
        "--padding-corpus",
        default="data/raw/padding/gutenberg_corpus.json",
        help="Optional local padding corpus to avoid Gutenberg downloads.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["naive", "structured"],
        help="Strategies to execute (default: %(default)s).",
    )
    parser.add_argument(
        "--fill-pcts",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Fill percentages to target (0-1 floats, default: %(default)s).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per ques/strategy/fill (default: %(default)s).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=1_000_000,
        help="Maximum context window per strategy (default: %(default)s).",
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
        default=512,
        help="Max completion tokens per call (default: %(default)s).",
    )
    parser.add_argument(
        "--per-minute-token-limit",
        type=int,
        default=env_limit,
        help=(
            "Rolling minute token cap enforced locally "
            "(default: %(default)s or PER_MINUTE_TOKEN_LIMIT env)."
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions for shakedown runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output and skip completed combinations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the run and write metadata without calling Gemini.",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus(path: Path) -> List[Mapping[str, str]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Corpus at {path} must be a list.")
    return data


def load_questions(path: Path, limit: Optional[int]) -> List[Mapping[str, object]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Questions at {path} must be a list.")
    if limit:
        return data[:limit]
    return data


def build_doc_index(documents: Sequence[Mapping[str, str]]) -> Dict[str, Mapping[str, str]]:
    index: Dict[str, Mapping[str, str]] = {}
    for doc in documents:
        for key in ("model_id", "dataset_id", "title", "url"):
            value = doc.get(key)
            if value:
                index[str(value)] = doc
    return index


def build_padding_generator(path: Path) -> Optional[padding_module.PaddingGenerator]:
    if not path:
        return None
    try:
        if path.exists():
            books = load_json(path)

            def _load_from_file(_: List[int]) -> List[Mapping[str, str]]:
                return books  # type: ignore[return-value]

            original_loader = padding_module.load_gutenberg_books
            padding_module.load_gutenberg_books = _load_from_file  # type: ignore[assignment]
            try:
                generator = padding_module.PaddingGenerator()
            finally:
                padding_module.load_gutenberg_books = original_loader  # type: ignore[assignment]
            return generator
        logger.warning(
            "Padding corpus %s not found; falling back to live Gutenberg fetch.",
            path,
        )
        return padding_module.PaddingGenerator()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Padding generator unavailable (%s); contexts will truncate.", exc)
        return None


def digest_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def validate_fill_percentages(fill_pcts: Iterable[float]) -> List[float]:
    validated = []
    for pct in fill_pcts:
        if not 0 < pct <= 1:
            raise ValueError(f"Fill percentage {pct} must be within (0, 1].")
        validated.append(float(pct))
    return validated


def record_key(question_id: str, strategy: str, fill_pct: float, repetition: int) -> str:
    return f"{question_id}::{strategy}::{round(fill_pct, 6)}::{repetition}"


def load_completed_records(path: Path) -> Dict[str, dict]:
    completed: Dict[str, dict] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record_key(
                record.get("question_id", "unknown"),
                record.get("strategy", "unknown"),
                float(record.get("fill_pct", 0)),
                int(record.get("repetition", 0)),
            )
            completed[key] = record
    return completed


def ensure_output_path(path: Path, resume: bool, dry_run: bool) -> None:
    if path.exists() and not (resume or dry_run):
        raise RuntimeError(
            f"{path} exists. Pass --resume to append or remove the file to restart."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def generate_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def main() -> None:
    args = parse_args()
    fill_pcts = validate_fill_percentages(args.fill_pcts)
    corpus_path = Path(args.corpus)
    question_path = Path(args.questions)
    output_path = Path(args.output)
    padding_path = Path(args.padding_corpus) if args.padding_corpus else None

    documents = load_corpus(corpus_path)
    questions = load_questions(question_path, args.max_questions)
    doc_index = build_doc_index(documents)
    padding_generator = build_padding_generator(padding_path) if padding_path else None

    strategy_handlers = {
        "naive": NaiveStrategyHandler(documents, doc_index, padding_generator),
        "structured": StructuredStrategyHandler(documents, doc_index, padding_generator),
    }

    unknown_strategies = [s for s in args.strategies if s not in strategy_handlers]
    if unknown_strategies:
        raise ValueError(f"Unsupported strategies requested: {', '.join(unknown_strategies)}")

    ensure_output_path(output_path, resume=args.resume, dry_run=args.dry_run)
    completed = load_completed_records(output_path) if args.resume else {}

    planned = len(questions) * len(args.strategies) * len(fill_pcts) * args.repetitions
    already_completed = 0
    for q in questions:
        question_id = q.get("question_id")
        if not question_id:
            continue
        for strategy in args.strategies:
            for fill_pct in fill_pcts:
                for rep in range(1, args.repetitions + 1):
                    if record_key(question_id, strategy, fill_pct, rep) in completed:
                        already_completed += 1
    remaining = planned - already_completed

    print(
        f"Experiment 1 runner\n"
        f"Questions      : {len(questions)} (limit={args.max_questions or 'all'})\n"
        f"Strategies     : {', '.join(args.strategies)}\n"
        f"Fill levels    : {', '.join(f'{p:.0%}' for p in fill_pcts)}\n"
        f"Repetitions    : {args.repetitions}\n"
        f"Planned calls  : {planned}\n"
        f"Completed skip : {already_completed}\n"
        f"Pending calls  : {remaining}\n"
        f"Dry run        : {args.dry_run}\n"
        f"Output         : {output_path.resolve()}"
    )

    throttle = PerMinuteTokenThrottle(args.per_minute_token_limit)
    client = None
    if not args.dry_run:
        try:
            client = GeminiClient()
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize GeminiClient. Set GOOGLE_API_KEY or use --dry-run."
            ) from exc

    mode = "a" if (args.resume or output_path.exists()) else "w"
    with output_path.open(mode, encoding="utf-8") as f_out:
        executed = 0
        skipped_limit = 0
        errored = 0
        for question in questions:
            question_id = question.get("question_id")
            question_text = question.get("question")
            if not question_id or not question_text:
                logger.warning("Skipping malformed question entry: %s", question)
                continue

            for strategy_name in args.strategies:
                handler = strategy_handlers[strategy_name]
                for fill_pct in fill_pcts:
                    try:
                        context = handler.build_context(
                            question,
                            fill_pct=fill_pct,
                            max_context_tokens=args.max_context_tokens,
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to build context for %s/%s %.0f%%: %s",
                            question_id,
                            strategy_name,
                            fill_pct * 100,
                            exc,
                        )
                        errored += args.repetitions
                        continue

                    prompt = generate_prompt(context.text, question_text)
                    prompt_tokens = count_tokens(prompt)
                    prompt_digest = digest_text(prompt)
                    context_digest = digest_text(context.text)
                    context_preview = context.text[:500]

                    for rep in range(1, args.repetitions + 1):
                        combo_key = record_key(question_id, strategy_name, fill_pct, rep)
                        if combo_key in completed:
                            continue

                        metadata = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "experiment_id": "exp1",
                            "question_id": question_id,
                            "question": question_text,
                            "strategy": strategy_name,
                            "fill_pct": fill_pct,
                            "repetition": rep,
                            "prompt_tokens_est": prompt_tokens,
                            "context_tokens": context.tokens,
                            "context_relevant_tokens": context.relevant_tokens,
                            "context_padding_tokens": context.padding_tokens,
                            "context_digest": context_digest,
                            "prompt_digest": prompt_digest,
                            "context_preview": context_preview,
                            "doc_ids": context.doc_ids,
                            "difficulty": question.get("difficulty"),
                            "answer_format": question.get("answer_format"),
                            "required_docs": question.get("required_docs"),
                            "source_url": question.get("source_url"),
                            "metadata": question.get("metadata"),
                        }

                        if args.dry_run:
                            print(
                                f"[DRY] {question_id} | {strategy_name} | "
                                f"{fill_pct:.0%} | rep {rep} | prompt ≈ {prompt_tokens:,} tokens"
                            )
                            record = {**metadata, "dry_run": True}
                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()
                            executed += 1
                            continue

                        session_id = (
                            f"exp1_{strategy_name}_fill{int(fill_pct*100)}_{question_id}_rep{rep}"
                        )
                        metadata["session_id"] = session_id

                        try:
                            wait_seconds = throttle.wait_for_budget(prompt_tokens)
                        except TokenLimitExceeded as exc:
                            logger.warning("Skipping %s due to limit: %s", session_id, exc)
                            record = {**metadata, "error": str(exc)}
                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()
                            skipped_limit += 1
                            continue

                        print(
                            f"Calling Gemini | {question_id} | {strategy_name} | "
                            f"{fill_pct:.0%} | rep {rep} | prompt ≈ {prompt_tokens:,} tokens"
                        )

                        try:
                            assert client is not None
                            response = client.generate_content(
                                prompt=prompt,
                                temperature=args.temperature,
                                max_output_tokens=args.max_output_tokens,
                                experiment_id="exp1",
                                session_id=session_id,
                            )
                        except Exception as exc:  # pragma: no cover - API failure
                            logger.error("Gemini call failed (%s): %s", session_id, exc)
                            record = {
                                **metadata,
                                "error": str(exc),
                                "throttle_wait_seconds": wait_seconds,
                            }
                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()
                            errored += 1
                            continue

                        actual_input_tokens = response.get("tokens_input") or prompt_tokens
                        throttle.record(actual_input_tokens)

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
                        f_out.flush()
                        executed += 1

        print(
            "\nRun complete\n"
            f"Executed records : {executed}\n"
            f"Skipped (limit)  : {skipped_limit}\n"
            f"Errors logged    : {errored}\n"
            f"Output file      : {output_path.resolve()}"
        )


if __name__ == "__main__":
    main()
