"""Experiment 1: Needle in Multiple Haystacks."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.config import config
from src.context_engineering.advanced_rag import AdvancedRAGPipeline
from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.rag import RAGPipeline
from src.context_engineering.structured import StructuredContextAssembler
from src.corpus.padding import PaddingGenerator
from src.experiments.base_experiment import BaseExperiment
from src.utils.logging import get_logger
from src.utils.throttle import PerMinuteTokenThrottle, TokenLimitExceeded
from src.utils.tokenizer import count_tokens

logger = get_logger(__name__)


class NeedleExperiment(BaseExperiment):
    """
    Run Experiment 1: Test retrieval accuracy across context strategies and fill levels.
    """

    def __init__(self) -> None:
        super().__init__(name="exp1")

        with open("data/questions/exp1_questions.json") as q_handle:
            self.questions = json.load(q_handle)
        with open("data/raw/exp1/hf_model_cards.json") as c_handle:
            self.corpus = json.load(c_handle)

        self.strategies = ["naive", "structured", "rag", "advanced_rag"]
        self.fill_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.repetitions = 3
        self.max_tokens = int(config.context_limit * 0.99)
        self.prompt_token_margin = 2_000
        self.max_run_attempts = 5

        self.padding_generator = PaddingGenerator()
        self._init_assemblers()

    def _init_assemblers(self):
        """Initialize and cache RAG pipelines"""
        cache_dir = Path("results/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # RAG
        rag_path = str(cache_dir / "exp1_rag")
        self.rag = RAGPipeline(padding_generator=self.padding_generator)
        if not self.rag.load_state(rag_path):
            logger.info("Building RAG index...")
            docs = [d['content'] for d in self.corpus]
            self.rag.chunk_documents(docs)
            self.rag.index_chunks()
            self.rag.save_state(rag_path)
            
        # Advanced RAG
        adv_path = str(cache_dir / "exp1_adv_rag")
        self.adv_rag = AdvancedRAGPipeline(padding_generator=self.padding_generator)
        if not self.adv_rag.load_state(adv_path):
            logger.info("Building Advanced RAG index...")
            docs = [d['content'] for d in self.corpus]
            self.adv_rag.chunk_documents(docs)
            self.adv_rag.index_chunks()
            self.adv_rag.save_state(adv_path)
            
        self.assemblers = {
            "naive": NaiveContextAssembler(),
            "structured": StructuredContextAssembler(),
            "rag": self.rag,
            "advanced_rag": self.adv_rag
        }

    def _get_run_key_from_result(self, result: Dict) -> str:
        return f"{result['question_id']}_{result['strategy']}_{result['fill_pct']}_{result['repetition']}"

    def _make_run_key(self, q_id, strat, fill, rep) -> str:
        return f"{q_id}_{strat}_{fill}_{rep}"

    def run(
        self,
        dry_run: bool = False,
        limit: Optional[int] = None,
        per_minute_token_limit: int = 240000,
        target_run_keys: Optional[Set[str]] = None,
    ) -> None:
        throttle = PerMinuteTokenThrottle(per_minute_token_limit)
        total_runs = len(self.questions) * len(self.strategies) * len(self.fill_levels) * self.repetitions
        selected_run_keys = set(target_run_keys) if target_run_keys else None
        planned_runs = len(selected_run_keys) if selected_run_keys else total_runs
        self.status.total_runs = total_runs

        logger.info(f"Starting Experiment 1: {planned_runs} runs planned.")

        new_results = 0
        attempted_runs = 0
        done_targets = set()

        for question in self.questions:
            for strategy in self.strategies:
                assembler = self.assemblers[strategy]
                for fill_pct in self.fill_levels:
                    for rep in range(self.repetitions):
                        if limit is not None and new_results >= limit:
                            logger.info(f"Hit limit of {limit} runs. Stopping.")
                            return

                        run_key = self._make_run_key(question['question_id'], strategy, fill_pct, rep)

                        if selected_run_keys and run_key not in selected_run_keys:
                            continue

                        if self.is_completed(run_key):
                            if selected_run_keys:
                                done_targets.add(run_key)
                                if done_targets == selected_run_keys:
                                    logger.info("All requested run keys already completed. Exiting.")
                                    return
                            continue

                        attempted_runs += 1
                        logger.info(f"Run {attempted_runs}/{planned_runs}: {run_key}")

                        # 1. Assemble Context
                        try:
                            if strategy in ["rag", "advanced_rag"]:
                                retrieved = assembler.retrieve(question['question'], top_k=10)
                                context = assembler.assemble_context_with_padding(retrieved, fill_pct, self.max_tokens)
                            else:
                                target_tokens = int(self.max_tokens * fill_pct)
                                context = assembler.assemble(self.corpus, target_tokens)
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.error("Context assembly failed for %s: %s", run_key, exc)
                            self.record_failure(run_key, "context_assembly")
                            if selected_run_keys and run_key in selected_run_keys:
                                done_targets.add(run_key)
                                if done_targets == selected_run_keys:
                                    logger.info("Processed all requested run keys. Exiting.")
                                    return
                            continue

                        prompt = (
                            "Answer the following question based on the provided documentation.\n\n"
                            f"Question: {question['question']}\n\n"
                            f"Documentation:\n{context}\n\nAnswer:"
                        )

                        if dry_run:
                            logger.info(f"[DRY RUN] Would execute {run_key}")
                            # Simulate success for dry run logic
                            self.record_result({
                                "question_id": question['question_id'],
                                "strategy": strategy,
                                "fill_pct": fill_pct,
                                "repetition": rep,
                                "response": "DRY RUN RESPONSE",
                                "dry_run": True
                            }, run_key)
                            new_results += 1
                            if selected_run_keys and run_key in selected_run_keys:
                                done_targets.add(run_key)
                                if done_targets == selected_run_keys:
                                    logger.info("Processed all requested run keys. Exiting.")
                                    return
                            continue

                        run_attempt = 0
                        while True:
                            run_attempt += 1
                            try:
                                prompt_tokens = count_tokens(prompt) + self.prompt_token_margin
                                waited = throttle.wait_for_budget(prompt_tokens)
                                if waited > 0:
                                    logger.info(f"Throttled for {waited:.1f}s")

                                start_time = time.time()
                                response = self.client.generate_content(
                                    prompt,
                                    temperature=0.0,
                                    experiment_id="exp1",
                                    session_id=run_key,
                                )
                                latency = time.time() - start_time
                                throttle.record(response['tokens_input'])

                                result = {
                                    "question_id": question['question_id'],
                                    "strategy": strategy,
                                    "fill_pct": fill_pct,
                                    "repetition": rep,
                                    "response": response['text'],
                                    "tokens_input": response['tokens_input'],
                                    "tokens_output": response['tokens_output'],
                                    "latency": latency,
                                    "cost": response['cost'],
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                }
                                self.record_result(result, run_key)
                                new_results += 1
                                logger.info(f"Success: {response['tokens_input']} tokens")
                                break

                            except TokenLimitExceeded as exc:
                                logger.warning(f"Token limit exceeded ({run_key}): {exc}")
                                self.record_failure(run_key, "token_limit_estimate")
                                time.sleep(10)
                                break
                            except Exception as exc:  # pylint: disable=broad-except
                                err_msg = str(exc)
                                if any(term in err_msg for term in ("ResourceExhausted", "429", "quota")):
                                    if run_attempt < self.max_run_attempts:
                                        wait_time = min(300, 60 * run_attempt)
                                        logger.warning(
                                            "ResourceExhausted for %s (attempt %d/%d). Sleeping %.1fs before retry.",
                                            run_key,
                                            run_attempt,
                                            self.max_run_attempts,
                                            wait_time,
                                        )
                                        time.sleep(wait_time)
                                        continue
                                    logger.error("ResourceExhausted persisted for %s: %s", run_key, err_msg)
                                    self.record_failure(run_key, "resource_exhausted")
                                    break

                                logger.error("API Failed for %s: %s", run_key, err_msg)
                                self.record_failure(run_key, "api_error")
                                time.sleep(5)
                                break

                        if selected_run_keys and run_key in selected_run_keys:
                            done_targets.add(run_key)
                            if done_targets == selected_run_keys:
                                logger.info("Processed all requested run keys. Exiting.")
                                return
