"""Experiment 2: Context Pollution."""
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
from src.corpus.pollution import PollutionInjector
from src.experiments.base_experiment import BaseExperiment
from src.utils.logging import get_logger
from src.utils.throttle import PerMinuteTokenThrottle, TokenLimitExceeded
from src.utils.tokenizer import count_tokens, truncate_to_tokens

logger = get_logger(__name__)


class PollutionExperiment(BaseExperiment):
    """Run Experiment 2: Test robustness to irrelevant information (pollution)."""

    def __init__(self) -> None:
        super().__init__(name="exp2")

        with open("data/questions/exp2_questions.json") as q_handle:
            self.questions = json.load(q_handle)
        
        # Load Base Corpus
        with open("data/raw/exp2/base_corpus.json") as c_handle:
            self.base_corpus = json.load(c_handle)
            
        # Load Padding Corpus
        self.padding_generator = PaddingGenerator()
        # PaddingGenerator exposes books via padding_books
        self.padding_corpus = self._ensure_tokens(self.padding_generator.padding_books)
        
        self.pollution_injector = PollutionInjector()

        self.strategies = ["naive", "structured", "rag", "advanced_rag"]
        self.pollution_levels = [50_000, 200_000, 500_000, 700_000, 950_000]
        self.repetitions = 3
        self.max_tokens = 1_000_000
        self.prompt_token_margin = 2_000
        self.max_run_attempts = 5
        
        self.naive = NaiveContextAssembler()
        self.structured = StructuredContextAssembler()
        self.rag = RAGPipeline(padding_generator=self.padding_generator)
        self.adv_rag = AdvancedRAGPipeline(padding_generator=self.padding_generator)

    def _get_run_key(self, q_id, strat, poll_level, rep) -> str:
        return f"{q_id}_{strat}_{poll_level}_{rep}"

    def _get_run_key_from_result(self, result: Dict) -> str:
        """Reconstruct run key from a stored result."""
        try:
            return self._get_run_key(
                result["question_id"],
                result["strategy"],
                result["pollution_level"],
                result["repetition"],
            )
        except KeyError:
            return ""

    def _get_pollution_corpus(self, target_tokens: int) -> List[Dict]:
        """Get a subset of padding corpus adding up to target_tokens."""
        selected_docs = []
        current_tokens = 0
        
        for doc in self.padding_corpus:
            if current_tokens >= target_tokens:
                break
            
            remaining = target_tokens - current_tokens
            if doc['tokens'] > remaining:
                trunc_content = truncate_to_tokens(doc['content'], remaining)
                new_doc = doc.copy()
                new_doc['content'] = trunc_content
                new_doc['tokens'] = count_tokens(trunc_content)
                selected_docs.append(new_doc)
                current_tokens += new_doc['tokens']
                break
            else:
                selected_docs.append(doc)
                current_tokens += doc['tokens']
                
        return selected_docs

    def _ensure_tokens(self, docs: List[Dict]) -> List[Dict]:
        """Ensure each doc has a 'tokens' field."""
        normalized = []
        for doc in docs:
            content = doc.get('content', '')
            tokens = doc.get('tokens')
            if tokens is None:
                tokens = count_tokens(content)
            new_doc = doc.copy()
            new_doc['tokens'] = tokens
            normalized.append(new_doc)
        return normalized

    def _update_rag_indices(self, pollution_level: int, current_corpus: List[Dict]) -> None:
        """Load or build RAG indices for the specific pollution level."""
        cache_dir = Path("results/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # RAG
        rag_path = str(cache_dir / f"exp2_rag_{pollution_level}")
        self.rag = RAGPipeline(padding_generator=self.padding_generator) 
        
        if not self.rag.load_state(rag_path):
            logger.info(f"Building RAG index for level {pollution_level}...")
            docs = [d['content'] for d in current_corpus]
            self.rag.chunk_documents(docs)
            self.rag.index_chunks()
            self.rag.save_state(rag_path)
        else:
            logger.info(f"Loaded RAG index for level {pollution_level}")

        # Advanced RAG
        adv_path = str(cache_dir / f"exp2_adv_rag_{pollution_level}")
        self.adv_rag = AdvancedRAGPipeline(padding_generator=self.padding_generator)
        
        if not self.adv_rag.load_state(adv_path):
            logger.info(f"Building Advanced RAG index for level {pollution_level}...")
            docs = [d['content'] for d in current_corpus]
            self.adv_rag.chunk_documents(docs)
            self.adv_rag.index_chunks()
            self.adv_rag.save_state(adv_path)
        else:
            logger.info(f"Loaded Advanced RAG index for level {pollution_level}")

    def run(
        self,
        dry_run: bool = False,
        limit: Optional[int] = None,
        per_minute_token_limit: int = 1000000,
        target_run_keys: Optional[Set[str]] = None,
    ) -> None:
        throttle = PerMinuteTokenThrottle(per_minute_token_limit)
        total_runs = len(self.questions) * len(self.strategies) * len(self.pollution_levels) * self.repetitions
        
        self.status.total_runs = total_runs
        logger.info(f"Starting Experiment 2: {total_runs} runs planned.")

        new_results = 0
        attempted_runs = 0

        for pollution_level in self.pollution_levels:
            logger.info(f"=== Processing Pollution Level: {pollution_level:,} tokens ===")
            
            base_docs = self._ensure_tokens(self.base_corpus)
            pollution_docs = self._ensure_tokens(self._get_pollution_corpus(pollution_level))
            current_corpus = base_docs + pollution_docs
            
            if not dry_run:
                self._update_rag_indices(pollution_level, current_corpus)
            
            for question in self.questions:
                for strategy in self.strategies:
                    for rep in range(self.repetitions):
                        if limit is not None and new_results >= limit:
                            logger.info(f"Hit limit of {limit} runs. Stopping.")
                            return

                        run_key = self._get_run_key(question['question_id'], strategy, pollution_level, rep)
                        
                        if target_run_keys and run_key not in target_run_keys:
                            continue
                            
                        if self.is_completed(run_key):
                            continue

                        attempted_runs += 1
                        logger.info(f"Run {attempted_runs}/{total_runs}: {run_key}")

                        if dry_run:
                            logger.info(f"[DRY RUN] Would execute {run_key}")
                            self.record_result({
                                "question_id": question['question_id'],
                                "strategy": strategy,
                                "pollution_level": pollution_level,
                                "repetition": rep,
                                "response": "DRY RUN RESPONSE",
                                "dry_run": True
                            }, run_key)
                            new_results += 1
                            continue
                            
                        try:
                            if strategy in ["rag", "advanced_rag"]:
                                if strategy == "rag":
                                    assembler = self.rag
                                else:
                                    assembler = self.adv_rag
                                    
                                retrieved = assembler.retrieve(question['question'], top_k=10)
                                context = "\n\n".join(retrieved)
                            else:
                                if strategy == "naive":
                                    context = self.naive.assemble(current_corpus, self.max_tokens)
                                else:
                                    context = self.structured.assemble(current_corpus, self.max_tokens)
                                    
                        except Exception as exc:
                            logger.error("Context assembly failed for %s: %s", run_key, exc)
                            self.record_failure(run_key, "context_assembly")
                            continue

                        # Enforce context budget safeguard
                        max_context_tokens = self.max_tokens - self.prompt_token_margin
                        if count_tokens(context) > max_context_tokens:
                            context = truncate_to_tokens(context, max_context_tokens)

                        prompt = (
                            "Answer the following question based on the provided documentation.\n\n"
                            f"Question: {question['question']}\n\n"
                            f"Documentation:\n{context}\n\nAnswer:"
                        )
                        
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
                                    experiment_id="exp2",
                                    session_id=run_key,
                                )
                                latency = response['latency']
                                throttle.record(response['tokens_input'])

                                result = {
                                    "question_id": question['question_id'],
                                    "strategy": strategy,
                                    "pollution_level": pollution_level,
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
                            except Exception as exc:
                                err_msg = str(exc)
                                if any(term in err_msg for term in ("ResourceExhausted", "429", "quota")):
                                    if run_attempt < self.max_run_attempts:
                                        wait_time = min(300, 60 * run_attempt)
                                        logger.warning(
                                            "ResourceExhausted for %s (attempt %d/%d). Sleeping %.1fs...",
                                            run_key, run_attempt, self.max_run_attempts, wait_time
                                        )
                                        time.sleep(wait_time)
                                        continue
                                    self.record_failure(run_key, "resource_exhausted")
                                    break

                                logger.error("API Failed for %s: %s", run_key, err_msg)
                                self.record_failure(run_key, "api_error")
                                time.sleep(5)
                                break
