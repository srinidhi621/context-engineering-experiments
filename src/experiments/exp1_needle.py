"""Experiment 1: Needle in Multiple Haystacks.
"""
import time
import json
from pathlib import Path
from typing import Dict, List

from src.experiments.base_experiment import BaseExperiment
from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.structured import StructuredContextAssembler
from src.context_engineering.rag import RAGPipeline
from src.context_engineering.advanced_rag import AdvancedRAGPipeline
from src.corpus.padding import PaddingGenerator
from src.utils.logging import get_logger
from src.utils.throttle import PerMinuteTokenThrottle, TokenLimitExceeded

logger = get_logger(__name__)

class NeedleExperiment(BaseExperiment):
    """
    Run Experiment 1: Test retrieval accuracy across context strategies and fill levels.
    """
    
    def __init__(self):
        super().__init__(name="exp1")
        
        # Load resources
        with open("data/questions/exp1_questions.json") as f:
            self.questions = json.load(f)
        with open("data/raw/exp1/hf_model_cards.json") as f:
            self.corpus = json.load(f)
            
        self.strategies = ["naive", "structured", "rag", "advanced_rag"]
        self.fill_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.repetitions = 3
        self.max_tokens = 900_000
        
        # Initialize Helpers
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

    def run(self, dry_run: bool = False, limit: int = None, per_minute_token_limit: int = 240000):
        throttle = PerMinuteTokenThrottle(per_minute_token_limit)
        total_runs = len(self.questions) * len(self.strategies) * len(self.fill_levels) * self.repetitions
        self.status.total_runs = total_runs
        
        logger.info(f"Starting Experiment 1: {total_runs} runs planned.")
        
        new_results = 0
        run_counter = 0
        
        for question in self.questions:
            for strategy in self.strategies:
                assembler = self.assemblers[strategy]
                for fill_pct in self.fill_levels:
                    for rep in range(self.repetitions):
                        if limit is not None and new_results >= limit:
                            logger.info(f"Hit limit of {limit} runs. Stopping.")
                            return

                        run_key = self._make_run_key(question['question_id'], strategy, fill_pct, rep)
                        run_counter += 1
                        
                        if self.is_completed(run_key):
                            continue
                            
                        logger.info(f"Run {run_counter}/{total_runs}: {run_key}")
                        
                        # 1. Assemble Context
                        try:
                            if strategy in ["rag", "advanced_rag"]:
                                retrieved = assembler.retrieve(question['question'], top_k=10)
                                context = assembler.assemble_context_with_padding(retrieved, fill_pct, self.max_tokens)
                            else:
                                target_tokens = int(self.max_tokens * fill_pct)
                                context = assembler.assemble(self.corpus, target_tokens)
                        except Exception as e:
                            logger.error(f"Context assembly failed for {run_key}: {e}")
                            self.status.failed_runs += 1
                            continue

                        prompt = f"Answer the following question based on the provided documentation.\n\nQuestion: {question['question']}\n\nDocumentation:\n{context}\n\nAnswer:"
                        
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
                            continue

                        # 2. API Call
                        try:
                            est_tokens = int(len(prompt) / 3.5)
                            waited = throttle.wait_for_budget(est_tokens)
                            if waited > 0:
                                logger.info(f"Throttled for {waited:.1f}s")
                                
                            start_time = time.time()
                            response = self.client.generate_content(
                                prompt, 
                                temperature=0.0,
                                experiment_id="exp1",
                                session_id=run_key
                            )
                            latency = time.time() - start_time
                            throttle.record(response['tokens_input'])
                            
                            # 3. Record Result
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
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            self.record_result(result, run_key)
                            new_results += 1
                            logger.info(f"Success: {response['tokens_input']} tokens")
                            
                        except TokenLimitExceeded as e:
                            logger.warning(f"Token limit exceeded: {e}")
                            time.sleep(10)
                        except Exception as e:
                            logger.error(f"API Failed for {run_key}: {e}")
                            self.status.failed_runs += 1
                            # Hard stop on rate limits
                            if "429" in str(e) or "quota" in str(e).lower():
                                logger.critical("Rate limit hit. Stopping experiment.")
                                return
                            time.sleep(5)
