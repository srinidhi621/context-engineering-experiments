#!/usr/bin/env python3
"""
Main runner for Experiment 1: Needle in Multiple Haystacks.

This script iterates through all questions, strategies, fill levels, and
repetitions, executing the experiment and recording the results.

It is idempotent and can be re-run to fill in missing results.
"""

import json
import time
import argparse
from pathlib import Path
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gemini_client import GeminiClient
from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.structured import StructuredContextAssembler
from src.context_engineering.rag import RAGPipeline
from src.context_engineering.advanced_rag import AdvancedRAGPipeline
from src.corpus.padding import PaddingGenerator
from src.utils.logging import get_logger
from src.utils.throttle import PerMinuteTokenThrottle, TokenLimitExceeded

logger = get_logger(__name__)

def run_experiment_1(dry_run: bool = False, per_minute_limit: int = 240000, limit: int = None):
    try:
        # --- 1. SETUP ---
        output_path = Path("results/raw/exp1_results.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_dir = Path("results/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Read existing results to allow for resumption
        completed_runs = set()
        if not dry_run and output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        run_key = (res['question_id'], res['strategy'], res['fill_pct'], res['repetition'])
                        completed_runs.add(run_key)
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info(f"Found {len(completed_runs)} existing results to skip.")

        # Load questions and corpora
        logger.info("Loading questions and corpora...")
        with open("data/questions/exp1_questions.json") as f:
            questions = json.load(f)
        with open("data/raw/exp1/hf_model_cards.json") as f:
            main_corpus = json.load(f)
        
        # Initialize client and throttle
        client = GeminiClient()
        throttle = PerMinuteTokenThrottle(per_minute_limit)
        
        # Initialize helpers
        logger.info("Initializing helpers...")
        # PaddingGenerator will reuse its internal cache if available
        padding_generator = PaddingGenerator()
        
        # --- RAG CACHING LOGIC ---
        rag_cache_prefix = str(cache_dir / "exp1_rag")
        rag_pipeline = RAGPipeline(padding_generator=padding_generator)
        
        if rag_pipeline.load_state(rag_cache_prefix):
            logger.info("Loaded RAG index from cache.")
        else:
            logger.info("Building RAG index...")
            main_corpus_docs = [doc['content'] for doc in main_corpus]
            rag_pipeline.chunk_documents(main_corpus_docs)
            rag_pipeline.index_chunks()
            rag_pipeline.save_state(rag_cache_prefix)
        
        adv_rag_cache_prefix = str(cache_dir / "exp1_adv_rag")
        advanced_rag_pipeline = AdvancedRAGPipeline(padding_generator=padding_generator)
        
        if advanced_rag_pipeline.load_state(adv_rag_cache_prefix):
             logger.info("Loaded Advanced RAG index from cache.")
        else:
            logger.info("Building Advanced RAG index...")
            main_corpus_docs = [doc['content'] for doc in main_corpus]
            advanced_rag_pipeline.chunk_documents(main_corpus_docs)
            advanced_rag_pipeline.index_chunks()
            advanced_rag_pipeline.save_state(adv_rag_cache_prefix)
        
        assemblers = {
            "naive": NaiveContextAssembler(),
            "structured": StructuredContextAssembler(),
            "rag": rag_pipeline,
            "advanced_rag": advanced_rag_pipeline
        }

        # --- 2. EXPERIMENT LOOP ---
        strategies = ["naive", "structured", "rag", "advanced_rag"]
        fill_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        repetitions = 3
        max_tokens = 1_000_000
        
        total_runs = len(questions) * len(strategies) * len(fill_levels) * repetitions
        run_counter = 0
        new_results_count = 0
        limit = args.limit

        logger.info(f"Starting Experiment 1... (Total planned runs: {total_runs})")
        
        for question in questions:
            if limit and new_results_count >= limit:
                logger.info(f"Hit run limit of {limit}. Stopping.")
                break
                
            for strategy in strategies:
                if limit and new_results_count >= limit:
                    break
                    
                assembler = assemblers[strategy]
                for fill_pct in fill_levels:
                    if limit and new_results_count >= limit:
                        break
                        
                    for rep in range(repetitions):
                        if limit and new_results_count >= limit:
                            break
                            
                        run_counter += 1
                        run_key = (question['question_id'], strategy, fill_pct, rep)

                        if run_key in completed_runs:
                            # logger.debug(f"Skipping completed run {run_key}")
                            continue
                        
                        logger.info(f"Run {run_counter}/{total_runs}: Q={question['question_id']}, Strat={strategy}, Fill={fill_pct*100}%, Rep={rep}")

                        # Assemble context based on strategy
                        try:
                            if strategy in ["rag", "advanced_rag"]:
                                retrieved = assembler.retrieve(question['question'], top_k=10)
                                context = assembler.assemble_context_with_padding(retrieved, fill_pct, max_tokens)
                            else: # naive, structured
                                target_tokens = int(max_tokens * fill_pct)
                                context = assembler.assemble(main_corpus, target_tokens)
                        except Exception as e:
                            logger.error(f"Context assembly failed: {e}")
                            continue
                        
                        prompt = f"Answer the following question based on the provided documentation.\n\nQuestion: {question['question']}\n\nDocumentation:\n{context}\n\nAnswer:"
                        
                        if dry_run:
                            logger.info(f"[DRY RUN] Would execute run for {run_key}")
                            continue

                        # Make API call with throttling
                        try:
                            # Estimate tokens (very rough: chars / 4 + safety margin)
                            est_tokens = int(len(prompt) / 3.5)
                            
                            # Wait for budget
                            throttle.wait_for_budget(est_tokens)
                            
                            start_time = time.time()
                            session_id = f"exp1_{question['question_id']}_{strategy}_fill{int(fill_pct*100)}_rep{rep}"
                            
                            response = client.generate_content(
                                prompt, 
                                temperature=0.0, 
                                experiment_id="exp1_needle", 
                                session_id=session_id
                            )
                            
                            latency = time.time() - start_time
                            
                            # Update throttle with actual usage
                            throttle.record(response['tokens_input'])
                            
                            result = {
                                "question_id": question['question_id'], "strategy": strategy, 
                                "fill_pct": fill_pct, "repetition": rep,
                                "response": response['text'], "tokens_input": response['tokens_input'],
                                "tokens_output": response['tokens_output'], "latency": latency,
                                "cost": response['cost'], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            with open(output_path, 'a') as f:
                                f.write(json.dumps(result) + '\n')
                            new_results_count += 1
                            
                            logger.info(f"Success: {response['tokens_input']:,} in, {response['tokens_output']} out, {latency:.2f}s")
                        
                        except TokenLimitExceeded as e:
                            logger.error(f"Token limit exceeded: {e}")
                        except Exception as e:
                            logger.error(f"API call failed for run {run_key}: {e}")
                            time.sleep(5) # Backoff on error

        logger.info(f"Experiment 1 run phase complete! Added {new_results_count} new results.")

    except Exception as e:
        logger.critical(f"A critical error occurred during the experiment run: {e}", exc_info=True)
        # Re-raise the exception after logging to ensure the script exits with an error code
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: Needle in Multiple Haystacks.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the run without making actual API calls.")
    parser.add_argument("--per-minute-token-limit", type=int, default=240000, help="Token limit per minute.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of successful runs (for testing).")
    args = parser.parse_args()
    run_experiment_1(dry_run=args.dry_run, per_minute_limit=args.per_minute_token_limit, limit=args.limit)
