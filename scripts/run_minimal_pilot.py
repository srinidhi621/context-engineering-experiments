#!/usr/bin/env python3
"""
Minimal pilot: Run 1 question with 2 strategies at 3 fill levels.
Total: 1 × 2 × 3 × 3 reps = 18 API calls.
This script is idempotent and can be re-run to fill in missing results.
"""

import json
import time
import argparse
from pathlib import Path
from src.models.gemini_client import GeminiClient
from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.rag import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_minimal_pilot(dry_run: bool = False):
    # --- 1. SETUP ---
    output_path = Path("results/pilot_minimal_results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing results to allow for resumption
    completed_session_ids = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    existing_result = json.loads(line)
                    # Use a unique key for each run
                    session_key = (
                        existing_result['strategy'], 
                        existing_result['fill_pct'], 
                        existing_result['repetition']
                    )
                    completed_session_ids.add(session_key)
                except (json.JSONDecodeError, KeyError):
                    continue # Ignore malformed lines or lines without necessary keys
    logger.info(f"Found {len(completed_session_ids)} existing results. Will skip them.")

    # Load corpus
    corpus_path = "data/raw/pilot/hf_model_cards.json"
    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    # Load question
    question_path = "data/questions/pilot_question_01.json"
    logger.info(f"Loading question from {question_path}...")
    with open(question_path) as f:
        question = json.load(f)[0]
    
    # Initialize client and assemblers
    client = GeminiClient()
    logger.info("Initializing context assemblers...")
    naive = NaiveContextAssembler()
    rag = RAGPipeline()
    
    logger.info("Indexing corpus for RAG...")
    documents = [doc['content'] for doc in corpus]
    rag.chunk_documents(documents, chunk_size=512, overlap=50)
    rag.index_chunks()
    
    # --- 2. EXPERIMENT LOOP ---
    strategies = ["naive", "rag"]
    fill_levels = [0.3, 0.5, 0.7]
    repetitions = 3
    max_tokens = 1_000_000
    
    total_api_calls = len(strategies) * len(fill_levels) * repetitions
    api_call_count = 0
    new_results_count = 0

    logger.info(f"Starting pilot experiment runs... (Total runs: {total_api_calls})")
    for strategy in strategies:
        for fill_pct in fill_levels:
            for rep in range(repetitions):
                api_call_count += 1
                session_key = (strategy, fill_pct, rep)

                if session_key in completed_session_ids:
                    logger.info(f"Run {api_call_count}/{total_api_calls}: Skipping already completed run {session_key}")
                    continue

                logger.info(f"Run {api_call_count}/{total_api_calls}: Strategy={strategy}, Fill={fill_pct*100}%, Rep={rep}")
                
                # Assemble context
                if strategy == "naive":
                    context = naive.assemble(corpus, int(max_tokens * fill_pct))
                else:  # RAG
                    retrieved = rag.retrieve(question['question'], top_k=5)
                    context = rag.assemble_context_with_padding(retrieved, fill_pct, max_tokens)
                
                prompt = f"Answer the following question based on the provided documentation.\n\nQuestion: {question['question']}\n\nDocumentation:\n{context}\n\nAnswer:"
                
                if dry_run:
                    logger.info("[DRY RUN] Skipping API call.")
                    continue

                # Make API call
                try:
                    start_time = time.time()
                    response = client.generate_content(prompt, temperature=0.0, experiment_id="pilot", session_id=f"{strategy}_fill{int(fill_pct*100)}_rep{rep}")
                    latency = time.time() - start_time
                    
                    result = {
                        "question_id": question['question_id'],
                        "strategy": strategy, "fill_pct": fill_pct, "repetition": rep,
                        "response": response['text'], "tokens_input": response['tokens_input'],
                        "tokens_output": response['tokens_output'], "latency": latency,
                        "cost": response['cost'], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Append result immediately
                    with open(output_path, 'a') as f:
                        f.write(json.dumps(result) + '\n')
                    new_results_count += 1
                    
                    logger.info(f"Success: {response['tokens_input']:,} input tokens, {response['tokens_output']:,} output tokens, {latency:.2f}s latency, Cost: ${response['cost']:.6f}")
                except Exception as e:
                    logger.error(f"API call failed for run {api_call_count}: {e}")
                
                time.sleep(2)

    logger.info(f"Pilot run phase complete! Added {new_results_count} new results to {output_path}")
    
    # --- 3. FINAL SUMMARY ---
    if not dry_run:
        final_results = []
        with open(output_path, 'r') as f:
            final_results = [json.loads(line) for line in f]
        
        total_cost = sum(r['cost'] for r in final_results)
        total_tokens = sum(r['tokens_input'] for r in final_results)
        print("\n" + "="*60)
        print("PILOT SUMMARY")
        print("="*60)
        print(f"Total API calls recorded: {len(final_results)}")
        print(f"Total input tokens: {total_tokens:,}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Results saved to: {output_path}")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the minimal pilot experiment. Can be re-run to fill in missing results.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the run without making actual API calls.")
    args = parser.parse_args()
    
    run_minimal_pilot(dry_run=args.dry_run)
