#!/usr/bin/env python3
"""
Run a comprehensive synthetic test of the entire experiment pipeline.

1. Generates a DUMMY results file (simulating API outputs).
2. Runs the analysis script on it.
3. Verifies the output files (CSV, Markdown).

This proves the "Analysis" phase works before we spend money on the "Run" phase.
"""

import json
import random
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.analyze_results import analyze_results

def run_synthetic_test():
    print("\n🤖 STARTING SYNTHETIC PIPELINE TEST...\n")
    
    # 1. Load Questions
    questions_path = Path("data/questions/exp1_questions.json")
    if not questions_path.exists():
        print("❌ Questions file missing!")
        return
        
    with open(questions_path, 'r') as f:
        questions = json.load(f)
        
    print(f"1. Loaded {len(questions)} questions.")
    
    # 2. Generate Dummy Results
    print("2. Generating dummy results (simulating API)...")
    dummy_file = Path("results/test_synthetic/raw_results.jsonl")
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    
    strategies = ["naive", "structured", "rag"]
    fill_pcts = [0.1, 0.5, 0.9]
    
    record_count = 0
    with open(dummy_file, 'w') as f:
        for q in questions:
            for strat in strategies:
                for fill in fill_pcts:
                    # Randomly decide if "correct" to test metrics
                    is_correct = random.choice([True, False])
                    
                    if is_correct:
                        response = q['ground_truth'] + " This is correct extra context."
                    else:
                        response = "I do not know the answer to this question."
                        
                    record = {
                        "question_id": q['question_id'],
                        "strategy": strat,
                        "fill_pct": fill,
                        "repetition": 1,
                        "response": response,
                        "tokens_input": 1000,
                        "tokens_output": 50,
                        "latency": 0.5,
                        "cost": 0.001,
                        "timestamp": "2025-01-01T12:00:00"
                    }
                    f.write(json.dumps(record) + "\n")
                    record_count += 1
                    
    print(f"   Generated {record_count} dummy records.")
    
    # 3. Run Analysis
    print("3. Running Analysis Module...")
    output_dir = Path("results/test_synthetic/analysis")
    
    try:
        analyze_results(
            input_file=str(dummy_file),
            questions_file=str(questions_path),
            output_dir=str(output_dir),
            use_llm_judge=True,
            mock_judge=True  # Vital: use mock judge to verify logic without API calls
        )
        print("   ✅ Analysis script finished without error.")
    except Exception as e:
        print(f"   ❌ Analysis script FAILED: {e}")
        return

    # 4. Verify Outputs
    print("4. Verifying outputs...")
    
    expected_files = ["scored_results.csv", "summary_metrics.csv", "analysis_report.md"]
    all_passed = True
    
    for fname in expected_files:
        fpath = output_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"   ✅ Found {fname} ({fpath.stat().st_size} bytes)")
        else:
            print(f"   ❌ Missing or empty: {fname}")
            all_passed = False
            
    # check content of summary
    if all_passed:
        df = pd.read_csv(output_dir / "summary_metrics.csv")
        print("\n   --- Summary Metrics Head ---")
        print(df.head())
        
        if 'f1' in df.columns and 'judge_score' in df.columns:
            print("\n   ✅ Metrics columns present (F1, Judge Score)")
        else:
            print("\n   ❌ Metrics columns missing!")
            all_passed = False

    if all_passed:
        print("\n✅ SYNTHETIC TEST PASSED! The analysis pipeline is ready.")
    else:
        print("\n❌ SYNTHETIC TEST FAILED.")

    # Cleanup
    # import shutil
    # shutil.rmtree("results/test_synthetic")

if __name__ == "__main__":
    run_synthetic_test()
