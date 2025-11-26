#!/usr/bin/env python3
"""
Analyze experiment results.

Reads a JSONL file of experiment results, calculates metrics (Exact Match, F1, LLM Judge),
and outputs a summary CSV and markdown report.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import exact_match_score, f1_score, contains_score, recall_score
from src.evaluation.judges import CorrectnessJudge, MockJudge
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_questions(questions_path: str) -> dict:
    """Load questions and map IDs to ground truth."""
    with open(questions_path, 'r') as f:
        data = json.load(f)
    return {q['question_id']: q for q in data}

def analyze_results(
    input_file: str, 
    questions_file: str, 
    output_dir: str, 
    use_llm_judge: bool = False,
    mock_judge: bool = False
):
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading results from {input_path}...")
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                continue
    
    logger.info(f"Loaded {len(results)} result records.")
    questions_map = load_questions(questions_file)
    
    # Initialize Judge
    if use_llm_judge:
        if mock_judge:
            judge = MockJudge()
            logger.info("Using MOCK Judge.")
        else:
            judge = CorrectnessJudge()
            logger.info("Using LLM Judge (Gemini).")
    else:
        judge = None
        logger.info("Skipping LLM Judge.")

    scored_records = []
    
    logger.info("Computing metrics...")
    for res in tqdm(results):
        q_id = res['question_id']
        if q_id not in questions_map:
            continue
            
        ground_truth = questions_map[q_id]['ground_truth']
        question_text = questions_map[q_id]['question']
        prediction = res.get('response', '')
        
        # 1. Heuristic Metrics
        em = exact_match_score(prediction, ground_truth)
        f1 = f1_score(prediction, ground_truth)
        recall = recall_score(prediction, ground_truth)
        contains = contains_score(prediction, ground_truth)
        
        record = res.copy()
        record.update({
            'exact_match': em,
            'f1': f1,
            'recall': recall,
            'contains_truth': contains
        })
        
        # 2. LLM Judge (Optional)
        if judge:
            try:
                eval_res = judge.evaluate(question_text, prediction, ground_truth)
                record['judge_score'] = eval_res.get('score', 0)
                record['judge_label'] = eval_res.get('label', 'Error')
                record['judge_reasoning'] = eval_res.get('reasoning', '')
            except Exception as e:
                logger.error(f"Judge error on {q_id}: {e}")
                record['judge_score'] = 0
        
        scored_records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(scored_records)
    
    # Save raw scored data
    raw_output = out_dir / "scored_results.csv"
    df.to_csv(raw_output, index=False)
    logger.info(f"Saved scored results to {raw_output}")
    
    # Aggregation
    metrics = ['f1', 'recall', 'exact_match', 'contains_truth']
    if use_llm_judge:
        metrics.append('judge_score')
        
    # Group by Strategy and Fill %
    summary = df.groupby(['strategy', 'fill_pct'])[metrics].mean().reset_index()
    summary_output = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_output, index=False)
    logger.info(f"Saved summary metrics to {summary_output}")
    
    # Generate Markdown Report
    report_path = out_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment Analysis Report\n\n")
        f.write(f"**Input:** {input_file}\n")
        f.write(f"**Records:** {len(df)}\n\n")
        
        f.write("## Summary by Strategy\n\n")
        f.write(summary.to_markdown(index=False))
        
        f.write("\n\n## Best Performing Configuration\n\n")
        best = summary.loc[summary['f1'].idxmax()]
        f.write(f"**Strategy:** {best['strategy']}\n")
        f.write(f"**Fill %:** {best['fill_pct']}\n")
        f.write(f"**F1 Score:** {best['f1']:.4f}\n")

    print(f"\nAnalysis Complete! Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument("--input", required=True, help="Path to raw .jsonl results")
    parser.add_argument("--questions", required=True, help="Path to questions.json")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-a-judge")
    parser.add_argument("--mock-judge", action="store_true", help="Use mock judge for testing")
    
    args = parser.parse_args()
    
    analyze_results(
        args.input, 
        args.questions, 
        args.output_dir, 
        use_llm_judge=args.judge,
        mock_judge=args.mock_judge
    )