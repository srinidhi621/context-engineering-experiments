#!/usr/bin/env python3
"""Automatically evaluate pilot results based on predefined criteria."""

import json
import sys
import re

def main():
    """Load results, automatically score them, and save."""
    
    results_path = "results/pilot_minimal_results.jsonl"
    question_path = "data/questions/pilot_question_01.json"
    output_path = "results/pilot_minimal_results_scored.jsonl"

    try:
        with open(results_path, 'r') as f:
            results = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(question_path, 'r') as f:
            question = json.load(f)[0]
    except FileNotFoundError:
        print(f"Error: Question file not found at {question_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Question: {question['question']}")
    criteria_text = question['evaluation_criteria']
    print(f"Evaluation Criteria: {criteria_text}\n")

    # Dynamically parse keywords (like "128k" or "262,144") from the criteria string
    required_substrings = re.findall(r'(\d+k?|\d{1,3}(?:,\d{3})*)', criteria_text)
    
    if not required_substrings:
        print(f"Error: Could not parse any keywords to check from criteria: '{criteria_text}'", file=sys.stderr)
        sys.exit(1)
        
    print(f"Automatically checking for presence of any of these keywords: {required_substrings}\n")
    
    for result in results:
        response_text = result.get('response', '').lower()
        # Check if any of the required substrings are in the response
        if any(sub.lower() in response_text for sub in required_substrings):
            result['correct'] = 1
        else:
            result['correct'] = 0
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # --- Print Final Summary ---
    correct_by_strategy = {}
    for result in results:
        strategy = result['strategy']
        if strategy not in correct_by_strategy:
            correct_by_strategy[strategy] = []
        correct_by_strategy[strategy].append(result['correct'])

    print("="*60)
    print("PILOT AUTO-EVALUATION SUMMARY")
    print("="*60)
    for strategy, scores in correct_by_strategy.items():
        if scores:
            accuracy = sum(scores) / len(scores) * 100
            print(f"{strategy}: {accuracy:.1f}% correct ({sum(scores)}/{len(scores)})")
    print("="*60)
    print(f"\nScored results saved to {output_path}")

if __name__ == "__main__":
    main()