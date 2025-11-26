"""
Human evaluation utilities.
"""
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

def generate_human_eval_sheet(
    input_file: str,
    output_file: str,
    sample_size: int = 50,
    seed: int = 42
):
    """
    Generate a CSV sheet for human evaluation by sampling from results.
    """
    df = pd.read_json(input_file, lines=True)
    
    # Sample equally from strategies if possible
    sampled = df.groupby('strategy', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 4), random_state=seed)
    )
    
    # Add evaluation columns
    sampled['human_score'] = ''
    sampled['human_reasoning'] = ''
    
    # Reorder columns for readability
    cols = ['question_id', 'strategy', 'fill_pct', 'response', 'human_score', 'human_reasoning']
    sampled = sampled[cols]
    
    sampled.to_csv(output_file, index=False)
    return len(sampled)