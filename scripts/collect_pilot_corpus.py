#!/usr/bin/env python3
"""Collect minimal corpus for pilot testing"""

import json
from pathlib import Path
from src.corpus.loaders import load_hf_curated_models

# Target: 10k tokens from recent model cards
corpus = load_hf_curated_models(
    after_date="2024-08-01",
    max_tokens=10000
)

# Define the output path
output_path = Path("data/raw/pilot/hf_model_cards.json")

# Ensure the directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save to the file
with open(output_path, 'w') as f:
    json.dump(corpus, f, indent=2)

print(f"Collected {len(corpus)} model cards, {sum(d['tokens'] for d in corpus)} tokens, and saved to {output_path}")
