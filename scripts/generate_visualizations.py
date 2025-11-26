#!/usr/bin/env python3
"""
Generate visualizations from experiment results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def generate_visualizations(input_csv: str, output_dir: str):
    """
    Generate standard plots for experiment analysis.
    """
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # 1. Degradation Curve (F1 vs Fill %)
    if 'fill_pct' in df.columns and 'f1' in df.columns and 'strategy' in df.columns:
        print("Generating Degradation Curve...")
        plt.figure()
        sns.lineplot(data=df, x='fill_pct', y='f1', hue='strategy', marker='o')
        plt.title('Performance Degradation by Context Fill %')
        plt.xlabel('Fill Percentage')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.0)
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(out_dir / 'degradation_curve_f1.png')
        plt.close()
    
    # 2. Strategy Comparison (Bar Chart)
    if 'strategy' in df.columns and 'f1' in df.columns:
        print("Generating Strategy Comparison...")
        plt.figure()
        # Calculate mean F1 across all fill levels for overall strategy comparison
        strategy_means = df.groupby('strategy')['f1'].mean().reset_index()
        sns.barplot(data=strategy_means, x='strategy', y='f1', palette='viridis')
        plt.title('Average Performance by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Mean F1 Score')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / 'strategy_comparison_f1.png')
        plt.close()

    # 3. Latency vs Strategy (if available)
    # Note: summary_metrics.csv might not have latency if it was aggregated. 
    # If input is raw results, we use that. If summary, we check if latency is there.
    if 'latency' in df.columns:
        print("Generating Latency Comparison...")
        plt.figure()
        sns.boxplot(data=df, x='strategy', y='latency', palette='rocket')
        plt.title('Latency Distribution by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Latency (s)')
        plt.tight_layout()
        plt.savefig(out_dir / 'latency_distribution.png')
        plt.close()

    print(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from metrics CSV.")
    parser.add_argument("--input", required=True, help="Path to summary_metrics.csv or scored_results.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for output plots")
    
    args = parser.parse_args()
    
    generate_visualizations(args.input, args.output_dir)
