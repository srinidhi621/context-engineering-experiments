#!/usr/bin/env python3
"""
Generate visualizations from experiment results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

# Define consistent color palette for all charts
COLOR_PALETTE = {
    'structured': '#2E86AB',    # Dark blue (best performer)
    'rag': '#4EA8DE',           # Medium blue
    'advanced_rag': '#7FC8F8',  # Light blue
    'naive': '#E76F51'          # Orange/Red (worst performer)
}

def generate_strategy_comparison_fixed(df: pd.DataFrame, out_dir: Path):
    """
    Generate fixed strategy comparison bar chart with proper y-axis scaling,
    error bars, and visual improvements.
    
    Fixes: y-axis 0-0.35 instead of 0-1.0, adds 95% CI, sorts by F1, adds value labels.
    """
    print("Generating Fixed Strategy Comparison...")
    
    # Calculate mean F1 per strategy (averaging across fill levels)
    strategy_stats = df.groupby('strategy')['f1'].agg(['mean', 'std', 'count']).reset_index()
    strategy_stats.columns = ['strategy', 'mean_f1', 'std_f1', 'count']
    
    # Calculate 95% CI using t-distribution
    # For 5 fill levels (n=5), df=4, t-critical ≈ 2.776
    n_fill_levels = strategy_stats['count'].iloc[0]  # Should be 5 for all strategies
    t_value = 2.776 if n_fill_levels == 5 else 1.96  # Fallback to z-score
    strategy_stats['ci_95'] = t_value * (strategy_stats['std_f1'] / np.sqrt(n_fill_levels))
    
    # Sort by mean F1 descending
    strategy_stats = strategy_stats.sort_values('mean_f1', ascending=False)
    
    # Map colors
    colors = [COLOR_PALETTE.get(s, '#888888') for s in strategy_stats['strategy']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars with error bars
    bars = ax.bar(
        strategy_stats['strategy'], 
        strategy_stats['mean_f1'],
        yerr=strategy_stats['ci_95'],
        capsize=5,
        color=colors,
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels on top of bars
    for bar, val in zip(bars, strategy_stats['mean_f1']):
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_ylabel('Mean F1 Score', fontsize=12)
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_title('Average Performance by Strategy (Experiment 1)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.35)  # FIXED: was 0-1.0, now 0-0.35
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at naive's score for reference
    naive_score = strategy_stats[strategy_stats['strategy'] == 'naive']['mean_f1'].values
    if len(naive_score) > 0:
        ax.axhline(y=naive_score[0], color='gray', linestyle='--', alpha=0.5, 
                   label=f'Naive baseline ({naive_score[0]:.3f})')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_strategy_comparison_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: exp1_strategy_comparison_fixed.png")


def generate_degradation_curve_fixed(df: pd.DataFrame, out_dir: Path):
    """
    Generate fixed degradation curve with proper y-axis scaling and danger zone highlighting.
    
    Fixes: y-axis 0-0.35 instead of 0-1.0, highlights 50-70% danger zone where naive collapses.
    Key insight: Naive drops to 0.019 F1 at 50% fill - this is the critical finding.
    
    Note: If per-run data is passed (multiple rows per strategy/fill_pct), this function
    aggregates to means first to produce clean summary lines.
    """
    print("Generating Fixed Degradation Curve...")
    
    # Check if this is per-run data (has question_id) or already aggregated summary data
    # Per-run data will have many rows per strategy/fill_pct combination
    rows_per_combo = df.groupby(['strategy', 'fill_pct']).size().mean()
    is_per_run = rows_per_combo > 1.5  # More than 1 row per combo = per-run data
    
    if is_per_run:
        print("  → Detected per-run data, aggregating to means...")
        # Aggregate to mean F1 per strategy/fill_pct
        df_agg = df.groupby(['strategy', 'fill_pct'])['f1'].mean().reset_index()
    else:
        df_agg = df.copy()
    
    # Ensure data is sorted by fill_pct for proper line plotting
    df_sorted = df_agg.sort_values(['strategy', 'fill_pct'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each strategy
    for strategy in ['structured', 'rag', 'advanced_rag', 'naive']:
        subset = df_sorted[df_sorted['strategy'] == strategy]
        if len(subset) == 0:
            continue
        ax.plot(
            subset['fill_pct'], 
            subset['f1'], 
            marker='o', 
            linewidth=2,
            markersize=8,
            label=strategy.replace('_', ' ').title(),
            color=COLOR_PALETTE.get(strategy, '#888888')
        )
    
    # Highlight the danger zone (50-70% fill)
    ax.axvspan(0.45, 0.75, alpha=0.1, color='red', label='Degradation Zone')
    
    # Add annotation for naive's collapse
    naive_50 = df_sorted[(df_sorted['strategy'] == 'naive') & (df_sorted['fill_pct'] == 0.5)]
    if len(naive_50) > 0:
        naive_f1_at_50 = naive_50['f1'].values[0]
        ax.annotate(
            'Naive collapses\nat 50% fill',
            xy=(0.5, naive_f1_at_50),
            xytext=(0.55, 0.08),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            color='red',
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xlabel('Fill Percentage', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Performance Degradation by Context Fill %', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.35)  # FIXED: was 0-1.0, now 0-0.35
    ax.set_xlim(0.05, 0.95)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'])
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_degradation_curve_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: exp1_degradation_curve_fixed.png")


def generate_latency_vs_tokens_scatter(df: pd.DataFrame, out_dir: Path):
    """
    Generate scatter plot showing latency vs context size (tokens_input).
    
    Replaces misleading latency boxplot. Key insight: latency scales with token count,
    not strategy. RAG processes ~92k tokens and stays fast; naive/structured scale linearly.
    """
    print("Generating Latency vs Tokens Scatter...")
    
    # Check if we have the required columns
    if 'tokens_input' not in df.columns or 'latency' not in df.columns:
        print("  ⚠ Skipping: tokens_input or latency columns not found (need per-run data)")
        return
    
    # Filter out extreme outliers (likely API timeouts/retries)
    # Use 99th percentile as cutoff
    latency_99th = df['latency'].quantile(0.99)
    df_filtered = df[df['latency'] <= latency_99th].copy()
    n_outliers = len(df) - len(df_filtered)
    if n_outliers > 0:
        print(f"  → Filtered {n_outliers} outliers (latency > {latency_99th:.1f}s)")
    
    # Sample for performance if dataset is very large (>5000 points)
    if len(df_filtered) > 5000:
        df_plot = df_filtered.sample(n=5000, random_state=42)
        print(f"  → Sampling 5000 points from {len(df_filtered)} total for performance")
    else:
        df_plot = df_filtered
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each strategy
    for strategy in ['naive', 'structured', 'rag', 'advanced_rag']:
        subset = df_plot[df_plot['strategy'] == strategy]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset['tokens_input'] / 1000,  # Convert to thousands
            subset['latency'],
            alpha=0.3,
            s=20,
            label=strategy.replace('_', ' ').title(),
            color=COLOR_PALETTE.get(strategy, '#888888')
        )
    
    # Add annotations
    # RAG annotation (stays constant)
    rag_data = df_plot[df_plot['strategy'].isin(['rag', 'advanced_rag'])]
    if len(rag_data) > 0:
        rag_median_tokens = rag_data['tokens_input'].median() / 1000
        rag_median_latency = rag_data['latency'].median()
        ax.annotate(
            'RAG: Stays fast\nregardless of corpus size',
            xy=(rag_median_tokens, rag_median_latency),
            xytext=(200, 15),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#4EA8DE', lw=1.5),
            color='#4EA8DE',
            fontweight='bold'
        )
    
    # Full context annotation (scales)
    full_context_data = df_plot[df_plot['strategy'].isin(['naive', 'structured'])]
    if len(full_context_data) > 0:
        # Find a high-token, high-latency point for annotation
        high_token_point = full_context_data.nlargest(100, 'tokens_input').iloc[0]
        ax.annotate(
            'Full context: Latency\nscales with tokens',
            xy=(high_token_point['tokens_input'] / 1000, high_token_point['latency']),
            xytext=(500, 70),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#E76F51', lw=1.5),
            color='#E76F51',
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xlabel('Context Size (thousands of tokens)', fontsize=12)
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency vs Context Size by Strategy\n(Each point is one API call)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_latency_vs_tokens.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: exp1_latency_vs_tokens.png")


def generate_strategy_fill_heatmap(df: pd.DataFrame, out_dir: Path):
    """
    Generate heatmap showing F1 scores for each strategy at each fill level.
    
    Key insight: Shows at a glance which strategy excels at which fill level.
    Highlights naive's collapse at 50% fill with a red border.
    """
    print("Generating Strategy × Fill % Interaction Heatmap...")
    
    # Check if we have the required columns
    if 'fill_pct' not in df.columns or 'f1' not in df.columns or 'strategy' not in df.columns:
        print("  ⚠ Skipping: fill_pct, f1, or strategy columns not found (need Exp1 summary data)")
        return
    
    # Pivot to create heatmap matrix
    pivot = df.pivot(index='strategy', columns='fill_pct', values='f1')
    
    # Reorder rows by mean F1 (descending)
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    
    # Rename columns for display
    pivot.columns = [f'{int(c*100)}%' for c in pivot.columns]
    
    # Rename index for display
    pivot.index = pivot.index.map(lambda x: x.replace('_', ' ').title())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='YlGn',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'F1 Score'},
        ax=ax,
        vmin=0,
        vmax=0.30
    )
    
    # Highlight the anomaly cell (Naive @ 50%)
    # Find the position of the naive row and 50% column
    try:
        naive_row = list(pivot.index).index('Naive')
        fifty_col = list(pivot.columns).index('50%')
        # Add red border to highlight the collapse
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((fifty_col, naive_row), 1, 1, fill=False, edgecolor='red', lw=3))
    except (ValueError, IndexError):
        pass  # Skip if naive or 50% not found
    
    ax.set_title('F1 Score by Strategy and Fill Level\n(Experiment 1: Needle in Multiple Haystacks)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Context Fill Percentage', fontsize=11)
    ax.set_ylabel('Strategy', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_strategy_fill_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: exp1_strategy_fill_heatmap.png")


def generate_relative_lift_chart(df: pd.DataFrame, out_dir: Path):
    """
    Generate horizontal bar chart showing relative improvement over naive baseline.
    
    Makes the headline finding unmistakable: "Structured beats naive by 68%"
    """
    print("Generating Relative Performance Lift Chart...")
    
    # Check if we have the required columns
    if 'strategy' not in df.columns or 'f1' not in df.columns:
        print("  ⚠ Skipping: strategy or f1 columns not found")
        return
    
    # Calculate mean F1 per strategy
    strategy_means = df.groupby('strategy')['f1'].mean().reset_index()
    strategy_means.columns = ['strategy', 'mean_f1']
    
    # Get naive baseline
    naive_f1 = strategy_means[strategy_means['strategy'] == 'naive']['mean_f1'].values
    if len(naive_f1) == 0:
        print("  ⚠ Skipping: naive strategy not found in data")
        return
    naive_f1 = naive_f1[0]
    
    # Calculate lift for each strategy
    strategy_means['lift'] = ((strategy_means['mean_f1'] - naive_f1) / naive_f1 * 100)
    
    # Sort by lift
    strategy_means = strategy_means.sort_values('lift', ascending=True)  # Ascending for horizontal bars
    
    # Map colors
    colors = [COLOR_PALETTE.get(s, '#888888') for s in strategy_means['strategy']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot horizontal bars
    bars = ax.barh(strategy_means['strategy'], strategy_means['lift'], 
                   color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val, strat in zip(bars, strategy_means['lift'], strategy_means['strategy']):
        if val == 0:
            label = 'Baseline'
            x_pos = 5
            ha = 'left'
        else:
            label = f'+{val:.0f}%'
            x_pos = val + 2
            ha = 'left'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label, 
                va='center', ha=ha, fontsize=11, fontweight='bold')
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Performance Lift vs Naive Baseline (%)', fontsize=12)
    ax.set_title('Relative Improvement Over Naive Context Stuffing\n(Experiment 1)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks(range(len(strategy_means)))
    ax.set_yticklabels([s.replace('_', ' ').title() for s in strategy_means['strategy']])
    
    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_relative_lift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: exp1_relative_lift.png")


def generate_pareto_plot(df: pd.DataFrame, out_dir: Path):
    """
    Generate Pareto frontier plot showing quality vs latency trade-offs.
    
    Shows which strategies are Pareto-optimal (can't improve quality without sacrificing latency).
    """
    print("Generating Quality-Latency Pareto Plot...")
    
    # Check if we have the required columns
    if 'strategy' not in df.columns or 'f1' not in df.columns or 'latency' not in df.columns:
        print("  ⚠ Skipping: strategy, f1, or latency columns not found")
        return
    
    # Calculate mean F1 and latency per strategy
    stats = df.groupby('strategy').agg({'f1': 'mean', 'latency': 'mean'}).reset_index()
    stats.columns = ['strategy', 'mean_f1', 'mean_latency']
    
    # Identify Pareto-optimal points
    # A point is dominated if another has both higher F1 AND lower latency
    pareto_points = []
    for i, row in stats.iterrows():
        dominated = False
        for j, other_row in stats.iterrows():
            if i != j:
                # Check if other point dominates this one
                if (other_row['mean_f1'] >= row['mean_f1'] and 
                    other_row['mean_latency'] <= row['mean_latency'] and 
                    (other_row['mean_f1'] > row['mean_f1'] or other_row['mean_latency'] < row['mean_latency'])):
                    dominated = True
                    break
        if not dominated:
            pareto_points.append((row['mean_f1'], row['mean_latency'], row['strategy']))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot each strategy
    for _, row in stats.iterrows():
        strategy = row['strategy']
        f1 = row['mean_f1']
        lat = row['mean_latency']
        color = COLOR_PALETTE.get(strategy, '#888888')
        
        ax.scatter(f1, lat, s=200, c=color, edgecolor='black', linewidth=1.5, zorder=3)
        
        # Add labels with smart positioning
        offset_x = 0.008 if strategy != 'structured' else -0.025
        offset_y = 1.5 if strategy != 'naive' else -3
        ax.annotate(
            strategy.replace('_', ' ').title(), 
            (f1 + offset_x, lat + offset_y), 
            fontsize=11, 
            fontweight='bold'
        )
    
    # Draw Pareto frontier (connect non-dominated points)
    if len(pareto_points) > 1:
        # Sort by F1 for drawing
        pareto_points.sort(key=lambda x: x[0])
        pareto_f1 = [p[0] for p in pareto_points]
        pareto_lat = [p[1] for p in pareto_points]
        ax.plot(pareto_f1, pareto_lat, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier', zorder=1)
    
    # Add quadrant labels
    ax.text(0.14, 50, 'Low Quality\nHigh Latency\n(Avoid)', ha='center', fontsize=9, 
            color='gray', alpha=0.7)
    ax.text(0.22, 30, 'High Quality\nLow Latency\n(Ideal)', ha='center', fontsize=9, 
            color='green', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Mean F1 Score (Quality →)', fontsize=12)
    ax.set_ylabel('Mean Latency in seconds (← Lower is better)', fontsize=12)
    ax.set_title('Quality vs Latency Trade-off\n(Experiment 1)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.10, 0.26)
    ax.set_ylim(25, 55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'pareto_quality_latency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: pareto_quality_latency.png")


def generate_summary_table(df: pd.DataFrame, out_dir: Path):
    """
    Generate publication-ready summary statistics table as an image.
    
    Provides a clean, embeddable table for articles showing key metrics.
    """
    print("Generating Summary Statistics Table...")
    
    # Check if we have the required columns
    if 'strategy' not in df.columns or 'f1' not in df.columns:
        print("  ⚠ Skipping: strategy or f1 columns not found")
        return
    
    # Calculate statistics per strategy
    strategy_stats = df.groupby('strategy').agg({
        'f1': ['mean', 'std'],
        'latency': 'mean' if 'latency' in df.columns else lambda x: None
    }).reset_index()
    
    # Flatten column names
    strategy_stats.columns = ['strategy', 'mean_f1', 'std_f1', 'mean_latency']
    
    # Calculate 95% CI (using t-distribution for small samples)
    # Assuming 5 fill levels per strategy (n=5), t-critical ≈ 2.776
    strategy_stats['ci_95'] = 2.776 * (strategy_stats['std_f1'] / np.sqrt(5))
    
    # Get naive baseline for comparison
    naive_f1 = strategy_stats[strategy_stats['strategy'] == 'naive']['mean_f1'].values
    if len(naive_f1) > 0:
        naive_f1 = naive_f1[0]
        strategy_stats['vs_naive'] = ((strategy_stats['mean_f1'] - naive_f1) / naive_f1 * 100)
    else:
        strategy_stats['vs_naive'] = 0
    
    # Find best fill % for each strategy (needs fill_pct column)
    if 'fill_pct' in df.columns:
        best_fill = df.loc[df.groupby('strategy')['f1'].idxmax()][['strategy', 'fill_pct']]
        strategy_stats = strategy_stats.merge(best_fill, on='strategy', how='left')
    else:
        strategy_stats['fill_pct'] = None
    
    # Sort by mean F1 descending
    strategy_stats = strategy_stats.sort_values('mean_f1', ascending=False)
    
    # Build display table
    table_data = []
    for _, row in strategy_stats.iterrows():
        strat = row['strategy'].replace('_', ' ').title()
        f1_str = f"{row['mean_f1']:.3f}"
        ci_str = f"±{row['ci_95']:.3f}"
        
        if row['strategy'] == 'naive':
            vs_naive_str = 'baseline'
        else:
            vs_naive_str = f"+{row['vs_naive']:.0f}%"
        
        best_fill_str = f"{int(row['fill_pct']*100)}%" if pd.notna(row.get('fill_pct')) else 'N/A'
        latency_str = f"{row['mean_latency']:.1f}s" if pd.notna(row['mean_latency']) else 'N/A'
        
        table_data.append([strat, f1_str, ci_str, vs_naive_str, best_fill_str, latency_str])
    
    # Create DataFrame for table
    df_table = pd.DataFrame(table_data, columns=[
        'Strategy', 'Mean F1', '95% CI', 'vs Naive', 'Best Fill %', 'Avg Latency'
    ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc='center',
        loc='center',
        colColours=['#2E86AB']*6
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(len(df_table.columns)):
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight the "+68%" cell (structured, vs Naive column)
    for i in range(1, len(df_table)+1):
        cell_text = table[(i, 3)].get_text().get_text()
        if '+68%' in cell_text or '+67%' in cell_text:  # Account for rounding
            table[(i, 3)].set_facecolor('#d4edda')
    
    plt.title('Experiment 1 Summary: Strategy Performance Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: summary_table.png")


def generate_pollution_robustness_fixed(df: pd.DataFrame, out_dir: Path):
    """
    Generate fixed pollution robustness chart for Experiment 2.
    
    Fixes: y-axis 0-0.40 instead of too wide, categorical x-axis for even spacing,
    highlights 950k zone where retrieval strategies jump ahead.
    Key insight: At 50k-700k pollution, all cluster at F1 ~0.05-0.07. 
    At 950k, RAG jumps to 0.31 while naive stays at 0.15.
    """
    print("Generating Fixed Pollution Robustness Chart...")
    
    # Check if we have the required columns
    if 'pollution_level' not in df.columns or 'f1' not in df.columns:
        print("  ⚠ Skipping: pollution_level or f1 columns not found (need Exp2 data)")
        return
    
    # Use categorical x-axis for even spacing
    pollution_labels = ['50k', '200k', '500k', '700k', '950k']
    pollution_values = [50000, 200000, 500000, 700000, 950000]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each strategy
    for strategy in ['rag', 'advanced_rag', 'structured', 'naive']:
        subset = df[df['strategy'] == strategy].copy()
        if len(subset) == 0:
            continue
        subset = subset.sort_values('pollution_level')
        
        # Map pollution levels to categorical positions
        x_positions = [pollution_values.index(p) for p in subset['pollution_level']]
        
        ax.plot(
            x_positions,
            subset['f1'].values,
            marker='o',
            linewidth=2,
            markersize=8,
            label=strategy.replace('_', ' ').title(),
            color=COLOR_PALETTE.get(strategy, '#888888')
        )
    
    # Highlight the 950k zone (position 4)
    ax.axvspan(3.5, 4.5, alpha=0.15, color='green', label='Retrieval advantage zone')
    
    # Add annotation for RAG's jump at 950k
    rag_950k = df[(df['strategy'] == 'rag') & (df['pollution_level'] == 950000)]
    if len(rag_950k) > 0:
        rag_f1 = rag_950k['f1'].values[0]
        ax.annotate(
            'RAG filters noise\nat extreme pollution',
            xy=(4, rag_f1),
            xytext=(2.5, 0.32),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            color='green',
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xticks(range(len(pollution_labels)))
    ax.set_xticklabels(pollution_labels)
    ax.set_xlabel('Pollution Level (irrelevant tokens added)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Robustness to Context Pollution (Experiment 2)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.40)  # FIXED: was likely 0-1.0, now 0-0.40
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'exp2_pollution_robustness_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: exp2_pollution_robustness_fixed.png")


def generate_visualizations(input_csv: str, output_dir: str):
    """
    Generate standard plots for experiment analysis.
    """
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from {input_path}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Detect if this is Exp1 data (has fill_pct) or Exp2 data (has pollution_level)
    is_exp1 = 'fill_pct' in df.columns
    is_exp2 = 'pollution_level' in df.columns
    is_per_run = 'question_id' in df.columns  # Per-run data has question_id
    
    if is_exp1:
        # Generate fixed Exp1 visualizations
        if 'strategy' in df.columns and 'f1' in df.columns:
            generate_strategy_comparison_fixed(df, out_dir)
            # Generate relative lift chart (needs summary data)
            if not is_per_run:
                generate_relative_lift_chart(df, out_dir)
        if 'fill_pct' in df.columns and 'f1' in df.columns and 'strategy' in df.columns:
            generate_degradation_curve_fixed(df, out_dir)
            # Generate heatmap (needs summary data, not per-run)
            if not is_per_run:
                generate_strategy_fill_heatmap(df, out_dir)
        if is_per_run:
            # Generate latency vs tokens scatter (needs per-run data)
            generate_latency_vs_tokens_scatter(df, out_dir)
            # Generate Pareto plot (needs per-run data for latency)
            generate_pareto_plot(df, out_dir)
            # Generate summary table (works best with per-run data for accurate stats)
            generate_summary_table(df, out_dir)
    
    if is_exp2:
        # Generate fixed Exp2 visualizations
        if 'strategy' in df.columns and 'f1' in df.columns and 'pollution_level' in df.columns:
            generate_pollution_robustness_fixed(df, out_dir)
    
    # Note: Old chart generation code (degradation_curve_f1.png, strategy_comparison_f1.png,
    # latency_distribution.png) has been removed. These have been replaced by:
    # - exp1_degradation_curve_fixed.png (proper y-axis, aggregated data)
    # - exp1_strategy_comparison_fixed.png (proper y-axis, error bars, sorted)
    # - exp1_latency_vs_tokens.png (scatter showing latency scales with tokens)
    # See ANALYSIS_CONCLUSIONS.md for rationale.

    print(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from metrics CSV.")
    parser.add_argument("--input", required=True, help="Path to summary_metrics.csv or scored_results.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for output plots")
    
    args = parser.parse_args()
    
    generate_visualizations(args.input, args.output_dir)
