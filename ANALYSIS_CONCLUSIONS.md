# Visualization Rebuild Plan for Context Engineering Experiments

**Purpose:** This document provides detailed specifications for rebuilding the visualizations that support ARTICLE_CONCLUSIONS.md. The current charts have scaling issues that make findings hard to interpret. This plan addresses those issues and adds new visualizations to better tell the experimental story.

**Target Audience:** Developers implementing these visualizations.

**Data Sources:**
- Experiment 1: `results/analysis/exp1_rerun/summary_metrics.csv` and `results/analysis/exp1_full/scored_results.csv`
- Experiment 2: `results/analysis/exp2/summary_metrics.csv` and `results/analysis/exp2/scored_results.csv`

---

## Table of Contents

1. [Critical Fixes to Existing Charts](#1-critical-fixes-to-existing-charts)
2. [Charts to Delete](#2-charts-to-delete)
3. [New Charts to Create](#3-new-charts-to-create)
4. [Data Reference](#4-data-reference)
5. [Implementation Checklist](#5-implementation-checklist)

---

## 1. Critical Fixes to Existing Charts

### 1.1 Fix: Strategy Comparison Bar Chart (Exp1)

**Current Location:** `results/analysis/exp1_full/strategy_comparison_f1.png`

**Problem:** 
- Y-axis is hardcoded to `plt.ylim(0, 1.0)` in `scripts/generate_visualizations.py` line 56
- Actual F1 data ranges from 0.136 to 0.228
- This makes all bars look tiny and nearly identical, hiding the 68% relative difference

**Fix Required:**

```python
# BEFORE (wrong)
plt.ylim(0, 1.0)

# AFTER (correct)
plt.ylim(0, 0.35)
```

**Additional Improvements:**
1. Add error bars showing 95% confidence intervals
2. Sort bars from highest to lowest F1 (Structured → RAG → Adv RAG → Naive)
3. Add value labels on top of each bar
4. Use a color scheme that distinguishes "engineered" (blue tones) from "naive" (red/orange)

**Expected Output:**
- Structured bar clearly taller than others
- Naive bar visibly shorter (~40% height of Structured)
- Error bars showing variance from 3 repetitions
- Clear visual that says "engineering helps"

**Data to Use (from exp1_rerun/summary_metrics.csv, averaged across fill levels):**
| Strategy | Mean F1 | 
|----------|---------|
| structured | 0.228 |
| rag | 0.221 |
| advanced_rag | 0.217 |
| naive | 0.136 |

**Code Snippet (reference implementation):**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('results/analysis/exp1_rerun/summary_metrics.csv')

# Calculate mean F1 per strategy (averaging across fill levels)
strategy_means = df.groupby('strategy')['f1'].agg(['mean', 'std']).reset_index()
strategy_means.columns = ['strategy', 'mean_f1', 'std_f1']

# Calculate 95% CI (using t-distribution, df=4 for 5 fill levels)
# CI = mean ± t * (std / sqrt(n))
n_fill_levels = 5
t_value = 2.776  # t-critical for 95% CI, df=4
strategy_means['ci_95'] = t_value * (strategy_means['std_f1'] / np.sqrt(n_fill_levels))

# Sort by mean F1 descending
strategy_means = strategy_means.sort_values('mean_f1', ascending=False)

# Define colors: engineered strategies in blues, naive in orange
color_map = {
    'structured': '#2E86AB',    # Blue
    'rag': '#4EA8DE',           # Light blue
    'advanced_rag': '#7FC8F8',  # Lighter blue
    'naive': '#E76F51'          # Orange/red
}
colors = [color_map[s] for s in strategy_means['strategy']]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with error bars
bars = ax.bar(
    strategy_means['strategy'], 
    strategy_means['mean_f1'],
    yerr=strategy_means['ci_95'],
    capsize=5,
    color=colors,
    edgecolor='black',
    linewidth=1
)

# Add value labels on top of bars
for bar, val in zip(bars, strategy_means['mean_f1']):
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
ax.set_ylim(0, 0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add horizontal line at naive's score for reference
naive_score = strategy_means[strategy_means['strategy'] == 'naive']['mean_f1'].values[0]
ax.axhline(y=naive_score, color='gray', linestyle='--', alpha=0.5, label=f'Naive baseline ({naive_score:.3f})')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('results/visualizations/exp1_strategy_comparison_fixed.png', dpi=150)
plt.close()
```

---

### 1.2 Fix: Fill % Degradation Curve (Exp1)

**Current Location:** `results/analysis/exp1_full/degradation_curve_f1.png`

**Problem:**
- Y-axis hardcoded to 0-1.0
- F1 values range from 0.018 to 0.262
- Lines appear flat and differences are invisible
- No confidence intervals shown

**Fix Required:**
1. Change y-axis to `plt.ylim(0, 0.35)`
2. Add shaded confidence bands (not just points)
3. Highlight the "danger zone" at 50-70% fill where naive collapses

**Data to Use (from exp1_rerun/summary_metrics.csv):**
```
strategy,fill_pct,f1
advanced_rag,0.1,0.262
advanced_rag,0.3,0.246
advanced_rag,0.5,0.219
advanced_rag,0.7,0.182
advanced_rag,0.9,0.176
naive,0.1,0.198
naive,0.3,0.188
naive,0.5,0.019  <-- ANOMALY: Drops to near zero
naive,0.7,0.085
naive,0.9,0.189
rag,0.1,0.242
rag,0.3,0.257
rag,0.5,0.221
rag,0.7,0.197
rag,0.9,0.189
structured,0.1,0.220
structured,0.3,0.228
structured,0.5,0.234
structured,0.7,0.229
structured,0.9,0.229
```

**Key Insight to Visualize:**
- Naive collapses at 50% fill (0.019 F1) — this is a critical finding
- Structured stays flat across all fill levels (~0.22-0.23)
- RAG degrades gradually but doesn't collapse

**Code Snippet:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results/analysis/exp1_rerun/summary_metrics.csv')

# Define colors
palette = {
    'structured': '#2E86AB',
    'rag': '#4EA8DE', 
    'advanced_rag': '#7FC8F8',
    'naive': '#E76F51'
}

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each strategy
for strategy in ['structured', 'rag', 'advanced_rag', 'naive']:
    subset = df[df['strategy'] == strategy]
    ax.plot(
        subset['fill_pct'], 
        subset['f1'], 
        marker='o', 
        linewidth=2,
        markersize=8,
        label=strategy.replace('_', ' ').title(),
        color=palette[strategy]
    )

# Highlight the danger zone (50-70% fill)
ax.axvspan(0.45, 0.75, alpha=0.1, color='red', label='Degradation Zone')

# Add annotation for naive's collapse
ax.annotate(
    'Naive collapses\nat 50% fill',
    xy=(0.5, 0.019),
    xytext=(0.55, 0.08),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    color='red'
)

# Formatting
ax.set_xlabel('Fill Percentage', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Performance Degradation by Context Fill %', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.35)
ax.set_xlim(0.05, 0.95)
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'])
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/exp1_degradation_curve_fixed.png', dpi=150)
plt.close()
```

---

### 1.3 Fix: Latency Distribution Chart (Exp1)

**Current Location:** `results/analysis/exp1_full/latency_distribution.png`

**Problem:**
- Current chart shows boxplot of latency per strategy
- This is misleading because latency correlates with tokens_input, not strategy per se
- RAG uses ~92k tokens while naive uses ~96k-862k tokens
- Comparing latency without context size is comparing apples to oranges

**Decision: REPLACE this chart**

Instead of a simple boxplot, create a scatter plot showing **Latency vs Context Size (tokens_input)**, colored by strategy. This reveals the true relationship.

See Section 3.3 for the replacement chart specification.

---

### 1.4 Fix: Pollution vs F1 Chart (Exp2)

**Current Location:** `results/analysis/exp2/pollution_vs_f1.png`

**Problem:**
- Y-axis likely set too wide
- F1 values range from 0.045 to 0.314
- The critical finding (RAG jumps at 950k) may not be visible

**Fix Required:**
1. Y-axis: `plt.ylim(0, 0.40)`
2. Add vertical annotation line at 950k highlighting "Retrieval wins here"
3. Use log scale or evenly-spaced categorical x-axis (pollution levels aren't evenly spaced: 50k, 200k, 500k, 700k, 950k)

**Data to Use (from exp2/summary_metrics.csv):**
```
pollution_level,strategy,f1
50000,advanced_rag,0.057
50000,naive,0.068
50000,rag,0.052
50000,structured,0.074
200000,advanced_rag,0.046
200000,naive,0.056
200000,rag,0.045
200000,structured,0.067
500000,advanced_rag,0.050
500000,naive,0.073
500000,rag,0.052
500000,structured,0.065
700000,advanced_rag,0.048
700000,naive,0.069
700000,rag,0.052
700000,structured,0.056
950000,advanced_rag,0.314  <-- JUMP
950000,naive,0.148
950000,rag,0.307           <-- JUMP
950000,structured,0.233
```

**Key Insight:**
At 50k-700k pollution, all strategies cluster at F1 ≈ 0.05-0.07. At 950k, retrieval strategies (RAG, Advanced RAG) jump to 0.31 while full-context strategies (naive, structured) stay at 0.15-0.23.

**Code Snippet:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/analysis/exp2/summary_metrics.csv')

# Use categorical x-axis for even spacing
pollution_labels = ['50k', '200k', '500k', '700k', '950k']
pollution_values = [50000, 200000, 500000, 700000, 950000]

palette = {
    'structured': '#2E86AB',
    'rag': '#4EA8DE', 
    'advanced_rag': '#7FC8F8',
    'naive': '#E76F51'
}

fig, ax = plt.subplots(figsize=(10, 6))

for strategy in ['rag', 'advanced_rag', 'structured', 'naive']:
    subset = df[df['strategy'] == strategy].sort_values('pollution_level')
    ax.plot(
        range(len(pollution_values)),  # categorical positions
        subset['f1'].values,
        marker='o',
        linewidth=2,
        markersize=8,
        label=strategy.replace('_', ' ').title(),
        color=palette[strategy]
    )

# Highlight the 950k zone
ax.axvspan(3.5, 4.5, alpha=0.15, color='green', label='Retrieval advantage zone')

# Add annotation
ax.annotate(
    'RAG filters noise\nat extreme pollution',
    xy=(4, 0.31),
    xytext=(2.5, 0.32),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
    color='green'
)

# Formatting
ax.set_xticks(range(len(pollution_labels)))
ax.set_xticklabels(pollution_labels)
ax.set_xlabel('Pollution Level (irrelevant tokens added)', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Robustness to Context Pollution (Experiment 2)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.40)
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/exp2_pollution_robustness_fixed.png', dpi=150)
plt.close()
```

---

## 2. Charts to Delete

### 2.1 DELETE: Pollution vs Exact Match (Exp2)

**Current Location:** `results/analysis/exp2/pollution_vs_em.png`

**Reason for Deletion:**
- Exact Match (EM) is **0.0 for every single data point** across all strategies, all pollution levels, all repetitions
- The chart is literally a flat line at zero
- It provides zero information and will confuse readers

**Root Cause of Zero EM:**
Exact match requires the model's response to exactly equal the ground truth string. Our questions have short ground truth answers (e.g., "1024 tokens", "v3.2"), but the model produces longer explanatory responses. This is a metric mismatch, not a model failure. F1 score handles this better by measuring token overlap.

**Action:** 
1. Delete `results/analysis/exp2/pollution_vs_em.png`
2. Do not regenerate this chart
3. Remove any references to "exact match" charts from documentation

---

### 2.2 DELETE/REPLACE: Original Latency Distribution Boxplot (Exp1)

**Current Location:** `results/analysis/exp1_full/latency_distribution.png`

**Reason for Deletion:**
- Boxplot of latency per strategy is misleading
- It suggests "RAG is faster than naive" but ignores that RAG processes ~10x fewer tokens
- The real insight is latency scales with context size, not strategy choice

**Action:**
1. Delete `results/analysis/exp1_full/latency_distribution.png`
2. Replace with "Latency vs Context Size Scatter Plot" (see Section 3.3)

---

## 3. New Charts to Create

### 3.1 NEW: Strategy × Fill % Interaction Heatmap (Exp1)

**Purpose:** Show at a glance which strategy excels at which fill level. Reveals the naive collapse at 50% fill and structured's stability.

**Output Location:** `results/visualizations/exp1_strategy_fill_heatmap.png`

**Data Structure:**
```
         | 10%   | 30%   | 50%   | 70%   | 90%
---------|-------|-------|-------|-------|-------
Adv RAG  | 0.262 | 0.246 | 0.219 | 0.182 | 0.176
RAG      | 0.242 | 0.257 | 0.221 | 0.197 | 0.189
Struct.  | 0.220 | 0.228 | 0.234 | 0.229 | 0.229
Naive    | 0.198 | 0.188 | 0.019 | 0.085 | 0.189
```

**Visual Specification:**
- Rows: Strategies (ordered by mean F1: Adv RAG, RAG, Structured, Naive)
- Columns: Fill percentages (10%, 30%, 50%, 70%, 90%)
- Color: Sequential colormap (e.g., `YlGn` or `Blues`) where darker = higher F1
- Annotations: Show F1 value in each cell
- Highlight: Red border or different color for the naive@50% cell (0.019) to draw attention

**Code Snippet:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('results/analysis/exp1_rerun/summary_metrics.csv')

# Pivot to create heatmap matrix
pivot = df.pivot(index='strategy', columns='fill_pct', values='f1')

# Reorder rows by mean F1
row_order = pivot.mean(axis=1).sort_values(ascending=False).index
pivot = pivot.loc[row_order]

# Rename columns for display
pivot.columns = ['10%', '30%', '50%', '70%', '90%']

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
naive_row = list(pivot.index).index('Naive')
fifty_col = list(pivot.columns).index('50%')
ax.add_patch(plt.Rectangle((fifty_col, naive_row), 1, 1, fill=False, edgecolor='red', lw=3))

ax.set_title('F1 Score by Strategy and Fill Level\n(Experiment 1: Needle in Multiple Haystacks)', fontsize=13, fontweight='bold')
ax.set_xlabel('Context Fill Percentage', fontsize=11)
ax.set_ylabel('Strategy', fontsize=11)

plt.tight_layout()
plt.savefig('results/visualizations/exp1_strategy_fill_heatmap.png', dpi=150)
plt.close()
```

---

### 3.2 NEW: Relative Performance Lift Chart (Exp1)

**Purpose:** Make the headline finding unmistakable: "Structured beats naive by 68%"

**Output Location:** `results/visualizations/exp1_relative_lift.png`

**Calculation:**
```
Lift = (Strategy_F1 - Naive_F1) / Naive_F1 * 100

Naive: 0.136 → 0% (baseline)
Advanced RAG: 0.217 → +60% lift
RAG: 0.221 → +63% lift  
Structured: 0.228 → +68% lift
```

**Visual Specification:**
- Horizontal bar chart (easier to read percentages)
- Naive at 0% (baseline, shown as vertical line)
- Other strategies as bars extending to the right
- Color: Green gradient (more lift = darker green)
- Labels: Show exact percentage on each bar

**Code Snippet:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Data
strategies = ['Naive', 'Advanced RAG', 'RAG', 'Structured']
mean_f1 = [0.136, 0.217, 0.221, 0.228]
naive_f1 = 0.136

# Calculate lift
lift = [(f1 - naive_f1) / naive_f1 * 100 for f1 in mean_f1]

# Colors
colors = ['#E76F51', '#7FC8F8', '#4EA8DE', '#2E86AB']

fig, ax = plt.subplots(figsize=(10, 5))

# Plot horizontal bars
bars = ax.barh(strategies, lift, color=colors, edgecolor='black', linewidth=1)

# Add value labels
for bar, val in zip(bars, lift):
    if val == 0:
        label = 'Baseline'
        x_pos = 5
    else:
        label = f'+{val:.0f}%'
        x_pos = val + 2
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, label, 
            va='center', ha='left', fontsize=11, fontweight='bold')

# Add vertical line at 0
ax.axvline(x=0, color='black', linewidth=1.5)

# Formatting
ax.set_xlabel('Performance Lift vs Naive Baseline (%)', fontsize=12)
ax.set_title('Relative Improvement Over Naive Context Stuffing\n(Experiment 1)', fontsize=14, fontweight='bold')
ax.set_xlim(-5, 85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/visualizations/exp1_relative_lift.png', dpi=150)
plt.close()
```

---

### 3.3 NEW: Latency vs Context Size Scatter (Exp1)

**Purpose:** Show that latency scales with context size, revealing the true cost of full-context strategies.

**Output Location:** `results/visualizations/exp1_latency_vs_tokens.png`

**Data Source:** `results/analysis/exp1_full/scored_results.csv` (has per-run tokens_input and latency)

**Visual Specification:**
- X-axis: tokens_input (log scale or linear, 50k → 900k)
- Y-axis: latency in seconds
- Points colored by strategy
- Add trend lines (linear regression) per strategy
- Key insight: RAG stays flat at ~20-30s, Naive/Structured scale linearly

**Expected Pattern:**
- RAG/Advanced RAG: Cluster at low tokens (~92k) with latency ~12-50s
- Naive: Spreads from 97k→759k with latency 9-55s
- Structured: Spreads from 98k→862k with latency 23-70s

**Code Snippet:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results/analysis/exp1_full/scored_results.csv')

# Sample for performance (or use all ~17k points if rendering is fast)
# df = df.sample(n=3000, random_state=42)

palette = {
    'structured': '#2E86AB',
    'rag': '#4EA8DE', 
    'advanced_rag': '#7FC8F8',
    'naive': '#E76F51'
}

fig, ax = plt.subplots(figsize=(12, 7))

for strategy in ['naive', 'structured', 'rag', 'advanced_rag']:
    subset = df[df['strategy'] == strategy]
    ax.scatter(
        subset['tokens_input'] / 1000,  # Convert to k
        subset['latency'],
        alpha=0.3,
        s=20,
        label=strategy.replace('_', ' ').title(),
        color=palette[strategy]
    )

# Add annotations
ax.annotate(
    'RAG: Stays fast\nregardless of corpus size',
    xy=(92, 20),
    xytext=(200, 15),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='#4EA8DE', lw=1.5),
    color='#4EA8DE'
)

ax.annotate(
    'Full context: Latency\nscales with tokens',
    xy=(700, 55),
    xytext=(500, 70),
    fontsize=10,
    arrowprops=dict(arrowstyle='->', color='#E76F51', lw=1.5),
    color='#E76F51'
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
plt.savefig('results/visualizations/exp1_latency_vs_tokens.png', dpi=150)
plt.close()
```

---

### 3.4 NEW: Quality-Latency Pareto Plot (Combined Exp1+Exp2)

**Purpose:** Show the trade-off frontier — which strategies are Pareto-optimal (can't improve quality without sacrificing latency).

**Output Location:** `results/visualizations/pareto_quality_latency.png`

**Data:** Aggregate from both exp1 and exp2 summary metrics.

**Visual Specification:**
- X-axis: Mean F1 Score (quality)
- Y-axis: Mean Latency (seconds) — note: lower is better
- Each point: One strategy (averaged across conditions)
- Different markers for Exp1 vs Exp2
- Connect Pareto-optimal points with a line

**Expected Pareto Frontier:**
- High quality, high latency: Structured (F1=0.228, latency=46s)
- Medium quality, low latency: RAG (F1=0.221, latency=40s) — probably Pareto-optimal
- Low quality, medium latency: Naive (F1=0.136, latency=33s) — dominated

**Code Snippet:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Aggregated data from exp1 summary
strategies = ['Naive', 'Structured', 'RAG', 'Advanced RAG']
exp1_f1 = [0.136, 0.228, 0.221, 0.217]
exp1_latency = [32.6, 45.8, 48.3, 35.3]  # From exp1_full summary_metrics.csv

# Note: For Exp2, RAG latency is ~1s because it retrieves only ~7k tokens
# That's a different comparison (128k window RAG vs 1M window)
# For Pareto, we should probably keep Exp1 data only for fair comparison

colors = ['#E76F51', '#2E86AB', '#4EA8DE', '#7FC8F8']

fig, ax = plt.subplots(figsize=(10, 7))

# Plot points
for i, (f1, lat, strat, col) in enumerate(zip(exp1_f1, exp1_latency, strategies, colors)):
    ax.scatter(f1, lat, s=200, c=col, edgecolor='black', linewidth=1.5, zorder=3)
    
    # Add labels
    offset_x = 0.008 if strat != 'Structured' else -0.025
    offset_y = 1.5 if strat != 'Naive' else -3
    ax.annotate(strat, (f1 + offset_x, lat + offset_y), fontsize=11, fontweight='bold')

# Draw Pareto frontier (connect non-dominated points)
# Sort by F1 and find Pareto optimal
# A point is dominated if another has both higher F1 AND lower latency
pareto_points = []
for i, (f1, lat) in enumerate(zip(exp1_f1, exp1_latency)):
    dominated = False
    for j, (f1_other, lat_other) in enumerate(zip(exp1_f1, exp1_latency)):
        if i != j and f1_other >= f1 and lat_other <= lat and (f1_other > f1 or lat_other < lat):
            dominated = True
            break
    if not dominated:
        pareto_points.append((f1, lat, strategies[i]))

# Sort Pareto points by F1 for drawing the frontier line
pareto_points.sort(key=lambda x: x[0])
if len(pareto_points) > 1:
    pareto_f1 = [p[0] for p in pareto_points]
    pareto_lat = [p[1] for p in pareto_points]
    ax.plot(pareto_f1, pareto_lat, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')

# Add quadrant labels
ax.text(0.14, 50, 'Low Quality\nHigh Latency\n(Avoid)', ha='center', fontsize=9, color='gray', alpha=0.7)
ax.text(0.22, 30, 'High Quality\nLow Latency\n(Ideal)', ha='center', fontsize=9, color='green', alpha=0.7)

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

# Invert y-axis might help (lower latency = better = higher on chart)
# ax.invert_yaxis()

plt.tight_layout()
plt.savefig('results/visualizations/pareto_quality_latency.png', dpi=150)
plt.close()
```

---

### 3.5 NEW: Summary Statistics Table (For Article Embedding)

**Purpose:** Sometimes a clean table communicates better than a chart. Create a publication-ready table.

**Output Location:** `results/visualizations/summary_table.png` (rendered as image for article)

**Table Content:**
| Strategy | Mean F1 | 95% CI | vs Naive | Best Fill % | Avg Latency |
|----------|---------|--------|----------|-------------|-------------|
| Structured | 0.228 | ±0.006 | **+68%** | 50% | 45.8s |
| RAG | 0.221 | ±0.028 | +63% | 30% | 48.3s |
| Advanced RAG | 0.217 | ±0.036 | +60% | 10% | 35.3s |
| Naive | 0.136 | ±0.074 | baseline | 10% | 32.6s |

**Code Snippet (using matplotlib table):**
```python
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Strategy': ['Structured', 'RAG', 'Advanced RAG', 'Naive'],
    'Mean F1': ['0.228', '0.221', '0.217', '0.136'],
    '95% CI': ['±0.006', '±0.028', '±0.036', '±0.074'],
    'vs Naive': ['+68%', '+63%', '+60%', 'baseline'],
    'Best Fill %': ['50%', '30%', '10%', '10%'],
    'Avg Latency': ['45.8s', '48.3s', '35.3s', '32.6s']
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#2E86AB']*6
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for j in range(len(df.columns)):
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Highlight the "vs Naive" column
for i in range(1, len(df)+1):
    if '+68%' in str(table[(i, 3)].get_text().get_text()):
        table[(i, 3)].set_facecolor('#d4edda')

plt.title('Experiment 1 Summary: Strategy Performance Comparison', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/visualizations/summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
```

---

## 4. Data Reference

### Experiment 1 Data Summary

**Source File:** `results/analysis/exp1_rerun/summary_metrics.csv`

| strategy | fill_pct | f1 | recall | exact_match | contains_truth |
|----------|----------|-----|--------|-------------|----------------|
| advanced_rag | 0.1 | 0.262 | 0.496 | 0.0 | 0.0 |
| advanced_rag | 0.3 | 0.246 | 0.522 | 0.0 | 0.003 |
| advanced_rag | 0.5 | 0.219 | 0.502 | 0.0 | 0.0 |
| advanced_rag | 0.7 | 0.182 | 0.505 | 0.0 | 0.0 |
| advanced_rag | 0.9 | 0.176 | 0.514 | 0.0 | 0.0 |
| naive | 0.1 | 0.198 | 0.432 | 0.0 | 0.0 |
| naive | 0.3 | 0.188 | 0.402 | 0.0 | 0.003 |
| naive | 0.5 | **0.019** | 0.344 | 0.0 | 0.0 |
| naive | 0.7 | 0.085 | 0.437 | 0.0 | 0.0 |
| naive | 0.9 | 0.189 | 0.415 | 0.0 | 0.0 |
| rag | 0.1 | 0.242 | 0.482 | 0.0 | 0.0 |
| rag | 0.3 | 0.257 | 0.522 | 0.0 | 0.003 |
| rag | 0.5 | 0.221 | 0.524 | 0.0 | 0.0 |
| rag | 0.7 | 0.197 | 0.505 | 0.0 | 0.002 |
| rag | 0.9 | 0.189 | 0.528 | 0.0 | 0.0 |
| structured | 0.1 | 0.220 | 0.425 | 0.0 | 0.0 |
| structured | 0.3 | 0.228 | 0.472 | 0.0 | 0.0 |
| structured | 0.5 | 0.234 | 0.464 | 0.0 | 0.0 |
| structured | 0.7 | 0.229 | 0.471 | 0.0 | 0.0 |
| structured | 0.9 | 0.229 | 0.476 | 0.0 | 0.002 |

**Key Observations:**
1. `exact_match` is always 0 — do not create charts for this metric
2. Naive at 50% fill drops to 0.019 — this is the critical anomaly
3. Structured stays stable at ~0.22-0.23 across all fill levels

### Experiment 2 Data Summary

**Source File:** `results/analysis/exp2/summary_metrics.csv`

| pollution_level | strategy | f1 | em | latency | tokens_input |
|-----------------|----------|-----|-----|---------|--------------|
| 50000 | advanced_rag | 0.057 | 0.0 | 1.02s | 6,807 |
| 50000 | naive | 0.068 | 0.0 | 2.84s | 111,547 |
| 50000 | rag | 0.052 | 0.0 | 1.02s | 6,783 |
| 50000 | structured | 0.074 | 0.0 | 3.00s | 112,397 |
| ... | ... | ... | ... | ... | ... |
| 950000 | advanced_rag | **0.314** | 0.0 | 1.39s | 6,758 |
| 950000 | naive | 0.148 | 0.0 | 20.17s | 991,108 |
| 950000 | rag | **0.307** | 0.0 | 1.55s | 6,887 |
| 950000 | structured | 0.233 | 0.0 | 20.00s | 1,002,311 |

**Key Observations:**
1. `em` (exact_match) is always 0 — DELETE the pollution_vs_em chart
2. RAG strategies use ~7k tokens; Naive/Structured use full context (111k → 1M)
3. At 950k pollution, RAG jumps to 0.31 F1 while Naive stays at 0.15

---

## 5. Implementation Checklist

### Files to Modify

- [ ] `scripts/generate_visualizations.py` — Fix y-axis limits, add new chart functions

### Files to Delete

- [ ] `results/analysis/exp2/pollution_vs_em.png`
- [ ] `results/analysis/exp1_full/latency_distribution.png` (replace with scatter)

### Files to Create

- [ ] `results/visualizations/exp1_strategy_comparison_fixed.png`
- [ ] `results/visualizations/exp1_degradation_curve_fixed.png`
- [ ] `results/visualizations/exp1_strategy_fill_heatmap.png`
- [ ] `results/visualizations/exp1_relative_lift.png`
- [ ] `results/visualizations/exp1_latency_vs_tokens.png`
- [ ] `results/visualizations/exp2_pollution_robustness_fixed.png`
- [ ] `results/visualizations/pareto_quality_latency.png`
- [ ] `results/visualizations/summary_table.png`

### Testing

After implementation, verify:
1. All chart y-axes show data clearly (no "flat" appearances)
2. Key findings are visually obvious (naive collapse, RAG jump at 950k)
3. Colors are consistent across charts (same palette for strategies)
4. Charts have proper titles, labels, and legends
5. No chart shows "exact_match" data (always zero)

---

## Appendix: Color Palette Reference

Use these colors consistently across all charts:

| Strategy | Hex Code | Description |
|----------|----------|-------------|
| Structured | `#2E86AB` | Dark blue (best performer) |
| RAG | `#4EA8DE` | Medium blue |
| Advanced RAG | `#7FC8F8` | Light blue |
| Naive | `#E76F51` | Orange/Red (worst performer) |

This creates a visual language: **Blue = Engineered = Good**, **Orange = Naive = Bad**

---

*Document Version: 1.0*  
*Last Updated: January 2, 2026*  
*Author: Analysis Team*

