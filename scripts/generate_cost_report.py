#!/usr/bin/env python3
"""
Generate comprehensive cost report with experiment and session breakdowns.
This script provides the full report requested: API calls, tokens, sessions, charges.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cost_monitor import get_monitor


def generate_comprehensive_report(output_format='text'):
    """Generate complete cost monitoring report"""
    
    monitor = get_monitor()
    summary = monitor.get_summary()
    
    if output_format == 'json':
        print(json.dumps(summary, indent=2, default=str))
        return
    
    # Text format - comprehensive report
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE COST MONITORING REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # === LIFETIME TOTALS ===
    print("🌍 LIFETIME TOTALS (All Time)")
    print("-" * 80)
    print(f"  Total API Calls:       {summary['total']['calls']:,}")
    print(f"  Total Tokens:          {summary['total']['tokens']:,}")
    print(f"  Total Cost:            ${summary['total']['cost']:.4f}")
    print(f"  Average Cost/Call:     ${summary['total']['cost'] / max(summary['total']['calls'], 1):.4f}")
    print(f"  Average Tokens/Call:   {summary['total']['tokens'] // max(summary['total']['calls'], 1):,}")
    print()
    
    # === TODAY ===
    print("📅 TODAY'S ACTIVITY")
    print("-" * 80)
    print(f"  Calls Today:           {summary['today']['calls']:,}")
    print(f"  Tokens Today:          {summary['today']['tokens']:,}")
    print(f"  Cost Today:            ${summary['today']['cost']:.4f}")
    print()
    
    # === BY EXPERIMENT ===
    if summary.get('by_experiment'):
        print("🧪 BREAKDOWN BY EXPERIMENT")
        print("-" * 80)
        total_exp_cost = 0
        for exp_id, data in sorted(summary['by_experiment'].items()):
            print(f"\n  Experiment: {exp_id}")
            print(f"    Calls:         {data['calls']:,}")
            print(f"    Input Tokens:  {data['input_tokens']:,}")
            print(f"    Output Tokens: {data['output_tokens']:,}")
            print(f"    Total Tokens:  {data['input_tokens'] + data['output_tokens']:,}")
            print(f"    Cost:          ${data['cost']:.4f}")
            print(f"    Avg Cost/Call: ${data['cost'] / max(data['calls'], 1):.4f}")
            total_exp_cost += data['cost']
        print(f"\n  Total Experiment Cost: ${total_exp_cost:.4f}")
        print()
    else:
        print("🧪 BREAKDOWN BY EXPERIMENT")
        print("-" * 80)
        print("  No experiment-tagged calls yet.")
        print("  TIP: Pass experiment_id to cost_monitor.record_call() to track by experiment.")
        print()
    
    # === BY SESSION ===
    if summary.get('by_session'):
        print("📊 BREAKDOWN BY SESSION (Last 10)")
        print("-" * 80)
        sessions = list(summary['by_session'].items())[-10:]
        for session_id, data in sessions:
            exp_tag = f" [{data.get('experiment_id', 'N/A')}]" if data.get('experiment_id') else ""
            print(f"\n  Session: {session_id}{exp_tag}")
            print(f"    Calls:  {data['calls']:,}")
            print(f"    Tokens: {data['input_tokens'] + data['output_tokens']:,}")
            print(f"    Cost:   ${data['cost']:.4f}")
        print()
    else:
        print("📊 BREAKDOWN BY SESSION")
        print("-" * 80)
        print("  No session-tagged calls yet.")
        print("  TIP: Pass session_id to cost_monitor.record_call() to track by session.")
        print()
    
    # === BY MODEL ===
    if summary.get('by_model'):
        print("🤖 BREAKDOWN BY MODEL")
        print("-" * 80)
        for model, data in summary['by_model'].items():
            print(f"\n  Model: {model}")
            print(f"    Calls:         {data['calls']:,}")
            print(f"    Input Tokens:  {data['input_tokens']:,}")
            print(f"    Output Tokens: {data['output_tokens']:,}")
            print(f"    Total Tokens:  {data['input_tokens'] + data['output_tokens']:,}")
            print(f"    Cost:          ${data['cost']:.4f}")
        print()
    
    # === BUDGET STATUS ===
    print("💳 BUDGET STATUS")
    print("-" * 80)
    budget_threshold = monitor.ALERT_THRESHOLDS['total']
    spent = summary['total']['cost']
    remaining = summary['budget_remaining']
    usage_pct = (spent / budget_threshold) * 100 if budget_threshold > 0 else 0
    
    print(f"  Budget Threshold:      ${budget_threshold:.2f}")
    print(f"  Total Spent:           ${spent:.4f}")
    print(f"  Remaining:             ${remaining:.4f}")
    print(f"  Usage:                 {usage_pct:.1f}%")
    
    # Visual bar
    bar_length = 50
    filled = int(bar_length * min(usage_pct / 100, 1))
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"  Progress:              [{bar}]")
    
    if usage_pct > 90:
        print(f"  ⚠️  WARNING: Approaching budget limit!")
    elif usage_pct > 75:
        print(f"  ⚠️  CAUTION: Over 75% of budget used")
    else:
        print(f"  ✅ Budget healthy")
    print()
    
    # === ALERTS ===
    if summary['recent_alerts']:
        print("⚠️  RECENT ALERTS (Last 5)")
        print("-" * 80)
        for alert in summary['recent_alerts']:
            print(f"  • {alert['message']}")
        print()
    else:
        print("✅ NO ALERTS - All thresholds within limits")
        print()
    
    # === SUMMARY STATS ===
    if summary['total']['calls'] > 0:
        print("📈 SUMMARY STATISTICS")
        print("-" * 80)
        avg_input = summary['total']['tokens'] // max(summary['total']['calls'], 1)
        print(f"  Average Tokens/Call:   {avg_input:,}")
        print(f"  Average Cost/Call:     ${summary['total']['cost'] / summary['total']['calls']:.4f}")
        
        if summary.get('by_experiment'):
            exp_count = len(summary['by_experiment'])
            print(f"  Total Experiments:     {exp_count}")
            print(f"  Avg Cost/Experiment:   ${summary['total']['cost'] / exp_count:.4f}")
        
        if summary.get('by_session'):
            session_count = len(summary['by_session'])
            print(f"  Total Sessions:        {session_count}")
            print(f"  Avg Cost/Session:      ${summary['total']['cost'] / session_count:.4f}")
        print()
    
    print("="*80)
    print()
    
    # === RECOMMENDATIONS ===
    print("💡 RECOMMENDATIONS")
    print("-" * 80)
    
    if summary['total']['calls'] == 0:
        print("  • No API calls recorded yet. Make your first call to start tracking.")
    elif usage_pct > 90:
        print("  • URGENT: Budget nearly exhausted. Stop experiments or increase budget.")
    elif usage_pct > 75:
        print("  • WARNING: Budget 75%+ used. Monitor carefully before large experiments.")
    else:
        remaining_calls_estimate = int(remaining / (spent / max(summary['total']['calls'], 1)))
        print(f"  • You can make approximately {remaining_calls_estimate:,} more calls at current")
        print(f"    average cost before hitting budget limit.")
    
    if not summary.get('by_experiment'):
        print("  • Consider tagging API calls with experiment_id for better tracking.")
    
    if not summary.get('by_session'):
        print("  • Consider tagging API calls with session_id for better tracking.")
    
    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive cost monitoring report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        help='Save report to file'
    )
    
    args = parser.parse_args()
    
    if args.save:
        import sys
        original_stdout = sys.stdout
        with open(args.save, 'w') as f:
            sys.stdout = f
            generate_comprehensive_report(args.format)
        sys.stdout = original_stdout
        print(f"✅ Report saved to: {args.save}")
    else:
        generate_comprehensive_report(args.format)


if __name__ == '__main__':
    main()

