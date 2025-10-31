#!/usr/bin/env python3
"""Monitor API costs and token usage in real-time"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cost_monitor import get_monitor, CostMonitor


def main():
    parser = argparse.ArgumentParser(
        description='Monitor API costs and token usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor_costs.py
    (Show current summary)
  
  python scripts/monitor_costs.py --json
    (Output as JSON)
  
  python scripts/monitor_costs.py --reset
    (Reset all monitoring data)
  
  python scripts/monitor_costs.py --by-day
    (Show daily breakdown)
        """
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--by-day',
        action='store_true',
        help='Show daily breakdown'
    )
    parser.add_argument(
        '--by-hour',
        action='store_true',
        help='Show hourly breakdown'
    )
    parser.add_argument(
        '--by-model',
        action='store_true',
        help='Show model breakdown'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset all monitoring data'
    )
    parser.add_argument(
        '--set-budget',
        type=float,
        help='Set total budget alert threshold (in USD)'
    )
    
    args = parser.parse_args()
    
    monitor = get_monitor()
    
    if args.reset:
        # Reset monitoring data
        state_file = Path('results/.cost_monitor_state.json')
        if state_file.exists():
            state_file.unlink()
            print("✅ Monitoring data reset")
        return
    
    if args.set_budget:
        monitor.ALERT_THRESHOLDS['total'] = args.set_budget
        print(f"✅ Budget threshold set to ${args.set_budget:.2f}")
        return
    
    if args.json:
        # Output JSON
        summary = monitor.get_summary()
        print(json.dumps(summary, indent=2, default=str))
        return
    
    # Print summary
    monitor.print_summary()
    
    # Show breakdowns if requested
    if args.by_model:
        print("\n📊 DETAILED BY-MODEL BREAKDOWN:")
        print("="*70)
        with open(monitor.state_file) as f:
            state = json.load(f)
        
        for model, data in state['by_model'].items():
            print(f"\n{model}:")
            print(f"  Calls: {data['calls']}")
            print(f"  Input tokens: {data['input_tokens']:,}")
            print(f"  Output tokens: {data['output_tokens']:,}")
            print(f"  Total tokens: {data['input_tokens'] + data['output_tokens']:,}")
            print(f"  Cost: ${data['cost']:.4f}")
    
    if args.by_day:
        print("\n📅 DETAILED DAILY BREAKDOWN:")
        print("="*70)
        with open(monitor.state_file) as f:
            state = json.load(f)
        
        for day, data in sorted(state['by_day'].items()):
            print(f"\n{day}:")
            print(f"  Calls: {data['calls']}")
            print(f"  Tokens: {data['tokens']:,}")
            print(f"  Cost: ${data['cost']:.4f}")
    
    if args.by_hour:
        print("\n⏰ DETAILED HOURLY BREAKDOWN (Last 24 hours):")
        print("="*70)
        with open(monitor.state_file) as f:
            state = json.load(f)
        
        hours = sorted(state['by_hour'].items())[-24:]
        for hour, data in hours:
            print(f"\n{hour}:")
            print(f"  Calls: {data['calls']}")
            print(f"  Tokens: {data['tokens']:,}")
            print(f"  Cost: ${data['cost']:.4f}")


if __name__ == '__main__':
    main()
