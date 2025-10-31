"""Cost monitoring and tracking for API calls and token usage"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import threading


class CostMonitor:
    """Monitor API costs, token usage, and provide alerts"""
    
    # Pricing for Gemini 2.5 Flash (as of Oct 2025)
    PRICING = {
        'gemini-2.5-flash': {
            'input': 0.075 / 1_000_000,      # $0.075 per 1M tokens
            'output': 0.3 / 1_000_000,       # $0.30 per 1M tokens
        },
        'text-embedding-004': {
            'input': 0.02 / 1_000_000,       # $0.02 per 1M tokens
            'output': 0.0,                   # Free output
        }
    }
    
    # Alert thresholds (in USD)
    ALERT_THRESHOLDS = {
        'per_call': 0.10,           # Alert if single call > $0.10
        'per_hour': 5.00,           # Alert if hourly total > $5.00
        'per_day': 50.00,           # Alert if daily total > $50.00
        'total': 150.00,            # Alert if total spend > $150.00 (approaching budget)
    }
    
    def __init__(self, state_file: str = 'results/.cost_monitor_state.json'):
        """
        Initialize cost monitor.
        
        Args:
            state_file: Path to persist monitoring state
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load monitoring state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'calls': [],
            'summary': {
                'total_calls': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_cost': 0.0,
                'last_reset': datetime.now().isoformat(),
            },
            'by_model': {},
            'by_hour': {},
            'by_day': {},
            'alerts': []
        }
    
    def _save_state(self):
        """Save monitoring state to disk"""
        with self.lock:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
    
    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prompt: Optional[str] = None,
        response: Optional[str] = None
    ) -> Dict:
        """
        Record an API call and calculate cost.
        
        Args:
            model: Model used (e.g., 'gemini-2.5-flash')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            prompt: Original prompt (for logging)
            response: Generated response (for logging)
            
        Returns:
            Dict with call details and cost breakdown
        """
        # Calculate cost
        pricing = self.PRICING.get(model, self.PRICING['gemini-2.5-flash'])
        input_cost = input_tokens * pricing['input']
        output_cost = output_tokens * pricing['output']
        total_cost = input_cost + output_cost
        
        # Create call record
        timestamp = datetime.now()
        call_record = {
            'timestamp': timestamp.isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'prompt_length': len(prompt.split()) if prompt else 0,
            'response_length': len(response.split()) if response else 0,
        }
        
        with self.lock:
            # Add to calls list
            self.state['calls'].append(call_record)
            
            # Update summary
            self.state['summary']['total_calls'] += 1
            self.state['summary']['total_input_tokens'] += input_tokens
            self.state['summary']['total_output_tokens'] += output_tokens
            self.state['summary']['total_cost'] += total_cost
            
            # Update by model
            if model not in self.state['by_model']:
                self.state['by_model'][model] = {
                    'calls': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cost': 0.0
                }
            self.state['by_model'][model]['calls'] += 1
            self.state['by_model'][model]['input_tokens'] += input_tokens
            self.state['by_model'][model]['output_tokens'] += output_tokens
            self.state['by_model'][model]['cost'] += total_cost
            
            # Update by hour
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            if hour_key not in self.state['by_hour']:
                self.state['by_hour'][hour_key] = {'calls': 0, 'cost': 0.0, 'tokens': 0}
            self.state['by_hour'][hour_key]['calls'] += 1
            self.state['by_hour'][hour_key]['cost'] += total_cost
            self.state['by_hour'][hour_key]['tokens'] += input_tokens + output_tokens
            
            # Update by day
            day_key = timestamp.strftime('%Y-%m-%d')
            if day_key not in self.state['by_day']:
                self.state['by_day'][day_key] = {'calls': 0, 'cost': 0.0, 'tokens': 0}
            self.state['by_day'][day_key]['calls'] += 1
            self.state['by_day'][day_key]['cost'] += total_cost
            self.state['by_day'][day_key]['tokens'] += input_tokens + output_tokens
            
            # Check for alerts
            self._check_alerts(total_cost, hour_key, day_key)
            
            self._save_state()
        
        return call_record
    
    def _check_alerts(self, call_cost: float, hour_key: str, day_key: str):
        """Check if any cost thresholds are exceeded"""
        alerts = []
        
        # Per-call alert
        if call_cost > self.ALERT_THRESHOLDS['per_call']:
            alerts.append({
                'type': 'per_call',
                'message': f"⚠️  High-cost API call: ${call_cost:.4f}",
                'threshold': self.ALERT_THRESHOLDS['per_call'],
                'actual': call_cost
            })
        
        # Per-hour alert
        hour_cost = self.state['by_hour'][hour_key]['cost']
        if hour_cost > self.ALERT_THRESHOLDS['per_hour']:
            alerts.append({
                'type': 'per_hour',
                'message': f"⚠️  Hourly cost exceeded: ${hour_cost:.2f} (threshold: ${self.ALERT_THRESHOLDS['per_hour']:.2f})",
                'threshold': self.ALERT_THRESHOLDS['per_hour'],
                'actual': hour_cost,
                'hour': hour_key
            })
        
        # Per-day alert
        day_cost = self.state['by_day'][day_key]['cost']
        if day_cost > self.ALERT_THRESHOLDS['per_day']:
            alerts.append({
                'type': 'per_day',
                'message': f"⚠️  Daily cost exceeded: ${day_cost:.2f} (threshold: ${self.ALERT_THRESHOLDS['per_day']:.2f})",
                'threshold': self.ALERT_THRESHOLDS['per_day'],
                'actual': day_cost,
                'day': day_key
            })
        
        # Total budget alert
        total_cost = self.state['summary']['total_cost']
        if total_cost > self.ALERT_THRESHOLDS['total']:
            alerts.append({
                'type': 'total_budget',
                'message': f"🚨 Total budget approaching limit: ${total_cost:.2f} / ${self.ALERT_THRESHOLDS['total']:.2f}",
                'threshold': self.ALERT_THRESHOLDS['total'],
                'actual': total_cost
            })
        
        self.state['alerts'].extend(alerts)
    
    def get_summary(self) -> Dict:
        """Get current monitoring summary"""
        with self.lock:
            today = datetime.now().strftime('%Y-%m-%d')
            today_data = self.state['by_day'].get(today, {'calls': 0, 'cost': 0.0, 'tokens': 0})
            
            return {
                'total': {
                    'calls': self.state['summary']['total_calls'],
                    'tokens': self.state['summary']['total_input_tokens'] + self.state['summary']['total_output_tokens'],
                    'cost': self.state['summary']['total_cost'],
                },
                'today': {
                    'calls': today_data['calls'],
                    'tokens': today_data['tokens'],
                    'cost': today_data['cost'],
                },
                'by_model': self.state['by_model'],
                'budget_remaining': self.ALERT_THRESHOLDS['total'] - self.state['summary']['total_cost'],
                'recent_alerts': self.state['alerts'][-5:] if self.state['alerts'] else []
            }
    
    def print_summary(self):
        """Print a formatted cost summary"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("💰 API COST MONITORING SUMMARY")
        print("="*70)
        
        # Total stats
        print(f"\n📊 TOTAL USAGE:")
        print(f"   Calls: {summary['total']['calls']}")
        print(f"   Tokens: {summary['total']['tokens']:,}")
        print(f"   Cost: ${summary['total']['cost']:.4f}")
        
        # Today stats
        print(f"\n📅 TODAY:")
        print(f"   Calls: {summary['today']['calls']}")
        print(f"   Tokens: {summary['today']['tokens']:,}")
        print(f"   Cost: ${summary['today']['cost']:.4f}")
        
        # By model
        if summary['by_model']:
            print(f"\n🤖 BY MODEL:")
            for model, data in summary['by_model'].items():
                print(f"   {model}:")
                print(f"      Calls: {data['calls']}")
                print(f"      Tokens: {data['input_tokens'] + data['output_tokens']:,}")
                print(f"      Cost: ${data['cost']:.4f}")
        
        # Budget
        print(f"\n💳 BUDGET:")
        print(f"   Threshold: ${self.ALERT_THRESHOLDS['total']:.2f}")
        print(f"   Spent: ${summary['total']['cost']:.4f}")
        print(f"   Remaining: ${summary['budget_remaining']:.4f}")
        print(f"   Usage: {(summary['total']['cost']/self.ALERT_THRESHOLDS['total']*100):.1f}%")
        
        # Recent alerts
        if summary['recent_alerts']:
            print(f"\n⚠️  RECENT ALERTS:")
            for alert in summary['recent_alerts']:
                print(f"   • {alert['message']}")
        else:
            print(f"\n✅ No alerts - all within thresholds")
        
        print("\n" + "="*70 + "\n")


# Global instance
_monitor = None


def get_monitor() -> CostMonitor:
    """Get or create global cost monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = CostMonitor()
    return _monitor
