"""
API Monitor - Comprehensive monitoring for API calls, rate limits, and costs.

Tracks all API calls with full persistence across runs:
- Every call tracked (tokens in/out, costs, latency)
- Per-run tracking (experiment_id, session_id)
- Per-day aggregation
- Budget enforcement
- Rate limit enforcement (RPM, TPM, RPD)

All data persisted to disk after EVERY call.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import threading

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelLimits:
    """Rate limits for a specific model"""
    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute  
    rpd: int  # Requests per day
    input_cost_per_1m: float  # Cost per 1M input tokens
    output_cost_per_1m: float  # Cost per 1M output tokens
    
    @classmethod
    def get_for_model(cls, model_name: str) -> 'ModelLimits':
        """Get limits for a specific model"""
        LIMITS = {
            'gemini-flash-latest': cls(
                rpm=15,
                tpm=1_000_000,
                rpd=1_500,
                input_cost_per_1m=0.00,  # Free tier
                output_cost_per_1m=0.00   # Free tier
            ),
            'gemini-2.0-flash-exp': cls(
                rpm=10,
                tpm=4_000_000,
                rpd=1_500,
                input_cost_per_1m=0.00,  # Free tier
                output_cost_per_1m=0.00   # Free tier
            ),
            'models/gemini-2.0-flash': cls(
                rpm=10,
                tpm=4_000_000,
                rpd=1_500,
                input_cost_per_1m=0.00,  # Free tier (Preview)
                output_cost_per_1m=0.00   # Free tier (Preview)
            ),
             'gemini-2.0-flash-lite-preview-02-05': cls(
                rpm=30,
                tpm=4_000_000,
                rpd=1_500,
                input_cost_per_1m=0.00,  # Free tier
                output_cost_per_1m=0.00   # Free tier
            ),
             'gemini-2.0-pro-exp-02-05': cls(
                rpm=2,
                tpm=32_000,
                rpd=50,
                input_cost_per_1m=0.00,  # Free tier
                output_cost_per_1m=0.00   # Free tier
            ),
            'gemini-2.5-flash': cls(
                rpm=10,
                tpm=250_000,
                rpd=250,
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.30
            ),
            'gemini-2.5-pro': cls(
                rpm=5,
                tpm=250_000,
                rpd=100,
                input_cost_per_1m=1.25,
                output_cost_per_1m=5.00
            ),
            'text-embedding-004': cls(
                rpm=1500,
                tpm=10_000_000,
                rpd=1_000,  # CRITICAL: Reduced to 1,000 to safely stay in Free Tier. Exceeding triggers Paid Tier.
                input_cost_per_1m=0.00,
                output_cost_per_1m=0.00
            )
        }
        
        for key, limits in LIMITS.items():
            if key in model_name.lower():
                return limits
        
        # Default to most restrictive
        logger.warning(f"Unknown model {model_name}, using conservative limits")
        return LIMITS['gemini-2.5-pro']


class APIMonitor:
    """
    Unified monitoring system that combines:
    - Rate limiting (RPM, TPM, RPD enforcement)
    - Cost tracking (tokens, charges, experiments, sessions)
    - Budget management (alerts and thresholds)
    
    This is the single source of truth for all API usage.
    """
    
    def __init__(
        self,
        model_name: str,
        state_file: str = "results/.monitor_state.json",
        budget_limit: float = 174.00
    ):
        """
        Initialize API monitor.
        
        Args:
            model_name: Primary model being used
            state_file: Path to persist state
            budget_limit: Maximum budget in USD (default: $174 from plan)
        """
        self.model_name = model_name
        self.limits = ModelLimits.get_for_model(model_name)
        self.state_file = Path(state_file)
        self.budget_limit = budget_limit
        self.lock = threading.RLock()  # Use RLock for reentrant locking
        
        # Load or initialize state
        self.state = self._load_state()
        
        logger.info(f"API monitor initialized for {model_name}")
        logger.info(f"Limits: RPM={self.limits.rpm}, TPM={self.limits.tpm:,}, RPD={self.limits.rpd}")
        logger.info(f"Budget: ${budget_limit:.2f}")
    
    def _load_state(self) -> Dict:
        """Load persisted state or create new"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    logger.info(f"Loaded state: {state['totals']['total_requests']} total requests")
                    return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, starting fresh")
        
        return {
            'totals': {
                'total_requests': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_cost': 0.0,
                'created_at': datetime.now().isoformat(),
            },
            'current_minute': {
                'requests': 0,
                'tokens': 0,
                'reset_time': time.time()
            },
            'current_day': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            },
            'by_model': {},
            'by_experiment': {},
            'by_session': {},
            'by_day': {},
            'calls': [],
            'alerts': []
        }
    
    def _save_state(self):
        """Persist state to disk"""
        with self.lock:
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
    
    def _reset_minute_if_needed(self):
        """Reset per-minute counters if 60 seconds elapsed"""
        now = time.time()
        if now - self.state['current_minute']['reset_time'] >= 60:
            self.state['current_minute']['requests'] = 0
            self.state['current_minute']['tokens'] = 0
            self.state['current_minute']['reset_time'] = now
    
    def _reset_day_if_needed(self):
        """Reset per-day counters if new day"""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.state['current_day']['date']:
            logger.info(f"Day rollover: {self.state['current_day']['requests']} requests yesterday")
            self.state['current_day'] = {
                'date': today,
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }
    
    def can_make_request(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 500
    ) -> Tuple[bool, str]:
        """
        Check if request can be made without exceeding limits.
        
        Returns:
            (can_proceed, reason) tuple
        """
        self._reset_minute_if_needed()
        self._reset_day_if_needed()
        
        estimated_total = estimated_input_tokens + estimated_output_tokens
        
        # Check RPM
        if self.state['current_minute']['requests'] >= self.limits.rpm:
            wait_time = 60 - (time.time() - self.state['current_minute']['reset_time'])
            return False, f"RPM limit ({self.limits.rpm}). Wait {wait_time:.0f}s"
        
        # Check TPM
        if self.state['current_minute']['tokens'] + estimated_total > self.limits.tpm:
            wait_time = 60 - (time.time() - self.state['current_minute']['reset_time'])
            return False, f"TPM limit ({self.limits.tpm:,}). Wait {wait_time:.0f}s"
        
        # Check RPD
        if self.state['current_day']['requests'] >= self.limits.rpd:
            return False, f"RPD limit ({self.limits.rpd}) - daily quota exhausted"
        
        # Check budget
        estimated_cost = (
            (estimated_input_tokens / 1_000_000) * self.limits.input_cost_per_1m +
            (estimated_output_tokens / 1_000_000) * self.limits.output_cost_per_1m
        )
        if self.state['totals']['total_cost'] + estimated_cost > self.budget_limit:
            return False, f"Budget limit (${self.budget_limit:.2f}) would be exceeded"
        
        return True, "OK"
    
    def wait_if_needed(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 500
    ) -> float:
        """
        Block until request can be made. Returns wait time in seconds.
        Raises RuntimeError if daily limit or budget exceeded.
        """
        start_time = time.time()
        
        while True:
            can_proceed, reason = self.can_make_request(
                estimated_input_tokens,
                estimated_output_tokens
            )
            
            if can_proceed:
                return time.time() - start_time
            
            # Handle hard stops (daily limit, budget)
            if "RPD limit" in reason or "Budget limit" in reason:
                logger.error(f"HARD STOP: {reason}")
                raise RuntimeError(f"Cannot proceed: {reason}")
            
            # Wait for minute-based limits
            logger.warning(f"Rate limit: {reason}")
            wait_time = 60 - (time.time() - self.state['current_minute']['reset_time']) + 1
            logger.info(f"Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
    
    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        experiment_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Record an API call with full tracking.
        
        Returns:
            Call record with all metadata
        """
        timestamp = datetime.now()
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        model_limits = ModelLimits.get_for_model(model)
        input_cost = (input_tokens / 1_000_000) * model_limits.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * model_limits.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        call_record = {
            'timestamp': timestamp.isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'latency': latency,
            'experiment_id': experiment_id,
            'session_id': session_id,
        }
        
        with self.lock:
            # Update totals
            self.state['totals']['total_requests'] += 1
            self.state['totals']['total_input_tokens'] += input_tokens
            self.state['totals']['total_output_tokens'] += output_tokens
            self.state['totals']['total_cost'] += total_cost
            
            # Update current minute
            self.state['current_minute']['requests'] += 1
            self.state['current_minute']['tokens'] += total_tokens
            
            # Update current day
            self.state['current_day']['requests'] += 1
            self.state['current_day']['tokens'] += total_tokens
            self.state['current_day']['cost'] += total_cost
            
            # Update by model
            if model not in self.state['by_model']:
                self.state['by_model'][model] = {
                    'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0
                }
            self.state['by_model'][model]['calls'] += 1
            self.state['by_model'][model]['input_tokens'] += input_tokens
            self.state['by_model'][model]['output_tokens'] += output_tokens
            self.state['by_model'][model]['cost'] += total_cost
            
            # Update by experiment
            if experiment_id:
                if experiment_id not in self.state['by_experiment']:
                    self.state['by_experiment'][experiment_id] = {
                        'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0
                    }
                self.state['by_experiment'][experiment_id]['calls'] += 1
                self.state['by_experiment'][experiment_id]['input_tokens'] += input_tokens
                self.state['by_experiment'][experiment_id]['output_tokens'] += output_tokens
                self.state['by_experiment'][experiment_id]['cost'] += total_cost
            
            # Update by session
            if session_id:
                if session_id not in self.state['by_session']:
                    self.state['by_session'][session_id] = {
                        'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 
                        'cost': 0.0, 'experiment_id': experiment_id
                    }
                self.state['by_session'][session_id]['calls'] += 1
                self.state['by_session'][session_id]['input_tokens'] += input_tokens
                self.state['by_session'][session_id]['output_tokens'] += output_tokens
                self.state['by_session'][session_id]['cost'] += total_cost
            
            # Update by day historical
            day_key = timestamp.strftime('%Y-%m-%d')
            if day_key not in self.state['by_day']:
                self.state['by_day'][day_key] = {'calls': 0, 'tokens': 0, 'cost': 0.0}
            self.state['by_day'][day_key]['calls'] += 1
            self.state['by_day'][day_key]['tokens'] += total_tokens
            self.state['by_day'][day_key]['cost'] += total_cost
            
            # Store call (keep last 1000 for analysis)
            self.state['calls'].append(call_record)
            if len(self.state['calls']) > 1000:
                self.state['calls'] = self.state['calls'][-1000:]
            
            # Check alerts
            self._check_alerts(total_cost)
            
            # ALWAYS save state to ensure persistence across runs
            self._save_state()
        
        return call_record
    
    def _check_alerts(self, call_cost: float):
        """Check for threshold violations and add alerts"""
        alerts = []
        
        # Budget alerts
        budget_used_pct = (self.state['totals']['total_cost'] / self.budget_limit) * 100
        if budget_used_pct > 90:
            alerts.append({
                'type': 'budget_critical',
                'message': f"🚨 CRITICAL: Budget at {budget_used_pct:.1f}% (${self.state['totals']['total_cost']:.2f}/${self.budget_limit:.2f})",
                'timestamp': datetime.now().isoformat()
            })
        elif budget_used_pct > 75:
            alerts.append({
                'type': 'budget_warning',
                'message': f"⚠️  WARNING: Budget at {budget_used_pct:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Daily quota alerts
        daily_used_pct = (self.state['current_day']['requests'] / self.limits.rpd) * 100
        if daily_used_pct > 90:
            alerts.append({
                'type': 'daily_quota_high',
                'message': f"⚠️  Daily quota at {daily_used_pct:.1f}% ({self.state['current_day']['requests']}/{self.limits.rpd})",
                'timestamp': datetime.now().isoformat()
            })
        
        # High cost per call
        if call_cost > 0.10:
            alerts.append({
                'type': 'high_cost_call',
                'message': f"⚠️  High-cost API call: ${call_cost:.4f}",
                'timestamp': datetime.now().isoformat()
            })
        
        self.state['alerts'].extend(alerts)
        
        # Keep only last 50 alerts
        if len(self.state['alerts']) > 50:
            self.state['alerts'] = self.state['alerts'][-50:]
    
    def get_status(self) -> Dict:
        """Get comprehensive current status"""
        self._reset_minute_if_needed()
        self._reset_day_if_needed()
        
        total_spent = self.state['totals']['total_cost']
        budget_remaining = self.budget_limit - total_spent
        budget_used_pct = (total_spent / self.budget_limit) * 100 if self.budget_limit > 0 else 0
        
        return {
            'model': self.model_name,
            'limits': {
                'rpm': self.limits.rpm,
                'tpm': self.limits.tpm,
                'rpd': self.limits.rpd
            },
            'current_usage': {
                'requests_this_minute': self.state['current_minute']['requests'],
                'requests_remaining_this_minute': self.limits.rpm - self.state['current_minute']['requests'],
                'tokens_this_minute': self.state['current_minute']['tokens'],
                'requests_today': self.state['current_day']['requests'],
                'requests_remaining_today': self.limits.rpd - self.state['current_day']['requests'],
                'tokens_today': self.state['current_day']['tokens'],
                'cost_today': self.state['current_day']['cost']
            },
            'totals': self.state['totals'],
            'budget': {
                'limit': self.budget_limit,
                'spent': total_spent,
                'remaining': budget_remaining,
                'used_pct': budget_used_pct
            },
            'by_experiment': self.state['by_experiment'],
            'by_session': self.state['by_session'],
            'by_model': self.state['by_model'],
            'recent_alerts': self.state['alerts'][-5:] if self.state['alerts'] else []
        }
    
    def print_status(self):
        """Print human-readable status"""
        status = self.get_status()
        
        print("\n" + "="*70)
        print(f"📊 UNIFIED API MONITOR - {self.model_name}")
        print("="*70)
        
        # Limits
        print("\n🔒 RATE LIMITS:")
        print(f"  RPM: {status['limits']['rpm']}")
        print(f"  TPM: {status['limits']['tpm']:,}")
        print(f"  RPD: {status['limits']['rpd']:,}")
        
        # Current usage
        print("\n📈 CURRENT USAGE:")
        print(f"  This Minute: {status['current_usage']['requests_this_minute']}/{status['limits']['rpm']} requests")
        print(f"  This Minute: {status['current_usage']['tokens_this_minute']:,} tokens")
        print(f"  Today: {status['current_usage']['requests_today']}/{status['limits']['rpd']} requests")
        print(f"  Today: {status['current_usage']['tokens_today']:,} tokens")
        print(f"  Today: ${status['current_usage']['cost_today']:.4f}")
        
        # Totals
        print("\n🌍 LIFETIME TOTALS:")
        print(f"  Requests: {status['totals']['total_requests']:,}")
        print(f"  Tokens: {status['totals']['total_input_tokens'] + status['totals']['total_output_tokens']:,}")
        print(f"    Input: {status['totals']['total_input_tokens']:,}")
        print(f"    Output: {status['totals']['total_output_tokens']:,}")
        print(f"  Cost: ${status['totals']['total_cost']:.4f}")
        
        # Budget
        print("\n💳 BUDGET STATUS:")
        print(f"  Limit: ${status['budget']['limit']:.2f}")
        print(f"  Spent: ${status['budget']['spent']:.4f}")
        print(f"  Remaining: ${status['budget']['remaining']:.4f}")
        print(f"  Used: {status['budget']['used_pct']:.1f}%")
        
        # Visual budget bar
        bar_length = 40
        filled = int(bar_length * min(status['budget']['used_pct'] / 100, 1))
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}]")
        
        # Experiments
        if status['by_experiment']:
            print("\n🧪 BY EXPERIMENT:")
            for exp_id, data in sorted(status['by_experiment'].items())[:5]:
                print(f"  {exp_id}: {data['calls']} calls, ${data['cost']:.4f}")
        
        # Alerts
        if status['recent_alerts']:
            print("\n⚠️  RECENT ALERTS:")
            for alert in status['recent_alerts']:
                print(f"  • {alert['message']}")
        else:
            print("\n✅ No alerts - all within limits")
        
        print("\n" + "="*70 + "\n")


# Global instance registry
_monitors = {}


def get_monitor(model_name: Optional[str] = None, budget_limit: float = 174.00) -> APIMonitor:
    """
    Get or create API monitor instance for a specific model.
    Maintains separate instances (and state files) per model to track independent quotas.
    """
    global _monitors
    
    if model_name is None:
        from ..config import config
        model_name = config.model_name
        
    if model_name not in _monitors:
        # Create unique state file for this model to ensure independent tracking
        safe_name = model_name.replace("-", "_").replace(".", "_")
        state_file = f"results/.monitor_state_{safe_name}.json"
        
        _monitors[model_name] = APIMonitor(
            model_name=model_name,
            state_file=state_file,
            budget_limit=budget_limit
        )
        
    return _monitors[model_name]

