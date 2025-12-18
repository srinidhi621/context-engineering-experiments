"""
Comprehensive rate limiter and quota tracker for Gemini API
Handles RPM, TPM, and RPD limits for free tier
"""
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from .logging import get_logger

logger = get_logger(__name__)

@dataclass
class RateLimits:
    """Rate limits for a specific model"""
    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    rpd: int  # Requests per day
    
    # Gemini 2.0 Flash Experimental (best for experiments)
    GEMINI_2_0_FLASH_EXP = {
        'rpm': 15,
        'tpm': 1_000_000,
        'rpd': 1_500
    }
    
    # Gemini 2.5 Flash
    GEMINI_2_5_FLASH = {
        'rpm': 10,
        'tpm': 250_000,
        'rpd': 250
    }
    
    # Gemini 2.5 Pro
    GEMINI_2_5_PRO = {
        'rpm': 5,
        'tpm': 250_000,
        'rpd': 100
    }
    
    @classmethod
    def get_limits(cls, model_name: str) -> 'RateLimits':
        """Get rate limits for a model"""
        if 'gemini-2.0-flash' in model_name.lower():
            limits = cls.GEMINI_2_0_FLASH_EXP
        elif 'gemini-2.5-flash' in model_name.lower():
            limits = cls.GEMINI_2_5_FLASH
        elif 'gemini-2.5-pro' in model_name.lower():
            limits = cls.GEMINI_2_5_PRO
        else:
            # Default to most restrictive
            logger.warning(f"Unknown model {model_name}, using conservative limits")
            limits = cls.GEMINI_2_5_PRO
        
        return cls(**limits)

@dataclass
class UsageStats:
    """Track usage statistics"""
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    requests_today: int = 0
    tokens_today: int = 0
    total_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    last_reset_minute: float = field(default_factory=time.time)
    last_reset_day: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UsageStats':
        return cls(**data)

class RateLimiter:
    """
    Comprehensive rate limiter that tracks and enforces all limits
    
    Features:
    - Tracks RPM, TPM, and RPD
    - Automatic time-based resets
    - Pre-call limit checking
    - Persistent state (survives restarts)
    - Automatic backoff recommendations
    """
    
    def __init__(self, model_name: str, state_file: str = "results/.rate_limiter_state.json"):
        self.model_name = model_name
        self.limits = RateLimits.get_limits(model_name)
        self.state_file = Path(state_file)
        
        # Load or initialize state
        self.stats = self._load_state()
        
        # Track request timestamps for accurate per-minute tracking
        self.request_timestamps = deque(maxlen=self.limits.rpm * 2)
        self.token_usage_minute = deque(maxlen=self.limits.rpm * 2)
        
        logger.info(f"Rate limiter initialized for {model_name}")
        logger.info(f"Limits: RPM={self.limits.rpm}, TPM={self.limits.tpm:,}, RPD={self.limits.rpd}")
    
    def _load_state(self) -> UsageStats:
        """Load persisted state or create new"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    stats = UsageStats.from_dict(data)
                    logger.info(f"Loaded state: {stats.total_requests} total requests")
                    return stats
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, starting fresh")
        
        return UsageStats()
    
    def _save_state(self):
        """Persist state to disk"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _reset_minute_counters(self):
        """Reset per-minute counters"""
        now = time.time()
        if now - self.stats.last_reset_minute >= 60:
            self.stats.requests_this_minute = 0
            self.stats.tokens_this_minute = 0
            self.stats.last_reset_minute = now
            
            # Clean old timestamps
            cutoff = now - 60
            while self.request_timestamps and self.request_timestamps[0] < cutoff:
                self.request_timestamps.popleft()
            while self.token_usage_minute and self.token_usage_minute[0][0] < cutoff:
                self.token_usage_minute.popleft()
    
    def _reset_daily_counters(self):
        """Reset per-day counters"""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.stats.last_reset_day:
            logger.info(f"Day rollover: {self.stats.requests_today} requests yesterday")
            self.stats.requests_today = 0
            self.stats.tokens_today = 0
            self.stats.last_reset_day = today
    
    def can_make_request(self, estimated_input_tokens: int, estimated_output_tokens: int = 500) -> Tuple[bool, str]:
        """
        Check if a request can be made without exceeding limits
        
        Args:
            estimated_input_tokens: Estimated input tokens for this request
            estimated_output_tokens: Estimated output tokens (default 500)
            
        Returns:
            (can_proceed, reason) tuple
        """
        self._reset_minute_counters()
        self._reset_daily_counters()
        
        estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Check RPM limit
        if self.stats.requests_this_minute >= self.limits.rpm:
            wait_time = 60 - (time.time() - self.stats.last_reset_minute)
            return False, f"RPM limit reached ({self.limits.rpm}). Wait {wait_time:.0f}s"
        
        # Check TPM limit
        if self.stats.tokens_this_minute + estimated_total_tokens > self.limits.tpm:
            wait_time = 60 - (time.time() - self.stats.last_reset_minute)
            return False, f"TPM limit would be exceeded ({self.limits.tpm:,}). Wait {wait_time:.0f}s"
        
        # Check RPD limit
        if self.stats.requests_today >= self.limits.rpd:
            tomorrow = datetime.now() + timedelta(days=1)
            midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
            wait_seconds = (midnight - datetime.now()).total_seconds()
            wait_hours = wait_seconds / 3600
            return False, f"RPD limit reached ({self.limits.rpd}). Wait {wait_hours:.1f} hours until midnight PT"
        
        return True, "OK"
    
    def wait_if_needed(self, estimated_input_tokens: int, estimated_output_tokens: int = 500) -> float:
        """
        Block until request can be made, return wait time
        
        Returns:
            Seconds waited
        """
        start_time = time.time()
        
        while True:
            can_proceed, reason = self.can_make_request(estimated_input_tokens, estimated_output_tokens)
            
            if can_proceed:
                return time.time() - start_time
            
            # Log why we're waiting
            if "RPD limit" in reason:
                logger.error(f"Hit daily limit! {reason}")
                logger.error("Consider:")
                logger.error("1. Stopping for today and resuming tomorrow")
                logger.error("2. Using multiple API keys with different projects")
                logger.error("3. Upgrading to paid tier")
                raise RuntimeError(f"Daily rate limit exceeded: {reason}")
            
            logger.warning(f"Rate limit: {reason}")
            
            # Calculate smart wait time
            if "RPM" in reason:
                wait_time = 60 - (time.time() - self.stats.last_reset_minute) + 1
            elif "TPM" in reason:
                wait_time = 60 - (time.time() - self.stats.last_reset_minute) + 1
            else:
                wait_time = 10
            
            logger.info(f"Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
    
    def record_request(self, input_tokens: int, output_tokens: int):
        """
        Record a completed request
        
        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens generated
        """
        now = time.time()
        total_tokens = input_tokens + output_tokens
        
        # Update counters
        self.stats.requests_this_minute += 1
        self.stats.tokens_this_minute += total_tokens
        self.stats.requests_today += 1
        self.stats.tokens_today += total_tokens
        self.stats.total_requests += 1
        self.stats.total_tokens_input += input_tokens
        self.stats.total_tokens_output += output_tokens
        
        # Track timestamps
        self.request_timestamps.append(now)
        self.token_usage_minute.append((now, total_tokens))
        
        # Persist state periodically (every 10 requests)
        if self.stats.total_requests % 10 == 0:
            self._save_state()
        
        logger.debug(f"Recorded: {input_tokens:,} in + {output_tokens:,} out tokens")
    
    def get_status(self) -> Dict:
        """Get current rate limiter status"""
        self._reset_minute_counters()
        self._reset_daily_counters()
        
        return {
            'model': self.model_name,
            'limits': {
                'rpm': self.limits.rpm,
                'tpm': self.limits.tpm,
                'rpd': self.limits.rpd
            },
            'current_usage': {
                'requests_this_minute': self.stats.requests_this_minute,
                'requests_remaining_this_minute': self.limits.rpm - self.stats.requests_this_minute,
                'tokens_this_minute': self.stats.tokens_this_minute,
                'tokens_remaining_this_minute': self.limits.tpm - self.stats.tokens_this_minute,
                'requests_today': self.stats.requests_today,
                'requests_remaining_today': self.limits.rpd - self.stats.requests_today,
                'tokens_today': self.stats.tokens_today
            },
            'lifetime': {
                'total_requests': self.stats.total_requests,
                'total_tokens_input': self.stats.total_tokens_input,
                'total_tokens_output': self.stats.total_tokens_output,
                'total_tokens': self.stats.total_tokens_input + self.stats.total_tokens_output
            },
            'utilization': {
                'rpm_pct': (self.stats.requests_this_minute / self.limits.rpm) * 100,
                'tpm_pct': (self.stats.tokens_this_minute / self.limits.tpm) * 100,
                'rpd_pct': (self.stats.requests_today / self.limits.rpd) * 100
            }
        }
    
    def estimate_time_for_requests(self, num_requests: int, avg_tokens_per_request: int = 500_000) -> Dict:
        """
        Estimate how long it will take to complete N requests
        
        Args:
            num_requests: Number of requests to estimate
            avg_tokens_per_request: Average tokens per request
            
        Returns:
            Dict with time estimates
        """
        # Calculate based on most restrictive limit
        
        # Time based on RPM limit
        minutes_by_rpm = num_requests / self.limits.rpm
        
        # Time based on TPM limit
        total_tokens = num_requests * avg_tokens_per_request
        minutes_by_tpm = total_tokens / self.limits.tpm
        
        # Time based on RPD limit
        days_by_rpd = num_requests / self.limits.rpd
        minutes_by_rpd = days_by_rpd * 24 * 60
        
        # Use most restrictive
        minutes_needed = max(minutes_by_rpm, minutes_by_tpm, minutes_by_rpd)
        hours_needed = minutes_needed / 60
        days_needed = hours_needed / 24
        
        bottleneck = "RPM"
        if minutes_needed == minutes_by_tpm:
            bottleneck = "TPM"
        elif minutes_needed == minutes_by_rpd:
            bottleneck = "RPD"
        
        return {
            'num_requests': num_requests,
            'avg_tokens_per_request': avg_tokens_per_request,
            'estimated_minutes': minutes_needed,
            'estimated_hours': hours_needed,
            'estimated_days': days_needed,
            'bottleneck': bottleneck,
            'breakdown': {
                'by_rpm': minutes_by_rpm,
                'by_tpm': minutes_by_tpm,
                'by_rpd': minutes_by_rpd
            },
            'recommendation': self._get_recommendation(num_requests, days_needed)
        }
    
    def _get_recommendation(self, num_requests: int, days_needed: float) -> str:
        """Get recommendation based on estimates"""
        if days_needed < 1:
            return "Can complete in less than 1 day with current limits"
        elif days_needed < 7:
            return f"Will take ~{days_needed:.1f} days. Consider spreading over multiple days"
        elif days_needed < 30:
            return f"Will take ~{days_needed:.1f} days. Strongly recommend: (1) Use multiple API keys, (2) Reduce dataset size, or (3) Upgrade to paid tier"
        else:
            return f"Will take ~{days_needed:.1f} days - NOT FEASIBLE with free tier. Must use paid tier or drastically reduce scope"
    
    def print_status(self):
        """Print human-readable status"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print(f"RATE LIMITER STATUS - {self.model_name}")
        print("="*60)
        
        print("\nLIMITS:")
        print(f"  RPM: {status['limits']['rpm']}")
        print(f"  TPM: {status['limits']['tpm']:,}")
        print(f"  RPD: {status['limits']['rpd']}")
        
        print("\nCURRENT USAGE (THIS MINUTE):")
        print(f"  Requests: {status['current_usage']['requests_this_minute']}/{status['limits']['rpm']}")
        print(f"  Tokens: {status['current_usage']['tokens_this_minute']:,}/{status['limits']['tpm']:,}")
        
        print("\nCURRENT USAGE (TODAY):")
        print(f"  Requests: {status['current_usage']['requests_today']}/{status['limits']['rpd']}")
        print(f"  Tokens: {status['current_usage']['tokens_today']:,}")
        
        print("\nLIFETIME TOTALS:")
        print(f"  Total Requests: {status['lifetime']['total_requests']:,}")
        print(f"  Total Tokens: {status['lifetime']['total_tokens']:,}")
        print(f"    Input: {status['lifetime']['total_tokens_input']:,}")
        print(f"    Output: {status['lifetime']['total_tokens_output']:,}")
        
        print("\nUTILIZATION:")
        print(f"  RPM: {status['utilization']['rpm_pct']:.1f}%")
        print(f"  TPM: {status['utilization']['tpm_pct']:.1f}%")
        print(f"  RPD: {status['utilization']['rpd_pct']:.1f}%")
        
        print("="*60 + "\n")

# Example usage
if __name__ == "__main__":
    # Initialize rate limiter
    limiter = RateLimiter("gemini-2.0-flash-lite-preview-02-05")
    
    # Check status
    limiter.print_status()
    
    # Estimate time for experiments
    print("\nESTIMATING TIME FOR 9,000 REQUESTS:")
    estimate = limiter.estimate_time_for_requests(
        num_requests=9_000,
        avg_tokens_per_request=500_000  # 500k average context
    )
    print(f"Estimated time: {estimate['estimated_days']:.1f} days")
    print(f"Bottleneck: {estimate['bottleneck']}")
    print(f"Recommendation: {estimate['recommendation']}")
