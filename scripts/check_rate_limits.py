#!/usr/bin/env python3
"""
Check current rate limit status
Run this anytime to see how much quota you've used
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rate_limiter import RateLimiter
import json

def main():
    model_name = "gemini-2.0-flash-exp"
    limiter = RateLimiter(model_name)
    
    # Print detailed status
    limiter.print_status()
    
    # Check if we're approaching limits
    status = limiter.get_status()
    
    print("⚠️  WARNINGS:")
    warnings = []
    
    if status['utilization']['rpm_pct'] > 80:
        warnings.append(f"  • RPM usage at {status['utilization']['rpm_pct']:.0f}% - slow down!")
    
    if status['utilization']['tpm_pct'] > 80:
        warnings.append(f"  • TPM usage at {status['utilization']['tpm_pct']:.0f}% - reduce token usage!")
    
    if status['utilization']['rpd_pct'] > 80:
        warnings.append(f"  • RPD usage at {status['utilization']['rpd_pct']:.0f}% - approaching daily limit!")
    
    if status['utilization']['rpd_pct'] > 95:
        warnings.append("  • ❌ CRITICAL: Very close to daily limit! Stop experiments for today.")
    
    if warnings:
        print("\n".join(warnings))
    else:
        print("  ✓ All limits healthy\n")
    
    # Save status to JSON
    status_file = Path("results/.rate_limit_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"Status saved to: {status_file}")

if __name__ == "__main__":
    main()
