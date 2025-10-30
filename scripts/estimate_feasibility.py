#!/usr/bin/env python3
"""
Estimate experiment feasibility with free tier rate limits
Run this BEFORE starting experiments to understand constraints
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rate_limiter import RateLimiter
from src.config import config

def main():
    print("\n" + "="*70)
    print("GEMINI FREE TIER FEASIBILITY ANALYSIS")
    print("="*70)
    
    # Model to use
    model_name = "gemini-2.0-flash-exp"  # Best free tier limits
    limiter = RateLimiter(model_name)
    
    # Experiment configurations
    experiments = {
        'Experiment 1 (Needle in Haystacks)': {
            'questions': 50,
            'conditions': 4,
            'fill_levels': 5,
            'repetitions': 3,
            'avg_tokens_per_request': 500_000  # Average 500k context
        },
        'Experiment 2 (Context Pollution)': {
            'questions': 20,
            'conditions': 4,
            'pollution_levels': 5,
            'repetitions': 3,
            'avg_tokens_per_request': 400_000  # Smaller contexts
        },
        'Experiment 3 (Multi-Turn Memory)': {
            'scenarios': 10,
            'turns': 5,
            'conditions': 4,
            'repetitions': 3,
            'avg_tokens_per_request': 300_000  # Growing context per turn
        },
        'Experiment 4 (Precision Retrieval)': {
            'questions': 60,
            'conditions': 4,
            'fill_levels': 5,
            'repetitions': 3,
            'avg_tokens_per_request': 500_000
        },
        'Calibration': {
            'tests': 180,  # 6 fills × 3 positions × 10 trials
            'avg_tokens_per_request': 200_000  # Smaller contexts
        }
    }
    
    # Calculate totals
    total_requests = 0
    total_days_needed = 0
    
    print("\nEXPERIMENT BREAKDOWN:\n")
    
    for exp_name, config_dict in experiments.items():
        # Calculate number of requests
        if exp_name == 'Calibration':
            num_requests = config_dict['tests']
        elif 'scenarios' in config_dict:
            num_requests = (config_dict['scenarios'] * config_dict['turns'] * 
                          config_dict['conditions'] * config_dict['repetitions'])
        else:
            num_requests = (config_dict['questions'] * config_dict['conditions'] * 
                          config_dict.get('fill_levels', config_dict.get('pollution_levels', 1)) * 
                          config_dict['repetitions'])
        
        avg_tokens = config_dict['avg_tokens_per_request']
        
        # Estimate time
        estimate = limiter.estimate_time_for_requests(num_requests, avg_tokens)
        
        print(f"{exp_name}:")
        print(f"  Total Requests: {num_requests:,}")
        print(f"  Avg Tokens/Request: {avg_tokens:,}")
        print(f"  Estimated Time: {estimate['estimated_days']:.2f} days")
        print(f"  Bottleneck: {estimate['bottleneck']}")
        print()
        
        total_requests += num_requests
        total_days_needed += estimate['estimated_days']
    
    print("="*70)
    print(f"TOTAL ACROSS ALL EXPERIMENTS:")
    print(f"  Total Requests: {total_requests:,}")
    print(f"  Estimated Total Time: {total_days_needed:.1f} days")
    print("="*70)
    
    # Reality check
    print("\n⚠️  REALITY CHECK:\n")
    
    if total_days_needed < 7:
        print("✓ Feasible with free tier in ~1 week")
    elif total_days_needed < 30:
        print("⚠️  Will take multiple weeks with free tier")
        print("   Recommendation: Spread experiments over several weeks")
    else:
        print("❌ NOT FEASIBLE with free tier alone")
        print("   Time needed:", f"{total_days_needed:.0f} days ({total_days_needed/30:.1f} months)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:\n")
    
    print("Option 1: REDUCE SCOPE (Stay on free tier)")
    print("  - Reduce repetitions from 3 to 2 (saves 33%)")
    print("  - Reduce fill levels from 5 to 3 (saves 40%)")
    print("  - Focus on 2-3 key experiments initially")
    reduced_requests = int(total_requests * 0.4)  # 60% reduction
    reduced_estimate = limiter.estimate_time_for_requests(
        reduced_requests,
        avg_tokens_per_request=400_000
    )
    print(f"  → Reduced to ~{reduced_requests:,} requests = {reduced_estimate['estimated_days']:.1f} days")
    
    print("\nOption 2: USE MULTIPLE API KEYS")
    print("  - Create 2-3 Google accounts")
    print("  - Get separate API keys (free tier per project)")
    print("  - Run experiments in parallel")
    parallel_days = total_days_needed / 3
    print(f"  → With 3 keys: ~{parallel_days:.1f} days")
    
    print("\nOption 3: UPGRADE TO PAID TIER")
    print("  - Gemini 2.0 Flash: $0 per million tokens (input & output)")
    print("  - Wait, it's FREE even on paid tier!")
    print("  - But rate limits increase significantly:")
    print("    • Free: 15 RPM, 1M TPM, 1,500 RPD")
    print("    • Paid Tier 1: 1,000 RPM, 4M TPM, unlimited RPD")
    paid_estimate = total_requests / 1000  # Minutes at 1000 RPM
    print(f"  → With paid tier: ~{paid_estimate/60:.1f} hours (vs {total_days_needed:.1f} days)")
    
    print("\nOption 4: HYBRID APPROACH (Recommended)")
    print("  - Use free tier for development/testing")
    print("  - Run 10% of experiments to validate methodology")
    print("  - Then upgrade to paid tier for full suite")
    print("  - Paid tier is still FREE for Gemini 2.0 Flash!")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Decide on scope (full vs reduced)")
    print("2. If staying on free tier:")
    print("   - Expect {:.0f} days minimum".format(total_days_needed))
    print("   - Plan to run experiments in batches")
    print("   - Monitor daily with: python scripts/check_rate_limits.py")
    print("3. If upgrading to paid tier:")
    print("   - Set up billing in Google AI Studio")
    print("   - Still FREE for Gemini 2.0 Flash")
    print("   - Just need credit card on file for rate limit increases")
    print("\nRun experiments with: python scripts/run_experiment.py --experiment exp1")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
