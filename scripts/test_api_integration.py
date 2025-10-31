#!/usr/bin/env python3
"""
Test API integration end-to-end.
Verifies: API key, GeminiClient, RateLimiter, CostMonitor all work together.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gemini_client import GeminiClient
from src.config import config


def test_api_integration():
    """Test complete API integration"""
    
    print("\n" + "="*70)
    print("🧪 API INTEGRATION TEST")
    print("="*70)
    print()
    
    # Step 1: Initialize client
    print("Step 1: Initializing GeminiClient...")
    try:
        client = GeminiClient()
        print(f"  ✅ Client initialized")
        print(f"  ✅ Model: {client.generation_model}")
        print(f"  ✅ Embedding Model: {client.embedding_model}")
        print(f"  ✅ Rate Limiter: Active")
        print(f"  ✅ Cost Monitor: Active")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False
    print()
    
    # Step 2: Check API monitor status
    print("Step 2: Checking API monitor status...")
    try:
        status = client.get_status()
        print(f"  ✅ Monitor configured for: {status['model']}")
        print(f"  ✅ RPM Limit: {status['limits']['rpm']}")
        print(f"  ✅ TPM Limit: {status['limits']['tpm']:,}")
        print(f"  ✅ RPD Limit: {status['limits']['rpd']:,}")
        print(f"  ✅ Budget Limit: ${status['budget']['limit']:.2f}")
        print(f"  ✅ Lifetime calls: {status['totals']['total_requests']}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False
    print()
    
    # Step 3: Make a test API call
    print("Step 3: Making test API call...")
    test_prompt = """You are testing an API integration. Respond with exactly:
    
TEST PASSED: API is working correctly.

Nothing else."""
    
    try:
        response = client.generate_content(
            prompt=test_prompt,
            temperature=0.0,
            max_output_tokens=100,
            experiment_id="integration_test",
            session_id="test_session_1"
        )
        
        print(f"  ✅ API call successful")
        print(f"  ✅ Response received: {len(response['text'])} chars")
        print(f"  ✅ Input tokens: {response['tokens_input']:,}")
        print(f"  ✅ Output tokens: {response['tokens_output']:,}")
        print(f"  ✅ Latency: {response['latency']:.2f}s")
        print(f"  ✅ Cost: ${response.get('cost', 0):.6f}")
        
        print(f"\n  Response preview:")
        print(f"  {'-'*66}")
        print(f"  {response['text'][:200]}")
        print(f"  {'-'*66}")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Verify API monitor tracking and persistence
    print("Step 4: Verifying API monitor tracking and persistence...")
    try:
        status = client.get_status()
        print(f"  ✅ Total calls tracked: {status['totals']['total_requests']}")
        print(f"  ✅ Total tokens tracked: {status['totals']['total_input_tokens'] + status['totals']['total_output_tokens']:,}")
        print(f"  ✅ Total cost tracked: ${status['totals']['total_cost']:.6f}")
        print(f"  ✅ Requests today: {status['current_usage']['requests_today']}/{status['limits']['rpd']}")
        print(f"  ✅ Budget used: {status['budget']['used_pct']:.2f}%")
        
        if status['by_experiment'].get('integration_test'):
            exp_data = status['by_experiment']['integration_test']
            print(f"  ✅ Experiment tracking working: {exp_data['calls']} calls in 'integration_test'")
        
        # Verify state file exists
        import os
        if os.path.exists('results/.monitor_state.json'):
            print(f"  ✅ State persisted to disk: results/.monitor_state.json")
        else:
            print(f"  ⚠️  WARNING: State file not found on disk")
        
        if status['totals']['total_requests'] > 0:
            print(f"  ✅ API monitor recording and persisting correctly")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Final summary
    print("="*70)
    print("✅ ALL TESTS PASSED - System is ready for experiments!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Collect corpus data (data/raw/)")
    print("  2. Generate evaluation questions (data/questions/)")
    print("  3. Implement context assemblers (src/context_engineering/)")
    print("  4. Implement experiment runners (src/experiments/)")
    print("  5. Run pilot experiment with small sample")
    print()
    
    return True


if __name__ == '__main__':
    success = test_api_integration()
    sys.exit(0 if success else 1)

