
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gemini_client import GeminiClient
from src.utils.tokenizer import count_tokens
from src.utils.monitor import get_monitor

def test_scientific_scaling():
    print("\n" + "="*70)
    print("🔬 SCIENTIFIC CONTEXT SCALING PROBE")
    print("="*70)
    
    # Initialize client
    client = GeminiClient()
    monitor = client.monitor
    
    print(f"Target Model: {client.generation_model}")
    print(f"TPM Limit: {monitor.limits.tpm:,}")
    
    # Targets for this run - carefully chosen to test boundaries
    # We test 300k first as requested to verify TPM impact
    targets = [300_000, 600_000, 990_000]
    
    # Use a more plausible repeating pattern to avoid "repetition penalty" hallucinations
    # This is still synthetic but looks more like sentence structure
    plausible_pattern = "The quick brown fox jumps over the lazy dog. " 
    pattern_tokens = count_tokens(plausible_pattern)
    
    for target in targets:
        print(f"\nTesting target size: {target:,} tokens...")
        
        # check if we have enough capacity in the current minute
        # The monitor tracks a rolling window.
        # We can inspect the monitor's current usage
        
        monitor._reset_minute_if_needed()
        tokens_used = monitor.state['current_minute']['tokens']
        remaining = monitor.limits.tpm - tokens_used
        
        print(f"  Current Minute Usage: {tokens_used:,} tokens")
        print(f"  Remaining Capacity: {remaining:,} tokens")
        
        if target > remaining:
            wait_time = 65 # Wait for the window to clear completely
            print(f"  ⚠️ Target exceeds remaining capacity. Waiting {wait_time}s for window to reset...")
            time.sleep(wait_time)
            
            # Re-check
            monitor._reset_minute_if_needed()
            tokens_used = monitor.state['current_minute']['tokens']
            print(f"  New Minute Usage: {tokens_used:,} tokens")
        
        # Generate Payload
        multiplier = int(target / pattern_tokens)
        payload = plausible_pattern * multiplier
        
        # Fine-tune
        current_count = count_tokens(payload)
        if current_count > target:
            # Trim char-wise approx
            payload = payload[:int(len(payload) * (target/current_count))]
        
        final_count = count_tokens(payload)
        print(f"  Generated payload: {final_count:,} tokens")
        
        prompt = f"This is a system test. Please ignore the repeating text below and reply with the word 'Confirmed'.\n\n{payload}\n\nTask: Reply with 'Confirmed'."
        
        try:
            print(f"  Sending request... ")
            start = time.time()
            response = client.generate_content(
                prompt=prompt,
                max_output_tokens=10,
                experiment_id="scientific_probe",
                session_id=f"size_{target}"
            )
            elapsed = time.time() - start
            
            print(f"  ✅ Success!")
            print(f"  ✅ Latency: {elapsed:.2f}s")
            print(f"  ✅ Input Tokens: {response['tokens_input']:,}")
            print(f"  ✅ Response: {response['text'].strip()}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            if "ResourceExhausted" in str(e) or "429" in str(e):
                print("  🛑 Hit Rate Limit / Quota")
                # If we hit a hard limit, stop.
                break
            
        # Mandatory cooldown between large requests to let the TPM window slide
        # If we just used 300k, and limit is 1M, we are fine, but if we used 600k next, total 900k.
        # It's safer to wait a bit.
        print("  Cooling down for 5s...")
        time.sleep(5)

if __name__ == "__main__":
    test_scientific_scaling()
