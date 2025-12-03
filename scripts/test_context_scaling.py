
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gemini_client import GeminiClient
from src.utils.tokenizer import count_tokens

def test_context_scaling():
    print("\n" + "="*70)
    print("📈 CONTEXT SCALING PROBE")
    print("="*70)
    
    client = GeminiClient()
    print(f"Target Model: {client.generation_model}")
    
    # Define target sizes
    # Probe high-context limits
    targets = [200_000, 600_000, 990_000]
    
    # Use a simple word that tokenizes predictably (usually 1 token)
    # "the " is often 1 token in many tokenizers, but let's use "word " and check.
    base_word = "word "
    
    for target in targets:
        print(f"\nTesting target size: {target:,} tokens...")
        
        # Create payload
        # We'll approximate first then trim
        # count_tokens uses tiktoken which is an approximation for Gemini but close enough for probing
        multiplier = int(target) # Assuming 1 token per "word " roughly, or we can adjust
        payload = base_word * multiplier
        
        current_count = count_tokens(payload)
        
        # Adjustment loop (simple)
        if current_count > target:
            payload = payload[:int(len(payload) * (target / current_count))]
        elif current_count < target:
            payload += base_word * (target - current_count)
            
        final_count = count_tokens(payload)
        print(f"  Generated payload: {final_count:,} tokens (approx)")
        
        prompt = f"This is a test of context scaling. Please reply with the word 'Verified'.\n\n{payload}\n\nTask: Reply with 'Verified'."
        
        try:
            print(f"  Sending request... (this may take time)")
            start = time.time()
            response = client.generate_content(
                prompt=prompt,
                max_output_tokens=10,
                experiment_id="scaling_probe",
                session_id=f"size_{target}"
            )
            elapsed = time.time() - start
            
            print(f"  ✅ Success!")
            print(f"  ✅ Latency: {elapsed:.2f}s")
            print(f"  ✅ Input Tokens (API): {response['tokens_input']:,}")
            print(f"  ✅ Response: {response['text'].strip()}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            # If we fail at a lower limit, ask user if we should continue
            # But for automation, we'll just log and continue to see if it's a fluke or hard limit
            
        # Sleep to respect limits and let buffers clear
        time.sleep(2)

if __name__ == "__main__":
    test_context_scaling()
