"""
Script to test tokenization discrepancy between tiktoken and Gemini API.
"""
import sys
import os
import google.generativeai as genai
import time

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.tokenizer import count_tokens, truncate_to_tokens
from src.config import api_config, config
from src.models.gemini_client import GeminiClient # Use the client for monitoring

def test_tokenizer_discrepancy():
    print("🧪 Testing Tokenizer Discrepancy...")
    
    # Initialize Gemini Client for API token counting
    # (Note: This client also does generation, but we only use count_tokens from it)
    client = GeminiClient()
    
    # Use the model set in config
    model_name = config.model_name
    
    # Create a long dummy text
    long_text = "The quick brown fox jumps over the lazy dog. " * 500000 # ~25M chars
    
    # Test target tokens
    target_tokens = 990_000
    
    # 1. Truncate locally using tiktoken
    print(f"1. Truncating locally to {target_tokens:,} tokens using tiktoken (cl100k_base)...")
    truncated_text = truncate_to_tokens(long_text, target_tokens)
    local_count = count_tokens(truncated_text)
    print(f"   Local tiktoken count: {local_count:,} tokens.")
    
    if local_count > target_tokens:
        print(f"   ❌ Local truncation failed: {local_count} > {target_tokens}")
        # This shouldn't happen if truncate_to_tokens is correct
    
    # 2. Ask Gemini API to count tokens for the locally truncated text
    print(f"2. Asking Gemini API ({model_name}) to count tokens for this text...")
    
    try:
        # We need a model instance for count_tokens API call
        gemini_model = genai.GenerativeModel(model_name)
        api_count_response = gemini_model.count_tokens(truncated_text)
        api_count = api_count_response.total_tokens
        
        print(f"   Gemini API count: {api_count:,} tokens.")
        
        if api_count > 1_000_000:
            print(f"   ❌ API count exceeds 1M context window: {api_count:,}")
        
        print(f"\n--- Discrepancy ---")
        print(f"Local ({local_count:,}) vs API ({api_count:,})")
        print(f"Difference: {api_count - local_count:,} tokens")
        print(f"Ratio (API/Local): {api_count / local_count:.4f}")

        # 3. Test sending this text via generate_content (expect it to pass with new limit)
        print("\n3. Testing API call with this truncated text (expect success)...")
        test_prompt = f"Summarize this text: {truncated_text}"
        
        # Use a very short max_output_tokens to reduce cost/time for probe
        response = client.generate_content(
            test_prompt,
            model=model_name,
            max_output_tokens=10,
            experiment_id="tokenizer_probe",
            retries=1 # Don't retry endlessly for this test
        )
        print(f"   ✅ API generate_content successful. Input tokens: {response['tokens_input']:,}")
        
        if response['tokens_input'] > 1_000_000:
            print(f"   ❌ API reported input tokens exceeded 1M: {response['tokens_input']:,}")
        
        print("\n✅ Tokenizer discrepancy test complete.")

    except Exception as e:
        print(f"   ❌ Error during API token counting or generation: {e}")

if __name__ == "__main__":
    test_tokenizer_discrepancy()
