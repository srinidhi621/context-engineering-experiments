"""
Probe API limits for different models.
"""
import time
import google.generativeai as genai
from src.config import api_config

def probe_limits():
    genai.configure(api_key=api_config.google_api_key)
    
    models = [
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-1.5-flash"
    ]
    
    # Approximate char count for tokens (1 token ~= 4 chars)
    # We use "a " repeated which is 2 chars per token roughly? 
    # Actually "word " is 5 chars = 1 token.
    # Let's use a standard repeat string.
    
    thresholds = [260_000, 510_000, 990_000, 1_100_000]
    
    print(f"{'Model':<25} | {'Tokens':<10} | {'Result':<10} | {'Details'}")
    print("-" * 70)
    
    for model_name in models:
        print(f"Testing {model_name}...")
        
        for tokens in thresholds:
            # Create dummy payload
            # "a " is 2 chars. 1 token is ~4 chars.
            # So we need tokens * 4 chars.
            # "a " * (tokens * 2) should be safe?
            # Let's use count_tokens to be sure.
            
            try:
                model = genai.GenerativeModel(model_name)
                
                # Calibrate payload size
                # Start with rough estimate
                dummy_text = "word " * tokens
                
                # Verify exact count (API call, fast)
                try:
                    count = model.count_tokens(dummy_text).total_tokens
                except Exception as e:
                    print(f"{model_name:<25} | {tokens:<10} | ⚠️ CountErr | {str(e)[:40]}")
                    continue
                
                # Adjust if way off (simple heuristic)
                if count < tokens * 0.9:
                    dummy_text += "word " * (tokens - count)
                    count = model.count_tokens(dummy_text).total_tokens
                
                display_tokens = f"{count:,}"
                
                # Attempt generation
                try:
                    response = model.generate_content(
                        dummy_text,
                        generation_config={"max_output_tokens": 1}
                    )
                    print(f"{model_name:<25} | {display_tokens:<10} | ✅ PASS    | Success")
                except Exception as e:
                    err = str(e)
                    if "429" in err:
                        reason = "429 Quota"
                    elif "400" in err:
                        reason = "400 Context"
                    else:
                        reason = "Error"
                    print(f"{model_name:<25} | {display_tokens:<10} | ❌ FAIL    | {reason}")
                    
                    # If we fail at a lower threshold, skip higher ones for this model
                    break
                    
                time.sleep(2) # Avoid RPM limits
                
            except Exception as e:
                print(f"{model_name:<25} | {tokens:<10} | ❌ ERROR   | {str(e)[:40]}")

if __name__ == "__main__":
    probe_limits()
