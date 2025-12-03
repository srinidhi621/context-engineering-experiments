
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gemini_client import GeminiClient

def test_embedding():
    print("\n" + "="*70)
    print("🧪 EMBEDDING API PROBE")
    print("="*70)
    
    try:
        client = GeminiClient()
        text = "This is a test sentence for embedding."
        print(f"Step 1: Requesting embedding for: '{text}'")
        
        result = client.embed_text(
            text=text,
            track_cost=True
        )
        
        print("Step 2: Validating response")
        if not result:
            print("❌ FAILED: No result returned")
            return False
            
        embedding = result
            
        print(f"  ✅ Embedding received")
        print(f"  ✅ Vector length: {len(embedding)}")
        print(f"  ✅ Sample (first 3): {embedding[:3]}")
        
        print("\n✅ EMBEDDING PROBE SUCCESSFUL")
        return True
        
    except Exception as e:
        print(f"\n❌ EMBEDDING PROBE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding()
    sys.exit(0 if success else 1)
