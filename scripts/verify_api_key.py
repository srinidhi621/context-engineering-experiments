#!/usr/bin/env python3
"""
Verify if a Google Gemini API key is active and working.
Gathers detailed information about API capabilities and rate limits.

Usage:
    python scripts/verify_api_key.py                    # Uses GOOGLE_API_KEY from .env
    python scripts/verify_api_key.py --key your_key    # Use specific key
    GOOGLE_API_KEY=your_key python scripts/verify_api_key.py
"""

import os
import sys
import argparse
import json
from datetime import datetime
import google.generativeai as genai


def verify_api_key(api_key: str, verbose: bool = False) -> dict:
    """
    Verify if an API key is active and gather capability information.
    
    Args:
        api_key: Google Gemini API key to test
        verbose: Show detailed testing output
        
    Returns:
        dict with comprehensive API information including capabilities and limits
    """
    result = {
        'is_active': False,
        'timestamp': datetime.now().isoformat(),
        'models': [],
        'primary_model': 'gemini-2.5-flash',
        'embedding_model': 'text-embedding-004',
        'test_response': '',
        'capabilities': [],
        'rate_limits': {
            'detected': False,
            'rpm': None,
            'tpm': None,
            'notes': ''
        },
        'api_key_type': 'unknown',
        'errors': []
    }
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Try to list available models
        if verbose:
            print("  📡 Fetching available models...")
        try:
            available_models = genai.list_models()
            model_names = [m.name.replace('models/', '') for m in available_models]
            result['models'] = model_names
            result['capabilities'].append("✓ Can list available models")
            
            if verbose:
                print(f"     Found {len(model_names)} models")
                for name in model_names[:5]:
                    print(f"     - {name}")
                if len(model_names) > 5:
                    print(f"     ... and {len(model_names) - 5} more")
        except Exception as e:
            result['errors'].append(f"Could not list models: {str(e)[:50]}")
        
        # Test with primary model
        if verbose:
            print(f"  🚀 Testing with {result['primary_model']}...")
        
        model = genai.GenerativeModel(result['primary_model'])
        result['capabilities'].append("✓ Can create GenerativeModel instance")
        
        # Make an elaborate test call
        test_prompt = """You are an AI assistant verifying API connectivity. 
Respond with EXACTLY these 3 lines:
1. ACTIVE
2. Ready for experiments
3. API Key verified

Nothing else."""
        
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            result['is_active'] = True
            result['test_response'] = response.text.strip()[:200]
            result['capabilities'].append("✓ Can generate content")
            
            # Check for rate limit headers in response
            if hasattr(response, 'headers'):
                headers = response.headers
                if 'x-ratelimit-limit-requests' in headers:
                    result['rate_limits']['rpm'] = headers['x-ratelimit-limit-requests']
                    result['rate_limits']['detected'] = True
                if 'x-ratelimit-limit-tokens' in headers:
                    result['rate_limits']['tpm'] = headers['x-ratelimit-limit-tokens']
                    result['rate_limits']['detected'] = True
            
            if verbose:
                print(f"     ✅ Response received: {result['test_response'][:50]}...")
        else:
            result['errors'].append("No response from API")
            return result
        
        # Test embedding model
        if verbose:
            print(f"  🔤 Testing embedding model {result['embedding_model']}...")
        try:
            embedding_response = genai.embed_content(
                model=result['embedding_model'],
                content="API key verified"
            )
            if embedding_response and 'embedding' in embedding_response:
                result['capabilities'].append("✓ Embedding model available")
                if verbose:
                    embedding_dim = len(embedding_response['embedding'])
                    print(f"     ✅ Embedding model works ({embedding_dim}D vectors)")
        except Exception as e:
            result['errors'].append(f"Embedding model not available: {str(e)[:50]}")
            if verbose:
                print(f"     ⚠️  Embedding model issue: {str(e)[:50]}")
        
        # Detect API key type based on available models
        if any('gemini-2.0' in m for m in result['models']):
            result['api_key_type'] = 'Standard (Gemini 2.0 available)'
        if any('gemini-1.5-pro' in m for m in result['models']):
            result['api_key_type'] = 'Premium (Pro model available)'
        
        # Test streaming capability
        if verbose:
            print("  ⚡ Testing streaming capability...")
        try:
            import base64
            # Create minimal test image (1x1 pixel PNG)
            test_image = base64.standard_b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
            )
            test_call = model.generate_content([
                "Describe this image in one word:",
                {"mime_type": "image/png", "data": test_image}
            ])
            result['capabilities'].append("✓ Vision/images supported")
            if verbose:
                print("     ✅ Vision capability available")
        except Exception as e:
            # Vision might not be available, that's ok
            if verbose:
                print(f"     ⚠️  Vision not available (OK): {str(e)[:40]}")
        
        # Get summary info
        if result['is_active']:
            result['rate_limits']['notes'] = (
                "Rate limits detected from API response headers" 
                if result['rate_limits']['detected'] 
                else "Contact Google for your organization's rate limits"
            )
        
        return result
            
    except Exception as e:
        error_str = str(e)
        result['is_active'] = False
        
        # Parse specific error types
        if 'API key' in error_str or 'authentication' in error_str.lower():
            result['errors'].append('❌ Authentication failed: Invalid or expired API key')
            result['api_key_type'] = 'Invalid'
        elif 'quota' in error_str.lower() or 'rate' in error_str.lower():
            result['errors'].append('⚠️  Rate limit or quota exceeded')
        elif 'permission' in error_str.lower():
            result['errors'].append('❌ Permission denied: API key lacks necessary permissions')
        elif 'not found' in error_str.lower():
            result['errors'].append('❌ Model not found: Check if model name is correct')
        else:
            result['errors'].append(f"❌ Error: {error_str[:100]}")
        
        return result


def print_result(result: dict, json_output: bool = False):
    """Pretty print the verification result."""
    if json_output:
        print(json.dumps(result, indent=2))
        return
    
    # Header
    print("\n" + "="*60)
    print("🔐 GOOGLE GEMINI API KEY VERIFICATION")
    print("="*60)
    
    # Status
    status = "✅ ACTIVE" if result['is_active'] else "❌ INACTIVE"
    print(f"\nStatus: {status}")
    print(f"Timestamp: {result['timestamp']}")
    
    if result['is_active']:
        # Test response
        print(f"\n📝 Test Response:")
        print(f"   {result['test_response']}")
        
        # API Key Type
        print(f"\n🔑 API Key Type: {result['api_key_type']}")
        
        # Available Models
        if result['models']:
            print(f"\n🤖 Available Models ({len(result['models'])}):")
            for model in result['models'][:10]:
                print(f"   • {model}")
            if len(result['models']) > 10:
                print(f"   ... and {len(result['models']) - 10} more")
        
        # Capabilities
        if result['capabilities']:
            print(f"\n⚡ Capabilities:")
            for cap in result['capabilities']:
                print(f"   {cap}")
        
        # Rate Limits
        print(f"\n📊 Rate Limits:")
        if result['rate_limits']['detected']:
            if result['rate_limits']['rpm']:
                print(f"   RPM: {result['rate_limits']['rpm']}")
            if result['rate_limits']['tpm']:
                print(f"   TPM: {result['rate_limits']['tpm']}")
        else:
            print(f"   ℹ️  {result['rate_limits']['notes']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"   • API key is ready for use")
        print(f"   • You can run experiments with {len(result['models'])} available models")
        if any('2.0' in m for m in result['models']):
            print(f"   • Gemini 2.0 models available for your experiments")
        if 'streaming' in [c.lower() for c in result['capabilities']]:
            print(f"   • Streaming responses supported")
    
    else:
        # Errors
        print(f"\n⚠️  Issues Found:")
        for error in result['errors']:
            print(f"   {error}")
        
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Verify API key is correct: GOOGLE_API_KEY=your_key")
        print(f"   2. Check key hasn't expired in Google Cloud Console")
        print(f"   3. Ensure Gemini API is enabled in your project")
        print(f"   4. Verify your IP/domain isn't blocked")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Verify Google Gemini API key and gather capability information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_api_key.py
    (Uses GOOGLE_API_KEY from .env)
  
  python scripts/verify_api_key.py --key AIzaSy...
    (Test specific key)
  
  python scripts/verify_api_key.py --verbose
    (Show detailed testing steps)
  
  python scripts/verify_api_key.py --json
    (Output as JSON for scripting)
        """
    )
    
    parser.add_argument(
        '--key',
        help='API key to verify (if not using .env)',
        default=None
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed testing steps'
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.key or os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ Error: No API key provided")
        print("\nSet GOOGLE_API_KEY in one of these ways:")
        print("  1. In .env file: GOOGLE_API_KEY=your_key")
        print("  2. As command argument: --key your_key")
        print("  3. As environment variable: export GOOGLE_API_KEY=your_key")
        sys.exit(1)
    
    # Verify the key
    if not args.json:
        print("🔍 Verifying API key and gathering information...\n")
    
    result = verify_api_key(api_key, verbose=args.verbose)
    
    print_result(result, json_output=args.json)
    
    # Exit with appropriate code
    sys.exit(0 if result['is_active'] else 1)


if __name__ == '__main__':
    main()
