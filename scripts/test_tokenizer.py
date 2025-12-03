import tiktoken
import sys

# Add project root to the Python path, so this script can be run from anywhere
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Attempting to get tiktoken encoding 'cl100k_base'...")
    encoding = tiktoken.get_encoding("cl100k_base")
    print("Successfully retrieved encoding.")
    print(f"Encoding object: {encoding}")
    print("\nTEST PASSED: The tiktoken library appears to be working correctly.")
except Exception as e:
    print("\n--- TEST FAILED ---")
    print("Failed to get tiktoken encoding.")
    print("This is the likely cause of the startup crash.")
    print(f"Error details: {e}")
