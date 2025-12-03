"""List available models"""
import google.generativeai as genai
from src.config import api_config

genai.configure(api_key=api_config.google_api_key)
for m in genai.list_models():
    print(f"name: {m.name}")
    print(f"description: {m.description}")
    print(f"input_token_limit: {m.input_token_limit}")
    print("-" * 20)
