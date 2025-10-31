"""Gemini API client wrapper with rate limiting and error handling"""

import google.generativeai as genai
from src.config import api_config, config
from src.utils.rate_limiter import RateLimiter
from typing import Optional, List, Dict, Any
import time


class GeminiClient:
    """Wrapper for Google Gemini API with rate limiting and error handling"""
    
    def __init__(self):
        """Initialize Gemini client with API key and rate limiter"""
        if not api_config.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_config.google_api_key)
        self.rate_limiter = RateLimiter(
            rpm=api_config.rate_limit_rpm,
            tpm=api_config.rate_limit_tpm,
            rpd=api_config.rate_limit_rpd
        )
        self.generation_model = config.model_name
        self.embedding_model = config.embedding_model_name
    
    def generate_content(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini model with rate limiting.
        
        Args:
            prompt: Input prompt/question
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens in response
            **kwargs: Additional arguments to pass to generate_content
            
        Returns:
            dict with 'text', 'tokens_input', 'tokens_output', 'latency'
        """
        model = model or self.generation_model
        
        # Wait if rate limit approaching
        self.rate_limiter.wait_if_needed(prompt_tokens=len(prompt.split()))
        
        try:
            start_time = time.time()
            
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens or 2048,
                },
                **kwargs
            )
            
            end_time = time.time()
            
            # Update rate limiter with actual usage
            if hasattr(response, 'usage_metadata'):
                self.rate_limiter.record_request(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.completion_token_count
                )
            
            return {
                'text': response.text,
                'tokens_input': getattr(response.usage_metadata, 'prompt_token_count', 0),
                'tokens_output': getattr(response.usage_metadata, 'completion_token_count', 0),
                'latency': end_time - start_time,
                'model': model
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate content: {str(e)}")
    
    def embed_text(
        self, 
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embeddings for text using Gemini embedding model.
        
        Args:
            text: Text to embed
            model: Embedding model to use (defaults to configured model)
            
        Returns:
            List of floats representing the embedding vector
        """
        model = model or self.embedding_model
        
        try:
            response = genai.embed_content(
                model=model,
                content=text
            )
            return response['embedding']
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def batch_embed_text(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text, model))
        return embeddings

