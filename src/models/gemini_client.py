"""Gemini API client wrapper with rate limiting and error handling"""

import google.generativeai as genai
from src.config import api_config, config
from src.utils.rate_limiter import RateLimiter
from src.utils.cost_monitor import get_monitor
from typing import Optional, List, Dict, Any
import time


class GeminiClient:
    """Wrapper for Google Gemini API with rate limiting, error handling, and cost tracking"""
    
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
        self.cost_monitor = get_monitor()
        self.generation_model = config.model_name
        self.embedding_model = config.embedding_model_name
    
    def generate_content(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        track_cost: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini model with rate limiting and cost tracking.
        
        Args:
            prompt: Input prompt/question
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens in response
            track_cost: Whether to track this call in cost monitor
            **kwargs: Additional arguments to pass to generate_content
            
        Returns:
            dict with 'text', 'tokens_input', 'tokens_output', 'latency', 'cost'
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
            
            # Extract token counts
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'completion_token_count', 0)
            
            # Track cost
            if track_cost:
                call_record = self.cost_monitor.record_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    prompt=prompt[:200],  # First 200 chars
                    response=response.text[:200] if response.text else None
                )
            
            # Update rate limiter with actual usage
            if hasattr(response, 'usage_metadata'):
                self.rate_limiter.record_request(
                    prompt_tokens=input_tokens,
                    output_tokens=output_tokens
                )
            
            result = {
                'text': response.text,
                'tokens_input': input_tokens,
                'tokens_output': output_tokens,
                'latency': end_time - start_time,
                'model': model
            }
            
            # Add cost if tracked
            if track_cost:
                result['cost'] = call_record['total_cost']
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate content: {str(e)}")
    
    def embed_text(
        self, 
        text: str,
        model: Optional[str] = None,
        track_cost: bool = True
    ) -> List[float]:
        """
        Generate embeddings for text using Gemini embedding model.
        
        Args:
            text: Text to embed
            model: Embedding model to use (defaults to configured model)
            track_cost: Whether to track this call in cost monitor
            
        Returns:
            List of floats representing the embedding vector
        """
        model = model or self.embedding_model
        
        try:
            response = genai.embed_content(
                model=model,
                content=text
            )
            
            # Track cost (embeddings are cheap, ~0.02 per 1M tokens input)
            if track_cost:
                # Approximate token count (1 token ~ 4 chars)
                approx_tokens = len(text) // 4
                self.cost_monitor.record_call(
                    model=model,
                    input_tokens=approx_tokens,
                    output_tokens=768,  # Output dimension size
                    prompt=text[:200]
                )
            
            return response['embedding']
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def batch_embed_text(
        self,
        texts: List[str],
        model: Optional[str] = None,
        track_cost: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            track_cost: Whether to track these calls
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text, model, track_cost))
        return embeddings
    
    def get_cost_summary(self) -> Dict:
        """Get current cost monitoring summary"""
        return self.cost_monitor.get_summary()
    
    def print_cost_summary(self):
        """Print formatted cost summary"""
        self.cost_monitor.print_summary()

