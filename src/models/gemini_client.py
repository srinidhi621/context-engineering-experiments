"""Gemini API client wrapper with unified monitoring (rate limiting + cost tracking)"""

import google.generativeai as genai
from src.config import api_config, config
from src.utils.unified_monitor import get_unified_monitor
from typing import Optional, List, Dict, Any
import time


class GeminiClient:
    """Wrapper for Google Gemini API with unified rate limiting and cost tracking"""
    
    def __init__(self, budget_limit: float = 174.00):
        """
        Initialize Gemini client with unified monitoring.
        
        Args:
            budget_limit: Maximum budget in USD (default: $174 from project plan)
        """
        if not api_config.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_config.google_api_key)
        self.generation_model = config.model_name
        self.embedding_model = config.embedding_model_name
        self.monitor = get_unified_monitor(
            model_name=self.generation_model,
            budget_limit=budget_limit
        )
    
    def generate_content(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        experiment_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini model with unified monitoring.
        
        Args:
            prompt: Input prompt/question
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens in response
            experiment_id: Optional experiment ID for tracking
            session_id: Optional session ID for tracking
            **kwargs: Additional arguments to pass to generate_content
            
        Returns:
            dict with 'text', 'tokens_input', 'tokens_output', 'latency', 'cost'
        """
        model = model or self.generation_model
        
        # Estimate token count (rough approximation: 1 token ~ 0.75 words)
        estimated_input_tokens = int(len(prompt.split()) * 1.33)
        estimated_output_tokens = max_output_tokens or 2048
        
        # Wait if rate limit approaching (enforces free tier limits & budget)
        self.monitor.wait_if_needed(
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens
        )
        
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
            latency = end_time - start_time
            
            # Extract token counts
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'completion_token_count', 0)
            
            # Record in unified monitor (handles both rate limiting and cost tracking)
            call_record = self.monitor.record_call(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                prompt=prompt[:200] if len(prompt) > 200 else prompt,
                response=response.text[:200] if response.text and len(response.text) > 200 else response.text,
                experiment_id=experiment_id,
                session_id=session_id
            )
            
            result = {
                'text': response.text,
                'tokens_input': input_tokens,
                'tokens_output': output_tokens,
                'latency': latency,
                'cost': call_record['total_cost'],
                'model': model
            }
            
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
    
    def get_status(self) -> Dict:
        """Get comprehensive monitoring status"""
        return self.monitor.get_status()
    
    def print_status(self):
        """Print formatted monitoring status"""
        self.monitor.print_status()

