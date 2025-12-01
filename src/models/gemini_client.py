"""Gemini API client wrapper with comprehensive monitoring"""

import google.generativeai as genai
from google.api_core import exceptions
from src.config import api_config, config
from src.utils.monitor import get_monitor
from src.utils.logging import get_logger
from typing import Optional, List, Dict, Any
import time
import re

logger = get_logger(__name__)


class GeminiClient:
    """Wrapper for Google Gemini API with rate limiting and cost tracking"""
    
    def __init__(self, budget_limit: float = 174.00):
        """
        Initialize Gemini client with API monitoring.
        
        Args:
            budget_limit: Maximum budget in USD (default: $174 from project plan)
        """
        if not api_config.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_config.google_api_key)
        self.generation_model = config.model_name
        self.embedding_model = config.embedding_model_name
        self.monitor = get_monitor(
            model_name=self.generation_model,
            budget_limit=budget_limit
        )
        self.embedding_monitor = get_monitor(
            model_name=self.embedding_model,
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
        retries: int = 3,
        initial_delay: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini model with comprehensive monitoring and retries.
        """
        model = model or self.generation_model
        
        estimated_input_tokens = int(len(prompt.split()) * 1.33)
        estimated_output_tokens = max_output_tokens or 2048
        
        self.monitor.wait_if_needed(
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens
        )
        
        delay = initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(retries):
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
                
                latency = time.time() - start_time
                
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'completion_token_count', 0)
                
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
                
                return {
                    'text': response.text,
                    'tokens_input': input_tokens,
                    'tokens_output': output_tokens,
                    'latency': latency,
                    'cost': call_record['total_cost'],
                    'model': model
                }
            except (exceptions.DeadlineExceeded, exceptions.ServiceUnavailable) as e:
                last_error = e
                logger.warning(f"API call failed with {type(e).__name__} on attempt {attempt + 1}/{retries}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except exceptions.ResourceExhausted as e:
                last_error = e
                match = re.search(r"retry_delay { seconds: (\d+) }", str(e))
                retry_after = int(match.group(1)) if match else delay * (2 ** attempt)
                logger.warning(f"API call failed with ResourceExhausted on attempt {attempt + 1}/{retries}. Retrying in {retry_after}s...")
                time.sleep(retry_after)
                delay *= 2 # Still exponentially backoff for next attempt if this doesn't fix it
            except Exception as e:
                last_error = e
                raise RuntimeError(f"Failed to generate content after non-retryable error: {str(e)}")
        
        raise RuntimeError(
            f"Failed to generate content after {retries} attempts. Last error: {last_error}"
        )
    
    def embed_text(
        self, 
        text: str,
        model: Optional[str] = None,
        track_cost: bool = True,
        retries: int = 3,
        initial_delay: int = 5
    ) -> List[float]:
        """
        Generate embeddings for text, with retries for transient errors.
        """
        model = model or self.embedding_model
        delay = initial_delay
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                start_time = time.time()
                response = genai.embed_content(
                    model=model,
                    content=text
                )
                latency = time.time() - start_time
                
                if track_cost:
                    approx_tokens = len(text) // 4
                    self.embedding_monitor.record_call(
                        model=model,
                        input_tokens=approx_tokens,
                        output_tokens=768,
                        latency=latency,
                        prompt=text[:200]
                    )
                
                return response['embedding']
            
            except Exception as e:
                last_error = e
                message = str(e)
                if "DeadlineExceeded" in message or "504" in message:
                    logger.warning(
                        "Embedding call failed with DeadlineExceeded on attempt %d/%d. Retrying in %ds...",
                        attempt + 1,
                        retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                elif "ResourceExhausted" in message or "429" in message:
                    logger.warning(
                        "Embedding call hit ResourceExhausted on attempt %d/%d. Sleeping %ds...",
                        attempt + 1,
                        retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(f"Failed to generate embedding: {message}")

        raise RuntimeError(f"Failed to generate embedding after {retries} attempts. Last error: {last_error}")
    
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
