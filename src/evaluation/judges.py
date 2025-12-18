"""
LLM-as-judge evaluation using Gemini.
"""
import json
import time
from typing import Dict, Optional

from src.models.gemini_client import GeminiClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

class LLMJudge:
    """
    LLM-as-a-Judge evaluator using Gemini.
    """
    
    def __init__(self, client: Optional[GeminiClient] = None, model_name: str = "gemini-2.0-flash-lite-preview-02-05"):
        self.client = client or GeminiClient()
        self.model_name = model_name
        self.tokenizer = Tokenizer()
        
    def evaluate(self, question: str, prediction: str, reference: str) -> Dict:
        """
        Evaluate the prediction against the reference.
        
        Returns:
            Dict containing 'score' (0-10), 'reasoning', and 'label' (Correct/Incorrect)
        """
        prompt = f"""
        You are an impartial judge evaluating the correctness of an answer produced by an AI model.
        
        Question: {question}
        
        Reference Answer (Ground Truth): {reference}
        
        Model Prediction: {prediction}
        
        Task:
        1. Compare the Model Prediction to the Reference Answer.
        2. Determine if the Model Prediction contains the core information required by the Reference Answer.
        3. Ignore minor stylistic differences or extra context if it doesn't contradict the truth.
        4. If the prediction is vague, incorrect, or missing key facts, mark it down.
        
        Output strictly in valid JSON format with the following keys:
        - "score": An integer from 0 to 10 (0 = completely wrong, 10 = perfect match).
        - "label": "Correct" (score >= 7) or "Incorrect" (score < 7).
        - "reasoning": A brief explanation of your decision.
        """
        
        try:
            response = self.client.generate_content(
                prompt, 
                model=self.model_name,
                max_output_tokens=256,
                temperature=0.0
            )
            
            text = response['text'].strip()
            
            # Clean up potential markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            
            result = json.loads(text.strip())
            return result
            
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            # Fallback for failure
            return {
                "score": 0,
                "label": "Error",
                "reasoning": f"Evaluation failed: {str(e)}"
            }

# Simple mock judge for dry/synthetic runs to avoid API costs
class MockJudge:
    def evaluate(self, question: str, prediction: str, reference: str) -> Dict:
        return {
            "score": 5,
            "label": "Mock",
            "reasoning": "This is a mock evaluation."
        }