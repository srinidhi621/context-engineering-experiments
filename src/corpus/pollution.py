"""
Pollution Injector for Experiment 2.
Injects irrelevant content (Project Gutenberg text) into base content at specified token levels.
"""

from typing import Optional
from src.utils.tokenizer import count_tokens, truncate_to_tokens

class PollutionInjector:
    """Injects pollution into text."""
    
    def inject_pollution(
        self, 
        base_content: str,
        pollution_content: str,
        target_pollution_tokens: int,
        strategy: str = 'append'
    ) -> str:
        """
        Inject pollution into base content.
        
        Args:
            base_content: The relevant content (e.g., documentation)
            pollution_content: The irrelevant content (e.g., Gutenberg text)
            target_pollution_tokens: How many tokens of pollution to add
            strategy: Injection strategy ('append' or 'interleave' - currently only 'append' implemented)
            
        Returns:
            Combined text containing base content + pollution.
        """
        if target_pollution_tokens <= 0:
            return base_content
            
        # Truncate pollution to target size
        pollution_segment = truncate_to_tokens(pollution_content, target_pollution_tokens)
        
        # Verify actual count (truncate is approximate or exact depending on implementation)
        # If truncate_to_tokens uses encoding.decode(tokens[:max]), it is exact.
        
        if strategy == 'append':
            return f"{base_content}\n\n{pollution_segment}"
        elif strategy == 'prepend':
             return f"{pollution_segment}\n\n{base_content}"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    def create_mixed_context(
        self,
        base_content: str,
        padding_corpus: list[dict],
        target_pollution_tokens: int
    ) -> str:
        """
        Helper to create mixed context from a list of padding docs.
        Concatenates padding docs until enough content is available, then injects.
        """
        # simple concatenation of padding docs
        full_pollution = "\n\n".join(doc['content'] for doc in padding_corpus)
        return self.inject_pollution(base_content, full_pollution, target_pollution_tokens)
