from typing import List, Dict
from src.utils.tokenizer import count_tokens, truncate_to_tokens

class NaiveContextAssembler:
    """Sequential document concatenation with no structure"""
    
    def assemble(self, documents: List[Dict], max_tokens: int) -> str:
        """
        Concatenate documents sequentially.
        
        Args:
            documents: List of dicts with 'content', 'title', 'url'
            max_tokens: Maximum tokens in output
        
        Returns:
            Assembled context string
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        # Simply concatenate with double newline separator
        context_parts = []
        total_tokens = 0
        
        for doc in documents:
            content = doc['content']
            tokens = count_tokens(content)
            
            if total_tokens + tokens <= max_tokens:
                context_parts.append(content)
                total_tokens += tokens
            else:
                # Truncate last document to fit
                remaining = max_tokens - total_tokens
                if remaining > 100:  # Only add if meaningful
                    truncated = truncate_to_tokens(content, remaining)
                    context_parts.append(truncated)
                break
        
        return "\n\n".join(context_parts)