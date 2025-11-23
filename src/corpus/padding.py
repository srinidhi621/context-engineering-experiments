from typing import List, Dict
import random
from src.corpus.loaders import load_gutenberg_books
from src.utils.tokenizer import count_tokens, truncate_to_tokens

class PaddingGenerator:
    """Generate irrelevant padding content to reach target fill %"""
    
    def __init__(self):
        # Pre-load some Gutenberg books for padding
        # Book IDs: 1342 (Pride & Prejudice), 84 (Frankenstein), 
        #           98 (A Tale of Two Cities), 1661 (Sherlock Holmes)
        self.padding_books = load_gutenberg_books([1342, 84, 98, 1661])
        self.padding_text = "\n\n".join([b['content'] for b in self.padding_books if b and 'content' in b])
    
    def generate_padding(self, target_tokens: int) -> str:
        """
        Generate padding text of target token count.
        
        Randomly samples from pre-loaded books to create padding.
        """
        if target_tokens <= 0:
            return ""
        
        total_tokens = count_tokens(self.padding_text)
        if total_tokens == 0:
            return ""

        # This is a simplified sampling method.
        if target_tokens >= total_tokens:
            # Need multiple copies to reach the target token count
            copies = (target_tokens // total_tokens) + 1
            result = (self.padding_text + "\n\n") * copies
        else:
            # Simplified approach: just use the start of the text.
            result = self.padding_text
        
        return truncate_to_tokens(result, target_tokens)
    
    def pad_to_fill_percentage(self, 
                               content: str, 
                               fill_pct: float,
                               max_context_tokens: int = 1_000_000) -> str:
        """
        Pad content to reach target fill percentage.
        
        Args:
            content: The actual relevant content
            fill_pct: Target fill percentage (0.0 to 1.0]
            max_context_tokens: Maximum context window size
        
        Returns:
            content + padding to reach fill_pct * max_context_tokens
        """
        if not (0 < fill_pct <= 1.0):
            raise ValueError("fill_pct must be between 0.0 and 1.0")

        target_total = int(max_context_tokens * fill_pct)
        content_tokens = count_tokens(content)
        
        if content_tokens >= target_total:
            # Content already exceeds target, truncate
            return truncate_to_tokens(content, target_total)
        
        padding_needed = target_total - content_tokens
        padding = self.generate_padding(padding_needed)
        
        # Interleave or append? For now, append.
        result = content + "\n\n" + padding
        
        # Final truncation to be safe
        return truncate_to_tokens(result, target_total)