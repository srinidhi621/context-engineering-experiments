"""Token counting utilities"""
import tiktoken

# Use GPT-4 tokenizer as approximation for Gemini
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """
    Count tokens in text
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens
    """
    return len(encoding.encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to max tokens
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

