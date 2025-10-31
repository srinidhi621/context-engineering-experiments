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

def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int = 0) -> list:
    """
    Split text into chunks by token count (not word/character count).
    
    Args:
        text: Text to chunk
        chunk_size: Number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    # Encode text to tokens
    tokens = encoding.encode(text)
    
    # If text fits in one chunk, return as-is
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk of tokens
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move to next chunk with overlap
        start += chunk_size - overlap
    
    return chunks

