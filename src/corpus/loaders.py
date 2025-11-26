"""Corpus loading utilities for Hugging Face Hub and Project Gutenberg"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_models, list_datasets
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from gutenbergpy import textget
from dotenv import load_dotenv

# Import token counting from centralized utilities
from src.utils.tokenizer import count_tokens

# Load environment variables
load_dotenv()


def load_hf_model_card(
    model_id: str, 
    after_date: str = "2024-08-01"
) -> Optional[Dict]:
    """
    Load model card (README) from Hugging Face Hub.
    
    Args:
        model_id: Model ID (e.g., "meta-llama/Llama-3.2-3B", "openai/whisper-large-v3")
        after_date: ISO date string, only fetch if modified after this date
    
    Returns:
        Dict with 'content', 'url', 'last_modified', 'tokens', 'model_id'
        Returns None if model doesn't meet criteria or on error
    """
    try:
        api = HfApi()
        
        # Get model info
        model_info = api.model_info(model_id, files_metadata=False)
        
        # Check if modified after target date
        cutoff_date = datetime.fromisoformat(after_date).replace(tzinfo=None)
        last_modified = model_info.lastModified.replace(tzinfo=None)
        
        if last_modified < cutoff_date:
            return None
        
        # Download model card (README.md)
        try:
            readme_path = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                repo_type="model"
            )
            
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            # Some models may not have a README
            print(f"  No README for {model_id}: {e}")
            return None
        
        # Count tokens
        tokens = count_tokens(content)
        
        # Only include if meaningful content
        if tokens < 50:
            return None
        
        return {
            "content": content,
            "url": f"https://huggingface.co/{model_id}",
            "last_modified": model_info.lastModified.isoformat(),
            "tokens": tokens,
            "model_id": model_id,
            "type": "model_card",
            "tags": model_info.tags[:10] if model_info.tags else []
        }
    
    except RepositoryNotFoundError:
        print(f"  Model not found: {model_id}")
        return None
    except Exception as e:
        print(f"  Error loading {model_id}: {e}")
        return None


def load_hf_curated_models(
    after_date: str = "2024-08-01",
    max_tokens: int = 50000
) -> List[Dict]:
    """
    Load model cards from curated list of well-documented, popular models.
    
    These models are known to have comprehensive documentation and are
    regularly updated. Much faster than scanning all recent models.
    
    Args:
        after_date: ISO date string, only fetch if modified after
        max_tokens: Maximum total tokens to collect
    
    Returns:
        List of dicts with model card content and metadata
    """
    # Curated list of models released/updated Sept-Dec 2024 (post-training cutoff)
    CURATED_MODELS = [
        # Meta - Llama 3.2 (Released Sept 2024) & Llama 3.3 (Dec 2024)
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-Guard-3-1B",
        "meta-llama/Llama-Guard-3-8B",
        
        # Alibaba - Qwen 2.5 (Released Sept 2024)
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        
        # Mistral AI - New releases (Sept-Oct 2024)
        "mistralai/Mistral-Small-Instruct-2409",
        "mistralai/Pixtral-12B-2409",
        "mistralai/Ministral-8B-Instruct-2410",
        
        # Microsoft - Phi 3.5 (Aug-Sept 2024)
        "microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-MoE-instruct",
        "microsoft/Phi-3.5-vision-instruct",
        
        # Google - Gemma 2 (Updated Oct 2024)
        "google/gemma-2-2b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b",
        "google/gemma-2-27b-it",
        
        # OpenAI - Whisper v3 Turbo (Oct 2024)
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
        
        # Stability AI - SD 3.5 (Oct 2024)
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "stabilityai/stable-diffusion-3.5-medium",
        
        # NVIDIA - Updated models (Sept-Oct 2024)
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
        
        # HuggingFace - Recent releases
        "HuggingFaceTB/SmolLM-135M",
        "HuggingFaceTB/SmolLM-360M",
        "HuggingFaceTB/SmolLM-1.7B",
        "HuggingFaceTB/SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-1.7B",
        
        # Cohere - Command R7B (Sept 2024)
        "CohereForAI/c4ai-command-r7b-12-2024",
        
        # AI21 Labs - Jamba 1.5 (Aug-Sept 2024)
        "ai21labs/Jamba-v0.1",
        
        # Anthropic (regularly updated)
        "Anthropic/claude-tokenizer",
        
        # Alibaba - Qwen Math & Coder (Sept-Oct 2024)
        "Qwen/Qwen2.5-Math-1.5B",
        "Qwen/Qwen2.5-Math-7B",
        "Qwen/Qwen2.5-Math-72B",
    ]
    
    documents = []
    total_tokens = 0
    
    print(f"Loading curated model cards (after {after_date})...")
    print(f"Target: {max_tokens:,} tokens\n")
    
    for model_id in CURATED_MODELS:
        if total_tokens >= max_tokens:
            break
        
        # Load model card
        doc = load_hf_model_card(model_id, after_date)
        
        if doc and total_tokens + doc['tokens'] <= max_tokens:
            documents.append(doc)
            total_tokens += doc['tokens']
            print(f"  ✓ {model_id} ({doc['tokens']:,} tokens)")
        elif not doc:
            print(f"  ⊘ {model_id} (not modified after {after_date})")
    
    print(f"\n{'='*60}")
    print(f"Collected {len(documents)} model cards")
    print(f"Total tokens: {total_tokens:,}")
    print(f"{'='*60}")
    
    return documents


def load_hf_model_cards(
    after_date: str = "2024-08-01",
    max_tokens: int = 50000,
    tags: Optional[List[str]] = None,
    search: Optional[str] = None
) -> List[Dict]:
    """
    Load multiple model cards from Hugging Face Hub by scanning recent updates.
    
    Note: Many recent models don't have READMEs. For faster, more reliable
    collection, use load_hf_curated_models() instead.
    
    Args:
        after_date: ISO date string, only fetch if modified after
        max_tokens: Maximum total tokens to collect
        tags: Filter by tags (e.g., ["text-generation", "pytorch"])
        search: Search query (e.g., "llama", "whisper")
    
    Returns:
        List of dicts with model card content and metadata
    """
    documents = []
    total_tokens = 0
    
    try:
        # Get recently updated models
        models = list_models(
            sort="lastModified",
            direction=-1,
            limit=500,  # Check more models to find enough valid ones
            tags=tags,
            search=search
        )
        
        print(f"Scanning recent models (after {after_date})...")
        
        for model in models:
            if total_tokens >= max_tokens:
                break
            
            # Load model card
            doc = load_hf_model_card(model.id, after_date)
            
            if doc and total_tokens + doc['tokens'] <= max_tokens:
                documents.append(doc)
                total_tokens += doc['tokens']
                print(f"  ✓ Loaded {model.id} ({doc['tokens']} tokens)")
            
            # Be polite - don't hammer the API
            if len(documents) % 10 == 0 and len(documents) > 0:
                print(f"  Progress: {len(documents)} models, {total_tokens:,} tokens")
    
    except Exception as e:
        print(f"Error loading model cards: {e}")
    
    print(f"\nCollected {len(documents)} model cards, {total_tokens:,} tokens total")
    return documents


def load_hf_dataset_cards(
    after_date: str = "2024-08-01",
    max_tokens: int = 50000,
    tags: Optional[List[str]] = None
) -> List[Dict]:
    """
    Load dataset cards from Hugging Face Hub.
    
    Args:
        after_date: ISO date string, only fetch if modified after
        max_tokens: Maximum total tokens to collect
        tags: Filter by tags (e.g., ["text-classification", "image-classification"])
    
    Returns:
        List of dicts with dataset card content and metadata
    """
    documents = []
    total_tokens = 0
    
    try:
        api = HfApi()
        
        # Get recently updated datasets
        datasets = list_datasets(
            sort="lastModified",
            direction=-1,
            limit=500,
            tags=tags
        )
        
        print(f"Scanning recent datasets (after {after_date})...")
        
        for dataset in datasets:
            if total_tokens >= max_tokens:
                break
            
            try:
                # Get dataset info
                dataset_info = api.dataset_info(dataset.id, files_metadata=False)
                
                # Check date
                cutoff_date = datetime.fromisoformat(after_date).replace(tzinfo=None)
                last_modified = dataset_info.lastModified.replace(tzinfo=None)
                
                if last_modified < cutoff_date:
                    continue
                
                # Download README
                readme_path = hf_hub_download(
                    repo_id=dataset.id,
                    filename="README.md",
                    repo_type="dataset"
                )
                
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tokens = count_tokens(content)
                
                if tokens < 50:
                    continue
                
                if total_tokens + tokens <= max_tokens:
                    doc = {
                        "content": content,
                        "url": f"https://huggingface.co/datasets/{dataset.id}",
                        "last_modified": dataset_info.lastModified.isoformat(),
                        "tokens": tokens,
                        "dataset_id": dataset.id,
                        "type": "dataset_card",
                        "tags": dataset_info.tags[:10] if dataset_info.tags else []
                    }
                    documents.append(doc)
                    total_tokens += tokens
                    print(f"  ✓ Loaded {dataset.id} ({tokens} tokens)")
            
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"Error loading dataset cards: {e}")
    
    print(f"\nCollected {len(documents)} dataset cards, {total_tokens:,} tokens total")
    return documents


def load_gutenberg_books(
    book_ids: List[int], 
    max_tokens: Optional[int] = None
) -> List[Dict]:
    """
    Load books from Project Gutenberg.
    
    Args:
        book_ids: List of Gutenberg book IDs (e.g., [1342, 84, 98])
                 Common IDs:
                 - 1342: Pride and Prejudice
                 - 84: Frankenstein
                 - 98: A Tale of Two Cities
                 - 1661: Sherlock Holmes Adventures
                 - 11: Alice in Wonderland
                 - 2701: Moby Dick
                 - 174: The Picture of Dorian Gray
                 - 1952: The Yellow Wallpaper
                 - 345: Dracula
                 - 74: Tom Sawyer
        max_tokens: Maximum total tokens to load (optional)
    
    Returns:
        List of dicts with 'content', 'title', 'book_id', 'tokens'
    """
    books = []
    total_tokens = 0
    
    for book_id in book_ids:
        try:
            if max_tokens and total_tokens >= max_tokens:
                break
            
            print(f"  Downloading Gutenberg book {book_id}...")
            
            # Download raw text bytes
            raw_text = textget.get_text_by_id(book_id)
            
            # Strip Gutenberg headers/footers
            raw_text = textget.strip_headers(raw_text)
            
            # Decode to string
            try:
                content = raw_text.decode('utf-8')
            except UnicodeDecodeError:
                # Try latin-1 as fallback
                content = raw_text.decode('latin-1')
            
            # Clean up excessive whitespace
            content = '\n'.join(line.rstrip() for line in content.split('\n'))
            
            # Count tokens
            tokens = count_tokens(content)
            
            # Truncate if needed
            if max_tokens and total_tokens + tokens > max_tokens:
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 1000:  # Only add if meaningful amount left
                    # Rough truncation by characters (will refine with tokenizer later)
                    char_ratio = remaining_tokens / tokens
                    content = content[:int(len(content) * char_ratio)]
                    tokens = count_tokens(content)
                else:
                    break
            
            books.append({
                "content": content,
                "title": f"Gutenberg Book {book_id}",  # Could fetch metadata if needed
                "book_id": book_id,
                "tokens": tokens,
                "source": "Project Gutenberg"
            })
            
            total_tokens += tokens
            print(f"  ✓ Loaded book {book_id} ({tokens:,} tokens)")
        
        except Exception as e:
            print(f"  ✗ Error loading book {book_id}: {e}")
            continue
    
    return books


# Example usage and testing
if __name__ == "__main__":
    print("Testing corpus loaders...\n")
    
    # Test 1: Load single model card
    print("=" * 60)
    print("TEST 1: Load single model card")
    print("=" * 60)
    doc = load_hf_model_card("meta-llama/Llama-3.2-3B", after_date="2024-08-01")
    if doc:
        print(f"✓ Loaded: {doc['model_id']}")
        print(f"  Tokens: {doc['tokens']}")
        print(f"  Last modified: {doc['last_modified']}")
        print(f"  URL: {doc['url']}")
    else:
        print("✗ Model card not found or not modified after cutoff date")
    
    # Test 2: Load curated model cards (FAST)
    print("\n" + "=" * 60)
    print("TEST 2: Load curated model cards (10k token limit)")
    print("=" * 60)
    docs = load_hf_curated_models(max_tokens=10000, after_date="2024-08-01")
    
    # Test 3: Load Gutenberg books
    print("\n" + "=" * 60)
    print("TEST 3: Load Gutenberg books")
    print("=" * 60)
    books = load_gutenberg_books([1342, 84], max_tokens=50000)
    print(f"\nLoaded {len(books)} books")
    print(f"Total tokens: {sum(b['tokens'] for b in books):,}")
    
    print("\n✓ All tests complete!")
