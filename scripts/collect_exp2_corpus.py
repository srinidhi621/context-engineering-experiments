#!/usr/bin/env python3
"""
Collect Experiment 2 Base Corpus.
Fetches documentation from 3-5 distinct open source libraries (not used in Exp 1).
Target: ~50k tokens total.
"""

import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict
from src.utils.tokenizer import count_tokens

# Target documentation URLs (dense technical content)
TARGET_URLS = [
    # FastAPI
    {
        "source": "FastAPI",
        "url": "https://fastapi.tiangolo.com/",
        "title": "FastAPI Home"
    },
    {
        "source": "FastAPI",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "title": "FastAPI First Steps"
    },
    {
        "source": "FastAPI",
        "url": "https://fastapi.tiangolo.com/tutorial/path-params/",
        "title": "FastAPI Path Parameters"
    },
    # Pydantic
    {
        "source": "Pydantic",
        "url": "https://docs.pydantic.dev/latest/",
        "title": "Pydantic Home"
    },
    {
        "source": "Pydantic",
        "url": "https://docs.pydantic.dev/latest/concepts/models/",
        "title": "Pydantic Models"
    },
    # SQLAlchemy
    {
        "source": "SQLAlchemy",
        "url": "https://docs.sqlalchemy.org/en/20/tutorial/index.html",
        "title": "SQLAlchemy Tutorial"
    },
    {
        "source": "SQLAlchemy",
        "url": "https://docs.sqlalchemy.org/en/20/orm/quickstart.html",
        "title": "SQLAlchemy ORM Quickstart"
    },
    # Celery
    {
        "source": "Celery",
        "url": "https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html",
        "title": "Celery First Steps"
    },
    # Requests
    {
        "source": "Requests",
        "url": "https://requests.readthedocs.io/en/latest/user/quickstart/",
        "title": "Requests Quickstart"
    },
    # Additional URLs to reach 50k
    {
        "source": "Pydantic",
        "url": "https://docs.pydantic.dev/latest/concepts/fields/",
        "title": "Pydantic Fields"
    },
    {
        "source": "SQLAlchemy",
        "url": "https://docs.sqlalchemy.org/en/20/tutorial/metadata.html",
        "title": "SQLAlchemy Metadata"
    },
    {
        "source": "Celery",
        "url": "https://docs.celeryq.dev/en/stable/getting-started/next-steps.html",
        "title": "Celery Next Steps"
    },
    {
        "source": "Requests",
        "url": "https://requests.readthedocs.io/en/latest/user/advanced/",
        "title": "Requests Advanced"
    }
]

TARGET_TOKENS = 50_000

def fetch_and_clean(url: str) -> str:
    """Fetch URL and extract main text content."""
    try:
        headers = {
            'User-Agent': 'ContextEngineeringBot/1.0 (Research Experiment; +https://github.com/srinidhi621/context-engineering-experiments)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts, styles, nav, footer to get mostly content
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        return clean_text
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def main():
    output_dir = Path("data/raw/exp2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "base_corpus.json"
    
    documents = []
    total_tokens = 0
    
    print(f"Collecting Experiment 2 Corpus (Target: {TARGET_TOKENS:,} tokens)...")
    
    for target in TARGET_URLS:
        if total_tokens >= TARGET_TOKENS:
            break
            
        print(f"Fetching {target['source']} - {target['title']} ({target['url']})...")
        
        content = fetch_and_clean(target['url'])
        if not content:
            continue
            
        tokens = count_tokens(content)
        print(f"  -> {tokens:,} tokens")
        
        # If a single page is huge, truncate it (unlikely for docs, but good safety)
        # But we want complete documents if possible.
        
        doc = {
            "content": content,
            "source": target['source'],
            "url": target['url'],
            "title": target['title'],
            "tokens": tokens,
            "type": "documentation"
        }
        
        documents.append(doc)
        total_tokens += tokens
        
        # Be polite
        time.sleep(1)
        
    print(f"\nTotal Collected: {total_tokens:,} tokens")
    print(f"Documents: {len(documents)}")
    
    if total_tokens < 20_000:
        print("⚠️  Warning: Collected tokens are low. You might need to add more URLs.")
    
    with open(output_file, 'w') as f:
        json.dump(documents, f, indent=2)
        
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
