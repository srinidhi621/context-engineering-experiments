"""Corpus loading utilities for GitHub and Project Gutenberg"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from github import Github, GithubException, Auth
from gutenbergpy import textget
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-4)
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def load_github_file(
    repo_name: str, 
    file_path: str, 
    after_date: str = "2024-08-01"
) -> Optional[Dict]:
    """
    Load single file from GitHub repository.
    
    Args:
        repo_name: Repository name in format "owner/repo" (e.g., "pytorch/pytorch")
        file_path: Path to file in repo (e.g., "README.md" or "docs/optimizer.md")
        after_date: ISO date string, only fetch if modified after this date
    
    Returns:
        Dict with 'content', 'url', 'last_modified', 'tokens', 'path'
        Returns None if file doesn't meet criteria or on error
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN not found in environment variables")
    
    try:
        # Initialize GitHub client
        auth = Auth.Token(github_token)
        g = Github(auth=auth)
        repo = g.get_repo(repo_name)
        
        # Get file content
        file_content = repo.get_contents(file_path)
        
        if isinstance(file_content, list):
            # If it's a directory, return None
            return None
        
        # Get last commit for this file to check modification date
        commits = repo.get_commits(path=file_path)
        if commits.totalCount == 0:
            return None
        
        last_commit = commits[0]
        last_modified = last_commit.commit.author.date
        
        # Check if modified after target date
        cutoff_date = datetime.fromisoformat(after_date)
        if last_modified.replace(tzinfo=None) < cutoff_date:
            return None
        
        # Decode content
        content = file_content.decoded_content.decode('utf-8')
        
        # Count tokens
        tokens = count_tokens(content)
        
        return {
            "content": content,
            "url": file_content.html_url,
            "last_modified": last_modified.isoformat(),
            "tokens": tokens,
            "path": file_path,
            "repo": repo_name,
            "sha": file_content.sha
        }
    
    except GithubException as e:
        print(f"GitHub API error for {repo_name}/{file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {repo_name}/{file_path}: {e}")
        return None


def load_github_docs(
    repo_name: str, 
    path: str = "docs/", 
    after_date: str = "2024-08-01",
    max_tokens: int = 50000,
    extensions: List[str] = [".md", ".rst", ".txt"]
) -> List[Dict]:
    """
    Load documentation from GitHub repository.
    
    Args:
        repo_name: Repository name in format "owner/repo"
        path: Path to documentation folder (default: "docs/")
        after_date: ISO date string, only fetch if modified after
        max_tokens: Maximum total tokens to collect
        extensions: File extensions to include
    
    Returns:
        List of dicts with file metadata and content
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN not found in environment variables")
    
    documents = []
    total_tokens = 0
    
    try:
        auth = Auth.Token(github_token)
        g = Github(auth=auth)
        repo = g.get_repo(repo_name)
        
        # Get contents of the path
        try:
            contents = repo.get_contents(path)
        except GithubException:
            # Try root README if docs/ doesn't exist
            try:
                contents = [repo.get_contents("README.md")]
            except:
                return []
        
        # Process files (recursively if needed)
        def process_contents(contents_list):
            nonlocal total_tokens
            
            for content_file in contents_list:
                if total_tokens >= max_tokens:
                    break
                
                # If it's a directory, recurse
                if content_file.type == "dir":
                    try:
                        nested_contents = repo.get_contents(content_file.path)
                        process_contents(nested_contents)
                    except:
                        continue
                
                # If it's a file with matching extension
                elif content_file.type == "file":
                    # Check extension
                    if not any(content_file.name.endswith(ext) for ext in extensions):
                        continue
                    
                    # Load the file
                    doc = load_github_file(repo_name, content_file.path, after_date)
                    
                    if doc and total_tokens + doc['tokens'] <= max_tokens:
                        documents.append(doc)
                        total_tokens += doc['tokens']
                        print(f"  ✓ Loaded {content_file.path} ({doc['tokens']} tokens)")
        
        # Start processing
        if isinstance(contents, list):
            process_contents(contents)
        else:
            # Single file
            doc = load_github_file(repo_name, contents.path, after_date)
            if doc:
                documents.append(doc)
    
    except GithubException as e:
        print(f"GitHub API error for {repo_name}: {e}")
    except Exception as e:
        print(f"Error loading docs from {repo_name}: {e}")
    
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
    
    # Test 1: Load single GitHub file
    print("=" * 60)
    print("TEST 1: Load single GitHub file")
    print("=" * 60)
    doc = load_github_file("pytorch/pytorch", "README.md", after_date="2024-08-01")
    if doc:
        print(f"✓ Loaded: {doc['path']}")
        print(f"  Tokens: {doc['tokens']}")
        print(f"  Last modified: {doc['last_modified']}")
        print(f"  URL: {doc['url']}")
    else:
        print("✗ File not found or not modified after cutoff date")
    
    # Test 2: Load documentation from a repo
    print("\n" + "=" * 60)
    print("TEST 2: Load docs from repository (10k token limit)")
    print("=" * 60)
    docs = load_github_docs("pytorch/pytorch", path="docs/", max_tokens=10000)
    print(f"\nLoaded {len(docs)} documents")
    print(f"Total tokens: {sum(d['tokens'] for d in docs):,}")
    
    # Test 3: Load Gutenberg books
    print("\n" + "=" * 60)
    print("TEST 3: Load Gutenberg books")
    print("=" * 60)
    books = load_gutenberg_books([1342, 84], max_tokens=50000)
    print(f"\nLoaded {len(books)} books")
    print(f"Total tokens: {sum(b['tokens'] for b in books):,}")
    
    print("\n✓ All tests complete!")
