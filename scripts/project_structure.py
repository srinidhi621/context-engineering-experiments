#!/usr/bin/env python3
"""
Project structure generator for Context Engineering Experiments
Run this script to create the complete directory structure and skeleton files
"""

import os
from pathlib import Path

# Project structure definition
PROJECT_STRUCTURE = {
    "src": {
        "__init__.py": "",
        "config.py": """# Configuration for experiments
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    \"\"\"Base configuration for all experiments\"\"\"
    model_name: str = "gemini-2.0-flash-lite-preview-02-05"
    context_limit: int = 1_000_000
    temperature: float = 0.0
    repetitions: int = 3
    fill_percentages: list = None
    
    def __post_init__(self):
        if self.fill_percentages is None:
            self.fill_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

@dataclass
class APIConfig:
    \"\"\"API configuration\"\"\"
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 1_000_000
    
# Global config instance
config = ExperimentConfig()
api_config = APIConfig()
""",
        "models": {
            "__init__.py": "",
            "gemini_client.py": """# Gemini API client wrapper
import time
import google.generativeai as genai
from typing import Dict, Any, Optional
from ..config import api_config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class GeminiClient:
    \"\"\"Wrapper for Gemini API with rate limiting and error handling\"\"\"
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite-preview-02-05"):
        genai.configure(api_key=api_config.google_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.last_call_time = 0
        self.min_interval = 60.0 / api_config.rate_limit_rpm
        
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        \"\"\"
        Generate response with rate limiting and retry logic
        
        Returns dict with:
            - response: str
            - tokens_input: int
            - tokens_output: int
            - latency_ttft: float (time to first token)
            - latency_total: float
        \"\"\"
        # Rate limiting
        time_since_last = time.time() - self.last_call_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature
                    )
                )
                
                end_time = time.time()
                self.last_call_time = end_time
                
                # Extract metrics
                usage = response.usage_metadata
                
                return {
                    'response': response.text,
                    'tokens_input': usage.prompt_token_count,
                    'tokens_output': usage.candidates_token_count,
                    'latency_ttft': None,  # Gemini doesn't provide TTFT
                    'latency_total': end_time - start_time,
                    'success': True
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        'response': '',
                        'error': str(e),
                        'success': False
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {'success': False}
"""
        },
        "context_engineering": {
            "__init__.py": "",
            "naive.py": """# Naïve context assembly
from typing import List
from ..utils.tokenizer import count_tokens

class NaiveContextAssembler:
    \"\"\"Simple sequential document concatenation\"\"\"
    
    def assemble(
        self,
        documents: List[str],
        target_tokens: int,
        padding_corpus: List[str] = None
    ) -> str:
        \"\"\"
        Dump documents sequentially until target_tokens reached
        
        Args:
            documents: List of document strings
            target_tokens: Target token count (for fill % control)
            padding_corpus: Optional padding to reach target_tokens
            
        Returns:
            Assembled context string
        \"\"\"
        context_parts = []
        current_tokens = 0
        
        # Add documents sequentially
        for doc in documents:
            doc_tokens = count_tokens(doc)
            if current_tokens + doc_tokens <= target_tokens:
                context_parts.append(doc)
                current_tokens += doc_tokens
            else:
                # Truncate to fit
                remaining = target_tokens - current_tokens
                context_parts.append(self._truncate_to_tokens(doc, remaining))
                current_tokens = target_tokens
                break
        
        # Add padding if needed to reach target_tokens
        if padding_corpus and current_tokens < target_tokens:
            padding_text = self._get_padding(
                padding_corpus,
                target_tokens - current_tokens
            )
            context_parts.append(\"\\n\\n[Additional Context]\\n\\n\" + padding_text)
        
        return \"\\n\\n\".join(context_parts)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        \"\"\"Truncate text to approximate token count\"\"\"
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        return text[:max_chars]
    
    def _get_padding(self, padding_corpus: List[str], target_tokens: int) -> str:
        \"\"\"Get padding text to reach target tokens\"\"\"
        padding_parts = []
        current = 0
        
        for pad_doc in padding_corpus:
            tokens = count_tokens(pad_doc)
            if current + tokens <= target_tokens:
                padding_parts.append(pad_doc)
                current += tokens
            else:
                padding_parts.append(self._truncate_to_tokens(
                    pad_doc,
                    target_tokens - current
                ))
                break
        
        return \"\\n\\n\".join(padding_parts)
""",
            "structured.py": """# Engineered context with structure
from typing import List, Dict
from ..utils.tokenizer import count_tokens

class StructuredContextAssembler:
    \"\"\"Hierarchical context with TOC and metadata\"\"\"
    
    def assemble(
        self,
        documents: List[Dict[str, str]],
        target_tokens: int,
        padding_corpus: List[str] = None
    ) -> str:
        \"\"\"
        Create structured context with navigation aids
        
        Args:
            documents: List of dicts with 'content', 'title', 'metadata'
            target_tokens: Target token count
            padding_corpus: Optional padding
            
        Returns:
            Structured context string
        \"\"\"
        # Generate table of contents
        toc = self._generate_toc(documents)
        
        # Add navigation instructions
        instructions = self._create_instructions()
        
        # Assemble structured documents
        doc_parts = []
        current_tokens = count_tokens(toc + instructions)
        
        for i, doc in enumerate(documents):
            structured_doc = self._structure_document(doc, i)
            doc_tokens = count_tokens(structured_doc)
            
            if current_tokens + doc_tokens <= target_tokens * 0.95:  # Leave room
                doc_parts.append(structured_doc)
                current_tokens += doc_tokens
            else:
                break
        
        # Combine all parts
        context = f\"{instructions}\\n\\n{toc}\\n\\n\" + \"\\n\\n\".join(doc_parts)
        
        # Add padding if needed
        if padding_corpus and count_tokens(context) < target_tokens:
            # Similar to naive padding
            pass
        
        return context
    
    def _generate_toc(self, documents: List[Dict]) -> str:
        \"\"\"Generate table of contents\"\"\"
        toc_lines = [\"# TABLE OF CONTENTS\\n\"]
        
        for i, doc in enumerate(documents):
            title = doc.get('title', f'Document {i+1}')
            metadata = doc.get('metadata', {})
            toc_lines.append(
                f\"{i+1}. {title} [ID: doc_{i}] \"
                f\"(Source: {metadata.get('source', 'unknown')})\"
            )
        
        return \"\\n\".join(toc_lines)
    
    def _create_instructions(self) -> str:
        \"\"\"Create navigation instructions\"\"\"
        return \"\"\"
<instructions>
Use the Table of Contents to navigate to relevant sections.
Each document has a unique ID (e.g., doc_0) for reference.
Cite specific documents when answering questions.
</instructions>
\"\"\"
    
    def _structure_document(self, doc: Dict, index: int) -> str:
        \"\"\"Add structure to individual document\"\"\"
        title = doc.get('title', f'Document {index+1}')
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        structured = f\"\"\"
<document id="doc_{index}">
<metadata>
  <title>{title}</title>
  <source>{metadata.get('source', 'unknown')}</source>
  <topic>{metadata.get('topic', 'general')}</topic>
</metadata>
<content>
{content}
</content>
</document>
\"\"\"
        return structured
""",
            "rag.py": """# Basic RAG pipeline
from typing import List, Dict
import numpy as np

class RAGPipeline:
    \"\"\"Basic retrieval-augmented generation pipeline\"\"\"
    
    def __init__(self, embedding_model: str = "text-embedding-004"):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        
    def index_documents(self, documents: List[str]):
        \"\"\"Create chunks and embeddings\"\"\"
        # Chunk documents
        self.chunks = self._chunk_documents(documents)
        
        # Generate embeddings (placeholder - implement with actual embedding API)
        self.embeddings = self._generate_embeddings(self.chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, any]]:
        \"\"\"Retrieve relevant chunks\"\"\"
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        
        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'chunk': self.chunks[i],
                'score': similarities[i],
                'index': i
            }
            for i in top_indices
        ]
    
    def assemble_context(
        self,
        retrieved_chunks: List[Dict],
        max_tokens: int = 128_000
    ) -> str:
        \"\"\"Assemble context from retrieved chunks\"\"\"
        # TODO: Implement proper assembly with token tracking
        context_parts = [chunk['chunk'] for chunk in retrieved_chunks]
        return \"\\n\\n\".join(context_parts)
    
    def _chunk_documents(
        self,
        documents: List[str],
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        \"\"\"Split documents into overlapping chunks\"\"\"
        # TODO: Implement semantic chunking
        chunks = []
        for doc in documents:
            # Simple character-based chunking (replace with token-aware)
            chars_per_token = 4
            chunk_chars = chunk_size * chars_per_token
            overlap_chars = overlap * chars_per_token
            
            for i in range(0, len(doc), chunk_chars - overlap_chars):
                chunks.append(doc[i:i + chunk_chars])
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        \"\"\"Generate embeddings for texts\"\"\"
        # TODO: Implement actual embedding API call
        # Placeholder: random embeddings
        return np.random.rand(len(texts), 768)
    
    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        \"\"\"Compute cosine similarities\"\"\"
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(
            doc_embeddings,
            axis=1,
            keepdims=True
        )
        return np.dot(doc_norms, query_norm)
""",
            "advanced_rag.py": """# Advanced RAG with hybrid search
from .rag import RAGPipeline

class AdvancedRAGPipeline(RAGPipeline):
    \"\"\"Advanced RAG with hybrid search and query decomposition\"\"\"
    
    def retrieve(self, query: str, top_k: int = 10):
        \"\"\"Hybrid search: vector + BM25\"\"\"
        # Vector search
        vector_results = super().retrieve(query, top_k=top_k*2)
        
        # BM25 search (TODO: implement)
        bm25_results = self._bm25_search(query, top_k=top_k*2)
        
        # Fusion (reciprocal rank fusion)
        fused = self._fuse_results(vector_results, bm25_results)
        
        # Rerank
        reranked = self._rerank(query, fused[:top_k])
        
        return reranked
    
    def _bm25_search(self, query: str, top_k: int):
        \"\"\"BM25 sparse retrieval\"\"\"
        # TODO: Implement BM25
        return []
    
    def _fuse_results(self, results1, results2, k: int = 60):
        \"\"\"Reciprocal rank fusion\"\"\"
        # TODO: Implement RRF
        return results1  # Placeholder
    
    def _rerank(self, query: str, candidates):
        \"\"\"Rerank candidates\"\"\"
        # TODO: Implement cross-encoder reranking
        return candidates
"""
        },
        "experiments": {
            "__init__.py": "",
            "base_experiment.py": """# Base experiment class
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BaseExperiment(ABC):
    \"\"\"Base class for all experiments\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.results = []
    
    @abstractmethod
    def setup(self):
        \"\"\"Load corpus and prepare experiment\"\"\"
        pass
    
    @abstractmethod
    def run(self):
        \"\"\"Execute experiment\"\"\"
        pass
    
    @abstractmethod
    def evaluate(self):
        \"\"\"Evaluate results\"\"\"
        pass
    
    def save_results(self, filepath: str):
        \"\"\"Save experiment results\"\"\"
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f\"Results saved to {filepath}\")
""",
            "exp1_needle.py": "# Experiment 1: Needle in Multiple Haystacks\nfrom .base_experiment import BaseExperiment\n\nclass NeedleExperiment(BaseExperiment):\n    pass",
            "exp2_pollution.py": "# Experiment 2: Context Pollution\nfrom .base_experiment import BaseExperiment\n\nclass PollutionExperiment(BaseExperiment):\n    pass",
            "exp3_memory.py": "# Experiment 3: Multi-Turn Memory\nfrom .base_experiment import BaseExperiment\n\nclass MemoryExperiment(BaseExperiment):\n    pass",
            "exp4_precision.py": "# Experiment 4: Precision Retrieval\nfrom .base_experiment import BaseExperiment\n\nclass PrecisionExperiment(BaseExperiment):\n    pass",
            "exp5_frontier.py": "# Experiment 5: Cost-Latency Frontier\nfrom .base_experiment import BaseExperiment\n\nclass FrontierExperiment(BaseExperiment):\n    pass"
        },
        "corpus": {
            "__init__.py": "",
            "loaders.py": """# Corpus loading utilities
from pathlib import Path
from typing import List, Dict
import json

def load_corpus(corpus_path: str) -> List[str]:
    \"\"\"Load corpus from directory\"\"\"
    path = Path(corpus_path)
    documents = []
    
    for file_path in path.glob('**/*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    
    return documents

def load_questions(question_file: str) -> List[Dict]:
    \"\"\"Load questions and ground truth\"\"\"
    with open(question_file, 'r') as f:
        return json.load(f)
""",
            "generators.py": "# Synthetic corpus generation\n\ndef generate_synthetic_docs(n_docs: int, tokens_per_doc: int):\n    pass",
            "padding.py": """# Padding generation for fill % control
from typing import List
import random

def generate_padding(
    target_tokens: int,
    padding_corpus: List[str],
    domain: str = None
) -> str:
    \"\"\"
    Generate domain-agnostic padding to reach target tokens
    
    Args:
        target_tokens: Target number of tokens
        padding_corpus: Pool of padding documents
        domain: Task domain to avoid (e.g., 'technology')
        
    Returns:
        Padding text
    \"\"\"
    # Sample from padding corpus
    selected = []
    current_tokens = 0
    
    # Shuffle to get variety
    shuffled = random.sample(padding_corpus, len(padding_corpus))
    
    for doc in shuffled:
        # Rough token estimate
        doc_tokens = len(doc) // 4
        
        if current_tokens + doc_tokens <= target_tokens:
            selected.append(doc)
            current_tokens += doc_tokens
        else:
            # Truncate last document
            remaining_chars = (target_tokens - current_tokens) * 4
            selected.append(doc[:remaining_chars])
            break
    
    return \"\\n\\n\".join(selected)
"""
        },
        "evaluation": {
            "__init__.py": "",
            "metrics.py": """# Evaluation metrics
from typing import Dict, Any

def evaluate_response(
    response: str,
    ground_truth: str,
    context: str
) -> Dict[str, Any]:
    \"\"\"
    Compute all evaluation metrics
    
    Returns:
        Dict with correctness, citation_accuracy, etc.
    \"\"\"
    return {
        'correctness': 0.0,  # TODO: Implement LLM judge
        'citation_accuracy': 0.0,
        'hallucination_rate': 0.0,
        'completeness': 0.0
    }
""",
            "judges.py": "# LLM-as-judge evaluation\n\ndef llm_judge_correctness(response, ground_truth):\n    pass",
            "human_eval.py": "# Human evaluation interface\n\ndef create_evaluation_task(response, question):\n    pass"
        },
        "utils": {
            "__init__.py": "",
            "tokenizer.py": """# Token counting utilities
import tiktoken

# Use GPT-4 tokenizer as approximation for Gemini
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    \"\"\"Count tokens in text\"\"\"
    return len(encoding.encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    \"\"\"Truncate text to max tokens\"\"\"
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
""",
            "logging.py": """# Structured logging
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    \"\"\"Get configured logger\"\"\"
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger
""",
            "stats.py": """# Statistical analysis utilities
from scipy import stats
import numpy as np

def paired_t_test(condition1, condition2):
    \"\"\"Paired t-test between two conditions\"\"\"
    t_stat, p_value = stats.ttest_rel(condition1, condition2)
    return {'t_statistic': t_stat, 'p_value': p_value}

def cohen_d(group1, group2):
    \"\"\"Cohen's d effect size\"\"\"
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
"""
        }
    },
    "scripts": {
        "run_experiment.py": """#!/usr/bin/env python3
\"\"\"Main experiment runner\"\"\"
import argparse
from src.experiments.exp1_needle import NeedleExperiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--conditions', type=str, default='all')
    args = parser.parse_args()
    
    print(f\"Running experiment: {args.experiment}\")
    # TODO: Implement

if __name__ == '__main__':
    main()
""",
        "run_calibration.py": "#!/usr/bin/env python3\\n# Baseline calibration script\\n",
        "analyze_results.py": "#!/usr/bin/env python3\\n# Results analysis script\\n",
        "generate_report.py": "#!/usr/bin/env python3\\n# Report generation script\\n"
    },
    "data": {
        "raw": {},
        "processed": {},
        "questions": {}
    },
    "results": {
        "raw": {},
        "metrics": {},
        "analysis": {},
        "visualizations": {}
    },
    "notebooks": {},
    "tests": {
        "__init__.py": "",
        "test_context_engineering.py": "# Tests for context engineering\\nimport pytest\\n",
        "test_corpus.py": "# Tests for corpus utilities\\nimport pytest\\n",
        "test_evaluation.py": "# Tests for evaluation metrics\\nimport pytest\\n",
        "test_models.py": "# Tests for model clients\\nimport pytest\\n"
    }
}

def create_structure(base_path: Path, structure: dict):
    \"\"\"Recursively create directory structure\"\"\"
    for name, content in structure.items():
        path = base_path / name
        
        if isinstance(content, dict):
            # It's a directory
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            # It's a file
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)

def create_additional_files(base_path: Path):
    \"\"\"Create additional configuration files\"\"\"
    
    # requirements.txt
    requirements = \"\"\"google-generativeai>=0.3.0
numpy>=1.24.0
scipy>=1.10.0
tiktoken>=0.5.0
pytest>=7.4.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
python-dotenv>=1.0.0
\"\"\"
    (base_path / "requirements.txt").write_text(requirements)
    
    # .env.example
    env_example = \"\"\"# Google AI API Key
GOOGLE_API_KEY=your_api_key_here

# Rate limits
RATE_LIMIT_RPM=60
RATE_LIMIT_TPM=1000000
\"\"\"
    (base_path / ".env.example").write_text(env_example)
    
    # .gitignore
    gitignore = \"\"\"# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.env

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Results
results/raw/*
results/metrics/*
!results/raw/.gitkeep

# Notebooks
.ipynb_checkpoints/
notebooks/*.ipynb

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
\"\"\"
    (base_path / ".gitignore").write_text(gitignore)
    
    # setup.py
    setup_py = \"\"\"from setuptools import setup, find_packages

setup(
    name="context-engineering-experiments",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tiktoken>=0.5.0",
    ],
    python_requires=">=3.10",
)
\"\"\"
    (base_path / "setup.py").write_text(setup_py)
    
    # Create .gitkeep files for empty directories
    for empty_dir in ["data/raw", "data/processed", "data/questions",
                      "results/raw", "results/metrics", "results/analysis",
                      "results/visualizations", "notebooks"]:
        gitkeep_path = base_path / empty_dir / ".gitkeep"
        gitkeep_path.parent.mkdir(parents=True, exist_ok=True)
        gitkeep_path.touch()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = Path("context-engineering-experiments")
    
    print(f"Creating project structure in: {project_path.absolute()}")
    
    # Create main structure
    project_path.mkdir(parents=True, exist_ok=True)
    create_structure(project_path, PROJECT_STRUCTURE)
    create_additional_files(project_path)
    
    print("✅ Project structure created successfully!")
    print("\\nNext steps:")
    print(f"  cd {project_path}")
    print("  python -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  cp .env.example .env")
    print("  # Edit .env with your API keys")
    print("  python scripts/run_calibration.py")
