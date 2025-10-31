"""Basic RAG pipeline with vector search using Gemini embeddings"""

from typing import List, Dict, Optional
import numpy as np
from src.models.gemini_client import GeminiClient

try:
    import faiss
except ImportError:
    faiss = None


class RAGPipeline:
    """Basic RAG (Retrieval-Augmented Generation) pipeline using Gemini embeddings"""
    
    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: Embedding model to use (uses config default if None)
        """
        self.client = GeminiClient()
        self.embedding_model = embedding_model
        self.chunks: List[Dict[str, any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
    
    def chunk_documents(
        self, 
        documents: List[str], 
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Dict[str, str]]:
        """
        Break documents into overlapping chunks.
        
        Args:
            documents: List of document strings
            chunk_size: Size of each chunk in tokens (approximate)
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of dicts with 'text', 'doc_id', 'chunk_id'
        """
        chunks = []
        
        for doc_id, doc in enumerate(documents):
            words = doc.split()
            chunk_start = 0
            chunk_id = 0
            
            while chunk_start < len(words):
                chunk_end = min(chunk_start + chunk_size, len(words))
                chunk_text = ' '.join(words[chunk_start:chunk_end])
                
                chunks.append({
                    'text': chunk_text,
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'word_count': len(chunk_text.split())
                })
                
                chunk_start = chunk_end - overlap
                chunk_id += 1
        
        self.chunks = chunks
        return chunks
    
    def index_chunks(self, chunks: Optional[List[Dict]] = None):
        """
        Generate embeddings and build vector index.
        
        Args:
            chunks: Chunks to index (uses self.chunks if None)
        """
        if chunks:
            self.chunks = chunks
        
        if not self.chunks:
            raise ValueError("No chunks to index")
        
        if not faiss:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu")
        
        # Generate embeddings
        chunk_texts = [c['text'] for c in self.chunks]
        embeddings = self.client.batch_embed_text(chunk_texts, self.embedding_model)
        
        # Convert to numpy array
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of dicts with chunk info and similarity scores
        """
        if not self.index:
            raise ValueError("Index not built. Call index_chunks() first")
        
        # Embed query
        query_embedding = np.array([
            self.client.embed_text(query, self.embedding_model)
        ], dtype=np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid result
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def assemble_context(
        self, 
        retrieved_chunks: List[Dict],
        max_tokens: int = 128_000
    ) -> str:
        """
        Assemble retrieved chunks into context string.
        
        Args:
            retrieved_chunks: Chunks from retrieve()
            max_tokens: Maximum tokens in final context
            
        Returns:
            Assembled context string
        """
        context_parts = []
        total_tokens = 0
        
        for chunk in retrieved_chunks:
            chunk_tokens = chunk.get('word_count', len(chunk['text'].split()))
            
            if total_tokens + chunk_tokens <= max_tokens:
                context_parts.append(chunk['text'])
                total_tokens += chunk_tokens
            else:
                break  # Stop if we'd exceed max
        
        return '\n\n'.join(context_parts)

