"""Basic RAG pipeline with vector search using Gemini embeddings."""

from __future__ import annotations

from typing import Callable, List, Dict, Optional, Sequence

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

from src.models.gemini_client import GeminiClient


EmbedManyFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]
EmbedOneFn = Callable[[str], Sequence[float]]


class RAGPipeline:
    """Basic RAG (Retrieval-Augmented Generation) pipeline using Gemini embeddings."""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        *,
        embed_many_fn: Optional[EmbedManyFn] = None,
        embed_one_fn: Optional[EmbedOneFn] = None,
        client: Optional[GeminiClient] = None,
        use_faiss: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            embedding_model: Embedding model name.
            embed_many_fn: Optional injection for batch embedding.
            embed_one_fn: Optional injection for single-text embedding.
            client: Optional GeminiClient (created lazily otherwise).
        """
        self.embedding_model = embedding_model
        self.embed_many_fn = embed_many_fn
        self.embed_one_fn = embed_one_fn
        self.client = client
        self.use_faiss = use_faiss and faiss is not None
        self.chunks: List[Dict[str, any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
   
    def chunk_documents(
        self,
        documents: List[str],
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[Dict[str, str]]:
        """
        Break documents into overlapping chunks (word-based approximation).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("chunk_size must be greater than overlap")

        chunks = []

        for doc_id, doc in enumerate(documents):
            words = doc.split()
            chunk_start = 0
            chunk_id = 0

            while chunk_start < len(words):
                chunk_end = min(chunk_start + chunk_size, len(words))
                chunk_text = " ".join(words[chunk_start:chunk_end])

                chunks.append(
                    {
                        "text": chunk_text,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "word_count": len(chunk_text.split()),
                    }
                )

                if chunk_end == len(words):
                    break

                chunk_start = max(chunk_start + chunk_size - overlap, chunk_start + 1)
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
        
        chunk_texts = [c['text'] for c in self.chunks]
        embeddings = self._embed_texts(chunk_texts)
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.use_faiss:
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
        if self.embeddings is None:
            raise ValueError("Index not built. Call index_chunks() first")
        
        query_vec = np.array([self._embed_query(query)], dtype=np.float32)
        top_k = min(top_k, len(self.chunks))
        
        if self.index is not None:
            distances, indices = self.index.search(query_vec, top_k)
            return self._format_results(indices[0], distances[0])
        
        similarities = self.embeddings @ query_vec.T
        ranked_indices = np.argsort(similarities[:, 0])[::-1][:top_k]
        ranked_scores = similarities[ranked_indices, 0]
        distances = 1 - ranked_scores
        return self._format_results(ranked_indices, distances)
    
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

    def assemble_context_with_padding(self, 
                                  retrieved_chunks: List[Dict],
                                  fill_pct: float,
                                  max_context_tokens: int = 1_000_000) -> str:
        """
        Assemble retrieved chunks and pad to match fill percentage.
        
        This is the KEY methodological control for H2.
        """
        from src.corpus.padding import PaddingGenerator
        
        # Assemble retrieved chunks
        # Note: The max_tokens for RAG is typically smaller (e.g., 128k)
        rag_max_tokens = min(max_context_tokens, 128_000)
        context = self.assemble_context(retrieved_chunks, rag_max_tokens)
        
        # Pad to match fill percentage of the total context window
        padder = PaddingGenerator()
        padded_context = padder.pad_to_fill_percentage(
            context, fill_pct, max_context_tokens
        )
        
        return padded_context

    def _embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self.embed_many_fn:
            return self.embed_many_fn(texts)
        if not self.client:
            self.client = GeminiClient()
        return self.client.batch_embed_text(list(texts), self.embedding_model)

    def _embed_query(self, query: str) -> Sequence[float]:
        if self.embed_one_fn:
            return self.embed_one_fn(query)
        if not self.client:
            self.client = GeminiClient()
        return self.client.embed_text(query, self.embedding_model)

    def _format_results(self, indices, distances) -> List[Dict]:
        results = []
        for rank, idx in enumerate(indices):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)].copy()
            chunk['similarity_score'] = float(1 / (1 + distances[rank]))
            chunk['rank'] = rank + 1
            results.append(chunk)
        return results
