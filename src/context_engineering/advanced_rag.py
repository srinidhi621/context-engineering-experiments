from typing import List, Dict
import pickle
from pathlib import Path

from .rag import RAGPipeline
from rank_bm25 import BM25Okapi
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedRAGPipeline(RAGPipeline):
    """
    An advanced RAG pipeline that uses hybrid search (BM25 + vector search)
    with Reciprocal Rank Fusion to combine results.
    """

    def __init__(self, reranker=None, padding_generator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_index = None
        self.reranker = reranker
        self.padding_generator = padding_generator

    def save_state(self, path_prefix: str):
        """Save basic RAG state + BM25 index."""
        super().save_state(path_prefix)
        
        if self.bm25_index:
            try:
                with open(f"{path_prefix}_bm25.pkl", 'wb') as f:
                    pickle.dump(self.bm25_index, f)
                logger.info("BM25 index saved.")
            except Exception as e:
                logger.error(f"Failed to save BM25 index: {e}")

    def load_state(self, path_prefix: str) -> bool:
        """Load basic RAG state + BM25 index."""
        if not super().load_state(path_prefix):
            return False
            
        try:
            bm25_path = Path(f"{path_prefix}_bm25.pkl")
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                logger.info("BM25 index loaded.")
            else:
                logger.warning("BM25 index file not found.")
                # Rebuild if chunks exist? For now just warn.
                if self.chunks:
                    logger.info("Rebuilding BM25 index from chunks...")
                    tokenized_corpus = [doc['text'].split(" ") for doc in self.chunks]
                    self.bm25_index = BM25Okapi(tokenized_corpus)
            
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return False

    def index_chunks(self, chunks: List[Dict] = None, **kwargs):
        """
        Builds both the vector index (FAISS) and the BM25 keyword index.
        """
        # Call the parent class's indexer to build the vector index
        super().index_chunks(chunks)
        
        if not self.chunks:
            raise ValueError("No chunks to index for BM25.")
        
        logger.info("Starting Advanced RAG indexing (BM25)...")
        tokenized_corpus = [doc['text'].split(" ") for doc in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("Finished Advanced RAG indexing (BM25).")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieves documents using hybrid search, RRF, and an optional reranker.
        """
        if self.bm25_index is None or self.embeddings is None:
            raise ValueError("Index not built. Call index_chunks() first.")

        # 1. Get results from both retrievers
        vector_results = super().retrieve(query, top_k=top_k)
        
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        bm25_results = []
        for i, doc_idx in enumerate(bm25_top_indices):
            chunk = self.chunks[doc_idx].copy()
            chunk['rank'] = i + 1
            chunk['similarity_score'] = bm25_scores[doc_idx]
            bm25_results.append(chunk)

        # 2. Combine results with Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        k = 60  # RRF constant

        for doc in vector_results:
            doc_id = (doc['doc_id'], doc['chunk_id'])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + doc['rank'])

        for doc in bm25_results:
            doc_id = (doc['doc_id'], doc['chunk_id'])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + doc['rank'])

        # 3. Sort by fused score and prepare for reranking
        sorted_docs = sorted(fused_scores.keys(), key=lambda did: fused_scores[did], reverse=True)
        
        initial_results = []
        for doc_id, chunk_id in sorted_docs:
            chunk = next((c for c in self.chunks if c['doc_id'] == doc_id and c['chunk_id'] == chunk_id), None)
            if chunk:
                initial_results.append(chunk)
        
        # 4. Apply reranker if it exists
        if self.reranker:
            reranked_chunks = self.reranker(query, initial_results)
        else:
            reranked_chunks = initial_results
        
        # 5. Format final top_k results
        final_results = []
        for i, chunk in enumerate(reranked_chunks):
            if len(final_results) >= top_k:
                break
            chunk_copy = chunk.copy()
            chunk_copy['rank'] = i + 1
            chunk_copy['rrf_score'] = fused_scores.get((chunk['doc_id'], chunk['chunk_id']), 0)
            final_results.append(chunk_copy)
            
        return final_results

    def assemble_context_with_padding(self, retrieved_chunks: List[Dict], fill_pct: float, max_context_tokens: int) -> str:
        """
        Assembles context and uses the injected padding_generator if available,
        otherwise falls back to the default PaddingGenerator. This override
        is necessary to make the class more testable.
        """
        # Assemble the retrieved chunks into a single context string.
        # We use a smaller max_tokens for the initial RAG context.
        rag_max_tokens = min(max_context_tokens, 128_000)
        context = self.assemble_context(retrieved_chunks, rag_max_tokens)
        
        # Use the injected padding generator from the test if it exists,
        # otherwise create a real one.
        if self.padding_generator:
            padder = self.padding_generator
        else:
            from src.corpus.padding import PaddingGenerator
            padder = PaddingGenerator()
            
        padded_context = padder.pad_to_fill_percentage(
            context, fill_pct, max_context_tokens
        )
        return padded_context