"""Advanced RAG pipeline with reranking and padding support."""

from __future__ import annotations

from typing import Callable, List, Dict, Optional, Sequence

from src.context_engineering.rag import RAGPipeline
from src.corpus.padding import PaddingGenerator

RerankerFn = Callable[[str, Sequence[Dict]], Sequence[Dict]]


class AdvancedRAGPipeline(RAGPipeline):
    """
    Extends the basic RAG pipeline with optional reranking and padding to
    match target fill percentages.
    """

    def __init__(
        self,
        *args,
        reranker: Optional[RerankerFn] = None,
        padding_generator: Optional[PaddingGenerator] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reranker = reranker
        self.padding_generator = padding_generator

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        chunks = super().retrieve(query, top_k)
        if self.reranker:
            reranked = self.reranker(query, chunks)
            return list(reranked)
        return chunks

    def assemble_context_with_padding(
        self,
        retrieved_chunks: Sequence[Dict],
        *,
        fill_pct: float,
        max_context_tokens: int,
    ) -> str:
        """
        Assemble context and pad to the desired fill percentage.
        """
        context = super().assemble_context(
            list(retrieved_chunks),
            max_tokens=max_context_tokens,
        )
        if not self.padding_generator:
            return context
        return self.padding_generator.pad_to_fill_percentage(
            context,
            fill_pct=fill_pct,
            max_context_tokens=max_context_tokens,
        )
