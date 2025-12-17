#!/usr/bin/env python3
"""Build the 950k RAG index for Exp2 (no generation).

Warning: This will issue ~1.6k embedding calls; stop early if RPD is low.
"""
from pathlib import Path
from src.experiments.exp2_pollution import PollutionExperiment
from src.context_engineering.rag import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger("rag_builder_950k")

def main() -> None:
    level = 950_000
    exp = PollutionExperiment()
    base_docs = exp._ensure_tokens(exp.base_corpus)
    pollution_docs = exp._ensure_tokens(exp._get_pollution_corpus(level))
    corpus = base_docs + pollution_docs

    cache_dir = Path("results/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rag_path = cache_dir / f"exp2_rag_{level}"

    logger.info(f"Building RAG index for level {level}")
    rag = RAGPipeline(padding_generator=exp.padding_generator)
    docs = [d['content'] for d in corpus]
    rag.chunk_documents(docs)
    rag.index_chunks()
    rag.save_state(str(rag_path))
    logger.info(f"RAG index built and saved to {rag_path}*")

if __name__ == "__main__":
    main()
