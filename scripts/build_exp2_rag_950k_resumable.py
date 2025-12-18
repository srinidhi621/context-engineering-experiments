#!/usr/bin/env python3
"""Build the 950k RAG index for Exp2 with resumable embedding batches.

- Keeps 512-word chunking (no methodology change).
- Embeds in batches, checkpoints partial embeddings, and can be re-run to resume
  if the daily embedding RPD cap is reached.
- Final state is saved to results/cache/exp2_rag_950000_* once all embeddings complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from src.experiments.exp2_pollution import PollutionExperiment
from src.context_engineering.rag import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger("rag_builder_950k_resumable")

LEVEL = 950_000
CACHE_DIR = Path("results/cache")
CHUNKS_PATH = CACHE_DIR / "exp2_rag_950000_chunks.json"
EMB_PATH = CACHE_DIR / "exp2_rag_950000_embeddings.partial.npy"
PROGRESS_PATH = CACHE_DIR / "exp2_rag_950000_progress.json"
FINAL_PREFIX = CACHE_DIR / "exp2_rag_950000"
BATCH_SIZE = 200  # modest to avoid RPM spikes


def load_chunks(exp: PollutionExperiment) -> List[dict]:
    """Load or create chunked documents for 950k level."""
    if CHUNKS_PATH.exists():
        logger.info("Loading existing chunks from %s", CHUNKS_PATH)
        with CHUNKS_PATH.open() as handle:
            return json.load(handle)

    base_docs = exp._ensure_tokens(exp.base_corpus)
    pollution_docs = exp._ensure_tokens(exp._get_pollution_corpus(LEVEL))
    corpus = base_docs + pollution_docs

    rag = RAGPipeline(padding_generator=exp.padding_generator)
    docs = [d["content"] for d in corpus]
    rag.chunk_documents(docs)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w") as handle:
        json.dump(rag.chunks, handle)
    logger.info("Chunked %d documents into %d chunks", len(corpus), len(rag.chunks))
    return rag.chunks


def load_progress():
    if PROGRESS_PATH.exists() and EMB_PATH.exists():
        data = json.load(PROGRESS_PATH.open())
        next_idx = data.get("next_idx", 0)
        partial = np.load(str(EMB_PATH))
        logger.info("Resuming from chunk %d with %d embeddings loaded", next_idx, len(partial))
        return next_idx, partial.tolist()
    return 0, []


def save_progress(next_idx: int, embeddings: List[List[float]]) -> None:
    np.save(str(EMB_PATH), np.array(embeddings, dtype=np.float32))
    with PROGRESS_PATH.open("w") as handle:
        json.dump({"next_idx": next_idx}, handle)
    logger.info("Checkpointed %d embeddings (next_idx=%d)", len(embeddings), next_idx)


def build_index() -> bool:
    exp = PollutionExperiment()
    chunks = load_chunks(exp)
    total = len(chunks)

    next_idx, embeddings = load_progress()
    rag = RAGPipeline(padding_generator=exp.padding_generator)
    rag.chunks = chunks

    try:
        while next_idx < total:
            batch_end = min(total, next_idx + BATCH_SIZE)
            batch_texts = [c["text"] for c in chunks[next_idx:batch_end]]
            logger.info("Embedding batch %d:%d of %d", next_idx, batch_end, total)
            try:
                batch_embs = rag._embed_texts(batch_texts)
            except Exception as exc:  # likely RPD limit
                logger.warning("Embedding halted at %d:%d due to: %s", next_idx, batch_end, exc)
                save_progress(next_idx, embeddings)
                return False
            embeddings.extend(batch_embs)
            next_idx = batch_end
            if next_idx % 200 == 0 or next_idx == total:
                save_progress(next_idx, embeddings)
    except KeyboardInterrupt:
        logger.warning("Interrupted; saving progress at %d", next_idx)
        save_progress(next_idx, embeddings)
        return False

    # All embeddings done; build index and save state
    rag.embeddings = np.array(embeddings, dtype=np.float32)
    rag.index_chunks()
    rag.save_state(str(FINAL_PREFIX))
    for path in (EMB_PATH, PROGRESS_PATH):
        if path.exists():
            path.unlink()
    logger.info("RAG index built and saved to %s*", FINAL_PREFIX)
    return True


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    build_index()
