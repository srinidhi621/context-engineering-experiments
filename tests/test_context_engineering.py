"""Tests for context engineering strategies."""

import pytest

from src.context_engineering.naive import NaiveContextAssembler
from src.context_engineering.structured import StructuredContextAssembler
from src.context_engineering.rag import RAGPipeline
from src.context_engineering.advanced_rag import AdvancedRAGPipeline
from src.utils.tokenizer import count_tokens


def test_naive_assembler_respects_token_budget():
    docs = [
        {"content": "Relevant details about llama models. " * 5},
        {"content": "Additional supplemental information. " * 5},
    ]
    max_tokens = count_tokens(docs[0]["content"]) + 10

    assembler = NaiveContextAssembler()
    assembled = assembler.assemble(docs, max_tokens=max_tokens)

    assert count_tokens(assembled) <= max_tokens
    assert docs[0]["content"].strip() in assembled


def test_naive_assembler_raises_for_invalid_budget():
    assembler = NaiveContextAssembler()
    with pytest.raises(ValueError):
        assembler.assemble([], max_tokens=0)


def test_structured_assembler_includes_metadata_and_toc():
    docs = [
        {
            "content": "First document body.",
            "title": "Doc One",
            "url": "https://example.com/one",
            "last_modified": "2024-09-01",
            "tokens": 20,
            "tags": ["llm"],
        },
        {
            "content": "Second document body.",
            "model_id": "doc-two",
        },
    ]
    assembler = StructuredContextAssembler()
    result = assembler.assemble(docs, max_tokens=500)

    assert "# Context Package" in result
    assert "## Table of Contents" in result
    assert "Doc One" in result and "doc-two" in result
    assert "- Source: https://example.com/one" in result
    assert count_tokens(result) <= 500


def test_rag_pipeline_retrieve_without_faiss():
    docs = [
        "alpha beta gamma delta epsilon",
        "theta iota kappa lambda mu",
    ]

    def embed_many(texts):
        return [[len(t), float(idx)] for idx, t in enumerate(texts)]

    def embed_one(text):
        return [len(text), 0.0]

    pipeline = RAGPipeline(
        embed_many_fn=embed_many, embed_one_fn=embed_one, use_faiss=False
    )
    chunks = [
        {"text": docs[0], "word_count": 5, "doc_id": 0, "chunk_id": 0},
        {"text": docs[1], "word_count": 5, "doc_id": 1, "chunk_id": 0},
    ]
    pipeline.index_chunks(chunks)

    results = pipeline.retrieve("alpha beta", top_k=2)
    assert results, "Expected retrieval results"
    assert all("similarity_score" in r for r in results)


def test_advanced_rag_applies_padding_and_reranker():
    docs = ["first doc text", "second doc text"]

    def embed_many(texts):
        return [[float(len(t)), 0.0] for t in texts]

    def embed_one(text):
        return [float(len(text)), 0.0]

    class FakePadding:
        def __init__(self):
            self.called_with = None

        def pad_to_fill_percentage(self, content, fill_pct, max_context_tokens):
            self.called_with = (content, fill_pct, max_context_tokens)
            return content + " | PAD"

    reranker_called = {"flag": False}

    def reranker(query, chunks):
        reranker_called["flag"] = True
        return list(reversed(chunks))

    padding = FakePadding()
    pipeline = AdvancedRAGPipeline(
        embed_many_fn=embed_many,
        embed_one_fn=embed_one,
        padding_generator=padding,
        reranker=reranker,
        use_faiss=False,
    )
    chunks = [
        {"text": docs[0], "word_count": 3, "doc_id": 0, "chunk_id": 0},
        {"text": docs[1], "word_count": 3, "doc_id": 1, "chunk_id": 0},
    ]
    pipeline.index_chunks(chunks)

    retrieved = pipeline.retrieve("query text", top_k=2)
    assert reranker_called["flag"]

    context = pipeline.assemble_context_with_padding(
        retrieved,
        fill_pct=0.5,
        max_context_tokens=200,
    )
    assert context.endswith("PAD")
    assert padding.called_with is not None
