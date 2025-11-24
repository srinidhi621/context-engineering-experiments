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


def test_structured_assembler_creates_xml_structure():
    docs = [
        {
            "content": "First document body.",
            "model_id": "model-one",
            "license": "mit",
        },
        {
            "content": "Second document body.",
            "model_id": "model-two",
        },
    ]
    assembler = StructuredContextAssembler()
    result = assembler.assemble(docs, max_tokens=500)

    # Check for overall XML structure
    assert "<context_package>" in result
    assert "</context_package>" in result
    assert "<table_of_contents>" in result
    
    # Check for TOC entries
    assert "model-one (id: doc_1)" in result
    assert "model-two (id: doc_2)" in result
    
    # Check for document structure and metadata
    assert '<document id="doc_1">' in result
    assert '<metadata>' in result
    assert "- license: mit" in result
    assert '<content>' in result
    assert "First document body." in result

    # Check token limit
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


def test_advanced_rag_hybrid_search():
    """
    Tests that the AdvancedRAGPipeline uses both BM25 and vector search.
    """
    docs = [
        "The quick brown fox jumps over the lazy dog.", # Vector-heavy
        "apple banana orange fruit salad", # Keyword-heavy
    ]
    
    # Mock embedding functions
    def embed_many_fn(texts):
        # Return simple embeddings, make "fox" and "apple" very different
        return [[1.0, 0.0] if "fox" in t else [0.0, 1.0] for t in texts]
    def embed_one_fn(text):
        if "animal" in text:  # Simulate semantic query for "fox"
            return [1.0, 0.0]
        elif "apple" in text:
            return [0.0, 1.0]
        return [0.0, 0.0]  # Default for other queries

    pipeline = AdvancedRAGPipeline(
        embed_many_fn=embed_many_fn, 
        embed_one_fn=embed_one_fn, 
        use_faiss=False
    )
    # Use whole docs as chunks for this simple test
    simple_chunks = [{"text": doc, "doc_id": i, "chunk_id": 0} for i, doc in enumerate(docs)]
    pipeline.index_chunks(simple_chunks)

    # Query with a keyword that BM25 should find easily
    keyword_query = "apple"
    keyword_results = pipeline.retrieve(keyword_query, top_k=1)
    assert keyword_results, "Expected results for keyword query"
    assert "apple banana orange" in keyword_results[0]['text']

    # Query with a semantic phrase that vector search should find easily
    # We will make the query "fast brown animal"
    # Our mock embedding will give this a vector similar to the "fox" document
    semantic_query = "fast brown animal"
    semantic_results = pipeline.retrieve(semantic_query, top_k=1)
    assert semantic_results, "Expected results for semantic query"
    assert "quick brown fox" in semantic_results[0]['text']

