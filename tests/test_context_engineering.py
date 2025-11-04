"""Tests for context engineering strategies."""

import pytest

from src.context_engineering.naive import NaiveContextAssembler
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
