"""Tests for corpus utilities."""

from typing import List

import pytest

from src.corpus import padding as padding_module
from src.utils.tokenizer import count_tokens


@pytest.fixture
def mock_padding_sources(monkeypatch):
    filler = "Irrelevant padding sentence for context control. " * 4
    filler_tokens = count_tokens(filler)

    def fake_loader(_: List[int]):
        return [
            {"content": filler, "book_id": 1342, "tokens": filler_tokens},
            {"content": filler[::-1], "book_id": 84, "tokens": filler_tokens},
        ]

    monkeypatch.setattr(padding_module, "load_gutenberg_books", fake_loader)


def test_padding_generator_matches_target_tokens(mock_padding_sources):
    generator = padding_module.PaddingGenerator()
    target_tokens = 120
    padding_text = generator.generate_padding(target_tokens)

    assert 0 < count_tokens(padding_text) <= target_tokens


def test_pad_to_fill_percentage_adds_expected_padding(mock_padding_sources):
    generator = padding_module.PaddingGenerator()
    base_content = "Key fact about the model context window. " * 3
    base_tokens = count_tokens(base_content)

    target_total = base_tokens + 30
    max_context_tokens = target_total * 2
    fill_pct = target_total / max_context_tokens

    padded = generator.pad_to_fill_percentage(
        base_content, fill_pct=fill_pct, max_context_tokens=max_context_tokens
    )

    total_tokens = count_tokens(padded)
    assert target_total - 5 <= total_tokens <= target_total
    assert base_content.strip() in padded
