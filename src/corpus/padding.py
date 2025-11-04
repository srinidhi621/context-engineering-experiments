"""Padding generation helpers for context fill control."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Iterable, List, Sequence, Tuple

from src.corpus.loaders import load_gutenberg_books
from src.utils.tokenizer import count_tokens, truncate_to_tokens


DEFAULT_PADDING_BOOK_IDS: Tuple[int, ...] = (1342, 84, 98, 1661)


@dataclass
class PaddingGenerator:
    """Generate irrelevant padding text from public-domain sources."""

    book_ids: Sequence[int] = DEFAULT_PADDING_BOOK_IDS
    separator: str = "\n\n"
    _sources: List[Tuple[str, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        books = load_gutenberg_books(list(self.book_ids))
        sources: List[Tuple[str, int]] = []
        for book in books:
            content = (book.get("content") or "").strip()
            if not content:
                continue
            tokens = count_tokens(content)
            if tokens == 0:
                continue
            sources.append((content, tokens))

        if not sources:
            raise RuntimeError("No padding sources available from Project Gutenberg.")

        self._sources = sources

    def generate_padding(self, target_tokens: int) -> str:
        """
        Produce padding text with the requested token count.

        Args:
            target_tokens: Desired number of tokens in the padding text.

        Returns:
            Padding string truncated to the requested size.
        """
        if target_tokens <= 0:
            return ""

        assembled_parts: List[str] = []
        accumulated_tokens = 0

        for content, tokens in cycle(self._sources):
            assembled_parts.append(content)
            accumulated_tokens += tokens
            if accumulated_tokens >= target_tokens:
                break

        combined = self.separator.join(assembled_parts)
        return truncate_to_tokens(combined, target_tokens)

    def pad_to_fill_percentage(
        self,
        content: str,
        fill_pct: float,
        max_context_tokens: int = 1_000_000,
    ) -> str:
        """
        Pad content to reach the target fill percentage of a context window.

        Args:
            content: Relevant content to include before padding.
            fill_pct: Desired context fill percentage (0 < fill_pct <= 1).
            max_context_tokens: Maximum size of the context window.

        Returns:
            Content combined with padding to match the requested fill.
        """
        if not 0 < fill_pct <= 1:
            raise ValueError("fill_pct must be within (0, 1].")
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive.")

        target_total = int(max_context_tokens * fill_pct)
        if target_total <= 0:
            raise ValueError("Target context size must be positive.")

        content_tokens = count_tokens(content)
        if content_tokens >= target_total:
            return truncate_to_tokens(content, target_total)

        padding_needed = target_total - content_tokens
        padding = self.generate_padding(padding_needed)

        if not padding:
            return content

        return self.separator.join(
            segment for segment in (content, padding) if segment.strip()
        )
