"""Naive context assembly utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from src.utils.tokenizer import count_tokens, truncate_to_tokens


class NaiveContextAssembler:
    """Simple sequential concatenation of documents within a token budget."""

    def __init__(self, separator: str = "\n\n") -> None:
        self.separator = separator

    def assemble(
        self,
        documents: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> str:
        """
        Concatenate document content until the token limit is reached.

        Args:
            documents: Sequence of dict-like objects with a ``content`` field.
            max_tokens: Token budget for the assembled context.

        Returns:
            A string containing concatenated document contents trimmed to fit the limit.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        context_parts: List[str] = []
        total_tokens = 0

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue

            content_tokens = count_tokens(content)
            if total_tokens + content_tokens <= max_tokens:
                context_parts.append(content)
                total_tokens += content_tokens
                continue

            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break

            truncated = truncate_to_tokens(content, remaining)
            if truncated.strip():
                context_parts.append(truncated)
                total_tokens += count_tokens(truncated)
            break

        return self.separator.join(context_parts)
