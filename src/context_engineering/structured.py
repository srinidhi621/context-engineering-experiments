"""Engineered context assembler with metadata and table of contents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.utils.tokenizer import count_tokens, truncate_to_tokens


@dataclass
class StructuredContextAssembler:
    """
    Assemble contexts with metadata blocks, table of contents, and
    per-document sections. Designed to reduce attention diffusion by
    clearly scoping each document.
    """

    context_title: str = "Context Package"
    include_toc: bool = True
    include_metadata: bool = True

    def assemble(
        self,
        documents: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not documents:
            raise ValueError("documents cannot be empty")

        sections: list[str] = []
        total_tokens = 0

        preamble = self._build_preamble(documents)
        preamble_tokens = count_tokens(preamble)

        if preamble_tokens >= max_tokens:
            return truncate_to_tokens(preamble, max_tokens)

        sections.append(preamble)
        total_tokens += preamble_tokens

        if self.include_toc:
            toc = self._build_toc(documents)
            toc_tokens = count_tokens(toc)
            if total_tokens + toc_tokens >= max_tokens:
                sections.append(truncate_to_tokens(toc, max_tokens - total_tokens))
                return "\n\n".join(sections)
            sections.append(toc)
            total_tokens += toc_tokens

        for idx, doc in enumerate(documents, start=1):
            section = self._build_section(idx, doc)
            section_tokens = count_tokens(section)
            if total_tokens + section_tokens <= max_tokens:
                sections.append(section)
                total_tokens += section_tokens
                continue

            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break
            truncated = truncate_to_tokens(section, remaining)
            if truncated.strip():
                sections.append(truncated)
                total_tokens = max_tokens
            break

        return "\n\n".join(sections)

    def _build_preamble(self, documents: Sequence[Mapping[str, str]]) -> str:
        lines = [
            f"# {self.context_title}",
            f"Total Documents: {len(documents)}",
        ]
        if self.include_metadata:
            domains = {doc.get("type", "document") for doc in documents}
            sources = {doc.get("source", "unknown") for doc in documents}
            lines.append(f"Domains: {', '.join(sorted(domains))}")
            lines.append(f"Sources: {', '.join(sorted(sources))}")
        return "\n".join(lines)

    def _build_toc(self, documents: Sequence[Mapping[str, str]]) -> str:
        toc_lines = ["## Table of Contents"]
        for idx, doc in enumerate(documents, start=1):
            title = self._doc_title(doc, idx)
            toc_lines.append(f"{idx}. {title}")
        return "\n".join(toc_lines)

    def _build_section(self, idx: int, doc: Mapping[str, str]) -> str:
        title = self._doc_title(doc, idx)
        header_lines = [f"## Document {idx}: {title}"]

        if self.include_metadata:
            meta_lines = []
            if doc.get("url"):
                meta_lines.append(f"- Source: {doc['url']}")
            if doc.get("last_modified"):
                meta_lines.append(f"- Last Modified: {doc['last_modified']}")
            if doc.get("tokens"):
                meta_lines.append(f"- Tokens: {doc['tokens']:,}")
            if doc.get("tags"):
                meta_lines.append(f"- Tags: {', '.join(doc['tags'])}")
            if meta_lines:
                header_lines.extend(meta_lines)

        body = doc.get("content", "").strip()
        section = "\n".join(header_lines) + "\n\n" + body
        return section

    @staticmethod
    def _doc_title(doc: Mapping[str, str], idx: int) -> str:
        return (
            doc.get("title")
            or doc.get("model_id")
            or doc.get("dataset_id")
            or f"Document {idx}"
        )
