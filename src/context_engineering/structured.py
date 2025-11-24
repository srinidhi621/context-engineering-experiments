from typing import List, Dict
from src.utils.tokenizer import count_tokens, truncate_to_tokens

class StructuredContextAssembler:
    """Assembles context with a structured, XML-like format."""

    def _format_metadata(self, doc: Dict) -> str:
        """Helper to format metadata into a consistent string."""
        lines = []
        # A preferred order for common keys ensures consistency
        preferred_keys = ['model_id', 'license', 'language', 'tags', 'last_modified']
        
        for key in preferred_keys:
            if key in doc and doc[key]:
                 lines.append(f"  - {key}: {doc[key]}")

        # Add other relevant keys that might exist
        for key, value in doc.items():
            if key not in preferred_keys and key not in ['content', 'tokens', 'url', 'source_model', 'source_url', 'experiment', 'question_id', 'type', 'difficulty', 'question', 'ground_truth', 'required_docs', 'evaluation_criteria', 'keywords']:
                if value:
                    lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    def assemble(self, documents: List[Dict], max_tokens: int) -> str:
        """
        Assembles documents into a structured XML-like string with a TOC.
        
        Args:
            documents: List of dicts with document content and metadata.
            max_tokens: Maximum tokens for the final output string.
            
        Returns:
            The assembled context string.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        doc_sections = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i+1}"
            metadata_str = self._format_metadata(doc)
            content = doc.get('content', '')
            
            doc_str = (
                f'<document id="{doc_id}">\n'
                f'<metadata>\n{metadata_str}\n</metadata>\n'
                f'<content>\n{content}\n</content>\n'
                f'</document>'
            )
            doc_sections.append({
                "id": doc_id,
                "title": doc.get('model_id', f"Untitled Document {i+1}"),
                "string": doc_str,
                "tokens": count_tokens(doc_str)
            })

        # Build header and Table of Contents
        toc_items = [f"  - Document {i+1}: {section['title']} (id: {section['id']})" for i, section in enumerate(doc_sections)]
        toc_str = "<table_of_contents>\n" + "\n".join(toc_items) + "\n</table_of_contents>"
        
        introduction = "<introduction>\nThis package contains several documents. Use the table of contents and document metadata to navigate and answer the user's question.\n</introduction>"
        
        header = f"<context_package>\n{introduction}\n\n{toc_str}"
        footer = "\n</context_package>"
        
        header_tokens = count_tokens(header)
        footer_tokens = count_tokens(footer)
        
        # Assemble the final context respecting the token limit
        final_parts = [header]
        current_tokens = header_tokens + footer_tokens

        for section in doc_sections:
            # Add 2 tokens for the double newline separator
            if current_tokens + section['tokens'] + 2 <= max_tokens:
                final_parts.append(section['string'])
                current_tokens += section['tokens'] + 2
            else:
                # For simplicity and to maintain structure, we don't truncate documents.
                # If a document doesn't fit, we stop adding more.
                break
        
        final_parts.append(footer)
        
        full_context = "\n\n".join(final_parts)

        # Final check to ensure we are within the budget. This can happen if token
        # counting has minor discrepancies. This is a safeguard.
        if count_tokens(full_context) > max_tokens:
            return truncate_to_tokens(full_context, max_tokens)

        return full_context