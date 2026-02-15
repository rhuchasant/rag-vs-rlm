"""Chunking strategies for RAG - the chunking approach significantly affects retrieval quality."""

from typing import List, Optional
import re


def chunk_by_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into overlapping chunks by approximate token count.
    Uses ~4 chars per token heuristic.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    target_size = chunk_size * 4  # ~4 chars per token

    for word in words:
        word_len = len(word) + 1
        if current_size + word_len > target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep last N words
            overlap_words = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_words:]
            current_size = sum(len(w) + 1 for w in current_chunk)
        current_chunk.append(word)
        current_size += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_by_semantic_boundaries(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Split by semantic boundaries (paragraphs, sections) to preserve context.
    Better for structured documents like data layouts.
    """
    if separators is None:
        separators = ["\n\n\n", "\n\n", "\n", ". ", " "]

    def _split(text: str, sep: str) -> List[str]:
        if sep == " ":
            return text.split()
        return text.split(sep)

    def _recursive_split(text: str, seps: List[str]) -> List[str]:
        if not text.strip():
            return []
        if not seps:
            return [text] if text.strip() else []

        sep = seps[0]
        parts = _split(text, sep)
        if sep != " ":
            parts = [p + (sep if i < len(parts) - 1 else "") 
                     for i, p in enumerate(parts)]

        chunks = []
        current = []
        current_tokens = 0
        target = chunk_size * 4

        overlap_parts = max(1, min(overlap, len(parts) // 2))  # Overlap in "parts"
        for part in parts:
            part_len = len(part)
            if current_tokens + part_len > target and current:
                chunks.append("".join(current))
                current = current[-overlap_parts:] if len(current) > overlap_parts else []
                current_tokens = sum(len(p) for p in current)
            current.append(part)
            current_tokens += part_len

        if current:
            chunks.append("".join(current))

        return chunks

    return _recursive_split(text, separators)


def chunk_excel_tabs(tab_contents: dict) -> List[tuple]:
    """
    Chunk multi-tab Excel content. Returns list of (tab_name, content) tuples.
    Each tab is a separate chunk - RAG typically retrieves top-k, which may
    miss later tabs (the real-world failure case).
    """
    return [(name, content) for name, content in tab_contents.items()]


def chunk_by_tabs(tab_contents: dict) -> List[str]:
    """
    One chunk per tab - fair comparison when document has explicit tab structure.
    Each chunk is prefixed with tab name for retrieval.
    """
    return [f"## {name}\n{content}" for name, content in tab_contents.items()]
