from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        normalized = text.strip()
        if not normalized:
            return []

        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = [part.strip() for part in re.split(sentence_pattern, normalized) if part.strip()]
        if not sentences:
            return [normalized]

        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            current.append(sentence)
            if len(current) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current).strip())
                current = []
        if current:
            chunks.append(" ".join(current).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        cleaned = text.strip()
        if not cleaned:
            return []
        parts = self._split(cleaned, list(self.separators))
        merged: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer} {part}".strip() if buffer else part
            if buffer and len(candidate) <= self.chunk_size:
                buffer = candidate
            elif buffer and len(candidate) > self.chunk_size:
                merged.append(buffer)
                buffer = part
            else:
                buffer = part
        if buffer:
            merged.append(buffer)
        return merged

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        pieces = [piece.strip() for piece in current_text.split(separator) if piece.strip()]
        if len(pieces) <= 1:
            return self._split(current_text, remaining_separators[1:])

        result: list[str] = []
        current: list[str] = []
        for piece in pieces:
            if not current:
                current = [piece]
                continue
            candidate = f"{current[-1]}{separator}{piece}".strip()
            if len(candidate) <= self.chunk_size:
                current[-1] = candidate
            else:
                result.append(current[-1])
                current = [piece]
        if current:
            result.append(current[-1])

        final: list[str] = []
        for item in result:
            if len(item) <= self.chunk_size:
                final.append(item)
            else:
                final.extend(self._split(item, remaining_separators[1:]))
        return final


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if not text:
            return {
                "fixed_size": {"count": 0, "avg_length": 0.0, "chunks": []},
                "by_sentences": {"count": 0, "avg_length": 0.0, "chunks": []},
                "recursive": {"count": 0, "avg_length": 0.0, "chunks": []},
            }

        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, min(50, chunk_size // 5))).chunk(text)
        sentences = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        return {
            "fixed_size": {
                "count": len(fixed),
                "avg_length": sum(len(c) for c in fixed) / len(fixed) if fixed else 0.0,
                "chunks": fixed,
            },
            "by_sentences": {
                "count": len(sentences),
                "avg_length": sum(len(c) for c in sentences) / len(sentences) if sentences else 0.0,
                "chunks": sentences,
            },
            "recursive": {
                "count": len(recursive),
                "avg_length": sum(len(c) for c in recursive) / len(recursive) if recursive else 0.0,
                "chunks": recursive,
            },
        }
