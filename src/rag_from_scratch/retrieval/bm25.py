"""A small, in-memory BM25 retriever for the project's chunks."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rag_from_scratch.models import Chunk


@dataclass(frozen=True)
class SearchResult:
    """A chunk and its BM25 relevance score."""

    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase text, remove accents and split on non-alphanumeric characters.

    This intentionally does not stem words or remove stop words, keeping the
    first BM25 baseline easy to inspect and reproduce.
    """
    normalized = unicodedata.normalize("NFKD", text.casefold())
    tokens: list[str] = []
    current: list[str] = []

    for character in normalized:
        if unicodedata.combining(character):
            continue
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()

    if current:
        tokens.append("".join(current))
    return tokens


class BM25Retriever:
    """Rank chunks with the Okapi BM25 scoring function.

    The index is kept in memory, which is sufficient for the current small
    corpus. The IDF term uses ``log(1 + (N - df + 0.5) / (df + 0.5))`` so
    scores stay non-negative. With ``include_metadata=True``, the source
    filename (without extension) and section title are appended to the indexed
    text and receive the same weight as chunk text.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        include_metadata: bool = False,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.chunks = tuple(chunks)
        self.k1 = k1
        self.b = b
        self.include_metadata = include_metadata
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for chunk in self.chunks:
            indexed_text = chunk.text
            if include_metadata:
                metadata = [Path(chunk.source_path).stem, chunk.section or ""]
                indexed_text = " ".join(part for part in (indexed_text, *metadata) if part)
            frequencies = Counter(tokenize(indexed_text))
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(sum(frequencies.values()))
            document_frequencies.update(frequencies.keys())

        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_count = len(self.chunks)
        self._inverse_document_frequencies = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def rank(self, query: str) -> list[SearchResult]:
        """Return all chunks with a positive score, highest score first.

        Ties retain the input chunk order. Query terms are counted once, so
        repeating a word in the question does not multiply its contribution.
        """
        query_terms = sorted(set(tokenize(query)))
        if not query_terms or self._average_document_length == 0:
            return []

        scored: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            document_length = self._document_lengths[index]
            score = 0.0

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue

                length_normalization = 1 - self.b + self.b * (
                    document_length / self._average_document_length
                )
                denominator = term_frequency + self.k1 * length_normalization
                score += self._inverse_document_frequencies[term] * (
                    term_frequency * (self.k1 + 1) / denominator
                )

            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [SearchResult(chunk=self.chunks[index], score=score) for index, score in scored]

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to ``top_k`` matching chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return self.rank(query)[:top_k]
