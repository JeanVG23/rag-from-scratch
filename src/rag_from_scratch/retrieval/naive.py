"""Naive word-overlap retriever used as the project's initial baseline (v0)."""

from __future__ import annotations

from collections.abc import Iterable

from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval.bm25 import SearchResult, tokenize


class NaiveOverlapRetriever:
    """Rank chunks by raw word intersection count with the query.

    Does not use IDF weighting or document length normalization. Serves as
    the minimal historical reference (v0) to measure the value of BM25 and embeddings.
    """

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        self._token_sets = [set(tokenize(c.text)) for c in self.chunks]

    def rank(self, query: str) -> list[SearchResult]:
        """Rank chunks by number of common terms with the query."""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scored: list[SearchResult] = []
        for idx, tokens in enumerate(self._token_sets):
            overlap = len(query_tokens & tokens)
            if overlap > 0:
                scored.append(SearchResult(chunk=self.chunks[idx], score=float(overlap)))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to ``top_k`` matching chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return self.rank(query)[:top_k]
