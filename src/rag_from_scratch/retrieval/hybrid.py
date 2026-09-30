"""Hybrid retrieval combining BM25 and Dense vector search via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval.bm25 import BM25Retriever, SearchResult
from rag_from_scratch.retrieval.dense import DenseRetriever


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[SearchResult]:
    """Fuse multiple ranked SearchResult lists using Reciprocal Rank Fusion (RRF).

    Formula for each document d:
        RRF_Score(d) = sum_{m in models} (weight_m / (k + rank_m(d)))
    where rank_m(d) is 1-indexed.

    Args:
        result_lists: Sequences of SearchResult from different retrievers.
        k: Smoothing constant (default 60, standard in IR literature).
        weights: Optional multiplier per result list (defaults to 1.0 each).

    Returns:
        Consolidated, deduplicated list of SearchResult sorted by descending RRF score.
    """
    if k <= 0:
        raise ValueError("RRF smoothing constant k must be greater than zero")
    if not result_lists:
        return []

    if weights is None:
        effective_weights = [1.0] * len(result_lists)
    else:
        if len(weights) != len(result_lists):
            raise ValueError("Length of weights must match length of result_lists")
        effective_weights = list(weights)

    chunk_map: dict[str, Chunk] = {}
    rrf_scores: dict[str, float] = {}

    for results, weight in zip(result_lists, effective_weights):
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = item.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (weight / (k + rank))

    fused = [SearchResult(chunk=chunk_map[cid], score=score) for cid, score in rrf_scores.items()]
    fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return fused


class HybridRetriever:
    """Combines BM25 and dense vector search via Reciprocal Rank Fusion (RRF).

    Provides the best of both worlds:
    - BM25 handles exact identifiers, Go syntax, and rare keywords.
    - Dense search handles conceptual and semantic queries with synonym mismatch.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        dense_model: str = "bge-m3",
        dense_index_path: str | Path | None = "data/indexes/bge-m3.jsonl",
        host: str | None = None,
        k: int = 60,
        weight_bm25: float = 1.0,
        weight_dense: float = 1.0,
        candidate_multiplier: int = 4,
    ) -> None:
        self.chunks = list(chunks)
        self.k = k
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense
        self.candidate_multiplier = max(1, candidate_multiplier)

        self.bm25 = (
            bm25
            if bm25 is not None
            else BM25Retriever(self.chunks, include_metadata=True)
        )
        self.dense = (
            dense
            if dense is not None
            else DenseRetriever(
                self.chunks,
                model=dense_model,
                index_path=dense_index_path,
                host=host,
            )
        )

    def rank(self, query: str, *, candidate_count: int | None = None) -> list[SearchResult]:
        """Rank chunks by fusing BM25 and Dense search via RRF."""
        if not self.chunks or not query.strip():
            return []

        if candidate_count is None:
            candidate_count = max(20, len(self.chunks))

        bm25_candidates = self.bm25.rank(query)[:candidate_count]
        dense_candidates = self.dense.rank(query)[:candidate_count]

        return reciprocal_rank_fusion(
            [bm25_candidates, dense_candidates],
            k=self.k,
            weights=[self.weight_bm25, self.weight_dense],
        )

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to ``top_k`` matching chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        candidate_count = max(top_k * self.candidate_multiplier, 20)
        return self.rank(query, candidate_count=candidate_count)[:top_k]
