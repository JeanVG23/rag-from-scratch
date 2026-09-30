"""Unit tests for BM25, Dense and Hybrid RRF retrieval."""

from __future__ import annotations

from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval.bm25 import BM25Retriever, SearchResult
from rag_from_scratch.retrieval.hybrid import reciprocal_rank_fusion, HybridRetriever


def _make_dummy_chunk(chunk_id: str, text: str, source_id: str = "src1") -> Chunk:
    words = text.split()
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_path=f"{source_id}.md",
        unit_id=f"{source_id}_u1",
        text=text,
        chunk_index=0,
        word_start=0,
        word_end=len(words),
        page=None,
        section=None,
    )


def test_reciprocal_rank_fusion_basic():
    c1 = _make_dummy_chunk("c1", "premier chunk")
    c2 = _make_dummy_chunk("c2", "deuxieme chunk")
    c3 = _make_dummy_chunk("c3", "troisieme chunk")

    # List 1: c1 rank 1, c2 rank 2
    # List 2: c2 rank 1, c3 rank 2
    list1 = [SearchResult(chunk=c1, score=10.0), SearchResult(chunk=c2, score=5.0)]
    list2 = [SearchResult(chunk=c2, score=0.9), SearchResult(chunk=c3, score=0.7)]

    # c2 is rank 2 in list1, rank 1 in list2 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.0325
    # c1 is rank 1 in list1, absent in list2 -> 1/61 ~ 0.01639
    # c3 is rank 2 in list2, absent in list1 -> 1/62 ~ 0.01613
    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(fused) == 3
    assert fused[0].chunk.chunk_id == "c2"
    assert fused[1].chunk.chunk_id == "c1"
    assert fused[2].chunk.chunk_id == "c3"


def test_reciprocal_rank_fusion_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_hybrid_retriever_mock():
    class DummyRetriever:
        def __init__(self, results):
            self.results = results

        def rank(self, query: str):
            return self.results

        def search(self, query: str, *, top_k: int = 5):
            return self.results[:top_k]

    c1 = _make_dummy_chunk("c1", "text one")
    c2 = _make_dummy_chunk("c2", "text two")

    bm25_mock = DummyRetriever([SearchResult(chunk=c1, score=5.0)])
    dense_mock = DummyRetriever([SearchResult(chunk=c2, score=0.8)])

    hybrid = HybridRetriever(
        chunks=[c1, c2],
        bm25=bm25_mock,
        dense=dense_mock,
        k=60,
    )

    res = hybrid.search("query", top_k=2)
    assert len(res) == 2
