"""Unit tests for high-level RAGPipeline and Naive retriever."""

from __future__ import annotations

from rag_from_scratch.models import Chunk
from rag_from_scratch.pipeline import RAGPipeline
from rag_from_scratch.retrieval.naive import NaiveOverlapRetriever


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


def test_naive_overlap_retriever():
    c1 = _make_dummy_chunk("c1", "agent autonome environnement")
    c2 = _make_dummy_chunk("c2", "goroutine channel synchronisation")

    retriever = NaiveOverlapRetriever([c1, c2])
    results = retriever.search("agent autonome", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score == 2.0


def test_rag_pipeline_abstention_on_empty():
    class EmptyRetriever:
        def search(self, query: str, *, top_k: int = 5):
            return []

    pipeline = RAGPipeline(retriever=EmptyRetriever(), model="dummy")
    response = pipeline.query("Question sans résultat")
    assert response.is_abstained
    assert "Je ne peux pas le déterminer" in response.answer
