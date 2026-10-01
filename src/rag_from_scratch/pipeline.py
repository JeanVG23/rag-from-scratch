"""High-level RAG Pipeline coordinating retrieval, context preparation, and generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_from_scratch.generation import (
    ABSTENTION,
    Citation,
    PreparedContext,
    generate_answer,
    prepare_context,
)
from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, SearchResult
from rag_from_scratch.retrieval_metrics import load_jsonl


@dataclass(frozen=True)
class RAGResponse:
    """Consolidated answer and evidence from the RAG pipeline."""

    question: str
    answer: str
    context: PreparedContext
    search_results: list[SearchResult]

    @property
    def citations(self) -> tuple[Citation, ...]:
        return self.context.citations

    @property
    def is_abstained(self) -> bool:
        return ABSTENTION in self.answer or self.answer.strip() == ABSTENTION


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Coordinates retrieval (Hybrid RRF, BM25, or Dense), deterministic context
    assembly with citations, and local Ollama answer generation.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        model: str | None = None,
        host: str | None = None,
        top_k: int = 5,
        timeout: float = 300.0,
        think: bool = False,
        num_predict: int = 384,
    ) -> None:
        self.retriever = retriever
        self.model = model or os.environ.get("OLLAMA_CHAT_MODEL", "qwen3.5:4b")
        self.host = host
        self.top_k = top_k
        self.timeout = timeout
        self.think = think
        self.num_predict = num_predict

    @classmethod
    def from_chunks(
        cls,
        chunks_path: str | Path = "data/gold/ia04_chunks.jsonl",
        *,
        retriever_type: str = "hybrid",
        dense_model: str = "bge-m3",
        index_path: str | Path | None = "data/indexes/bge-m3.jsonl",
        model: str = "qwen3.5:4b",
        top_k: int = 5,
        host: str | None = None,
        rrf_k: int = 60,
    ) -> RAGPipeline:
        """Instantiate a ready-to-query pipeline directly from a chunk JSONL file."""
        records = load_jsonl(Path(chunks_path))
        chunks = [Chunk(**r) for r in records]

        method = retriever_type.lower()
        if method == "dense":
            retriever = DenseRetriever(
                chunks,
                model=dense_model,
                index_path=index_path,
                host=host,
            )
        elif method == "hybrid":
            retriever = HybridRetriever(
                chunks,
                dense_model=dense_model,
                dense_index_path=index_path,
                host=host,
                k=rrf_k,
            )
        else:
            retriever = BM25Retriever(chunks, include_metadata=True)

        return cls(retriever=retriever, model=model, host=host, top_k=top_k)

    def query(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        """Retrieve relevant context and generate a grounded, cited answer."""
        k = top_k or self.top_k
        results = self.retriever.search(question, top_k=k)
        context = prepare_context(results)
        answer = generate_answer(
            question,
            context,
            model=self.model,
            host=self.host,
            timeout=self.timeout,
            think=self.think,
            num_predict=self.num_predict,
        )
        return RAGResponse(
            question=question,
            answer=answer,
            context=context,
            search_results=results,
        )
