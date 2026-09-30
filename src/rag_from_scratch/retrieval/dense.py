"""Dense vector retriever using Ollama embeddings and cosine similarity."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from rag_from_scratch.embeddings import embed_texts
from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval.bm25 import SearchResult

try:
    import numpy as np
except ImportError:
    np = None


def _unit_vector_py(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


class DenseRetriever:
    """Rank chunks using dense vector embeddings and cosine similarity.

    Embeddings can be pre-loaded from a JSONL index file or computed dynamically
    via Ollama's local embedding endpoint. Document vectors are pre-normalized to
    unit length so cosine scoring is a simple dot product.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        model: str = "bge-m3",
        index_path: str | Path | None = None,
        host: str | None = None,
        auto_index: bool = True,
    ) -> None:
        self.model = model
        self.host = host
        self.chunks: list[Chunk] = list(chunks)
        self._chunk_by_id = {c.chunk_id: c for c in self.chunks}

        embeddings_map: dict[str, list[float]] = {}
        if index_path is not None:
            path = Path(index_path)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            embeddings_map[record["chunk_id"]] = record["embedding"]

        # Check for missing chunks and compute if allowed
        missing_chunks = [c for c in self.chunks if c.chunk_id not in embeddings_map]
        if missing_chunks:
            if not auto_index:
                raise ValueError(
                    f"{len(missing_chunks)} chunks are missing embeddings in index "
                    f"{index_path} and auto_index is False"
                )
            texts = [c.text for c in missing_chunks]
            new_embeddings = embed_texts(self.model, texts, host=self.host)
            for chunk, emb in zip(missing_chunks, new_embeddings):
                embeddings_map[chunk.chunk_id] = emb

            if index_path is not None:
                path = Path(index_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    for chunk in self.chunks:
                        if chunk.chunk_id in embeddings_map:
                            f.write(
                                json.dumps(
                                    {
                                        "model": self.model,
                                        "chunk_id": chunk.chunk_id,
                                        "embedding": embeddings_map[chunk.chunk_id],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

        # Store unit vectors aligned with self.chunks
        self._doc_vectors: list[list[float]] = []
        for c in self.chunks:
            emb = embeddings_map[c.chunk_id]
            self._doc_vectors.append(_unit_vector_py(emb))

        if np is not None and self._doc_vectors:
            self._matrix = np.array(self._doc_vectors, dtype=np.float32)
        else:
            self._matrix = None

    def rank(self, query: str) -> list[SearchResult]:
        """Rank all chunks by cosine similarity with the query vector."""
        if not self.chunks or not query.strip():
            return []

        query_emb = embed_texts(self.model, [query], host=self.host)[0]

        if self._matrix is not None:
            q_vec = np.array(query_emb, dtype=np.float32)
            norm = float(np.linalg.norm(q_vec))
            if norm > 0:
                q_vec = q_vec / norm
            scores = self._matrix @ q_vec
            ranked_indices = np.argsort(-scores)
            return [
                SearchResult(chunk=self.chunks[idx], score=float(scores[idx]))
                for idx in ranked_indices
            ]

        # Pure Python fallback
        q_unit = _unit_vector_py(query_emb)
        scored: list[tuple[Chunk, float]] = []
        for chunk, doc_unit in zip(self.chunks, self._doc_vectors):
            dot = sum(a * b for a, b in zip(q_unit, doc_unit))
            scored.append((chunk, dot))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [SearchResult(chunk=ch, score=sc) for ch, sc in scored]

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to ``top_k`` matching chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return self.rank(query)[:top_k]
