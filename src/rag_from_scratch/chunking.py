"""Simple, inspectable text chunking for the first RAG iteration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rag_from_scratch.models import Chunk, TextUnit


def chunk_unit(
    unit: TextUnit,
    *,
    max_words: int = 400,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Split one page or section into overlapping word windows.

    A chunk never crosses a ``TextUnit``, so its page or section reference stays
    unambiguous. Word offsets are zero-based, with ``word_end`` exclusive.
    """
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between zero and max_words - 1")

    spans = [match.span() for match in re.finditer(r"\S+", unit.text)]
    if not spans:
        return []

    chunks: list[Chunk] = []
    step = max_words - overlap_words

    for chunk_index, word_start in enumerate(range(0, len(spans), step)):
        word_end = min(word_start + max_words, len(spans))
        text_start = spans[word_start][0]
        text_end = spans[word_end - 1][1]
        chunk_text = unit.text[text_start:text_end].strip()

        identity = "\x1f".join(
            (
                unit.source_id,
                unit.source_path,
                unit.unit_id,
                str(chunk_index),
                str(word_start),
                str(word_end),
            )
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_id=unit.source_id,
                source_path=unit.source_path,
                unit_id=unit.unit_id,
                text=chunk_text,
                chunk_index=chunk_index,
                word_start=word_start,
                word_end=word_end,
                page=unit.page,
                section=unit.section,
            )
        )

        if word_end == len(spans):
            break

    return chunks


def chunk_units(
    units: Iterable[TextUnit],
    *,
    max_words: int = 400,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Chunk multiple extracted pages or sections in input order."""
    chunks: list[Chunk] = []
    for unit in units:
        chunks.extend(
            chunk_unit(
                unit,
                max_words=max_words,
                overlap_words=overlap_words,
            )
        )
    return chunks


def write_chunks_jsonl(chunks: Iterable[Chunk], output_path: str | Path) -> None:
    """Write chunks as UTF-8 JSON Lines, creating the output directory if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
