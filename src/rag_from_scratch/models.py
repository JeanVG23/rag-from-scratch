"""Small data structures shared by the RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextUnit:
    """An extracted piece of one source, such as a PDF page or Markdown section."""

    source_id: str
    source_path: str
    unit_id: str
    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A searchable passage that keeps its source location."""

    chunk_id: str
    source_id: str
    source_path: str
    unit_id: str
    text: str
    chunk_index: int
    word_start: int
    word_end: int
    page: int | None = None
    section: str | None = None
