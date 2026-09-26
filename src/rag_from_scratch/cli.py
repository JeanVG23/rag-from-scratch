"""Command-line entry point for building local RAG artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_from_scratch.chunking import chunk_units, write_chunks_jsonl
from rag_from_scratch.generation import GenerationError, generate_answer, prepare_context
from rag_from_scratch.ingestion import load_ia04_units, write_units_jsonl
from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval import BM25Retriever


def _build_chunks(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    units = load_ia04_units(data_dir)
    chunks = chunk_units(
        units,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )

    silver_path = data_dir / "silver" / "ia04_units.jsonl"
    gold_path = data_dir / "gold" / "ia04_chunks.jsonl"
    write_units_jsonl(units, silver_path)
    write_chunks_jsonl(chunks, gold_path)

    pdf_units = sum(unit.page is not None for unit in units)
    markdown_units = len(units) - pdf_units
    print(f"Text units: {len(units)} ({markdown_units} Markdown sections, {pdf_units} PDF pages)")
    print(f"Chunks: {len(chunks)} (max {args.max_words} words, overlap {args.overlap_words})")
    print(f"Extracted units: {silver_path}")
    print(f"Chunks: {gold_path}")
    return 0


def _load_chunks(path: Path) -> list[Chunk]:
    """Read chunks written by ``build-chunks`` from a JSONL file."""
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                chunks.append(Chunk(**value))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"invalid chunk JSON on line {line_number}: {error}") from error
    return chunks


def _search(args: argparse.Namespace) -> int:
    try:
        chunks = _load_chunks(args.chunks)
        retriever = BM25Retriever(
            chunks,
            k1=args.k1,
            b=args.b,
            include_metadata=args.include_metadata,
        )
        results = retriever.search(" ".join(args.question), top_k=args.top_k)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if not results:
        print("Aucun passage ne correspond aux termes de la question.")
        return 0

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        location = []
        if chunk.page is not None:
            location.append(f"page {chunk.page}")
        if chunk.section:
            location.append(f"section {chunk.section}")
        citation = f" ({', '.join(location)})" if location else ""
        print(
            f"[{rank}] score={result.score:.4f} "
            f"{chunk.source_path} [{chunk.source_id}]{citation}"
        )
        print(chunk.text)
        if rank < len(results):
            print()
    return 0


def _ask(args: argparse.Namespace) -> int:
    model = args.model or os.environ.get("OLLAMA_CHAT_MODEL")
    if not model:
        print(
            "Error: specify a chat model with --model or set OLLAMA_CHAT_MODEL.",
            file=sys.stderr,
        )
        return 2

    try:
        chunks = _load_chunks(args.chunks)
        retriever = BM25Retriever(
            chunks,
            k1=args.k1,
            b=args.b,
            include_metadata=True,
        )
        results = retriever.search(" ".join(args.question), top_k=args.top_k)
        context = prepare_context(results)
        answer = generate_answer(
            " ".join(args.question),
            context,
            model=model,
            host=args.host,
            timeout=args.timeout,
            think=args.think,
            num_predict=args.num_predict,
        )
    except (OSError, ValueError, GenerationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(answer)
    if context.citations:
        print("\nRéférences fournies au modèle :")
        for citation in context.citations:
            print(
                f"[{citation.label}] {citation.source_path} "
                f"({citation.locator()}; chunk {citation.chunk_id})"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="rag-from-scratch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-chunks", help="extract IA04 and write chunk JSONL")
    build.add_argument("--data-dir", default="data")
    build.add_argument("--max-words", type=int, default=400)
    build.add_argument("--overlap-words", type=int, default=50)
    build.set_defaults(handler=_build_chunks)

    search = subparsers.add_parser("search", help="search IA04 chunks with BM25")
    search.add_argument("question", nargs="+", help="question or search terms")
    search.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--k1", type=float, default=1.5)
    search.add_argument("--b", type=float, default=0.75)
    search.add_argument(
        "--content-only",
        action="store_false",
        dest="include_metadata",
        help="rank chunk text without the source filename and section title",
    )
    search.set_defaults(include_metadata=True)
    search.set_defaults(handler=_search)

    ask = subparsers.add_parser("ask", help="answer a question with BM25 and a local Ollama model")
    ask.add_argument("question", nargs="+", help="question about IA04")
    ask.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    ask.add_argument("--model", help="Ollama chat model (or set OLLAMA_CHAT_MODEL)")
    ask.add_argument("--host", help="Ollama host (defaults to OLLAMA_HOST or localhost)")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--k1", type=float, default=1.5)
    ask.add_argument("--b", type=float, default=0.75)
    ask.add_argument("--timeout", type=float, default=300)
    ask.add_argument("--num-predict", type=int, default=384)
    ask.add_argument("--think", action="store_true", help="enable model reasoning when supported")
    ask.set_defaults(handler=_ask)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
