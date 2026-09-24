"""Command-line entry point for building local RAG artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_from_scratch.chunking import chunk_units, write_chunks_jsonl
from rag_from_scratch.ingestion import load_ia04_units, write_units_jsonl


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


def main() -> int:
    parser = argparse.ArgumentParser(prog="rag-from-scratch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-chunks", help="extract IA04 and write chunk JSONL")
    build.add_argument("--data-dir", default="data")
    build.add_argument("--max-words", type=int, default=400)
    build.add_argument("--overlap-words", type=int, default=50)
    build.set_defaults(handler=_build_chunks)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
