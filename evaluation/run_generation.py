"""Generate private review responses for the IA04 answer-quality evaluation set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_from_scratch.generation import GenerationError, generate_answer, prepare_context
from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval import BM25Retriever
from rag_from_scratch.retrieval_metrics import load_jsonl


def run_generation_review(
    *,
    chunks_path: Path,
    questions_path: Path,
    output_path: Path,
    model: str,
    host: str | None = None,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    timeout: float = 300,
    limit: int | None = None,
    overwrite: bool = False,
    think: bool = False,
    num_predict: int = 384,
) -> int:
    """Generate answers and save a local JSONL worksheet for human scoring."""
    chunks = [Chunk(**record) for record in load_jsonl(chunks_path)]
    questions = load_jsonl(questions_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    if not questions:
        raise ValueError(f"No questions found in {questions_path}")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if num_predict <= 0:
        raise ValueError("num_predict must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if limit is not None:
        questions = questions[:limit]
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"{output_path} already exists; choose another --output or pass --overwrite "
            "to replace it"
        )

    retriever = BM25Retriever(chunks, k1=k1, b=b, include_metadata=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with output_path.open("w", encoding="utf-8") as output:
        for number, question in enumerate(questions, start=1):
            results = retriever.search(question["question"], top_k=top_k)
            context = prepare_context(results)
            answer = generate_answer(
                question["question"],
                context,
                model=model,
                host=host,
                timeout=timeout,
                think=think,
                num_predict=num_predict,
            )
            record: dict[str, Any] = {
                "generated_at": generated_at,
                "model": model,
                "generation": {
                    "temperature": 0,
                    "think": think,
                    "num_predict": num_predict,
                },
                "retrieval": {
                    "method": "BM25",
                    "top_k": top_k,
                    "k1": k1,
                    "b": b,
                    "include_metadata": True,
                },
                "question_id": question["id"],
                "question": question["question"],
                "answerable": question.get("answerable"),
                "expected_abstention": question.get("expected_abstention"),
                "abstention_reason": question.get("abstention_reason"),
                "answer": answer,
                "expected_answer_points": [
                    {**point, "coverage_score": None}
                    for point in question.get("expected_answer_points", [])
                ],
                "retrieved_passages": [
                    {
                        "citation": passage.citation.label,
                        "source_path": passage.citation.source_path,
                        "source_id": passage.citation.source_id,
                        "chunk_id": passage.citation.chunk_id,
                        "page": passage.citation.page,
                        "section": passage.citation.section,
                        "score": passage.score,
                        "text": passage.text,
                    }
                    for passage in context.passages
                ],
                "scores": {
                    "groundedness_0_to_2": None,
                    "citation_quality_0_to_2": None,
                    "abstention_0_or_1": None,
                    "review_notes": "",
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{number}/{len(questions)}] {question['id']} générée")

    print(f"Fiche de notation locale : {output_path}")
    return len(questions)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate IA04 answers into a private human-review JSONL worksheet"
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/private/ia04_questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/generation_review.jsonl"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_CHAT_MODEL"),
        help="Ollama chat model (defaults to OLLAMA_CHAT_MODEL)",
    )
    parser.add_argument("--host", help="Ollama host (defaults to OLLAMA_HOST or localhost)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--think", action="store_true", help="enable model reasoning when supported")
    parser.add_argument("--limit", type=int, help="generate only the first N questions")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing review file (including any scores and notes)",
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("specify --model or set OLLAMA_CHAT_MODEL")

    try:
        run_generation_review(
            chunks_path=args.chunks,
            questions_path=args.questions,
            output_path=args.output,
            model=args.model,
            host=args.host,
            top_k=args.top_k,
            k1=args.k1,
            b=args.b,
            timeout=args.timeout,
            limit=args.limit,
            overwrite=args.overwrite,
            think=args.think,
            num_predict=args.num_predict,
        )
    except (OSError, ValueError, GenerationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
