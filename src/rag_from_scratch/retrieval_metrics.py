"""Shared metrics for evaluating chunk retrieval methods."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a UTF-8 JSON Lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_locator(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", without_marks)
    return " ".join(word[:-1] if len(word) > 4 and word.endswith("s") else word for word in words)


def _chunk_matches_reference(chunk: dict[str, Any], reference: dict[str, Any]) -> bool:
    if chunk["source_id"] != reference["source_id"]:
        return False

    locator = reference.get("locator", {})
    locator_type = locator.get("type")
    locator_value = locator.get("value")
    if locator_type == "page":
        return chunk.get("page") == locator_value
    if locator_type != "section" or not isinstance(locator_value, str):
        return True

    actual = _normalize_locator(chunk.get("section") or "")
    if not actual:
        return False
    actual_tokens = set(actual.split())
    candidates = [_normalize_locator(part) for part in locator_value.split(";") if part.strip()]
    for expected in candidates:
        if not expected:
            continue
        if expected in actual:
            return True
        expected_tokens = set(expected.split())
        if expected_tokens and expected_tokens.issubset(actual_tokens):
            return True
    return False


def score_rankings(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    rankings: list[list[tuple[int, float]]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute Recall@1/3/5 and MRR from ranked chunk indices."""
    answerable = [question for question in questions if question.get("answerable")]
    if not answerable:
        raise ValueError("The evaluation set contains no answerable questions")
    if len(rankings) != len(answerable):
        raise ValueError("The number of rankings must match the answerable questions")

    recall_counts = {1: 0, 3: 0, 5: 0}
    total_references = sum(len(question.get("relevant_sources", [])) for question in answerable)
    reciprocal_ranks: list[float] = []
    per_question: list[dict[str, Any]] = []

    for question, ranking in zip(answerable, rankings):
        references = question.get("relevant_sources", [])
        first_relevant_rank = None
        for rank, (chunk_index, _) in enumerate(ranking, start=1):
            if any(_chunk_matches_reference(chunks[chunk_index], ref) for ref in references):
                first_relevant_rank = rank
                break
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)

        found_by_k = {}
        for k in recall_counts:
            retrieved = {chunk_index for chunk_index, _ in ranking[:k]}
            found = sum(
                any(_chunk_matches_reference(chunks[index], ref) for index in retrieved)
                for ref in references
            )
            recall_counts[k] += found
            found_by_k[k] = found / len(references) if references else 0.0

        per_question.append(
            {
                "id": question["id"],
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": reciprocal_ranks[-1],
                "recall_by_k": found_by_k,
                "top_chunks": [
                    {"chunk_id": chunks[index]["chunk_id"], "score": score}
                    for index, score in ranking[:5]
                ],
            }
        )

    metrics = {
        f"recall@{k}": count / total_references if total_references else 0.0
        for k, count in recall_counts.items()
    }
    metrics["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return metrics, per_question
