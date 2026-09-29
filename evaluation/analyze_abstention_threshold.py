"""Measure a top-1 BM25 score threshold as a pilot abstention signal."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval import BM25Retriever, tokenize
from rag_from_scratch.retrieval_metrics import load_jsonl


COMMON_QUERY_TOKENS = {"a", "d", "en", "l"}


def _metadata_only_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [
        replace(
            chunk,
            text=" ".join(
                part
                for part in (Path(chunk.source_path).stem, chunk.section or "")
                if part
            ),
        )
        for chunk in chunks
    ]


def _top_score(ranker: BM25Retriever, query: str) -> float:
    results = ranker.rank(query)
    return results[0].score if results else 0.0


def _confusion(
    questions: list[dict[str, Any]],
    scores: dict[str, float],
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = fn = fp = tn = 0
    for question in questions:
        predicted_answerable = scores[question["id"]] >= threshold
        expected_answerable = bool(question.get("answerable"))
        if expected_answerable and predicted_answerable:
            tp += 1
        elif expected_answerable:
            fn += 1
        elif predicted_answerable:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def _variant_scores(
    *,
    name: str,
    questions: list[dict[str, Any]],
    score_query,
    threshold: float,
) -> dict[str, Any]:
    scores = {question["id"]: score_query(question["question"]) for question in questions}
    positives = [scores[q["id"]] for q in questions if q.get("answerable")]
    negatives = [scores[q["id"]] for q in questions if not q.get("answerable")]
    min_positive = min(positives)
    max_negative = max(negatives)
    midpoint = (min_positive + max_negative) / 2 if min_positive > max_negative else None
    threshold_confusion = _confusion(questions, scores, threshold)
    midpoint_confusion = _confusion(questions, scores, midpoint) if midpoint is not None else None
    return {
        "name": name,
        "min_positive_score": min_positive,
        "max_negative_score": max_negative,
        "separation_gap": min_positive - max_negative,
        "q16_top_score": scores.get("q16"),
        "threshold": threshold,
        "threshold_confusion_tp_fn_fp_tn": threshold_confusion,
        "posthoc_midpoint": midpoint,
        "midpoint_confusion_tp_fn_fp_tn": midpoint_confusion,
    }


def run_analysis(
    *,
    chunks_path: Path,
    questions_path: Path,
    output_path: Path,
    report_path: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    chunks = [Chunk(**record) for record in load_jsonl(chunks_path)]
    questions = load_jsonl(questions_path)
    if not chunks or not questions:
        raise ValueError("The evaluation corpus and question set must be non-empty")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    combined = BM25Retriever(chunks, include_metadata=True)
    text_only = BM25Retriever(chunks)
    metadata_only = BM25Retriever(_metadata_only_chunks(chunks))

    variants = [
        _variant_scores(
            name="BM25 courant : texte + métadonnées concaténés",
            questions=questions,
            score_query=lambda query: _top_score(combined, query),
            threshold=threshold,
        ),
        _variant_scores(
            name="Texte seulement",
            questions=questions,
            score_query=lambda query: _top_score(text_only, query),
            threshold=threshold,
        ),
        _variant_scores(
            name="Métadonnées et mots fréquents retirés de la requête",
            questions=questions,
            score_query=lambda query: _top_score(
                combined,
                " ".join(token for token in tokenize(query) if token not in COMMON_QUERY_TOKENS),
            ),
            threshold=threshold,
        ),
    ]

    for alpha in (0.25, 0.5, 1.0):
        def score_separate_fields(query: str, *, weight: float = alpha) -> float:
            text_scores = {
                result.chunk.chunk_id: result.score for result in text_only.rank(query)
            }
            metadata_scores = {
                result.chunk.chunk_id: result.score for result in metadata_only.rank(query)
            }
            return max(
                (
                    text_scores.get(chunk.chunk_id, 0.0)
                    + weight * metadata_scores.get(chunk.chunk_id, 0.0)
                    for chunk in chunks
                ),
                default=0.0,
            )

        variants.append(
            _variant_scores(
                name=f"BM25 texte + {alpha:.2f} × métadonnées, index séparés",
                questions=questions,
                score_query=score_separate_fields,
                threshold=threshold,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(variants, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Analyse exploratoire d’un seuil BM25 pour l’abstention",
        "",
        f"Seuil fixe testé : `{threshold:g}`. Jeu pilote privé : {len(questions)} questions.",
        "Le seuil prédit « répondable » si le score BM25 top 1 est supérieur ou égal à cette valeur.",
        "Les résultats ne sont pas une mesure indépendante : le seuil postérieur indiqué est calculé sur le même jeu.",
        "",
        "| Variante | Score min. répondable | Score max. négatif | Écart | q16 | Seuil fixe TP/FN/FP/TN | Milieu post hoc TP/FN/FP/TN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        fixed = "/".join(map(str, variant["threshold_confusion_tp_fn_fp_tn"]))
        midpoint = variant["posthoc_midpoint"]
        midpoint_result = variant["midpoint_confusion_tp_fn_fp_tn"]
        midpoint_cell = (
            f"{midpoint:.3f}: " + "/".join(map(str, midpoint_result))
            if midpoint is not None and midpoint_result is not None
            else "aucune séparation"
        )
        report_lines.append(
            f"| {variant['name']} | {variant['min_positive_score']:.3f} | "
            f"{variant['max_negative_score']:.3f} | {variant['separation_gap']:.3f} | "
            f"{variant['q16_top_score']:.3f} | {fixed} | {midpoint_cell} |"
        )
    report_lines.extend(
        [
            "",
            "Dans la colonne de confusion, l’ordre est `TP/FN/FP/TN`. Un seuil appris sur ces 16 questions serait optimiste : il n’y a que deux questions sans réponse et aucune partition de calibration distincte.",
            "Les scores changent quand le mode BM25 ou le poids des champs change. Le seuil fixe n’est donc valable que pour une configuration figée, et les chiffres de ce jeu ne justifient pas son activation par défaut.",
            "",
            f"Détails agrégés sans texte privé : `{output_path}`.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Local details: {output_path}")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a BM25 threshold for abstention")
    parser.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    parser.add_argument(
        "--questions", type=Path, default=Path("evaluation/private/ia04_questions.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluations/abstention_threshold.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("experiments/v6-abstention-threshold.md")
    )
    parser.add_argument("--threshold", type=float, default=12.0)
    args = parser.parse_args()
    run_analysis(
        chunks_path=args.chunks,
        questions_path=args.questions,
        output_path=args.output,
        report_path=args.report,
        threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
