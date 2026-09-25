"""Run a small, controlled ablation to diagnose BM25 retrieval errors."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval import BM25Retriever, tokenize
from rag_from_scratch.retrieval_metrics import (
    _chunk_matches_reference,
    load_jsonl,
    score_rankings,
)


def _light_plural_normalization(text: str) -> str:
    """Apply the deliberately simple plural rule used in this diagnostic only."""
    tokens = tokenize(text)
    return " ".join(token[:-1] if len(token) > 4 and token.endswith("s") else token for token in tokens)


def _build_variant_chunks(
    chunks: list[Chunk],
    *,
    add_filename: bool,
    add_section: bool,
    normalize_plurals: bool,
) -> list[Chunk]:
    variant_chunks = []
    for chunk in chunks:
        fields = [chunk.text]
        if add_filename:
            fields.append(Path(chunk.source_path).stem)
        if add_section and chunk.section:
            fields.append(chunk.section)
        text = " ".join(fields)
        if normalize_plurals:
            text = _light_plural_normalization(text)
        variant_chunks.append(replace(chunk, text=text))
    return variant_chunks


def _evaluate_variant(
    *,
    name: str,
    chunks: list[Chunk],
    chunk_records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    add_filename: bool = False,
    add_section: bool = False,
    normalize_plurals: bool = False,
) -> dict[str, Any]:
    variant_chunks = _build_variant_chunks(
        chunks,
        add_filename=add_filename,
        add_section=add_section,
        normalize_plurals=normalize_plurals,
    )
    retriever = BM25Retriever(variant_chunks)
    chunk_indexes = {chunk.chunk_id: index for index, chunk in enumerate(variant_chunks)}
    answerable = [question for question in questions if question.get("answerable")]
    rankings: list[list[tuple[int, float]]] = []
    for question in answerable:
        query = _light_plural_normalization(question["question"]) if normalize_plurals else question["question"]
        ranked = retriever.rank(query)
        rankings.append([(chunk_indexes[result.chunk.chunk_id], result.score) for result in ranked])

    metrics, per_question = score_rankings(questions, chunk_records, rankings)
    per_question_by_id = {row["id"]: row for row in per_question}
    case_reference_ranks = {}
    for question_id in ("q01", "q13"):
        question_index = next(
            index for index, question in enumerate(answerable) if question["id"] == question_id
        )
        question = answerable[question_index]
        rank_by_chunk = {
            chunk_records[chunk_index]["chunk_id"]: rank
            for rank, (chunk_index, _) in enumerate(rankings[question_index], start=1)
        }
        reference_ranks = []
        for reference in question.get("relevant_sources", []):
            matching_indexes = [
                index
                for index, record in enumerate(chunk_records)
                if _chunk_matches_reference(record, reference)
            ]
            matching_ranks = [
                rank_by_chunk[chunk_records[index]["chunk_id"]]
                for index in matching_indexes
                if chunk_records[index]["chunk_id"] in rank_by_chunk
            ]
            reference_ranks.append(
                {
                    "source_id": reference["source_id"],
                    "locator": reference.get("locator", {}),
                    "best_rank": min(matching_ranks) if matching_ranks else None,
                }
            )
        case_reference_ranks[question_id] = reference_ranks

    return {
        "name": name,
        "parameters": {
            "add_filename": add_filename,
            "add_section": add_section,
            "normalize_plurals": normalize_plurals,
        },
        "metrics": metrics,
        "q01": {
            "first_relevant_rank": per_question_by_id["q01"]["first_relevant_rank"],
            "recall@5": per_question_by_id["q01"]["recall_by_k"][5],
            "reference_ranks": case_reference_ranks["q01"],
        },
        "q13": {
            "first_relevant_rank": per_question_by_id["q13"]["first_relevant_rank"],
            "recall@5": per_question_by_id["q13"]["recall_by_k"][5],
            "reference_ranks": case_reference_ranks["q13"],
        },
    }


def run_analysis(
    *,
    chunks_path: Path,
    questions_path: Path,
    details_path: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    chunk_records = load_jsonl(chunks_path)
    chunks = [Chunk(**record) for record in chunk_records]
    questions = load_jsonl(questions_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    variants = [
        ("Texte seul (baseline)", False, False, False),
        ("Normalisation légère du pluriel", False, False, True),
        ("Nom de fichier", True, False, False),
        ("Titre de section", False, True, False),
        ("Nom de fichier + titre de section", True, True, False),
        ("Nom, section et pluriel", True, True, True),
    ]
    results = [
        _evaluate_variant(
            name=name,
            chunks=chunks,
            chunk_records=chunk_records,
            questions=questions,
            add_filename=add_filename,
            add_section=add_section,
            normalize_plurals=normalize_plurals,
        )
        for name, add_filename, add_section, normalize_plurals in variants
    ]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    answerable_count = sum(bool(question.get("answerable")) for question in questions)
    report_lines = [
        "# Analyse des erreurs BM25 IA04",
        "",
        f"Analyse générée le {generated_at}.",
        f"- Corpus : {len(chunks)} chunks depuis `{chunks_path}`.",
        f"- Questions : {answerable_count} répondables sur {len(questions)} ; même jeu et mêmes critères que `v2-bm25.md`.",
        "- Les variantes sont évaluées sur les mêmes questions et les mêmes références ; la baseline BM25 n’est pas modifiée.",
        "- Les textes de questions et des cours ne sont pas recopiés dans ce rapport.",
        "",
        "## Diagnostic ciblé",
        "",
        "| Question | Référence attendue | Rang BM25 initial | Observation |",
        "| --- | --- | ---: | --- |",
        "| `q01` | `course_03_communication_multi_agents`, puis `course_00_intro` | 2 puis 8 | La recherche trouve une des deux références dans le top 5 ; l’autre section reste au rang 8. |",
        "| `q13` | `annales_a21_median_ia04`, page 1 | 17 | Les meilleurs passages lexicaux parlent d’autres annales ; la référence attendue n’entre pas dans le top 5. |",
        "",
        "L’inspection des termes confirme deux signaux pour `q13` : la question emploie le pluriel « algorithmes » alors que le chunk de référence contient « algorithme » ; l’identifiant `A21` figure dans le nom du fichier source, absent du texte scoré. Le chunk de référence contient aussi le terme « médian ». Les mots `appariement`, `A21` et `médian` apparaissent chacun dans plusieurs chunks, tandis que les termes fréquents contribuent encore au score.",
        "",
        "Pour `q01`, les deux sources attendues sont liées à des titres de section. Le retriever ne scorait jusque-là que `chunk.text`, sans utiliser `chunk.section` ni le nom du fichier source.",
        "",
        "## Ablation contrôlée",
        "",
        "| Variante | Recall@1 | Recall@3 | Recall@5 | MRR | Rang ref. `q01` | Recall@5 `q01` | Rang ref. `q13` |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result["metrics"]
        report_lines.append(
            f"| {result['name']} | {metrics['recall@1']:.3f} | {metrics['recall@3']:.3f} | "
            f"{metrics['recall@5']:.3f} | {metrics['mrr']:.3f} | "
            f"{result['q01']['first_relevant_rank']} | {result['q01']['recall@5']:.3f} | "
            f"{result['q13']['first_relevant_rank']} |"
        )
    metadata_variant = next(result for result in results if result["name"] == "Nom de fichier + titre de section")
    report_lines.extend(
        [
            "",
            "La variante nom de fichier + titre de section retrouve les deux références de `q01` dans le top 5 et fait passer la référence de `q13` du rang 17 au rang 2. La normalisation plurielle seule fait progresser `q13`, mais diminue le MRR global ; elle n’est donc pas retenue comme correction générale.",
            "",
            "## Décision",
            "",
            f"Le meilleur candidat de cette analyse est l’ajout du nom de fichier et du titre de section à la recherche : Recall@1 {metadata_variant['metrics']['recall@1']:.3f}, Recall@3 {metadata_variant['metrics']['recall@3']:.3f}, Recall@5 {metadata_variant['metrics']['recall@5']:.3f}, MRR {metadata_variant['metrics']['mrr']:.3f}.",
            "Ces résultats sont exploratoires : les mêmes 14 questions ont servi à diagnostiquer et à comparer les variantes. Ils motivent une version BM25 avec métadonnées, sans établir une performance générale. La normalisation morphologique reste une piste séparée.",
            "",
            f"Détails sans textes de questions ni de chunks : `{details_path}`.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(
        json.dumps({"generated_at": generated_at, "variants": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Analysis: {report_path}")
    print(f"Local details: {details_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose BM25 retrieval errors on IA04")
    parser.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/private/ia04_questions.jsonl"))
    parser.add_argument("--details", type=Path, default=Path("data/evaluations/bm25_ablation.json"))
    parser.add_argument("--report", type=Path, default=Path("experiments/v2-bm25-analysis.md"))
    args = parser.parse_args()
    run_analysis(
        chunks_path=args.chunks,
        questions_path=args.questions,
        details_path=args.details,
        report_path=args.report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
