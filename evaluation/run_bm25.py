"""Evaluate the in-memory BM25 retriever on the private IA04 question set."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_from_scratch.models import Chunk
from rag_from_scratch.retrieval import BM25Retriever
from rag_from_scratch.retrieval_metrics import load_jsonl, score_rankings


def run_bm25_evaluation(
    *,
    chunks_path: Path,
    questions_path: Path,
    data_dir: Path,
    report_path: Path,
    k1: float = 1.5,
    b: float = 0.75,
    include_metadata: bool = False,
) -> dict[str, Any]:
    chunk_records = load_jsonl(chunks_path)
    chunks = [Chunk(**record) for record in chunk_records]
    questions = load_jsonl(questions_path)
    answerable = [question for question in questions if question.get("answerable")]
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    if not answerable:
        raise ValueError("The evaluation set contains no answerable questions")

    started = time.perf_counter()
    retriever = BM25Retriever(chunks, k1=k1, b=b, include_metadata=include_metadata)
    indexing_seconds = time.perf_counter() - started

    chunk_indexes = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    if len(chunk_indexes) != len(chunks):
        raise ValueError("Chunk IDs must be unique for evaluation")

    rankings: list[list[tuple[int, float]]] = []
    query_seconds = 0.0
    for question in answerable:
        started = time.perf_counter()
        results = retriever.rank(question["question"])
        query_seconds += time.perf_counter() - started
        rankings.append(
            [(chunk_indexes[result.chunk.chunk_id], result.score) for result in results]
        )

    metrics, per_question = score_rankings(questions, chunk_records, rankings)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result: dict[str, Any] = {
        "generated_at": generated_at,
        "method": "BM25",
        "k1": k1,
        "b": b,
        "include_metadata": include_metadata,
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "answerable_question_count": len(answerable),
        "indexing_seconds": indexing_seconds,
        "query_seconds_total": query_seconds,
        "query_seconds_average": query_seconds / len(answerable),
        "metrics": metrics,
        "per_question": per_question,
    }

    missed = [entry["id"] for entry in per_question if entry["first_relevant_rank"] is None]
    outside_top_five = [
        entry["id"]
        for entry in per_question
        if entry["first_relevant_rank"] is not None and entry["first_relevant_rank"] > 5
    ]
    details_filename = "bm25_with_metadata.json" if include_metadata else "bm25.json"
    details_path = data_dir / "evaluations" / details_filename
    report_lines = [
        "# Recherche BM25 IA04" + (" avec métadonnées" if include_metadata else ""),
        "",
        f"Évaluation générée le {generated_at}.",
        f"- Chunks : {len(chunks)} depuis `{chunks_path}`.",
        f"- Questions répondables : {len(answerable)} sur {len(questions)} ; les questions sans réponse sont exclues des métriques, conformément au protocole.",
        f"- Paramètres : `k1={k1:g}`, `b={b:g}`.",
        "- Recherche : BM25 implémenté en Python, index calculé en mémoire.",
        "- Champs indexés : texte du chunk, nom du fichier source sans extension et titre de section." if include_metadata else "- Champ indexé : texte du chunk uniquement.",
        "- Tokenisation : minuscules, accents retirés, séparation sur les caractères non alphanumériques ; aucun stemming ni mot vide retiré.",
        "- Recall@k : proportion micro des références document/section/page retrouvées ; MRR : rang du premier résultat pertinent, moyenné par question.",
        "",
        "| Méthode | Recall@1 | Recall@3 | Recall@5 | MRR | Indexation (s) | Requête moyenne (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| BM25 (`k1={k1:g}`, `b={b:g}`) | {metrics['recall@1']:.3f} | {metrics['recall@3']:.3f} | {metrics['recall@5']:.3f} | {metrics['mrr']:.3f} | {indexing_seconds:.4f} | {query_seconds / len(answerable) * 1000:.3f} |",
        "",
        "## Erreurs observées",
        "",
        f"- Aucune référence retrouvée : {len(missed)} question(s)" + (f" (`{', '.join(missed)}`)." if missed else "."),
        f"- Première référence après le top 5 : {len(outside_top_five)} question(s)" + (f" (`{', '.join(outside_top_five)}`)." if outside_top_five else "."),
        "- Les identifiants de questions sont indiqués sans reproduire leur texte ni le contenu des cours.",
        "",
        "## Limites",
        "",
        "- Ce score est un premier résultat sur un jeu pilote privé de petite taille ; il ne mesure pas la qualité des réponses générées.",
        "- La tokenisation initiale ne ramène pas les formes fléchies à leur racine et peut manquer des correspondances par synonymie ou paraphrase.",
        "- Les métadonnées sont concaténées au texte et utilisent le même poids BM25 ; une version ultérieure pourra scorer les champs séparément." if include_metadata else "- Les métadonnées source et section ne sont pas indexées dans cette baseline.",
        "- L’extraction PDF utilise `pdftotext -layout` ; l’ordre de lecture des pages à colonnes, tableaux et matrices peut affecter les résultats.",
        "- Les questions sans réponse ne sont pas évaluées ici pour l’abstention ; cette capacité sera mesurée avec la génération.",
        "",
        f"Les classements détaillés par question et les identifiants des chunks sont conservés localement dans `{details_path}`.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"BM25 Recall@1={metrics['recall@1']:.3f} Recall@3={metrics['recall@3']:.3f} "
        f"Recall@5={metrics['recall@5']:.3f} MRR={metrics['mrr']:.3f}"
    )
    print(f"Report: {report_path}")
    print(f"Local details: {details_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate BM25 on the private IA04 retrieval set")
    parser.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/private/ia04_questions.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--include-metadata", action="store_true")
    args = parser.parse_args()
    report_path = args.report or (
        Path("experiments/v3-bm25-metadata.md")
        if args.include_metadata
        else Path("experiments/v2-bm25.md")
    )
    run_bm25_evaluation(
        chunks_path=args.chunks,
        questions_path=args.questions,
        data_dir=args.data_dir,
        report_path=report_path,
        k1=args.k1,
        b=args.b,
        include_metadata=args.include_metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
