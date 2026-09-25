"""Compare local embedding models against the private IA04 retrieval set."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_from_scratch.embeddings import embed_texts, get_model_info
from rag_from_scratch.retrieval_metrics import load_jsonl as _load_jsonl
from rag_from_scratch.retrieval_metrics import score_rankings as _score_questions


@dataclass(frozen=True)
class ModelSpec:
    name: str
    query_prefix: str = ""
    document_prefix: str = ""

    def format_query(self, question: str) -> str:
        if self.name.startswith("qwen3-embedding"):
            instruction = (
                "Instruct: Given a question about the IA04 course, retrieve relevant passages "
                f"that answer the question.\nQuery: {question}"
            )
            return instruction
        return f"{self.query_prefix}{question}"

    def format_document(self, text: str) -> str:
        return f"{self.document_prefix}{text}"


MODEL_SPECS = (
    ModelSpec(name="bge-m3"),
    ModelSpec(name="qwen3-embedding:0.6b"),
    ModelSpec(
        name="corvojaeger/jina-embeddings-v5-text-small",
        query_prefix="Query: ",
        document_prefix="Document: ",
    ),
)


def _unit_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _rank_chunks(query_vector: list[float], document_vectors: list[list[float]]) -> list[tuple[int, float]]:
    normalized_query = _unit_vector(query_vector)
    scored = []
    for index, vector in enumerate(document_vectors):
        normalized_document = _unit_vector(vector)
        score = sum(left * right for left, right in zip(normalized_query, normalized_document))
        scored.append((index, score))
    return sorted(scored, key=lambda result: (-result[1], result[0]))


def _slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-")


def _write_index(path: Path, model: str, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for chunk, vector in zip(chunks, vectors):
            output.write(
                json.dumps(
                    {"model": model, "chunk_id": chunk["chunk_id"], "embedding": vector},
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_comparison(
    *,
    chunks_path: Path,
    questions_path: Path,
    data_dir: Path,
    report_path: Path,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    chunks = _load_jsonl(chunks_path)
    questions = _load_jsonl(questions_path)
    specs = [spec for spec in MODEL_SPECS if model_names is None or spec.name in model_names]
    if model_names and {spec.name for spec in specs} != set(model_names):
        raise ValueError(f"Unknown model name(s): {sorted(set(model_names) - {spec.name for spec in specs})}")
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    answerable = [question for question in questions if question.get("answerable")]
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        print(f"Embedding {len(chunks)} chunks with {spec.name}...")
        model_info = get_model_info(spec.name)
        document_texts = [spec.format_document(chunk["text"]) for chunk in chunks]
        started = time.perf_counter()
        document_vectors = embed_texts(spec.name, document_texts)
        document_seconds = time.perf_counter() - started

        started = time.perf_counter()
        query_vectors = embed_texts(spec.name, [spec.format_query(q["question"]) for q in answerable])
        query_seconds = time.perf_counter() - started
        if document_vectors and query_vectors and len(document_vectors[0]) != len(query_vectors[0]):
            raise ValueError(f"Embedding dimensions differ between queries and documents for {spec.name}")

        rankings = [_rank_chunks(query_vector, document_vectors) for query_vector in query_vectors]
        metrics, per_question = _score_questions(answerable, chunks, rankings)
        dimension = len(document_vectors[0]) if document_vectors else 0
        index_path = data_dir / "indexes" / f"{_slug(spec.name)}.jsonl"
        _write_index(index_path, spec.name, chunks, document_vectors)

        results[spec.name] = {
            "ollama_name": model_info["name"],
            "ollama_digest": model_info["digest"],
            "ollama_size_bytes": model_info["size"],
            "dimension": dimension,
            "document_embedding_seconds": document_seconds,
            "query_embedding_seconds": query_seconds,
            "metrics": metrics,
            "per_question": per_question,
            "index_path": str(index_path),
            "query_format": "instruction prefix" if spec.name.startswith("qwen3-embedding") else spec.query_prefix or "raw query",
            "document_format": spec.document_prefix or "raw document",
        }
        print(
            f"  dim={dimension} Recall@1={metrics['recall@1']:.3f} "
            f"Recall@3={metrics['recall@3']:.3f} Recall@5={metrics['recall@5']:.3f} "
            f"MRR={metrics['mrr']:.3f}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_lines = [
        "# Comparaison des embeddings IA04",
        "",
        f"Évaluation générée le {generated_at}.",
        f"- Chunks : {len(chunks)} depuis `{chunks_path}`.",
        f"- Questions répondables : {len(answerable)} sur {len(questions)} ; les questions sans réponse sont exclues, conformément au protocole.",
            "- Recherche : similarité cosinus sur les vecteurs normalisés, top-k évalué à 1, 3 et 5.",
        "- Recall@k : proportion micro des références document/section/page retrouvées ; MRR : rang du premier résultat pertinent, moyenné par question.",
        "",
        "| Modèle Ollama | Dimension | Recall@1 | Recall@3 | Recall@5 | MRR | Embedding corpus (s) | Embedding questions (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, result in results.items():
        metrics = result["metrics"]
        report_lines.append(
            f"| `{model}` | {result['dimension']} | {metrics['recall@1']:.3f} | "
            f"{metrics['recall@3']:.3f} | {metrics['recall@5']:.3f} | {metrics['mrr']:.3f} | "
            f"{result['document_embedding_seconds']:.2f} | {result['query_embedding_seconds']:.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## Versions Ollama",
            "",
            *[
                f"- `{model}` : `{result['ollama_name']}`, digest `{result['ollama_digest']}`, "
                f"{result['ollama_size_bytes'] / (1024 ** 3):.2f} Gio téléchargés."
                for model, result in results.items()
            ],
            "",
            "## Paramètres et limites",
            "",
            "- Découpage initial : fenêtres de 400 mots avec 50 mots de chevauchement, sans traverser une page ou une section.",
            "- Qwen3 reçoit une instruction sur les requêtes et les passages restent bruts ; Jina reçoit les préfixes `Query:` et `Document:` ; BGE-M3 reçoit le texte brut.",
            "- Jina v5 est utilisé via un [paquet Ollama communautaire](https://ollama.com/corvojaeger/jina-embeddings-v5-text-small) ; le modèle amont est sous licence [CC BY-NC 4.0](https://jina.ai/en-US/models/jina-embeddings-v5-text-small/), donc ce résultat reste une comparaison locale de projet pédagogique.",
            "- Les scores portent sur le petit jeu pilote privé de 14 questions répondables. Ils donnent un premier signal pour IA04, pas un classement général des modèles.",
            "- L’extraction PDF utilise actuellement `pdftotext -layout`. L’audit a relevé des pages à colonnes et des tableaux/matrices ; leur ordre de lecture n’est pas validé sur toutes les pages, ce qui peut influencer le résultat.",
            "- Les index vectoriels restent locaux dans `data/indexes/` et ne sont pas distribués.",
            "",
            "Les classements détaillés par question et les vecteurs sont conservés dans les artefacts locaux sous `data/`.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    detailed_path = data_dir / "evaluations" / "embedding_comparison.json"
    detailed_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_path.write_text(
        json.dumps({"generated_at": generated_at, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(f"Local details: {detailed_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Ollama embedding models on IA04")
    parser.add_argument("--chunks", type=Path, default=Path("data/gold/ia04_chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/private/ia04_questions.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=Path("experiments/v1-embedding.md"))
    parser.add_argument("--model", action="append", dest="models", help="model from the built-in comparison list")
    args = parser.parse_args()
    run_comparison(
        chunks_path=args.chunks,
        questions_path=args.questions,
        data_dir=args.data_dir,
        report_path=args.report,
        model_names=args.models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
