"""LlamaIndex baseline pipeline for IA04 end-to-end evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import Document, PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pypdf import PdfReader

from rag_from_scratch.ingestion import _markdown_sections

FRENCH_QA_PROMPT = PromptTemplate(
    "Tu es un assistant pédagogique pour le cours IA04.\n"
    "Réponds à la question en français en te basant UNIQUEMENT sur les extraits de contexte fournis ci-dessous.\n"
    "Si le contexte ne contient pas les informations nécessaires pour répondre avec certitude à la question "
    "ou si le fait demandé est absent du contexte, réponds EXACTEMENT :\n"
    "Je ne peux pas le déterminer à partir du corpus IA04 fourni.\n"
    "N'invente aucun fait, aucun chiffre et aucune date.\n\n"
    "---------------------\n"
    "Contexte :\n"
    "{context_str}\n"
    "---------------------\n"
    "Question : {query_str}\n"
    "Réponse : "
)


def load_ia04_nodes(
    data_dir: str | Path = "data",
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> tuple[list[BaseNode], dict[str, Any]]:
    """Ingest and chunk IA04 documents using LlamaIndex SentenceSplitter, preserving metadata."""
    root = Path(data_dir)
    manifest_path = root / "manifest.jsonl"
    raw_dir = root / "raw"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ia04_records = [r for r in records if r.get("path", "").startswith("IA04/")]
    if not ia04_records:
        raise ValueError(f"No IA04 sources found in {manifest_path}")

    documents: list[Document] = []
    for r in ia04_records:
        fpath = raw_dir / r["path"]
        if not fpath.is_file():
            raise FileNotFoundError(f"Source file missing: {fpath}")

        source_id = r["source_id"]
        fmt = r.get("format")

        if fmt == "md":
            text = fpath.read_text(encoding="utf-8")
            for sec_title, body in _markdown_sections(text):
                documents.append(
                    Document(
                        text=body,
                        metadata={
                            "source_id": source_id,
                            "file_name": fpath.name,
                            "section": sec_title or None,
                            "page": None,
                        },
                    )
                )
        elif fmt == "pdf":
            reader = PdfReader(str(fpath))
            for p_idx, page in enumerate(reader.pages, start=1):
                p_text = page.extract_text() or ""
                documents.append(
                    Document(
                        text=p_text,
                        metadata={
                            "source_id": source_id,
                            "file_name": fpath.name,
                            "section": None,
                            "page": p_idx,
                        },
                    )
                )

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    stats = {
        "raw_files": len(ia04_records),
        "raw_documents_or_pages": len(documents),
        "total_nodes": len(nodes),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    return nodes, stats


def create_llamaindex_pipeline(
    nodes: list[BaseNode],
    *,
    embed_model_name: str = "bge-m3",
    llm_model_name: str = "qwen3.5:4b",
    temperature: float = 0.0,
    top_k: int = 5,
    timeout: float = 300.0,
) -> tuple[VectorStoreIndex, Any]:
    """Create a VectorStoreIndex and configured QueryEngine with reasoning disabled."""
    embed_model = OllamaEmbedding(model_name=embed_model_name, request_timeout=timeout)
    llm = Ollama(
        model=llm_model_name,
        temperature=temperature,
        thinking=False,
        request_timeout=timeout,
        additional_kwargs={"num_predict": 384},
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=FRENCH_QA_PROMPT,
        llm=llm,
        response_mode="compact",
    )
    return index, query_engine
