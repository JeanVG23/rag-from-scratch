"""Small Ollama client for embedding text in batches."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

from rag_from_scratch.ollama import ollama_base_url


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot create the requested embeddings."""


def get_model_info(model: str, *, host: str | None = None) -> dict[str, int | str]:
    """Return the installed Ollama digest and byte size for a model alias."""
    base_url = ollama_base_url(host)
    request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EmbeddingError(f"Could not read installed model versions from Ollama: {error}") from error

    model_entries = result.get("models", [])
    requested = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    for entry in model_entries:
        if entry.get("name") in {model, requested}:
            return {"name": entry["name"], "digest": entry.get("digest", "unknown"), "size": entry.get("size", 0)}
    raise EmbeddingError(f"Model {model!r} is not installed in the local Ollama library")


def embed_texts(
    model: str,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    host: str | None = None,
    timeout: float = 600,
) -> list[list[float]]:
    """Embed texts with Ollama's local ``/api/embed`` endpoint."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not texts:
        return []

    base_url = ollama_base_url(host)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise EmbeddingError(
                f"Could not reach Ollama at {base_url}: {error}. Start Ollama and pull {model!r}."
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EmbeddingError(f"Ollama returned invalid JSON for model {model!r}") from error

        batch_embeddings = result.get("embeddings")
        if not isinstance(batch_embeddings, list) or len(batch_embeddings) != len(batch):
            raise EmbeddingError(f"Ollama returned an unexpected embedding response for {model!r}")
        embeddings.extend(batch_embeddings)

    return embeddings
