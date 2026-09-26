"""Shared helpers for the local Ollama HTTP API."""

from __future__ import annotations

import os


def ollama_base_url(host: str | None = None) -> str:
    """Return a normalized Ollama host URL, using ``OLLAMA_HOST`` if unset."""
    value = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    return value if "://" in value else f"http://{value}"
