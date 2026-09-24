"""Read the IA04 source files as page- or section-level text units."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from rag_from_scratch.models import TextUnit


class IngestionError(RuntimeError):
    """Raised when a source cannot be read or extracted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return (section path, text) pairs, keeping each Markdown heading with its body."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current_lines
        body = "".join(current_lines).strip()
        if body:
            title = " > ".join(title for _, title in heading_stack)
            sections.append((title, body))
        current_lines = []

    for line in text.splitlines(keepends=True):
        fence = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            if not in_fence:
                in_fence = True
            elif fence.group(1)[0] == "`" or fence.group(1)[0] == "~":
                in_fence = False
            current_lines.append(line)
            continue

        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        current_lines.append(line)

    flush()
    return sections


def _pdf_page_text(path: Path, page: int) -> str:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise IngestionError("pdftotext is required to extract IA04 PDF pages")

    result = subprocess.run(
        [
            executable,
            "-f",
            str(page),
            "-l",
            str(page),
            "-layout",
            "-enc",
            "UTF-8",
            str(path),
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise IngestionError(f"Could not extract page {page} from {path}: {detail}")
    return result.stdout.strip()


def load_ia04_units(
    data_dir: str | Path = "data",
    *,
    verify_hashes: bool = True,
) -> list[TextUnit]:
    """Load local IA04 Markdown sections and PDF pages from the data manifest."""
    root = Path(data_dir)
    raw_dir = root / "raw"
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise IngestionError(f"Corpus manifest not found: {manifest_path}")

    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ia04_records = [record for record in records if record.get("path", "").startswith("IA04/")]
    if not ia04_records:
        raise IngestionError(f"No IA04 sources found in {manifest_path}")

    units: list[TextUnit] = []
    for record in ia04_records:
        relative_path = record["path"]
        source_path = raw_dir / relative_path
        if not source_path.is_file():
            raise IngestionError(f"Manifest source is missing: {source_path}")

        expected_hash = record.get("sha256")
        if verify_hashes and expected_hash and _sha256(source_path) != expected_hash:
            raise IngestionError(f"Source changed since the manifest was written: {source_path}")

        source_id = record["source_id"]
        if record.get("format") == "md":
            text = source_path.read_text(encoding="utf-8")
            for index, (section, body) in enumerate(_markdown_sections(text), start=1):
                unit_id = f"section-{index:04d}"
                units.append(
                    TextUnit(
                        source_id=source_id,
                        source_path=relative_path,
                        unit_id=unit_id,
                        text=body,
                        section=section or None,
                    )
                )
        elif record.get("format") == "pdf":
            pages = record.get("pages")
            if not isinstance(pages, int) or pages <= 0:
                raise IngestionError(f"PDF page count is missing from manifest: {relative_path}")
            for page in range(1, pages + 1):
                text = _pdf_page_text(source_path, page)
                if text:
                    units.append(
                        TextUnit(
                            source_id=source_id,
                            source_path=relative_path,
                            unit_id=f"page-{page:04d}",
                            text=text,
                            page=page,
                        )
                    )
        else:
            raise IngestionError(f"Unsupported IA04 source format: {record.get('format')!r}")

    return units


def write_units_jsonl(units: list[TextUnit], output_path: str | Path) -> None:
    """Write extracted units as UTF-8 JSON Lines."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for unit in units:
            output.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")
