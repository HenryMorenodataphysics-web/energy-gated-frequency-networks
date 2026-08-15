from __future__ import annotations

import json
import sqlite3
import hashlib
import re
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    title TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_items (
    reference_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    location TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    document_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
CREATE TABLE IF NOT EXISTS repairs (
    repair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    symptom TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    source_document_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_document_id) REFERENCES documents(document_id)
);
"""


def load_evidence(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_database(database_path: Path, evidence: dict[str, Any]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA)
        for item in evidence.get("items", []):
            connection.execute(
                "INSERT OR REPLACE INTO evidence "
                "(evidence_id, source_path, title, payload_json) VALUES (?, ?, ?, ?)",
                (
                    item["evidence_id"],
                    item["source_path"],
                    item["title"],
                    json.dumps(item["payload"], ensure_ascii=False),
                ),
            )
            connection.execute(
                "INSERT INTO history (evidence_id) VALUES (?)",
                (item["evidence_id"],),
            )
        for reference in evidence.get("references", []):
            connection.execute(
                "INSERT OR REPLACE INTO reference_items "
                "(reference_id, title, location, note) VALUES (?, ?, ?, ?)",
                (reference["reference_id"], reference["title"], reference["location"], reference["note"]),
            )


def retrieve_context(database_path: Path, query: str = "") -> list[dict[str, Any]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT evidence_id, source_path, title, payload_json FROM evidence "
            "WHERE title LIKE ? OR source_path LIKE ? ORDER BY evidence_id",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
    return [
        {
            "evidence_id": evidence_id,
            "source_path": source_path,
            "title": title,
            "payload": json.loads(payload_json),
        }
        for evidence_id, source_path, title, payload_json in rows
    ]


def extract_document_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            from io import BytesIO

            reader = PdfReader(BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pypdf in requirements-dashboard.txt") from exc
    text = content.decode("utf-8", errors="replace")
    if suffix == ".json":
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def ingest_document(
    database_path: Path,
    filename: str,
    content: bytes,
    metadata: dict[str, Any] | None = None,
) -> str:
    text = extract_document_text(filename, content)
    if not text:
        raise ValueError("The document contains no extractable text.")
    digest = hashlib.sha256(content).hexdigest()
    document_id = f"doc-{digest[:16]}"
    chunks = _chunk_text(text)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO documents "
            "(document_id, filename, document_type, content_hash, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (document_id, filename, Path(filename).suffix.lower().lstrip("."), digest, json.dumps(metadata or {})),
        )
        connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        connection.executemany(
            "INSERT INTO document_chunks (chunk_id, document_id, chunk_index, content) VALUES (?, ?, ?, ?)",
            [(f"{document_id}-{index:04d}", document_id, index, chunk) for index, chunk in enumerate(chunks)],
        )
    return document_id


def add_repair(
    database_path: Path,
    machine_id: str,
    symptom: str,
    action: str,
    outcome: str,
    source_document_id: str | None = None,
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO repairs (machine_id, symptom, action, outcome, source_document_id) VALUES (?, ?, ?, ?, ?)",
            (machine_id, symptom, action, outcome, source_document_id),
        )


def retrieve_knowledge(database_path: Path, query: str, limit: int = 8) -> list[dict[str, Any]]:
    terms = [term for term in re.findall(r"\w+", query.lower()) if len(term) > 2]
    if not terms:
        return []
    clauses = " OR ".join("LOWER(c.content) LIKE ?" for _ in terms)
    parameters = [f"%{term}%" for term in terms]
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT c.chunk_id, d.document_id, d.filename, c.content "
            "FROM document_chunks c JOIN documents d ON d.document_id = c.document_id "
            f"WHERE {clauses} ORDER BY c.chunk_index LIMIT ?",
            (*parameters, limit),
        ).fetchall()
    return [
        {"chunk_id": chunk_id, "document_id": document_id, "filename": filename, "content": content}
        for chunk_id, document_id, filename, content in rows
    ]
