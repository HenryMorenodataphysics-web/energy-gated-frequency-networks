import json

from src.egfn_context import initialize_database, retrieve_context
from src.egfn_context import ingest_document, retrieve_knowledge


def test_evidence_round_trip(tmp_path):
    evidence = {
        "items": [
            {
                "evidence_id": "egfn-001",
                "source_path": "outputs/example.json",
                "title": "example",
                "payload": {"auc": 0.82},
            }
        ],
        "references": [],
    }
    database_path = tmp_path / "context.sqlite3"
    initialize_database(database_path, evidence)
    assert retrieve_context(database_path, "example")[0]["payload"] == {"auc": 0.82}


def test_document_ingestion_and_keyword_retrieval(tmp_path):
    database_path = tmp_path / "knowledge.sqlite3"
    initialize_database(database_path, {"items": [], "references": []})
    document_id = ingest_document(database_path, "manual.txt", b"Valve vibration requires bearing inspection.")
    results = retrieve_knowledge(database_path, "bearing vibration")
    assert document_id.startswith("doc-")
    assert results[0]["filename"] == "manual.txt"
