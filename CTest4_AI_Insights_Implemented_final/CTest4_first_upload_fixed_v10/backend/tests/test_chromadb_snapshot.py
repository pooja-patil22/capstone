"""Regression test for the human-readable ChromaDB snapshot scope."""

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_SOURCE = (PROJECT_ROOT / "frontend" / "app.py").read_text(encoding="utf-8")


def _load_snapshot_builder():
    tree = ast.parse(APP_SOURCE)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_build_current_chromadb_snapshot_records"
    )
    namespace = {"DEFAULT_EMBEDDING_MODEL": "all-MiniLM-L6-v2"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace[node.name]


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def get(self, include=None):
        return {
            "ids": [record["id"] for record in self.records],
            "documents": [record["document"] for record in self.records],
            "metadatas": [record["metadata"] for record in self.records],
        }


def test_snapshot_reads_all_persisted_vectors_not_only_latest_batch():
    build_snapshot = _load_snapshot_builder()

    records = []
    for batch_no in (1, 2):
        for index in range(10):
            records.append(
                {
                    "id": f"mongo-{batch_no}-{index}_chunk_0",
                    "document": f"run case id: RUN-{batch_no:03d}. test case id: TC-{index:03d}",
                    "metadata": {
                        "mongo_record_id": f"mongo-{batch_no}-{index}",
                        "upload_batch_id": f"batch-{batch_no}",
                        "run_case_id": f"RUN-{batch_no:03d}",
                        "test_case_id": f"TC-{index:03d}",
                        "test_case_name": f"Test {batch_no}-{index}",
                        "module_name": "Module",
                        "status": "PASSED",
                        "chunk_index": 0,
                        "chunk_count": 1,
                    },
                }
            )

    snapshot = build_snapshot(FakeCollection(records))

    assert len(snapshot) == 20
    assert sum(len(record["chunks"]) for record in snapshot) == 20
    assert {chunk["metadata"]["upload_batch_id"] for record in snapshot for chunk in record["chunks"]} == {
        "batch-1", "batch-2"
    }
    assert snapshot[0]["chunks"][0]["metadata"]["test_case_id"] == "TC-000"
