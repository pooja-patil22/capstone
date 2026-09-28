"""Safety regression test for MongoDB -> ChromaDB cleanup.

If ChromaDB insertion fails, the raw MongoDB records must remain untouched.
"""

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Keep this regression test runnable even when the full application
# dependencies are not installed in the test environment.
try:
    import pymongo  # noqa: F401
except ImportError:
    sys.modules["pymongo"] = SimpleNamespace(MongoClient=object)

try:
    import chromadb  # noqa: F401
except ImportError:
    sys.modules["chromadb"] = SimpleNamespace(PersistentClient=object)

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    sys.modules["sentence_transformers"] = SimpleNamespace(SentenceTransformer=object)


class TestChromaFailureRetainsMongoData(unittest.TestCase):
    def test_chromadb_failure_does_not_delete_mongodb_records(self):
        module = importlib.import_module("mongo_to_chroma")

        raw_records = [
            {"_id": "mongo-1", "upload_batch_id": "batch-1", "status": "failed"},
            {"_id": "mongo-2", "upload_batch_id": "batch-1", "status": "passed"},
        ]

        collection = SimpleNamespace(
            find=lambda query: raw_records,
            delete_many=unittest.mock.Mock(name="delete_many"),
            count_documents=unittest.mock.Mock(name="count_documents"),
        )

        class FakeDatabase:
            def __getitem__(self, name):
                return collection

        class FakeMongoClient:
            def __init__(self):
                self.admin = SimpleNamespace(
                    command=unittest.mock.Mock(return_value={"ok": 1})
                )
                self.close = unittest.mock.Mock(name="mongo_close")
                self.database = FakeDatabase()

            def __getitem__(self, name):
                return self.database

        mongo_client = FakeMongoClient()

        # Use a real-looking Chroma collection whose write operation fails.
        chroma_collection = SimpleNamespace(
            upsert=unittest.mock.Mock(
                name="upsert",
                side_effect=RuntimeError("simulated ChromaDB insertion failure"),
            ),
            get=unittest.mock.Mock(name="get"),
            count=unittest.mock.Mock(return_value=0),
        )
        chroma_client = SimpleNamespace(
            get_or_create_collection=unittest.mock.Mock(
                return_value=chroma_collection
            )
        )

        class FakeEmbeddingArray:
            def tolist(self):
                return [[0.1, 0.2, 0.3] for _ in raw_records]

        embedding_model = SimpleNamespace(
            encode=unittest.mock.Mock(return_value=FakeEmbeddingArray())
        )

        with patch.object(module, "MongoClient", return_value=mongo_client), \
             patch.object(module.chromadb, "PersistentClient", return_value=chroma_client), \
             patch.object(module, "SentenceTransformer", return_value=embedding_model):
            with self.assertRaises(RuntimeError):
                module.main()

        # Critical safety assertion: cleanup must never be attempted after
        # ChromaDB insertion failure.
        collection.delete_many.assert_not_called()
        collection.count_documents.assert_not_called()
        chroma_collection.upsert.assert_called_once()
        mongo_client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
