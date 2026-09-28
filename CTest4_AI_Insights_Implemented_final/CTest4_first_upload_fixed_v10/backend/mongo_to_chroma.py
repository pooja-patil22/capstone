"""
Standalone MongoDB -> ChromaDB vectorization utility.

This utility mirrors the application's post-MongoDB pipeline:
MongoDB -> text preprocessing -> chunking -> embeddings -> ChromaDB.

The Streamlit application performs the same work automatically after upload.
"""

from pymongo import MongoClient
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
import sys

# Add current directory to path for imports when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mongo_data_filter import filter_mongodb_records


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "qa_test_analyzer"
MONGO_COLLECTION = "test_execution_data"

CHROMA_PATH = "./backend/chroma_db"
CHROMA_COLLECTION = "test_execution_history"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logging.basicConfig(
    filename="./pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mongo_to_chroma")


def clean_value(value):
    if value is None:
        return ""

    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return value.item() if hasattr(value, "item") else value


def create_document(record):
    """Convert one MongoDB record into searchable text."""
    parts = []

    for key, value in record.items():
        if key == "_id" or str(key).startswith("_"):
            continue

        value = clean_value(value)
        if value != "":
            parts.append(f"{str(key).replace('_', ' ')}: {value}")

    return ". ".join(parts)


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    words = str(text or "").split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - min(chunk_overlap, chunk_size - 1))
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


def main():
    print("Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    try:
        mongo_client.admin.command("ping")
        collection = mongo_client[DB_NAME][MONGO_COLLECTION]
        records = list(collection.find({}))

        print(f"MongoDB records found: {len(records)}")

        # Apply the same MongoDB retrieval filter used by the application
        # before chunking/embedding. Keep `records` intact for source IDs and
        # post-Chroma deletion; only `clean_records` enters preprocessing.
        clean_records = filter_mongodb_records(records)
        print("MongoDB retrieval filter: 7 approved columns")

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )

        model = SentenceTransformer(EMBEDDING_MODEL)

        ids = []
        documents = []
        metadatas = []

        for source_record, record in zip(records, clean_records):
            mongo_id = source_record.get("_id")
            if mongo_id is None:
                continue

            source_document = create_document(record)
            chunks = chunk_text(source_document)

            for chunk_index, chunk in enumerate(chunks):
                vector_id = f"mongo_{str(mongo_id)}_chunk_{chunk_index}"

                metadata = {}
                for key, value in record.items():
                    value = clean_value(value)
                    if value == "":
                        continue

                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, (list, tuple)):
                        metadata[key] = "; ".join(str(item) for item in value if item)
                    else:
                        metadata[key] = str(value)

                source_file_name = str(source_record.get("source_file", "") or "")
                file_stem = Path(source_file_name).stem if source_file_name else "legacy"
                # Preserve the upload batch as internal Chroma routing metadata.
                # It is not part of the seven-column clean dataset passed into
                # chunking/embedding, but Analytics Engine needs it to identify
                # a batch after MongoDB raw records are removed.
                upload_batch_id = source_record.get("upload_batch_id")
                if upload_batch_id:
                    metadata["upload_batch_id"] = str(upload_batch_id)
                metadata["namespace"] = f"{CHROMA_COLLECTION}::{file_stem}"
                metadata["embedding_model"] = EMBEDDING_MODEL
                metadata["chunk_index"] = chunk_index
                metadata["chunk_count"] = len(chunks)
                metadata["mongo_record_id"] = str(mongo_id)

                ids.append(vector_id)
                documents.append(chunk)
                metadatas.append(metadata)

        if ids:
            print(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = model.encode(
                documents,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).tolist()

            try:
                chroma_collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                persisted = chroma_collection.get(ids=ids, include=["metadatas"])
                persisted_ids = set(persisted.get("ids", []) or [])
                missing_ids = [vector_id for vector_id in ids if vector_id not in persisted_ids]
                if missing_ids:
                    raise RuntimeError(
                        f"ChromaDB verification failed: {len(missing_ids)} vector(s) are missing."
                    )
                logger.info(
                    "ChromaDB insertion SUCCESS | vectors=%s | collection=%s",
                    len(ids), CHROMA_COLLECTION,
                )
            except Exception:
                logger.exception(
                    "ChromaDB insertion/verification FAILURE | vectors=%s",
                    len(ids),
                )
                # MongoDB raw data is deliberately retained on ChromaDB failure.
                raise

            logger.info(
                "MongoDB raw-data retained SUCCESS | records=%s (not deleted)",
                len(records),
            )
            print("MongoDB raw data retained:", len(records))


        print("================================")
        print("MongoDB -> ChromaDB vectorization complete")
        print("MongoDB records:", len(records))
        print("Text chunks:", len(ids))
        print("Embedding model:", EMBEDDING_MODEL)
        print("ChromaDB path:", CHROMA_PATH)
        print("ChromaDB collection:", CHROMA_COLLECTION)
        print("ChromaDB vectors:", chroma_collection.count())
        print("================================")

    finally:
        mongo_client.close()


if __name__ == "__main__":
    main()
