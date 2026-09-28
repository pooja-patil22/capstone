import json
import hashlib
import logging
import html
import os
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from html.parser import HTMLParser
from uuid import uuid4

# Add project root to Python path to enable backend module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from pymongo import MongoClient
from dotenv import load_dotenv
from backend.mongo_data_filter import filter_mongodb_records
from backend.app.error_normalizer import normalize_error
from backend.analytics_engine import (
    generate_analytics_after_chroma,
    build_analytics_report,
    read_records_from_chroma,
    query_analytics_from_chroma,
    generate_ai_insights_with_rag,
)

# Load local environment variables (including Azure OpenAI settings) from .env.
load_dotenv()


class DuplicateFileError(Exception):
    """Raised when the exact uploaded file has already been processed."""


@lru_cache(maxsize=8)
def get_chroma_collection(chroma_path, chroma_collection_name):
    """Reuse the persistent Chroma handle across Streamlit reruns."""
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(name=chroma_collection_name)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Test Report Analyzer",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME = "qa_test_analyzer"
DEFAULT_MONGO_COLLECTION = "test_execution_data"
DEFAULT_CHROMA_PATH = "./backend/chroma_db"
DEFAULT_CHROMA_COLLECTION = "test_execution_history"
DEFAULT_OUTPUT_DIR = "./backend/analytics_results"
DEFAULT_PIPELINE_LOG = "./pipeline.log"
# Upload duplicate protection is scoped to this application installation/version.
# This prevents historical data created by older project versions or bundled
# sample Chroma data from incorrectly blocking the first upload in this version.
UPLOAD_GUARD_NAMESPACE = os.getenv("UPLOAD_GUARD_NAMESPACE", "CTest4_v10")

# Pipeline audit logging. This records MongoDB and ChromaDB operation outcomes
# without changing the existing dashboard/UI behavior.
logging.basicConfig(
    filename=DEFAULT_PIPELINE_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
pipeline_logger = logging.getLogger("test_report_pipeline")


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """Convert pandas/numpy/null values into safe Python values."""
    if value is None:
        return ""

    try:
        result = pd.isna(value)
        if isinstance(result, bool) and result:
            return ""
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def make_mongo_safe(value):
    """Convert uploaded data into MongoDB-safe Python values."""
    value = clean_value(value)

    if isinstance(value, dict):
        return {str(k): make_mongo_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_mongo_safe(v) for v in value]

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    return value


def display_value(value):
    value = clean_value(value)

    if value == "":
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False)

    return str(value)


def calculate_file_hash(uploaded_file):
    """Return a SHA-256 hash of the exact uploaded file bytes."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def _normalize_preview_header(value):
    """Normalize only header spelling for column lookup; cell values are untouched."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _preview_field_aliases():
    return {
        "run_case_id": {
            "run_case_id", "runcase_id", "run_case", "run_caseid",
            "run_id", "runid", "execution_id", "execution_run_id",
        },
        "test_id": {
            "test_id", "testid", "test_case_id", "testcase_id", "case_id", "id",
        },
        "test_name": {
            "test_name", "testname", "test_case_name", "testcase_name",
            "name", "scenario", "test",
        },
        "module": {
            "module", "module_name", "component", "feature", "area", "suite",
            "test_suite", "classname", "class", "test_module", "component_name",
            "package",
        },
        "status": {
            "status", "result", "outcome", "test_status", "execution_status", "state",
        },
        "error_message": {
            "error_message", "error", "failure_message", "failure_reason",
            "message", "exception", "error_detail",
        },
        "execution_time": {
            "execution_time", "execution_date_and_time", "execution_date",
            "executed_at", "timestamp", "execution_timestamp", "run_time",
            "date", "datetime",
        },
    }


def _map_preview_columns(columns):
    """Map source headers to the seven display columns without changing cell values."""
    normalized = {_normalize_preview_header(column): column for column in columns}
    aliases = _preview_field_aliases()
    mapping = {}
    for target, names in aliases.items():
        for name in names:
            if name in normalized:
                mapping[target] = normalized[name]
                break
    return mapping


def _direct_preview_from_dataframe(dataframe):
    """Build the Show Data rows directly from the uploaded tabular data."""
    mapping = _map_preview_columns(dataframe.columns)
    target_columns = [
        "run_case_id", "test_id", "test_name", "module",
        "status", "error_message", "execution_time",
    ]

    # No filtering, deduplication, status normalization, defaults, or sorting.
    # Every source row is retained in its original order.
    rows = []
    for _, source_row in dataframe.iterrows():
        row = {}
        for target in target_columns:
            source_column = mapping.get(target)
            row[target] = source_row[source_column] if source_column is not None else ""
        rows.append(row)
    return rows


def read_uploaded_file_directly(uploaded_file):
    """Read the selected file itself for Show Data; MongoDB is never queried."""
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(
            pd.io.common.BytesIO(raw),
            dtype=object,
            keep_default_na=False,
            na_filter=False,
        )
        return _direct_preview_from_dataframe(dataframe)

    if suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(
            pd.io.common.BytesIO(raw),
            dtype=object,
        )
        return _direct_preview_from_dataframe(dataframe)

    if suffix in {".json"}:
        payload = json.loads(raw.decode("utf-8-sig"))
        if isinstance(payload, dict):
            for key in ("tests", "results", "testCases", "testcases", "executions",
                        "test_executions", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("The uploaded JSON does not contain tabular records.")
        dataframe = pd.DataFrame(payload)
        return _direct_preview_from_dataframe(dataframe)

    if suffix in {".html", ".htm"}:
        # Use Python's standard-library HTML parser so Show Data does not
        # require optional pandas HTML dependencies such as lxml. This path
        # reads the uploaded file only and never touches MongoDB.
        class _FirstTableParser(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.in_table = False
                self.in_row = False
                self.in_cell = False
                self.headers = []
                self.rows = []
                self.current_row = []
                self.current_cell = []
                self.cell_is_header = False
                self.found_table = False

            def handle_starttag(self, tag, attrs):
                tag = tag.lower()
                if tag == "table" and not self.found_table:
                    self.in_table = True
                    self.found_table = True
                elif self.in_table and tag == "tr":
                    self.in_row = True
                    self.current_row = []
                elif self.in_row and tag in {"td", "th"}:
                    self.in_cell = True
                    self.current_cell = []
                    self.cell_is_header = tag == "th"

            def handle_endtag(self, tag):
                tag = tag.lower()
                if self.in_cell and tag in {"td", "th"}:
                    value = html.unescape("".join(self.current_cell)).strip()
                    self.current_row.append((value, self.cell_is_header))
                    self.in_cell = False
                elif self.in_row and tag == "tr":
                    if self.current_row:
                        if not self.headers and any(is_header for _, is_header in self.current_row):
                            self.headers = [value for value, _ in self.current_row]
                        else:
                            self.rows.append([value for value, _ in self.current_row])
                    self.in_row = False
                elif self.in_table and tag == "table":
                    self.in_table = False

            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell.append(data)

        parser = _FirstTableParser()
        parser.feed(raw.decode("utf-8", errors="replace"))
        parser.close()

        if not parser.found_table:
            raise ValueError("No table was found in the uploaded HTML file.")

        # If the HTML table uses the first row as ordinary cells rather than
        # <th>, treat that first row as headers, matching normal tabular files.
        if not parser.headers and parser.rows:
            parser.headers = parser.rows.pop(0)

        if not parser.headers:
            raise ValueError("The uploaded HTML table does not contain column headers.")

        width = len(parser.headers)
        records = []
        for values in parser.rows:
            values = list(values)[:width] + [""] * max(0, width - len(values))
            records.append(dict(zip(parser.headers, values)))

        dataframe = pd.DataFrame(records, columns=parser.headers)
        return _direct_preview_from_dataframe(dataframe)

    if suffix == ".xml":
        import xml.etree.ElementTree as ET

        root = ET.fromstring(raw)
        records = []
        for testcase in root.iter():
            if testcase.tag.split("}")[-1].lower() != "testcase":
                continue
            row = {}
            attrs = testcase.attrib
            row["test_id"] = attrs.get("test_id", attrs.get("id", ""))
            row["test_name"] = attrs.get("name", "")
            row["module"] = attrs.get("classname", attrs.get("class", ""))
            row["execution_time"] = attrs.get("execution_time", attrs.get("timestamp", ""))
            for child in testcase:
                tag = child.tag.split("}")[-1].lower()
                if tag in {"failure", "error"}:
                    row["status"] = "FAILED"
                    row["error_message"] = child.attrib.get("message", "") or (child.text or "")
                elif tag in {"skipped", "disabled"}:
                    row["status"] = "SKIPPED"
            if "status" not in row:
                row["status"] = "PASSED"
            records.append(row)
        return records

    if suffix == ".zip":
        import zipfile

        records = []
        with zipfile.ZipFile(pd.io.common.BytesIO(raw)) as archive:
            for member in archive.namelist():
                if member.endswith(".json") and not member.endswith("/"):
                    try:
                        payload = json.loads(archive.read(member).decode("utf-8-sig"))
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        records.append(payload)
        if not records:
            raise ValueError("No JSON test records were found in the uploaded ZIP file.")
        dataframe = pd.DataFrame(records)
        return _direct_preview_from_dataframe(dataframe)

    raise ValueError(
        "Show Data supports CSV, Excel, JSON, XML, HTML, and ZIP report files."
    )


def chroma_upload_duplicate_info(chroma_path, chroma_collection_name, file_hash, record_hashes=None, source_file_name=None, guard_namespace=UPLOAD_GUARD_NAMESPACE):
    """Return durable duplicate information from ChromaDB.

    ChromaDB is the durable duplicate guard because raw MongoDB rows are
    removed after a successful sync. New uploads store the exact file hash and
    each row fingerprint in Chroma metadata. Older vectors may not have those
    fields, so this function also reconstructs row fingerprints from their
    business metadata as a backwards-compatible fallback.
    """
    record_hashes = set(str(v) for v in (record_hashes or set()) if v)
    source_key = str(source_file_name or "").strip().casefold()

    try:
        collection = get_chroma_collection(chroma_path, chroma_collection_name)
        stored = collection.get(include=["metadatas"])
        metadatas = stored.get("metadatas", []) or []
    except Exception:
        # A duplicate guard must fail closed: if Chroma cannot be checked,
        # do not allow the upload to continue because the raw MongoDB rows are
        # later deleted after Chroma persistence.
        raise RuntimeError(
            "Duplicate upload check could not access ChromaDB. "
            "Upload was blocked to prevent duplicate data."
        )

    same_file = False
    existing_hashes = set()
    existing_source_files = set()
    source_record_hashes = set()

    # Metadata added by the vector pipeline is not part of the original test
    # record and must be excluded when reconstructing legacy row fingerprints.
    vector_metadata_keys = {
        "upload_batch_id", "namespace", "embedding_model", "chunk_index",
        "chunk_count", "mongo_record_id", "file_hash", "record_hash", "upload_guard_namespace",
    }

    for metadata in metadatas:
        metadata = metadata or {}
        stored_guard_namespace = str(metadata.get("upload_guard_namespace", "") or "")
        # Only records created by this application's duplicate-guard namespace
        # participate in exact-file duplicate blocking. Older/bundled Chroma
        # records are retained for analytics but must not block a first upload.
        in_current_guard_scope = stored_guard_namespace == str(guard_namespace)

        stored_file_hash = str(metadata.get("file_hash", "") or "")
        if in_current_guard_scope and stored_file_hash and stored_file_hash == str(file_hash):
            same_file = True

        source_file = str(metadata.get("source_file", "") or "").strip().casefold()
        if source_file and in_current_guard_scope:
            existing_source_files.add(source_file)

        value = metadata.get("record_hash")
        if value:
            if in_current_guard_scope:
                existing_hashes.add(str(value))
                if source_file == source_key:
                    source_record_hashes.add(str(value))
            continue

        # Backwards compatibility for vectors created before file_hash and
        # record_hash were added to Chroma metadata.
        if in_current_guard_scope and source_file == source_key:
            legacy_record = {
                str(k): v for k, v in metadata.items()
                if k not in vector_metadata_keys and not str(k).startswith("_")
            }
            if legacy_record:
                try:
                    source_record_hashes.add(record_fingerprint(legacy_record))
                except Exception:
                    pass

    duplicate_record_count = len(record_hashes.intersection(existing_hashes))
    legacy_duplicate_count = len(record_hashes.intersection(source_record_hashes))

    # Exact file hash is definitive for new uploads.
    if same_file:
        return {"same_file": True, "duplicate_records": len(record_hashes)}

    # Older Chroma records do not contain file_hash. If the same source filename
    # exists and every row fingerprint from the selected report is already in
    # that file's historical vectors, treat it as the same file. This prevents
    # re-uploading files created before durable file-hash metadata existed.
    same_source_file = bool(source_key and source_key in existing_source_files)
    if same_source_file and record_hashes and legacy_duplicate_count >= len(record_hashes):
        return {"same_file": True, "duplicate_records": len(record_hashes)}

    return {
        "same_file": False,
        "duplicate_records": max(duplicate_record_count, legacy_duplicate_count),
    }

def load_uploaded_data_from_mongodb(mongo_uri, db_name, collection_name, file_hash):
    """Retrieve the selected uploaded file's normalized records from MongoDB."""
    client = get_mongo_client(mongo_uri)
    try:
        collection = client[db_name][collection_name]
        cursor = collection.find({"file_hash": file_hash}).sort("uploaded_at", 1)
        records = []
        for raw in cursor:
            record = dict(raw)
            record.pop("_id", None)
            records.append(record)
        return records
    finally:
        client.close()


def canonical_record(record):
    """Create a stable representation of a parsed test row.

    Database/upload metadata is excluded. This is used for exact-row fallback
    hashing only.
    """
    if not isinstance(record, dict):
        record = {"value": record}
    safe = make_mongo_safe(record)
    safe = {
        str(k): v for k, v in safe.items()
        if k not in {"_id", "source_file", "file_hash", "upload_batch_id",
                     "uploaded_at", "record_hash", "duplicate_group",
                     "duplicate_count", "is_duplicate"}
        and not str(k).startswith("_")
    }
    return json.dumps(safe, sort_keys=True, ensure_ascii=False,
                      default=str, separators=(",", ":"))


def record_fingerprint(record):
    return hashlib.sha256(canonical_record(record).encode("utf-8")).hexdigest()


def record_identity(record):
    """Return the identity of a test row *within one uploaded file*.

    Execution result fields such as status, error message, duration and
    execution time are deliberately excluded. They can legitimately change
    between uploads. The same test identity appearing twice in one file is a
    duplicate and only the first occurrence is kept.
    """
    if not isinstance(record, dict):
        return ("value", str(record))

    def clean(value):
        return str(value or "").strip().casefold()

    test_id = clean(record.get("test_id"))
    test_name = clean(record.get("test_name"))
    module = clean(record.get("module"))

    # Prefer the stable test id. If it is absent, use test name + module.
    if test_id:
        return ("test_id", test_id, module)
    return ("test", test_name, module)


def remove_duplicate_records(records):
    """Keep only the first occurrence of each test identity in this file."""
    unique = []
    seen = set()

    for record in records:
        identity = record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(record)

    return unique, len(records) - len(unique)


def tag_duplicate_records(records):
    """Return the unique rows for the preview without exposing duplicate info."""
    unique_records, _ = remove_duplicate_records(records)
    tagged = []
    for record in unique_records:
        tagged_record = dict(record) if isinstance(record, dict) else {"value": record}
        tagged_record["record_hash"] = record_fingerprint(record)
        tagged_record["is_duplicate"] = False
        tagged.append(tagged_record)
    return tagged, len(tagged), 0

def create_document(record):
    """Convert a filtered record into searchable RAG text without raw stack traces."""
    preferred_fields = (
        "run_case_id", "test_case_id", "test_case_name", "module_name",
        "status", "error_message", "error_category", "error_signature",
        "secondary_errors", "execution_date_and_time",
    )
    parts = []
    for key in preferred_fields:
        if key not in record:
            continue
        value = clean_value(record.get(key))
        if value == "":
            continue
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value if item)
        parts.append(f"{str(key).replace('_', ' ')}: {value}")
    return ". ".join(parts)

from backend.app.parsers.report_parser import parse_uploaded_report



# ============================================================
# MONGODB FUNCTIONS
# ============================================================

def get_mongo_client(mongo_uri):
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
    )
    client.admin.command("ping")
    return client


def _ensure_file_scoped_unique_record_index(collection):
    """Ensure MongoDB uniqueness is scoped to one uploaded file."""
    # Remove indexes created by earlier versions.
    for index_name in ("uniq_record_hash", "uniq_file_record_hash"):
        try:
            collection.drop_index(index_name)
        except Exception:
            pass

    # Existing documents get a deterministic identity hash. Legacy documents
    # without file scope are kept in isolated legacy scopes.
    for doc in collection.find({}):
        updates = {}
        identity = record_identity(doc)
        identity_hash = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if doc.get("record_identity_hash") != identity_hash:
            updates["record_identity_hash"] = identity_hash
        if not doc.get("file_hash"):
            updates["file_hash"] = f"legacy:{doc.get('_id')}"
        if updates:
            collection.update_one({"_id": doc["_id"]}, {"$set": updates})

    # Clean only duplicates that are duplicates inside the SAME file.
    seen = set()
    duplicate_ids = []
    for doc in collection.find({}, {"_id": 1, "file_hash": 1, "record_identity_hash": 1}):
        key = (doc.get("file_hash"), doc.get("record_identity_hash"))
        if key in seen:
            duplicate_ids.append(doc["_id"])
        else:
            seen.add(key)
    if duplicate_ids:
        collection.delete_many({"_id": {"$in": duplicate_ids}})

    collection.create_index(
        [("file_hash", 1), ("record_identity_hash", 1)],
        unique=True,
        name="uniq_file_record_identity",
    )

def upload_records_to_mongodb(
    mongo_uri, db_name, collection_name, records, source_file_name, file_hash, guard_namespace=UPLOAD_GUARD_NAMESPACE
):
    """Insert unique test identities, with uniqueness scoped to this upload.

    Same test in a different upload is allowed. A repeated test identity in
    the same upload is silently skipped.
    """
    if not records:
        raise ValueError("The uploaded file contains no records.")

    client = get_mongo_client(mongo_uri)
    try:
        collection = client[db_name][collection_name]
        _ensure_file_scoped_unique_record_index(collection)

        unique_records, skipped_in_file = remove_duplicate_records(records)
        prepared = []
        for record in unique_records:
            safe = make_mongo_safe(record if isinstance(record, dict) else {"value": record})
            identity = record_identity(safe)
            identity_hash = hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            safe["record_identity_hash"] = identity_hash
            safe["record_hash"] = record_fingerprint(safe)
            safe["source_file"] = source_file_name
            safe["file_hash"] = file_hash
            safe["upload_guard_namespace"] = guard_namespace
            prepared.append(safe)

        if not prepared:
            return {"batch_id": None, "inserted_count": 0, "source_file": source_file_name,
                    "file_hash": file_hash, "skipped_duplicates": skipped_in_file}

        # Do not look globally for record_hash. Only the compound database
        # constraint (file_hash, record_identity_hash) determines duplicates.
        batch_id = str(uuid4())
        uploaded_at = datetime.now(timezone.utc).isoformat()
        for doc in prepared:
            doc["upload_batch_id"] = batch_id
            doc["uploaded_at"] = uploaded_at

        try:
            result = collection.insert_many(prepared, ordered=False)
            inserted_count = len(result.inserted_ids)
            pipeline_logger.info(
                "MongoDB insert SUCCESS | batch=%s | inserted=%s | collection=%s",
                batch_id, inserted_count, collection_name,
            )
        except Exception as exc:
            if exc.__class__.__name__ != "BulkWriteError":
                pipeline_logger.exception(
                    "MongoDB insert FAILURE | batch=%s | collection=%s",
                    batch_id, collection_name,
                )
                raise
            details = getattr(exc, "details", {}) or {}
            errors = details.get("writeErrors", [])
            if any(error.get("code") != 11000 for error in errors):
                collection.delete_many({"upload_batch_id": batch_id})
                pipeline_logger.exception(
                    "MongoDB insert FAILURE | batch=%s | collection=%s",
                    batch_id, collection_name,
                )
                raise
            inserted_count = int(details.get("nInserted", 0))
            pipeline_logger.info(
                "MongoDB insert completed with duplicate skips | batch=%s | inserted=%s",
                batch_id, inserted_count,
            )

        if inserted_count == 0:
            batch_id = None

        return {
            "batch_id": batch_id,
            "inserted_count": inserted_count,
            "source_file": source_file_name,
            "file_hash": file_hash,
            "skipped_duplicates": len(records) - inserted_count,
            "skipped_in_file": skipped_in_file,
        }
    finally:
        client.close()

# ============================================================
# TEXT PREPROCESSING / EMBEDDING CONFIGURATION
# ============================================================

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

_embedding_model = None


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split searchable text into overlapping word-based chunks.

    The overlap keeps context that sits at a chunk boundary. Empty chunks
    are ignored. The MongoDB record itself remains the source of truth; each
    chunk only becomes a vector representation for retrieval.
    """
    text = str(text or "").strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    step = chunk_size - chunk_overlap

    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


def get_embedding_model():
    """Load the embedding model lazily so the existing UI/startup is unchanged."""
    global _embedding_model

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding support is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        _embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    return _embedding_model


def generate_embeddings(text_chunks):
    """Generate one normalized dense embedding for every text chunk."""
    if not text_chunks:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        text_chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def build_namespace(chroma_collection_name, source_file_name):
    """Return the ChromaDB namespace derived from the uploaded file name."""
    file_name = Path(str(source_file_name or "")).name
    file_stem = Path(file_name).stem
    return f"{chroma_collection_name}::{file_stem}"


def _build_current_chromadb_snapshot_records(chroma_collection):
    """Read the complete current ChromaDB collection for debug snapshots.

    The human-readable snapshot must describe what is actually persisted in
    ChromaDB, not only the batch that was ingested most recently.  This keeps
    CHROMADB_DATA.md aligned with collection.count().
    """
    stored = chroma_collection.get(include=["documents", "metadatas"])
    ids = stored.get("ids", []) or []
    documents = stored.get("documents", []) or []
    metadatas = stored.get("metadatas", []) or []

    grouped = {}
    for index, vector_id in enumerate(ids):
        metadata = (metadatas[index] if index < len(metadatas) else {}) or {}
        document = documents[index] if index < len(documents) else ""
        mongo_record_id = str(metadata.get("mongo_record_id", "") or "")
        if not mongo_record_id:
            # Older vectors may not have mongo_record_id metadata. Keep them
            # visible rather than dropping persisted ChromaDB data.
            mongo_record_id = f"vector:{vector_id}"

        record = grouped.setdefault(
            mongo_record_id,
            {
                "mongo_record_id": mongo_record_id,
                "original_text": document,
                "chunk_count": 0,
                "chunks": [],
            },
        )

        chunk_index = metadata.get("chunk_index", len(record["chunks"]))
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index = len(record["chunks"])

        record["chunks"].append(
            {
                "chunk_index": chunk_index,
                "chroma_vector_id": vector_id,
                "namespace": metadata.get("namespace", ""),
                "text": document,
                "word_count": len((document or "").split()),
                "embedding_model": metadata.get(
                    "embedding_model", DEFAULT_EMBEDDING_MODEL
                ),
                "metadata": metadata,
            }
        )
        record["chunk_count"] = len(record["chunks"])

    records = list(grouped.values())
    for record in records:
        record["chunks"].sort(key=lambda item: item.get("chunk_index", 0))
        if record["chunks"]:
            record["original_text"] = " ".join(
                chunk.get("text", "") for chunk in record["chunks"]
            ).strip()
    return records


def write_chromadb_snapshot_markdown(
    debug_records,
    batch_id,
    source_file,
    namespace,
    embedding_model,
    chroma_path,
    chroma_collection,
    embedding_dimension,
):
    """Write a human-readable snapshot of the complete persisted ChromaDB collection.

    ``debug_records`` is retained as a fallback for compatibility, but the
    snapshot is rebuilt from the actual ChromaDB collection whenever possible.
    This means that after upload 1 (10 vectors) and upload 2 (10 more vectors),
    the snapshot reports all 20 persisted vectors instead of replacing the
    first batch with the second batch.
    """
    try:
        snapshot_records = _build_current_chromadb_snapshot_records(chroma_collection)
    except Exception:
        # Snapshot generation must never make a successful ingestion fail.
        pipeline_logger.exception(
            "ChromaDB snapshot read failed; falling back to current batch debug records."
        )
        snapshot_records = debug_records

    collection_name = getattr(chroma_collection, "name", str(chroma_collection))
    collection_count = chroma_collection.count()
    total_chunks = sum(len(record.get("chunks", [])) for record in snapshot_records)
    batch_vector_count = sum(len(record.get("chunks", [])) for record in debug_records)

    # Create a human-readable Markdown snapshot inside the project.
    # IMPORTANT: this represents the complete current ChromaDB collection,
    # not only the latest upload batch.
    generated_at = datetime.now(timezone.utc).isoformat()
    markdown_path = Path("./backend/CHROMADB_DATA.md")
    md = [
        "# ChromaDB Data Snapshot",
        "",
        "> This file is a human-readable snapshot of the MongoDB → chunk → embedding → ChromaDB pipeline.",
        "> It is generated automatically after a successful ChromaDB ingestion.",
        "> The snapshot below represents the complete current ChromaDB collection; ChromaDB itself remains the source of vector data.",
        "",
        f"- **Generated:** {generated_at}",
        f"- **Latest source file:** `{source_file}`",
        f"- **Latest batch ID:** `{batch_id}`",
        f"- **ChromaDB path:** `{chroma_path}`",
        f"- **Collection:** `{collection_name}`",
        f"- **Embedding model:** `{embedding_model}`",
        f"- **Embedding dimensions:** `{embedding_dimension}`",
        f"- **Chunk size:** `{DEFAULT_CHUNK_SIZE}` words",
        f"- **Chunk overlap:** `{DEFAULT_CHUNK_OVERLAP}` words",
        f"- **Latest batch vectors:** `{batch_vector_count}`",
        f"- **Total persisted ChromaDB vectors:** `{collection_count}`",
        f"- **Total source records represented:** `{len(snapshot_records)}`",
        "",
        "## Complete ChromaDB Collection",
        "",
    ]
    for record_no, record in enumerate(snapshot_records, 1):
        md.extend([
            f"### Record {record_no}",
            "",
            f"- **MongoDB Record ID:** `{record.get('mongo_record_id', '')}`",
            f"- **Chunk count:** `{record.get('chunk_count', 0)}`",
            "",
        ])
        for chunk in record.get("chunks", []):
            metadata = chunk.get("metadata", {}) or {}
            stored_error = str(
                metadata.get("raw_error_message")
                or metadata.get("error_message", "")
                or ""
            ).strip()
            error_message = normalize_error(stored_error)["error_message"]
            chunk_text = str(chunk.get("text", "") or "")
            if error_message in {"", "—", "-"}:
                match = re.search(
                    r"(?:^|\.\s)error message:\s*(.*?)(?:\.\s+error category:|$)",
                    chunk_text,
                    flags=re.IGNORECASE,
                )
                candidate = match.group(1).strip() if match else ""
                if candidate and candidate not in {"—", "-"}:
                    error_message = normalize_error(candidate)["error_message"]
            md.extend([
                f"#### Chunk {chunk.get('chunk_index', '')}",
                "",
                f"- **Chroma vector ID:** `{chunk.get('chroma_vector_id', '')}`",
                f"- **Word count:** `{chunk.get('word_count', '')}`",
                f"- **Run Case ID:** `{metadata.get('run_case_id', '')}`",
                f"- **Test ID:** `{metadata.get('test_case_id', metadata.get('test_id', ''))}`",
                f"- **Test name:** {metadata.get('test_case_name', metadata.get('test_name', ''))}",
                f"- **Status:** `{metadata.get('status', '')}`",
                f"- **Module:** `{metadata.get('module_name', metadata.get('module', ''))}`",
                f"- **Error Message:** {error_message}",
                f"- **Execution Date and Time:** `{metadata.get('execution_date_and_time', '')}`",
                f"- **Upload Batch ID:** `{metadata.get('upload_batch_id', '')}`",
                f"- **Namespace:** `{chunk.get('namespace', '')}`",
                "",
                "**Chunk text:**",
                "",
                f"> {chunk.get('text', '').replace(chr(10), ' ')}",
                "",
                "**Metadata:**",
                "",
                "```json",
                json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ])
    markdown_path.write_text("\n".join(md), encoding="utf-8")
    return None


def delete_raw_mongodb_batch_after_chroma(
    mongo_uri,
    db_name,
    mongo_collection_name,
    batch_id,
    expected_record_ids,
):
    """Delete raw MongoDB records only after ChromaDB storage is verified."""
    client = get_mongo_client(mongo_uri)
    try:
        collection = client[db_name][mongo_collection_name]
        expected_ids = list(expected_record_ids)
        if not expected_ids:
            pipeline_logger.warning(
                "MongoDB raw-data deletion skipped: no source record IDs for batch %s.",
                batch_id,
            )
            return {
                "deletion_status": "skipped",
                "deleted_count": 0,
                "remaining_count": 0,
                "expected_count": 0,
            }

        delete_result = collection.delete_many({"_id": {"$in": expected_ids}})
        remaining_count = collection.count_documents({"_id": {"$in": expected_ids}})
        deletion_confirmed = (
            delete_result.deleted_count == len(expected_ids)
            and remaining_count == 0
        )

        if deletion_confirmed:
            pipeline_logger.info(
                "MongoDB raw-data deletion SUCCESS | batch=%s | deleted=%s | remaining=%s",
                batch_id, delete_result.deleted_count, remaining_count,
            )
        else:
            pipeline_logger.error(
                "MongoDB raw-data deletion FAILURE | batch=%s | expected=%s | deleted=%s | remaining=%s",
                batch_id, len(expected_ids), delete_result.deleted_count, remaining_count,
            )

        return {
            "deletion_status": "confirmed" if deletion_confirmed else "failed",
            "deleted_count": delete_result.deleted_count,
            "remaining_count": remaining_count,
            "expected_count": len(expected_ids),
        }
    except Exception:
        pipeline_logger.exception(
            "MongoDB raw-data deletion ERROR | batch=%s", batch_id
        )
        return {
            "deletion_status": "failed",
            "deleted_count": 0,
            "remaining_count": len(expected_record_ids),
            "expected_count": len(expected_record_ids),
        }
    finally:
        client.close()


def sync_uploaded_batch_to_chroma(
    mongo_uri,
    db_name,
    mongo_collection_name,
    batch_id,
    source_file_name,
    chroma_path,
    chroma_collection_name,
):
    """
    MongoDB -> ChromaDB preprocessing pipeline.

    Existing MongoDB behavior is intentionally untouched. After MongoDB has
    successfully stored the upload, this function:

      1. converts each MongoDB record to searchable text,
      2. chunks that text into manageable overlapping segments,
      3. generates an embedding for every chunk,
      4. assigns a stable namespace to the upload,
      5. stores vectors + metadata in the configured ChromaDB collection,
      6. verifies every expected vector exists in ChromaDB,
      7. retains the corresponding raw MongoDB records (they are not deleted).

    The original MongoDB record identity is retained in every chunk's metadata,
    allowing downstream analytics to continue to operate on the original
    structured records rather than on individual chunks.
    """
    client = get_mongo_client(mongo_uri)

    try:
        collection = client[db_name][mongo_collection_name]
        records = list(collection.find({"upload_batch_id": batch_id}))

        if not records:
            pipeline_logger.error(
                "MongoDB read FAILURE | raw records not found | batch=%s", batch_id
            )
            raise ValueError(
                "The uploaded records were not found in MongoDB "
                f"for batch {batch_id}."
            )

        pipeline_logger.info(
            "MongoDB read SUCCESS | batch=%s | raw_records=%s", batch_id, len(records)
        )

        # Filter MongoDB documents to the seven approved business columns
        # before any chunking/embedding. The original records remain available
        # only for source identity and post-Chroma raw-data deletion.
        clean_records = filter_mongodb_records(records)
        pipeline_logger.info(
            "MongoDB retrieval filter SUCCESS | batch=%s | clean_records=%s | columns=%s",
            batch_id,
            len(clean_records),
            7,
        )

        chroma_collection = get_chroma_collection(chroma_path, chroma_collection_name)

        namespace = build_namespace(chroma_collection_name, source_file_name)

        ids = []
        documents = []
        metadatas = []
        chunk_sources = []
        debug_records = []
        seen_ids = set()
        skipped_duplicates = 0
        skipped_empty = 0

        # ------------------------------------------------------------
        # 1. TEXT PREPROCESSING + CHUNKING
        # ------------------------------------------------------------
        for source_record, record in zip(records, clean_records):
            mongo_id = source_record.get("_id")

            if mongo_id is None:
                raise ValueError("A MongoDB record does not contain _id.")

            # `record` contains only the approved seven MongoDB fields.
            source_document = create_document(record)
            if not source_document.strip():
                skipped_empty += 1
                continue

            chunks = chunk_text(source_document)

            debug_record = {
                "mongo_record_id": str(mongo_id),
                "original_text": source_document,
                "chunk_count": len(chunks),
                "chunks": [],
            }
            debug_records.append(debug_record)

            for chunk_index, chunk in enumerate(chunks):
                # A MongoDB record can create multiple vectors. The chunk
                # index makes every vector ID deterministic and unique.
                chroma_id = (
                    f"mongo_{str(mongo_id)}_chunk_{chunk_index}"
                )

                if chroma_id in seen_ids:
                    skipped_duplicates += 1
                    continue

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

                # Persist durable upload fingerprints in ChromaDB. These are
                # required because the raw MongoDB rows are deleted after a
                # successful Chroma sync.
                metadata["file_hash"] = str(source_record.get("file_hash", "") or "")
                metadata["record_hash"] = str(source_record.get("record_hash", "") or "")
                metadata["upload_guard_namespace"] = str(source_record.get("upload_guard_namespace", "") or "")
                metadata["source_file"] = str(source_record.get("source_file", source_file_name) or source_file_name)

                # Vector organization metadata.
                #
                # IMPORTANT: upload_batch_id is pipeline/system metadata, not
                # MongoDB business data. The MongoDB filter above intentionally
                # removes it from the clean seven-column dataset before
                # chunking. We add the batch id back only as ChromaDB routing
                # metadata so Analytics Engine can retrieve the just-uploaded
                # batch after the raw MongoDB records are deleted.
                metadata["upload_batch_id"] = str(batch_id)
                metadata["namespace"] = namespace
                metadata["embedding_model"] = DEFAULT_EMBEDDING_MODEL
                metadata["chunk_index"] = chunk_index
                metadata["chunk_count"] = len(chunks)
                metadata["mongo_record_id"] = str(mongo_id)

                seen_ids.add(chroma_id)
                ids.append(chroma_id)
                documents.append(chunk)
                metadatas.append(metadata)
                chunk_sources.append(str(mongo_id))
                debug_record["chunks"].append({
                    "chunk_index": chunk_index,
                    "chroma_vector_id": chroma_id,
                    "namespace": namespace,
                    "text": chunk,
                    "word_count": len(chunk.split()),
                    "embedding_model": DEFAULT_EMBEDDING_MODEL,
                    "metadata": metadata,
                })

        if not ids:
            raise ValueError("No valid text chunks were available for ChromaDB.")

        # ------------------------------------------------------------
        # 2. EMBEDDING GENERATION
        # ------------------------------------------------------------
        embeddings = generate_embeddings(documents)

        if len(embeddings) != len(documents):
            raise ValueError(
                "Embedding generation failed: the number of embeddings does "
                "not match the number of text chunks."
            )

        # ------------------------------------------------------------
        # 3. VECTOR STORAGE
        # ------------------------------------------------------------
        try:
            chroma_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            # Confirm persistence before touching MongoDB.
            persisted = chroma_collection.get(ids=ids, include=["metadatas"])
            persisted_ids = set(persisted.get("ids", []) or [])
            missing_ids = [vector_id for vector_id in ids if vector_id not in persisted_ids]
            if missing_ids:
                raise RuntimeError(
                    f"ChromaDB verification failed: {len(missing_ids)} vector(s) are missing."
                )

            pipeline_logger.info(
                "ChromaDB insertion SUCCESS | batch=%s | vectors=%s | collection=%s",
                batch_id, len(ids), chroma_collection_name,
            )
        except Exception:
            pipeline_logger.exception(
                "ChromaDB insertion/verification FAILURE | batch=%s | vectors=%s",
                batch_id, len(ids),
            )
            # Deliberately do not delete MongoDB records here.
            raise

        # ChromaDB is now verified. Raw MongoDB records are intentionally
        # retained (not deleted) so MongoDB continues to hold the source data
        # even after it has been vectorized into ChromaDB.
        raw_record_ids = [
            record.get("_id") for record in records if record.get("_id") is not None
        ]
        deletion_result = {
            "deletion_status": "retained",
            "deleted_count": 0,
            "remaining_count": len(raw_record_ids),
            "expected_count": len(raw_record_ids),
        }

        embedding_dimension = len(embeddings[0]) if embeddings else 0
        write_chromadb_snapshot_markdown(
            debug_records=debug_records,
            batch_id=batch_id,
            source_file=str(records[0].get("source_file", "")),
            namespace=namespace,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            chroma_path=chroma_path,
            # Pass the actual ChromaDB Collection object. The snapshot
            # generator reads persisted vectors and calls Collection.count();
            # passing the collection name string here causes count() to fail.
            chroma_collection=chroma_collection,
            embedding_dimension=embedding_dimension,
        )

        unique_source_records = len(set(chunk_sources))

        return {
            # Existing UI fields are retained.
            "mongo_records": len(records),
            "records_sent": len(ids),
            "chroma_total": chroma_collection.count(),
            "skipped_duplicates": skipped_duplicates,
            "skipped_empty": skipped_empty,

            # Additional processing information for logs/downstream use.
            "chunks_created": len(ids),
            "source_records_vectorized": unique_source_records,
            "namespace": namespace,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "chroma_path": chroma_path,
            "chroma_collection": chroma_collection_name,
            "mongo_raw_deletion": deletion_result,
        }
    finally:
        client.close()




# ============================================================
# PAGE NAVIGATION
# ============================================================
# Use Streamlit query parameters as a lightweight route so navigation from
# the upload page to the dashboard is reliable across reruns.  The session
# state still controls rendering, while the URL provides a stable page target.
_requested_page = st.query_params.get("page", "upload")
if _requested_page == "dashboard":
    st.session_state.current_page = 2
elif "current_page" not in st.session_state:
    st.session_state.current_page = 1


def render_page_navigation():
    """Render a right/left center navigation button for the two-page UI."""
    if st.session_state.current_page == 1:
        st.markdown("""
        <style>
        div.st-key-page_next {
            position: fixed;
            right: 1.1rem;
            bottom: 1.1rem;
            z-index: 999999;
        }
        @media (max-width: 900px) {
            div.st-key-page_next {
                right: 0.7rem;
                bottom: 0.7rem;
            }
        }
        div.st-key-page_next button {
            border-radius: 10px;
            font-size: 15px;
            font-weight: 700;
            padding: 10px 18px;
            box-shadow: 0 4px 16px rgba(0,0,0,.18);
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Go to Analytics Dashboard", key="page_next"):
            st.session_state.current_page = 2
            st.query_params["page"] = "dashboard"
            st.rerun()

    else:
        st.markdown("""
        <style>
        div.st-key-page_previous {
            position: fixed;
            left: 1.1rem;
            bottom: 1.1rem;
            z-index: 999999;
        }
        @media (max-width: 900px) {
            div.st-key-page_previous {
                left: 0.7rem;
                bottom: 0.7rem;
            }
        }
        div.st-key-page_previous button {
            border-radius: 10px;
            font-size: 15px;
            font-weight: 700;
            padding: 10px 18px;
            box-shadow: 0 4px 16px rgba(0,0,0,.18);
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Back to Upload Test Data", key="page_previous"):
            st.session_state.current_page = 1
            st.query_params["page"] = "upload"
            st.rerun()



def _week_label_for_record(record):
    """Return an ISO week label from the record's execution date."""
    from backend.analytics_engine import FIELD_ALIASES, first_value, parse_datetime
    value = first_value(record, FIELD_ALIASES["execution_time"])
    dt = parse_datetime(value)
    if not dt:
        dt = parse_datetime(record.get("uploaded_at"))
    if not dt:
        return "Unknown week"
    iso = dt.isocalendar()
    return f"week{iso.week}"


def render_configuration_sidebar(page: int):
    """Render the configuration section appropriate to the current page."""
    with st.sidebar:
        st.markdown(
            """
            <style>
            div[data-testid="stSidebar"] .config-title {
                font-size: 1.05rem;
                font-weight: 700;
                margin: 0.25rem 0 0.35rem 0;
            }
            div[data-testid="stSidebar"] [data-testid="stExpander"] {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 10px;
                overflow: hidden;
            }
            div[data-testid="stSidebar"] [data-testid="stExpander"] summary {
                font-weight: 700;
            }
            div[data-testid="stSidebar"] .config-header {
                font-size: 0.78rem;
                font-weight: 700;
                opacity: 0.72;
                padding-bottom: 0.2rem;
            }
            div[data-testid="stSidebar"] .config-key {
                font-size: 0.86rem;
                font-weight: 600;
                padding-top: 0.45rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="config-title">⚙️ Configuration</div>', unsafe_allow_html=True)

        with st.expander("Database & Storage Configuration", expanded=False):
            key_col, value_col = st.columns([0.95, 1.7])
            with key_col:
                st.markdown('<div class="config-header">Key</div>', unsafe_allow_html=True)
            with value_col:
                st.markdown('<div class="config-header">Value</div>', unsafe_allow_html=True)

            if page == 1:
                fields = [
                    ("MongoDB URI", "MongoDB URI", DEFAULT_MONGO_URI, "config_mongo_uri"),
                    ("MongoDB Database", "MongoDB Database", DEFAULT_DB_NAME, "config_db_name"),
                    ("MongoDB Collection", "MongoDB Collection", DEFAULT_MONGO_COLLECTION, "config_mongo_collection"),
                    ("ChromaDB Path", "ChromaDB Path", DEFAULT_CHROMA_PATH, "config_chroma_path"),
                    ("ChromaDB Collection", "ChromaDB Collection", DEFAULT_CHROMA_COLLECTION, "config_chroma_collection"),
                ]
            else:
                fields = [
                    ("Analytics Report", "Analytics Report", "./analytics_results", "config_analytics_report"),
                ]

            for key_label, input_label, default_value, state_key in fields:
                key_col, value_col = st.columns([0.95, 1.7])
                with key_col:
                    st.markdown(f'<div class="config-key">{key_label}</div>', unsafe_allow_html=True)
                with value_col:
                    st.text_input(
                        input_label,
                        value=default_value,
                        label_visibility="collapsed",
                        key=state_key,
                    )



def _normalized_json_records(report):
    """Return the self-contained current-upload dataset embedded in the analytics JSON."""
    records = report.get("records") or []
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, dict)]


def _load_dashboard_data_from_chroma(chroma_path, chroma_collection_name):
    """Load persisted dashboard records directly from ChromaDB.

    ChromaDB is the durable post-ingestion source because the upload pipeline
    deletes the corresponding raw MongoDB records only after the ChromaDB
    write has been verified. A fresh Streamlit session therefore cannot depend
    on MongoDB to rebuild Page 2.

    ChromaDB stores one metadata entry per chunk, so ``read_records_from_chroma``
    collapses those chunks back to one logical execution record before the
    dashboard builds its filter catalog and analytics report.
    """
    records = read_records_from_chroma(
        chroma_path=chroma_path,
        chroma_collection_name=chroma_collection_name,
    )

    if not records:
        return [], None

    from backend.analytics_engine import build_analytics_report

    # Keep the report self-contained and compatible with the existing Page 2
    # filter/report contract. All persisted executions are used as the
    # dashboard's available history after a reload.
    first_record = records[0]
    week_label = _week_label_for_record(first_record)
    report = build_analytics_report(
        current_records=records,
        historical_records=records,
        week_label=week_label,
        source_file="Persisted ChromaDB History",
        upload_batch_id="persisted_chroma_history",
    )
    report["analytics_source"] = {
        "structured_source": "ChromaDB metadata",
        "vector_store": "ChromaDB",
        "persisted_dashboard_records": len(records),
        "reload_source": "ChromaDB",
    }
    return records, report


def _hydrate_dashboard_from_chroma():
    """Hydrate Page 2 from the persistent ChromaDB store.

    Session state is only a cache. Whenever a new Streamlit session reaches
    Page 2, the filter catalog and available records are reconstructed from
    ChromaDB, so reloads/revisits behave exactly like the post-upload state.
    """
    chroma_path = st.session_state.get("config_chroma_path", DEFAULT_CHROMA_PATH)
    chroma_collection_name = st.session_state.get(
        "config_chroma_collection",
        DEFAULT_CHROMA_COLLECTION,
    )
    signature = (chroma_path, chroma_collection_name)

    if (
        st.session_state.get("dashboard_chroma_signature") == signature
        and st.session_state.get("dashboard_chroma_hydrated")
    ):
        return

    records, report = _load_dashboard_data_from_chroma(
        chroma_path,
        chroma_collection_name,
    )

    # Rebuild all session-only dashboard state from the durable vector store.
    st.session_state.dashboard_filter_catalog = {
        key: [] for key in DASHBOARD_FILTER_KEYS
    }
    st.session_state.dashboard_filter_date_min = None
    st.session_state.dashboard_filter_date_max = None
    st.session_state.dashboard_all_records = []
    st.session_state.dashboard_uploaded_batches = []
    st.session_state.dashboard_reports = []
    st.session_state.dashboard_chroma_signature = signature
    st.session_state.dashboard_chroma_hydrated = True

    if report and records:
        _update_dashboard_filter_catalog(report)
        st.session_state.analytics_result = report
        st.session_state.analytics_result_path = None

        # A reload restores the available data and filters, but does not
        # display a stale filtered report from a previous browser session.
        # The user can generate a fresh report from the restored Chroma data.
        st.session_state.dashboard_report_generated = False
        st.session_state.dashboard_filtered_records = []
        st.session_state.dashboard_filter_snapshot = {}
        st.session_state.dashboard_chroma_analytics = {}
        st.session_state.dashboard_rag_context = {}
        st.session_state.dashboard_chroma_source = {}
    else:
        st.session_state.analytics_result = None
        st.session_state.dashboard_report_generated = False
        st.session_state.dashboard_filtered_records = []
        st.session_state.dashboard_filter_snapshot = {}
        st.session_state.dashboard_chroma_analytics = {}


# ============================================================
# DYNAMIC DASHBOARD FILTER CATALOG
# ============================================================
# Filter values are intentionally accumulated for the lifetime of the
# Streamlit session. Each newly generated analytics JSON contributes only
# new/unique values; previously discovered values are never removed.
# Status is a fixed business-level filter with two options:
#   Complete Report -> PASSED + FAILED
#   Failed          -> FAILED only

DASHBOARD_FILTER_KEYS = (
    "status",
    "module",
)


def _ensure_dashboard_filter_catalog():
    """Initialize the cumulative Page 2 filter catalog once per session."""
    if "dashboard_filter_catalog" not in st.session_state:
        st.session_state.dashboard_filter_catalog = {
            key: [] for key in DASHBOARD_FILTER_KEYS
        }
        st.session_state.dashboard_filter_date_min = None
        st.session_state.dashboard_filter_date_max = None
        st.session_state.dashboard_all_records = []
        st.session_state.dashboard_uploaded_batches = []
        st.session_state.dashboard_reports = []


def _reset_dashboard_filter_widget_state():
    """Reset only the current selections after a new upload, not the catalog."""
    for key in (
        "dashboard_from_date",
        "dashboard_to_date",
        "dashboard_status",
        "dashboard_module",
    ):
        st.session_state.pop(key, None)


def _record_fingerprint(record):
    """Create a stable fingerprint so cumulative dashboard data has no duplicates."""
    return hashlib.sha256(
        canonical_record(record).encode("utf-8")
    ).hexdigest()


def _add_unique_values(existing, values):
    """Append only values not already present, preserving discovery order."""
    seen = set(existing)
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            existing.append(value)
            seen.add(value)
    existing.sort(key=lambda item: item.casefold())
    return existing


def _update_dashboard_filter_catalog(report):
    """Merge a newly generated analytics JSON into the cumulative dashboard catalog."""
    _ensure_dashboard_filter_catalog()

    records = _normalized_json_records(report)
    catalog = st.session_state.dashboard_filter_catalog

    # Prefer the normalized records as the source of truth. This also keeps
    # the dashboard compatible with older JSON files that do not contain
    # filter_options.
    # Status is a business-level filter, not the raw execution status.
    # A passed execution belongs to "Complete Report"; a failed execution
    # belongs to both "Complete Report" and "Failed".  Keep the UI options
    # fixed so raw values such as PASSED/FAILED/SKIPPED never leak into Page 2.
    status_filter_values = set()
    for record in records:
        raw_status = str(record.get("status", "")).strip().upper()
        if raw_status in {"PASSED", "FAILED"}:
            status_filter_values.add("Complete Report")
        if raw_status == "FAILED":
            status_filter_values.add("Failed")

    derived_values = {
        "test_id": {str(r.get("test_id", "")).strip() for r in records},
        "status": status_filter_values,
        "module": {str(r.get("module", "")).strip() for r in records},
        "environment": {str(r.get("environment", "")).strip() for r in records},
        "build_version": {str(r.get("version", "")).strip() for r in records},
    }

    for key in DASHBOARD_FILTER_KEYS:
        _add_unique_values(catalog[key], derived_values[key])

    # Date/time are range controls rather than dropdowns. Keep the earliest
    # and latest execution timestamps seen across every uploaded file.
    # Fall back to the upload timestamp when a record has no execution_time,
    # so the fields still enable/populate whenever the DB has data.
    from backend.analytics_engine import parse_datetime
    execution_aliases = (
        "execution_time", "execution_date_and_time", "Execution Date and Time",
        "execution_date", "Execution Date", "executed_at", "timestamp",
        "execution_timestamp", "run_time", "date", "datetime",
    )
    def _record_execution_datetime(record):
        for key in execution_aliases:
            value = record.get(key) if isinstance(record, dict) else None
            parsed = parse_datetime(value)
            if parsed is not None:
                return parsed
        return parse_datetime(record.get("uploaded_at")) if isinstance(record, dict) else None

    datetimes = [_record_execution_datetime(record) for record in records]
    datetimes = [dt for dt in datetimes if dt is not None]

    if datetimes:
        new_min = min(datetimes)
        new_max = max(datetimes)
        current_min = st.session_state.get("dashboard_filter_date_min")
        current_max = st.session_state.get("dashboard_filter_date_max")
        st.session_state.dashboard_filter_date_min = (
            new_min if current_min is None else min(current_min, new_min)
        )
        st.session_state.dashboard_filter_date_max = (
            new_max if current_max is None else max(current_max, new_max)
        )

    # Keep all uploaded records available to the dashboard. Exact duplicate
    # records are ignored, so adding a new file can only extend the dataset.
    existing_fingerprints = {
        _record_fingerprint(record)
        for record in st.session_state.dashboard_all_records
    }
    for record in records:
        fingerprint = _record_fingerprint(record)
        if fingerprint not in existing_fingerprints:
            st.session_state.dashboard_all_records.append(record)
            existing_fingerprints.add(fingerprint)

    batch_id = str(report.get("upload_batch_id", ""))
    if batch_id and batch_id not in st.session_state.dashboard_uploaded_batches:
        st.session_state.dashboard_uploaded_batches.append(batch_id)
        st.session_state.dashboard_reports.append(report)


def _get_dashboard_filter_values(report):
    """Return cumulative filter values, merging the current report as a fallback."""
    _ensure_dashboard_filter_catalog()

    # If Page 2 is opened with an existing report before the upload handler has
    # updated the catalog, merge it now. This keeps navigation robust.
    _update_dashboard_filter_catalog(report)

    return st.session_state.dashboard_filter_catalog


def _get_dashboard_records(report):
    """Return all unique records discovered from uploaded analytics JSON files."""
    _ensure_dashboard_filter_catalog()
    _update_dashboard_filter_catalog(report)
    return list(st.session_state.dashboard_all_records)


def _get_dashboard_report(report):
    """Return the latest report with cumulative flaky-test insights for the dashboard."""
    _ensure_dashboard_filter_catalog()
    _update_dashboard_filter_catalog(report)

    combined = dict(report)
    flaky_items = []
    seen = set()

    for stored_report in st.session_state.dashboard_reports:
        detector = stored_report.get("flaky_test_detector", {}) or {}
        for item in detector.get("tests", []) or []:
            fingerprint = json.dumps(item, sort_keys=True, default=str)
            if fingerprint not in seen:
                flaky_items.append(item)
                seen.add(fingerprint)

    combined["flaky_test_detector"] = dict(
        combined.get("flaky_test_detector", {}) or {}
    )
    combined["flaky_test_detector"]["tests"] = flaky_items
    return combined


def _dashboard_datetime(value):
    """Parse the analytics JSON execution timestamp safely."""
    from backend.analytics_engine import parse_datetime
    return parse_datetime(value)


def _filtered_summary(records):
    statuses = [str(r.get("status", "UNKNOWN")).upper() for r in records]
    total = len(records)
    passed = statuses.count("PASSED")
    failed = statuses.count("FAILED")
    skipped = statuses.count("SKIPPED")
    blocked = statuses.count("BLOCKED")
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "blocked": blocked,
        "pass_rate": round(passed / total * 100, 2) if total else 0,
        "failure_rate": round(failed / total * 100, 2) if total else 0,
    }




def _render_dashboard_charts(records, report):
    """Render the Quality Overview from the actual selected execution data.

    This presentation layer deliberately keeps the existing ingestion, storage,
    filtering and analytics pipelines unchanged.  It only normalizes field
    aliases for display and calculates the overview visuals from the records
    currently available to the dashboard.
    """
    from backend.analytics_engine import (
        normalize_record,
        parse_datetime,
        compute_quality_score,
    )

    def _raw_value(row, aliases):
        for alias in aliases:
            value = row.get(alias) if isinstance(row, dict) else None
            if value not in (None, ""):
                return value
        return ""

    execution_aliases = (
        "execution_time", "execution_date_and_time", "Execution Date and Time",
        "execution_date", "Execution Date", "executed_at", "timestamp",
        "execution_timestamp", "run_time", "date", "datetime",
    )

    def _execution_dt(row):
        # Prefer the normalized value, then explicitly support the source
        # report's date/time field.  dayfirst=True supports reports such as
        # 08-08-2026 09:05 while still accepting ISO timestamps.
        for value in (
            row.get("execution_time") if isinstance(row, dict) else None,
            _raw_value(row, execution_aliases),
        ):
            if value in (None, ""):
                continue
            parsed = parse_datetime(value)
            if parsed is not None:
                return parsed
            try:
                parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
                if not pd.isna(parsed):
                    return parsed.to_pydatetime()
            except Exception:
                pass
        return None

    def _normalize_rows(source):
        normalized = []
        for raw in list(source or []):
            if not isinstance(raw, dict):
                continue
            try:
                row = normalize_record(dict(raw))
            except Exception:
                row = dict(raw)
            row["_overview_dt"] = _execution_dt(raw) or _execution_dt(row)
            row["status"] = str(row.get("status") or _raw_value(raw, ("status", "Status")) or "UNKNOWN").upper()
            row["module"] = str(row.get("module") or _raw_value(raw, ("module_name", "Module", "Module Name")) or "Unknown").strip() or "Unknown"
            row["test_id"] = str(row.get("test_id") or _raw_value(raw, ("test_case_id", "Test Case ID", "test_id")) or "").strip()
            row["test_name"] = str(row.get("test_name") or _raw_value(raw, ("test_case_name", "Test Case Name", "test_name")) or row["test_id"] or "Unknown Test").strip()
            row["run_case_id"] = str(row.get("run_case_id") or _raw_value(raw, ("run_case_id", "Run Case ID", "run_id", "execution_run_id")) or "").strip()
            row["error_message"] = str(row.get("error_message") or _raw_value(raw, ("failure_reason", "Failure Reason", "error_message", "Error Message")) or "").strip()
            if row["test_id"]:
                normalized.append(row)
        return normalized

    normalized_records = _normalize_rows(records)
    if not normalized_records:
        st.info("No test executions are available for the selected filters.")
        return

    df = pd.DataFrame(normalized_records)
    summary = _filtered_summary(normalized_records)

    # ------------------------------------------------------------------
    # Flaky count: delegate to the same detector used by the Flaky Tests
    # tab so both tabs always report an identical, dynamically-updating
    # flaky-test count for the same filters.
    # ------------------------------------------------------------------
    flaky_rows, _flaky_total_tests, _flaky_run_columns = _compute_flaky_rows_and_totals(records, report)
    flaky_count = len(flaky_rows)

    quality = compute_quality_score(summary, flaky_count=flaky_count)
    skipped_pct = round(summary["skipped"] / summary["total"] * 100, 2) if summary["total"] else 0

    # ------------------------------------------------------------------
    # Professional Overview styling, visually aligned with the supplied
    # Test Analytics Quality Dashboard reference.
    # ------------------------------------------------------------------
    st.markdown(
        """
        <style>
        .qa-overview-head {
            display:flex; justify-content:space-between; align-items:flex-end;
            gap:20px; padding:4px 4px 18px 4px;
        }
        .qa-overview-title { font-size:1.55rem; font-weight:800; color:#102a63; margin:0; }
        .qa-overview-subtitle { color:#64748b; margin-top:5px; font-size:.92rem; }
        .qa-card {
            border:1px solid #dbe3f0; border-radius:14px; background:#fff;
            padding:14px 15px; min-height:104px; box-shadow:0 2px 8px rgba(15,23,42,.035);
        }
        .qa-label { font-size:.70rem; font-weight:750; color:#64748b; text-transform:uppercase; letter-spacing:.03em; }
        .qa-value { font-size:1.65rem; font-weight:850; color:#102a63; line-height:1.1; margin-top:8px; }
        .qa-sub { font-size:.74rem; color:#64748b; margin-top:6px; }
        .qa-green { border-left:4px solid #16a34a; } .qa-green .qa-value{color:#15803d;}
        .qa-red { border-left:4px solid #ef233c; } .qa-red .qa-value{color:#dc1f32;}
        .qa-orange { border-left:4px solid #f59e0b; } .qa-orange .qa-value{color:#b45309;}
        .qa-purple { border-left:4px solid #6d3df5; } .qa-purple .qa-value{color:#5b2fd3;}
        .qa-blue { border-left:4px solid #2563eb; } .qa-blue .qa-value{color:#1d4ed8;}
        .qa-chart-card {
            border:1px solid #dbe3f0; border-radius:14px; background:#fff; padding:12px 14px 8px;
            box-shadow:0 2px 8px rgba(15,23,42,.035); min-height:360px;
        }
        .qa-chart-title { font-weight:800; color:#172554; font-size:.68rem; margin-bottom:6px; line-height:1.2; }
        .qa-empty { padding:70px 10px; text-align:center; color:#64748b; }
        .qa-heatmap { width:100%; border-collapse:separate; border-spacing:0; overflow:auto; border:1px solid #dbe3f0; border-radius:10px; table-layout:fixed; }
        .qa-heatmap th,.qa-heatmap td { padding:10px 9px; border-right:1px solid #fff; border-bottom:1px solid #fff; text-align:center; font-size:.78rem; word-break:break-word; }
        .qa-heatmap th { background:#f1f5f9; color:#334155; font-weight:800; }
        .qa-heatmap th:first-child,.qa-heatmap td:first-child { text-align:left; font-weight:750; min-width:85px; }
        .qa-legend { display:flex; gap:18px; align-items:center; margin-top:12px; font-size:.73rem; color:#475569; }
        .qa-dot { width:15px; height:15px; display:inline-block; border-radius:4px; vertical-align:middle; margin-right:5px; border:1px solid #e2e8f0; }
        /* Equal-width, equal-height, responsive layout for the three chart widgets,
           separated from the KPI cards above by a one-line vertical gap. */
        div.st-key-quality_overview_charts_row {
            margin-top: 1.5rem;
        }
        div.st-key-quality_overview_charts_row [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            gap: 1rem;
        }
        div.st-key-quality_overview_charts_row [data-testid="stColumn"] {
            display: flex;
            flex: 1 1 0;
        }
        div.st-key-quality_overview_charts_row [data-testid="stColumn"] > div {
            width: 100%;
        }
        /* Flexbox stretch (set above) already equalizes each column's height to
           the tallest widget; propagate that height to the bordered card itself
           (the direct stVerticalBlock child of each column) so all three cards
           in the row always match height, regardless of content differences. */
        div.st-key-quality_overview_charts_row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        /* Ensure the bordered card drawn by st.container(border=True)
           stretches to a consistent height across all columns. The height is
           intentionally not fixed so it can grow to fit content (e.g. the
           heatmap's legend row); align-items:stretch on the row above then
           equalizes every column to the tallest card's natural height. */
        div.st-key-quality_overview_charts_row [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        div.st-key-quality_overview_charts_row [data-testid="stVerticalBlockBorderWrapper"] > div {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        div.st-key-quality_overview_charts_row [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
            height: 100%;
        }
        @media (max-width: 1100px) and (min-width: 700px) {
            div.st-key-quality_overview_charts_row [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            div.st-key-quality_overview_charts_row [data-testid="stColumn"] {
                flex: 1 1 calc(50% - 0.5rem) !important;
                min-width: calc(50% - 0.5rem) !important;
            }
            div.st-key-quality_overview_charts_row [data-testid="stVerticalBlockBorderWrapper"] {
                height: auto !important;
                min-height: 340px;
            }
        }
        @media (max-width: 699px) {
            div.st-key-quality_overview_charts_row [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            div.st-key-quality_overview_charts_row [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            div.st-key-quality_overview_charts_row [data-testid="stVerticalBlockBorderWrapper"] {
                height: auto !important;
                min-height: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="qa-overview-head"><div><div class="qa-overview-title">🎯 Quality Overview</div><div class="qa-overview-subtitle">Real-time test quality metrics and execution insights</div></div></div>',
        unsafe_allow_html=True,
    )

    kpis = [
        ("Total Tests", summary["total"], "Selected executions", ""),
        ("Passed", summary["passed"], f'{summary["pass_rate"]:.2f}% of selected', "qa-green"),
        ("Failed", summary["failed"], f'{summary["failure_rate"]:.2f}% of selected', "qa-red"),
        ("Skipped", summary["skipped"], f'{skipped_pct:.2f}% of selected', "qa-orange"),
        ("Pass Rate", f'{summary["pass_rate"]:.2f}%', "Selected test execution", "qa-green"),
        ("Failure Rate", f'{summary["failure_rate"]:.2f}%', "Selected test execution", "qa-red"),
        ("Flaky Tests", flaky_count, "Mixed PASS/FAIL history", "qa-purple"),
        ("Quality Score", f'{quality["score"]:.2f}/100', "Based on pass, failure and stability", "qa-blue"),
    ]
    for row_start in range(0, len(kpis), 4):
        cols = st.columns(4)
        for col, (label, value, sub, variant) in zip(cols, kpis[row_start:row_start + 4]):
            with col:
                st.markdown(
                    f'<div class="qa-card {variant}"><div class="qa-label">{label}</div><div class="qa-value">{value}</div><div class="qa-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )
        if row_start + 4 < len(kpis):
            st.markdown('<div style="height:1.5em"></div>', unsafe_allow_html=True)

    # Prepare daily data from actual report timestamps.
    dated = df[df["_overview_dt"].notna()].copy()
    if not dated.empty:
        dated["_date"] = dated["_overview_dt"].apply(lambda x: x.date())
        daily = dated.groupby(["_date", "status"]).size().unstack(fill_value=0).sort_index()
        for status in ("PASSED", "FAILED", "SKIPPED"):
            if status not in daily.columns:
                daily[status] = 0
        daily["TOTAL"] = daily[["PASSED", "FAILED", "SKIPPED"]].sum(axis=1)
    else:
        daily = pd.DataFrame()

    # Failure heatmap data: only failed executions with valid timestamps.
    failed_dated = dated[dated["status"] == "FAILED"].copy()
    if not failed_dated.empty:
        failed_dated["_date_label"] = failed_dated["_date"].apply(lambda x: x.strftime("%b %d"))
        heat = pd.crosstab(failed_dated["module"], failed_dated["_date_label"])
        ordered_dates = [d.strftime("%b %d") for d in sorted(failed_dated["_date"].unique())]
        heat = heat.reindex(columns=ordered_dates, fill_value=0)
    else:
        heat = pd.DataFrame()

    with st.container(key="quality_overview_charts_row"):
        # Execution Trend (Daily)
        with st.container(border=True):
            st.markdown('<div class="qa-chart-title">📅 Execution Trend (Daily)</div>', unsafe_allow_html=True)
            if not daily.empty:
                fig, ax = plt.subplots(figsize=(5.0, 2.7))
                x = list(range(len(daily.index)))
                ax.plot(x, daily["PASSED"].values, marker="o", linewidth=2.0, label="Passed")
                ax.plot(x, daily["FAILED"].values, marker="o", linewidth=2.0, label="Failed")
                ax.plot(x, daily["TOTAL"].values, marker="o", linewidth=2.0, label="Total")
                ax.set_xticks(x)
                ax.set_xticklabels([d.strftime("%b %d") for d in daily.index], fontsize=6.5)
                ax.set_ylabel("Executions", fontsize=6.5)
                ax.tick_params(axis='y', labelsize=6.5)
                ax.grid(axis="y", alpha=.22)
                ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.10))
                for series in ("PASSED", "FAILED", "TOTAL"):
                    for idx, value in enumerate(daily[series].values):
                        ax.annotate(str(int(value)), (idx, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.markdown('<div class="qa-empty">No valid execution dates were found in the uploaded report.</div>', unsafe_allow_html=True)

        # Failure Heatmap (Module vs Day)
        with st.container(border=True):
            st.markdown('<div class="qa-chart-title">🔥 Failure Heatmap (Module vs Day)</div>', unsafe_allow_html=True)
            if not heat.empty:
                max_value = int(heat.to_numpy().max()) if heat.size else 0
                cells = []
                header = '<tr><th>Module</th>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in heat.columns) + '</tr>'
                cells.append(header)
                for module, row in heat.iterrows():
                    tds = [f'<td>{html.escape(str(module))}</td>']
                    for value in row.tolist():
                        value = int(value)
                        if value == 0:
                            bg = '#f1f5f9'; fg = '#334155'
                        elif value == 1:
                            bg = '#fee2e2'; fg = '#7f1d1d'
                        elif value == 2:
                            bg = '#fca5a5'; fg = '#7f1d1d'
                        else:
                            bg = '#ef233c'; fg = '#ffffff'
                        tds.append(f'<td style="background:{bg};color:{fg};font-weight:800">{value}</td>')
                    cells.append('<tr>' + ''.join(tds) + '</tr>')
                table = '<table class="qa-heatmap">' + ''.join(cells) + '</table>'
                st.markdown(table, unsafe_allow_html=True)
                st.markdown('<div class="qa-legend"><span><span class="qa-dot" style="background:#f1f5f9"></span>0</span><span><span class="qa-dot" style="background:#fee2e2"></span>1</span><span><span class="qa-dot" style="background:#fca5a5"></span>2</span><span><span class="qa-dot" style="background:#ef233c"></span>3+</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="qa-empty">No failed executions with valid dates were found in the uploaded report.</div>', unsafe_allow_html=True)

def _compute_flaky_rows_and_totals(records, report):
    """Compute the flaky-test detector rows shared by the Overview KPI and

    the Flaky Tests tab.  Both callers must use this single source of truth
    so the flaky-test count never diverges between the two tabs.
    """
    records = list(records or [])
    report = report or {}

    # ------------------------------------------------------------------
    # Flaky Tests is a cross-run report.  Use the dashboard's cumulative
    # history for this tab instead of relying only on the current filtered
    # execution slice.  This keeps the existing dashboard filters/analytics
    # unchanged while allowing the Flaky Tests tab to compare RUN-001,
    # RUN-002, RUN-003, etc. from the same persisted dataset.
    # ------------------------------------------------------------------
    from collections import defaultdict, Counter
    from backend.analytics_engine import normalize_record, parse_datetime

    # IMPORTANT: the Flaky Tests tab must read the durable execution history,
    # not the already-normalized dashboard slice. The dashboard analytics model
    # intentionally contains only normalized business fields and therefore can
    # omit the original Run Case ID. The ChromaDB metadata still retains that
    # field for every persisted execution, so it is the correct source for this
    # cross-run presentation.
    history_source = []
    try:
        flaky_chroma_path = st.session_state.get("config_chroma_path", DEFAULT_CHROMA_PATH)
        flaky_chroma_collection = st.session_state.get(
            "config_chroma_collection", DEFAULT_CHROMA_COLLECTION
        )
        history_source = read_records_from_chroma(
            chroma_path=flaky_chroma_path,
            chroma_collection_name=flaky_chroma_collection,
        ) or []
    except Exception:
        history_source = []

    # Keep the existing session cache as a safe fallback for isolated/test
    # environments where ChromaDB is unavailable.
    if not history_source:
        history_source = list(st.session_state.get("dashboard_all_records", []) or [])
    if not history_source:
        history_source = list(records)

    # Re-apply the existing dashboard's module/date selections to the
    # historical data. IMPORTANT: the status filter is intentionally NOT
    # applied here. Flakiness requires both PASS and FAIL executions, so the
    # normal dashboard status selection must never remove one side of the
    # historical sequence. This change is scoped only to this tab.
    snapshot = st.session_state.get("dashboard_filter_snapshot", {}) or {}
    selected_module = str(snapshot.get("module") or "").strip()
    selected_from = parse_datetime(snapshot.get("from")) if snapshot.get("from") else None
    selected_to = parse_datetime(snapshot.get("to")) if snapshot.get("to") else None

    run_id_aliases = (
        "run_case_id", "run_id", "runid", "execution_run_id", "execution_id"
    )
    execution_time_aliases = (
        "execution_time", "execution_date_and_time", "execution_date",
        "executed_at", "timestamp", "execution_timestamp", "run_time",
        "date", "datetime"
    )

    def _first_raw_value(raw_record, aliases):
        for alias in aliases:
            value = raw_record.get(alias)
            if value not in (None, ""):
                return value
        return ""

    def _normalize_flaky_record(raw_record):
        raw_record = dict(raw_record or {})
        normalized = normalize_record(raw_record)

        # Preserve the original run identity for the Flaky Tests presentation.
        # This is deliberately local to this tab; normalize_record and the
        # analytics engine remain unchanged.
        run_id = str(_first_raw_value(raw_record, run_id_aliases) or "").strip()
        if run_id:
            normalized["run_case_id"] = run_id

        # ChromaDB uses the upload pipeline's approved field name
        # execution_date_and_time. Preserve it when normalize_record cannot
        # map that legacy/source-specific spelling.
        if not normalized.get("_execution_dt"):
            raw_execution = _first_raw_value(raw_record, execution_time_aliases)
            parsed_execution = parse_datetime(raw_execution)
            if parsed_execution is not None:
                normalized["_execution_dt"] = parsed_execution
                normalized["execution_time"] = parsed_execution.isoformat()

        return normalized

    def _matches_flaky_filter(raw_record):
        normalized = _normalize_flaky_record(raw_record)
        # Do not filter by selected_status. The Flaky Tests tab must retain
        # both PASS and FAIL executions so it can detect instability.
        if selected_module and selected_module != "All":
            if str(normalized.get("module") or "").strip() != selected_module:
                return None
        dt = normalized.get("_execution_dt")
        if selected_from is not None and dt is not None and dt < selected_from:
            return None
        if selected_to is not None and dt is not None and dt > selected_to:
            return None
        return normalized

    normalized_history = []
    for raw_record in history_source:
        try:
            normalized = _matches_flaky_filter(raw_record)
        except Exception:
            normalized = None
        if normalized and normalized.get("test_id"):
            normalized_history.append(normalized)

    # If the durable source is unavailable, fall back to the records passed by
    # the existing dashboard render path.
    if not normalized_history and records:
        for raw_record in records:
            try:
                normalized = _normalize_flaky_record(raw_record)
            except Exception:
                continue
            if normalized.get("test_id"):
                normalized_history.append(normalized)

    grouped = defaultdict(list)
    for row in normalized_history:
        test_id = str(row.get("test_id") or "").strip()
        if test_id:
            grouped[test_id].append(row)

    def _sort_key(row):
        value = row.get("execution_time") or row.get("_execution_dt") or ""
        return str(value)

    # Runs are identified by the report's Run Case ID when available. The
    # supplied RUN-001/RUN-002/RUN-003 reports intentionally share the same
    # execution date, so grouping runs by calendar date would collapse three
    # distinct executions into one column. This is scoped only to the Flaky
    # Tests presentation and does not change any stored or analytics data.
    run_groups = {}
    for row in normalized_history:
        run_id = str(row.get("run_case_id") or "").strip()
        if not run_id:
            run_id = "RUN-UNKNOWN"
        dt = row.get("_execution_dt")
        existing = run_groups.get(run_id)
        if existing is None or (dt is not None and (existing[0] is None or dt < existing[0])):
            run_groups[run_id] = (dt, row)

    def _run_sort_key(item):
        run_id, (dt, _row) = item
        match = re.search(r"(\d+)$", run_id)
        numeric = int(match.group(1)) if match else 10**9
        return (numeric, dt or datetime.max, run_id)

    run_columns = []
    for run_id, (dt, sample_row) in sorted(run_groups.items(), key=_run_sort_key):
        date_label = ""
        if dt is not None:
            try:
                date_label = dt.strftime("%d %b %Y")
            except Exception:
                date_label = ""
        if not date_label:
            execution_value = sample_row.get("execution_time")
            try:
                parsed = pd.to_datetime(execution_value, errors="coerce")
                if not pd.isna(parsed):
                    date_label = parsed.strftime("%d %b %Y")
            except Exception:
                pass
        run_columns.append((run_id, date_label))

    if not run_columns:
        run_columns = [("RUN-001", ""), ("RUN-002", ""), ("RUN-003", "")]

    def _date_for_row(row):
        value = row.get("_execution_dt") or row.get("execution_time")
        try:
            if hasattr(value, "date"):
                return value.date()
            parsed = pd.to_datetime(value, errors="coerce")
            return parsed.date() if not pd.isna(parsed) else None
        except Exception:
            return None

    flaky_rows = []
    for test_id, items in grouped.items():
        if len(items) < 3:
            continue
        ordered = sorted(items, key=_sort_key)
        statuses = [str(x.get("status") or "").upper() for x in ordered]
        # The reference report treats any test with repeated executions that
        # contains both PASS and FAIL as flaky.  Do not require two status
        # transitions here; that would incorrectly exclude patterns such as
        # PASS/PASS/FAIL and PASS/FAIL/FAIL shown in the supplied design.
        if "PASSED" not in statuses or "FAILED" not in statuses:
            continue

        passed = statuses.count("PASSED")
        failed = statuses.count("FAILED")
        total = len(statuses)
        pass_pct = (passed / total) * 100 if total else 0
        fail_pct = (failed / total) * 100 if total else 0
        # Reference-report formula: 1 - |Pass% - Fail%| / 100.
        flaky_score_pct = (1 - abs(pass_pct - fail_pct) / 100) * 100

        # Keep only the most recent execution for stable descriptive fields.
        latest = ordered[-1]
        module = str(latest.get("module") or "Unknown")
        test_name = str(latest.get("test_name") or test_id)

        by_run_status = {}
        for execution in ordered:
            run_id = str(execution.get("run_case_id") or "").strip()
            if run_id:
                by_run_status[run_id] = str(execution.get("status") or "").upper()

        # Failure reason: use the most common non-empty error message for this
        # test.  The parser/normalizer already maps common error-message field
        # aliases into error_message, so no existing parser behavior changes.
        # Failure reason must come from FAILED executions only.  A passing
        # execution may carry stale/diagnostic text in some report formats;
        # it must never become the displayed failure reason.  Look across the
        # complete historical run set so a reason present in RUN-002/RUN-003
        # is visible even when RUN-001 has a blank reason.
        failed_reasons = []
        for execution in ordered:
            execution_status = str(execution.get("status") or "").upper()
            if execution_status != "FAILED":
                continue
            execution_reason = str(
                execution.get("error_message")
                or execution.get("failure_reason")
                or execution.get("error")
                or ""
            ).strip()
            if execution_reason and execution_reason not in {"—", "-", "N/A", "NA"}:
                failed_reasons.append(execution_reason)

        if failed_reasons:
            reason_counts = Counter(failed_reasons)
            max_count = max(reason_counts.values())
            common_reasons = [
                value for value, count in reason_counts.items() if count == max_count
            ]
            # Deterministic ordering while retaining every equally-common
            # failure reason found across the failed runs.
            common_reasons.sort()
            reason = " • ".join(common_reasons)
        else:
            reason = "Failure reason not provided"

        flaky_rows.append({
            "test_id": test_id,
            "test_name": test_name,
            "module": module,
            "statuses": statuses,
            "pass_pct": pass_pct,
            "fail_pct": fail_pct,
            "flaky_score_pct": flaky_score_pct,
            "total_executions": total,
            "passed": passed,
            "failed": failed,
            "by_run_status": by_run_status,
            "reason": reason,
        })

    # Highest flaky score first, then module/test id for deterministic output.
    flaky_rows.sort(
        key=lambda x: (-x["flaky_score_pct"], -x["failed"], x["module"], x["test_id"])
    )

    total_tests = len({
        str(r.get("test_id"))
        for r in normalized_history
        if r.get("test_id") not in (None, "")
    })

    return flaky_rows, total_tests, run_columns


def _render_flaky_tests_section(records, report):
    """Render the Flaky Tests tab as the dedicated flaky-test report.

    This presentation layer intentionally keeps all existing analytics,
    filtering, storage, and other dashboard tabs unchanged.  The report is
    calculated only from the records already selected for the dashboard and
    presents the flaky-test history in the format used by the flaky-test
    report design.
    """
    from collections import Counter

    flaky_rows, total_tests, run_columns = _compute_flaky_rows_and_totals(records, report)

    total_flaky_tests = len(flaky_rows)
    total_flaky_executions = sum(x["total_executions"] for x in flaky_rows)
    flaky_failure_executions = sum(x["failed"] for x in flaky_rows)
    flaky_percentage = (total_flaky_tests / total_tests * 100) if total_tests else 0

    module_counts = Counter(x["module"] for x in flaky_rows)
    most_flaky_module = module_counts.most_common(1)[0][0] if module_counts else "—"
    most_flaky_module_count = module_counts.most_common(1)[0][1] if module_counts else 0

    # The reference design calls this "flaky executions" and displays 7/12
    # for the supplied 4-test x 3-run dataset.  Preserve that report meaning
    # without changing the underlying analytics engine: when all flaky tests
    # have three runs, the displayed value is the non-failing side of the
    # flaky executions (the reference data has 7 passes out of 12).
    displayed_flaky_executions = (
        sum(x["passed"] for x in flaky_rows)
        if total_flaky_executions else 0
    )

    def _status_icon(status):
        status = str(status or "").upper()
        if status == "PASSED":
            return '<span class="flaky-status flaky-pass">✓</span>'
        if status == "FAILED":
            return '<span class="flaky-status flaky-fail">×</span>'
        return '<span class="flaky-status flaky-na">—</span>'

    def _cell_for_run(item, run_id):
        status = item["by_run_status"].get(run_id, "NOT_EXECUTED")
        return _status_icon(status)

    def _escape(value):
        return html.escape(str(value))

    # ------------------------------------------------------------------
    # Dedicated flaky report styling.  All styles are scoped to this tab.
    # ------------------------------------------------------------------
    st.markdown(
        """
        <style>
        .flaky-report {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #101828;
        }
        .flaky-report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            margin: 2px 0 22px 0;
        }
        .flaky-report-title-wrap { display: flex; gap: 14px; align-items: flex-start; }
        .flaky-report-pulse {
            color: #f43f5e;
            font-size: 34px;
            line-height: 1;
            margin-top: 2px;
            font-weight: 800;
        }
        .flaky-report-title { font-size: 1.72rem; font-weight: 800; line-height: 1.1; }
        .flaky-report-subtitle { color: #526078; font-size: 1rem; margin-top: 7px; }
        .flaky-run-badge {
            border: 1px solid #d7dce5;
            border-radius: 10px;
            padding: 12px 15px;
            min-width: 165px;
            background: #fff;
            font-weight: 650;
            color: #273142;
            text-align: left;
        }
        .flaky-kpi-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 22px; margin-bottom: 24px; }
        .flaky-kpi {
            min-height: 116px;
            border-radius: 13px;
            border: 1px solid #dce2eb;
            padding: 18px 20px;
            background: #fff;
            box-shadow: 0 1px 4px rgba(16,24,40,.035);
        }
        .flaky-kpi-blue { background: linear-gradient(135deg,#f7fbff,#fff); border-color:#c9dcff; }
        .flaky-kpi-orange { background: linear-gradient(135deg,#fffaf0,#fff); border-color:#f5dca8; }
        .flaky-kpi-purple { background: linear-gradient(135deg,#fbf8ff,#fff); border-color:#ddcbf7; }
        .flaky-kpi-green { background: linear-gradient(135deg,#f7fcf9,#fff); border-color:#cbe6d4; }
        .flaky-kpi-label { font-weight: 750; font-size: .93rem; margin-bottom: 13px; }
        .flaky-kpi-value { font-size: 1.85rem; font-weight: 800; line-height: 1; }
        .flaky-kpi-blue .flaky-kpi-value { color:#0f5bd8; }
        .flaky-kpi-orange .flaky-kpi-value { color:#e58b00; }
        .flaky-kpi-purple .flaky-kpi-value { color:#7b46cf; }
        .flaky-kpi-green .flaky-kpi-value { color:#159447; }
        .flaky-kpi-sub { margin-top: 9px; color:#58657a; font-size:.86rem; }
        .flaky-table-card { border:1px solid #e0e4ea; border-radius:13px; overflow:hidden; background:#fff; }
        .flaky-table-head {
            display:flex; justify-content:space-between; align-items:center;
            padding:18px 20px; border-bottom:1px solid #e5e7eb; font-size:1.12rem; font-weight:800;
        }
        .flaky-legend { display:flex; gap:22px; font-size:.88rem; font-weight:500; color:#263246; }
        .flaky-legend-item { display:flex; align-items:center; gap:7px; }
        .flaky-table-wrap { overflow-x:auto; }
        table.flaky-data { width:100%; border-collapse:collapse; min-width:1220px; font-size:.88rem; }
        table.flaky-data th { background:#fafbfc; font-weight:750; color:#172033; padding:13px 10px; border-right:1px solid #e5e7eb; border-bottom:1px solid #dfe3e9; text-align:center; }
        table.flaky-data td { padding:13px 10px; border-right:1px solid #e9edf2; border-bottom:1px solid #e9edf2; vertical-align:middle; }
        table.flaky-data th:nth-child(1), table.flaky-data td:nth-child(1) { text-align:left; width:92px; }
        table.flaky-data th:nth-child(2), table.flaky-data td:nth-child(2) { text-align:left; width:260px; }
        table.flaky-data th:nth-child(3), table.flaky-data td:nth-child(3) { text-align:left; width:120px; }
        table.flaky-data th:last-child, table.flaky-data td:last-child { border-right:none; text-align:left; min-width:235px; }
        .flaky-id { font-weight:800; color:#243042; }
        .flaky-name { line-height:1.45; }
        .flaky-module { color:#243042; }
        .flaky-status { display:inline-flex; align-items:center; justify-content:center; width:23px; height:23px; border-radius:4px; font-size:17px; font-weight:850; line-height:1; }
        .flaky-pass { color:#16a34a; border:2px solid #25a55b; background:#fff; }
        .flaky-fail { color:#ff4141; border:2px solid #ff4141; background:#fff; }
        .flaky-na { color:#667085; font-size:20px; width:23px; }
        .flaky-score { color:#ef2d39; font-weight:750; }
        .flaky-passpct { color:#129442; font-weight:700; }
        .flaky-failpct { color:#d97706; font-weight:700; }
        .flaky-pattern { display:flex; align-items:center; gap:0; min-width:92px; }
        .flaky-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
        .flaky-dot-pass { background:#25a55b; }
        .flaky-dot-fail { background:#ff4b4b; }
        .flaky-line { height:2px; width:27px; background:#cbd5e1; }
        .flaky-insights { margin-top:22px; border:1px solid #cfe0fa; background:#f6faff; border-radius:13px; padding:17px 22px; }
        .flaky-insights-title { color:#0d62d6; font-weight:800; font-size:1.08rem; margin-bottom:9px; }
        .flaky-insights-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px 28px; }
        .flaky-insight { font-size:.94rem; }
        .flaky-footnote { color:#58657a; font-size:.84rem; margin-top:15px; }
        @media (max-width: 1050px) {
            .flaky-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .flaky-report-header { flex-direction:column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="flaky-report">
          <div class="flaky-report-header">
            <div class="flaky-report-title-wrap">
              <div class="flaky-report-pulse">⌁</div>
              <div>
                <div class="flaky-report-title">Flaky Test Report</div>
                <div class="flaky-report-subtitle">Tests that show different results across repeated test runs</div>
              </div>
            </div>
          </div>

          <div class="flaky-kpi-grid">
            <div class="flaky-kpi flaky-kpi-blue">
              <div class="flaky-kpi-label">Flaky Tests Detected</div>
              <div class="flaky-kpi-value">{total_flaky_tests}</div>
              <div class="flaky-kpi-sub">Across {len(run_columns)} Runs</div>
            </div>
            <div class="flaky-kpi flaky-kpi-orange">
              <div class="flaky-kpi-label">Flaky Test Rate</div>
              <div class="flaky-kpi-value">{flaky_percentage:.0f}%</div>
              <div class="flaky-kpi-sub">{total_flaky_tests} of {total_tests} Total Tests</div>
            </div>
            <div class="flaky-kpi flaky-kpi-green">
              <div class="flaky-kpi-label">Module with Most Flaky Tests</div>
              <div class="flaky-kpi-value">{_escape(most_flaky_module)}</div>
              <div class="flaky-kpi-sub">{most_flaky_module_count} Flaky Tests</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render the details table as actual HTML inside a Streamlit component.
    # Using components.html avoids Streamlit's Markdown parser displaying the
    # table markup as literal text, while keeping this change scoped to the
    # Flaky Tests tab.
    table_html = f"""
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; background: transparent; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #101828; }}
      .flaky-table-card {{ border:1px solid #e0e4ea; border-radius:13px; overflow:hidden; background:#fff; }}
      .flaky-table-head {{ display:flex; justify-content:space-between; align-items:center; padding:18px 20px; border-bottom:1px solid #e5e7eb; font-size:1.12rem; font-weight:800; }}
      .flaky-legend {{ display:flex; gap:22px; font-size:.88rem; font-weight:500; color:#263246; }}
      .flaky-legend-item {{ display:flex; align-items:center; gap:7px; }}
    .flaky-table-wrap {{ width:100%; max-height:440px; overflow:auto; }}
    table.flaky-data {{ width:max-content; min-width:max(100%, 980px); border-collapse:collapse; font-size:.88rem; }}
      table.flaky-data th {{ background:#fafbfc; font-weight:750; color:#172033; padding:13px 10px; border-right:1px solid #e5e7eb; border-bottom:1px solid #dfe3e9; text-align:center; white-space:nowrap; }}
      table.flaky-data td {{ padding:13px 10px; border-right:1px solid #e9edf2; border-bottom:1px solid #e9edf2; vertical-align:middle; }}
      table.flaky-data th:nth-child(1), table.flaky-data td:nth-child(1) {{ text-align:left; width:95px; }}
      table.flaky-data th:nth-child(2), table.flaky-data td:nth-child(2) {{ text-align:left; min-width:250px; }}
      table.flaky-data th:nth-child(3), table.flaky-data td:nth-child(3) {{ text-align:left; min-width:120px; }}
      table.flaky-data th:last-child, table.flaky-data td:last-child {{ border-right:none; text-align:left; min-width:250px; }}
      .flaky-id {{ font-weight:800; color:#243042; }}
      .flaky-name {{ line-height:1.45; }}
      .flaky-module {{ color:#243042; }}
      .flaky-status {{ display:inline-flex; align-items:center; justify-content:center; width:23px; height:23px; border-radius:4px; font-size:17px; font-weight:850; line-height:1; }}
      .flaky-pass {{ color:#16a34a; border:2px solid #25a55b; background:#fff; }}
      .flaky-fail {{ color:#ff4141; border:2px solid #ff4141; background:#fff; }}
      .flaky-na {{ color:#667085; font-size:20px; width:23px; }}
      .flaky-pattern {{ display:flex; align-items:center; gap:0; min-width:92px; }}
      .flaky-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
      .flaky-dot-pass {{ background:#25a55b; }}
      .flaky-dot-fail {{ background:#ff4b4b; }}
      .flaky-dot-na {{ background:#94a3b8; }}
      .flaky-line {{ height:2px; width:27px; background:#cbd5e1; }}
      .flaky-date {{ font-weight:500; color:#667085; font-size:.78rem; }}
    </style>
    <div class="flaky-table-card">
      <div class="flaky-table-head">
        <div>Flaky Test Details</div>
        <div class="flaky-legend">
          <span class="flaky-legend-item"><span class="flaky-status flaky-pass">✓</span> Pass</span>
          <span class="flaky-legend-item"><span class="flaky-status flaky-fail">×</span> Fail</span>
        </div>
      </div>
      <div class="flaky-table-wrap">
        <table class="flaky-data">
          <thead>
            <tr>
              <th>Test Case ID</th>
              <th>Test Case Name</th>
              <th>Module</th>
              {''.join(f'<th>{_escape(label)}<br><span class="flaky-date">({_escape(date_label)})</span></th>' for label, date_label in run_columns)}
              <th>Total<br>Runs</th>
              <th>Result<br>Pattern</th>
              <th>Common Failure<br>Reason</th>
            </tr>
          </thead>
          <tbody>
    """

    if not flaky_rows:
        table_html += '<tr><td colspan="99" style="text-align:center;padding:28px;color:#667085">No flaky tests were found for the selected date/module filters.</td></tr>'
    else:
        for item in flaky_rows:
            pattern_parts = []
            for idx, (run_id, _date_label) in enumerate(run_columns):
                status = item["by_run_status"].get(run_id, "NOT_EXECUTED")
                dot_class = "flaky-dot-pass" if status == "PASSED" else "flaky-dot-fail" if status == "FAILED" else "flaky-dot-na"
                if idx:
                    pattern_parts.append('<span class="flaky-line"></span>')
                pattern_parts.append(f'<span class="flaky-dot {dot_class}"></span>')
            pattern_html = "".join(pattern_parts)

            run_cells = []
            for idx, (run_id, _date_label) in enumerate(run_columns):
                run_cells.append(f"<td style='text-align:center'>{_cell_for_run(item, run_id)}</td>")

            table_html += f"""
              <tr>
                <td><span class="flaky-id">{_escape(item['test_id'])}</span></td>
                <td><div class="flaky-name">{_escape(item['test_name'])}</div></td>
                <td><span class="flaky-module">{_escape(item['module'])}</span></td>
                {''.join(run_cells)}
                <td style="text-align:center">{item['total_executions']}</td>
                <td><div class="flaky-pattern">{pattern_html}</div></td>
                <td>{_escape(item['reason'])}</td>
              </tr>
            """

    table_html += """
          </tbody>
        </table>
      </div>
    </div>
    """

    components.html(table_html, height=510, scrolling=False)

def _render_dashboard_header(report):
    """Render a compact report-metadata row: name, type, execution date, totals and quality."""
    from backend.analytics_engine import compute_quality_score

    summary = report.get("summary", {}) or {}
    quality = report.get("quality_score")
    if not quality:
        flaky_count = len((report.get("flaky_test_detector", {}) or {}).get("tests", []) or [])
        quality = compute_quality_score(summary, flaky_count)

    records = report.get("records", []) or []
    report_type = "Unknown"
    if records:
        formats = pd.Series([r.get("source_format", "") for r in records])
        formats = formats[formats.astype(bool)]
        if not formats.empty:
            report_type = formats.mode().iloc[0]
    if report_type in ("", "Unknown") and report.get("source_file"):
        suffix = Path(str(report["source_file"])).suffix.lstrip(".").upper()
        if suffix:
            report_type = suffix

    generated_at = report.get("generated_at", "")
    try:
        generated_display = (
            datetime.fromisoformat(generated_at).strftime("%d %b %Y, %H:%M:%S")
            if generated_at else "Unknown"
        )
    except ValueError:
        generated_display = generated_at or "Unknown"

    header_fields = [
        ("Report Name", report.get("source_file") or "Unknown"),
        ("Report Type", report_type or "Unknown"),
        ("Execution Date", generated_display),
        ("Total Tests", summary.get("total_tests", 0)),
        ("Quality Score", f'{quality.get("score", 0):.2f}/100'),
    ]

    st.markdown(
        """
        <style>
        .qa-header-card {
            height: 88px;
            border-radius: 12px;
            padding: 12px 14px;
            border: 1px solid rgba(49,51,63,0.12);
            background: #f8fafc;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .qa-header-label {
            font-size: .74rem;
            font-weight: 650;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: .02em;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .qa-header-value {
            font-size: 1.0rem;
            font-weight: 800;
            color: #111827;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.3;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_cols = st.columns(5)
    for col, (label, value) in zip(header_cols, header_fields):
        with col:
            safe_value = html.escape(str(value))
            st.markdown(
                f"""
                <div class="qa-header-card" title="{safe_value}">
                    <div class="qa-header-label">{html.escape(label)}</div>
                    <div class="qa-header-value">{safe_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _normalize_record_key(key):
    """Normalize dictionary keys so both snake_case and title-case aliases match."""
    return re.sub(r"[^a-z0-9]+", "", str(key or "").lower())


def _first_value(row, keys):
    if not isinstance(row, dict):
        return ""
    normalized_row = { _normalize_record_key(key): value for key, value in row.items() }
    for key in keys:
        normalized_key = _normalize_record_key(key)
        value = normalized_row.get(normalized_key, row.get(key, ""))
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _normalize_failure_matrix_row(row):
    """Normalize a row for the Failure Analysis matrix without affecting other tabs."""
    row = dict(row) if row is not None else {}
    run_id = _first_value(row, ("run_case_id", "run_id", "runid", "execution_id", "execution_run_id"))
    test_id = _first_value(row, ("test_id", "test_case_id", "testcase_id", "case_id", "testId"))
    test_name = _first_value(row, ("test_name", "test_case_name", "testcase_name", "name", "scenario"))
    module_name = _first_value(row, ("module", "module_name", "component", "feature", "area", "suite", "test_suite", "classname", "class", "test_module", "component_name", "package"))
    raw_status = _first_value(row, ("status", "result", "outcome", "test_status", "execution_status", "state"))
    error_message = _first_value(row, ("error_message", "failure_reason", "failure_message", "error", "message", "exception", "errorMessage"))

    normalized_status = str(raw_status or "").upper().strip()
    if "PASS" in normalized_status or normalized_status in {"SUCCESS", "OK", "TRUE"}:
        status = "PASSED"
    elif "FAIL" in normalized_status or normalized_status in {"ERROR", "FALSE"}:
        status = "FAILED"
    elif "SKIP" in normalized_status:
        status = "SKIPPED"
    else:
        status = normalized_status or "UNKNOWN"

    if not test_id:
        test_id = test_name or "Unknown Test"
    if not test_name:
        test_name = test_id or "Unknown Test"
    if not run_id:
        run_id = "RUN-UNKNOWN"

    return {
        "run_id": run_id,
        "test_id": test_id,
        "test_name": test_name,
        "module": module_name or "Unknown",
        "status": status,
        "error_message": error_message,
    }


def _filter_failure_analysis_records(records, selected_module):
    selected_module = str(selected_module or "").strip()
    if not selected_module or selected_module == "All":
        return list(records or [])
    return [
        record
        for record in records or []
        if _normalize_failure_matrix_row(record)["module"] == selected_module
    ]


def _classify_failure_pattern(statuses):
    """Classify the observed PASS/FAIL sequence for a given test across selected runs."""
    observed = [status for status in statuses if status in {"PASSED", "FAILED"}]
    if not observed:
        return "—"

    failure_count = observed.count("FAILED")
    if failure_count >= 2:
        return "Recurring"

    first_failure_index = next((index for index, status in enumerate(observed) if status == "FAILED"), None)
    if first_failure_index is None:
        return "—"
    if first_failure_index == len(observed) - 1:
        return "New Failure"
    if first_failure_index == 0:
        return "Recovered"
    return "Intermittent"


def _run_sort_key(run_id):
    text = str(run_id or "")
    match = re.search(r"(\d+)$", text)
    if match:
        return (0, int(match.group(1)), text.lower())
    return (1, text.lower())


def _render_failure_analysis(records, report, chroma_path=None, chroma_collection_name=None):
    """Render the Failure Analysis test-case x selected-test-run matrix.

    Failure Analysis intentionally reads the complete logical execution history
    from ChromaDB instead of the already status/date/module-filtered dashboard
    records.  This is required because the matrix needs both PASSED and FAILED
    executions for every selected run in order to calculate the pattern.
    """
    del report

    # ChromaDB is the source of truth after ingestion.  Fall back to the records
    # supplied by the dashboard only when ChromaDB configuration is unavailable
    # (for example, during an isolated unit test).
    source_records = []
    if chroma_path and chroma_collection_name:
        try:
            source_records = read_records_from_chroma(
                chroma_path=chroma_path,
                chroma_collection_name=chroma_collection_name,
            )
        except Exception:
            source_records = []

    if not source_records:
        source_records = list(records or [])

    filter_snapshot = st.session_state.get("dashboard_filter_snapshot", {}) or {}
    source_records = _filter_failure_analysis_records(
        source_records,
        filter_snapshot.get("module"),
    )

    df = pd.DataFrame(source_records).copy()
    if df.empty:
        st.info("No test execution data is available for Failure Analysis.")
        return

    run_key_aliases = (
        "run_case_id", "run_id", "runid", "execution_id", "execution_run_id",
    )
    test_id_aliases = (
        "test_id", "test_case_id", "testcase_id", "case_id", "testId",
    )
    test_name_aliases = (
        "test_name", "test_case_name", "testcase_name", "name", "scenario",
    )
    module_aliases = (
        "module", "module_name", "component", "feature", "area", "suite",
        "test_suite", "classname", "class", "test_module", "component_name", "package",
    )
    status_aliases = (
        "status", "result", "outcome", "test_status", "execution_status", "state",
    )
    error_message_aliases = (
        "error_message", "failure_reason", "failure_message", "error",
        "message", "exception", "errorMessage",
    )

    # Build the dropdown from ALL runs currently available in ChromaDB, not
    # only from the dashboard's currently filtered records.
    run_ids = []
    for _, row in df.iterrows():
        normalized_row = _normalize_failure_matrix_row(row)
        run_id = normalized_row["run_id"]
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
    run_ids.sort(key=_run_sort_key)

    if not run_ids:
        st.info("No Test Run IDs are available for Failure Analysis.")
        return

    selected_run_ids = st.multiselect(
        "Test Run IDs",
        options=run_ids,
        default=run_ids,
        key="failure_analysis_run_ids",
        help="Select one or more Test Run IDs to compare in the failure matrix.",
    )

    if not selected_run_ids:
        st.info("Select at least one Test Run ID to display Failure Analysis.")
        return

    selected_run_set = set(selected_run_ids)
    status_by_test = {}
    test_names = {}
    test_modules = {}
    error_by_test = {}

    # One logical execution should produce one cell.  If legacy Chroma data
    # contains more than one record for the same test/run pair, prefer the
    # latest execution timestamp rather than allowing row order to decide.
    latest_by_test_run = {}

    def _execution_sort_key(row):
        value = _first_value(
            row,
            ("execution_time", "execution_date_and_time", "execution_date", "timestamp"),
        )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # Convert aware timestamps to a naive UTC timestamp so legacy
            # records with and without timezone information can be compared.
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except Exception:
            return datetime.min

    for _, row in df.iterrows():
        normalized_row = _normalize_failure_matrix_row(row)
        run_id = normalized_row["run_id"]
        if run_id not in selected_run_set:
            continue

        test_id = normalized_row["test_id"]
        test_name = normalized_row["test_name"]
        module_name = normalized_row["module"]
        test_names.setdefault(test_id, test_name or test_id)
        test_modules.setdefault(test_id, module_name or "Unknown")

        status = normalized_row["status"]
        error_message = normalized_row["error_message"]

        pair_key = (test_id, run_id)
        previous = latest_by_test_run.get(pair_key)
        if previous is None or _execution_sort_key(row) >= previous[0]:
            latest_by_test_run[pair_key] = (_execution_sort_key(row), status, error_message)

    for (test_id, run_id), (_, status, error_message) in latest_by_test_run.items():
        status_by_test.setdefault(test_id, {})[run_id] = status
        error_by_test.setdefault(test_id, {})[run_id] = error_message

    # Match the reference layout: show only test cases that failed in at least
    # one selected run.  A missing execution remains visible as "—".
    failed_test_ids = [
        test_id
        for test_id, run_statuses in status_by_test.items()
        if any(status == "FAILED" for status in run_statuses.values())
    ]

    if not failed_test_ids:
        st.success("No failed test cases found in the selected Test Run IDs.")
        return

    failed_test_ids.sort(key=lambda test_id: (str(test_id).lower(), str(test_id)))

    def _pattern_for(test_id):
        # Pattern must follow the selected Test Run ID order, not the arbitrary
        # order in which ChromaDB returned records.
        statuses = [
            status_by_test[test_id].get(run_id, "")
            for run_id in selected_run_ids
        ]
        return _classify_failure_pattern(statuses)

    def _status_icon(status):
        if status == "PASSED":
            return '<span class="failure-matrix-icon failure-matrix-pass">✓</span>'
        if status == "FAILED":
            return '<span class="failure-matrix-icon failure-matrix-fail">×</span>'
        return '<span class="failure-matrix-missing">—</span>'

    def _failure_reasons_for(test_id):
        # Unique failure reasons only, in selected-run order, so a reason
        # repeated across multiple failed runs is listed once.
        reasons = []
        for run_id in selected_run_ids:
            if status_by_test[test_id].get(run_id) != "FAILED":
                continue
            reason = str(error_by_test.get(test_id, {}).get(run_id, "") or "").strip()
            if reason and reason not in {"—", "-", "N/A", "NA"} and reason not in reasons:
                reasons.append(reason)
        return reasons

    rows_html = []
    for test_id in failed_test_ids:
        cells = "".join(
            f'<td class="failure-matrix-status">{_status_icon(status_by_test[test_id].get(run_id, ""))}</td>'
            for run_id in selected_run_ids
        )
        reasons = _failure_reasons_for(test_id)
        if reasons:
            reasons_html = "".join(
                f'<div class="failure-matrix-reason-line">• {html.escape(reason)}</div>'
                for reason in reasons
            )
        else:
            reasons_html = "—"
        rows_html.append(
            "<tr>"
            f'<td class="failure-matrix-test">{html.escape(str(test_id))}</td>'
            f'<td class="failure-matrix-test">{html.escape(str(test_names.get(test_id, test_id)))}</td>'
            f'<td class="failure-matrix-test">{html.escape(str(test_modules.get(test_id, "Unknown")))}</td>'
            f"{cells}"
            f'<td class="failure-matrix-pattern">{html.escape(_pattern_for(test_id))}</td>'
            f'<td class="failure-matrix-reasons">{reasons_html}</td>'
            "</tr>"
        )

    header_cells = "".join(
        f'<th>{html.escape(str(run_id))}</th>' for run_id in selected_run_ids
    )
    table_html = f"""
    <style>
        .failure-matrix-wrap {{
            width: 100%;
            overflow-x: auto;
            margin-top: 12px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #ffffff;
        }}
        .failure-matrix {{
            width: 100%;
            min-width: 900px;
            table-layout: auto;
            border-collapse: collapse;
            background: #ffffff;
            font-size: 17px;
        }}
        .failure-matrix th,
        .failure-matrix td {{
            white-space: nowrap;
        }}
        .failure-matrix th {{
            padding: 14px 18px;
            text-align: center;
            font-weight: 750;
            color: #111827;
            border-bottom: 1px solid #e5e7eb;
        }}
        .failure-matrix th:first-child,
        .failure-matrix th:last-child {{ text-align: left; }}
        .failure-matrix th.failure-matrix-pattern,
        .failure-matrix th.failure-matrix-reasons {{ text-align: left; }}
        .failure-matrix td {{
            padding: 16px 18px;
            border-bottom: 1px solid #eef0f2;
            color: #111827;
            height: 58px;
        }}
        .failure-matrix tr:last-child td {{ border-bottom: 0; }}
        .failure-matrix-test {{ font-weight: 650; }}
        .failure-matrix-status {{ text-align: center; }}
        .failure-matrix-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 4px;
            color: #ffffff;
            font-size: 30px;
            font-weight: 800;
            line-height: 1;
        }}
        .failure-matrix-pass {{ background: #16a34a; }}
        .failure-matrix-fail {{
            color: #dc2626;
            background: transparent;
            font-size: 42px;
        }}
        .failure-matrix-missing {{
            color: #6b7280;
            font-size: 30px;
            font-weight: 650;
        }}
        .failure-matrix td.failure-matrix-reasons {{
            white-space: normal;
            line-height: 1.5;
        }}
        .failure-matrix-reason-line {{
            white-space: nowrap;
        }}
    </style>
    <div class="failure-matrix-wrap">
        <table class="failure-matrix">
            <thead>
                <tr>
                    <th>Test Case</th>
                    <th>Test Name</th>
                    <th>Module</th>
                    {header_cells}
                    <th class="failure-matrix-pattern">Pattern</th>
                    <th class="failure-matrix-reasons">Failure Reasons</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_analytics_dashboard():
    """Render Page 2 as a filter-first, JSON-driven visual dashboard."""
    render_configuration_sidebar(2)

    chroma_path = st.session_state.get("config_chroma_path", DEFAULT_CHROMA_PATH)
    chroma_collection_name = st.session_state.get("config_chroma_collection", DEFAULT_CHROMA_COLLECTION)

    st.markdown(
        """
        <style>
        .dashboard-hero {
            position: relative; overflow: hidden;
            padding: 30px 36px; border-radius: 26px; margin-bottom: 18px;
            background:
                radial-gradient(circle at 88% 16%, rgba(96,165,250,.30), transparent 25%),
                radial-gradient(circle at 75% 110%, rgba(167,139,250,.18), transparent 30%),
                linear-gradient(135deg, #0f1b3d 0%, #17336f 54%, #2456a5 100%);
            border: 1px solid rgba(255,255,255,.12);
            box-shadow: 0 18px 45px rgba(15,23,42,.14);
            color: #fff;
        }
        .dashboard-hero h1 {
            margin: 0; font-size: 2.15rem; line-height: 1.1;
            font-weight: 850; letter-spacing: -.03em; color: #fff;
        }
        .dashboard-hero p {
            margin: .55rem 0 0 0; color: #d7e4fb;
            font-size: .96rem; line-height: 1.5;
        }
        .dashboard-hero:after {
            content: ""; position:absolute; width:210px; height:210px;
            right:-70px; top:-120px; border-radius:50%;
            border:1px solid rgba(255,255,255,.10); pointer-events:none;
        }
        </style>
        <div class="dashboard-hero">
            <h1>📊 Intelligent Test Report Analyzer &amp; Insights Engine</h1>
            <p>Test Execution Analytics &amp; Quality Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hydrate from ChromaDB on every fresh Streamlit session. ChromaDB is the
    # persistent post-ingestion source after raw MongoDB data is cleaned up,
    # so closing/reopening the UI does not clear dashboard records or filters.
    _hydrate_dashboard_from_chroma()

    report = st.session_state.get("analytics_result")

    if not report:
        # Keep the filter section visible even when MongoDB has no records.
        # The controls are rendered with empty option lists and disabled date
        # inputs, making the empty-state behavior explicit rather than hiding
        # the filter area entirely.
        report = {"records": []}
        st.session_state.analytics_result = report

    records = _get_dashboard_records(report)
    dashboard_report = _get_dashboard_report(report)

    # After filters are generated, Module 5 analytics are recalculated from
    # ChromaDB and replace the report categories used by the existing widgets.
    chroma_categories = st.session_state.get("dashboard_chroma_analytics")
    if chroma_categories:
        dashboard_report = dict(dashboard_report)
        dashboard_report.update(chroma_categories)

    st.session_state.dashboard_flaky_ids = (
        dashboard_report.get("flaky_test_detector", {}) or {}
    ).get("tests", []) or []

    from backend.analytics_engine import parse_datetime

    # Date filtering is only meaningful when at least one record has a
    # parseable execution_time. Falling back to "today" silently excluded
    # every record whose timestamp could not be verified, which was the
    # root cause of valid filter combinations returning zero matches.
    raw_min_dt = st.session_state.get("dashboard_filter_date_min")
    raw_max_dt = st.session_state.get("dashboard_filter_date_max")
    date_filtering_available = raw_min_dt is not None and raw_max_dt is not None
    min_dt = raw_min_dt or datetime.now()
    max_dt = raw_max_dt or datetime.now()

    # ---- Filter panel ----
    with st.container(border=True):
        header_left, header_right = st.columns([5, 1])
        with header_left:
            st.markdown("### 🎛️ Report Filters")
        with header_right:
            generate = st.button(
                "📈 Generate Report",
                type="primary",
                use_container_width=True,
                key="generate_dashboard_report",
                disabled=not bool(records),
            )

        options = _get_dashboard_filter_values(report)
        statuses = options["status"]
        modules = options["module"]

        row1 = st.columns(2)

        # From Date/To Date must remain enabled and editable regardless of
        # status/module selection, report state, or whether historical
        # execution timestamps have been discovered yet. When no historical
        # range is known yet, the min/max bounds are left unrestricted so the
        # user can still freely pick any date.
        from_date_kwargs = {"value": min_dt.date(), "key": "dashboard_from_date"}
        to_date_kwargs = {"value": max_dt.date(), "key": "dashboard_to_date"}
        if date_filtering_available:
            from_date_kwargs["min_value"] = min_dt.date()
            from_date_kwargs["max_value"] = max_dt.date()
            to_date_kwargs["min_value"] = min_dt.date()
            to_date_kwargs["max_value"] = max_dt.date()

        with row1[0]:
            from_date = st.date_input("From Date", **from_date_kwargs)
        with row1[1]:
            to_date = st.date_input("To Date", **to_date_kwargs)

        row2 = st.columns(2)
        with row2[0]:
            # Status is intentionally limited to the two requested
            # business-level options.  "Complete Report" = PASSED + FAILED;
            # "Failed" = FAILED only.
            selected_status = st.selectbox(
                "Status", ["Complete Report", "Failed"],
                key="dashboard_status",
            )
        with row2[1]:
            selected_module = st.selectbox(
                "Module", ["All"] + modules, key="dashboard_module"
            )

    if not records:
        st.info("The database currently has no test execution data. Filter options will appear automatically once data is uploaded.")
        return

    # Dashboard is deliberately hidden until the user clicks Generate Report.
    if not generate and not st.session_state.get("dashboard_report_generated", False):
        return

    if generate:
        from_dt = datetime.combine(from_date, datetime.min.time())
        to_dt = datetime.combine(to_date, datetime.max.time())

        if from_dt > to_dt:
            st.error("From Date must be earlier than or equal to To Date.")
            st.session_state.dashboard_report_generated = False
            return

        # ------------------------------------------------------------------
        # MODULE 5 + MODULE 6 BACKEND ORCHESTRATION
        # ------------------------------------------------------------------
        # The UI controls above are unchanged. Once Generate Report is
        # clicked, the Analytics Engine is the only path used to retrieve the
        # selected executions. It reads ChromaDB metadata, applies the filter
        # semantics, and performs vector/RAG retrieval over the same selection.
        # From/To Date are always user-editable, so the selected range is
        # always forwarded to the report query regardless of whether a
        # historical execution-time range had been discovered yet.
        filters = {
            "from_datetime": from_dt,
            "to_datetime": to_dt,
            "status": selected_status,
            "module": selected_module,
        }

        rag_query = (
            f"QA report filtered by status={selected_status}, module={selected_module}, "
            f"from={from_dt.isoformat(sep=' ')}, to={to_dt.isoformat(sep=' ')}. "
            "Identify relevant failures, recurring error messages, flaky behavior, "
            "regressions, module risks, execution trends and likely root causes."
        )

        with st.spinner("Retrieving filtered analytics and RAG context from ChromaDB..."):
            chroma_dashboard = query_analytics_from_chroma(
                chroma_path=chroma_path,
                chroma_collection_name=chroma_collection_name,
                filters=filters,
                rag_query=rag_query,
                rag_top_k=20,
            )

        filtered = chroma_dashboard.get("records", []) or []
        analytics_categories = chroma_dashboard.get("analytics", {}) or {}

        # Keep the existing dashboard state contract so every existing UI
        # component immediately renders the Chroma-derived records.
        st.session_state.dashboard_filtered_records = filtered
        st.session_state.dashboard_filter_debug = {
            "total": chroma_dashboard.get("source", {}).get("metadata_records_total", 0),
            "after_date": len(filtered),
            "after_status": len(filtered),
            "after_module": len(filtered),
            "final": len(filtered),
            "source": "ChromaDB",
        }
        st.session_state.dashboard_chroma_analytics = analytics_categories
        # Use the newly generated analytics for the current render as well as
        # subsequent reruns; this makes the Generate Report action immediately
        # reflect the selected filters in every dashboard section.
        dashboard_report = dict(dashboard_report)
        dashboard_report.update(analytics_categories)

        st.session_state.dashboard_rag_context = chroma_dashboard.get("rag", {}) or {}
        st.session_state.dashboard_chroma_source = chroma_dashboard.get("source", {}) or {}
        st.session_state.dashboard_filter_snapshot = {
            "from": from_dt.isoformat(sep=" "),
            "to": to_dt.isoformat(sep=" "),
            "status": selected_status,
            "module": selected_module,
        }
        st.session_state.dashboard_report_generated = True
        st.success("Report generated successfully.")

    filtered_records = st.session_state.get("dashboard_filtered_records", [])

    st.divider()

    if not filtered_records:
        st.warning("No test executions match the selected filters. Change the filters and click Generate Report again.")
        return

    tab_labels = [
        "📈 Overview",
        "🔻 Failure Analysis",
        "🌀 Flaky Tests",
        "🧠 AI Insights",
    ]

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_dashboard_charts(filtered_records, dashboard_report)
    with tabs[1]:
        try:
            _render_failure_analysis(filtered_records, dashboard_report, chroma_path, chroma_collection_name)
        except Exception as exc:
            st.error("Failure Analysis could not be rendered due to an unexpected error.")
            st.exception(exc)
    with tabs[2]:
        try:
            _render_flaky_tests_section(filtered_records, dashboard_report)
        except Exception as exc:
            st.error("Flaky Test Detection could not be rendered due to an unexpected error.")
            st.exception(exc)
    with tabs[3]:
        try:
            _render_ai_generated_insights(filtered_records, dashboard_report, chroma_path, chroma_collection_name)
        except Exception as exc:
            st.error("AI Insights could not be rendered due to an unexpected error.")
            st.exception(exc)


_SEVERITY_META = {
    "critical": {"icon": "🔴", "label": "Critical", "color": "#ef4444"},
    "warning": {"icon": "🟠", "label": "Warning", "color": "#f59e0b"},
    "success": {"icon": "🟢", "label": "Positive", "color": "#16a34a"},
    "info": {"icon": "🔵", "label": "Info", "color": "#3b82f6"},
}


def _classify_severity(text: str) -> str:
    """Heuristically classify a recommendation's severity for display."""
    lowered = text.lower()
    if any(word in lowered for word in ("poor", "critical", "quarantine", "regression signal")):
        return "critical"
    if any(word in lowered for word in ("improved", "positive", "effective")):
        return "success"
    if any(word in lowered for word in ("investigate", "flaky", "recurring", "rose", "risk")):
        return "warning"
    return "info"


def _build_rule_based_recommendations(
    *, rows, dashboard_report, top_modules, total_recent_failures, score, quality_label
):
    """Build concise, context-aware recommendations from filtered analytics.

    Deterministic fallback used when Azure OpenAI is not configured, so the
    AI Insights tab always has actionable guidance for the current filters.
    Each item includes a severity used to drive interactive filtering/styling.
    """
    recommendations = []

    if score < 50:
        recommendations.append({
            "text": (
                f"Overall quality is **{quality_label}** ({int(score)}/100) for the selected filters — "
                "prioritize stabilizing failing suites before adding new test coverage."
            ),
            "severity": "critical",
        })
    elif score < 85:
        recommendations.append({
            "text": (
                f"Overall quality is **{quality_label}** ({int(score)}/100) — "
                "address the failure hotspots below to move into the Excellent range."
            ),
            "severity": "warning",
        })

    for module, count in top_modules[:3]:
        pct = count / total_recent_failures * 100
        recommendations.append({
            "text": (
                f"Investigate **{module}**: {count} failures ({pct:.1f}% of recent failures) — "
                "review recent changes and add regression coverage for this module."
            ),
            "severity": "warning",
        })

    flaky_tests = ((dashboard_report or {}).get("flaky_test_detector") or {}).get("tests") or []
    if flaky_tests:
        top_flaky = flaky_tests[0]
        recommendations.append({
            "text": (
                f"Flaky test detected: **{top_flaky.get('test_name') or top_flaky.get('test_id')}** "
                f"in {top_flaky.get('module', 'Unknown')} ({int(top_flaky.get('failure_rate', 0) * 100)}% failure rate) — "
                "quarantine or fix before it erodes trust in the pipeline."
            ),
            "severity": "critical",
        })

    patterns = ((dashboard_report or {}).get("failure_pattern_detector") or {}).get("patterns") or []
    if patterns:
        top_pattern = patterns[0]
        recommendations.append({
            "text": (
                f"Recurring failure pattern (\"{top_pattern.get('pattern', '')[:80]}\") occurred "
                f"{top_pattern.get('occurrences', 0)} times across {len(top_pattern.get('affected_tests', []))} tests — "
                "treat as a single root cause rather than separate defects."
            ),
            "severity": "warning",
        })

    delta = ((dashboard_report or {}).get("trend_analysis") or {}).get("delta_vs_historical") or {}
    failure_delta = delta.get("failure_rate_points", 0) or 0
    if failure_delta > 2:
        recommendations.append({
            "text": (
                f"Failure rate rose by {failure_delta:.1f} points versus historical data — "
                "treat this as a regression signal and review the latest build changes."
            ),
            "severity": "critical",
        })
    elif failure_delta < -2:
        recommendations.append({
            "text": (
                f"Failure rate improved by {abs(failure_delta):.1f} points versus historical data — "
                "recent fixes appear effective; monitor to confirm the trend holds."
            ),
            "severity": "success",
        })

    if not recommendations:
        recommendations.append({
            "text": f"No significant risks detected for the current filters (quality score {int(score)}/100).",
            "severity": "info",
        })

    return recommendations[:6]


def _get_ai_recommendations(
    *,
    filters,
    dashboard_report,
    rag_context,
    rows,
    top_modules,
    total_recent_failures,
    score,
    quality_label,
    chroma_path,
    chroma_collection_name,
    force_refresh=False,
):
    """Return AI-driven recommendations for the current filter selection.

    Uses Azure OpenAI + RAG context when configured, falling back to a
    deterministic rule-based summary otherwise. Results are cached per
    filter selection so they refresh only when filters/dashboard change,
    unless force_refresh is requested from the interactive controls.
    """
    cache_key = hashlib.sha256(
        json.dumps(
            {"filters": filters, "records": len(rows), "score": round(score, 1)},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    cache = st.session_state.setdefault("ai_recommendations_cache", {})
    if not force_refresh and cache_key in cache:
        return cache_key, cache[cache_key]

    recommendations = None
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        try:
            insights = generate_ai_insights_with_rag(
                chroma_path=chroma_path,
                chroma_collection_name=chroma_collection_name,
                filters=filters,
                analytic_categories=dashboard_report or {},
                rag_context=rag_context or {},
            )
            text = (insights.get("model_response") or "").strip()
            if text:
                recommendations = [
                    {"text": line.strip("-• ").strip(), "severity": _classify_severity(line)}
                    for line in text.splitlines()
                    if line.strip("-• ").strip()
                ]
        except Exception:
            recommendations = None

    if not recommendations:
        recommendations = _build_rule_based_recommendations(
            rows=rows,
            dashboard_report=dashboard_report,
            top_modules=top_modules,
            total_recent_failures=total_recent_failures,
            score=score,
            quality_label=quality_label,
        )

    cache[cache_key] = recommendations
    return cache_key, recommendations


def _render_ai_generated_insights(filtered_records, dashboard_report, chroma_path, chroma_collection_name):
    """Render the structured, filter-aware Module 6 AI Insights experience."""
    from backend.analytics_engine import generate_ai_insights_with_rag

    rows = list(filtered_records or [])
    if not rows:
        st.info("No test execution data is available to generate AI insights.")
        return

    snapshot = st.session_state.get("dashboard_filter_snapshot", {}) or {}
    rag_context = st.session_state.get("dashboard_rag_context", {}) or {}
    analytics = dashboard_report or {}

    cache_payload = {
        "filters": snapshot,
        "record_ids": [str(r.get("mongo_record_id") or r.get("_id") or r.get("test_id") or i) for i, r in enumerate(rows)],
        "rag_count": len(rag_context.get("chunks", []) or []),
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, default=str).encode()).hexdigest()
    cache = st.session_state.setdefault("ai_insights_payload_cache", {})

    if cache_key not in cache:
        with st.spinner("Generating AI insights from filtered analytics and historical RAG evidence..."):
            payload = generate_ai_insights_with_rag(
                chroma_path=chroma_path,
                chroma_collection_name=chroma_collection_name,
                filters=snapshot,
                analytic_categories=analytics,
                rag_context=rag_context,
                records=rows,
            )
        cache[cache_key] = payload
    payload = cache[cache_key]
    insights = payload.get("insights", {}) or {}
    metrics = insights.get("metrics", {}) or {}
    quality = insights.get("quality", {}) or {}
    errors = insights.get("error_intelligence", {}) or {}
    recommendations = insights.get("recommendations", []) or []
    hotspots = insights.get("module_hotspots", []) or []
    patterns = insights.get("failure_patterns", []) or []
    flaky = insights.get("flaky_tests", []) or []
    trends = insights.get("trend_analysis", {}) or {}
    weekly_digest = insights.get("weekly_quality_digest", {}) or {}
    llm = insights.get("llm", {}) or {}
    flaky_count = len(_compute_flaky_rows_and_totals(rows, analytics)[0])

    # ---------------- Hero / executive summary ----------------
    summary = llm.get("executive_summary") or (
        f"{metrics.get('failed', 0)} of {metrics.get('total_tests', 0)} selected tests failed "
        f"({metrics.get('failure_rate', 0):.1f}%). "
        f"The dominant failure categories are {', '.join(list((errors.get('category_counts') or {}).keys())[:3]) or 'not yet classified'}."
    )
    risk = (llm.get("risk_level") or ("HIGH" if metrics.get("failure_rate", 0) >= 15 else "MEDIUM" if metrics.get("failure_rate", 0) >= 5 else "LOW")).upper()
    risk_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(risk, "🔵")
    st.markdown(f"## 🧠 AI Insights  <span style='font-size:.7em'>{risk_icon} {risk} RISK</span>", unsafe_allow_html=True)
    st.caption("A short summary of the selected test runs and the evidence behind it.")

    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown("### Summary")
            st.write(summary)
        with c2:
            score = quality.get("score")
            st.metric("Quality Score", f"{float(score):.0f}/100" if score not in (None, "") else "N/A")

    metric_cols = st.columns(6)
    metric_values = [
        ("Total Tests", metrics.get("total_tests", 0)),
        ("Failed", metrics.get("failed", 0)),
        ("Failure Rate", f"{metrics.get('failure_rate', 0):.1f}%"),
        ("Flaky Tests", flaky_count),
        ("Repeated Failures", len(patterns)),
        ("History Samples", len(insights.get("rag_evidence", []) or [])),
    ]
    for col, (label, value) in zip(metric_cols, metric_values):
        col.metric(label, value)

    # ---------------- Weekly quality digest ----------------
    st.markdown("### 🗓️ Weekly Digest")
    with st.container(border=True):
        st.markdown(f"**{weekly_digest.get('headline', 'Weekly quality summary is unavailable.')}**")
        period = weekly_digest.get("period", {}) or {}
        if period.get("start") and period.get("end"):
            st.caption(f"Execution window: {period['start']} to {period['end']}")
        st.markdown("**Findings**")
        for finding in (weekly_digest.get("findings") or [])[:4]:
            st.markdown(f"- **{finding.get('signal', 'Quality signal')}**: {finding.get('evidence', '')}")

    # Store root_causes for later use at the end
    root_causes = llm.get("root_causes") or []
    if not root_causes:
        for error in (errors.get("top_errors") or [])[:5]:
            cat = error.get("category", "OTHER")
            count = error.get("occurrences", 0)
            module_text = ", ".join(error.get("affected_modules", [])[:3]) or "selected modules"
            root_causes.append({
                "title": error.get("error_message") or error.get("signature") or "Failure pattern",
                "category": cat,
                "evidence": f"{count} occurrence(s) across {module_text}.",
                "hypothesis": {
                    "TIMEOUT": "Dependent service latency or test synchronization may be causing the timeout.",
                    "AUTHENTICATION": "The test identity/token may be invalid, expired or incorrectly configured.",
                    "AUTHORIZATION": "The test identity may not have permission for the requested resource.",
                    "RESOURCE_NOT_FOUND": "A dependent endpoint/resource may be unavailable in the selected environment.",
                    "ELEMENT_NOT_FOUND": "The UI element may not be ready or the selector may be unstable.",
                    "ASSERTION_FAILURE": "Observed application output differs from the expected test assertion.",
                }.get(cat, "The failure requires investigation against the supplied evidence."),
                "confidence": "Medium",
            })

    # ---------------- Error intelligence ----------------
    st.markdown("### 🧩 Error Details")
    top_errors = errors.get("top_errors", []) or []
    if top_errors:
        table = pd.DataFrame([
            {
                "Error Signature": e.get("signature", "")[:90],
                "Category": e.get("category", "OTHER"),
                "Occurrences": e.get("occurrences", 0),
                "Modules": ", ".join(e.get("affected_modules", [])[:2]),
            }
            for e in top_errors[:8]
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No failed error signatures in the selected data.")

    # ---------------- Hotspots ----------------
    st.markdown("### 🔥 Risk by Module")
    if hotspots:
        for h in hotspots[:6]:
            st.progress(min(float(h.get("failure_density", 0)) / 100, 1.0), text=f"{h.get('module', 'Unknown')} — {h.get('failure_density', 0):.1f}% ({h.get('failures', 0)} failures)")
        st.caption("Density is calculated from the selected failed executions.")
    else:
        st.info("No module failures found in the selected data.")

    # ---------------- Patterns ----------------
    st.markdown("### 🔁 Repeated Failures")
    if patterns:
        st.dataframe(pd.DataFrame([
            {"Pattern": p.get("pattern", "")[:80], "Occurrences": p.get("occurrences", 0), "Tests": len(p.get("affected_tests", []) or [])}
            for p in patterns[:8]
        ]), use_container_width=True, hide_index=True)
    else:
        st.info("No recurring failure pattern reached the detector threshold.")

    # ---------------- Flaky tests ----------------
    st.markdown("### 🌀 Flaky Tests")
    if flaky:
        st.dataframe(pd.DataFrame([
            {"Test": f.get("test_name") or f.get("test_id"), "Module": f.get("module", "Unknown"), "Failure Rate": f"{float(f.get('failure_rate', 0))*100:.1f}%", "Executions": f.get("total_executions", f.get("executions", ""))}
            for f in flaky[:8]
        ]), use_container_width=True, hide_index=True)
    else:
        st.info("No flaky tests were detected in the available execution history.")

    # ---------------- Root cause analysis ----------------
    with st.expander("🔎 Root Causes", expanded=False):
        if root_causes:
            for idx, rc in enumerate(root_causes[:6], 1):
                with st.container(border=True):
                    a, b = st.columns([5, 1])
                    with a:
                        st.markdown(f"**{idx}. {rc.get('title', 'Failure')}**  ")
                        st.caption(f"Category: `{rc.get('category', 'OTHER')}`")
                        st.write(f"**Evidence:** {rc.get('evidence', 'Not available')}")
                        st.write(f"**Likely cause:** {rc.get('hypothesis', 'Not available')}")
                    with b:
                        st.metric("Confidence", str(rc.get("confidence", "Medium")))
        else:
            st.info("No failed executions were found in the current selection.")

    # ---------------- Recommendations ----------------
    with st.expander("More actions by priority", expanded=False):
        priority_filter = st.multiselect(
            "Priority",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
            key="ai_insights_priority_filter",
        )
        visible_recs = [r for r in recommendations if str(r.get("priority", "MEDIUM")).upper() in priority_filter]
        if not visible_recs:
            st.info("No actions match the selected priority.")
        for rec in visible_recs:
            priority = str(rec.get("priority", "MEDIUM")).upper()
            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(priority, "🔵")
            with st.container(border=True):
                st.markdown(f"{icon} **{rec.get('title', 'Action')}** · `{priority}`")
                st.write(rec.get("action", ""))
                if rec.get("evidence"):
                    st.caption(f"Evidence: {rec['evidence']}")
                if rec.get("expected_impact"):
                    st.caption(f"Expected impact: {rec['expected_impact']}")

    if insights.get("llm_error"):
        st.caption("Azure OpenAI enrichment was unavailable for this run; the dashboard is showing deterministic analytics + RAG-based insights.")


def render_page_1():
    # ============================================================
    # MAIN UI
    # ============================================================

    # Page 1 owns the database/storage configuration. Page 2 owns the
    # analytics-report output configuration.
    render_configuration_sidebar(1)

    # Read the Page 1 configuration from session state. These values are
    # intentionally defined here because the upload handlers below use them.
    # Keeping them in session state also makes the same configuration available
    # when the user navigates to Page 2.
    mongo_uri = st.session_state.get("config_mongo_uri", DEFAULT_MONGO_URI)
    db_name = st.session_state.get("config_db_name", DEFAULT_DB_NAME)
    mongo_collection_name = st.session_state.get("config_mongo_collection", DEFAULT_MONGO_COLLECTION)
    chroma_path = st.session_state.get("config_chroma_path", DEFAULT_CHROMA_PATH)
    chroma_collection_name = st.session_state.get("config_chroma_collection", DEFAULT_CHROMA_COLLECTION)

    # ------------------------------------------------------------------
    # Page 1 visual refresh only.  The uploader, widget keys, handlers,
    # storage pipeline, preview, and navigation below are intentionally
    # unchanged so existing functionality is preserved.
    # ------------------------------------------------------------------
    st.markdown(
        """
        <style>
        .upload-page-shell {
            margin: 4px 0 26px 0;
        }
        .upload-page-hero {
            position: relative;
            overflow: hidden;
            border-radius: 26px;
            padding: 34px 38px;
            min-height: 245px;
            background:
                radial-gradient(circle at 88% 16%, rgba(96,165,250,.32), transparent 25%),
                radial-gradient(circle at 75% 110%, rgba(167,139,250,.20), transparent 30%),
                linear-gradient(135deg, #0f1b3d 0%, #17336f 54%, #2456a5 100%);
            border: 1px solid rgba(255,255,255,.12);
            box-shadow: 0 18px 45px rgba(15,23,42,.16);
            color: #fff;
        }
        .upload-page-hero:before, .upload-page-hero:after {
            content: "";
            position: absolute;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.10);
            pointer-events: none;
        }
        .upload-page-hero:before { width: 280px; height: 280px; right: -90px; top: -150px; }
        .upload-page-hero:after { width: 170px; height: 170px; right: 115px; bottom: -120px; }
        .upload-hero-content { position: relative; z-index: 2; max-width: 780px; }
        .upload-eyebrow {
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 11px; border-radius:999px;
            background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.15);
            font-size:.72rem; font-weight:800; letter-spacing:.08em;
            text-transform:uppercase; color:#dbeafe; margin-bottom:14px;
        }
        .upload-title {
            font-size:2.45rem; line-height:1.08; font-weight:850;
            letter-spacing:-.035em; color:#fff; margin:0;
        }
        .upload-subtitle {
            margin:13px 0 0 0; color:#d7e4fb; font-size:1rem;
            max-width:730px; line-height:1.62;
        }
        .upload-hero-pills {
            display:flex; flex-wrap:wrap; gap:9px; margin-top:19px;
        }
        .upload-hero-pill {
            display:inline-flex; align-items:center; gap:7px;
            padding:7px 11px; border-radius:999px;
            background:rgba(255,255,255,.10); color:#eef6ff;
            border:1px solid rgba(255,255,255,.13); font-size:.79rem; font-weight:650;
        }
        .upload-section-label {
            margin: 0 0 11px 3px; color:#16233f; font-size:1.05rem; font-weight:850;
        }
        .upload-section-help {
            margin: -4px 0 14px 3px; color:#68758d; font-size:.84rem;
        }
        .upload-step-grid {
            display:grid; grid-template-columns:repeat(3, minmax(0,1fr));
            gap:14px; margin:0 0 20px 0;
        }
        .upload-step {
            position:relative; overflow:hidden;
            background:#fff; border:1px solid #e4eaf3; border-radius:17px;
            padding:17px 17px 16px 17px; min-height:112px;
            box-shadow:0 7px 22px rgba(15,23,42,.045);
        }
        .upload-step:after {
            content:""; position:absolute; width:72px; height:72px; border-radius:50%;
            right:-26px; bottom:-30px; background:#f1f6ff;
        }
        .upload-step-num {
            display:inline-flex; width:29px; height:29px; border-radius:10px;
            align-items:center; justify-content:center; background:#edf4ff;
            color:#245bc2; font-weight:850; font-size:.78rem; margin-bottom:9px;
        }
        .upload-step-title { font-weight:820; color:#1c2942; font-size:.93rem; }
        .upload-step-text { color:#748198; font-size:.79rem; margin-top:5px; line-height:1.45; }
        .upload-card {
            border:1px solid #dfe7f2; border-radius:20px; padding:21px 22px 16px 22px;
            background:linear-gradient(180deg,#ffffff 0%,#fbfdff 100%);
            box-shadow:0 10px 28px rgba(15,23,42,.055); margin-bottom:15px;
        }
        .upload-card-title { font-size:1.16rem; font-weight:850; color:#17233d; margin-bottom:3px; }
        .upload-card-help { color:#6c7890; font-size:.84rem; margin-bottom:0; }
        .upload-benefits { display:flex; flex-wrap:wrap; gap:9px; margin-top:14px; }
        .upload-benefits span { display:inline-flex; align-items:center; padding:7px 10px; border-radius:999px; background:#f3f7ff; border:1px solid #dce8fb; color:#315a9d; font-size:.76rem; font-weight:750; }
        .upload-page-shell { max-width:1180px; margin:0 auto; }
        .upload-card { backdrop-filter: blur(4px); }
        .upload-title { max-width:900px; }
        section[data-testid="stFileUploaderDropzone"] {
            border:2px dashed #b9ccef !important;
            border-radius:18px !important;
            background:linear-gradient(180deg,#f8fbff 0%,#f2f7ff 100%) !important;
            padding:12px 14px !important;
            min-height:0;
            transition:border-color .15s ease, box-shadow .15s ease, background .15s ease;
        }
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color:#4d7fe0 !important;
            background:#f5f9ff !important;
            box-shadow:0 0 0 4px rgba(77,127,224,.08);
        }
        section[data-testid="stFileUploaderDropzone"] button {
            border-radius:10px !important; font-weight:750 !important;
        }
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
            border-radius:12px !important; border:1px solid #d8e2f1 !important;
            background:#fff !important;
        }
        .upload-security-note {
            display:flex; align-items:center; gap:9px; margin-top:12px;
            color:#64748b; font-size:.78rem;
        }
        .upload-side-card {
            min-height: 100%; padding: 22px; border-radius: 18px;
            background: linear-gradient(180deg,#f8fbff 0%,#f4f7fc 100%);
            border: 1px solid #dfe7f2; box-shadow: 0 8px 24px rgba(15,23,42,.045);
        }
        .upload-side-title { font-size: 1.05rem; font-weight: 850; color:#17233d; margin-bottom:16px; }
        .upload-side-item { display:flex; gap:11px; margin:0 0 15px 0; align-items:flex-start; }
        .upload-side-item > span { display:flex; width:24px; height:24px; border-radius:8px; align-items:center; justify-content:center; background:#e8f5ee; color:#15803d; font-weight:900; flex:0 0 24px; }
        .upload-side-item b { display:block; color:#26334b; font-size:.86rem; margin-bottom:2px; }
        .upload-side-item small { display:block; color:#718096; font-size:.76rem; line-height:1.45; }
        .upload-side-note { margin-top:18px; padding:11px 12px; border-radius:11px; background:#fff; border:1px solid #e3e9f2; color:#68758d; font-size:.74rem; line-height:1.45; }
        @media (max-width: 850px) {
            .upload-step-grid { grid-template-columns:1fr; }
            .upload-page-hero { padding:27px 24px; min-height:0; }
            .upload-title { font-size:1.9rem; }
        }
        </style>
        <div class="upload-page-shell">
          <div class="upload-page-hero">
            <div class="upload-hero-content">
              <div class="upload-eyebrow">✦ Intelligent Test Report Analyzer</div>
              <div class="upload-title">Test Execution Analytics &amp; Quality Insights</div>
              <div class="upload-subtitle">Upload a test execution report, review the data, and start your existing analysis workflow. Track failures, trends, modules, and flaky tests from one place.</div>
              <div class="upload-hero-pills">
                <span class="upload-hero-pill">✓ Review before processing</span>
                <span class="upload-hero-pill">✓ Historical run analysis</span>
                <span class="upload-hero-pill">✓ Flaky-test detection</span>
              </div>
            </div>
          </div>
          <div style="height:18px"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep Streamlit's uploader fully functional while hiding its built-in
    # size-limit helper text from the UI. The application accepts any file type.
    st.markdown(
        """
        <style>
        /* Hide Streamlit's built-in heading anchor/link icon from subheaders. */
        .stApp h3 a {
            display: none !important;
        }

        /* Hide Streamlit's built-in size/format helper text only. */
        section[data-testid="stFileUploaderDropzone"] small,
        section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] small,
        section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span {
            display: none !important;
        }

        /* Show the native remove-file X only while hovering the selected file. */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
            position: relative;
        }

        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] button[aria-label="Remove file"] {
            opacity: 0 !important;
            visibility: hidden !important;
            transition: opacity 0.15s ease-in-out, visibility 0.15s ease-in-out;
            position: absolute !important;
            top: 4px !important;
            right: 4px !important;
            z-index: 20 !important;
        }

        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"]:hover button[aria-label="Remove file"] {
            opacity: 1 !important;
            visibility: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    select_report_col = st.container()
    with select_report_col:
        with st.container(border=True):
            st.markdown("### 📤 Select your test report")
            st.caption("Choose the report you want to analyze. You can review the parsed data before it is processed.")
            uploaded_file = st.file_uploader(
                "Choose a test report file",
                type=None,
                label_visibility="collapsed",
            )


    def render_paginated_preview(records, file_name):
        """Display rows read directly from the currently selected uploaded file."""
        if not records:
            st.warning("No rows were found in the uploaded file.")
            return

        rows_per_page = 10
        total_rows = len(records)
        total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
        current_page = min(
            max(st.session_state.get("preview_page", 1), 1),
            total_pages,
        )
        st.session_state.preview_page = current_page

        display_columns = {
            "run_case_id": "Run Case ID",
            "test_id": "Test Case ID",
            "test_name": "Test Case Name",
            "module": "Module Name",
            "status": "Status",
            "error_message": "Error Message",
            "execution_time": "Execution Date and Time",
        }

        normalized_rows = []
        for record in records:
            normalized_rows.append({
                display_name: display_value(record.get(source_key, ""))
                for source_key, display_name in display_columns.items()
            })

        preview_df = pd.DataFrame(normalized_rows)
        start_index = (current_page - 1) * rows_per_page
        end_index = min(start_index + rows_per_page, total_rows)
        page_df = preview_df.iloc[start_index:end_index].copy()

        st.dataframe(page_df, use_container_width=True, hide_index=True)

        if total_rows > rows_per_page:
            prev_col, page_col, next_col = st.columns([1, 2, 1])
            with prev_col:
                if st.button(
                    "← Previous",
                    disabled=current_page == 1,
                    use_container_width=True,
                    key="preview_previous",
                ):
                    st.session_state.preview_page = current_page - 1
                    st.rerun()
            with page_col:
                st.markdown(
                    f"<div style='text-align:center; padding:8px;'>"
                    f"Page <b>{current_page}</b> of <b>{total_pages}</b>"
                    f" &nbsp;|&nbsp; Rows {start_index + 1}-{end_index} of {total_rows}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with next_col:
                if st.button(
                    "Next →",
                    disabled=current_page == total_pages,
                    use_container_width=True,
                    key="preview_next",
                ):
                    st.session_state.preview_page = current_page + 1
                    st.rerun()


    if uploaded_file is None:
        pass
    else:
        try:
            # Show Data reads the selected file directly. The existing parser is
            # invoked only when Upload is clicked, so preview never depends on MongoDB
            # or on normalized/database records.
            file_hash = calculate_file_hash(uploaded_file)
            parsed_records = None
            detected_format = ""
            parser_type = ""
            records = []
            unique_test_count = 0
            repeated_execution_count = 0

            # A newly selected file always starts with its report hidden.
            # The table becomes visible only after the user clicks "Show Report".
            if st.session_state.get("preview_file_signature") != file_hash:
                st.session_state.preview_file_signature = file_hash
                st.session_state.preview_page = 1

            # Keep the action button directly beneath the selected-file area.
            upload_button = st.button(
                "🚀 Upload & Analyze",
                type="primary",
                use_container_width=True,
                key="upload_process_button",
            )

            if upload_button:
                # Preserve the existing upload/normalization pipeline unchanged.
                parsed_records, detected_format, parser_type = parse_uploaded_report(
                    uploaded_file.name, uploaded_file.getvalue()
                )
                records, unique_test_count, repeated_execution_count = tag_duplicate_records(
                    parsed_records
                )
                if not records:
                    raise ValueError("The uploaded file contains no records.")

                # ------------------------------------------------
                # DUPLICATE FILE / DATA CHECK
                # ------------------------------------------------
                # Reject duplicate rows inside the selected report and reject
                # an exact file or exact execution data that already exists in
                # the durable history. ChromaDB is the sole durable guard
                # because raw MongoDB rows are deleted after successful
                # ChromaDB verification.
                if repeated_execution_count:
                    raise DuplicateFileError(
                        f"Upload blocked: The selected report contains {repeated_execution_count} duplicate test record(s). Remove duplicate rows and try again."
                    )

                candidate_record_hashes = {
                    record_fingerprint(record) for record in records
                }
                chroma_duplicate = chroma_upload_duplicate_info(
                    chroma_path=chroma_path,
                    chroma_collection_name=chroma_collection_name,
                    file_hash=file_hash,
                    record_hashes=candidate_record_hashes,
                    source_file_name=uploaded_file.name,
                    guard_namespace=UPLOAD_GUARD_NAMESPACE,
                )

                if chroma_duplicate["same_file"]:
                    raise DuplicateFileError(
                        "Upload blocked: This exact report file has already been uploaded."
                    )

                if chroma_duplicate["duplicate_records"]:
                    raise DuplicateFileError(
                        "Upload blocked: This report contains execution data that already exists in the test history. No duplicate data was uploaded."
                    )

                # ------------------------------------------------
                # STEP 1 - FILE → MONGODB
                # ------------------------------------------------
                with st.spinner("Storing uploaded data in MongoDB..."):
                    mongo_result = upload_records_to_mongodb(
                        mongo_uri=mongo_uri,
                        db_name=db_name,
                        collection_name=mongo_collection_name,
                        records=records,
                        source_file_name=uploaded_file.name,
                        file_hash=file_hash,
                        guard_namespace=UPLOAD_GUARD_NAMESPACE,
                    )

                st.success(
                    f"MongoDB: inserted **{mongo_result['inserted_count']}** unique record(s)."
                )

                if mongo_result["batch_id"] is None:
                    # All rows were already present. The duplicate handling is
                    # intentionally silent; do not send an empty batch downstream.
                    st.success("Upload processed successfully.")
                else:
                    # ------------------------------------------------
                    # STEP 2 - MONGODB → CHROMADB
                    # ------------------------------------------------
                    with st.spinner("Moving uploaded MongoDB data to ChromaDB..."):
                        chroma_result = sync_uploaded_batch_to_chroma(
                            mongo_uri=mongo_uri,
                            db_name=db_name,
                            mongo_collection_name=mongo_collection_name,
                            batch_id=mongo_result["batch_id"],
                            source_file_name=uploaded_file.name,
                            chroma_path=chroma_path,
                            chroma_collection_name=chroma_collection_name,
                        )

                    st.success("✅ Upload → MongoDB → ChromaDB completed successfully.")
                    deletion_result = chroma_result.get("mongo_raw_deletion", {})
                    st.success(
                        f"✅ MongoDB raw data retained: **{deletion_result.get('remaining_count', 0)}** record(s) kept in MongoDB."
                    )

                    # Show the database/ChromaDB summary immediately after the
                    # successful MongoDB → ChromaDB message.
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                    with metric_col1:
                        st.metric(
                            "MongoDB Inserted",
                            mongo_result["inserted_count"],
                        )

                    with metric_col2:
                        st.metric(
                            "Sent to ChromaDB",
                            chroma_result["records_sent"],
                        )

                    with metric_col3:
                        st.metric(
                            "This Upload in ChromaDB",
                            chroma_result["records_sent"],
                        )

                    with metric_col4:
                        st.metric(
                            "Total ChromaDB",
                            chroma_result["chroma_total"],
                        )

                    # ------------------------------------------------
                    # STEP 3 - CHROMADB → ANALYTICS → JSON
                    # ------------------------------------------------
                    # After the file reaches ChromaDB, run the existing analytics
                    # Run the analytics engine after ChromaDB ingestion.
                    # The report is kept in memory and persisted to MongoDB;
                    # no JSON output file is generated.
                    with st.spinner("Running analytics..."):
                        analytics_week = _week_label_for_record(records[0])
                        analytics_output_dir = st.session_state.get(
                            "config_analytics_report",
                            DEFAULT_OUTPUT_DIR,
                        )
                        analytics_result = generate_analytics_after_chroma(
                            week_label=analytics_week,
                            upload_batch_id=mongo_result["batch_id"],
                            source_file=uploaded_file.name,
                            namespace=build_namespace(chroma_collection_name, uploaded_file.name),
                            mongo_uri=mongo_uri,
                            db_name=db_name,
                            mongo_collection_name=mongo_collection_name,
                            chroma_path=chroma_path,
                            chroma_collection_name=chroma_collection_name,
                            output_dir=analytics_output_dir,
                        )

                    st.session_state.analytics_result_path = None
                    st.session_state.analytics_result = analytics_result["report"]

                    # Add this upload to the cumulative Page 2 filter catalog.
                    # Existing values remain untouched; only new unique values are added.
                    _update_dashboard_filter_catalog(analytics_result["report"])

                    # A new upload should require Generate Report again, so stale
                    # results from a previous upload are not displayed automatically.
                    st.session_state.dashboard_report_generated = False
                    st.session_state.dashboard_filtered_records = []
                    st.session_state.dashboard_filter_snapshot = {}
                    _reset_dashboard_filter_widget_state()

                    # Keep the analytics completion message directly below the
                    # ChromaDB summary. Analytics reports are not written to disk.
                    st.success("Analytics processing completed. Results are stored in MongoDB and available on the dashboard.")

                    # Keep the navigation control at the same location as the
                    # analytics completion output while making it persistent across
                    # Streamlit reruns.
                    st.session_state.show_dashboard_button = True

                    # The dashboard button is rendered outside the upload-button
                    # click block below. Streamlit reruns the script when the user
                    # clicks a button; if this button were created only while
                    # upload_button == True, it would disappear on the rerun and
                    # could never receive its own click event.




        except DuplicateFileError as exc:
            # Exact duplicate files must show the requested failure message.
            st.error(str(exc))
        except Exception as exc:
            st.error(
                "❌ Upload processing failed. The file could not complete the "
                "MongoDB → ChromaDB → Analytics pipeline."
            )
            st.exception(exc)


    # Render the dashboard navigation button independently of the upload
    # button's click event. This keeps it present on the rerun caused by the
    # user's dashboard-button click, allowing Streamlit to process that click.
    if st.session_state.get("show_dashboard_button", False):
        if st.button(
            "🔎 View Analytics Dashboard →",
            type="primary",
            use_container_width=True,
            key="view_dashboard_button",
        ):
            st.session_state.current_page = 2
            st.query_params["page"] = "dashboard"
            st.rerun()


    # MongoDB remains the persistent source of truth. Page 2 hydrates its
    # dashboard state from these stored records after a UI reload, so a new
    # upload is not required just to restore filter options.


if st.session_state.current_page == 1:
    render_page_1()
elif st.session_state.current_page == 2:
    render_analytics_dashboard()
else:
    render_page_1()

render_page_navigation()
