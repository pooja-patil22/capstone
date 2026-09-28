"""
MongoDB retrieval filter for downstream test-execution processing.

This module is intentionally a thin filter layer:
MongoDB documents -> seven approved business fields + normalized error metadata -> chunking/embedding.

It does not modify MongoDB documents and it does not perform deduplication,
sorting, or other business logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping

try:
    from backend.app.error_normalizer import normalize_error
except ModuleNotFoundError:
    from app.error_normalizer import normalize_error


ALLOWED_FIELDS = (
    "run_case_id",
    "test_case_id",
    "test_case_name",
    "module_name",
    "status",
    "error_message",
    "execution_date_and_time",
)

# Derived error fields are retained separately from the seven business fields.
ERROR_FIELDS = (
    "raw_error_message",
    "error_category",
    "error_signature",
    "secondary_errors",
)

FIELD_ALIASES = {
    "run_case_id": (
        "run_case_id", "runcase_id", "run_case", "run_caseid",
        "run_id", "runid", "execution_id", "execution_run_id",
    ),
    "test_case_id": (
        "test_case_id", "testcase_id", "test_id", "testid", "case_id",
    ),
    "test_case_name": (
        "test_case_name", "testcase_name", "test_name", "testname",
        "name", "scenario", "test",
    ),
    "module_name": (
        "module_name", "module", "component", "feature", "area", "suite",
        "test_suite", "classname", "class", "test_module", "component_name",
        "package",
    ),
    "status": (
        "status", "result", "outcome", "test_status", "execution_status", "state",
    ),
    "error_message": (
        "error_message", "error", "failure_message", "failure_reason",
        "message", "exception", "error_detail", "errormessage",
    ),
    "execution_date_and_time": (
        "execution_date_and_time", "execution_date_time", "execution_time", "execution_date",
        "executed_at", "timestamp", "execution_timestamp", "run_time",
        "date", "datetime",
    ),
}


def _normalize_field_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(value or "").strip().lower()
    ).strip("_")


def _is_blank(value: Any) -> bool:
    return value is None or value == ""


def _find_source_value(record: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    normalized = {
        _normalize_field_name(key): key
        for key in record.keys()
    }
    for alias in aliases:
        source_key = normalized.get(_normalize_field_name(alias))
        if source_key is not None:
            value = record.get(source_key)
            if not _is_blank(value):
                return value
    return ""


def filter_mongodb_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return the seven approved fields plus derived error metadata.

    Any unrelated MongoDB field is discarded before the record reaches
    chunking/embedding. Error fields are normalized here as a safety net for
    older MongoDB records that were stored before parser normalization.
    """
    result = {
        field: _find_source_value(record, FIELD_ALIASES[field])
        for field in ALLOWED_FIELDS
    }

    raw_error = record.get("raw_error_message")
    if _is_blank(raw_error):
        raw_error = result["error_message"]
    if not _is_blank(raw_error):
        error_details = normalize_error(raw_error)
        result["error_message"] = error_details["error_message"]
        for field in ERROR_FIELDS:
            result[field] = error_details[field]

    return result


def filter_mongodb_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Filter a MongoDB result set while preserving record order."""
    return [filter_mongodb_record(record) for record in records]
