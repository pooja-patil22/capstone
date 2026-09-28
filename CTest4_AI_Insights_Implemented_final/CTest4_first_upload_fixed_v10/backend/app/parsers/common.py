import json
import re
from datetime import datetime
from typing import Any

from ..error_normalizer import normalize_error


def sanitize_error_message(value):
    """Backward-compatible helper returning the normalized primary error."""
    return normalize_error(value)["error_message"]


def make_mongo_safe(value: Any):
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(k): make_mongo_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_mongo_safe(v) for v in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def normalize_status(value):
    text = str(value or "").strip().lower()
    if text in {"passed", "pass", "success", "successful", "ok", "passed test"}:
        return "PASSED"
    if text in {"failed", "fail", "failure", "error", "broken"}:
        return "FAILED"
    if text in {"skipped", "skip", "ignored", "disabled"}:
        return "SKIPPED"
    if text in {"blocked", "block"}:
        return "BLOCKED"
    if "pass" in text:
        return "PASSED"
    if "fail" in text or "error" in text or "broken" in text:
        return "FAILED"
    if "skip" in text:
        return "SKIPPED"
    return text.upper().replace(" ", "_") if text else "UNKNOWN"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def first_non_empty(data, keys, default=""):
    for key in keys:
        value = data.get(key)
        if value in (None, ""):
            continue
        try:
            if value != value:  # NaN
                continue
        except Exception:
            pass
        return value
    return default


def canonical_test_record(
    run_case_id="",
    test_id="",
    test_name="",
    status="",
    module="",
    error_message="",
    execution_time="",
    duration="",
    version="",
    environment="",
    source_format="",
    **extra,
):
    def _blank_nan(value):
        try:
            return "" if value != value else value
        except Exception:
            return value

    run_case_id = _blank_nan(run_case_id)
    test_id = _blank_nan(test_id)
    test_name = _blank_nan(test_name)
    status = _blank_nan(status)
    module = _blank_nan(module)
    error_message = _blank_nan(error_message)
    execution_time = _blank_nan(execution_time)
    error_details = normalize_error(error_message)

    record = {
        "run_case_id": str(run_case_id or "").strip(),
        "test_id": str(test_id or test_name or "").strip(),
        "test_name": str(test_name or test_id or "Unknown Test").strip(),
        "status": normalize_status(status),
        "module": str(module or "Unknown").strip(),
        "raw_error_message": error_details["raw_error_message"],
        "error_message": error_details["error_message"],
        "error_category": error_details["error_category"],
        "error_signature": error_details["error_signature"],
        "secondary_errors": error_details["secondary_errors"],
        "execution_time": str(execution_time or "").strip(),
        "duration": safe_float(duration) if duration not in ("", None) else "",
        "version": str(version or "Unknown").strip(),
        "environment": str(environment or "Unknown").strip(),
        "source_format": source_format,
    }
    for key, value in extra.items():
        if value not in ("", None):
            record[key] = make_mongo_safe(value)
    return make_mongo_safe(record)
