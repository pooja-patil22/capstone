"""
Analytics Engine for QA Test Report Analyzer

Flow:
    Uploaded file
        -> MongoDB
        -> ChromaDB
        -> Analytics Engine
        -> in-memory analytics report + MongoDB analytics_results collection

The engine produces ONE analytics result per upload/week containing:
    1. Flaky Test Detector
    2. Failure Pattern Detector
    3. Trend Analysis
    4. Heatmap Generator

It reads the uploaded batch from MongoDB for the current report and also
uses historical MongoDB records where historical context is required
(e.g. flaky detection and trends).

Dependencies:
    pip install pymongo pandas

This module does not delete or replace ChromaDB data.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from pymongo import MongoClient
import chromadb

try:
    from backend.app.error_normalizer import normalize_error
except ModuleNotFoundError:
    from app.error_normalizer import normalize_error

# Load local environment variables (including Azure OpenAI settings) from .env.
load_dotenv()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME = "qa_test_analyzer"
DEFAULT_MONGO_COLLECTION = "test_execution_data"
DEFAULT_OUTPUT_DIR = "./analytics_results"

DEFAULT_RAG_TOP_K = 12
DEFAULT_RAG_QUERY_TOP_K = 20


@lru_cache(maxsize=8)
def _get_chroma_collection(chroma_path: str, chroma_collection_name: str):
    """Reuse Chroma handles across analytics calls in this process."""
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(name=chroma_collection_name)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

STATUS_ALIASES = {
    "passed": "PASSED",
    "pass": "PASSED",
    "success": "PASSED",
    "successful": "PASSED",
    "ok": "PASSED",
    "true": "PASSED",

    "failed": "FAILED",
    "fail": "FAILED",
    "failure": "FAILED",
    "error": "FAILED",
    "false": "FAILED",

    "skipped": "SKIPPED",
    "skip": "SKIPPED",
    "ignored": "SKIPPED",

    "blocked": "BLOCKED",
    "block": "BLOCKED",

    "not executed": "NOT_EXECUTED",
    "not_executed": "NOT_EXECUTED",
    "not run": "NOT_EXECUTED",
    "not_run": "NOT_EXECUTED",
}

FIELD_ALIASES = {
    "test_id": [
        "test_id", "testId", "test_case_id", "test_case",
        "testcase_id", "testCaseId", "case_id"
    ],
    "test_name": [
        "test_name", "testName", "name", "scenario",
        "test_case_name", "testcase_name"
    ],
    "status": [
        "status", "result", "outcome", "test_status",
        "execution_status", "testResult"
    ],
    "module": [
        "module", "module_name", "component", "feature",
        "area", "suite", "test_suite", "test_module", "component_name", "package"
    ],
    "error_message": [
        "error_message", "error", "failure_message", "failure_reason",
        "failure_reason", "message", "exception", "errorMessage"
    ],
    "execution_time": [
        "execution_time", "execution_date_and_time", "Execution Date and Time",
        "execution_date_time", "execution_date", "Execution Date", "executed_at",
        "timestamp", "execution_timestamp", "run_time"
    ],
    "start_time": ["start_time", "startTime", "started_at"],
    "end_time": ["end_time", "endTime", "completed_at", "finished_at"],
    "duration": [
        "duration", "duration_seconds", "execution_duration",
        "elapsed_time", "time_taken"
    ],
    "version": [
        "version", "build_version", "build", "build_number",
        "release", "release_version", "app_version", "build_no", "buildno"
    ],
    "environment": [
        "environment", "env", "test_environment", "env_name", "test_env", "stage"
    ],
    "source_format": [
        "source_format", "sourceFormat", "report_format", "parser_type"
    ],
}


def clean_value(value: Any) -> Any:
    """Return a JSON/analysis friendly value."""
    if value is None:
        return ""

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # ObjectId and similar BSON values
    if not isinstance(value, (str, int, float, bool, list, dict)):
        try:
            return str(value)
        except Exception:
            return ""

    return value


def safe_string(value: Any) -> str:
    value = clean_value(value)
    if value == "":
        return ""
    return str(value).strip()


def first_value(record: Dict[str, Any], aliases: Sequence[str]) -> Any:
    for field in aliases:
        if field in record:
            value = record.get(field)
            if value not in (None, ""):
                return value
    return ""


def normalize_status(value: Any) -> str:
    text = safe_string(value).lower().strip()
    if text in STATUS_ALIASES:
        return STATUS_ALIASES[text]

    # Common compound values
    if "pass" in text:
        return "PASSED"
    if "fail" in text or "error" in text:
        return "FAILED"
    if "skip" in text:
        return "SKIPPED"
    if "block" in text:
        return "BLOCKED"

    return text.upper().replace(" ", "_") if text else "UNKNOWN"


def parse_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    text = safe_string(value)
    if not text:
        return None

    # Allure commonly stores timestamps as Unix epoch milliseconds.
    try:
        numeric = float(text)
        if numeric > 100000000000:
            return datetime.fromtimestamp(numeric / 1000.0)
        if numeric > 1000000000:
            return datetime.fromtimestamp(numeric)
    except (TypeError, ValueError, OverflowError):
        pass

    # ISO and common timestamp formats
    candidates = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    normalized = text.replace("Z", "").replace("+00:00", "")

    try:
        return datetime.fromisoformat(normalized)
    except Exception:
        pass

    for fmt in candidates:
        try:
            return datetime.strptime(normalized, fmt)
        except Exception:
            continue

    return None


def parse_duration(record: Dict[str, Any]) -> Optional[float]:
    raw = first_value(record, FIELD_ALIASES["duration"])

    if raw not in (None, ""):
        try:
            value = float(raw)
            # Treat very large values as milliseconds.
            if value > 100000:
                value = value / 1000.0
            return round(value, 3)
        except Exception:
            pass

    start = parse_datetime(first_value(record, FIELD_ALIASES["start_time"]))
    end = parse_datetime(first_value(record, FIELD_ALIASES["end_time"]))

    if start and end:
        seconds = (end - start).total_seconds()
        if seconds >= 0:
            return round(seconds, 3)

    return None


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map different report schemas into one common analytics model."""
    execution_id = safe_string(record.get("execution_id", ""))
    test_id = safe_string(first_value(record, FIELD_ALIASES["test_id"]))
    test_name = safe_string(first_value(record, FIELD_ALIASES["test_name"]))
    status = normalize_status(first_value(record, FIELD_ALIASES["status"]))
    module = safe_string(first_value(record, FIELD_ALIASES["module"])) or "Unknown"
    raw_error_message = safe_string(
        record.get("raw_error_message")
        or first_value(record, FIELD_ALIASES["error_message"])
    )
    error_details = normalize_error(raw_error_message)
    error_message = error_details["error_message"]
    version = safe_string(first_value(record, FIELD_ALIASES["version"])) or "Unknown"
    environment = safe_string(
        first_value(record, FIELD_ALIASES["environment"])
    ) or "Unknown"
    source_format = safe_string(
        first_value(record, FIELD_ALIASES["source_format"])
    ) or "Unknown"

    execution_dt = parse_datetime(
        first_value(record, FIELD_ALIASES["execution_time"])
    )

    duration = parse_duration(record)

    mongo_id = safe_string(record.get("_id", ""))

    # Duplicate tagging is applied at upload time (app.py) and preserved here
    # unchanged; older records without these fields default to "not a duplicate".
    duplicate_group = safe_string(record.get("duplicate_group", ""))
    duplicate_count = record.get("duplicate_count", 1) or 1
    is_duplicate = bool(record.get("is_duplicate", False))

    return {
        "mongo_id": mongo_id,
        "execution_id": execution_id,
        "test_id": test_id or test_name or mongo_id,
        "test_name": test_name or test_id or "Unknown Test",
        "status": status,
        "module": module,
        "raw_error_message": error_details["raw_error_message"],
        "error_message": error_message,
        "error_category": error_details["error_category"],
        "error_signature": error_details["error_signature"],
        "secondary_errors": error_details["secondary_errors"],
        "execution_time": execution_dt.isoformat() if execution_dt else "",
        "_execution_dt": execution_dt,
        "duration_seconds": duration,
        "version": version,
        "environment": environment,
        "source_format": source_format,
        "upload_batch_id": safe_string(record.get("upload_batch_id", "")),
        "duplicate_group": duplicate_group,
        "duplicate_count": duplicate_count,
        "is_duplicate": is_duplicate,
    }


def json_safe(value: Any) -> Any:
    """Recursively make output JSON serializable."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# ---------------------------------------------------------------------------
# 1. Flaky Test Detector
# ---------------------------------------------------------------------------

def detect_flaky_tests(
    records: Sequence[Dict[str, Any]],
    minimum_executions: int = 3,
) -> Dict[str, Any]:
    """
    Detect tests that alternate between PASSED and FAILED across executions.

    A test is considered a flaky candidate when:
      - it has at least minimum_executions,
    - it has both PASSED and FAILED executions,
    - it has at least 2 status transitions,
      - and it is not simply always failing.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        test_id = record["test_id"]
        if test_id and test_id != "Unknown":
            groups[test_id].append(record)

    flaky_tests: List[Dict[str, Any]] = []

    for test_id, items in groups.items():
        if len(items) < minimum_executions:
            continue

        ordered = sorted(
            items,
            key=lambda x: x["_execution_dt"] or datetime.min,
        )

        statuses = [
            item["status"]
            for item in ordered
            if item["status"] in {"PASSED", "FAILED", "SKIPPED", "BLOCKED"}
        ]

        passed = statuses.count("PASSED")
        failed = statuses.count("FAILED")

        if passed == 0 or failed == 0:
            continue

        transitions = sum(
            1 for a, b in zip(statuses, statuses[1:]) if a != b
        )

        if transitions < 2:
            continue

        failure_rate = failed / len(statuses)

        # A test that fails almost every time is a failing test, not a
        # particularly useful flaky candidate.
        if failure_rate in (0, 1):
            continue

        # More transitions + balanced pass/fail behavior = higher score.
        transition_score = min(transitions / max(len(statuses) - 1, 1), 1.0)
        balance_score = 1.0 - abs(0.5 - failure_rate) * 2
        flaky_score = round(
            min(1.0, 0.6 * transition_score + 0.4 * balance_score),
            3,
        )

        flaky_tests.append({
            "test_id": test_id,
            "test_name": ordered[-1]["test_name"],
            "module": ordered[-1]["module"],
            "total_executions": len(statuses),
            "passed": passed,
            "failed": failed,
            "failure_rate": round(failure_rate, 3),
            "status_transitions": transitions,
            "flaky_score": flaky_score,
            "classification": "FLAKY",
            "last_status": statuses[-1],
            "version": ordered[-1]["version"],
        })

    flaky_tests.sort(
        key=lambda x: (
            x["flaky_score"],
            x["status_transitions"],
            x["failure_rate"],
        ),
        reverse=True,
    )

    return {
        "detector": "Flaky Test Detector",
        "flaky_test_count": len(flaky_tests),
        "tests": flaky_tests,
    }


# ---------------------------------------------------------------------------
# 2. Failure Pattern Detector
# ---------------------------------------------------------------------------

_DYNAMIC_PATTERNS = [
    (r"\b[0-9a-f]{8,}\b", "<ID>"),
    (r"\b\d{4,}\b", "<NUMBER>"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>"),
    (r"https?://\S+", "<URL>"),
    (r"'[^']{8,}'", "'<VALUE>'"),
    (r'"[^"]{8,}"', '"<VALUE>"'),
]


def normalize_failure_message(message: str) -> str:
    text = safe_string(message).lower()

    if not text:
        return "Unknown Failure"

    for pattern, replacement in _DYNAMIC_PATTERNS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    # Keep the message useful but prevent huge stack traces becoming a pattern.
    if len(text) > 220:
        text = text[:220] + "..."

    return text


def detect_failure_patterns(
    records: Sequence[Dict[str, Any]],
    minimum_occurrences: int = 2,
) -> Dict[str, Any]:
    patterns: Dict[str, Dict[str, Any]] = {}

    for record in records:
        if record["status"] != "FAILED":
            continue

        normalized = normalize_failure_message(record["error_message"])

        entry = patterns.setdefault(
            normalized,
            {
                "pattern": normalized,
                "occurrences": 0,
                "modules": Counter(),
                "tests": set(),
                "versions": Counter(),
                "examples": [],
            },
        )

        entry["occurrences"] += 1
        entry["modules"][record["module"]] += 1
        entry["tests"].add(record["test_id"])
        entry["versions"][record["version"]] += 1

        if len(entry["examples"]) < 3 and record["error_message"]:
            entry["examples"].append(record["error_message"])

    result = []

    for pattern, entry in patterns.items():
        if entry["occurrences"] < minimum_occurrences:
            continue

        result.append({
            "pattern": pattern,
            "occurrences": entry["occurrences"],
            "affected_modules": [
                {"module": module, "count": count}
                for module, count in entry["modules"].most_common()
            ],
            "affected_tests": sorted(entry["tests"]),
            "affected_versions": [
                {"version": version, "count": count}
                for version, count in entry["versions"].most_common()
            ],
            "examples": entry["examples"],
        })

    result.sort(key=lambda x: x["occurrences"], reverse=True)

    return {
        "detector": "Failure Pattern Detector",
        "pattern_count": len(result),
        "patterns": result,
    }


# ---------------------------------------------------------------------------
# 3. Trend Analysis
# ---------------------------------------------------------------------------

def trend_analysis(
    current_records: Sequence[Dict[str, Any]],
    historical_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def summarize(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(items)
        passed = sum(x["status"] == "PASSED" for x in items)
        failed = sum(x["status"] == "FAILED" for x in items)
        skipped = sum(x["status"] == "SKIPPED" for x in items)
        blocked = sum(x["status"] == "BLOCKED" for x in items)

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "blocked": blocked,
            "pass_rate": round((passed / total) * 100, 2) if total else 0,
            "failure_rate": round((failed / total) * 100, 2) if total else 0,
        }

    current = summarize(current_records)
    historical = summarize(historical_records)

    def delta(current_value: float, old_value: float) -> float:
        return round(current_value - old_value, 2)

    # Group historical data by version.
    by_version: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in historical_records:
        by_version[item["version"]].append(item)

    version_trends = []

    for version, items in by_version.items():
        summary = summarize(items)
        version_trends.append({
            "version": version,
            **summary,
        })

    # Sort by latest known execution time when available.
    version_trends.sort(
        key=lambda x: x["version"]
    )

    # Group by execution date.
    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in historical_records:
        dt = item["_execution_dt"]
        if dt:
            by_date[dt.date().isoformat()].append(item)

    daily_trends = []

    for day, items in sorted(by_date.items()):
        daily_trends.append({
            "date": day,
            **summarize(items),
        })

    return {
        "analysis": "Trend Analysis",
        "current_upload": current,
        "historical_overview": historical,
        "delta_vs_historical": {
            "pass_rate_points": delta(
                current["pass_rate"],
                historical["pass_rate"],
            ),
            "failure_rate_points": delta(
                current["failure_rate"],
                historical["failure_rate"],
            ),
            "test_count": current["total_tests"] - historical["total_tests"],
        },
        "by_version": version_trends,
        "by_date": daily_trends,
    }


# ---------------------------------------------------------------------------
# 4. Heatmap Generator
# ---------------------------------------------------------------------------

def generate_heatmap(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce matrix-friendly data:
        rows    = modules
        columns = PASSED / FAILED / SKIPPED / BLOCKED
        values  = execution counts

    The UI can render this directly as a heatmap.
    """
    statuses = ["PASSED", "FAILED", "SKIPPED", "BLOCKED"]

    matrix: Dict[str, Counter] = defaultdict(Counter)

    for record in records:
        module = record["module"] or "Unknown"
        status = record["status"] if record["status"] in statuses else "FAILED"
        matrix[module][status] += 1

    modules = sorted(matrix.keys())

    rows = []
    for module in modules:
        rows.append({
            "module": module,
            **{
                status: matrix[module].get(status, 0)
                for status in statuses
            },
        })

    return {
        "generator": "Heatmap Generator",
        "rows": modules,
        "columns": statuses,
        "matrix": rows,
    }


# ---------------------------------------------------------------------------
# 5. Quality Score
# ---------------------------------------------------------------------------

def compute_quality_score(
    summary: Dict[str, Any],
    flaky_count: int = 0,
) -> Dict[str, Any]:
    """
    Transparent quality score built from four weighted contributions:
        Pass Rate (50%) + Failure Rate (20%) + Flaky Tests (20%) + Stability (10%)

    Each contribution is capped by its own weight, so the total always falls
    between 0 and 100.
    """
    total_tests = summary.get("total_tests", 0) or 0
    pass_rate = summary.get("pass_rate", 0) or 0
    failure_rate = summary.get("failure_rate", 0) or 0
    skipped = summary.get("skipped", 0) or 0

    skipped_rate = round((skipped / total_tests) * 100, 2) if total_tests else 0
    flaky_rate = round((flaky_count / total_tests) * 100, 2) if total_tests else 0

    pass_rate_contribution = round(pass_rate * 0.50, 2)
    failure_rate_contribution = round((100 - failure_rate) * 0.20, 2)
    flaky_contribution = round((100 - flaky_rate) * 0.20, 2)
    stability_contribution = round((100 - skipped_rate) * 0.10, 2)

    score = round(
        pass_rate_contribution
        + failure_rate_contribution
        + flaky_contribution
        + stability_contribution,
        2,
    )
    score = max(0.0, min(100.0, score))

    return {
        "score": score,
        "pass_rate_contribution": pass_rate_contribution,
        "failure_rate_contribution": failure_rate_contribution,
        "flaky_contribution": flaky_contribution,
        "stability_contribution": stability_contribution,
        "flaky_rate": flaky_rate,
        "skipped_rate": skipped_rate,
        "explanation": (
            "Quality score is based on pass rate, failure rate and test stability."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Execution Duration Analytics
# ---------------------------------------------------------------------------

def duration_analytics(
    records: Sequence[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """Total/average duration, slowest/fastest tests and duration by module."""
    entries = []
    for record in records:
        duration = record.get("duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            entries.append({
                "test_id": record.get("test_id", ""),
                "test_name": record.get("test_name", "") or record.get("test_id", ""),
                "module": record.get("module", "") or "Unknown",
                "status": record.get("status", ""),
                "duration_seconds": round(float(duration), 3),
            })

    if not entries:
        return {
            "total_duration_seconds": 0,
            "average_duration_seconds": 0,
            "slowest_tests": [],
            "fastest_tests": [],
            "duration_by_module": [],
            "tests_with_duration": 0,
        }

    total_duration = sum(item["duration_seconds"] for item in entries)
    average_duration = round(total_duration / len(entries), 3)

    slowest = sorted(entries, key=lambda x: x["duration_seconds"], reverse=True)[:top_n]
    fastest = sorted(entries, key=lambda x: x["duration_seconds"])[:top_n]

    by_module: Dict[str, List[float]] = defaultdict(list)
    for item in entries:
        by_module[item["module"]].append(item["duration_seconds"])

    duration_by_module = [
        {
            "module": module,
            "total_duration_seconds": round(sum(values), 3),
            "average_duration_seconds": round(sum(values) / len(values), 3),
            "test_count": len(values),
        }
        for module, values in by_module.items()
    ]
    duration_by_module.sort(key=lambda x: x["total_duration_seconds"], reverse=True)

    return {
        "total_duration_seconds": round(total_duration, 3),
        "average_duration_seconds": average_duration,
        "slowest_tests": slowest,
        "fastest_tests": fastest,
        "duration_by_module": duration_by_module,
        "tests_with_duration": len(entries),
    }


# ---------------------------------------------------------------------------
# 7. Module-wise Analytics
# ---------------------------------------------------------------------------

def module_wise_analytics(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-module totals, status breakdown and pass/failure percentages."""
    modules: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "blocked": 0}
    )

    for record in records:
        module = record.get("module") or "Unknown"
        stats = modules[module]
        stats["total"] += 1
        status = record.get("status")
        if status == "PASSED":
            stats["passed"] += 1
        elif status == "FAILED":
            stats["failed"] += 1
        elif status == "SKIPPED":
            stats["skipped"] += 1
        elif status == "BLOCKED":
            stats["blocked"] += 1

    result = []
    for module, stats in modules.items():
        total = stats["total"]
        result.append({
            "module": module,
            **stats,
            "pass_rate": round(stats["passed"] / total * 100, 2) if total else 0,
            "failure_rate": round(stats["failed"] / total * 100, 2) if total else 0,
        })

    result.sort(key=lambda x: x["total"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# 8. Duplicate / Repeated Execution Analysis
# ---------------------------------------------------------------------------

def duplicate_execution_analysis(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Group records sharing the same duplicate_group (test_id/test_name/module)
    so repeated executions are reported for analytics without ever deleting
    them from MongoDB or the analytics dataset.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = record.get("duplicate_group") or record.get("test_id", "")
        groups[key].append(record)

    total_executions = len(records)
    unique_tests = len(groups)
    repeated_executions = total_executions - unique_tests

    rows = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        statuses_seen = sorted({item.get("status", "UNKNOWN") for item in items})
        rows.append({
            "test_id": items[0].get("test_id", ""),
            "test_name": items[0].get("test_name", ""),
            "module": items[0].get("module", ""),
            "execution_count": len(items),
            "duplicate_count": len(items) - 1,
            "statuses_seen": statuses_seen,
        })

    rows.sort(key=lambda x: x["execution_count"], reverse=True)

    return {
        "total_executions": total_executions,
        "unique_tests": unique_tests,
        "repeated_executions": repeated_executions,
        "repeated_test_groups": rows,
    }


# ---------------------------------------------------------------------------
# Overall analytics report
# ---------------------------------------------------------------------------

def build_analytics_report(
    current_records: Sequence[Dict[str, Any]],
    historical_records: Sequence[Dict[str, Any]],
    week_label: str,
    source_file: str = "",
    upload_batch_id: str = "",
) -> Dict[str, Any]:
    current = [
        normalize_record(record)
        for record in current_records
    ]

    historical = [
        normalize_record(record)
        for record in historical_records
    ]

    current_clean = [x for x in current if x["test_id"]]
    historical_clean = [x for x in historical if x["test_id"]]

    summary = {
        "total_tests": len(current_clean),
        "passed": sum(x["status"] == "PASSED" for x in current_clean),
        "failed": sum(x["status"] == "FAILED" for x in current_clean),
        "skipped": sum(x["status"] == "SKIPPED" for x in current_clean),
        "blocked": sum(x["status"] == "BLOCKED" for x in current_clean),
    }

    summary["pass_rate"] = round(
        (summary["passed"] / summary["total_tests"]) * 100,
        2,
    ) if summary["total_tests"] else 0

    summary["failure_rate"] = round(
        (summary["failed"] / summary["total_tests"]) * 100,
        2,
    ) if summary["total_tests"] else 0

    # Total executions vs unique tests: duplicate_group defaults to test_id
    # when the upload layer did not tag duplicates (older records).
    duplicate_groups = {
        x.get("duplicate_group") or x["test_id"] for x in current_clean
    }
    summary["unique_tests"] = len(duplicate_groups)
    summary["duplicate_executions"] = summary["total_tests"] - summary["unique_tests"]

    report = {
        "report_type": "QA Test Analytics Report",
        "report_id": f"analytics_result_{week_label}_{upload_batch_id}",
        "week": week_label,
        "source_file": source_file,
        "upload_batch_id": upload_batch_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,

        # Current-upload normalized records are embedded in the generated JSON.
        # Page 2 uses this self-contained dataset to populate every filter
        # option without re-querying MongoDB or exposing raw database data.
        "records": [
            {
                **{
                    key: value
                    for key, value in item.items()
                    if key != "_execution_dt"
                },
                # Business-level status routing used by Page 2.
                # FAILED intentionally belongs to both options.
                "status_filter": (
                    ["Complete Report", "Failed"]
                    if item.get("status") == "FAILED"
                    else ["Complete Report"]
                    if item.get("status") == "PASSED"
                    else []
                ),
            }
            for item in current_clean
        ],

        # Requested categorization
        "flaky_test_detector": detect_flaky_tests(historical_clean),
        "failure_pattern_detector": detect_failure_patterns(current_clean),
        "trend_analysis": trend_analysis(
            current_clean,
            historical_clean,
        ),
        "heatmap_generator": generate_heatmap(current_clean),
    }

    # Additional dashboard metrics, computed only from the current upload so
    # they always match the just-processed batch (e.g. 2498 records in, 2498
    # records reflected here). Flaky count is restricted to test ids present
    # in this batch so the score reflects this upload, not the whole history.
    current_test_ids = {item["test_id"] for item in current_clean if item.get("test_id")}
    flaky_in_batch = [
        item
        for item in report["flaky_test_detector"].get("tests", [])
        if item.get("test_id") in current_test_ids
    ]
    report["module_analytics"] = module_wise_analytics(current_clean)
    report["duration_analytics"] = duration_analytics(current_clean)
    report["quality_score"] = compute_quality_score(summary, flaky_count=len(flaky_in_batch))
    report["duplicate_analysis"] = duplicate_execution_analysis(current_clean)


    # Store distinct filter values in the JSON as well, so Page 2 can
    # populate its controls immediately from the generated analytics file.
    report["filter_options"] = {
        "test_id": sorted({item["test_id"] for item in current_clean if item.get("test_id")}),
        # Status is a business-level filter. Failed records are members of
        # both "Complete Report" and "Failed"; passed records are members of
        # "Complete Report" only.
        "status": [
            status
            for status in ("Complete Report", "Failed")
            if any(
                status in _dashboard_status_mapping(item.get("status"))
                for item in current_clean
            )
        ],
        "module": sorted({item["module"] for item in current_clean if item.get("module")}),
        "environment": sorted({item["environment"] for item in current_clean if item.get("environment")}),
        "build_version": sorted({item["version"] for item in current_clean if item.get("version")}),
        "date_time": sorted({
            item["execution_time"][:19]
            for item in current_clean
            if item.get("execution_time")
        }),
    }

    return json_safe(report)


# ---------------------------------------------------------------------------
# ChromaDB integration
# ---------------------------------------------------------------------------

def _deduplicate_chroma_records(metadatas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse chunk metadata into one logical test execution per Mongo record."""
    records: List[Dict[str, Any]] = []
    seen_source_ids = set()
    for metadata in metadatas or []:
        if not metadata:
            continue
        item = dict(metadata)
        source_id = item.get("mongo_record_id")
        if source_id:
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
        records.append(item)
    return records


def read_records_from_chroma(
    *,
    chroma_path: str,
    chroma_collection_name: str,
    upload_batch_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read logical test-execution records directly from ChromaDB metadata.

    ChromaDB is the post-ingestion source for Module 5/6. A single logical
    execution may have several vector chunks, so chunk metadata is collapsed
    by ``mongo_record_id`` before analytics are calculated.
    """
    collection = _get_chroma_collection(chroma_path, chroma_collection_name)

    if namespace:
        result = collection.get(
            where={"namespace": namespace},
            include=["metadatas"],
        )
    elif upload_batch_id:
        result = collection.get(
            where={"upload_batch_id": upload_batch_id},
            include=["metadatas"],
        )
    else:
        result = collection.get(include=["metadatas"])

    return _deduplicate_chroma_records(result.get("metadatas") or [])


def _chroma_where_for_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a Chroma metadata predicate from dashboard filter conditions.

    The UI is not changed. This function translates the existing dashboard
    filters into ChromaDB predicates where possible; final post-filtering is
    still applied because historical data can contain mixed timestamp formats.
    """
    filters = filters or {}
    clauses = []

    test_id = filters.get("test_id")
    module = filters.get("module")
    environment = filters.get("environment")
    build_version = filters.get("build_version")
    status = str(filters.get("status") or "").strip()

    if test_id and test_id != "All":
        clauses.append({"test_id": str(test_id)})
    if module and module != "All":
        clauses.append({"module": str(module)})
    if environment and environment != "All":
        clauses.append({"environment": str(environment)})
    if build_version and build_version != "All":
        clauses.append({"version": str(build_version)})

    if status == "Failed":
        clauses.append({"status": "FAILED"})
    elif status == "Complete Report":
        clauses.append({"status": {"$in": ["PASSED", "FAILED"]}})

    # Chroma supports $and for multiple metadata predicates.
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _dashboard_status_mapping(status: Any) -> List[str]:
    """Map a raw execution status to the business-level Page 2 filters.

    Business rule:
      - PASSED -> Complete Report
      - FAILED -> Complete Report + Failed
      - anything else -> neither option
    """
    normalized = str(status or "").strip().upper()
    if normalized in {"PASSED", "PASS", "SUCCESS", "SUCCESSFUL", "OK", "TRUE"}:
        return ["Complete Report"]
    if normalized in {"FAILED", "FAIL", "FAILURE", "ERROR", "FALSE"}:
        return ["Complete Report", "Failed"]
    return []


def _post_filter_dashboard_records(
    records: Sequence[Dict[str, Any]],
    filters: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply exact Page 2 business-level status semantics after Chroma retrieval."""
    filters = filters or {}
    status = str(filters.get("status") or "").strip()
    module = str(filters.get("module") or "").strip()
    environment = str(filters.get("environment") or "").strip()
    build_version = str(filters.get("build_version") or "").strip()
    test_id = str(filters.get("test_id") or "").strip()

    from_dt = filters.get("from_datetime")
    to_dt = filters.get("to_datetime")

    def value(record, key):
        return str(record.get(key, "") if record.get(key) is not None else "").strip()

    result = []
    for record in records:
        # ChromaDB may contain the source report field names (for example
        # test_case_id/module_name/execution_date_and_time) because the upload
        # pipeline intentionally preserves the seven approved business
        # columns. Normalize only for filtering so Page 2 filters match the
        # uploaded report data exactly.
        normalized = normalize_record(dict(record))
        record_status = str(normalized.get("status") or value(record, "status")).upper()
        status_mapping = _dashboard_status_mapping(record_status)

        # Page 2 business rule:
        #   Complete Report = ALL passed + ALL failed
        #   Failed         = failed only
        if status and status in {"Complete Report", "Failed"}:
            if status not in status_mapping:
                continue
        if module and module != "All" and str(normalized.get("module") or value(record, "module")) != module:
            continue
        if environment and environment != "All" and str(normalized.get("environment") or value(record, "environment")) != environment:
            continue
        if build_version and build_version != "All" and str(normalized.get("version") or value(record, "version")) != build_version:
            continue
        if test_id and test_id != "All" and str(normalized.get("test_id") or value(record, "test_id")) != test_id:
            continue

        if from_dt is not None or to_dt is not None:
            dt = normalized.get("_execution_dt") or parse_datetime(record.get("execution_time"))
            if dt is None:
                for key in ("execution_date_and_time", "execution_date", "Execution Date and Time", "executed_at", "timestamp", "execution_timestamp", "run_time", "date", "datetime"):
                    dt = parse_datetime(record.get(key))
                    if dt is not None:
                        break
            if dt is not None:
                if from_dt is not None and dt < from_dt:
                    continue
                if to_dt is not None and dt > to_dt:
                    continue
            # Preserve the existing dashboard behavior for unparseable dates:
            # do not silently remove a record merely because its timestamp is
            # unavailable.

        enriched_record = dict(record)
        enriched_record["status_filter"] = status_mapping
        result.append(enriched_record)

    return result


def retrieve_chroma_rag_context(
    *,
    chroma_path: str,
    chroma_collection_name: str,
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> Dict[str, Any]:
    """Module 6 RAG retrieval from ChromaDB.

    Retrieves the most relevant chunks plus their embeddings, metadata,
    distances and source IDs. The filter predicate narrows the vector search
    to the dashboard selection before semantic ranking.
    """
    collection = _get_chroma_collection(chroma_path, chroma_collection_name)
    total = collection.count()

    if not total:
        return {
            "query": query_text,
            "chunks": [],
            "logical_records": [],
            "retrieved_count": 0,
        }

    kwargs = {
        "query_texts": [query_text or "test execution quality failures trends"],
        "n_results": min(max(1, int(top_k)), total),
        "include": ["documents", "metadatas", "distances", "embeddings"],
    }
    where = _chroma_where_for_filters(filters)
    if where:
        kwargs["where"] = where

    try:
        result = collection.query(**kwargs)
    except Exception:
        # Metadata predicates can fail on older/mixed Chroma collections. The
        # exact dashboard semantics are still enforced by post-filtering.
        kwargs.pop("where", None)
        result = collection.query(**kwargs)

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    embeddings = (result.get("embeddings") or [[]])[0]

    chunks = []
    for index, document in enumerate(documents):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        chunks.append({
            "document": document,
            "metadata": metadata,
            "distance": distances[index] if index < len(distances) else None,
            "embedding": embeddings[index] if index < len(embeddings) else None,
            "mongo_record_id": metadata.get("mongo_record_id"),
            "chunk_index": metadata.get("chunk_index"),
        })

    logical_records = _post_filter_dashboard_records(
        _deduplicate_chroma_records(metadatas),
        filters,
    )

    # If a Chroma-side predicate was used, ensure returned chunks belong to
    # records that survive the exact post-filter.
    allowed_ids = {
        str(item.get("mongo_record_id"))
        for item in logical_records
        if item.get("mongo_record_id") is not None
    }
    filtered_chunks = [
        chunk for chunk in chunks
        if not allowed_ids or str(chunk.get("mongo_record_id")) in allowed_ids
    ]

    return {
        "query": query_text,
        "chunks": filtered_chunks,
        "logical_records": logical_records,
        "retrieved_count": len(filtered_chunks),
        "embedding_dimension": (
            len(filtered_chunks[0]["embedding"])
            if filtered_chunks and filtered_chunks[0].get("embedding") is not None
            else 0
        ),
    }


def query_analytics_from_chroma(
    *,
    chroma_path: str,
    chroma_collection_name: str,
    filters: Optional[Dict[str, Any]] = None,
    rag_query: str = "",
    rag_top_k: int = DEFAULT_RAG_TOP_K,
) -> Dict[str, Any]:
    """Module 5 analytics + Module 6 RAG orchestration.

    1. Read all logical executions from ChromaDB metadata.
    2. Apply dashboard filters to the Chroma-derived records.
    3. Recompute the four requested analytics categories from that selection.
    4. Run semantic RAG retrieval against the same filtered context.
    """
    all_records = read_records_from_chroma(
        chroma_path=chroma_path,
        chroma_collection_name=chroma_collection_name,
    )
    filtered_records = _post_filter_dashboard_records(all_records, filters)

    current_batch_ids = {
        str(item.get("upload_batch_id"))
        for item in filtered_records
        if item.get("upload_batch_id")
    }
    # Trend/flaky analysis needs the full Chroma history, while the displayed
    # current selection remains filter-scoped.
    historical_records = all_records

    normalized_current = [normalize_record(item) for item in filtered_records]
    normalized_history = [normalize_record(item) for item in historical_records]
    normalized_current = [item for item in normalized_current if item.get("test_id")]
    normalized_history = [item for item in normalized_history if item.get("test_id")]

    flaky = detect_flaky_tests(normalized_history)
    patterns = detect_failure_patterns(normalized_current)
    trends = trend_analysis(normalized_current, normalized_history)
    heatmap = generate_heatmap(normalized_current)

    query = rag_query or (
        "test quality failures flaky tests recurring error patterns "
        "regressions module risks and recommended actions"
    )
    rag = retrieve_chroma_rag_context(
        chroma_path=chroma_path,
        chroma_collection_name=chroma_collection_name,
        query_text=query,
        filters=filters,
        top_k=rag_top_k,
    )

    return {
        "records": filtered_records,
        "analytics": {
            "flaky_test_detector": flaky,
            "failure_pattern_detector": patterns,
            "trend_analysis": trends,
            "heatmap_generator": heatmap,
        },
        "rag": rag,
        "source": {
            "analytics_source": "ChromaDB",
            "vector_retrieval": True,
            "metadata_records_total": len(all_records),
            "metadata_records_filtered": len(filtered_records),
            "historical_records": len(historical_records),
            "batch_ids_in_selection": sorted(current_batch_ids),
        },
    }


def validate_chroma_batch(
    *,
    chroma_path: str,
    chroma_collection_name: str,
    upload_batch_id: str,
) -> int:
    """Verify that the current upload exists in ChromaDB."""
    records = read_records_from_chroma(
        chroma_path=chroma_path,
        chroma_collection_name=chroma_collection_name,
        namespace=namespace,
        upload_batch_id=upload_batch_id,
    )
    if not records:
        raise ValueError(
            "Analytics Engine could not find the uploaded batch in ChromaDB. "
            f"upload_batch_id={upload_batch_id}"
        )
    return len(records)


# ---------------------------------------------------------------------------
# Main integration function
# ---------------------------------------------------------------------------

def generate_analytics_after_chroma(
    *,
    week_label: str,
    upload_batch_id: str,
    source_file: str,
    namespace: Optional[str] = None,
    mongo_uri: str = DEFAULT_MONGO_URI,
    db_name: str = DEFAULT_DB_NAME,
    mongo_collection_name: str = DEFAULT_MONGO_COLLECTION,
    chroma_path: str = "./chroma_db",
    chroma_collection_name: str = "test_execution_history",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Generate Module 5 analytics from ChromaDB after ingestion.

    No pre-Chroma stage is modified. Once vectors are persisted, ChromaDB
    becomes the Analytics Engine's sole retrieval and persistence source;
    reports are rebuilt from ChromaDB on demand and never written to MongoDB.
    """
    current_chroma_records = read_records_from_chroma(
        chroma_path=chroma_path,
        chroma_collection_name=chroma_collection_name,
        namespace=namespace,
        upload_batch_id=upload_batch_id,
    )
    if not current_chroma_records:
        raise ValueError(
            "Analytics Engine could not find the uploaded batch in ChromaDB. "
            f"upload_batch_id={upload_batch_id}"
        )

    historical_chroma_records = read_records_from_chroma(
        chroma_path=chroma_path,
        chroma_collection_name=chroma_collection_name,
    )

    report = build_analytics_report(
        current_records=current_chroma_records,
        historical_records=historical_chroma_records,
        week_label=week_label,
        source_file=source_file,
        upload_batch_id=upload_batch_id,
    )

    report["analytics_source"] = {
        "structured_source": "ChromaDB metadata",
        "vector_store": "ChromaDB",
        "current_upload_chromadb_records": len(current_chroma_records),
        "historical_chromadb_records": len(historical_chroma_records),
        "analytics_engine_retrieval": True,
    }

    chroma_collection = _get_chroma_collection(chroma_path, chroma_collection_name)
    report["chroma_db"] = {
        "collection": chroma_collection_name,
        "path": chroma_path,
        "current_upload_records": len(current_chroma_records),
        "total_records": chroma_collection.count(),
    }

    return {
        "report": report,
        "json_path": None,
        "report_id": report["report_id"],
        "current_records": len(current_chroma_records),
        "historical_records": len(historical_chroma_records),
        "chroma_total": report["chroma_db"]["total_records"],
    }


# ---------------------------------------------------------------------------
# Module 6 - RAG + Azure OpenAI AI Insights
# ---------------------------------------------------------------------------

def _azure_openai_client():
    """Create an Azure OpenAI client from environment configuration."""
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for Azure OpenAI AI Insights."
        ) from exc

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _ai_error_intelligence(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate normalized error metadata for the AI Insights layer."""
    failures = [r for r in records if normalize_status(first_value(r, FIELD_ALIASES["status"])) == "FAILED"]
    groups: Dict[str, Dict[str, Any]] = {}
    category_counts = Counter()
    secondary_counts = Counter()
    for record in failures:
        message = safe_string(first_value(record, ["error_message", "error", "failure_message", "failure_reason", "message"]))
        category = safe_string(record.get("error_category", "")) or "OTHER"
        signature = safe_string(record.get("error_signature", "")) or normalize_failure_message(message)
        secondary = record.get("secondary_errors", [])
        if isinstance(secondary, str):
            secondary = [secondary] if secondary else []
        category_counts[category] += 1
        for item in secondary:
            secondary_counts[str(item)] += 1
        key = signature or message or "Unknown Failure"
        entry = groups.setdefault(key, {
            "signature": key, "error_message": message or key, "category": category,
            "occurrences": 0, "affected_tests": set(), "affected_modules": set(),
            "secondary_errors": set(),
        })
        entry["occurrences"] += 1
        test_id = safe_string(first_value(record, FIELD_ALIASES["test_id"]))
        module = safe_string(first_value(record, FIELD_ALIASES["module"])) or "Unknown"
        if test_id: entry["affected_tests"].add(test_id)
        entry["affected_modules"].add(module)
        entry["secondary_errors"].update(str(x) for x in secondary if x)

    top_errors = []
    for item in sorted(groups.values(), key=lambda x: x["occurrences"], reverse=True)[:8]:
        top_errors.append({
            "signature": item["signature"],
            "error_message": item["error_message"],
            "category": item["category"],
            "occurrences": item["occurrences"],
            "affected_tests": sorted(item["affected_tests"])[:12],
            "affected_modules": sorted(item["affected_modules"])[:12],
            "secondary_errors": sorted(item["secondary_errors"])[:8],
        })
    return {
        "total_failures": len(failures),
        "category_counts": dict(category_counts.most_common()),
        "secondary_error_counts": dict(secondary_counts.most_common(10)),
        "top_errors": top_errors,
    }


def _build_weekly_quality_digest(
    *, records: Sequence[Dict[str, Any]], errors: Dict[str, Any],
    hotspots: Sequence[Dict[str, Any]], patterns: Sequence[Dict[str, Any]],
    flaky: Sequence[Dict[str, Any]], trends: Dict[str, Any],
    recommendations: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a concise, evidence-backed digest for the selected week."""
    rows = list(records or [])
    dates = [
        parse_datetime(first_value(row, FIELD_ALIASES["execution_time"]))
        for row in rows
    ]
    dates = [value.date().isoformat() for value in dates if value]
    period = {
        "start": min(dates) if dates else None,
        "end": max(dates) if dates else None,
        "label": f"{min(dates)} to {max(dates)}" if dates else "Selected week",
    }

    metrics = {
        "total": len(rows),
        "failed": errors.get("total_failures", 0),
        "failure_rate": round((errors.get("total_failures", 0) / len(rows)) * 100, 1) if rows else 0.0,
    }
    findings = []
    if metrics["failed"]:
        top_error = (errors.get("top_errors") or [])[0] if errors.get("top_errors") else None
        if top_error:
            findings.append({
                "signal": "Dominant failure pattern",
                "evidence": f"{top_error['occurrences']} occurrence(s) of {top_error['signature']} across {len(top_error['affected_tests'])} test(s).",
            })
    if hotspots:
        hotspot = hotspots[0]
        findings.append({
            "signal": "Highest-risk module",
            "evidence": f"{hotspot['module']} accounts for {hotspot['failures']} failures ({hotspot['failure_density']}% of failures).",
        })
    if flaky:
        test = flaky[0]
        findings.append({
            "signal": "Flakiness is affecting confidence",
            "evidence": f"{test.get('test_name') or test.get('test_id')} failed in {float(test.get('failure_rate', 0)) * 100:.1f}% of {test.get('total_executions', test.get('executions', 0))} execution(s).",
        })
    delta = (trends.get("delta_vs_historical") or {}).get("failure_rate_points")
    if delta is not None and abs(float(delta)) > 2:
        direction = "up" if float(delta) > 0 else "down"
        findings.append({
            "signal": "Weekly trend",
            "evidence": f"Failure rate is {direction} {abs(float(delta)):.1f} percentage points versus historical data.",
        })
    if not findings:
        findings.append({"signal": "No material quality risk detected", "evidence": "The selected week contains no failed executions or significant risk signals."})

    remediation = []
    for recommendation in recommendations:
        remediation.append({
            "priority": recommendation.get("priority", "MEDIUM"),
            "action": recommendation.get("action", "Review the reported failures and verify the fix in the next run."),
            "evidence": recommendation.get("evidence", "Selected execution data"),
        })
    if not remediation and metrics["failed"] == 0:
        remediation.append({
            "priority": "LOW",
            "action": "Keep the current suite configuration and monitor the next weekly run for regressions.",
            "evidence": "No failed executions in the selected week.",
        })

    return {
        "period": period,
        "metrics": metrics,
        "headline": f"{metrics['failed']} of {metrics['total']} tests failed ({metrics['failure_rate']:.1f}%) during {period['label']}.",
        "findings": findings[:4],
        "remediation": remediation[:3],
    }


def _build_ai_insights_payload(
    *, records: Sequence[Dict[str, Any]], analytics_categories: Dict[str, Any],
    rag_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a deterministic, structured AI Insights payload before LLM enrichment."""
    rows = list(records or [])
    total = len(rows)
    passed = sum(1 for r in rows if normalize_status(first_value(r, FIELD_ALIASES["status"])) == "PASSED")
    failed = sum(1 for r in rows if normalize_status(first_value(r, FIELD_ALIASES["status"])) == "FAILED")
    skipped = sum(1 for r in rows if normalize_status(first_value(r, FIELD_ALIASES["status"])) == "SKIPPED")
    failure_rate = round((failed / total) * 100, 1) if total else 0.0
    pass_rate = round((passed / total) * 100, 1) if total else 0.0

    modules = Counter()
    failed_modules = Counter()
    for r in rows:
        module = safe_string(first_value(r, FIELD_ALIASES["module"])) or "Unknown"
        modules[module] += 1
        if normalize_status(first_value(r, FIELD_ALIASES["status"])) == "FAILED": failed_modules[module] += 1
    hotspots = [
        {"module": m, "failures": c, "failure_density": round(c / failed * 100, 1) if failed else 0.0}
        for m, c in failed_modules.most_common(8)
    ]

    flaky = (analytics_categories.get("flaky_test_detector") or {}).get("tests", []) or []
    patterns = (analytics_categories.get("failure_pattern_detector") or {}).get("patterns", []) or []
    trends = analytics_categories.get("trend_analysis") or {}
    quality = analytics_categories.get("quality_score") or compute_quality_score(
        {"total_tests": total, "pass_rate": pass_rate, "failure_rate": failure_rate, "skipped": skipped},
        flaky_count=len(flaky),
    )
    errors = _ai_error_intelligence(rows)

    recommendations = []
    if failed:
        if errors["category_counts"].get("AUTHENTICATION", 0):
            recommendations.append({"priority": "HIGH", "title": "Investigate authentication failures", "action": "Verify token generation, expiry and test-environment credentials.", "evidence": f"{errors['category_counts']['AUTHENTICATION']} authentication failure(s)."})
        if errors["category_counts"].get("AUTHORIZATION", 0):
            recommendations.append({"priority": "HIGH", "title": "Investigate authorization failures", "action": "Verify service permissions, roles and endpoint access for the test identity.", "evidence": f"{errors['category_counts']['AUTHORIZATION']} authorization failure(s)."})
        if errors["category_counts"].get("TIMEOUT", 0):
            recommendations.append({"priority": "HIGH", "title": "Reduce timeout failures", "action": "Investigate dependent-service latency and replace fixed waits with readiness conditions.", "evidence": f"{errors['category_counts']['TIMEOUT']} timeout failure(s)."})
        if errors["category_counts"].get("ELEMENT_NOT_FOUND", 0):
            recommendations.append({"priority": "MEDIUM", "title": "Improve element stability", "action": "Use stable test IDs and explicit element visibility/readiness checks.", "evidence": f"{errors['category_counts']['ELEMENT_NOT_FOUND']} element-not-found failure(s)."})
        if errors["category_counts"].get("RESOURCE_NOT_FOUND", 0):
            recommendations.append({"priority": "MEDIUM", "title": "Investigate missing resources", "action": "Verify dependent URLs/endpoints and environment data before test execution.", "evidence": f"{errors['category_counts']['RESOURCE_NOT_FOUND']} resource-not-found failure(s)."})
    if flaky:
        top = flaky[0]
        recommendations.append({"priority": "HIGH", "title": "Stabilize the most flaky test", "action": "Investigate synchronization, environment and external-service dependencies before trusting the result.", "evidence": f"{top.get('test_name') or top.get('test_id')} has a {round(float(top.get('failure_rate', 0))*100, 1)}% failure rate."})
    if hotspots:
        h = hotspots[0]
        recommendations.append({"priority": "MEDIUM", "title": f"Review {h['module']} hotspot", "action": "Review recent changes and add regression coverage around the dominant failing scenarios.", "evidence": f"{h['failures']} failures ({h['failure_density']}% of failures)."})
    delta = (trends.get("delta_vs_historical") or {}).get("failure_rate_points")
    if delta is not None and float(delta) > 2:
        recommendations.append({"priority": "HIGH", "title": "Investigate rising failure trend", "action": "Compare the latest build with the last stable build and inspect newly introduced failure signatures.", "evidence": f"Failure rate is up {float(delta):.1f} percentage points versus historical data."})
    recommendations = recommendations[:8]
    weekly_digest = _build_weekly_quality_digest(
        records=rows,
        errors=errors,
        hotspots=hotspots,
        patterns=patterns,
        flaky=flaky,
        trends=trends,
        recommendations=recommendations,
    )

    return {
        "metrics": {"total_tests": total, "passed": passed, "failed": failed, "skipped": skipped, "pass_rate": pass_rate, "failure_rate": failure_rate},
        "quality": quality,
        "error_intelligence": errors,
        "flaky_tests": flaky[:8],
        "failure_patterns": patterns[:8],
        "trend_analysis": trends,
        "module_hotspots": hotspots,
        "recommendations": recommendations,
        "weekly_quality_digest": weekly_digest,
        "rag_evidence": [
            {"document": x.get("document", ""), "metadata": x.get("metadata", {}), "distance": x.get("distance")}
            for x in ((rag_context or {}).get("chunks", []) or [])[:DEFAULT_RAG_TOP_K]
        ],
    }


def generate_ai_insights_with_rag(
    *,
    chroma_path: str,
    chroma_collection_name: str,
    filters: Optional[Dict[str, Any]],
    analytic_categories: Dict[str, Any],
    rag_context: Dict[str, Any],
    records: Optional[Sequence[Dict[str, Any]]] = None,
    deployment: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate structured AI Insights from the exact filtered selection + RAG history.

    The deterministic payload is always produced first. Azure OpenAI is an optional
    enrichment layer; if unavailable/failing, the structured fallback remains valid.
    """
    selected_records = list(records or [])
    base = _build_ai_insights_payload(
        records=selected_records,
        analytics_categories=analytic_categories or {},
        rag_context=rag_context or {},
    )

    deployment_name = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if not (deployment_name and endpoint and api_key):
        return {"provider": "Deterministic analytics fallback", "filters": filters or {}, "insights": base, "llm_used": False}

    try:
        client = _azure_openai_client()
        prompt = {
            "filters": filters or {},
            "structured_analytics": base,
            "retrieved_historical_evidence": base["rag_evidence"],
        }
        system_prompt = (
            "You are a senior QA reliability engineer. Improve the supplied structured AI Insights "
            "without inventing facts. Return JSON only with keys: executive_summary, risk_level, "
            "root_causes, recommendations, key_takeaways. root_causes must contain title, category, "
            "evidence, hypothesis, confidence. recommendations must contain priority, title, action, "
            "evidence, expected_impact. Distinguish evidence from hypotheses. Use only supplied data."
        )
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
            ],
        )
        text = response.choices[0].message.content or "{}"
        llm = json.loads(text)
        base["llm"] = llm
        if llm.get("recommendations"):
            base["recommendations"] = llm["recommendations"][:8]
        return {"provider": "Azure OpenAI", "deployment": deployment_name, "filters": filters or {}, "insights": base, "llm_used": True}
    except Exception as exc:
        base["llm_error"] = str(exc)
        return {"provider": "Deterministic analytics fallback", "filters": filters or {}, "insights": base, "llm_used": False}


# ---------------------------------------------------------------------------
# Optional command-line usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate QA analytics report from MongoDB."
    )
    parser.add_argument("--week", required=True, help="Example: week1")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-file", default="")
    parser.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    parser.add_argument("--db", default=DEFAULT_DB_NAME)
    parser.add_argument("--collection", default=DEFAULT_MONGO_COLLECTION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    result = generate_analytics_after_chroma(
        week_label=args.week,
        upload_batch_id=args.batch_id,
        source_file=args.source_file,
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        mongo_collection_name=args.collection,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("Analytics Engine completed")
    print("Report ID:", result["report_id"])
    print("Current upload records:", result["current_records"])
    print("Historical records:", result["historical_records"])
    print("Report persisted to MongoDB; no JSON report file was generated.")
    print("=" * 60)
