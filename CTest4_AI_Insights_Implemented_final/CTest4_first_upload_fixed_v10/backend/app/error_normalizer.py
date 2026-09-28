"""Deterministic error extraction, cleaning, categorization and signature generation.

The normalizer deliberately does not use an LLM: report ingestion must be
stable, fast, testable and safe before data reaches MongoDB/ChromaDB.
"""
from __future__ import annotations

import re
from html import unescape
from typing import Any, Dict, List


HTTP_ERROR_RE = re.compile(
    r"\b(401|403|404|408|409|429|500|502|503|504)\b\s*(?:[-:]?\s*"
    r"(Unauthorized|Forbidden|Not Found|Request Timeout|Conflict|Too Many Requests|"
    r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)\b"
    r"|(?:\([^)]*(?:Unauthorized|Forbidden|Not Found|Request Timeout|Conflict|Too Many Requests|"
    r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)[^)]*\)))",
    re.IGNORECASE,
)

STACK_FRAME_RE = re.compile(r"^\s*at\s+.+$", re.IGNORECASE)
NODE_INTERNAL_RE = re.compile(r"^(?:node:|internal/|node_modules/)", re.IGNORECASE)
PATH_FRAME_RE = re.compile(r"^\s*(?:at\s+)?(?:[A-Za-z]:[\\/]|/|file://).+$", re.IGNORECASE)
BROWSER_NOISE_RE = re.compile(
    r"^(?:\[?BROWSER CONSOLE ERROR\]?\s*:?)|(?:browser console error\s*:?)$",
    re.IGNORECASE,
)
BROWSER_RESOURCE_RE = re.compile(
    r"^(?:failed to load resource|the server responded with a status of)\b",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?|"
    r"\d{1,2}/\d{1,2}/\d{2,4}[ T]\d{1,2}:\d{2}(?::\d{2})?)\b"
)
EMPTY_ERROR_MARKERS = {"", "-", "—", "n/a", "na", "none", "null"}


def _text(value: Any) -> str:
    text = (
        str(value or "")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    return "" if text.casefold() in EMPTY_ERROR_MARKERS else text


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_secondary_errors(raw_error: Any) -> List[str]:
    """Find important HTTP/browser errors anywhere in the raw report text."""
    text = _text(raw_error)
    if not text:
        return []
    found = []
    for match in HTTP_ERROR_RE.finditer(text):
        status = match.group(1)
        label = match.group(2)
        if not label:
            label_match = re.search(r"\(([^)]*(?:Unauthorized|Forbidden|Not Found|Request Timeout|Conflict|Too Many Requests|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)[^)]*)\)", match.group(0), re.IGNORECASE)
            label = label_match.group(1) if label_match else ""
        if label:
            found.append(f"{status} {label.title()}")
    return _dedupe(found)


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if STACK_FRAME_RE.match(stripped):
        return True
    if PATH_FRAME_RE.match(stripped):
        return True
    if NODE_INTERNAL_RE.match(stripped):
        return True
    if BROWSER_NOISE_RE.match(stripped):
        return True
    if BROWSER_RESOURCE_RE.match(stripped) and not re.search(r"\b(?:401|403|404|408|409|429|500|502|503|504)\b", stripped):
        return True
    if stripped.lower().startswith(("execution date and time:", "browser console:")):
        return True
    return False


def _strip_inline_stack(text: str) -> str:
    # Common Cucumber/Node errors can have stack frames on the same line.
    text = re.sub(r"\s+at\s+(?:[A-Za-z_$][\w$<>.]*)?\s*\([^\n]*\)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+at\s+(?:[A-Za-z_$][\w$<>.]*)?\s*(?:\([^\n]*\))?$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _clean_markup_and_symbols(text: str) -> str:
    """Remove markup and non-readable symbols while preserving useful punctuation."""
    text = unescape(text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b\ufeff]", " ", text)
    text = re.sub(r"[^\w\s.,:;!?()\[\]{}'\"/_#@%$=-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _clean_candidate(line: str) -> str:
    value = _clean_markup_and_symbols(_strip_inline_stack(line.strip()))
    # Remove source prefixes while retaining the actual message.
    value = re.sub(r"^(?:AssertionError\s*\[[^\]]+\]\s*:\s*)", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(?:AssertionError|TimeoutError|TypeError|ReferenceError|ValueError|"
        r"SyntaxError|HTTPError|NotFoundError|Error)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Error)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = TIMESTAMP_RE.sub("", value)
    value = re.sub(r"\b(?:line| at line)\s+\d+\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\(?(?:line|column)\s*[:=]?\s*\d+\)?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bat\s*[,.;]?\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+([,.;])", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip(" \t-:")
    return value



def _strip_paths(text: str) -> str:
    # Remove filesystem paths that sometimes leak into exception messages.
    text = re.sub(r"(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"]+[\\/])+[^\s'\"]*", "", text)
    text = re.sub(r"\b(?:node_modules|dist|build|src|features|steps)[\\/][^\s]+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def _candidate_score(text: str) -> int:
    low = text.lower()
    score = 0
    if re.search(r"\btimed?\s*out\b|timeout", low): score += 80
    if re.search(r"assertion|expected .*received|to equal|to be|mismatch|product count mismatch|page title mismatch", low): score += 90
    if re.search(r"strict mode violation|locator\.|element|selector|not found|no such element", low): score += 70
    if re.search(r"401|unauthorized|authentication|login|token", low): score += 65
    if re.search(r"403|forbidden|authorization|permission", low): score += 65
    if re.search(r"404|not found|resource", low): score += 65
    if re.search(r"\b(?:type|reference|value|syntax)error\b|cannot read properties|undefined", low): score += 60
    if low.startswith(("error", "failed", "exception")): score += 20
    if len(text) < 500: score += 10
    return score


def extract_primary_error(raw_error: Any, secondary_errors: List[str] | None = None) -> str:
    """Extract the most useful human-readable failure while dropping noise."""
    text = _text(raw_error)
    if not text:
        return ""

    secondary_errors = secondary_errors or extract_secondary_errors(text)
    candidates = []
    for line in text.split("\n"):
        if _is_noise_line(line):
            continue
        candidate = _clean_candidate(line)
        if not candidate:
            continue
        # Drop common report boilerplate, but retain meaningful HTTP errors.
        if BROWSER_RESOURCE_RE.match(candidate) and not re.search(r"\b(?:401|403|404)\b", candidate):
            continue
        candidates.append(candidate)

    if candidates:
        candidates = _dedupe(candidates)

        # Prefer the assertion headline when a multi-line mismatch includes
        # additional "Expected/Received" detail lines underneath it.
        headline = next(
            (candidate for candidate in candidates if re.search(r"mismatch|assertion|expected .*received|to equal|to be|failed", candidate, flags=re.IGNORECASE)),
            None,
        )
        if headline is not None:
            primary = headline
        else:
            scored = sorted(enumerate(candidates), key=lambda pair: (-_candidate_score(pair[1]), len(pair[1]), pair[0]))
            primary = scored[0][1]
    else:
        primary = ""

    # If the only useful signal was an HTTP response, retain it as the primary error.
    if not primary and secondary_errors:
        primary = secondary_errors[0]

    # Never let a primary error carry a duplicate browser prefix or stack suffix.
    primary = _strip_paths(primary)
    primary = re.sub(r"\s+", " ", primary).strip()
    return primary


def categorize_error(error_message: Any, secondary_errors: List[str] | None = None) -> str:
    """Categorize the primary failure first; use secondary HTTP errors as fallback signal."""
    primary = _text(error_message).lower()
    secondary = " ".join(secondary_errors or []).lower()

    if re.search(r"timeout|timed out|time out|exceeded\s+\d+(?:\.\d+)?\s*ms", primary):
        return "TIMEOUT"
    if re.search(
        r"assertionerror|assertion failure|expected .*received|to equal|to be|"
        r"mismatch|expect\s*\(\s*locator|tohave(?:text|value)|toequal|tobe",
        primary,
    ):
        return "ASSERTION_FAILURE"
    if re.search(r"strict mode violation|element not found|no such element|locator.*resolved to 0|unable to find|cannot find.*element", primary):
        return "ELEMENT_NOT_FOUND"
    if re.search(r"\b401\b|unauthorized|authentication|invalid token|login required", primary):
        return "AUTHENTICATION"
    if re.search(r"\b403\b|forbidden|authorization|permission denied|access denied", primary):
        return "AUTHORIZATION"
    if re.search(r"\b404\b|not found|no such resource", primary):
        return "RESOURCE_NOT_FOUND"
    if re.search(r"\b(?:500|502|503|504)\b|internal server error|bad gateway|service unavailable|gateway timeout", primary):
        return "SERVER_ERROR"

    if re.search(r"\b401\b|unauthorized|authentication", secondary):
        return "AUTHENTICATION"
    if re.search(r"\b403\b|forbidden|authorization", secondary):
        return "AUTHORIZATION"
    if re.search(r"\b404\b|not found", secondary):
        return "RESOURCE_NOT_FOUND"
    if re.search(r"\b(?:500|502|503|504)\b|internal server error|bad gateway|service unavailable|gateway timeout", secondary):
        return "SERVER_ERROR"
    return "OTHER"


def _shorten_primary_error(primary: str, category: str) -> str:
    """Use stable, concise wording for framework-generated failures."""
    low = primary.casefold()
    if category == "TIMEOUT":
        if re.search(
            r"(?:locator|page\.|getby[a-z]+\(|waitfor|click|fill|press)"
            r".*(?:timeout|timed out|exceeded\s+\d+(?:\.\d+)?\s*ms)",
            low,
        ):
            return "Element interaction timed out"
    if category == "ELEMENT_NOT_FOUND":
        return "Element not found"
    if category == "ASSERTION_FAILURE" and re.search(
        r"expect\s*\(\s*locator|tohave(?:text|value)|toequal|tobe",
        low,
    ):
        return "Expected value did not match actual value"
    if category == "SERVER_ERROR" and re.fullmatch(
        r"(?:error\s*:\s*)?(?:500\s+)?internal server error", low
    ):
        return "Internal server error"
    return primary


def normalize_error_signature(error_message: Any) -> str:
    """Normalize dynamic identifiers/values without changing the primary meaning."""
    text = _text(error_message)
    if not text:
        return ""

    # URLs first so their numeric path/query values are not normalized separately.
    text = re.sub(r"\b(?:https?|ftp)://[^\s)\]>]+", "URL", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfile://[^\s)\]>]+", "FILE", text, flags=re.IGNORECASE)
    # ISO timestamps and common date/time forms.
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b", "TIMESTAMP", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\b", "TIMESTAMP", text)
    # UUIDs and common prefixed execution/test IDs.
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "ID", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:RUN|TC|TEST|CASE|ID)[-_][A-Za-z0-9_-]+\b", "ID", text, flags=re.IGNORECASE)
    # Hex addresses / hashes.
    text = re.sub(r"\b0x[0-9a-f]+\b", "HEX", text, flags=re.IGNORECASE)
    # Standalone numeric values, including milliseconds/counts/status codes.
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Signatures are punctuation-light to make semantically identical failures match.
    text = re.sub(r"[,:;]+", "", text)
    text = re.sub(r"\s*([()\[\]{}])\s*", r"\1", text)
    text = re.sub(r"[.]+$", "", text)
    return text


def normalize_error_summary(raw_error: Any) -> Dict[str, str]:
    """Return the dashboard/storage contract: a consistent JSON-like summary.

    Contract:
    - recognized error -> {"category": <category>, "signature": <signature>}
    - no error -> {"category": "None", "signature": "No errors detected"}
    - uncertain -> {"category": "Unknown", "signature": "Unrecognized error"}
    """
    raw = _text(raw_error)
    if not raw:
        return {"category": "None", "signature": "No errors detected"}

    secondary = extract_secondary_errors(raw)
    primary = extract_primary_error(raw, secondary)
    category = categorize_error(primary, secondary)
    primary = _shorten_primary_error(primary, category)
    signature = normalize_error_signature(primary).strip()

    if not primary or not signature:
        if not primary:
            return {"category": "None", "signature": "No errors detected"}
        return {"category": "Unknown", "signature": "Unrecognized error"}

    if category == "OTHER":
        return {"category": "Unknown", "signature": "Unrecognized error"}

    return {
        "category": category,
        "signature": signature or primary,
    }


def normalize_error(raw_error: Any) -> Dict[str, Any]:
    """Return all durable error representations used by MongoDB and ChromaDB."""
    raw = _text(raw_error)
    secondary = extract_secondary_errors(raw)
    primary = extract_primary_error(raw, secondary)
    category = categorize_error(primary, secondary)
    primary = _shorten_primary_error(primary, category)
    return {
        "raw_error_message": raw,
        "error_message": primary,
        "error_category": category,
        "error_signature": normalize_error_signature(primary),
        "secondary_errors": secondary,
    }
