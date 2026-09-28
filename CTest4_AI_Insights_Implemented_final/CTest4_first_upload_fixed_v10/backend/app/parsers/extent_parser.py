import re
from html import unescape
from bs4 import BeautifulSoup
from .common import canonical_test_record

def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


_DATETIME_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(:\d{2})?"),
    re.compile(r"\d{2}/\d{2}/\d{4}[ ,]+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?", re.IGNORECASE),
    re.compile(r"\d{2}-\d{2}-\d{4}[ ,]+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?", re.IGNORECASE),
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{2}/\d{2}/\d{4}"),
    re.compile(r"\d{2}-\d{2}-\d{4}"),
]

_TIME_ATTR_NAMES = (
    "data-started-time", "data-start-time", "data-startedtime", "data-start",
    "data-time", "data-timestamp", "data-date", "data-started",
    "data-end-time", "data-endtime",
)


def _extract_datetime_text(text: str) -> str:
    text = text or ""
    for pattern in _DATETIME_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return ""


def _extract_node_time(node) -> str:
    """Read a per-test execution timestamp from attributes or nearby elements."""
    for attr in _TIME_ATTR_NAMES:
        value = node.get(attr)
        if value:
            extracted = _extract_datetime_text(str(value)) or _collapse_spaces(str(value))
            if extracted:
                return extracted

    time_node = (
        node.select_one(".test-time")
        or node.select_one(".time-info")
        or node.select_one(".timestamp")
        or node.select_one("[class*='time']")
        or node.select_one("[class*='date']")
    )
    if time_node:
        extracted = _extract_datetime_text(time_node.get_text(" ", strip=True))
        if extracted:
            return extracted

    return ""


def _extract_report_level_time(soup) -> str:
    """Fall back to the report's own generated/run timestamp when present."""
    for selector in (
        "#rp-report-datetime", ".report-datetime", ".test-run-time",
        "[class*='report-time']", "[id*='report-time']", "[class*='generated']",
        "[class*='run-time']",
    ):
        node = soup.select_one(selector)
        if node:
            extracted = _extract_datetime_text(node.get_text(" ", strip=True))
            if extracted:
                return extracted

    return _extract_datetime_text(soup.get_text(" ", strip=True))


def _extract_status(text: str) -> str:
    normalized = (text or "").strip().lower()
    if not normalized:
        return ""

    if re.search(r"\b(fail(?:ed|ure)?|error|broken|fatal)\b", normalized):
        return "FAILED"
    if re.search(r"\b(skip(?:ped)?|ignored|disabled)\b", normalized):
        return "SKIPPED"
    if re.search(r"\b(pass(?:ed)?|success(?:ful)?)\b", normalized):
        return "PASSED"
    return ""


def _extract_name(node, fallback: str) -> str:
    candidate = (
        node.get("data-name")
        or node.get("title")
        or node.get("name")
        or ""
    )

    if not candidate:
        name_node = (
            node.select_one(".test-name")
            or node.select_one(".test-name-node")
            or node.select_one(".node-name")
            or node.select_one(".name")
            or node.select_one("[class*='test-name']")
            or node.select_one("[class*='node-name']")
        )
        if name_node:
            candidate = name_node.get_text(" ", strip=True)

    if not candidate:
        candidate = fallback

    return _collapse_spaces(candidate)[:250]


def _extract_error_message(node) -> str:
    """Extract raw failure text from common ExtentReports error containers."""
    for attribute in (
        "data-error-message", "data-error", "data-message",
        "data-exception", "data-stacktrace", "data-stack-trace",
    ):
        value = node.get(attribute)
        if value:
            return str(value).strip()

    candidates = []
    for descendant in node.select(
        ".error-message, .error, .exception, .stacktrace, .stack-trace, "
        "[class*='error'], [class*='exception'], [class*='stacktrace'], "
        "[class*='stack-trace'], pre"
    ):
        text = descendant.get_text("\n", strip=True)
        if text:
            candidates.append(text)

    unique = []
    seen = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return "\n".join(unique)


def _extract_script_pairs(script_text: str):
    pairs = []
    seen = set()

    patterns = [
        re.compile(
            r"(?:['\"]?(?:name|testName|test_name|test|scenario|nodeName)['\"]?\s*:\s*['\"](?P<name>[^'\"]{1,350})['\"]).{0,2000}?"
            r"(?:['\"]?(?:status|statusLabel|state|result)['\"]?\s*:\s*['\"]?(?P<status>[A-Za-z_]+)['\"]?)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"(?:['\"]?(?:status|statusLabel|state|result)['\"]?\s*:\s*['\"]?(?P<status>[A-Za-z_]+)['\"]?).{0,2000}?"
            r"(?:['\"]?(?:name|testName|test_name|test|scenario|nodeName)['\"]?\s*:\s*['\"](?P<name>[^'\"]{1,350})['\"])",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(script_text or ""):
            raw_name = unescape(match.group("name") or "")
            raw_status = match.group("status") or ""
            name = _collapse_spaces(raw_name)
            status = _extract_status(raw_status)
            if not name or not status:
                continue
            dedupe_key = (name.casefold(), status)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            pairs.append((name, status))

    return pairs


def parse_extent_html(raw_bytes: bytes):
    """Parse common ExtentReports HTML structures into the common schema."""
    html = raw_bytes.decode("utf-8-sig", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    records = []
    seen_records = set()
    # Per-test timestamps are preferred; the report's own generated/run
    # timestamp is used only when no test node carries its own date/time.
    report_time = _extract_report_level_time(soup)

    def add_record(name: str, status: str, index_hint: int, run_case_id: str = "", test_id: str = "", module: str = "", error_message: str = "", execution_time: str = ""):
        if not name or not status:
            return
        key = (name.casefold(), status)
        if key in seen_records:
            return
        seen_records.add(key)
        records.append(canonical_test_record(
            run_case_id=run_case_id,
            test_id=test_id or str(index_hint),
            test_name=name,
            status=status,
            module=module or "Unknown",
            error_message=error_message,
            execution_time=execution_time or report_time,
            source_format="Extent Report",
        ))

    selectors = [
        ".test-item",
        ".test-item-content",
        ".test-node",
        ".test",
        "li.test",
        "[class*='test-item']",
        "[class*='test-node']",
        "[class*='status-']",
        "[data-status]",
    ]
    nodes = []
    for selector in selectors:
        found = soup.select(selector)
        if found:
            nodes = found
            break

    for index, node in enumerate(nodes, start=1):
        visible = _collapse_spaces(" ".join(node.stripped_strings))
        if not visible:
            continue
        classes = " ".join(node.get("class", []))
        data_status = " ".join(
            str(node.get(key) or "")
            for key in ("data-status", "status", "data-result", "result")
        )
        combined = f"{classes} {data_status} {visible}".lower()
        status = _extract_status(combined)
        if not status:
            continue

        name = _extract_name(node, visible)
        run_case_id = (
            node.get("data-run-case-id") or node.get("data-runcase-id")
            or node.get("data-run-id") or node.get("run-case-id") or ""
        )
        test_id = (
            node.get("data-test-case-id") or node.get("data-testcase-id")
            or node.get("data-test-id") or node.get("data-id") or ""
        )
        module = node.get("data-module") or node.get("data-module-name") or ""
        error_message = _extract_error_message(node)
        execution_time = _extract_node_time(node)
        add_record(name, status, index, run_case_id, test_id, module, error_message, execution_time)

    # Fallback for tabular Extent-style outputs.
    if not records:
        for row in soup.select("tr"):
            cells = [
                _collapse_spaces(cell.get_text(" ", strip=True))
                for cell in row.select("th, td")
            ]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            status = ""
            for cell in cells:
                status = _extract_status(cell)
                if status:
                    break
            if not status:
                continue

            name = ""
            execution_time = ""
            for cell in cells:
                if _extract_status(cell):
                    continue
                if not execution_time:
                    execution_time = _extract_datetime_text(cell)
                if not name and len(cell) > 1:
                    name = cell
            if name:
                add_record(name, status, len(records) + 1, execution_time=execution_time)

    # Fallback for Extent versions that keep the test data in script JSON.
    if not records:
        script_text = "\n".join(script.get_text() for script in soup.find_all("script"))
        for index, (name, status) in enumerate(_extract_script_pairs(script_text), start=1):
            add_record(name, status, index)

    if not records:
        raise ValueError("Could not extract test execution data from the ExtentReports HTML file.")
    return records
