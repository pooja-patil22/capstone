import csv
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup

from .allure_parser import parse_allure_json, parse_allure_zip
from .extent_parser import parse_extent_html
from .junit_parser import parse_junit_xml
from .common import canonical_test_record, first_non_empty


def _safe_status(value):
    text = str(value or "").strip().lower()
    if "fail" in text or "error" in text or "broken" in text:
        return "FAILED"
    if "skip" in text or "ignore" in text or "disable" in text:
        return "SKIPPED"
    if "pass" in text or "success" in text:
        return "PASSED"
    return "UNKNOWN"


def _fallback_records(records, source_format, name_prefix="Record"):
    normalized = []
    for index, raw in enumerate(records or [], start=1):
        data = raw if isinstance(raw, dict) else {"value": raw}
        text_name = (
            data.get("test_name")
            or data.get("name")
            or data.get("title")
            or data.get("id")
            or f"{name_prefix} {index}"
        )
        text_status = (
            data.get("status")
            or data.get("result")
            or data.get("outcome")
            or data.get("state")
            or "UNKNOWN"
        )
        normalized.append(canonical_test_record(
            run_case_id=str(data.get("run_case_id") or data.get("runcase_id") or data.get("run_case") or data.get("run_id") or ""),
            test_id=str(data.get("test_id") or data.get("id") or index),
            test_name=str(text_name),
            status=_safe_status(text_status),
            module=str(data.get("module") or data.get("component") or "Unknown"),
            error_message=str(
                data.get("error_message")
                or data.get("error")
                or data.get("failure_message")
                or data.get("failure_reason")
                or data.get("message")
                or data.get("exception")
                or data.get("error_detail")
                or ""
            ),
            execution_time=str(data.get("execution_time") or data.get("timestamp") or ""),
            duration=data.get("duration", ""),
            version=str(data.get("version") or "Unknown"),
            environment=str(data.get("environment") or "Unknown"),
            source_format=source_format,
            raw_payload=data,
        ))
    if normalized:
        return normalized
    return [canonical_test_record(
        test_id="1",
        test_name=f"{name_prefix} 1",
        status="UNKNOWN",
        module="Unknown",
        source_format=source_format,
    )]


def _xml_element_to_dict(element):
    payload = {}
    for child in list(element):
        tag = child.tag.rsplit("}", 1)[-1]
        value = (child.text or "").strip()
        if value:
            payload[tag] = value
    text = (element.text or "").strip()
    if text and not payload:
        payload[element.tag.rsplit("}", 1)[-1]] = text
    return payload


def _parse_generic_html(raw_bytes):
    text = _decode_text(raw_bytes)
    soup = BeautifulSoup(text, "html.parser")

    tabular = []
    for table in soup.select("table"):
        headers = [th.get_text(" ", strip=True) for th in table.select("tr th")]
        for row in table.select("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.select("td")]
            if not cells:
                continue
            if headers and len(headers) == len(cells):
                tabular.append({headers[i] or f"column_{i+1}": cells[i] for i in range(len(cells))})
            else:
                tabular.append({f"column_{i+1}": value for i, value in enumerate(cells)})

    if tabular:
        try:
            return _parse_tabular_records(tabular, "HTML"), "HTML", "generic_html"
        except ValueError:
            return _fallback_records(tabular, "HTML", "HTML Row"), "HTML", "generic_html"

    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    kv_rows = []
    for line in lines:
        pairs = re.findall(r"([A-Za-z][\w .-]*)\s*[:=]\s*(.*?)(?=\s*[,|]\s*[A-Za-z][\w .-]*\s*[:=]|$)", line)
        if pairs:
            kv_rows.append({k.strip(): v.strip() for k, v in pairs})

    if kv_rows:
        try:
            return _parse_tabular_records(kv_rows, "HTML"), "HTML", "generic_html"
        except ValueError:
            return _fallback_records(kv_rows, "HTML", "HTML Row"), "HTML", "generic_html"

    return _fallback_records([{"text": " ".join(lines[:5])}] if lines else [{}], "HTML", "HTML Content"), "HTML", "generic_html"


def _parse_pdf_report(raw_bytes):
    text_content = ""
    extraction_errors = []

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))
        text_content = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        extraction_errors.append(exc)

    if not text_content.strip():
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            text_content = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            extraction_errors.append(exc)

    if not text_content.strip():
        if extraction_errors:
            raise ValueError(
                "PDF upload is supported, but text could not be extracted. "
                "Install pypdf for reliable PDF parsing."
            )
        raise ValueError("PDF upload is supported, but the file contained no extractable text.")

    rows = []
    for line in text_content.splitlines():
        line = line.strip()
        if not line:
            continue
        pairs = re.findall(r"([A-Za-z][\w .-]*)\s*[:=]\s*(.*?)(?=\s*[,|]\s*[A-Za-z][\w .-]*\s*[:=]|$)", line)
        if pairs:
            rows.append({k.strip(): v.strip() for k, v in pairs})

    if rows:
        try:
            return _parse_tabular_records(rows, "PDF"), "PDF", "pdf_report"
        except ValueError:
            return _fallback_records(rows, "PDF", "PDF Record"), "PDF", "pdf_report"

    fallback_lines = [line.strip() for line in text_content.splitlines() if line.strip()][:20]
    return _fallback_records(
        [{"line": line, "status": _safe_status(line)} for line in fallback_lines] or [{}],
        "PDF",
        "PDF Line",
    ), "PDF", "pdf_report"


def _normalize_header(value):
    """Match mongo_data_filter's lenient normalization so headers like
    'Execution Date & Time' or 'Executed-At' still resolve to a known alias."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _parse_tabular_records(records, source_format):
    aliases = {
        "run_case_id": ["run_case_id", "runcase_id", "run_case", "run_caseid", "run_id", "runid", "execution_id", "execution_run_id"],
        "test_id": ["test_id", "testid", "id", "test_case_id", "testcase_id", "case_id"],
        "test_name": ["test_name", "testname", "name", "scenario", "test_case_name", "testcase_name", "test"],
        "status": ["status", "result", "outcome", "test_status", "execution_status", "state"],
        "module": ["module", "module_name", "component", "feature", "area", "suite", "test_suite", "classname", "class", "test_module", "component_name", "package"],
        "error_message": ["error_message", "error", "failure_message", "failure_reason", "message", "exception", "error_detail"],
        "execution_time": ["execution_time", "execution_date_and_time", "execution_date_time", "execution_date", "executed_at", "timestamp", "execution_timestamp", "run_time", "date", "datetime"],
        "duration": ["duration", "duration_seconds", "execution_duration", "elapsed_time", "time_taken", "time"],
        "version": ["version", "build_version", "build", "build_number", "release", "release_version", "app_version", "build_no", "buildno"],
        "environment": ["environment", "env", "test_environment", "env_name", "test_env", "stage"],
    }
    normalized = []
    for index, raw in enumerate(records, start=1):
        source = {_normalize_header(k): v for k, v in (raw or {}).items()}
        get = lambda key: first_non_empty(source, aliases[key], "")
        test_name = get("test_name")
        status = get("status")
        if not test_name and not status:
            continue
        run_case_id = get("run_case_id")
        test_id = get("test_id") or test_name or str(index)
        normalized.append(canonical_test_record(
            run_case_id=run_case_id,
            test_id=test_id,
            test_name=test_name or test_id,
            status=status or "UNKNOWN",
            module=get("module"),
            error_message=get("error_message"),
            execution_time=get("execution_time"),
            duration=get("duration"),
            version=get("version"),
            environment=get("environment"),
            source_format=source_format,
        ))
    if not normalized:
        raise ValueError(
            "No test execution records could be identified. Expected fields such as "
            "test_name/name/test and status/result/outcome."
        )
    return normalized


def _decode_text(raw_bytes):
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _looks_like_xml(text):
    stripped = text.lstrip()
    return stripped.startswith("<?xml") or stripped.startswith("<") and bool(re.search(r"<[^>]+>", stripped[:5000]))


def _looks_like_html(text):
    sample = text[:10000].lower()
    return "<html" in sample or "<!doctype html" in sample or "<body" in sample


def _parse_generic_json(raw_bytes):
    data = json.loads(_decode_text(raw_bytes))
    if isinstance(data, dict):
        # Prefer the generic schema when the JSON explicitly contains the
        # application's test-case fields. This prevents ordinary JSON test
        # rows with a "status" field from being misclassified as Allure.
        explicit_test_fields = {
            "run_case_id", "runCaseId", "run_id", "runId",
            "test_id", "testId", "test_case_id",
            "test_name", "testName", "test_case_name",
            "module", "module_name", "error_message", "errorMessage",
            "execution_time", "executionTime",
        }
        if any(key in data for key in explicit_test_fields):
            try:
                return _parse_tabular_records([data], "JSON"), "JSON", "generic_json"
            except ValueError:
                pass

        # Allure result object.
        if _is_allure_result(data):
            return parse_allure_json(raw_bytes), "Allure", "allure_json"
        # Common report wrappers: {tests: [...]} / {results: [...]} / {testCases: [...]}
        for key in ("tests", "results", "testCases", "testcases", "executions", "test_executions", "data"):
            value = data.get(key)
            if isinstance(value, list) and value:
                try:
                    return _parse_tabular_records(value, "JSON"), "JSON", "generic_json"
                except ValueError:
                    return _fallback_records(value, "JSON", "JSON Record"), "JSON", "generic_json"
        try:
            return _parse_tabular_records([data], "JSON"), "JSON", "generic_json"
        except ValueError:
            return _fallback_records([data], "JSON", "JSON Record"), "JSON", "generic_json"
    if isinstance(data, list):
        if not data:
            raise ValueError("The JSON file contains no records.")
        explicit_test_fields = {
            "run_case_id", "runCaseId", "run_id", "runId",
            "test_id", "testId", "test_case_id",
            "test_name", "testName", "test_case_name",
            "module", "module_name", "error_message", "errorMessage",
            "execution_time", "executionTime",
        }
        if any(isinstance(x, dict) and any(key in x for key in explicit_test_fields) for x in data):
            try:
                return _parse_tabular_records(data, "JSON"), "JSON", "generic_json"
            except ValueError:
                pass
        if any(isinstance(x, dict) and _is_allure_result(x) for x in data):
            return parse_allure_json(raw_bytes), "Allure", "allure_json"
        try:
            return _parse_tabular_records(data, "JSON"), "JSON", "generic_json"
        except ValueError:
            return _fallback_records(data, "JSON", "JSON Record"), "JSON", "generic_json"
    raise ValueError("The JSON file does not contain test execution records.")


def _is_allure_result(value):
    return isinstance(value, dict) and any(
        key in value for key in (
            "statusDetails", "historyId", "uuid", "fullName", "testCaseId",
            "start", "stop", "labels",
        )
    )


def _parse_generic_xml(raw_bytes):
    # JUnit parser is deliberately selected by content, not extension.
    try:
        root = ET.fromstring(raw_bytes)
        tags = {element.tag.rsplit("}", 1)[-1].lower() for element in root.iter()}
        if "testcase" in tags:
            return parse_junit_xml(raw_bytes), "JUnit XML", "junit_xml"
        rows = []
        for element in root.iter():
            children = list(element)
            if not children:
                continue
            row = _xml_element_to_dict(element)
            if row:
                rows.append(row)
        if not rows:
            single = _xml_element_to_dict(root)
            if single:
                rows = [single]
        if rows:
            try:
                return _parse_tabular_records(rows, "XML"), "XML", "generic_xml"
            except ValueError:
                return _fallback_records(rows, "XML", "XML Record"), "XML", "generic_xml"
    except ET.ParseError:
        pass
    raise ValueError("XML file was accepted but no extractable records were found.")


def _parse_generic_csv(raw_bytes):
    text = _decode_text(raw_bytes)
    sample = text[:10000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    rows = df.to_dict(orient="records")
    try:
        return _parse_tabular_records(rows, "CSV"), "CSV", "tabular"
    except ValueError:
        return _fallback_records(rows, "CSV", "CSV Row"), "CSV", "tabular"


def _parse_zip_any(raw_bytes):
    """Parse report files contained in a ZIP, not only Allure ZIPs."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        members = [n for n in archive.namelist() if not n.endswith("/")]
        allure_members = [n for n in members if n.lower().endswith("-result.json")]
        if allure_members:
            return parse_allure_zip(raw_bytes), "Allure", "allure_zip"

        all_records = []
        detected = []
        for member in members:
            lower = member.lower()
            # Avoid images, binaries and metadata directories.
            if any(lower.endswith(ext) for ext in (".xml", ".json", ".csv", ".html", ".htm", ".xlsx", ".xls")):
                try:
                    content = archive.read(member)
                    records, fmt, _ = parse_uploaded_report(member, content)
                    all_records.extend(records)
                    detected.append(fmt)
                except Exception:
                    continue
        if all_records:
            unique_formats = ", ".join(dict.fromkeys(detected))
            return all_records, unique_formats or "ZIP", "zip_bundle"
    raise ValueError("The ZIP file does not contain a recognized test report.")


def detect_report_type(filename, raw_bytes):
    """Detect report format from extension and file content, so extensions are not mandatory."""
    name = (filename or "").lower()
    extension = name.rsplit(".", 1)[-1] if "." in name else ""
    text = _decode_text(raw_bytes)
    stripped = text.lstrip()

    # Office formats (.xlsx/.xls/.xlsm) are themselves ZIP containers, so they
    # must be excluded here to avoid being misdetected as generic ZIP bundles.
    office_extensions = {"xlsx", "xls", "xlsm", "docx", "pptx"}
    if extension not in office_extensions and (extension == "zip" or raw_bytes[:4] == b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
                names = [n.lower() for n in archive.namelist()]
                if any(n.endswith("-result.json") for n in names):
                    return "allure_zip"
                if any(n.endswith((".xml", ".json", ".csv", ".html", ".htm", ".xlsx", ".xls")) for n in names):
                    return "zip_bundle"
        except zipfile.BadZipFile:
            pass

    if extension in {"xlsx", "xls", "xlsm"}:
        return "tabular_excel"
    if extension == "csv":
        return "tabular_csv"
    if extension == "pdf" or raw_bytes[:5] == b"%PDF-":
        return "pdf_report"

    if extension == "json" or stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(text)
            explicit_test_fields = {
                "run_case_id", "runCaseId", "run_id", "runId",
                "test_id", "testId", "test_case_id",
                "test_name", "testName", "test_case_name",
                "module", "module_name", "error_message", "errorMessage",
                "execution_time", "executionTime",
            }
            if isinstance(data, dict):
                if any(key in data for key in explicit_test_fields):
                    return "generic_json"
                if _is_allure_result(data):
                    return "allure_json"
            if isinstance(data, list):
                if any(isinstance(x, dict) and any(key in x for key in explicit_test_fields) for x in data):
                    return "generic_json"
                if any(_is_allure_result(x) for x in data):
                    return "allure_json"
            return "generic_json"
        except Exception:
            pass

    # Extension is checked first and trusted immediately for xml/html, since
    # content-sniffing alone cannot reliably tell them apart: any HTML page
    # also "looks like XML" (starts with "<" and contains tags).
    if extension == "xml":
        return "junit_xml"
    if extension in {"html", "htm"}:
        return "extent_html"

    # Without a disambiguating extension, HTML is checked first because its
    # signature (<html>, <!doctype html>, <body>) is more specific than the
    # generic "starts with a tag" check used for XML.
    if _looks_like_html(text):
        return "extent_html"
    if _looks_like_xml(text):
        return "junit_xml"

    # Generic delimited text with a header containing likely test fields.
    first_lines = "\n".join(text.splitlines()[:5]).lower()
    if re.search(r"(?:test_name|testname|test[_ ]?case|status|result|outcome)\s*[:=]", first_lines):
        return "generic_text"
    if any(token in first_lines for token in ("test_name", "testname", "test case", "testcase", "status", "result", "outcome")) and any(d in first_lines for d in (",", ";", "\t", "|")):
        return "generic_csv"

    return "generic_text"


def parse_uploaded_report(filename, raw_bytes):
    """Route any uploaded file to a dedicated parser or generic content parser."""
    report_type = detect_report_type(filename, raw_bytes)

    if report_type == "junit_xml":
        try:
            return parse_junit_xml(raw_bytes), "JUnit XML", report_type
        except Exception:
            return _parse_generic_xml(raw_bytes)
    if report_type == "allure_json":
        return parse_allure_json(raw_bytes), "Allure", report_type
    if report_type == "allure_zip":
        return parse_allure_zip(raw_bytes), "Allure", report_type
    if report_type == "zip_bundle":
        return _parse_zip_any(raw_bytes)
    if report_type == "extent_html":
        # Standard HTML tables containing the requested test fields should
        # use the generic tabular parser so every uploaded value is preserved.
        html_sample = _decode_text(raw_bytes)[:30000].lower()
        if "<table" in html_sample and re.search(
            r"(run[_ ]?case|test[_ ]?case|test[_ ]?name|testname|status|module)",
            html_sample,
        ):
            try:
                return _parse_generic_html(raw_bytes)
            except Exception:
                pass
        try:
            return parse_extent_html(raw_bytes), "Extent Report", report_type
        except Exception:
            records, detected_format, parser_type = _parse_generic_html(raw_bytes)
            if not any(record.get("status") in {"PASSED", "FAILED", "SKIPPED", "BLOCKED"} for record in records):
                raise ValueError(
                    "The HTML file does not contain recognizable test execution results. "
                    "Upload an ExtentReports HTML file or the complete Allure results ZIP."
                )
            return records, detected_format, parser_type
    if report_type == "pdf_report":
        return _parse_pdf_report(raw_bytes)
    if report_type == "tabular_csv":
        return _parse_generic_csv(raw_bytes)
    if report_type == "tabular_excel":
        lower = filename.lower()
        df = pd.read_excel(io.BytesIO(raw_bytes), engine="xlrd" if lower.endswith(".xls") else None)
        rows = df.to_dict(orient="records")
        try:
            return _parse_tabular_records(rows, "Excel"), "Excel", report_type
        except ValueError:
            return _fallback_records(rows, "Excel", "Excel Row"), "Excel", report_type
    if report_type == "generic_json":
        return _parse_generic_json(raw_bytes)
    if report_type == "generic_csv":
        return _parse_generic_csv(raw_bytes)
    if report_type == "generic_text":
        # Readable text/log reports are accepted. First try common key/value
        # formats, then fall back to delimited text.
        records = []
        for line in _decode_text(raw_bytes).splitlines():
            line = line.strip()
            if not line:
                continue
            pairs = re.findall(r"([A-Za-z][\w .-]*)\s*[:=]\s*(.*?)(?=\s*[,|]\s*[A-Za-z][\w .-]*\s*[:=]|$)", line)
            if pairs:
                records.append({k.strip(): v.strip() for k, v in pairs})
        if records:
            try:
                return _parse_tabular_records(records, "Text/Log"), "Text/Log", report_type
            except ValueError:
                return _fallback_records(records, "Text/Log", "Text Record"), "Text/Log", report_type
        try:
            return _parse_generic_csv(raw_bytes)
        except Exception:
            pass
        lines = [line.strip() for line in _decode_text(raw_bytes).splitlines() if line.strip()]
        if lines:
            return _fallback_records(
                [{"line": line, "status": _safe_status(line)} for line in lines[:50]],
                "Text/Log",
                "Text Line",
            ), "Text/Log", report_type

    raise ValueError(
        "The file was accepted, but no test execution records could be extracted. "
        "The parser supports JUnit XML, Allure, ExtentReports, CSV/Excel, JSON, PDF, ZIP bundles, and readable text/log reports."
    )
