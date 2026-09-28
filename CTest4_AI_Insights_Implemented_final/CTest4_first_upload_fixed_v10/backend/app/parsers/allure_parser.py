import io
import json
import zipfile
from datetime import datetime, timezone
from .common import canonical_test_record, first_non_empty


def _execution_time(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
    return str(value or "")


def parse_allure_result(result: dict):
    status = result.get("status", "")
    details = result.get("statusDetails") or {}
    labels = result.get("labels") or []

    module = ""
    suite = ""
    for label in labels:
        if not isinstance(label, dict):
            continue
        name = str(label.get("name", "")).lower()
        value = label.get("value", "")
        if name in {"suite", "parentsuite", "subsuite", "feature"} and not suite:
            suite = value
        if name in {"package", "parentsuite", "suite"} and not module:
            module = value

    start = result.get("start", "")
    stop = result.get("stop", "")
    duration = ""
    if start and stop:
        try:
            duration = (float(stop) - float(start)) / 1000.0
        except (TypeError, ValueError):
            pass

    error_message = ""
    if isinstance(details, dict):
        message = str(details.get("message") or "").strip()
        trace = str(details.get("trace") or "").strip()
        if message and trace and trace != message:
            error_message = f"{message}\n{trace}"
        else:
            error_message = message or trace

    run_case_id = result.get("run_case_id") or result.get("runCaseId") or result.get("run_id") or result.get("runId") or ""
    for label in labels:
        if isinstance(label, dict) and str(label.get("name", "")).lower() in {"run_case_id", "runcaseid", "run_id", "runid"}:
            run_case_id = label.get("value", "") or run_case_id
            break

    return canonical_test_record(
        run_case_id=run_case_id,
        test_id=result.get("testCaseId") or result.get("historyId") or result.get("uuid", ""),
        test_name=result.get("name") or result.get("fullName", "Unknown Test"),
        status=status,
        module=module or suite or "Unknown",
        error_message=error_message,
        execution_time=_execution_time(start),
        duration=duration,
        source_format="Allure",
    )


def parse_allure_json(raw_bytes: bytes):
    try:
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Allure JSON: {exc}") from exc

    if isinstance(data, dict):
        if "status" in data or "name" in data:
            return [parse_allure_result(data)]
        raise ValueError("The JSON file does not look like an Allure result file.")
    if isinstance(data, list):
        records = [parse_allure_result(item) for item in data if isinstance(item, dict) and ("status" in item or "name" in item)]
        if records:
            return records
    raise ValueError("Unsupported Allure JSON structure.")


def parse_allure_zip(raw_bytes: bytes):
    records = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            json_names = [name for name in archive.namelist() if name.lower().endswith("-result.json")]
            if not json_names:
                raise ValueError("The ZIP does not contain Allure *-result.json files.")
            for name in json_names:
                try:
                    result = json.loads(archive.read(name).decode("utf-8-sig"))
                    if isinstance(result, dict):
                        records.append(parse_allure_result(result))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP file is invalid.") from exc

    if not records:
        raise ValueError("No valid Allure test result files were found in the ZIP.")
    return records
