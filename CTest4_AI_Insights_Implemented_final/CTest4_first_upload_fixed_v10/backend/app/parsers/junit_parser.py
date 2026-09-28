import xml.etree.ElementTree as ET
from .common import canonical_test_record


def _local_tag(tag):
    return tag.rsplit("}", 1)[-1].lower() if tag else ""


def parse_junit_xml(raw_bytes: bytes):
    """Parse a JUnit XML report into the application's common test schema."""
    try:
        root = ET.fromstring(raw_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid JUnit XML: {exc}") from exc

    if _local_tag(root.tag) not in {"testsuite", "testsuites"}:
        raise ValueError("The XML file is not a supported JUnit test report.")

    testcases = root.findall(".//{*}testcase")
    records = []
    parent_map = {child: parent for parent in root.iter() for child in parent}

    def _suite_timestamp(testcase):
        """JUnit stores the run timestamp on the parent <testsuite>, not per-case."""
        node = testcase
        while node is not None:
            if _local_tag(node.tag) == "testsuite":
                for attr_key, attr_value in node.attrib.items():
                    if str(attr_key).lower() == "timestamp":
                        return attr_value
            node = parent_map.get(node)
        return ""

    for index, testcase in enumerate(testcases, start=1):
        attrs = {str(k).lower(): v for k, v in testcase.attrib.items()}
        name = attrs.get("name", "") or attrs.get("testname", "")
        classname = attrs.get("classname", "") or attrs.get("class", "")
        run_case_id = attrs.get("run_case_id", "") or attrs.get("runcaseid", "") or attrs.get("run_id", "") or attrs.get("runid", "")
        test_id = (
            attrs.get("test_case_id", "") or attrs.get("testcaseid", "")
            or attrs.get("testid", "") or attrs.get("id", "")
            or f"{classname}:{name}".strip(":") or str(index)
        )
        module = (
            attrs.get("module", "") or attrs.get("module_name", "")
            or classname or "Unknown"
        )
        duration = attrs.get("time", "")
        execution_time = (
            attrs.get("timestamp", "") or attrs.get("execution_time", "")
            or _suite_timestamp(testcase)
        )

        failure = testcase.find("./{*}failure")
        error = testcase.find("./{*}error")
        skipped = testcase.find("./{*}skipped")

        if failure is not None:
            status = "FAILED"
            message = failure.attrib.get("message", "")
            details = (failure.text or "").strip()
            error_message = "\n".join(part for part in (message, details) if part)
        elif error is not None:
            status = "FAILED"
            message = error.attrib.get("message", "")
            details = (error.text or "").strip()
            error_message = "\n".join(part for part in (message, details) if part)
        elif skipped is not None:
            status = "SKIPPED"
            error_message = ""
        else:
            status = "PASSED"
            error_message = ""

        records.append(canonical_test_record(
            run_case_id=run_case_id,
            test_id=test_id,
            test_name=name or test_id or f"Test {index}",
            status=status,
            module=module,
            error_message=error_message,
            execution_time=execution_time,
            duration=duration,
            source_format="JUnit XML",
        ))

    if not records:
        raise ValueError("No <testcase> entries were found in the JUnit XML report.")
    return records
