import json

import pytest

from app.parsers import parse_uploaded_report


def test_unrecognized_html_is_rejected_instead_of_creating_unknown_record():
    payload = b"<html><head><title>Application page</title></head><body>Welcome</body></html>"

    with pytest.raises(ValueError, match="does not contain recognizable test execution results"):
        parse_uploaded_report("index.html", payload)


def test_junit_xml_is_detected_and_mapped():
    payload = b'''<testsuite timestamp="2026-08-28T10:00:00Z">
        <testcase testCaseId="TC-1" name="Login" module="Auth">
            <failure message="bad password" />
        </testcase>
    </testsuite>'''

    records, report_type, parser_type = parse_uploaded_report("report.xml", payload)

    assert (report_type, parser_type) == ("JUnit XML", "junit_xml")
    assert records[0]["test_id"] == "TC-1"
    assert records[0]["test_name"] == "Login"
    assert records[0]["module"] == "Auth"
    assert records[0]["status"] == "FAILED"
    assert records[0]["error_message"] == "bad password"
    assert records[0]["execution_time"] == "2026-08-28T10:00:00Z"


def test_allure_json_is_detected_and_mapped():
    payload = json.dumps({
        "uuid": "u1",
        "testCaseId": "TC-2",
        "fullName": "Checkout",
        "name": "Checkout",
        "status": "passed",
        "start": 1756375200000,
        "labels": [{"name": "suite", "value": "Payments"}],
    }).encode()

    records, report_type, parser_type = parse_uploaded_report("result.json", payload)

    assert (report_type, parser_type) == ("Allure", "allure_json")
    assert records[0]["test_id"] == "TC-2"
    assert records[0]["module"] == "Payments"
    assert records[0]["status"] == "PASSED"
    assert "T" in records[0]["execution_time"]


def test_extent_html_is_detected_and_mapped():
    payload = b'''<html><div class="test-item" data-test-id="TC-3"
        data-run-case-id="RUN-3" data-module="Orders"
        data-error-message="timeout">
        <span class="test-name">Create order</span>
        <span class="status-fail">Failed</span>
        <time class="test-time">2026-08-28 11:00:00</time>
    </div></html>'''

    records, report_type, parser_type = parse_uploaded_report("extent.html", payload)

    assert (report_type, parser_type) == ("Extent Report", "extent_html")
    assert records[0]["run_case_id"] == "RUN-3"
    assert records[0]["test_id"] == "TC-3"
    assert records[0]["test_name"] == "Create order"
    assert records[0]["module"] == "Orders"
    assert records[0]["status"] == "FAILED"
    assert records[0]["error_message"] == "timeout"
    assert records[0]["execution_time"] == "2026-08-28 11:00:00"


def test_extent_nested_error_container_is_captured_and_normalized():
    payload = b'''<html><div class="test-item" data-test-id="TC-3b">
        <span class="test-name">Open product</span>
        <span class="status-fail">Failed</span>
        <div class="error-message">TimeoutError: locator.click: Timeout 30000ms exceeded
            <pre>at step (C:\\project\\steps.js:10:2)</pre>
        </div>
    </div></html>'''

    records, _, _ = parse_uploaded_report("extent.html", payload)
    record = records[0]

    assert "TimeoutError" in record["raw_error_message"]
    assert "steps.js" in record["raw_error_message"]
    assert record["error_message"] == "Element interaction timed out"
    assert record["error_category"] == "TIMEOUT"

def test_junit_preserves_raw_and_adds_error_metadata():
    payload = b'''<testsuite><testcase testCaseId="TC-4" name="Payment"><failure message="Error: function timed out, ensure the promise resolves within 5000 milliseconds">at x (C:\\project\\test.js:1:2)</failure></testcase></testsuite>'''
    records, _, _ = parse_uploaded_report("payment.xml", payload)
    record = records[0]
    assert record["raw_error_message"].startswith("Error: function timed out")
    assert record["error_message"].startswith("function timed out")
    assert record["error_category"] == "TIMEOUT"
    assert record["error_signature"] == "function timed out ensure the promise resolves within N milliseconds"
    assert "test.js:1:2" in record["raw_error_message"]


def test_allure_preserves_raw_and_http_secondary_error():
    payload = json.dumps({
        "uuid": "u-http",
        "testCaseId": "TC-5",
        "name": "Load order",
        "status": "failed",
        "statusDetails": {
            "message": "AssertionError: Expected 1 but received 0",
            "trace": "AssertionError: Expected 1 but received 0\\n at x (/app/test.js:2:3)\\n[Browser Console Error] Failed to load resource: the server responded with a status of 403 (Forbidden)",
        },
    }).encode()
    records, _, _ = parse_uploaded_report("allure.json", payload)
    record = records[0]
    assert record["raw_error_message"].startswith("AssertionError")
    assert record["error_message"] == "Expected 1 but received 0"
    assert record["secondary_errors"] == ["403 Forbidden"]
    assert record["error_category"] == "ASSERTION_FAILURE"


def test_extent_preserves_error_metadata():
    payload = b'''<html><div class="test-item" data-test-id="TC-6" data-error-message="Error: 404 Not Found\nat x (/app/test.js:1:2)"><span class="test-name">Get order</span><span class="status-fail">Failed</span></div></html>'''
    records, _, _ = parse_uploaded_report("extent.html", payload)
    record = records[0]
    assert record["raw_error_message"].startswith("Error: 404 Not Found")
    assert record["error_message"] == "404 Not Found"
    assert record["secondary_errors"] == ["404 Not Found"]
    assert record["error_category"] == "RESOURCE_NOT_FOUND"


def test_generic_cucumber_style_json_gets_same_error_normalization():
    payload = json.dumps([{
        "run_id": "RUN-7",
        "test_id": "TC-7",
        "test_name": "Checkout",
        "module": "Checkout",
        "status": "failed",
        "error_message": "TypeError: Cannot read properties of undefined (reading 'click')\n at World.<anonymous> (/app/steps/checkout.js:8:1)",
    }]).encode()
    records, _, _ = parse_uploaded_report("cucumber.json", payload)
    record = records[0]
    assert record["error_message"] == "Cannot read properties of undefined (reading 'click')"
    assert record["raw_error_message"].startswith("TypeError:")
    assert record["error_category"] == "OTHER"


def test_generic_json_failure_aliases_are_normalized():
    payload = json.dumps([{
        "name": "Missing order",
        "status": "failed",
        "failure_message": "Error: 404 Not Found\nat fetch (/app/test.js:4:2)",
    }]).encode()

    records, _, _ = parse_uploaded_report("generic.json", payload)

    assert records[0]["raw_error_message"].startswith("Error: 404 Not Found")
    assert records[0]["error_message"] == "404 Not Found"
    assert records[0]["error_category"] == "RESOURCE_NOT_FOUND"
