from app.error_normalizer import (
    categorize_error,
    extract_primary_error,
    extract_secondary_errors,
    normalize_error,
    normalize_error_signature,
    normalize_error_summary,
)


def test_timeout_primary_error_drops_stack_and_normalizes_signature():
    raw = """Error: function timed out, ensure the promise resolves within 5000 milliseconds
    at World.<anonymous> (C:\\project\\features\\steps\\checkout.js:42:17)
    at async runStep (node_modules/@cucumber/cucumber/src/runtime.ts:101:9)
    """
    result = normalize_error(raw)

    assert result["raw_error_message"] == raw.strip()
    assert result["error_message"] == "function timed out, ensure the promise resolves within 5000 milliseconds"
    assert result["error_category"] == "TIMEOUT"
    assert result["error_signature"] == "function timed out ensure the promise resolves within N milliseconds"
    assert "checkout.js" not in result["error_message"]
    assert result["secondary_errors"] == []


def test_assertion_primary_error_and_browser_401_secondary_error_are_preserved():
    raw = """[BROWSER CONSOLE ERROR] Failed to load resource: the server responded with a status of 401 (Unauthorized)
AssertionError [ERR_ASSERTION]: Expected product count 999, but received 0
at CustomWorld.<anonymous> (/workspace/features/steps/cart.js:12:4)
"""
    result = normalize_error(raw)

    assert result["error_message"] == "Expected product count 999, but received 0"
    assert result["error_category"] == "ASSERTION_FAILURE"
    assert result["secondary_errors"] == ["401 Unauthorized"]
    assert "BROWSER CONSOLE ERROR" not in result["error_message"]
    assert "cart.js" not in result["error_message"]


def test_http_categories():
    assert categorize_error("401 Unauthorized", ["401 Unauthorized"]) == "AUTHENTICATION"
    assert categorize_error("403 Forbidden", ["403 Forbidden"]) == "AUTHORIZATION"
    assert categorize_error("404 Not Found", ["404 Not Found"]) == "RESOURCE_NOT_FOUND"


def test_element_not_found_category():
    message = "locator('#checkout').click: strict mode violation: locator resolved to 0 elements"
    assert categorize_error(message) == "ELEMENT_NOT_FOUND"


def test_secondary_http_errors_are_unique_and_ordered():
    raw = "401 Unauthorized\n403 Forbidden\n401 Unauthorized\n404 Not Found"
    assert extract_secondary_errors(raw) == [
        "401 Unauthorized",
        "403 Forbidden",
        "404 Not Found",
    ]


def test_signature_normalizes_urls_ids_dates_and_numbers():
    message = "Request https://example.test/orders/123?id=987 at 2026-08-28T10:20:30Z failed for RUN-004"
    signature = normalize_error_signature(message)
    assert signature == "Request URL at TIMESTAMP failed for ID"


def test_primary_error_falls_back_to_http_signal():
    raw = "Failed to load resource: the server responded with a status of 404 (Not Found)\nat fetch (node:internal/http:1:2)"
    assert extract_primary_error(raw) == "Failed to load resource: the server responded with a status of 404 (Not Found)"


def test_vector_document_uses_clean_error_not_raw_stack():
    # Keep this test independent from Chroma/SentenceTransformer imports.
    from backend.app.error_normalizer import normalize_error

    normalized = normalize_error(
        "Error: function timed out, ensure the promise resolves within 5000 milliseconds\n"
        "at World (/workspace/node_modules/foo.js:10:2)"
    )
    searchable = f"error message: {normalized['error_message']}. error signature: {normalized['error_signature']}"
    assert "node_modules" not in searchable
    assert "function timed out" in searchable
    assert "N milliseconds" in searchable


def test_playwright_timeout_gets_concise_message():
    result = normalize_error("TimeoutError: locator.click: Timeout 30000ms exceeded")

    assert result["error_message"] == "Element interaction timed out"
    assert result["error_category"] == "TIMEOUT"


def test_playwright_timeout_with_waiting_text_is_not_misclassified_as_assertion():
    result = normalize_error(
        'TimeoutError: locator("[data-test="inventory-item-name"]") exceeded '
        "10000ms while waiting for the product details link to become visible."
    )

    assert result["error_message"] == "Element interaction timed out"
    assert result["error_category"] == "TIMEOUT"
    assert "locator" not in result["error_message"].lower()


def test_playwright_assertion_selector_is_replaced_with_human_description():
    result = normalize_error(
        'Error: expect(locator).toHaveText(expected) failed\n'
        'Expected: "Sauce Labs Backpack"\n'
        'Received: "Sauce Labs Bike Light"'
    )

    assert result["error_message"] == "Expected value did not match actual value"
    assert result["error_category"] == "ASSERTION_FAILURE"


def test_element_not_found_gets_concise_message():
    result = normalize_error("Error: locator('#submit') resolved to 0 elements")

    assert result["error_message"] == "Element not found"
    assert result["error_category"] == "ELEMENT_NOT_FOUND"


def test_common_http_and_unknown_errors_are_normalized():
    assertion = normalize_error("AssertionError: expected 2 but received 1")
    not_found = normalize_error("404 Not Found")
    server_error = normalize_error("500 Internal Server Error")
    unknown = normalize_error("Unexpected test failure")

    assert assertion["error_message"] == "expected 2 but received 1"
    assert assertion["error_category"] == "ASSERTION_FAILURE"
    assert not_found["error_message"] == "404 Not Found"
    assert not_found["error_category"] == "RESOURCE_NOT_FOUND"
    assert server_error["error_message"] == "Internal server error"
    assert server_error["error_category"] == "SERVER_ERROR"
    assert unknown["error_message"] == "Unexpected test failure"
    assert unknown["error_category"] == "OTHER"


def test_report_placeholders_are_not_stored_as_errors():
    result = normalize_error("—")

    assert result["raw_error_message"] == ""
    assert result["error_message"] == ""
    assert result["error_signature"] == ""
    assert result["error_category"] == "OTHER"
    assert result["secondary_errors"] == []


def test_normalizer_removes_timestamps_line_numbers_and_duplicate_messages():
    raw = (
        "Error: request failed at 2026-08-28T10:20:30Z, line 42\n"
        "request failed at 2026-08-28T10:20:30Z, line 42\n"
        "at runner (/workspace/test.js:42:7)"
    )

    result = normalize_error(raw)

    assert result["error_message"] == "request failed"
    assert "2026-08-28" not in result["error_message"]
    assert "line 42" not in result["error_message"]


def test_normalizer_removes_html_and_non_readable_symbols():
    raw = "<div>!!! Product count mismatch &#40;expected 6, received 5&#41; \u200b</div>"

    result = normalize_error(raw)

    assert result["error_message"] == "!!! Product count mismatch (expected 6, received 5)"
    assert "<div>" not in result["error_message"]
    assert "\u200b" not in result["error_message"]
    assert result["error_category"] == "ASSERTION_FAILURE"


def test_multiline_assertions_prefer_headline_mismatch_over_expected_received_details():
    raw = (
        "AssertionError: Product count mismatch.\n"
        "Expected: 6\n"
        "Received: 5"
    )

    result = normalize_error(raw)

    assert result["error_message"] == "Product count mismatch."
    assert result["error_category"] == "ASSERTION_FAILURE"
    assert result["error_signature"] == "Product count mismatch"


def test_page_title_assertion_signature_is_clean_and_not_doubled():
    raw = (
        "AssertionError: Page title mismatch.\n"
        'Expected: "Swag Labs"\n'
        'Received: "Sauce Labs"'
    )

    result = normalize_error(raw)

    assert result["error_message"] == "Page title mismatch."
    assert result["error_category"] == "ASSERTION_FAILURE"
    assert result["error_signature"] == "Page title mismatch"


def test_normalize_error_summary_contract_for_dashboard_and_storage():
    recognized = normalize_error_summary("AssertionError: Product count mismatch.\nExpected: 6\nReceived: 5")
    assert recognized == {"category": "ASSERTION_FAILURE", "signature": "Product count mismatch"}

    empty = normalize_error_summary("—")
    assert empty == {"category": "None", "signature": "No errors detected"}

    uncertain = normalize_error_summary("something weird happened")
    assert uncertain == {"category": "Unknown", "signature": "Unrecognized error"}
