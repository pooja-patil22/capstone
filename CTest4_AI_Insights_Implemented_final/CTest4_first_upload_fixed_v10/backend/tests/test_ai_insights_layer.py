from backend.analytics_engine import (
    _build_ai_insights_payload,
    detect_failure_patterns,
    normalize_record,
)


def test_analytics_uses_normalized_message_and_keeps_raw_error():
    raw_error = (
        "TimeoutError: locator.click: Timeout 30000ms exceeded\n"
        "at checkout (C:\\project\\steps.js:10:2)"
    )

    normalized = normalize_record({
        "test_id": "TC-1",
        "test_name": "Checkout",
        "status": "FAILED",
        "error_message": raw_error,
    })

    assert normalized["raw_error_message"] == raw_error
    assert normalized["error_message"] == "Element interaction timed out"
    assert normalized["error_category"] == "TIMEOUT"

    patterns = detect_failure_patterns([normalized], minimum_occurrences=1)
    assert patterns["patterns"][0]["examples"] == ["Element interaction timed out"]


def test_ai_insights_payload_is_filter_scoped_and_structured():
    records = [
        {
            "test_id": "TC-1", "test_name": "Payment", "module": "Checkout",
            "status": "FAILED", "error_message": "401 Unauthorized",
            "error_category": "AUTHENTICATION", "error_signature": "N Unauthorized",
            "secondary_errors": ["404 Not Found"],
        },
        {
            "test_id": "TC-2", "test_name": "Login", "module": "Login",
            "status": "PASSED", "error_message": "",
        },
    ]
    payload = _build_ai_insights_payload(
        records=records,
        analytics_categories={},
        rag_context={"chunks": [{"document": "historical 401", "metadata": {"test_id": "TC-OLD"}, "distance": 0.2}]},
    )
    assert payload["metrics"]["total_tests"] == 2
    assert payload["metrics"]["failed"] == 1
    assert payload["error_intelligence"]["category_counts"]["AUTHENTICATION"] == 1
    assert payload["error_intelligence"]["secondary_error_counts"]["404 Not Found"] == 1
    assert payload["recommendations"]
    assert payload["weekly_quality_digest"]["metrics"]["failed"] == 1
    assert payload["weekly_quality_digest"]["remediation"][0]["evidence"]
    assert payload["rag_evidence"][0]["metadata"]["test_id"] == "TC-OLD"


def test_ai_insights_fallback_never_requires_azure_openai():
    payload = _build_ai_insights_payload(
        records=[{"test_id": "TC-1", "status": "PASSED", "module": "Login"}],
        analytics_categories={},
        rag_context={},
    )
    assert payload["metrics"]["passed"] == 1
    assert payload["metrics"]["failed"] == 0
    assert "quality" in payload


