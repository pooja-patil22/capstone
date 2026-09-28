from frontend.app import (
    _classify_failure_pattern,
    _filter_failure_analysis_records,
    _normalize_failure_matrix_row,
)


def test_normalize_failure_matrix_row_uses_legacy_aliases_and_maps_failures():
    row = {
        "Test ID": "TC-100",
        "Test Name": "Login flow",
        "Module": "Auth",
        "Run Case ID": "RUN-3",
        "Status": "Error",
        "Failure Message": "Null pointer",
    }

    normalized = _normalize_failure_matrix_row(row)

    assert normalized["run_id"] == "RUN-3"
    assert normalized["test_id"] == "TC-100"
    assert normalized["test_name"] == "Login flow"
    assert normalized["module"] == "Auth"
    assert normalized["status"] == "FAILED"
    assert normalized["error_message"] == "Null pointer"


def test_classify_failure_pattern_handles_mixed_pass_fail_history():
    assert _classify_failure_pattern(["PASSED", "FAILED", "PASSED"]) == "Intermittent"
    assert _classify_failure_pattern(["FAILED", "PASSED"]) == "Recovered"
    assert _classify_failure_pattern(["PASSED", "FAILED"]) == "New Failure"


def test_filter_failure_analysis_records_keeps_only_selected_module_history():
    records = [
        {"test_id": "TC-01", "module": "Homepage", "status": "PASSED"},
        {"test_id": "TC-01", "module": "Homepage", "status": "FAILED"},
        {"test_id": "TC-02", "module": "Product Details", "status": "FAILED"},
    ]

    filtered = _filter_failure_analysis_records(records, "Homepage")

    assert filtered == records[:2]
