from backend.app.parsers.common import sanitize_error_message


def test_sanitize_error_message_keeps_real_assertion_and_drops_browser_noise():
    noisy = (
        "[BROWSER CONSOLE ERROR] Failed to load resource: the server responded with a status of 401 (Unauthorized)\n"
        "AssertionError [ERR_ASSERTION]: Expected product count 999, but received 0\n"
        "at CustomWorld.<anonymous> (...)"
    )

    cleaned = sanitize_error_message(noisy)

    assert "BROWSER CONSOLE ERROR" not in cleaned
    assert "Failed to load resource" not in cleaned
    assert "Expected product count 999, but received 0" in cleaned
    assert "AssertionError" not in cleaned
