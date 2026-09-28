import unittest
import sys
from pathlib import Path

# Add backend to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mongo_data_filter import filter_mongodb_record, filter_mongodb_records


class MongoDataFilterTests(unittest.TestCase):
    def test_only_approved_columns_are_returned(self):
        record = {
            "_id": "mongo-id",
            "upload_batch_id": "batch-1",
            "Run Case ID": "RUN-001",
            "Test Case ID": "TC-001",
            "Test Case Name": "Login",
            "Module Name": "Authentication",
            "Status": "PASSED",
            "Error Message": "",
            "Execution Date and Time": "2026-08-20 10:00:00",
            "unexpected_field": "must be ignored",
            "debug_payload": {"secret": "must be ignored"},
        }

        filtered = filter_mongodb_record(record)

        self.assertEqual(
            set(filtered),
            {
                "run_case_id",
                "test_case_id",
                "test_case_name",
                "module_name",
                "status",
                "error_message",
                "execution_date_and_time",
            },
        )
        self.assertNotIn("unexpected_field", filtered)
        self.assertNotIn("debug_payload", filtered)
        self.assertNotIn("upload_batch_id", filtered)
        self.assertEqual(filtered["run_case_id"], "RUN-001")
        self.assertEqual(filtered["test_case_id"], "TC-001")
        self.assertEqual(filtered["test_case_name"], "Login")
        self.assertEqual(filtered["module_name"], "Authentication")

    def test_record_order_is_preserved(self):
        records = [
            {"run_id": "RUN-001", "test_id": "TC-001"},
            {"run_id": "RUN-002", "test_id": "TC-002"},
        ]
        filtered = filter_mongodb_records(records)
        self.assertEqual(filtered[0]["run_case_id"], "RUN-001")
        self.assertEqual(filtered[1]["run_case_id"], "RUN-002")

    def test_error_fields_are_normalized_before_chroma_processing(self):
        record = {
            "status": "FAILED",
            "error_message": (
                "AssertionError [ERR_ASSERTION]: Expected product count 999, but received 0\n"
                "[BROWSER CONSOLE ERROR] Failed to load resource: 401 (Unauthorized)"
            ),
        }

        filtered = filter_mongodb_record(record)

        self.assertEqual(
            filtered["error_message"],
            "Expected product count 999, but received 0",
        )
        self.assertEqual(filtered["error_category"], "ASSERTION_FAILURE")
        self.assertEqual(filtered["secondary_errors"], ["401 Unauthorized"])
        self.assertIn("AssertionError", filtered["raw_error_message"])


if __name__ == "__main__":
    unittest.main()
