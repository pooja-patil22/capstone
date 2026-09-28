from .report_parser import detect_report_type, parse_uploaded_report
from .junit_parser import parse_junit_xml
from .allure_parser import parse_allure_json, parse_allure_zip
from .extent_parser import parse_extent_html

__all__ = [
    "detect_report_type",
    "parse_uploaded_report",
    "parse_junit_xml",
    "parse_allure_json",
    "parse_allure_zip",
    "parse_extent_html",
]
