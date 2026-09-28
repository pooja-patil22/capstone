import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Importing app.py starts Streamlit, so extract the helper without importing the UI.
app_source = (ROOT.parent.parent / "frontend" / "app.py").read_text(encoding="utf-8")
start = app_source.index("def build_namespace(")
end = app_source.index("\n\ndef _build_current_chromadb_snapshot_records", start)
namespace_code = app_source[start:end]

exec("from pathlib import Path\n" + namespace_code, globals())

def test_namespace_uses_uploaded_file_name_without_extension():
    assert build_namespace("test_execution_history", "results_Aug20.html") == (
        "test_execution_history::results_Aug20"
    )

def test_namespace_uses_file_stem_for_path_like_name():
    assert build_namespace("test_execution_history", "reports/results_Aug20.json") == (
        "test_execution_history::results_Aug20"
    )
