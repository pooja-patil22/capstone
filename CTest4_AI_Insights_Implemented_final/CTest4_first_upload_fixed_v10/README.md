# Intelligent Test Report Analyzer & Insights Engine — Duplicate-Protected

## Duplicate protection

This version prevents duplicate uploads at two levels:

1. **Exact file duplicate:** the application calculates a SHA-256 hash of the complete uploaded file. Uploading the same file again is rejected, even if the filename is changed.
2. **Duplicate filename:** if the filename already exists in the MongoDB test collection, the upload is rejected.
3. **Duplicate rows inside a file:** exact duplicate rows are removed before MongoDB insertion. The UI tells you how many duplicate rows were removed.
4. **MongoDB registry:** every successful upload is recorded in the `upload_registry` collection with a unique `file_hash`. MongoDB enforces uniqueness on that hash, preventing race-condition duplicates.
5. **ChromaDB:** MongoDB `_id` remains the ChromaDB ID, and ChromaDB uses `upsert`, so the same MongoDB record cannot create another vector with a different ID.

## Upload flow

`File → duplicate check → remove duplicate rows → MongoDB → ChromaDB → Analytics Engine → analytics_results/`

The analytics result filename remains the input filename with only the extension changed to `.json`.

Example:
`qa_test_report.csv` → analytics report held in memory and persisted to MongoDB (no JSON file is generated)

## Important

The first upload made with this version creates the `upload_registry` collection automatically.

Older MongoDB records created by a previous version may not have `file_hash`. The application still blocks reuse of an existing filename. For complete historical cleanup, remove old duplicate data once before starting fresh.

## Run

1. Start MongoDB.
2. Open this folder in VS code.
3. Create and configure a local virtual environment:

```bash
setup_venv.bat
```

This creates `.venv` inside the project, upgrades `pip`, and installs all dependencies from `requirements.txt` into that environment.

4. Activate the virtual environment:

```bash
.venv\Scripts\activate
```

5. Run:

```bash
streamlit run app.py
```

Or double-click `run.bat`.

## Dependency management in .venv

- `requirements.txt` contains exact pinned versions from the project virtual environment.
- After adding/updating packages in `.venv`, refresh pins with:

```bash
lock_requirements.bat
```

This keeps all dependencies managed inside the project-local virtual environment.

## Expected duplicate behavior

- Upload `Report1.csv` → accepted.
- Upload `Report1.csv` again → rejected.
- Rename the exact same file to `Report1_New.csv` and upload → rejected because the file content hash is the same.
- Upload a genuinely different report with a new filename → accepted.
- If one uploaded file contains repeated identical rows → repeated rows are removed before insertion.


## Duplicate upload message
If the exact same file is uploaded again, the application now shows only:

`Upload failed: this file is already in the system. Please choose a different file.`

The internal exception, batch ID, filename details, and traceback are hidden from the user for this case.


## Supported Test Report Formats

The upload pipeline automatically detects the uploaded report format and filters it into a common test-execution schema before database storage.

- **CSV / Excel** — detects fields such as `test_name`, `test_id`, `status`, `module`, `error`, `duration`, `execution_time`, `version`, and `environment`.
- **JUnit XML** — extracts each `<testcase>` and maps pass/fail/skipped status, class/module, failure/error message, and duration.
- **Allure JSON / ZIP** — extracts Allure `*-result.json` files, including test name, history/UUID, status, suite/feature, status details, timestamps, and duration.
- **ExtentReports HTML** — extracts test names and execution status from common ExtentReports HTML structures.

All normalized records use the same schema, so MongoDB, ChromaDB and the Analytics Engine work independently of the original report format.

**Recommended Allure upload:** ZIP the complete Allure `results` folder and upload the ZIP file.

### Module 2 – Report Ingestion Layer

The upload flow now follows the architecture diagram and uses dedicated parsers:

```text
User Upload
    │
    ▼
Report Type Detection
    ├── JUnit XML  ──> app/parsers/junit_parser.py
    ├── Allure     ──> app/parsers/allure_parser.py
    └── Extent     ──> app/parsers/extent_parser.py
                         │
                         ▼
                 Common Data Model
                         │
                         ▼
                  MongoDB → ChromaDB
```

`app/parsers/report_parser.py` is the Module-2 router. It detects the report type and sends the file to the correct parser before the data reaches MongoDB. This keeps report-specific extraction logic separate from database and analytics logic.

Supported Module-2 report formats:
- JUnit XML (`.xml`)
- Allure result JSON (`.json`)
- Allure results ZIP (`.zip`, containing `*-result.json`)
- ExtentReports HTML (`.html`, `.htm`)
- CSV / Excel continue to use the common tabular normalization path.

All parser outputs use the same normalized fields: `test_id`, `test_name`, `status`, `module`, `error_message`, `execution_time`, `duration`, `version`, `environment`, and `source_format`.


## Vector Processing – Module 5

After MongoDB successfully stores an upload, the existing UI and MongoDB flow remain unchanged. The application then performs:

```text
MongoDB
  ↓
Text preprocessing
  ↓
Chunking (500 words, 50-word overlap)
  ↓
Embedding generation (all-MiniLM-L6-v2)
  ↓
Namespace assignment
  ↓
ChromaDB Persistent Collection
```

Each MongoDB record is converted to searchable text and split into overlapping chunks. Every chunk receives a dense embedding and is stored in the configured ChromaDB collection.

Each vector stores metadata including:
- `namespace`
- `embedding_model`
- `chunk_index`
- `chunk_count`
- `mongo_record_id`
- the original MongoDB fields

The namespace is stable for an uploaded file and is based on the configured ChromaDB collection plus the upload's SHA-256 file hash.

Install the new dependency with:

```bash
pip install -r requirements.txt
```

The first embedding run may download the `all-MiniLM-L6-v2` model from Hugging Face and cache it locally.

## Module 5 + Module 6 — Analytics Retrieval, RAG and Azure OpenAI

The post-ingestion architecture is now:

```text
MongoDB → ChromaDB
             │
             ▼
     Module 5 Analytics Engine
             │
      ┌──────┼───────────────┐
      ▼      ▼               ▼
   Metadata  Embeddings   Filtered vectors
      │                       │
      ├── Flaky Test Detector
      ├── Failure Pattern Detector
      ├── Trend Analysis
      └── Heatmap Generator
                              │
                              ▼
                    Module 6 RAG Retrieval
                              │
                 relevant chunks + metadata
                              │
                              ▼
                       Azure OpenAI
                              │
                              ▼
                     AI Insights output
```

### Module 5 behavior

The ingestion pipeline before ChromaDB is unchanged. After vectors are persisted, `analytics_engine.py` reads the logical test executions directly from ChromaDB metadata. Chunk-level vectors are collapsed by `mongo_record_id` for deterministic analytics.

The four requested analytics categories are calculated from the ChromaDB-derived records:
- Flaky Test Detector
- Failure Pattern Detector
- Trend Analysis
- Heatmap Generator

The generated upload JSON now records ChromaDB as the analytics retrieval source.

### Module 6 behavior

When the existing Page 2 **Generate Report** action is used, the backend sends the existing filter selections to the Analytics Engine. The Analytics Engine:
1. Retrieves the complete logical record set from ChromaDB metadata.
2. Applies the dashboard filter semantics to that ChromaDB-derived data.
3. Recomputes the four analytics categories for the selected records.
4. Runs semantic RAG retrieval over ChromaDB using the filter-aware query.
5. Retrieves document chunks, metadata, distances and embeddings.
6. Stores that RAG context in the existing dashboard session state.
7. The AI Insights section passes the filtered analytics plus retrieved evidence to Azure OpenAI.

The semantic top-k RAG result is **context only**; it never replaces the complete filtered dashboard dataset.

### Azure OpenAI configuration

Install the added dependency:

```bash
pip install -r requirements.txt
```

Configure:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT_NAME
```

A template is included in `.env.azure-openai.example`.

The application does not load or display an API key in the UI. If Azure OpenAI is not configured or temporarily unavailable, the existing deterministic analytics-based insights remain available.

### Architecture boundary

No parser, upload, duplicate handling, MongoDB insertion, or MongoDB → ChromaDB ingestion logic was changed for Module 5/6. The dashboard now generates the filtered report when **Generate Report** is clicked. The Overview shows the requested Quality Overview KPIs without Duplicate Executions or Average Duration, uses two analytics graphs per row, and no longer includes the Overview AI Insights section. The Duplicates, Duration, and Test Results tabs were removed; AI Insights remains available as its own tab.



## Azure OpenAI configuration (Module 6)

The backend automatically loads a local `.env` file using `python-dotenv`. Copy `.env.azure-openai.example` to `.env` in this project root and fill in the Azure OpenAI values:

```env
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<azure-openai-api-key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT_NAME=<chat-model-deployment-name>
```

The `.env` file is ignored by Git. Do not commit or share the API key.


### Analytics report storage

The Analytics Engine does not generate or write a JSON report file to `analytics_results/`. Analytics results are persisted to the MongoDB `analytics_results` collection and kept in memory for the dashboard/RAG/LLM flow.
## ChromaDB Human-Readable Inspection

The project does not expose a ChromaDB viewer in the Streamlit frontend. After a successful MongoDB → ChromaDB ingestion, the pipeline writes a human-readable snapshot to `backend/CHROMADB_DATA.md`.

- `backend/CHROMADB_DATA.md` — easy-to-read chunks, vector IDs, metadata, collection, embedding model, and chunking information.
- `backend/chroma_db/` — the actual persistent ChromaDB storage; do not edit its internal files manually.

`backend/CHROMADB_DATA.md` is refreshed after each successful ingestion and is rebuilt from the complete current ChromaDB collection. For example, after two successful uploads of 10 records each, it reports all 20 persisted vectors rather than only the latest 10-record batch.


## Error normalization

Error messages are normalized before vectorization. The project preserves `raw_error_message` for MongoDB debugging while using the cleaned `error_message` and `error_signature` for ChromaDB embeddings. See `ERROR_NORMALIZATION.md` for the field contract, supported report formats, examples, and test commands.

## AI Insights Layer

The dashboard's **AI Insights** tab is implemented as a filter-aware Module 6 layer over Analytics Engine + ChromaDB RAG. It provides executive summary, root-cause hypotheses, actionable recommendations, error intelligence, flaky-test interpretation, module hotspots, trends and historical RAG evidence. See `AI_INSIGHTS.md` for the data flow, Azure OpenAI configuration and tests.
