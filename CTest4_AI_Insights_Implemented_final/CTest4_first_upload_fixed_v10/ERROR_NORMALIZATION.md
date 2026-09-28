# Error Normalization / Cleaning

The ingestion pipeline now normalizes report errors before MongoDB-to-ChromaDB chunking/embedding.

## Data flow

`Cucumber/JUnit/Allure/Extent parser -> Error Normalizer -> MongoDB -> seven-field filter + derived error metadata -> chunking -> embedding -> ChromaDB -> Analytics/RAG/LLM`

### Stored fields

For every parsed error, the record contains:

- `raw_error_message` — original parser output, retained for MongoDB debugging.
- `error_message` — cleaned primary failure message.
- `error_category` — `TIMEOUT`, `ASSERTION_FAILURE`, `ELEMENT_NOT_FOUND`, `AUTHENTICATION`, `AUTHORIZATION`, `RESOURCE_NOT_FOUND`, `SERVER_ERROR`, or `OTHER`.
- `error_signature` — deterministic normalized form with dynamic values replaced by tokens such as `N`, `ID`, `TIMESTAMP`, and `URL`.
- `secondary_errors` — important secondary HTTP errors such as `401 Unauthorized`, `403 Forbidden`, and `404 Not Found`.

## Embedding behavior

`raw_error_message` is **never included in the searchable embedding document**. The embedding text uses the cleaned `error_message` plus compact derived signals (`error_category`, `error_signature`, and `secondary_errors`). This prevents stack traces, filesystem paths, Node/Cucumber internals, and browser-console noise from dominating semantic retrieval.

The raw error remains in MongoDB as part of the uploaded source record. ChromaDB metadata retains the derived error fields and source-routing metadata for retrieval/analytics.

## Supported reports

The normalization is applied by the shared `canonical_test_record()` path, so it covers:

- Cucumber-style JSON/generic JSON reports
- JUnit XML
- Allure JSON and Allure ZIP result files
- ExtentReports HTML
- Generic supported tabular report formats that use the common parser path

## Run tests

From the project root:

```text
PYTHONPATH=.:backend pytest -q backend/tests/test_error_normalizer.py backend/tests/test_error_message_sanitization.py backend/tests/test_report_parsers.py backend/tests/test_mongo_data_filter.py
```

The focused backend regression suite currently passes with **23 tests** (including the existing ChromaDB failure-safety, namespace, snapshot, parser, and Mongo filter tests).

For the complete project suite, install the dependencies from `backend/requirements.txt` first and then run:

```text
PYTHONPATH=.:backend pytest -q backend/tests
```

## Example

Input:

```text
Error: function timed out, ensure the promise resolves within 5000 milliseconds
    at World.<anonymous> (C:\project\features\steps\checkout.js:42:17)
    at async runStep (node_modules/@cucumber/cucumber/src/runtime.ts:101:9)
```

Stored/embedded primary error:

```text
function timed out, ensure the promise resolves within 5000 milliseconds
```

Signature:

```text
function timed out ensure the promise resolves within N milliseconds
```

Framework-specific errors are reduced to stable messages when the framework
adds mostly diagnostic wording. For example:

```text
TimeoutError: locator.click: Timeout 30000ms exceeded
```

becomes:

```text
Element interaction timed out
```

A browser-console `401 Unauthorized`, `403 Forbidden`, or `404 Not Found` found alongside the primary error is preserved in `secondary_errors` and can influence categorization when the primary error itself does not provide a category.
