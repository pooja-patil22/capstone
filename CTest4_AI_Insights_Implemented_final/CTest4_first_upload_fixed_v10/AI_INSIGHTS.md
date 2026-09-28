# AI Insights Layer

The dashboard's **AI Insights** tab is Module 6 of the Intelligent Test Report Analyser & Insights Engine.

## Flow

```text
Page 2 filters
   -> ChromaDB logical records
   -> Analytics Engine
      - Flaky Test Detector
      - Failure Pattern Detector
      - Trend Analysis
      - Heatmap Generator
   -> filtered RAG retrieval from ChromaDB
   -> structured AI Insights payload
   -> Azure OpenAI enrichment (optional)
   -> AI Insights tab
```

The AI tab is deliberately filter-aware: it receives the exact `filtered_records` produced by the Generate Report action. Historical context is supplied separately through the RAG retrieval result.

## Displayed sections

1. **AI Executive Summary** – concise assessment of the selected executions and quality score.
2. **Key metrics** – total, failed, failure rate, flaky tests, recurring patterns and RAG evidence count.
3. **Likely Root Causes** – evidence, category, hypothesis and confidence. LLM hypotheses are kept separate from observed evidence.
4. **AI Recommendations** – HIGH/MEDIUM/LOW remediation actions with evidence and expected impact.
5. **Error Intelligence** – cleaned error signatures, categories, occurrence counts, affected tests/modules and secondary HTTP errors.
6. **Module Hotspots** – failure density for the selected data.
7. **Recurring Failure Patterns** – repeated normalized signatures detected by the Analytics Engine.
8. **Flaky Test Intelligence** – top flaky tests from historical execution analysis.
9. **Trend & Risk Signals** – historical failure-rate movement when available.
10. **Historical Evidence (RAG)** – retrieved ChromaDB evidence used for the AI reasoning.

## Azure OpenAI

Set these variables in the project's `.env` file when LLM enrichment is required:

```text
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
```

If Azure OpenAI is not configured or the request fails, the tab remains functional using deterministic Analytics Engine + RAG-derived insights. The UI indicates that fallback mode was used.

## Testing

Run from the project root after installing `backend/requirements.txt`:

```bash
python -m pytest backend/tests/test_ai_insights_layer.py -q
python -m pytest backend/tests -q
```

The first test verifies that the AI payload is structured, filter-scoped, includes normalized error intelligence and preserves RAG evidence. The second verifies the existing regression suite.
