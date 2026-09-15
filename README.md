# Sturdy RAG evaluation

This repository packages reproducible, supplied-document-filtered retrieval and
answer evaluations for ContractNLI, BioASQ, and FinQA. Tables contain raw method
results. Packaged files are listed in `MANIFEST.json`; source lineage is in
`provenance/SOURCES.md`.

## ContractNLI

ContractNLI contains 6,173 questions. `Sturdy` means four ranked sentence
matches, each expanded with two neighboring sentences on either side. Overlap is
retained only in the first ranked window where it occurs. BM25 ranks four
packaged subparagraphs. Both methods are restricted to the supplied contract.

### Exact containment

| Method | Top 1 | Top 2 | Top 3 | Top 4 | Full docs | Top 4 when possible |
|---|---:|---:|---:|---:|---:|---:|
| Sturdy | 47.59% | 53.82% | 56.52% | 57.95% | 62.40% | 92.86% |
| BM25 | 32.30% | 42.69% | 47.34% | 50.12% | 62.40% | 80.32% |
| `intfloat/e5-small-v2` | 25.63% | 35.67% | 41.83% | 45.81% | 62.40% | 73.42% |
| OpenAI embedding, model unspecified | 37.96% | 46.35% | 49.98% | 51.85% | 62.40% | 83.10% |

Containment Unicode-normalizes and lowercases text, retains alphanumeric tokens,
normalizes whitespace, and tests whether the complete normalized golden evidence
appears in cumulative Top-k context. `Top 4 when possible` divides Top-4
containment by the full-document ceiling. The supplied archive does not identify
the OpenAI embedding model, so no model name is inferred.

### End-to-end answers

The supplied generic grounded-answer prompt has this appended instruction:

```text
Answer True/Entailment or False/Contradiction, followed by a brief explanation.
```

| Generator | Judge | Metric | Sturdy | BM25 |
|---|---|---|---:|---:|
| Gemini 2.5 Flash | Claude Haiku 4.5 | Correct | 82.2939% | 77.9524% |
| Gemini 2.5 Flash | Claude Sonnet 4.5 | Mean score (1--5) | 4.3802 | 4.2085 |
| GPT-4o mini | Claude Haiku 4.5 | Correct | 84.5780% | 81.3381% |
| GPT-4o mini | Claude Sonnet 4.5 | Mean score (1--5) | 4.4262 | 4.3016 |

Every cell contains 6,173 aligned rows with zero final errors. Haiku is an
independent boolean judge and Sonnet an independent 1--5 judge. Score
distributions and paired counts are under `results/e2e/contractnli/paired/`.
Public summaries do not report a deterministic label-parser metric.

## BioASQ

BioASQ contains 4,387 packaged questions. Retrieval is restricted to supplied
document IDs. The Sturdy denominator retains 23 empty/error queries.

### Exact containment

| Method | Top 1 | Top 2 | Top 3 | Top 4 | Full docs | Top 4 when possible |
|---|---:|---:|---:|---:|---:|---:|
| Sturdy | 5.43% | 7.27% | 8.98% | 9.92% | 12.99% | 76.32% |
| BM25 | 5.84% | 7.96% | 9.21% | 10.10% | 12.99% | 77.72% |
| `intfloat/e5-small-v2` | 5.24% | 7.61% | 9.10% | 9.92% | 12.99% | 76.32% |
| OpenAI embedding, model unspecified | 5.29% | 7.75% | 8.94% | 9.64% | 12.99% | 74.21% |

### End-to-end answers

| Generator and judge | Sturdy | BM25 |
|---|---:|---:|
| GPT-4o mini | 74.1965% | 74.4472% |
| GPT-5.6 Luna, low | 84.7276% | 84.2261% |

These runs use the same model for generation and judging and have zero final
errors. They are diagnostics, not independent cross-model adjudication.

## FinQA

FinQA contains 6,251 questions. Retrieval is restricted to the supplied
financial document. Literal containment is not applicable because answers often
require arithmetic over text or tables.

### End-to-end answers

| Generator and judge | Sturdy | BM25 |
|---|---:|---:|
| GPT-4o mini | 40.9694% | 40.6975% |
| GPT-5.6 Luna, low | 75.6199% | 75.9558% |

These runs use the same model for generation and judging and have zero errors.

## Reproduction

Use Python 3.11 or newer:

```bash
uv sync
uv run python scripts/validate_package.py
uv run python scripts/write_manifest.py --check
```

Dataset entry points are grouped under `scripts/contractnli/`,
`scripts/bioasq/`, and `scripts/finqa/`. Each contains `retrieve_sturdy.py`,
`retrieve_bm25.py`, `generate.py`, and `judge.py`; ContractNLI and BioASQ also
contain `exact.py`.

ContractNLI uses `index.search(level="sentence", context=2, limit=4, ...)`.
The SDK defines `context=2` as two neighboring units on each side, in source
order, without crossing a document boundary. The SDK does not promise overlap
deduplication, so `scripts/contractnli/retrieve_sturdy.py` performs cumulative
deduplication across ranked windows.

```bash
uv run python scripts/contractnli/retrieve_sturdy.py
uv run python scripts/contractnli/retrieve_bm25.py
uv run python scripts/contractnli/exact.py
bin/run-contractnli-full-dual-label-matrix

uv run python scripts/bioasq/retrieve_sturdy.py
uv run python scripts/bioasq/retrieve_bm25.py
uv run python scripts/bioasq/exact.py
```

Sturdy retrieval requires `STURDY_ORG_ID` and `STURDY_API_KEY`. API generation
and judging use the provider variables expected by the runners, including
`OPENAI_API_KEY` and `OPENROUTER_API_KEY`. Credentials remain external.

ContractNLI Sturdy retrieval is split into two Parquet parts solely to keep
files below common hosting limits. Local checkpoint copies are ignored and do
not appear in `MANIFEST.json`.

## Publication

No license has been selected and this repository is not approved for public
publication. Do not publish or change visibility until licenses are reviewed.
