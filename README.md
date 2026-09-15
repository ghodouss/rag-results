# Sturdy RAG evaluation

Analysis-ready results are ten Parquet files. Each row is one question and
contains the question, reference answer, ranked retrievals, and rendered
context. ContractNLI and BioASQ files include normalized exact-containment
flags. Sturdy and BM25 files also contain all generated answers and judgments.
Model-specific generation and judgment columns are namespaced by model.

```text
results/
  contractnli/{sturdy,bm25,e5,openai}.parquet
  bioasq/{sturdy,bm25,e5,openai}.parquet
  finqa/{sturdy,bm25}.parquet
```

Load a result directly with pandas:

```python
import pandas as pd

sturdy = pd.read_parquet("results/contractnli/sturdy.parquet")
```

## ContractNLI

- [Sturdy results](results/contractnli/sturdy.parquet)
- [BM25 results](results/contractnli/bm25.parquet)
- [E5 results](results/contractnli/e5.parquet)
- [OpenAI embedding results](results/contractnli/openai.parquet)
- Rows per file: 6,173

### Normalized exact containment

62.40% of supplied documents contain the golden reference evidence span exactly.

| Method | Top 1 | Top 2 | Top 3 | Top 4 |
|---|---:|---:|---:|---:|
| Sturdy | 47.59% | 53.82% | 56.52% | 57.95% |
| BM25 | 32.30% | 42.69% | 47.34% | 50.12% |
| `intfloat/e5-small-v2` | 25.63% | 35.67% | 41.83% | 45.81% |
| OpenAI embedding, model unspecified | 37.96% | 46.35% | 49.98% | 51.85% |

### End-to-end scoring

| Generator | Judge | Metric | Sturdy | BM25 |
|---|---|---|---:|---:|
| Gemini 2.5 Flash | Claude Haiku 4.5 | Correct | 82.2939% | 77.9524% |
| Gemini 2.5 Flash | Claude Sonnet 4.5 | Mean score (1--5) | 4.3802 | 4.2085 |
| GPT-4o mini | Claude Haiku 4.5 | Correct | 84.5780% | 81.3381% |
| GPT-4o mini | Claude Sonnet 4.5 | Mean score (1--5) | 4.4262 | 4.3016 |

## BioASQ

- [Sturdy results](results/bioasq/sturdy.parquet)
- [BM25 results](results/bioasq/bm25.parquet)
- [E5 results](results/bioasq/e5.parquet)
- [OpenAI embedding results](results/bioasq/openai.parquet)
- Rows per file: 4,387

### Normalized exact containment

12.99% of supplied documents contain the golden answer exactly.

| Method | Top 1 | Top 2 | Top 3 | Top 4 |
|---|---:|---:|---:|---:|
| Sturdy | 5.43% | 7.27% | 8.98% | 9.92% |
| BM25 | 5.84% | 7.96% | 9.21% | 10.10% |
| `intfloat/e5-small-v2` | 5.24% | 7.61% | 9.10% | 9.92% |
| OpenAI embedding, model unspecified | 5.29% | 7.75% | 8.94% | 9.64% |

### End-to-end scoring

| Generator and judge | Sturdy | BM25 |
|---|---:|---:|
| GPT-4o mini | 74.1965% | 74.4472% |
| GPT-5.6 Luna, low | 84.7276% | 84.2261% |

## FinQA

- [Sturdy results](results/finqa/sturdy.parquet)
- [BM25 results](results/finqa/bm25.parquet)
- Rows per file: 6,251

### End-to-end scoring

| Generator and judge | Sturdy | BM25 |
|---|---:|---:|
| GPT-4o mini | 40.9694% | 40.6975% |
| GPT-5.6 Luna, low | 75.6199% | 75.9558% |

## Reproduction

Use Python 3.11 or newer:

```bash
uv sync
uv run python scripts/build_result_parquets.py
uv run python scripts/summarize_results.py
uv run python scripts/validate_package.py
uv run python scripts/write_manifest.py --check
```

Dataset entry points are grouped under `scripts/contractnli/`,
`scripts/bioasq/`, and `scripts/finqa/`. Each has `retrieve_sturdy.py`,
`retrieve_bm25.py`, `generate.py`, and `judge.py`; ContractNLI and BioASQ also
have `exact.py`. The detailed resumable run artifacts supplying the E2E columns
for the six Sturdy/BM25 files are under `artifacts/e2e/`.

ContractNLI Sturdy retrieval uses
`index.search(level="sentence", context=2, limit=4, ...)` and deduplicates
overlap across ranked windows. Sturdy retrieval requires `STURDY_ORG_ID` and
`STURDY_API_KEY`. Generation and judging use the provider variables expected by
the runners, including `OPENAI_API_KEY` and `OPENROUTER_API_KEY`. Credentials
remain external.

`scripts/summarize_results.py` computes the tables above directly from the ten
Parquets. `MANIFEST.json` records every distributed file's size, hash, and
Parquet dimensions. Source lineage and exact prompt text are documented in
`provenance/SOURCES.md`.

## Publication

No license has been selected and this repository is not approved for public
publication. Do not publish or change visibility until licenses are reviewed.
