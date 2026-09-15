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
Top-k percentages use those documents as the denominator.

| Method | Top 1 | Top 2 | Top 3 | Top 4 |
|---|---:|---:|---:|---:|
| Sturdy | 76.27% | 86.24% | 90.58% | 92.86% |
| BM25 | 51.77% | 68.41% | 75.86% | 80.32% |
| `intfloat/e5-small-v2` | 41.07% | 57.17% | 67.03% | 73.42% |
| OpenAI embedding, model unspecified | 60.83% | 74.27% | 80.09% | 83.10% |

### End-to-end scoring

| Generator | Judge | Metric | Sturdy | BM25 |
|---|---|---|---:|---:|
| Gemini 2.5 Flash | Claude Haiku 4.5 | Correct | 82.2939% | 77.9524% |
| Gemini 2.5 Flash | Claude Sonnet 4.5 | Mean score (1--5) | 4.3802 | 4.2085 |
| Gemini 2.5 Flash | Claude Sonnet 4 | Mean score (1--5) | 4.3523 | 4.1827 |
| GPT-4o mini | Claude Haiku 4.5 | Correct | 84.5780% | 81.3381% |
| GPT-4o mini | Claude Sonnet 4.5 | Mean score (1--5) | 4.4262 | 4.3016 |
| GPT-4o mini | Claude Sonnet 4 | Mean score (1--5) | 4.4218 | 4.2856 |

## BioASQ

- [Sturdy results](results/bioasq/sturdy.parquet)
- [BM25 results](results/bioasq/bm25.parquet)
- [E5 results](results/bioasq/e5.parquet)
- [OpenAI embedding results](results/bioasq/openai.parquet)
- Rows per file: 4,387

### Normalized exact containment

12.99% of supplied documents contain the golden answer exactly.
Top-k percentages use those documents as the denominator.

| Method | Top 1 | Top 2 | Top 3 | Top 4 |
|---|---:|---:|---:|---:|
| Sturdy | 41.75% | 55.96% | 69.12% | 76.32% |
| BM25 | 44.91% | 61.23% | 70.88% | 77.72% |
| `intfloat/e5-small-v2` | 40.35% | 58.60% | 70.00% | 76.32% |
| OpenAI embedding, model unspecified | 40.70% | 59.65% | 68.77% | 74.21% |

### End-to-end scoring

| Generator | Judge | Metric | Sturdy | BM25 |
|---|---|---|---:|---:|
| Gemini 2.5 Flash | Claude Haiku 4.5 | Correct | 71.5067% | 71.8714% |
| Gemini 2.5 Flash | Claude Sonnet 4.5 | Mean score (1--5) | 3.5700 | 3.5817 |
| Gemini 2.5 Flash | Claude Sonnet 4 | Mean score (1--5) | 3.5131 | 3.5295 |
| GPT-4o mini | Claude Haiku 4.5 | Correct | 78.7326% | 78.9150% |
| GPT-4o mini | Claude Sonnet 4.5 | Mean score (1--5) | 4.0000 | 4.0078 |
| GPT-4o mini | Claude Sonnet 4 | Mean score (1--5) | 3.9601 | 3.9713 |

Sonnet 4.5 declined 42 Gemini judgments (20 Sturdy and 22 BM25) and 56 GPT-4o-mini
judgments (29 Sturdy and 27 BM25) because of content filtering. Per the
evaluation policy, each refusal is retained as `content_filter_score_excluded`
and excluded from the 1--5 mean.

Sonnet 4 returned one unusable BioASQ response. Per the evaluation policy, it
is retained as `judge_failure_scored_incorrect` and counted as score 1.

## FinQA

- [Sturdy results](results/finqa/sturdy.parquet)
- [BM25 results](results/finqa/bm25.parquet)
- Rows per file: 6,251

### End-to-end scoring

| Generator | Judge | Metric | Sturdy | BM25 |
|---|---|---|---:|---:|
| Gemini 2.5 Flash | Claude Haiku 4.5 | Correct | 44.7288% | 43.9130% |
| Gemini 2.5 Flash | Claude Sonnet 4.5 | Mean score (1--5) | 3.1613 | 3.1235 |
| Gemini 2.5 Flash | Claude Sonnet 4 | Mean score (1--5) | 3.1184 | 3.0753 |
| GPT-4o mini | Claude Haiku 4.5 | Correct | 47.1285% | 47.2404% |
| GPT-4o mini | Claude Sonnet 4.5 | Mean score (1--5) | 3.4190 | 3.4292 |
| GPT-4o mini | Claude Sonnet 4 | Mean score (1--5) | 3.3855 | 3.3814 |

Sonnet 4 returned two unusable FinQA responses. Per the evaluation policy,
each is retained as `judge_failure_scored_incorrect` and counted as score 1.

## Reproduction

Use Python 3.11 or newer:

```bash
uv sync
uv run python scripts/run_gemini_independent_judges.py bioasq --concurrency 512
uv run python scripts/run_gemini_independent_judges.py finqa --concurrency 512
uv run python scripts/run_gpt4o_independent_judges.py --concurrency 512
uv run python scripts/run_sonnet4_judgments.py --concurrency 256
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
