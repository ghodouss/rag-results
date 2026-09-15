# Sources and evaluation provenance

This file records the lineage needed to interpret or reproduce the packaged
results. Paths in older provenance records identify the original local source;
the corresponding portable files now live under `data/`, `artifacts/`, and
`results/`.

## Dataset cohorts

| Dataset | Cohort | Splits | Source documents | Retrieval unit |
|---|---:|---|---:|---|
| ContractNLI | 6,173 | 4,272 dev; 1,901 test | 607 | sentence windows for final Sturdy E2E; subparagraphs for BM25 |
| BioASQ | 4,387 | 3,073 dev; 1,314 test | 28,001 | document |
| FinQA | 6,251 | 4,282 dev; 1,969 test | 2,110 | paragraph |

Every retrieval is restricted to documents supplied with that question.
BioASQ qrels use stable source-row identity to avoid attaching evidence from a
different record when question text repeats. Its Sturdy result retains 23
questions with no ranked result rather than silently dropping them. FinQA has
complete Top-4 retrieval rows with no retrieval errors, but no literal-answer
containment or qrel result is reported because answers often require arithmetic.

## Sturdy retrieval

The shared retrieval configuration is recorded in `config/datasets.json`.
BioASQ uses document-level index
`ix-01a05ef3-63f0-724e-9b01-566fa4426ea6`
(`bio-asq-v5-improved`). FinQA uses paragraph-level index
`ix-01a07d68-810e-7222-8e82-efbc498aee94`
(`search-lab-deadline-finqa-v20k-k192`). Both use Top 4, semantic-search weight
0.30, cutoff 0.00001, and a strict supplied-document filter.

ContractNLI retrieval uses four ranked Sturdy sentence anchors with two
neighboring sentences on either side. Overlaps are deduplicated by retrieval
rank and context never crosses a document boundary. Literal containment uses a
canonical source-order projection of each cumulative retrieved set; generation
uses ranked-window rendering. The packaged inputs are:

- `data/contractnli/sentence-source/sentence-anchors.parquet`
- `data/contractnli/sentence-source/sentence-snapshot.parquet`
- `data/contractnli/retrieval/sturdy-a4-r2.part-00000.parquet`
- `data/contractnli/retrieval/sturdy-a4-r2.part-00001.parquet`

`scripts/contractnli/retrieve_sturdy.py` reproduces retrieval through the SDK.
The anchors and sentence snapshot are retained to audit the packaged contexts.

## Baselines

BM25 is Robertson BM25 with `k1=1.2`, `b=0.75`, global retrieval-unit document
frequency, deterministic lowercase alphanumeric tokenization, and strict
candidate filtering. `src/rag_eval/bm25.py` and `scripts/retrieve_bm25.py`
implement it.

The E5 condition is identified as `intfloat/e5-small-v2`. The supplied archive
did not record the model behind its OpenAI embedding condition; the canonical
identifier is therefore `openai/embedding-model-unspecified`. The importer
preserves that uncertainty and no OpenAI embedding model name should be inferred.
The portable normalized retrieval files are under
`data/<dataset>/retrieval/`; `scripts/import_neural_results.py` documents the
archive schema and `scripts/score_neural_results.py` performs exact normalized
containment scoring. Only the rows matching the packaged question cohorts are
used in the README comparison.

## ContractNLI generation and independent judging

The complete paired matrix is
`contractnli-full-dual-label-sturdy-bm25-6173-v1`, with 6,173 questions for each
of `sturdy-a4-r2-ranked-windows` and `bm25-top4`. Generators are
`google/gemini-2.5-flash` and `openai/gpt-4o-mini`, called through OpenRouter.

Generator prompt version: `supplied-generic-dual-label-v1`.

```text
You are a helpful assistant. Answer the question based only on the provided context. If the context does not contain enough information to answer, say so. Be concise.

Answer True/Entailment or False/Contradiction, followed by a brief explanation.
```

Generator user-message template:

```text
Context:
{context}

Question:
{question}
```

The independent binary judge is `anthropic/claude-haiku-4.5`, prompt version
`supplied-binary-v1`. It receives the question, golden answer, and predicted
answer and returns only structured JSON `{"correct": true|false}`. The
independent scale judge is `anthropic/claude-sonnet-4.5`, prompt version
`supplied-scale-1-5-v1`. It uses the supplied 1--5 meaning-match rubric and
returns structured JSON with integer `score` and a one-sentence `rationale`.
The complete byte-level prompt text, user templates, and JSON schemas are stored
in `artifacts/e2e/contractnli/provenance.json` and implemented by
`scripts/run_contractnli_full_dual_label_matrix.py` together with
`scripts/run_contractnli_full_prompt_matrix.py`.

All ContractNLI summaries expose only those independent judge outputs. The
internal deterministic mapping of the dual label is an integrity aid, not a
public evaluation metric, and is intentionally absent from public summaries.

## BioASQ and FinQA generation and judging

Both datasets use generator prompt version `grounded-answer-v1`:

```text
You are a question-answering assistant.
Answer the question using only the supplied context. Be concise. If the answer
is a number, return just the number. Otherwise answer in one or two sentences.
If the context is insufficient, say that the answer cannot be determined from
the supplied context.
```

The user message is `Context:\n{top-four excerpts}\n\nQuestion:
{question}\n\nAnswer:`. Generators are `gpt-4o-mini` and `gpt-5.6-luna`;
Luna uses reasoning effort `low`, while GPT-4o mini records
`not_supported`.

Judgment prompt version `reference-answer-judge-v1` asks whether the candidate
adequately answers the question relative to the reference answer, allows meaning
equivalence without exact wording, and emits structured `CORRECT` or `INCORRECT`
plus a reason. Each output was judged by the same model that generated it. These
complete same-model diagnostics are useful, but they are not independent
adjudication. Exact prompts and layouts are implemented in
`scripts/generate_answers.py` and `scripts/judge_answers.py`.

## Reproduction and validation map

- Prepare portable question/document data: `scripts/prepare_data.py`.
- Export the exact indexed corpus: `scripts/export_index_corpus.py`.
- Build ContractNLI/BioASQ evidence qrels: `scripts/build_retrieval_qrels.py`.
- Rebuild retrieval with each dataset's `retrieve_bm25.py` and
  `retrieve_sturdy.py` entry points under `scripts/<dataset>/`.
- Combine and score canonical retrieval: `scripts/build_handoff.py`.
- Import supplied embedding conditions with `scripts/import_neural_results.py`.
- Score exact containment with `scripts/contractnli/exact.py` and
  `scripts/bioasq/exact.py`.
- Run/resume BioASQ and FinQA E2E: `scripts/run_llm_eval_suite.py`.
- Run/resume ContractNLI E2E with `scripts/contractnli/generate.py` and
  `scripts/contractnli/judge.py`.
- Validate identities, coverage, errors, summaries, and hosting limits with
  `scripts/validate_package.py`.
- Regenerate or check file hashes and dimensions: `scripts/write_manifest.py`.

The package intentionally ships a lightweight artifact validator rather than
the development test suite used while constructing the results.

## Secrets and release status

Sturdy API credentials are supplied at runtime through `STURDY_ORG_ID` and
`STURDY_API_KEY`. Generation and judging credentials remain in provider-specific
environment variables, including `OPENAI_API_KEY` and `OPENROUTER_API_KEY`.
No credential or DEK is part of this repository.

No license has been selected, dataset redistribution has not been cleared here,
and the repository has not been approved for public publication. Review source
dataset licenses and choose a repository license before publishing or changing
visibility.
