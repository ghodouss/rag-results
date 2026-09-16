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
The ContractNLI OpenAI retrieval is a pure OpenAI-embedding run and is
normalized to this canonical identifier before packaging.
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
independent scale judges are `anthropic/claude-sonnet-4.5` and
`anthropic/claude-sonnet-4`, both using prompt version
`supplied-scale-1-5-v1`. They use the supplied 1--5 meaning-match rubric and
return JSON with integer `score` and a one-sentence `rationale`.
The complete byte-level prompt text, user templates, and JSON schemas are stored
in `artifacts/e2e/contractnli/provenance.json` and implemented by
`scripts/run_contractnli_full_dual_label_matrix.py` together with
`scripts/run_contractnli_full_prompt_matrix.py`.

All ContractNLI summaries expose only those independent judge outputs. The
internal deterministic mapping of the dual label is an integrity aid, not a
public evaluation metric, and is intentionally absent from public summaries.

The E5 and OpenAI retrievals use the same 6,173-question
dual-label prompt and the same Gemini 2.5 Flash and GPT-4o-mini generators.
They are independently judged by Haiku 4.5 binary and Sonnet 4 on the 1--5
scale. All 24,692 embedding-retrieval generations and 49,384 judgments are
complete with no final errors or fallbacks. Their resumable artifacts are under
`artifacts/e2e/embedding-retrievals/contractnli/`.

## BioASQ and FinQA generation and judging

The final independent-judge matrices use `google/gemini-2.5-flash` through
OpenRouter with generator prompt version `grounded-answer-v2-openrouter`, plus
the existing direct-OpenAI `gpt-4o-mini` generations with prompt version
`grounded-answer-v1`. The prompt text is identical:

```text
You are a question-answering assistant.
Answer the question using only the supplied context. Be concise. If the answer
is a number, return just the number. Otherwise answer in one or two sentences.
If the context is insufficient, say that the answer cannot be determined from
the supplied context.
```

The user message is `Context:\n{top-four excerpts}\n\nQuestion:
{question}\n\nAnswer:`. Independent binary judgments use
`anthropic/claude-haiku-4.5` and prompt version
`reference-answer-binary-judge-v2-openrouter`; independent 1--5 judgments use
both `anthropic/claude-sonnet-4.5` and `anthropic/claude-sonnet-4` with prompt
version `reference-answer-scale-1-5-judge-v1-openrouter`. The judges receive the
question, reference answer, and generated answer through OpenRouter. Haiku 4.5
and Sonnet 4.5 use provider-enforced JSON schemas. Sonnet 4 uses the identical
JSON-only prompt with local schema validation because its available OpenRouter
route does not accept the structured-output parameter. A binary result is
`CORRECT` or `INCORRECT`; the scale result is an integer score from 1 through 5
plus a concise reason.

For the E5 and OpenAI retrieval conditions, both Gemini 2.5 Flash and
`openai/gpt-4o-mini` were regenerated through OpenRouter with
`grounded-answer-v2-openrouter`. Haiku 4.5 supplies the binary verdict and
Sonnet 4 supplies the 1--5 score. BioASQ contains 17,548 such generations and
35,096 judgments; FinQA contains 25,004 generations and 50,008 judgments.
Every embedding-retrieval generation and judgment completed without a final
error or fallback. These artifacts live under
`artifacts/e2e/embedding-retrievals/{bioasq,finqa}/`.

The BioASQ Sonnet 4.5 provider declined 42 Gemini judgments and 56 GPT-4o-mini
judgments with `content_filter` on benign benchmark questions involving
pathogens, vaccines, or toxins. At the user's direction, each refusal is recorded as
`content_filter_score_excluded` with a null score and excluded from the mean.
No FinQA judgment used this fallback. Older GPT-4o-mini and GPT-5.6-Luna
same-model judgments remain under
`artifacts/e2e/` as historical diagnostics and are not reported as the final
independent-judge results.

Sonnet 4 produced three unusable responses after validation retries: one
BioASQ judgment and two FinQA judgments. At the user's direction, each is
recorded as `judge_failure_scored_incorrect` and counted as score 1. All other
Sonnet 4 judgments contain validated scores from 1 through 5.

Judgment prompt version `reference-answer-judge-v1` asks whether the candidate
adequately answers the question relative to the reference answer, allows meaning
equivalence without exact wording, and emits structured `CORRECT` or `INCORRECT`
plus a reason. Each output was judged by the same model that generated it. These
complete same-model diagnostics are useful, but they are not independent
adjudication. Their artifacts retain the historical prompt-version metadata.

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
- Run/resume the final BioASQ and FinQA independent-judge E2E matrices:
  `scripts/run_gemini_independent_judges.py`.
- Run/resume Haiku and Sonnet judging of the existing GPT-4o-mini generations:
  `scripts/run_gpt4o_independent_judges.py`.
- Run/resume ContractNLI E2E with `scripts/contractnli/generate.py` and
  `scripts/contractnli/judge.py`.
- Run/resume the complete E5/OpenAI embedding E2E matrix with
  `scripts/run_embedding_e2e_matrix.py`.
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
