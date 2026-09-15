#!/usr/bin/env python3
"""Run the approved full ContractNLI Sturdy-versus-BM25 prompt matrix."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_sentence_ordering_inputs as ordering
import run_contractnli_prompt_pilot as pilot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet
from rag_eval.llm import (
    client,
    completion_text,
    json_object_text,
    model_slug,
    response_route,
    text_sha256,
    usage_fields,
    with_retries,
)


RESULTS_ROOT = ROOT / "artifacts" / "experiments" / "contractnli-full-true-false-matrix-6173"
QUESTIONS_SOURCE = ROOT / "data/contractnli/questions.parquet"
SENTENCE_CACHE = ROOT / "data/contractnli/sentence-source"
SENTENCE_ANCHORS_SOURCE = SENTENCE_CACHE / "sentence-anchors.parquet"
SENTENCE_SNAPSHOT_SOURCE = SENTENCE_CACHE / "sentence-snapshot.parquet"
STURDY_PILOT_REFERENCE = (
    ROOT / "provenance/frozen-pilot"
    / "contractnli-sturdy-a4-r2-ranked-windows-1000.parquet"
)
BM25_SOURCE = ROOT / "data/contractnli/retrieval/bm25.parquet"
EXPECTED_ROWS = 6_173
EXPECTED_PILOT_ROWS = 1_000
CONDITIONS = {
    "sturdy-a4-r2-ranked-windows": "sturdy-sentence-a4-r2-ranked-windows",
    "bm25-top4": "bm25-top4",
}
GENERATOR_MODELS = ("google/gemini-2.5-flash", "openai/gpt-4o-mini")
GENERATOR_PROMPT_VERSION = "supplied-generic-true-false-v1"
GENERATOR_SYSTEM_PROMPT = (
    pilot.BASELINE_SYSTEM_PROMPT
    + "\n\nAnswer True or False, followed by a brief explanation."
)
HAIKU_MODEL = "anthropic/claude-haiku-4.5"
HAIKU_PROMPT_VERSION = "supplied-binary-v1"
HAIKU_SYSTEM_PROMPT = pilot.ORIGINAL_JUDGE_SYSTEM_PROMPT
SONNET_MODEL = "anthropic/claude-sonnet-4.5"
SONNET_PROMPT_VERSION = "supplied-scale-1-5-v1"
SONNET_SYSTEM_PROMPT = (
    "You are an expert evaluator. You will be given a question, a golden "
    "(reference) answer, and a predicted answer. Score the predicted answer on "
    "a scale of 1 to 5 based on how well it matches the golden answer.\n\n"
    "Scoring rubric:\n"
    "  5 = Perfect match in meaning (may differ in wording)\n"
    "  4 = Mostly correct with minor omissions or extra detail\n"
    "  3 = Partially correct, captures the main idea but misses key details\n"
    "  2 = Somewhat related but largely incorrect or incomplete\n"
    "  1 = Completely wrong or irrelevant\n\n"
    "Respond with ONLY a JSON object in this format:\n"
    '{"score": <1-5>, "rationale": "<one sentence>"}'
)
JUDGE_USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\nGolden answer:\n{golden}"
    "\n\nPredicted answer:\n{predicted}"
)
MATRIX_VERSION = "contractnli-full-sturdy-bm25-6173-v1"


class ScaleJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    rationale: str


def judge_user_prompt(question: str, golden: str, predicted: str) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question, golden=golden, predicted=predicted
    )


def _joined_excerpts(frame: pd.DataFrame) -> pd.Series:
    columns = [f"excerpt_{rank}" for rank in range(1, 5)]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"BM25 source missing excerpt columns: {missing}")
    return frame[columns].fillna("").astype(str).apply(
        lambda row: "\n\n".join(value for value in row if value.strip()), axis=1
    )


def build_sturdy_contexts(
    questions: pd.DataFrame, anchors: pd.DataFrame, snapshots: pd.DataFrame
) -> pd.DataFrame:
    """Rebuild the exact rank-window framing used by the frozen 1,000-row pilot."""
    questions = questions.sort_values(["source_row", "query_id"], kind="stable").copy()
    if len(questions) != EXPECTED_ROWS or questions.query_id.astype(str).duplicated().any():
        raise ValueError("ContractNLI questions must contain 6,173 unique query IDs")
    query_ids = set(questions.query_id.astype(str))
    anchors = anchors[anchors.query_id.astype(str).isin(query_ids)].copy()
    anchors["query_id"] = anchors.query_id.astype(str)
    anchors["doc_id"] = anchors.doc_id.astype(str)
    snapshots = snapshots.copy()
    snapshots["doc_id"] = snapshots.doc_id.astype(str)
    grouped = {query_id: group for query_id, group in anchors.groupby("query_id")}
    if set(grouped) != query_ids:
        raise ValueError("questions and sentence-anchor query IDs differ")
    contexts: dict[str, str] = {}
    for row in questions.itertuples(index=False):
        query_id = str(row.query_id)
        retrieval = ordering.retrieve_windows(grouped[query_id], snapshots, radius=2, top_k=4)
        allowed = {str(value) for value in json.loads(row.allowed_doc_ids_json)}
        if retrieval["doc_id"] not in allowed:
            raise ValueError(f"{query_id}: anchor document is not allowed")
        contexts[query_id] = "\n\n".join(
            rendered
            for rendered in map(ordering.render_source_units, retrieval["windows"])
            if rendered
        )
    result = questions.copy()
    result["retrieved_context"] = result.query_id.astype(str).map(contexts)
    result["retrieval_error"] = ""
    result["retrieval_method"] = "sturdy"
    return result


def _condition_frame(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    required = {
        "dataset", "query_id", "split", "source_row", "question", "choice",
        "golden_answer", "retrieval_error",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{condition} source missing columns: {missing}")
    result = frame.copy()
    if condition == "bm25-top4":
        result = result[result.retrieval_method.astype(str).eq("bm25")].copy()
        result["retrieved_context"] = _joined_excerpts(result)
    elif "retrieved_context" not in result:
        raise ValueError("Sturdy source missing retrieved_context")
    result = result.sort_values(["source_row", "query_id"], kind="stable").reset_index(drop=True)
    if len(result) != EXPECTED_ROWS:
        raise ValueError(f"{condition}: expected {EXPECTED_ROWS} rows, found {len(result)}")
    if result.query_id.astype(str).duplicated().any():
        raise ValueError(f"{condition}: duplicate query IDs")
    errors = result.retrieval_error.fillna("").astype(str).ne("")
    empty = result.retrieved_context.fillna("").astype(str).str.strip().eq("")
    if errors.any() or empty.any():
        raise ValueError(
            f"{condition}: {int(errors.sum())} retrieval errors, "
            f"{int(empty.sum())} empty contexts"
        )
    result["reference_evidence"] = result.golden_answer.astype(str)
    result["golden_answer"] = result.choice.astype(str)
    result["retrieval_method"] = CONDITIONS[condition]
    result["selection_rank"] = range(1, len(result) + 1)
    result["selection_version"] = MATRIX_VERSION
    result["selection_key"] = result.query_id.astype(str)
    result["retrieval_context_sha256"] = result.retrieved_context.astype(str).map(text_sha256)
    columns = [
        "selection_rank", "selection_version", "selection_key", "dataset",
        "query_id", "split", "source_row", "question", "golden_answer",
        "reference_evidence", "retrieval_method", "retrieved_context",
        "retrieval_context_sha256", "retrieval_error",
    ]
    return result[columns].copy()


def prepare_inputs(
    questions_source: Path, anchors_source: Path, snapshot_source: Path,
    sturdy_pilot_reference: Path, bm25_source: Path, output_root: Path,
) -> dict[str, pd.DataFrame]:
    sturdy_raw = build_sturdy_contexts(
        pd.read_parquet(questions_source), pd.read_parquet(anchors_source),
        pd.read_parquet(snapshot_source),
    )
    frames = {
        "sturdy-a4-r2-ranked-windows": _condition_frame(
            sturdy_raw, "sturdy-a4-r2-ranked-windows"
        ),
        "bm25-top4": _condition_frame(pd.read_parquet(bm25_source), "bm25-top4"),
    }
    sturdy = frames["sturdy-a4-r2-ranked-windows"].set_index("query_id")
    bm25 = frames["bm25-top4"].set_index("query_id")
    if set(sturdy.index) != set(bm25.index):
        raise ValueError("retrieval conditions have different query IDs")
    for column in ("source_row", "split", "question", "golden_answer"):
        if not sturdy[column].sort_index().equals(bm25[column].sort_index()):
            raise ValueError(f"retrieval conditions disagree on {column}")
    pilot_reference = pd.read_parquet(sturdy_pilot_reference).set_index("query_id")
    if len(pilot_reference) != EXPECTED_PILOT_ROWS or pilot_reference.index.duplicated().any():
        raise ValueError(
            f"frozen Sturdy pilot reference must contain {EXPECTED_PILOT_ROWS:,} unique rows"
        )
    rebuilt_pilot = sturdy.loc[pilot_reference.index]
    if not rebuilt_pilot.retrieved_context.equals(pilot_reference.retrieved_context):
        mismatches = int(
            rebuilt_pilot.retrieved_context.ne(pilot_reference.retrieved_context).sum()
        )
        raise ValueError(
            f"rebuilt Sturdy contexts differ from frozen pilot on {mismatches} rows"
        )
    for column in ("source_row", "split", "question", "golden_answer"):
        if not rebuilt_pilot[column].equals(pilot_reference[column]):
            raise ValueError(f"rebuilt Sturdy pilot identities disagree on {column}")
    input_root = output_root / "input"
    for condition, frame in frames.items():
        atomic_parquet(frame, input_root / f"{condition}.parquet")
    provenance = {
        "experiment": MATRIX_VERSION,
        "scope": "paired full ContractNLI comparison: honest Sturdy A4 +/-2 versus packaged BM25 Top 4",
        "rows_per_condition": EXPECTED_ROWS,
        "sources": {
            "questions": {
                "path": str(questions_source.resolve()),
                "sha256": pilot.file_sha256(questions_source),
            },
            "sturdy-a4-r2-ranked-windows": {
                "anchors_path": str(anchors_source.resolve()),
                "anchors_sha256": pilot.file_sha256(anchors_source),
                "snapshot_path": str(snapshot_source.resolve()),
                "snapshot_sha256": pilot.file_sha256(snapshot_source),
                "pilot_reference_path": str(sturdy_pilot_reference.resolve()),
                "pilot_reference_sha256": pilot.file_sha256(sturdy_pilot_reference),
                "pilot_reference_exact_context_matches": len(pilot_reference),
            },
            "bm25-top4": {
                "path": str(bm25_source.resolve()),
                "sha256": pilot.file_sha256(bm25_source),
            },
        },
        "generator_models": list(GENERATOR_MODELS),
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "generator_system_prompt": GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt_template": pilot.USER_PROMPT_TEMPLATE,
        "judges": {
            HAIKU_MODEL: {
                "prompt_version": HAIKU_PROMPT_VERSION,
                "system_prompt": HAIKU_SYSTEM_PROMPT,
                "user_prompt_template": pilot.JUDGE_USER_PROMPT_TEMPLATE,
                "schema": pilot.CorrectJudgment.model_json_schema(),
            },
            SONNET_MODEL: {
                "prompt_version": SONNET_PROMPT_VERSION,
                "system_prompt": SONNET_SYSTEM_PROMPT,
                "user_prompt_template": JUDGE_USER_PROMPT_TEMPLATE,
                "schema": ScaleJudgment.model_json_schema(),
            },
        },
        "api_backend": "openrouter",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return frames


def generation_path(output_root: Path, condition: str, model: str) -> Path:
    return (
        output_root / condition / "generations"
        / f"{model_slug(model)}__{GENERATOR_PROMPT_VERSION}.parquet"
    )


def haiku_path(output_root: Path, condition: str, model: str) -> Path:
    stem = generation_path(output_root, condition, model).stem
    return (
        output_root / condition / "judgments"
        / f"{stem}__judge-{model_slug(HAIKU_MODEL)}__{HAIKU_PROMPT_VERSION}.parquet"
    )


def sonnet_path(output_root: Path, condition: str, model: str) -> Path:
    stem = generation_path(output_root, condition, model).stem
    return (
        output_root / condition / "judgments"
        / f"{stem}__judge-{model_slug(SONNET_MODEL)}__{SONNET_PROMPT_VERSION}.parquet"
    )


def scale_columns() -> list[str]:
    return pilot.generation_columns() + [
        "judge_model", "judge_prompt_version", "judge_system_prompt_sha256",
        "judge_user_prompt_sha256", "score", "rationale", "judge_error",
        "judge_response_id", "judge_input_tokens", "judge_output_tokens",
        "judge_total_tokens", "judge_response_model", "judge_response_provider",
    ]


async def judge_scale_one(api, semaphore, row) -> dict:
    prompt = judge_user_prompt(
        str(row.question), str(row.golden_answer), str(row.generated_answer)
    )
    base = row._asdict() | {
        "judge_model": SONNET_MODEL,
        "judge_prompt_version": SONNET_PROMPT_VERSION,
        "judge_system_prompt_sha256": text_sha256(SONNET_SYSTEM_PROMPT),
        "judge_user_prompt_sha256": text_sha256(prompt),
    }
    async with semaphore:
        try:
            request_options = {}
            if SONNET_MODEL != "anthropic/claude-sonnet-4":
                request_options = {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "contractnli_full_matrix_score",
                            "strict": True,
                            "schema": ScaleJudgment.model_json_schema(),
                        },
                    },
                    "extra_body": {"provider": {"require_parameters": True}},
                }
            response = await with_retries(lambda: api.chat.completions.create(
                model=SONNET_MODEL,
                messages=[
                    {"role": "system", "content": SONNET_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                **request_options,
            ))
            parsed = ScaleJudgment.model_validate_json(
                json_object_text(completion_text(response))
            )
            if not parsed.rationale.strip():
                raise ValueError("model returned empty rationale")
            usage = usage_fields(response)
            route = response_route(response)
            return base | {
                "score": int(parsed.score), "rationale": parsed.rationale,
                "judge_error": "", "judge_response_id": str(response.id),
                "judge_input_tokens": usage["input_tokens"],
                "judge_output_tokens": usage["output_tokens"],
                "judge_total_tokens": usage["total_tokens"],
                "judge_response_model": route["response_model"],
                "judge_response_provider": route["response_provider"],
            }
        except Exception as error:
            return base | {
                "score": None, "rationale": "",
                "judge_error": type(error).__name__, "judge_response_id": "",
                "judge_input_tokens": 0, "judge_output_tokens": 0,
                "judge_total_tokens": 0, "judge_response_model": "",
                "judge_response_provider": "",
            }


def write_scale_summary(frame: pd.DataFrame, target: Path) -> dict:
    scores = pd.to_numeric(frame.score)
    summary = {
        "generator_model": str(frame.generator_model.iloc[0]),
        "generator_prompt_version": str(frame.generator_prompt_version.iloc[0]),
        "judge_model": SONNET_MODEL,
        "judge_prompt_version": SONNET_PROMPT_VERSION,
        "questions": int(len(frame)),
        "generation_errors": int(frame.generation_error.fillna("").ne("").sum()),
        "judge_errors": int(frame.judge_error.fillna("").ne("").sum()),
        "mean_score": float(scores.mean()),
        **{f"score_{score}": int(scores.eq(score).sum()) for score in range(1, 6)},
    }
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(target.with_suffix(".summary.csv"), index=False)
    return summary


async def run_scale_judgment(generation: Path, target: Path, concurrency: int,
                             checkpoint_every: int) -> Path:
    generations = pd.read_parquet(generation)
    checkpoint = target.parent / ".checkpoints" / target.name
    expected = generations[["query_id", "generation_sha256"]].copy()
    expected["judge_user_prompt_sha256"] = [
        text_sha256(judge_user_prompt(
            str(row.question), str(row.golden_answer), str(row.generated_answer)
        ))
        for row in generations.itertuples(index=False)
    ]
    prior_path = target if target.exists() else checkpoint
    successful = pilot._successful_prior(
        prior_path, expected,
        {
            "judge_model": SONNET_MODEL,
            "judge_prompt_version": SONNET_PROMPT_VERSION,
            "judge_system_prompt_sha256": text_sha256(SONNET_SYSTEM_PROMPT),
            "cohort_sha256": str(generations.cohort_sha256.iloc[0]),
        },
        ("generation_sha256", "judge_user_prompt_sha256"), "judge_error",
    )
    records = {
        str(row.query_id): row._asdict()
        for row in successful.itertuples(index=False)
    }
    pending = [
        row for row in generations.itertuples(index=False)
        if str(row.query_id) not in records
    ]
    print(
        f"judge {generation.stem} / {SONNET_MODEL}: "
        f"{len(records):,} done, {len(pending):,} pending",
        flush=True,
    )
    if pending:
        api = client()
        semaphore = asyncio.Semaphore(concurrency)
        try:
            for offset in range(0, len(pending), checkpoint_every):
                completed = await asyncio.gather(*(
                    judge_scale_one(api, semaphore, row)
                    for row in pending[offset:offset + checkpoint_every]
                ))
                records.update((row["query_id"], row) for row in completed)
                atomic_parquet(
                    pd.DataFrame(records.values(), columns=scale_columns()), checkpoint
                )
                print(f"  {len(records):,}/{len(generations):,}", flush=True)
        finally:
            await api.close()
    result = pd.DataFrame(records.values(), columns=scale_columns())
    result = result.sort_values("selection_rank").reset_index(drop=True)
    atomic_parquet(result, checkpoint)
    errors = result.judge_error.fillna("").ne("")
    if errors.any() or len(result) != len(generations):
        raise SystemExit(
            f"judgment incomplete: {int(errors.sum())} errors, "
            f"{len(result):,}/{len(generations):,} rows; rerun to retry"
        )
    atomic_parquet(result, target)
    write_scale_summary(result, target)
    return target


def exact_sign_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def write_paired_summaries(output_root: Path, models: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    sturdy_condition, bm25_condition = CONDITIONS
    for model in models:
        sturdy = pd.read_parquet(haiku_path(output_root, sturdy_condition, model)).set_index("query_id")
        bm25 = pd.read_parquet(haiku_path(output_root, bm25_condition, model)).set_index("query_id")
        if set(sturdy.index) != set(bm25.index):
            raise ValueError(f"{model}/haiku: paired query IDs differ")
        a = sturdy.judge_correct.astype(bool).sort_index()
        b = bm25.judge_correct.astype(bool).sort_index()
        sturdy_only, bm25_only = int((a & ~b).sum()), int((~a & b).sum())
        rows.append({
            "generator_model": model, "judge_model": HAIKU_MODEL,
            "judge_type": "binary", "questions": len(a),
            "sturdy_metric": float(a.mean()), "bm25_metric": float(b.mean()),
            "sturdy_minus_bm25": float(a.mean() - b.mean()),
            "both_correct": int((a & b).sum()), "sturdy_only": sturdy_only,
            "bm25_only": bm25_only, "both_incorrect": int((~a & ~b).sum()),
            "paired_exact_pvalue": exact_sign_pvalue(sturdy_only, bm25_only),
        })
        sturdy = pd.read_parquet(sonnet_path(output_root, sturdy_condition, model)).set_index("query_id")
        bm25 = pd.read_parquet(sonnet_path(output_root, bm25_condition, model)).set_index("query_id")
        if set(sturdy.index) != set(bm25.index):
            raise ValueError(f"{model}/sonnet: paired query IDs differ")
        a = pd.to_numeric(sturdy.score).sort_index()
        b = pd.to_numeric(bm25.score).sort_index()
        delta = a - b
        wins, losses = int(delta.gt(0).sum()), int(delta.lt(0).sum())
        row = {
            "generator_model": model, "judge_model": SONNET_MODEL,
            "judge_type": "scale-1-5", "questions": len(a),
            "sturdy_metric": float(a.mean()), "bm25_metric": float(b.mean()),
            "sturdy_minus_bm25": float(delta.mean()),
            "sturdy_wins": wins, "bm25_wins": losses,
            "ties": int(delta.eq(0).sum()),
            "paired_exact_pvalue": exact_sign_pvalue(wins, losses),
        }
        row.update({f"sturdy_score_{score}": int(a.eq(score).sum()) for score in range(1, 6)})
        row.update({f"bm25_score_{score}": int(b.eq(score).sum()) for score in range(1, 6)})
        row.update({
            f"paired_score_{sturdy_score}_to_{bm25_score}": int(
                (a.eq(sturdy_score) & b.eq(bm25_score)).sum()
            )
            for sturdy_score in range(1, 6)
            for bm25_score in range(1, 6)
        })
        row.update({f"paired_delta_{delta_value:+d}": int(delta.eq(delta_value).sum()) for delta_value in range(-4, 5)})
        rows.append(row)
    summary_root = output_root / "paired"
    summary_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_root / "summary.csv", index=False)
    (summary_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def validate_outputs(output_root: Path, models: tuple[str, ...]) -> None:
    inputs = {
        condition: pd.read_parquet(output_root / "input" / f"{condition}.parquet")
        for condition in CONDITIONS
    }
    sturdy_input = inputs["sturdy-a4-r2-ranked-windows"].set_index("query_id")
    bm25_input = inputs["bm25-top4"].set_index("query_id")
    if set(sturdy_input.index) != set(bm25_input.index):
        raise ValueError("prepared input query IDs differ")
    for column in ("source_row", "split", "question", "golden_answer"):
        if not sturdy_input[column].sort_index().equals(bm25_input[column].sort_index()):
            raise ValueError(f"prepared inputs disagree on {column}")
    for condition, expected in inputs.items():
        if len(expected) != EXPECTED_ROWS or expected.query_id.nunique() != EXPECTED_ROWS:
            raise ValueError(f"{condition}: invalid input coverage")
        expected = expected.set_index("query_id")
        for model in models:
            generation = pd.read_parquet(generation_path(output_root, condition, model))
            if len(generation) != EXPECTED_ROWS or generation.query_id.nunique() != EXPECTED_ROWS:
                raise ValueError(f"{condition}/{model}: invalid generation coverage")
            if generation.generation_error.fillna("").ne("").any():
                raise ValueError(f"{condition}/{model}: generation errors remain")
            if generation.generated_answer.fillna("").str.strip().eq("").any():
                raise ValueError(f"{condition}/{model}: empty generations")
            if set(generation.generator_prompt_version.astype(str)) != {GENERATOR_PROMPT_VERSION}:
                raise ValueError(f"{condition}/{model}: wrong generator prompt")
            if set(generation.generator_model.astype(str)) != {model}:
                raise ValueError(f"{condition}/{model}: wrong generator model")
            if set(generation.generator_system_prompt_sha256.astype(str)) != {text_sha256(GENERATOR_SYSTEM_PROMPT)}:
                raise ValueError(f"{condition}/{model}: wrong generator prompt hash")
            if set(generation.query_id.astype(str)) != set(expected.index.astype(str)):
                raise ValueError(f"{condition}/{model}: generation key mismatch")
            generation = generation.set_index("query_id")
            if not generation.retrieval_context_sha256.sort_index().equals(
                expected.retrieval_context_sha256.sort_index()
            ):
                raise ValueError(f"{condition}/{model}: context hash mismatch")
            haiku = pd.read_parquet(haiku_path(output_root, condition, model))
            if len(haiku) != EXPECTED_ROWS or haiku.query_id.nunique() != EXPECTED_ROWS:
                raise ValueError(f"{condition}/{model}/haiku: invalid coverage")
            if haiku.judge_error.fillna("").ne("").any():
                raise ValueError(f"{condition}/{model}/haiku: errors remain")
            if set(haiku.judge_model.astype(str)) != {HAIKU_MODEL}:
                raise ValueError(f"{condition}/{model}/haiku: wrong judge")
            if set(haiku.judge_prompt_version.astype(str)) != {HAIKU_PROMPT_VERSION}:
                raise ValueError(f"{condition}/{model}/haiku: wrong prompt version")
            if set(haiku.judge_system_prompt_sha256.astype(str)) != {text_sha256(HAIKU_SYSTEM_PROMPT)}:
                raise ValueError(f"{condition}/{model}/haiku: wrong prompt hash")
            haiku = haiku.set_index("query_id")
            if set(haiku.index.astype(str)) != set(generation.index.astype(str)):
                raise ValueError(f"{condition}/{model}/haiku: key mismatch")
            if not haiku.generation_sha256.sort_index().equals(
                generation.generation_sha256.sort_index()
            ):
                raise ValueError(f"{condition}/{model}/haiku: generation hash mismatch")
            sonnet = pd.read_parquet(sonnet_path(output_root, condition, model))
            if len(sonnet) != EXPECTED_ROWS or sonnet.query_id.nunique() != EXPECTED_ROWS:
                raise ValueError(f"{condition}/{model}/sonnet: invalid coverage")
            if sonnet.judge_error.fillna("").ne("").any():
                raise ValueError(f"{condition}/{model}/sonnet: errors remain")
            if set(sonnet.judge_model.astype(str)) != {SONNET_MODEL}:
                raise ValueError(f"{condition}/{model}/sonnet: wrong judge")
            if set(sonnet.judge_prompt_version.astype(str)) != {SONNET_PROMPT_VERSION}:
                raise ValueError(f"{condition}/{model}/sonnet: wrong prompt version")
            if set(sonnet.judge_system_prompt_sha256.astype(str)) != {text_sha256(SONNET_SYSTEM_PROMPT)}:
                raise ValueError(f"{condition}/{model}/sonnet: wrong prompt hash")
            scores = pd.to_numeric(sonnet.score)
            if not scores.between(1, 5).all() or sonnet.rationale.fillna("").str.strip().eq("").any():
                raise ValueError(f"{condition}/{model}/sonnet: invalid score output")
            sonnet = sonnet.set_index("query_id")
            if set(sonnet.index.astype(str)) != set(generation.index.astype(str)):
                raise ValueError(f"{condition}/{model}/sonnet: key mismatch")
            if not sonnet.generation_sha256.sort_index().equals(
                generation.generation_sha256.sort_index()
            ):
                raise ValueError(f"{condition}/{model}/sonnet: generation hash mismatch")
    write_paired_summaries(output_root, models)
    print(f"valid: {len(CONDITIONS)} conditions x {len(models)} models x {EXPECTED_ROWS:,} rows")


def validate_generations(output_root: Path, models: tuple[str, ...]) -> None:
    inputs = {
        condition: pd.read_parquet(output_root / "input" / f"{condition}.parquet")
        for condition in CONDITIONS
    }
    sturdy = inputs["sturdy-a4-r2-ranked-windows"].set_index("query_id")
    bm25 = inputs["bm25-top4"].set_index("query_id")
    if set(sturdy.index) != set(bm25.index):
        raise ValueError("prepared input query IDs differ")
    for column in ("source_row", "split", "question", "golden_answer"):
        if not sturdy[column].sort_index().equals(bm25[column].sort_index()):
            raise ValueError(f"prepared inputs disagree on {column}")
    for condition, expected_frame in inputs.items():
        if len(expected_frame) != EXPECTED_ROWS or expected_frame.query_id.nunique() != EXPECTED_ROWS:
            raise ValueError(f"{condition}: invalid input coverage")
        expected = expected_frame.set_index("query_id")
        for model in models:
            generation = pd.read_parquet(generation_path(output_root, condition, model))
            if len(generation) != EXPECTED_ROWS or generation.query_id.nunique() != EXPECTED_ROWS:
                raise ValueError(f"{condition}/{model}: invalid generation coverage")
            if generation.generation_error.fillna("").ne("").any():
                raise ValueError(f"{condition}/{model}: generation errors remain")
            if generation.generated_answer.fillna("").str.strip().eq("").any():
                raise ValueError(f"{condition}/{model}: empty generations")
            if set(generation.generator_model.astype(str)) != {model}:
                raise ValueError(f"{condition}/{model}: wrong generator model")
            if set(generation.generator_prompt_version.astype(str)) != {GENERATOR_PROMPT_VERSION}:
                raise ValueError(f"{condition}/{model}: wrong generator prompt")
            if set(generation.generator_system_prompt_sha256.astype(str)) != {text_sha256(GENERATOR_SYSTEM_PROMPT)}:
                raise ValueError(f"{condition}/{model}: wrong generator prompt hash")
            generation = generation.set_index("query_id")
            if set(generation.index.astype(str)) != set(expected.index.astype(str)):
                raise ValueError(f"{condition}/{model}: generation key mismatch")
            if not generation.retrieval_context_sha256.sort_index().equals(
                expected.retrieval_context_sha256.sort_index()
            ):
                raise ValueError(f"{condition}/{model}: context hash mismatch")
    print(
        f"generation gate valid: {len(CONDITIONS)} conditions x "
        f"{len(models)} models x {EXPECTED_ROWS:,} rows",
        flush=True,
    )


async def run(args) -> None:
    output_root = args.output_root.resolve()
    if args.stage in ("prepare", "all"):
        all_frames = prepare_inputs(
            args.questions_source.resolve(), args.sentence_anchors.resolve(),
            args.sentence_snapshot.resolve(), args.sturdy_pilot_reference.resolve(),
            args.bm25_source.resolve(), output_root,
        )
    else:
        all_frames = {
            condition: pd.read_parquet(output_root / "input" / f"{condition}.parquet")
            for condition in CONDITIONS
        }
    frames = {
        condition: all_frames[condition]
        for condition in args.conditions
    }
    if args.stage == "prepare":
        return
    if args.stage in ("generate", "all"):
        for condition, frame in frames.items():
            for model in args.generator_models:
                await pilot.run_generation(
                    frame, model, GENERATOR_PROMPT_VERSION, GENERATOR_SYSTEM_PROMPT,
                    output_root / condition, args.concurrency, args.checkpoint_every,
                )
    if args.stage in ("judge", "all"):
        for condition in args.conditions:
            condition_root = output_root / condition
            for model in args.generator_models:
                generation = generation_path(output_root, condition, model)
                if not generation.exists():
                    raise FileNotFoundError(f"missing generation: {generation}")
                if HAIKU_MODEL in args.judge_models:
                    await pilot.run_judgment(
                        generation, HAIKU_MODEL, HAIKU_PROMPT_VERSION,
                        HAIKU_SYSTEM_PROMPT, condition_root,
                        args.concurrency, args.checkpoint_every,
                    )
                if SONNET_MODEL in args.judge_models:
                    await run_scale_judgment(
                        generation, sonnet_path(output_root, condition, model),
                        args.concurrency, args.checkpoint_every,
                    )
    if args.stage == "validate-generations":
        validate_generations(output_root, tuple(args.generator_models))
    if args.stage in ("summarize", "all"):
        write_paired_summaries(output_root, tuple(args.generator_models))
    if args.stage in ("validate", "all"):
        validate_outputs(output_root, tuple(args.generator_models))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-source", type=Path, default=QUESTIONS_SOURCE)
    parser.add_argument("--sentence-anchors", type=Path, default=SENTENCE_ANCHORS_SOURCE)
    parser.add_argument("--sentence-snapshot", type=Path, default=SENTENCE_SNAPSHOT_SOURCE)
    parser.add_argument(
        "--sturdy-pilot-reference", type=Path, default=STURDY_PILOT_REFERENCE
    )
    parser.add_argument("--bm25-source", type=Path, default=BM25_SOURCE)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--generator-models", nargs="+", default=GENERATOR_MODELS)
    parser.add_argument(
        "--judge-models", nargs="+", choices=(HAIKU_MODEL, SONNET_MODEL),
        default=(HAIKU_MODEL, SONNET_MODEL),
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS),
        default=tuple(CONDITIONS),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument(
        "--stage", choices=(
            "prepare", "generate", "validate-generations", "judge",
            "summarize", "validate", "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")
    if len(set(args.generator_models)) != len(args.generator_models):
        parser.error("generator models must be unique")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("conditions must be unique")
    if len(set(args.judge_models)) != len(args.judge_models):
        parser.error("judge models must be unique")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
