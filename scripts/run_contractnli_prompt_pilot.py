#!/usr/bin/env python3
"""Run the frozen, Sturdy-only ContractNLI prompt-optimization pilot."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

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


RESULTS_ROOT = ROOT / "artifacts" / "experiments" / "contractnli-prompt-optimization-100"
FROZEN_INPUT = (
    ROOT
    / "artifacts/experiments/contractnli-sentence-a4-r2-ordering-evidence-1000"
    / "inputs/ranked-windows.parquet"
)
EXPECTED_RETRIEVAL_METHOD = "sturdy-sentence-a4-r2-ranked-windows"
SELECTION_VERSION = "contractnli-frozen-dev-stable-hash-100-v1"
SELECTION_SALT = "contractnli-prompt-optimization-100-v1"
GENERATOR_MODELS = ("google/gemini-2.5-flash", "openai/gpt-4o-mini")
JUDGE_MODELS = ("anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.5")

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based only on the provided "
    "context. If the context does not contain enough information to answer, say "
    "so. Be concise."
)
GENERATOR_TWEAK = "Answer with Entailment or Contradiction."
GENERATOR_REVISION = "Begin your answer with Entailment or Contradiction."
USER_PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}"

ORIGINAL_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. You will be given a question, a golden "
    "(reference) answer, and a predicted answer. Determine whether the predicted "
    "answer is correct.\n\n"
    "An answer is correct if it conveys the same essential meaning as the golden "
    "answer, even if the wording differs. Minor differences in precision, extra "
    "context, or phrasing are acceptable. An answer is incorrect if it contradicts "
    "the golden answer, misses the key point, or fails to answer the question.\n\n"
    "Respond with ONLY a JSON object in this format:\n"
    '{"correct": true}  or  {"correct": false}'
)
JUDGE_CLARIFICATION = (
    "A one-word answer is acceptable when it gives the correct label. The predicted "
    "answer does not need to match the reference wording."
)
JUDGE_USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\nGolden answer:\n{golden_answer}"
    "\n\nPredicted answer:\n{generated_answer}"
)

GENERATOR_PROMPTS = {
    "supplied-generic-v1": BASELINE_SYSTEM_PROMPT,
    "supplied-generic-answer-label-v1": (
        BASELINE_SYSTEM_PROMPT + "\n\n" + GENERATOR_TWEAK
    ),
}
REVISION_PROMPT = {
    "supplied-generic-begin-label-v1": (
        BASELINE_SYSTEM_PROMPT + "\n\n" + GENERATOR_REVISION
    )
}
JUDGE_PROMPTS = {
    "supplied-binary-v1": ORIGINAL_JUDGE_SYSTEM_PROMPT,
    "supplied-binary-label-clarification-v1": (
        ORIGINAL_JUDGE_SYSTEM_PROMPT + "\n\n" + JUDGE_CLARIFICATION
    ),
}
CLARIFIED_JUDGE_VERSION = "supplied-binary-label-clarification-v1"


class CorrectJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_selection_key(query_id: str) -> str:
    return text_sha256(f"{SELECTION_SALT}\n{query_id}")


def select_development_rows(frame: pd.DataFrame, size: int = 100) -> pd.DataFrame:
    required = {
        "dataset", "query_id", "split", "source_row", "question",
        "golden_answer", "retrieval_method", "retrieved_context",
        "retrieval_error",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frozen input missing columns: {', '.join(missing)}")
    methods = set(frame.retrieval_method.astype(str))
    if methods != {EXPECTED_RETRIEVAL_METHOD}:
        raise ValueError(f"unexpected retrieval methods: {sorted(methods)}")
    if frame.query_id.astype(str).duplicated().any():
        raise ValueError("frozen input contains duplicate query IDs")
    failed = frame.retrieval_error.fillna("").astype(str).ne("")
    if failed.any():
        raise ValueError(f"frozen input contains {int(failed.sum())} retrieval errors")
    development = frame[frame.split.astype(str).str.lower().eq("dev")].copy()
    if len(development) < size:
        raise ValueError(f"need {size} development rows, found {len(development)}")
    development["selection_key"] = development.query_id.astype(str).map(
        stable_selection_key
    )
    development = development.sort_values(
        ["selection_key", "source_row", "query_id"], kind="stable"
    ).head(size).copy()
    development.insert(0, "selection_rank", range(1, len(development) + 1))
    development["selection_version"] = SELECTION_VERSION
    development["retrieval_context_sha256"] = (
        development.retrieved_context.astype(str).map(text_sha256)
    )
    return development.reset_index(drop=True)


def cohort_sha256(frame: pd.DataFrame) -> str:
    rows = [
        {
            "selection_rank": int(row.selection_rank),
            "query_id": str(row.query_id),
            "source_row": int(row.source_row),
            "context_sha256": str(row.retrieval_context_sha256),
        }
        for row in frame.sort_values("selection_rank").itertuples(index=False)
    ]
    return text_sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")))


def prepare_cohort(source: Path, target: Path) -> pd.DataFrame:
    frame = select_development_rows(pd.read_parquet(source))
    atomic_parquet(frame, target)
    return frame


def generation_system_prompts(include_revision: bool = False) -> dict[str, str]:
    prompts = dict(GENERATOR_PROMPTS)
    if include_revision:
        prompts.update(REVISION_PROMPT)
    return prompts


def user_prompt(context: str, question: str) -> str:
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)


def judge_user_prompt(question: str, golden_answer: str, generated_answer: str) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )


_LEADING_LABEL = re.compile(
    r"^\s*(Entailment|Contradiction)(?=$|[\s:.,;!?)\]}'\"\-—])",
    re.IGNORECASE,
)


def parse_label(answer: str) -> str | None:
    match = _LEADING_LABEL.match(str(answer))
    return match.group(1).capitalize() if match else None


def generation_columns() -> list[str]:
    return [
        "dataset", "query_id", "split", "source_row", "selection_rank",
        "selection_version", "selection_key", "cohort_sha256",
        "retrieval_method", "retrieval_context_sha256", "retrieved_context",
        "question", "golden_answer", "generator_model", "generator_prompt_version",
        "generator_system_prompt_sha256", "generator_user_prompt_sha256",
        "api_backend", "generated_answer", "parsed_label", "label_correct",
        "label_parse_error", "generation_error", "response_id", "input_tokens",
        "output_tokens", "total_tokens", "response_model", "response_provider",
        "generation_sha256",
    ]


async def generate_one(api, semaphore, row, model: str, version: str, system: str,
                       cohort_hash: str) -> dict:
    context = str(row.retrieved_context)
    prompt = user_prompt(context, str(row.question))
    base = {
        "dataset": str(row.dataset), "query_id": str(row.query_id),
        "split": str(row.split), "source_row": int(row.source_row),
        "selection_rank": int(row.selection_rank),
        "selection_version": str(row.selection_version),
        "selection_key": str(row.selection_key), "cohort_sha256": cohort_hash,
        "retrieval_method": str(row.retrieval_method),
        "retrieval_context_sha256": text_sha256(context),
        "retrieved_context": context, "question": str(row.question),
        "golden_answer": str(row.golden_answer), "generator_model": model,
        "generator_prompt_version": version,
        "generator_system_prompt_sha256": text_sha256(system),
        "generator_user_prompt_sha256": text_sha256(prompt),
        "api_backend": "openrouter",
    }
    async with semaphore:
        try:
            response = await with_retries(lambda: api.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            ))
            answer = completion_text(response)
            if not answer:
                raise ValueError("model returned empty output")
            label = parse_label(answer)
            generation_hash = text_sha256(answer)
            return base | {
                "generated_answer": answer, "parsed_label": label or "",
                "label_correct": bool(label == str(row.golden_answer)),
                "label_parse_error": "" if label else "no_leading_contractnli_label",
                "generation_error": "", "response_id": str(response.id),
                **usage_fields(response), **response_route(response),
                "generation_sha256": generation_hash,
            }
        except Exception as error:
            return base | {
                "generated_answer": "", "parsed_label": "", "label_correct": False,
                "label_parse_error": "generation_error",
                "generation_error": type(error).__name__, "response_id": "",
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "response_model": "", "response_provider": "",
                "generation_sha256": text_sha256(""),
            }


def _successful_prior(path: Path, expected: pd.DataFrame, checks: dict[str, str],
                      hash_columns: tuple[str, ...], error_column: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    prior = pd.read_parquet(path)
    required = {"query_id", error_column, *hash_columns, *checks}
    missing = sorted(required - set(prior.columns))
    if missing:
        raise ValueError(f"incompatible checkpoint {path}: missing {missing}")
    if prior.query_id.astype(str).duplicated().any():
        raise ValueError(f"incompatible checkpoint {path}: duplicate query IDs")
    expected_ids = set(expected.query_id.astype(str))
    if not set(prior.query_id.astype(str)).issubset(expected_ids):
        raise ValueError(f"incompatible checkpoint {path}: wrong cohort")
    for column, value in checks.items():
        if set(prior[column].astype(str)) != {str(value)}:
            raise ValueError(f"incompatible checkpoint {path}: {column}")
    for hash_column in hash_columns:
        expected_hashes = dict(zip(
            expected.query_id.astype(str), expected[hash_column].astype(str)
        ))
        if any(str(row[hash_column]) != expected_hashes[str(row.query_id)]
               for _, row in prior.iterrows()):
            raise ValueError(f"incompatible checkpoint {path}: {hash_column}")
    return prior[prior[error_column].fillna("").astype(str).eq("")].copy()


def write_generation_summary(frame: pd.DataFrame, target: Path) -> None:
    parsed = frame.label_parse_error.fillna("").eq("")
    summary = {
        "generator_model": str(frame.generator_model.iloc[0]),
        "generator_prompt_version": str(frame.generator_prompt_version.iloc[0]),
        "questions": int(len(frame)),
        "generation_errors": int(frame.generation_error.fillna("").ne("").sum()),
        "parsed_labels": int(parsed.sum()),
        "label_parse_rate": float(parsed.mean()),
        "label_correct": int(frame.label_correct.fillna(False).sum()),
        "label_accuracy": float(frame.label_correct.fillna(False).mean()),
        "selection_metric": False,
    }
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(target.with_suffix(".summary.csv"), index=False)


async def run_generation(cohort: pd.DataFrame, model: str, version: str, system: str,
                         output_root: Path, concurrency: int,
                         checkpoint_every: int) -> Path:
    target = output_root / "generations" / f"{model_slug(model)}__{version}.parquet"
    checkpoint = target.parent / ".checkpoints" / target.name
    cohort_hash = cohort_sha256(cohort)
    expected = cohort[["query_id", "retrieval_context_sha256"]].copy()
    expected["generator_user_prompt_sha256"] = [
        text_sha256(user_prompt(str(row.retrieved_context), str(row.question)))
        for row in cohort.itertuples(index=False)
    ]
    prior_path = target if target.exists() else checkpoint
    successful = _successful_prior(
        prior_path, expected,
        {
            "generator_model": model, "generator_prompt_version": version,
            "generator_system_prompt_sha256": text_sha256(system),
            "cohort_sha256": cohort_hash, "api_backend": "openrouter",
        },
        ("retrieval_context_sha256", "generator_user_prompt_sha256"),
        "generation_error",
    )
    records = {str(row.query_id): row._asdict() for row in successful.itertuples(index=False)}
    pending = [row for row in cohort.itertuples(index=False) if str(row.query_id) not in records]
    print(f"generate {model} / {version}: {len(records)} done, {len(pending)} pending")
    if pending:
        api = client()
        semaphore = asyncio.Semaphore(concurrency)
        try:
            for offset in range(0, len(pending), checkpoint_every):
                completed = await asyncio.gather(*(
                    generate_one(api, semaphore, row, model, version, system, cohort_hash)
                    for row in pending[offset:offset + checkpoint_every]
                ))
                records.update((row["query_id"], row) for row in completed)
                atomic_parquet(pd.DataFrame(records.values(), columns=generation_columns()), checkpoint)
        finally:
            await api.close()
    result = pd.DataFrame(records.values(), columns=generation_columns())
    result = result.sort_values("selection_rank").reset_index(drop=True)
    atomic_parquet(result, checkpoint)
    errors = result.generation_error.fillna("").ne("")
    if errors.any() or len(result) != len(cohort):
        raise SystemExit(
            f"generation incomplete: {int(errors.sum())} errors, "
            f"{len(result)}/{len(cohort)} rows; rerun to retry"
        )
    atomic_parquet(result, target)
    write_generation_summary(result, target)
    return target


def judgment_columns() -> list[str]:
    return generation_columns() + [
        "judge_model", "judge_prompt_version", "judge_system_prompt_sha256",
        "judge_user_prompt_sha256", "judge_correct", "judge_error",
        "judge_response_id", "judge_input_tokens", "judge_output_tokens",
        "judge_total_tokens", "judge_response_model", "judge_response_provider",
    ]


async def judge_one(api, semaphore, row, model: str, version: str,
                    system: str) -> dict:
    prompt = judge_user_prompt(
        str(row.question), str(row.golden_answer), str(row.generated_answer)
    )
    base = row._asdict() | {
        "judge_model": model, "judge_prompt_version": version,
        "judge_system_prompt_sha256": text_sha256(system),
        "judge_user_prompt_sha256": text_sha256(prompt),
    }
    async with semaphore:
        try:
            response = await with_retries(lambda: api.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "contractnli_prompt_pilot_correctness",
                        "strict": True,
                        "schema": CorrectJudgment.model_json_schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            ))
            parsed = CorrectJudgment.model_validate_json(
                json_object_text(completion_text(response))
            )
            usage = usage_fields(response)
            route = response_route(response)
            return base | {
                "judge_correct": bool(parsed.correct), "judge_error": "",
                "judge_response_id": str(response.id),
                "judge_input_tokens": usage["input_tokens"],
                "judge_output_tokens": usage["output_tokens"],
                "judge_total_tokens": usage["total_tokens"],
                "judge_response_model": route["response_model"],
                "judge_response_provider": route["response_provider"],
            }
        except Exception as error:
            return base | {
                "judge_correct": False, "judge_error": type(error).__name__,
                "judge_response_id": "", "judge_input_tokens": 0,
                "judge_output_tokens": 0, "judge_total_tokens": 0,
                "judge_response_model": "", "judge_response_provider": "",
            }


def write_judgment_summary(frame: pd.DataFrame, target: Path) -> dict:
    summary = {
        "generator_model": str(frame.generator_model.iloc[0]),
        "generator_prompt_version": str(frame.generator_prompt_version.iloc[0]),
        "judge_model": str(frame.judge_model.iloc[0]),
        "judge_prompt_version": str(frame.judge_prompt_version.iloc[0]),
        "questions": int(len(frame)),
        "judge_errors": int(frame.judge_error.fillna("").ne("").sum()),
        "correct": int(frame.judge_correct.fillna(False).sum()),
        "accuracy": float(frame.judge_correct.fillna(False).mean()),
        "label_accuracy": float(frame.label_correct.fillna(False).mean()),
        "selection_metric": str(frame.judge_prompt_version.iloc[0]) == CLARIFIED_JUDGE_VERSION,
    }
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(target.with_suffix(".summary.csv"), index=False)
    return summary


async def run_judgment(generation_path: Path, judge_model: str, version: str,
                       system: str, output_root: Path, concurrency: int,
                       checkpoint_every: int) -> Path:
    generations = pd.read_parquet(generation_path)
    stem = generation_path.stem
    target = output_root / "judgments" / f"{stem}__judge-{model_slug(judge_model)}__{version}.parquet"
    checkpoint = target.parent / ".checkpoints" / target.name
    expected = generations[["query_id", "generation_sha256"]].copy()
    expected["judge_user_prompt_sha256"] = [
        text_sha256(judge_user_prompt(
            str(row.question), str(row.golden_answer), str(row.generated_answer)
        ))
        for row in generations.itertuples(index=False)
    ]
    prior_path = target if target.exists() else checkpoint
    successful = _successful_prior(
        prior_path, expected,
        {
            "judge_model": judge_model, "judge_prompt_version": version,
            "judge_system_prompt_sha256": text_sha256(system),
            "cohort_sha256": str(generations.cohort_sha256.iloc[0]),
        },
        ("generation_sha256", "judge_user_prompt_sha256"), "judge_error",
    )
    records = {str(row.query_id): row._asdict() for row in successful.itertuples(index=False)}
    pending = [row for row in generations.itertuples(index=False) if str(row.query_id) not in records]
    print(f"judge {stem} / {judge_model} / {version}: {len(records)} done, {len(pending)} pending")
    if pending:
        api = client()
        semaphore = asyncio.Semaphore(concurrency)
        try:
            for offset in range(0, len(pending), checkpoint_every):
                completed = await asyncio.gather(*(
                    judge_one(api, semaphore, row, judge_model, version, system)
                    for row in pending[offset:offset + checkpoint_every]
                ))
                records.update((row["query_id"], row) for row in completed)
                atomic_parquet(pd.DataFrame(records.values(), columns=judgment_columns()), checkpoint)
        finally:
            await api.close()
    result = pd.DataFrame(records.values(), columns=judgment_columns())
    result = result.sort_values("selection_rank").reset_index(drop=True)
    atomic_parquet(result, checkpoint)
    errors = result.judge_error.fillna("").ne("")
    if errors.any() or len(result) != len(generations):
        raise SystemExit(
            f"judgment incomplete: {int(errors.sum())} errors, "
            f"{len(result)}/{len(generations)} rows; rerun to retry"
        )
    atomic_parquet(result, target)
    write_judgment_summary(result, target)
    return target


def write_provenance(output_root: Path, source: Path, cohort: pd.DataFrame,
                     generator_models: list[str], judge_models: list[str],
                     prompts: dict[str, str]) -> None:
    provenance = {
        "experiment": "contractnli-prompt-optimization-100",
        "scope": "Sturdy-only prompt optimization; not a retrieval comparison",
        "source": str(source.resolve()), "source_sha256": file_sha256(source),
        "source_retrieval_method": EXPECTED_RETRIEVAL_METHOD,
        "selection": {
            "version": SELECTION_VERSION, "salt": SELECTION_SALT,
            "rule": "filter split=dev; sort by SHA256(salt + newline + query_id), source_row, query_id; take first 100",
            "rows": len(cohort), "cohort_sha256": cohort_sha256(cohort),
            "query_ids": cohort.sort_values("selection_rank").query_id.astype(str).tolist(),
        },
        "generator_models": generator_models, "judge_models": judge_models,
        "generator_prompts": prompts,
        "generator_user_prompt_template": USER_PROMPT_TEMPLATE,
        "judge_prompts": JUDGE_PROMPTS,
        "judge_user_prompt_template": JUDGE_USER_PROMPT_TEMPLATE,
        "selection_metric": "mean clarified boolean-judge accuracy; deterministic leading-label accuracy is a sanity check",
        "api_backend": "openrouter",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def freeze_selected_prompt(output_root: Path) -> None:
    summaries = []
    for path in sorted((output_root / "judgments").glob("*.summary.json")):
        row = json.loads(path.read_text())
        if row.get("judge_prompt_version") == CLARIFIED_JUDGE_VERSION:
            summaries.append(row)
    if not summaries:
        raise ValueError("no complete clarified-judge summaries to select from")
    grouped: dict[str, list[float]] = {}
    for row in summaries:
        grouped.setdefault(row["generator_prompt_version"], []).append(float(row["accuracy"]))
    candidates = [
        {
            "generator_prompt_version": version,
            "mean_clarified_judge_accuracy": sum(scores) / len(scores),
            "cells": len(scores),
            "system_prompt": (GENERATOR_PROMPTS | REVISION_PROMPT)[version],
        }
        for version, scores in grouped.items()
    ]
    winner = sorted(
        candidates,
        key=lambda row: (
            -row["mean_clarified_judge_accuracy"],
            len(row["system_prompt"].encode("utf-8")),
            row["generator_prompt_version"],
        ),
    )[0]
    frozen = {
        "selection_rule": "highest mean clarified boolean-judge accuracy across completed generator-model and judge-model cells; ties choose shortest UTF-8 system prompt, then version",
        "winner": winner,
        "judge_prompt_version": CLARIFIED_JUDGE_VERSION,
        "judge_system_prompt": JUDGE_PROMPTS[CLARIFIED_JUDGE_VERSION],
        "candidates": candidates,
    }
    (output_root / "selected-prompts.json").write_text(json.dumps(frozen, indent=2) + "\n")


async def run(args) -> None:
    output_root = args.output_root.resolve()
    cohort_path = output_root / "input" / "dev-100.parquet"
    cohort = prepare_cohort(args.input.resolve(), cohort_path)
    prompts = generation_system_prompts(args.include_revision)
    write_provenance(
        output_root, args.input.resolve(), cohort,
        list(args.generator_models), list(args.judge_models), prompts,
    )
    print(f"prepared {len(cohort)} rows -> {cohort_path}")
    if args.stage == "prepare":
        return
    generation_paths = []
    for model in args.generator_models:
        for version, system in prompts.items():
            path = output_root / "generations" / f"{model_slug(model)}__{version}.parquet"
            if args.stage in ("all", "generate"):
                path = await run_generation(
                    cohort, model, version, system, output_root,
                    args.concurrency, args.checkpoint_every,
                )
            if path.exists():
                generation_paths.append(path)
            elif args.stage == "judge":
                raise FileNotFoundError(f"missing generation artifact: {path}")
    if args.stage in ("all", "judge"):
        for generation_path in generation_paths:
            for judge_model in args.judge_models:
                for version, system in JUDGE_PROMPTS.items():
                    await run_judgment(
                        generation_path, judge_model, version, system,
                        output_root, args.concurrency, args.checkpoint_every,
                    )
        freeze_selected_prompt(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FROZEN_INPUT)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--generator-models", nargs="+", default=GENERATOR_MODELS)
    parser.add_argument("--judge-models", nargs="+", default=JUDGE_MODELS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--stage", choices=("prepare", "generate", "judge", "all"), default="all")
    parser.add_argument(
        "--include-revision", action="store_true",
        help=("Also test the one authorized revision: original plus "
              "'Begin your answer with Entailment or Contradiction.'"),
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")
    if len(set(args.generator_models)) != len(args.generator_models):
        parser.error("generator models must be unique")
    if len(set(args.judge_models)) != len(args.judge_models):
        parser.error("judge models must be unique")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
