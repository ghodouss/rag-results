#!/usr/bin/env python3
"""Independently judge generated answers against the packaged reference answers."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config
from rag_eval.llm import (
    client,
    completion_text,
    effective_reasoning_effort,
    json_object_text,
    model_slug,
    reasoning_arguments,
    response_route,
    text_sha256,
    usage_fields,
    with_retries,
)


PROMPT_VERSIONS = {
    "binary": "reference-answer-binary-judge-v2-openrouter",
    "scale-1-5": "reference-answer-scale-1-5-judge-v1-openrouter",
}
BINARY_INSTRUCTIONS = """You are an expert answer evaluator. Judge whether the candidate
answer correctly answers the question relative to the reference answer.

Mark CORRECT when the candidate does not disagree with the reference and
adequately answers the question. Exact wording is not required, and a concise
true/false answer can match a verbose reference. Mark INCORRECT when the
candidate contradicts the reference, gives a materially different answer,
declines to answer, or says the answer cannot be determined.

Return only a JSON object with string fields `verdict` and `reason`. The verdict
must be exactly `CORRECT` or `INCORRECT`."""
SCALE_INSTRUCTIONS = """You are an expert answer evaluator. Score how well the
candidate answer answers the question relative to the reference answer.

Use this scale:
1 = fully incorrect, contradictory, or no usable answer
2 = mostly incorrect, with only minor correct information
3 = partially correct but missing or misstating important information
4 = mostly correct, with only a minor omission or imprecision
5 = fully correct and adequate; exact wording is not required

Return only a JSON object with fields `score` and `reason`. The score must be an
integer from 1 through 5 and the reason must be a concise string."""


class BinaryJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["CORRECT", "INCORRECT"]
    reason: str


class ScaleJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: Literal[1, 2, 3, 4, 5]
    reason: str


def judgment_contract(mode: str):
    if mode == "binary":
        return BINARY_INSTRUCTIONS, BinaryJudgment, "binary_judgment"
    if mode == "scale-1-5":
        return SCALE_INSTRUCTIONS, ScaleJudgment, "scale_judgment"
    raise ValueError(f"unsupported judge mode: {mode}")


def output_columns() -> list[str]:
    return [
        "dataset", "query_id", "retrieval_method", "question", "golden_answer",
        "generator_model", "generator_reasoning_effort", "generated_answer",
        "generation_error", "judge_model", "judge_reasoning_effort",
        "judge_mode", "judge_prompt_version", "api_backend", "verdict", "score",
        "reason", "judge_error", "judge_fallback",
        "judge_response_id", "judge_input_tokens", "judge_output_tokens",
        "judge_total_tokens", "judge_response_model", "judge_response_provider",
        "generation_sha256",
    ]


async def judge_one(
    api, semaphore, row, model, reasoning_effort, judge_mode="binary"
):
    instructions, result_model, schema_name = judgment_contract(judge_mode)
    base = {
        "dataset": str(row.dataset), "query_id": str(row.query_id),
        "retrieval_method": str(row.retrieval_method), "question": str(row.question),
        "golden_answer": str(row.golden_answer),
        "generator_model": str(row.generator_model),
        "generator_reasoning_effort": str(row.generator_reasoning_effort),
        "generated_answer": str(row.generated_answer),
        "generation_error": str(row.generation_error), "judge_model": model,
        "judge_reasoning_effort": effective_reasoning_effort(model, reasoning_effort),
        "judge_mode": judge_mode,
        "judge_prompt_version": PROMPT_VERSIONS[judge_mode],
        "api_backend": "openrouter",
        "generation_sha256": text_sha256(
            f"{row.generated_answer}\n{row.generation_error}"
        ),
    }
    if str(row.generation_error):
        return base | {
            "verdict": "INCORRECT" if judge_mode == "binary" else "",
            "score": None,
            "reason": "Answer generation failed.",
            "judge_error": "generation_error", "judge_response_id": "",
            "judge_fallback": "",
            "judge_input_tokens": 0, "judge_output_tokens": 0,
            "judge_total_tokens": 0, "judge_response_model": "",
            "judge_response_provider": "",
        }

    prompt = (f"Question: {row.question}\n\nReference answer: {row.golden_answer}"
              f"\n\nCandidate answer: {row.generated_answer}")
    async with semaphore:
        try:
            extra_body = reasoning_arguments(model, reasoning_effort).get(
                "extra_body", {}
            )
            request_options = {}
            if model != "anthropic/claude-sonnet-4":
                extra_body["provider"] = {"require_parameters": True}
                request_options["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": result_model.model_json_schema(),
                    },
                }
            if extra_body:
                request_options["extra_body"] = extra_body
            max_tokens = 1024 if model == "anthropic/claude-sonnet-4" else 256
            for validation_attempt in range(4):
                response = await with_retries(lambda: api.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    **request_options,
                ))
                try:
                    result_text = json_object_text(completion_text(response))
                    if model == "anthropic/claude-sonnet-4":
                        payload = json.loads(result_text)
                        if judge_mode == "scale-1-5" and "reason" not in payload:
                            payload["reason"] = next(
                                (
                                    value for key, value in payload.items()
                                    if key != "score" and isinstance(value, str)
                                ),
                                "No reason returned.",
                            )
                        if judge_mode == "scale-1-5":
                            payload = {
                                "score": payload.get("score"),
                                "reason": payload.get("reason"),
                            }
                        parsed = result_model.model_validate(payload)
                    else:
                        parsed = result_model.model_validate_json(result_text)
                    break
                except ValueError:
                    message = response.choices[0].message
                    if response.choices[0].finish_reason == "content_filter":
                        usage = usage_fields(response)
                        return base | {
                            "verdict": "INCORRECT" if judge_mode == "binary" else "",
                            "score": None,
                            "reason": (
                                "Judge returned no result because of content filtering; "
                                "excluded from the 1--5 mean by evaluation policy."
                            ),
                            "judge_error": "",
                            "judge_fallback": "content_filter_score_excluded",
                            "judge_response_id": str(response.id),
                            "judge_input_tokens": usage["input_tokens"],
                            "judge_output_tokens": usage["output_tokens"],
                            "judge_total_tokens": usage["total_tokens"],
                            "judge_response_model": response_route(response)["response_model"],
                            "judge_response_provider": response_route(response)["response_provider"],
                        }
                    if validation_attempt == 3:
                        raise ValueError(
                            "model returned invalid structured output: "
                            f"content={completion_text(response)[:500]!r}, "
                            f"refusal={getattr(message, 'refusal', None)!r}, "
                            f"finish_reason={response.choices[0].finish_reason!r}, "
                            f"reasoning_length={len(str(getattr(message, 'reasoning', '') or ''))}"
                        )
                    await asyncio.sleep(0.25 * (validation_attempt + 1))
            usage = usage_fields(response)
            return base | {
                "verdict": getattr(parsed, "verdict", ""),
                "score": getattr(parsed, "score", None),
                "reason": parsed.reason,
                "judge_error": "", "judge_fallback": "",
                "judge_response_id": str(response.id),
                "judge_input_tokens": usage["input_tokens"],
                "judge_output_tokens": usage["output_tokens"],
                "judge_total_tokens": usage["total_tokens"],
                "judge_response_model": response_route(response)["response_model"],
                "judge_response_provider": response_route(response)["response_provider"],
            }
        except Exception as error:
            return base | {
                "verdict": "", "score": None, "reason": "",
                "judge_error": f"{type(error).__name__}: {str(error)[:500]}",
                "judge_fallback": "",
                "judge_response_id": "", "judge_input_tokens": 0,
                "judge_output_tokens": 0, "judge_total_tokens": 0,
                "judge_response_model": "", "judge_response_provider": "",
            }


def write_summary(frame: pd.DataFrame, target: Path) -> None:
    rows = []
    for method, group in frame.groupby("retrieval_method", sort=True):
        judged = group.judge_error.fillna("").eq("")
        mode = str(group.judge_mode.iloc[0])
        row = {
            "dataset": str(group.dataset.iloc[0]),
            "retrieval_method": str(method),
            "generator_model": str(group.generator_model.iloc[0]),
            "judge_model": str(group.judge_model.iloc[0]),
            "judge_mode": mode,
            "questions": int(len(group)),
            "generation_errors": int(group.generation_error.fillna("").ne("").sum()),
            "judge_errors": int((~judged).sum()),
            "judge_fallbacks": int(
                group.judge_fallback.fillna("").astype(str).ne("").sum()
            ),
            "judged_questions": int(judged.sum()),
            "input_tokens": int(group.judge_input_tokens.sum()),
            "output_tokens": int(group.judge_output_tokens.sum()),
        }
        if mode == "binary":
            row |= {
                "correct": int(group.loc[judged].verdict.eq("CORRECT").sum()),
                "accuracy": float(
                    group.loc[judged].verdict.eq("CORRECT").mean()
                ),
            }
        else:
            scores = pd.to_numeric(group.loc[judged, "score"])
            row |= {
                "mean_score": float(scores.mean()),
                "scored_questions": int(scores.notna().sum()),
                "ignored_scores": int(scores.isna().sum()),
                **{
                    f"score_{score}": int(scores.eq(score).sum())
                    for score in range(1, 6)
                },
            }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(target.with_suffix(".summary.csv"), index=False)
    target.with_suffix(".summary.json").write_text(
        json.dumps(rows, indent=2) + "\n")


async def run(args) -> Path:
    generator_slug = model_slug(args.generator_model)
    source = (args.input or args.results_root / args.dataset / "generations" /
              f"{generator_slug}.parquet")
    generations = pd.read_parquet(source)
    if "dataset" in generations:
        generations = generations[
            generations.dataset.astype(str).str.lower().eq(args.dataset)
        ].copy()
    if args.packaged_only:
        questions = pd.read_parquet(
            ROOT / "data" / args.dataset / "questions.parquet",
            columns=["question", "golden_answer"],
        )
        packaged_keys = set(
            questions.astype(str).itertuples(index=False, name=None)
        )
        generations = generations[[
            key in packaged_keys
            for key in generations[["question", "golden_answer"]]
            .astype(str).itertuples(index=False, name=None)
        ]].copy()
    generations = generations[
        generations.generator_model.astype(str).eq(args.generator_model)
    ].copy()
    if args.methods:
        generations = generations[
            generations.retrieval_method.astype(str).isin(args.methods)
        ].copy()
    expected_model = set(generations.generator_model.astype(str))
    if expected_model != {args.generator_model}:
        raise ValueError(f"generation file model mismatch: {expected_model}")
    if args.limit:
        generations = generations.head(args.limit)

    judge_slug = model_slug(args.judge_model)
    target = (args.output or args.results_root / args.dataset / "judgments" /
              f"{generator_slug}__judge-{judge_slug}.parquet")
    checkpoint = target.parent / ".checkpoints" / target.name
    prior_path = target if target.exists() else checkpoint
    prior = pd.read_parquet(prior_path) if prior_path.exists() else pd.DataFrame()
    if not prior.empty:
        if set(prior.judge_model.astype(str)) != {args.judge_model}:
            raise ValueError(f"existing output at {prior_path} uses another judge")
        expected_effort = effective_reasoning_effort(
            args.judge_model, args.reasoning_effort
        )
        expected_config = {
            "generator_model": args.generator_model,
            "judge_reasoning_effort": expected_effort,
            "judge_mode": args.judge_mode,
            "judge_prompt_version": PROMPT_VERSIONS[args.judge_mode],
            "api_backend": "openrouter",
        }
        for column, expected in expected_config.items():
            if column not in prior or set(prior[column].astype(str)) != {expected}:
                raise ValueError(
                    f"existing output at {prior_path} has incompatible {column}"
                )
        expected_generations = {
            (str(row.query_id), str(row.retrieval_method)): text_sha256(
                f"{row.generated_answer}\n{row.generation_error}"
            )
            for row in generations.itertuples(index=False)
        }
        if "generation_sha256" not in prior or any(
            str(row.generation_sha256)
            != expected_generations.get((str(row.query_id), str(row.retrieval_method)))
            for row in prior.itertuples(index=False)
        ):
            raise ValueError(
                f"existing output at {prior_path} uses different generated answers"
            )
        if (
            args.judge_mode == "scale-1-5"
            and args.judge_model == "anthropic/claude-sonnet-4"
        ):
            failed = prior.judge_error.fillna("").ne("")
            prior.loc[failed, "score"] = 1
            prior.loc[failed, "reason"] = (
                "Judge did not return a usable result; counted as incorrect by "
                "evaluation policy."
            )
            prior.loc[failed, "judge_error"] = ""
            prior.loc[failed, "judge_fallback"] = "judge_failure_scored_incorrect"
        successful = prior[prior.judge_error.fillna("").eq("")]
    else:
        successful = prior
    records = {
        (str(row.query_id), str(row.retrieval_method)): row._asdict()
        for row in successful.itertuples(index=False)
    }
    pending = [row for row in generations.itertuples(index=False)
               if (str(row.query_id), str(row.retrieval_method)) not in records]
    print(f"{args.dataset}/{args.generator_model}->{args.judge_model}: "
          f"{len(records):,} done, {len(pending):,} pending", flush=True)

    api = client()
    semaphore = asyncio.Semaphore(args.concurrency)
    for offset in range(0, len(pending), args.checkpoint_every):
        batch = pending[offset:offset + args.checkpoint_every]
        completed = await asyncio.gather(*(
            judge_one(
                api, semaphore, row, args.judge_model, args.reasoning_effort,
                args.judge_mode,
            )
            for row in batch
        ))
        records.update({(row["query_id"], row["retrieval_method"]): row
                        for row in completed})
        current = pd.DataFrame(records.values(), columns=output_columns())
        atomic_parquet(current, checkpoint)
        print(f"  {len(records):,}/{len(generations):,}", flush=True)
    await api.close()

    result = pd.DataFrame(records.values(), columns=output_columns())
    result = result.sort_values(["query_id", "retrieval_method"]).reset_index(drop=True)
    errors = result.judge_error.fillna("").ne("")
    atomic_parquet(result, checkpoint)
    if errors.any():
        raise SystemExit(f"{int(errors.sum())} judge errors remain; rerun to retry")
    if len(result) != len(generations):
        raise SystemExit(f"expected {len(generations):,} rows, found {len(result):,}")
    atomic_parquet(result, target)
    write_summary(result, target)
    print(f"complete -> {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=config())
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-mode", default="binary", choices=("binary", "scale-1-5")
    )
    parser.add_argument("--reasoning-effort", default="none",
                        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--packaged-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-root", type=Path, default=ROOT / "artifacts" / "e2e")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
