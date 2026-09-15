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
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config
from rag_eval.llm import (
    client,
    effective_reasoning_effort,
    model_slug,
    reasoning_arguments,
    usage_fields,
    with_retries,
)


PROMPT_VERSION = "reference-answer-judge-v1"
INSTRUCTIONS = """You are an expert answer evaluator. Judge whether the candidate
answer correctly answers the question relative to the reference answer.

Mark CORRECT when the candidate does not disagree with the reference and
adequately answers the question. Exact wording is not required, and a concise
true/false answer can match a verbose reference. Mark INCORRECT when the
candidate contradicts the reference, gives a materially different answer,
declines to answer, or says the answer cannot be determined."""


class Judgment(BaseModel):
    verdict: Literal["CORRECT", "INCORRECT"]
    reason: str


def output_columns() -> list[str]:
    return [
        "dataset", "query_id", "retrieval_method", "question", "golden_answer",
        "generator_model", "generator_reasoning_effort", "generated_answer",
        "generation_error", "judge_model", "judge_reasoning_effort",
        "judge_prompt_version", "verdict", "reason", "judge_error",
        "judge_response_id", "judge_input_tokens", "judge_output_tokens",
        "judge_total_tokens",
    ]


async def judge_one(api, semaphore, row, model, reasoning_effort):
    base = {
        "dataset": str(row.dataset), "query_id": str(row.query_id),
        "retrieval_method": str(row.retrieval_method), "question": str(row.question),
        "golden_answer": str(row.golden_answer),
        "generator_model": str(row.generator_model),
        "generator_reasoning_effort": str(row.generator_reasoning_effort),
        "generated_answer": str(row.generated_answer),
        "generation_error": str(row.generation_error), "judge_model": model,
        "judge_reasoning_effort": effective_reasoning_effort(model, reasoning_effort),
        "judge_prompt_version": PROMPT_VERSION,
    }
    if str(row.generation_error):
        return base | {
            "verdict": "INCORRECT", "reason": "Answer generation failed.",
            "judge_error": "generation_error", "judge_response_id": "",
            "judge_input_tokens": 0, "judge_output_tokens": 0,
            "judge_total_tokens": 0,
        }

    prompt = (f"Question: {row.question}\n\nReference answer: {row.golden_answer}"
              f"\n\nCandidate answer: {row.generated_answer}")
    async with semaphore:
        try:
            response = await with_retries(lambda: api.responses.parse(
                model=model,
                instructions=INSTRUCTIONS,
                input=prompt,
                text_format=Judgment,
                max_output_tokens=2048,
                store=False,
                **reasoning_arguments(model, reasoning_effort),
            ))
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("model returned no parsed judgment")
            usage = usage_fields(response)
            return base | {
                "verdict": parsed.verdict, "reason": parsed.reason,
                "judge_error": "", "judge_response_id": str(response.id),
                "judge_input_tokens": usage["input_tokens"],
                "judge_output_tokens": usage["output_tokens"],
                "judge_total_tokens": usage["total_tokens"],
            }
        except Exception as error:
            return base | {
                "verdict": "", "reason": "", "judge_error": type(error).__name__,
                "judge_response_id": "", "judge_input_tokens": 0,
                "judge_output_tokens": 0, "judge_total_tokens": 0,
            }


def write_summary(frame: pd.DataFrame, target: Path) -> None:
    rows = []
    for method, group in frame.groupby("retrieval_method", sort=True):
        judged = group.judge_error.fillna("").eq("")
        rows.append({
            "dataset": str(group.dataset.iloc[0]),
            "retrieval_method": str(method),
            "generator_model": str(group.generator_model.iloc[0]),
            "judge_model": str(group.judge_model.iloc[0]),
            "questions": int(len(group)),
            "generation_errors": int(group.generation_error.fillna("").ne("").sum()),
            "judge_errors": int((~judged).sum()),
            "judged_questions": int(judged.sum()),
            "correct": int(group.loc[judged].verdict.eq("CORRECT").sum()),
            "accuracy": float(group.loc[judged].verdict.eq("CORRECT").mean()),
            "input_tokens": int(group.judge_input_tokens.sum()),
            "output_tokens": int(group.judge_output_tokens.sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(target.with_suffix(".summary.csv"), index=False)
    target.with_suffix(".summary.json").write_text(
        json.dumps(rows, indent=2) + "\n")


async def run(args) -> Path:
    generator_slug = model_slug(args.generator_model)
    source = (args.input or ROOT / "artifacts" / "e2e" / args.dataset / "generations" /
              f"{generator_slug}.parquet")
    generations = pd.read_parquet(source)
    expected_model = set(generations.generator_model.astype(str))
    if expected_model != {args.generator_model}:
        raise ValueError(f"generation file model mismatch: {expected_model}")
    if args.limit:
        generations = generations.head(args.limit)

    judge_slug = model_slug(args.judge_model)
    target = (args.output or ROOT / "artifacts" / "e2e" / args.dataset / "judgments" /
              f"{generator_slug}__judge-{judge_slug}.parquet")
    checkpoint = target.parent / ".checkpoints" / target.name
    prior_path = target if target.exists() else checkpoint
    prior = pd.read_parquet(prior_path) if prior_path.exists() else pd.DataFrame()
    if not prior.empty:
        if set(prior.judge_model.astype(str)) != {args.judge_model}:
            raise ValueError(f"existing output at {prior_path} uses another judge")
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
            judge_one(api, semaphore, row, args.judge_model, args.reasoning_effort)
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
    parser.add_argument("--reasoning-effort", default="low",
                        choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
