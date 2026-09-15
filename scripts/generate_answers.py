#!/usr/bin/env python3
"""Generate grounded answers from each packaged top-four retrieval context."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config
from rag_eval.llm import (
    client,
    completion_text,
    effective_reasoning_effort,
    model_slug,
    reasoning_arguments,
    response_route,
    text_sha256,
    usage_fields,
    with_retries,
)


PROMPT_VERSION = "grounded-answer-v2-openrouter"
INSTRUCTIONS = """You are a question-answering assistant.
Answer the question using only the supplied context. Be concise. If the answer
is a number, return just the number. Otherwise answer in one or two sentences.
If the context is insufficient, say that the answer cannot be determined from
the supplied context."""


def context_for(row) -> str:
    for column in ("retrieved_context", "neural_retrieved"):
        if hasattr(row, column):
            return str(getattr(row, column))
    return "\n\n".join(
        str(getattr(row, f"excerpt_{rank}"))
        for rank in range(1, 5)
        if str(getattr(row, f"excerpt_{rank}")).strip()
    )


def output_columns() -> list[str]:
    return [
        "dataset", "query_id", "retrieval_method", "question", "golden_answer",
        "generator_model", "generator_reasoning_effort", "prompt_version",
        "api_backend",
        "generated_answer", "generation_error", "response_id", "input_tokens",
        "output_tokens", "total_tokens", "response_model", "response_provider",
        "retrieval_context_sha256",
    ]


async def generate_one(api, semaphore, dataset, row, model, reasoning_effort):
    base = {
        "dataset": dataset,
        "query_id": str(row.query_id),
        "retrieval_method": str(row.retrieval_method),
        "question": str(row.question),
        "golden_answer": str(row.golden_answer),
        "generator_model": model,
        "generator_reasoning_effort": effective_reasoning_effort(model, reasoning_effort),
        "prompt_version": PROMPT_VERSION,
        "api_backend": "openrouter",
    }
    context = context_for(row)
    base["retrieval_context_sha256"] = text_sha256(context)
    if not context:
        return base | {
            "generated_answer": "Cannot be determined from the supplied context.",
            "generation_error": "", "response_id": "",
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "response_model": "", "response_provider": "",
        }

    async with semaphore:
        try:
            response = await with_retries(lambda: api.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content":
                     f"Context:\n{context}\n\nQuestion: {row.question}\n\nAnswer:"},
                ],
                max_tokens=2048,
                **reasoning_arguments(model, reasoning_effort),
            ))
            answer = completion_text(response)
            if not answer:
                raise ValueError("model returned empty output")
            return base | {
                "generated_answer": answer, "generation_error": "",
                "response_id": str(response.id), **usage_fields(response),
                **response_route(response),
            }
        except Exception as error:
            return base | {
                "generated_answer": "", "generation_error": type(error).__name__,
                "response_id": "", "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "response_model": "", "response_provider": "",
            }


async def run(args) -> Path:
    source = args.input or ROOT / "data" / args.dataset / "retrieval_results.parquet"
    frame = pd.read_parquet(source)
    if "dataset" in frame:
        frame = frame[frame.dataset.astype(str).str.lower().eq(args.dataset)].copy()
    if args.packaged_only:
        questions = pd.read_parquet(
            ROOT / "data" / args.dataset / "questions.parquet",
            columns=["question", "golden_answer"],
        )
        packaged_keys = set(
            questions.astype(str).itertuples(index=False, name=None)
        )
        frame = frame[[
            key in packaged_keys
            for key in frame[["question", "golden_answer"]]
            .astype(str).itertuples(index=False, name=None)
        ]].copy()
    if "retrieval_method" not in frame:
        if not args.method_name:
            raise ValueError("input has no retrieval_method; pass --method-name")
        frame["retrieval_method"] = args.method_name
    if args.methods:
        frame = frame[frame.retrieval_method.isin(args.methods)].copy()
    if "query_id" not in frame:
        frame["query_id"] = [
            f"external-{args.dataset}-{index}" for index in frame.index
        ]
    sort_columns = [
        column
        for column in ("source_row", "query_id", "retrieval_method")
        if column in frame
    ]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit)

    slug = model_slug(args.model)
    target = (
        args.output
        or args.results_root / args.dataset / "generations" / f"{slug}.parquet"
    )
    checkpoint = target.parent / ".checkpoints" / target.name
    prior_path = target if target.exists() else checkpoint
    prior = pd.read_parquet(prior_path) if prior_path.exists() else pd.DataFrame()
    if not prior.empty:
        if set(prior.generator_model.astype(str)) != {args.model}:
            raise ValueError(f"existing output at {prior_path} uses another model")
        expected_effort = effective_reasoning_effort(args.model, args.reasoning_effort)
        expected_config = {
            "generator_reasoning_effort": expected_effort,
            "prompt_version": PROMPT_VERSION,
            "api_backend": "openrouter",
        }
        for column, expected in expected_config.items():
            if column not in prior or set(prior[column].astype(str)) != {expected}:
                raise ValueError(
                    f"existing output at {prior_path} has incompatible {column}"
                )
        expected_contexts = {
            (str(row.query_id), str(row.retrieval_method)): text_sha256(context_for(row))
            for row in frame.itertuples(index=False)
        }
        if "retrieval_context_sha256" not in prior or any(
            str(row.retrieval_context_sha256)
            != expected_contexts.get((str(row.query_id), str(row.retrieval_method)))
            for row in prior.itertuples(index=False)
        ):
            raise ValueError(
                f"existing output at {prior_path} uses different retrieval contexts"
            )
        successful = prior[prior.generation_error.fillna("").eq("")]
    else:
        successful = prior
    records = {
        (str(row.query_id), str(row.retrieval_method)): row._asdict()
        for row in successful.itertuples(index=False)
    }
    pending = [row for row in frame.itertuples(index=False)
               if (str(row.query_id), str(row.retrieval_method)) not in records]
    print(f"{args.dataset}/{args.model}: {len(records):,} done, "
          f"{len(pending):,} pending", flush=True)

    api = client()
    semaphore = asyncio.Semaphore(args.concurrency)
    for offset in range(0, len(pending), args.checkpoint_every):
        batch = pending[offset:offset + args.checkpoint_every]
        completed = await asyncio.gather(*(
            generate_one(api, semaphore, args.dataset, row, args.model,
                         args.reasoning_effort)
            for row in batch
        ))
        records.update({(row["query_id"], row["retrieval_method"]): row
                        for row in completed})
        current = pd.DataFrame(records.values(), columns=output_columns())
        atomic_parquet(current, checkpoint)
        print(f"  {len(records):,}/{len(frame):,}", flush=True)
    await api.close()

    result = pd.DataFrame(records.values(), columns=output_columns())
    result = result.sort_values(["query_id", "retrieval_method"]).reset_index(drop=True)
    errors = result.generation_error.fillna("").ne("")
    atomic_parquet(result, checkpoint)
    if errors.any():
        raise SystemExit(f"{int(errors.sum())} generation errors remain; rerun to retry")
    if len(result) != len(frame):
        raise SystemExit(f"expected {len(frame):,} rows, found {len(result):,}")
    atomic_parquet(result, target)
    print(f"complete -> {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=config())
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="none",
                        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--method-name")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--packaged-only", action="store_true")
    parser.add_argument("--results-root", type=Path, default=ROOT / "artifacts" / "e2e")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
