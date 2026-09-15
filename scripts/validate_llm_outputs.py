#!/usr/bin/env python3
"""Validate completeness and join integrity for generation and judgment outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, config
from rag_eval.llm import effective_reasoning_effort, model_slug


DEFAULT_MODELS = ("gpt-4o-mini", "gpt-5.6-luna")


def keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame.query_id.astype(str), frame.retrieval_method.astype(str)))


def validate(dataset: str, model: str) -> None:
    retrieval = pd.read_parquet(ROOT / "data" / dataset / "retrieval_results.parquet")
    expected_keys = keys(retrieval)
    expected_effort = effective_reasoning_effort(model, "low")
    slug = model_slug(model)

    generation_path = ROOT / "results" / dataset / "generations" / f"{slug}.parquet"
    generation = pd.read_parquet(generation_path)
    if len(generation) != len(expected_keys) or keys(generation) != expected_keys:
        raise ValueError(f"{dataset}/{model}: incomplete generation key set")
    if generation.duplicated(["query_id", "retrieval_method"]).any():
        raise ValueError(f"{dataset}/{model}: duplicate generation rows")
    if set(generation.generator_model.astype(str)) != {model}:
        raise ValueError(f"{dataset}/{model}: generator identity mismatch")
    if set(generation.generator_reasoning_effort.astype(str)) != {expected_effort}:
        raise ValueError(f"{dataset}/{model}: generation reasoning setting mismatch")
    if generation.generation_error.fillna("").ne("").any():
        raise ValueError(f"{dataset}/{model}: unresolved generation errors")
    if generation.generated_answer.fillna("").str.strip().eq("").any():
        raise ValueError(f"{dataset}/{model}: empty generated answers")

    judgment_path = (ROOT / "results" / dataset / "judgments" /
                     f"{slug}__judge-{slug}.parquet")
    judgment = pd.read_parquet(judgment_path)
    if len(judgment) != len(expected_keys) or keys(judgment) != expected_keys:
        raise ValueError(f"{dataset}/{model}: incomplete judgment key set")
    if judgment.duplicated(["query_id", "retrieval_method"]).any():
        raise ValueError(f"{dataset}/{model}: duplicate judgment rows")
    if set(judgment.judge_model.astype(str)) != {model}:
        raise ValueError(f"{dataset}/{model}: judge identity mismatch")
    if set(judgment.judge_reasoning_effort.astype(str)) != {expected_effort}:
        raise ValueError(f"{dataset}/{model}: judge reasoning setting mismatch")
    if judgment.judge_error.fillna("").ne("").any():
        raise ValueError(f"{dataset}/{model}: unresolved judge errors")
    if not set(judgment.verdict.astype(str)) <= {"CORRECT", "INCORRECT"}:
        raise ValueError(f"{dataset}/{model}: invalid verdict")
    if judgment.reason.fillna("").str.strip().eq("").any():
        raise ValueError(f"{dataset}/{model}: empty judgment reason")

    summary = pd.read_csv(judgment_path.with_suffix(".summary.csv"))
    for method, group in judgment.groupby("retrieval_method"):
        row = summary[summary.retrieval_method == method]
        if len(row) != 1 or int(row.questions.iloc[0]) != len(group):
            raise ValueError(f"{dataset}/{model}/{method}: invalid summary count")
        accuracy = float(group.verdict.eq("CORRECT").mean())
        if abs(float(row.accuracy.iloc[0]) - accuracy) > 1e-12:
            raise ValueError(f"{dataset}/{model}/{method}: summary accuracy mismatch")
    print(f"{dataset}/{model}: valid; {len(judgment):,} generated and judged rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=config())
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            validate(dataset, model)


if __name__ == "__main__":
    main()
