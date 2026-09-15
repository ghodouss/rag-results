#!/usr/bin/env python3
"""Import the two neural retrieval conditions from Chris's archive."""
from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet


IDENTITY_COLUMNS = ["dataset", "question", "golden_answer"]
GENERATOR_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
CONDITIONS = {
    "e5": {
        "retrieval_method": "intfloat/e5-small-v2",
        "retrieval_column": "neural_retrieved",
        "generation_prefix": "neural",
        "context_length_column": "neural_context_len",
    },
    "openai": {
        "retrieval_method": "openai/embedding-model-unspecified",
        "retrieval_column": "openai_embed_retrieved",
        "generation_prefix": "openai_embed",
        "context_length_column": "openai_embed_context_len",
    },
}
EXCERPT_SEPARATOR = re.compile(r"\n\s*\.\.\.\s*\n")


def read_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    generations = []
    with zipfile.ZipFile(path) as archive:
        for dataset in ("bioasq", "contractnli", "finqa"):
            name = f"{dataset}_generation.parquet"
            generations.append(pd.read_parquet(io.BytesIO(archive.read(name))))
        with archive.open("retrieval_results.csv.gz") as compressed:
            with gzip.GzipFile(fileobj=compressed) as csv_stream:
                retrieval = pd.read_csv(csv_stream, keep_default_na=False)
    return pd.concat(generations, ignore_index=True), retrieval


def validate_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label}: missing columns: {', '.join(missing)}")
    null_columns = frame[columns].columns[frame[columns].isna().any()].tolist()
    if null_columns:
        raise ValueError(f"{label}: null values in: {', '.join(null_columns)}")


def add_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["dataset"] = result.dataset.astype(str).str.lower()
    ordinal = result.groupby("dataset", sort=False).cumcount()
    result["query_id"] = [
        f"chris-{dataset}-{index:06d}"
        for dataset, index in zip(result.dataset, ordinal)
    ]
    return result


def validate_alignment(generation: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    for dataset in sorted(retrieval.dataset.unique()):
        generated = generation[generation.dataset.eq(dataset)][IDENTITY_COLUMNS]
        retrieved = retrieval[retrieval.dataset.eq(dataset)][IDENTITY_COLUMNS]
        if not generated.reset_index(drop=True).equals(retrieved.reset_index(drop=True)):
            raise ValueError(f"{dataset}: generation and retrieval rows do not align")


def long_generations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_identity(frame)
    rows = []
    for condition in CONDITIONS.values():
        for generator, model in GENERATOR_MODELS.items():
            answer_column = f"{condition['generation_prefix']}_{generator}"
            selected = frame[[
                "dataset", "query_id", "question", "golden_answer",
                condition["context_length_column"], answer_column,
            ]].copy()
            selected["retrieval_method"] = condition["retrieval_method"]
            selected["generator_model"] = model
            selected["generator_reasoning_effort"] = "not_recorded"
            selected["prompt_version"] = "chris-neural-conditions"
            selected["api_backend"] = "not_recorded"
            selected["generation_error"] = ""
            selected = selected.rename(columns={
                condition["context_length_column"]: "retrieved_context_length",
                answer_column: "generated_answer",
            })
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def long_retrieval(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_identity(frame)
    rows = []
    for condition in CONDITIONS.values():
        context_column = condition["retrieval_column"]
        excerpts = frame[context_column].map(EXCERPT_SEPARATOR.split)
        if not excerpts.map(lambda values: 1 <= len(values) <= 4).all():
            raise ValueError("neural retrieval must contain one to four excerpts")
        selected = frame[[
            "dataset", "query_id", "question", "golden_answer", context_column,
        ]].copy()
        selected["retrieval_method"] = condition["retrieval_method"]
        selected["retrieved_count"] = excerpts.map(len)
        for rank in range(1, 5):
            selected[f"excerpt_{rank}"] = excerpts.map(
                lambda values: values[rank - 1] if len(values) >= rank else ""
            )
        selected = selected.rename(columns={context_column: "retrieved_context"})
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="path to neural-conditions.zip")
    args = parser.parse_args()

    generation, retrieval = read_archive(args.archive)
    generation = add_identity(generation)
    retrieval = add_identity(retrieval)
    required_generation = IDENTITY_COLUMNS.copy()
    required_retrieval = IDENTITY_COLUMNS.copy()
    for condition in CONDITIONS.values():
        required_generation.append(condition["context_length_column"])
        required_retrieval.append(condition["retrieval_column"])
        required_generation.extend(
            f"{condition['generation_prefix']}_{generator}"
            for generator in GENERATOR_MODELS
        )
    validate_columns(generation, required_generation, "generation")
    validate_columns(retrieval, required_retrieval, "retrieval")
    validate_alignment(generation, retrieval)

    target = ROOT / "data" / "neural-embeddings"
    atomic_parquet(long_generations(generation), target / "generation_results.parquet")
    atomic_parquet(long_retrieval(retrieval), target / "retrieval_results.parquet")
    print(f"Imported {len(retrieval):,} questions x 2 retrieval conditions")


if __name__ == "__main__":
    main()
