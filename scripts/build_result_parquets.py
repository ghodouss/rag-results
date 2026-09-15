#!/usr/bin/env python3
"""Build one analysis-ready, wide result Parquet per dataset and retriever."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "artifacts" / "e2e"
sys.path.insert(0, str(ROOT / "src"))
from rag_eval.common import atomic_parquet, normalize  # noqa: E402

DATASETS = ("contractnli", "bioasq", "finqa")
METHODS = ("sturdy", "bm25", "e5", "openai")
EXACT_DATASETS = ("contractnli", "bioasq")
METHODS_BY_DATASET = {
    "contractnli": ("sturdy", "bm25", "e5", "openai"),
    "bioasq": ("sturdy", "bm25", "e5", "openai"),
    "finqa": ("sturdy", "bm25"),
}
NEURAL_FILES = {
    "e5": "e5-small-v2.parquet",
    "openai": "openai-embedding-model-unspecified.parquet",
}


def slug(value: object) -> str:
    """Return a stable, column-name-safe identifier."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def require_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate rows for {keys}")


def load_retrieval(dataset: str, method: str) -> pd.DataFrame:
    if method in NEURAL_FILES:
        questions = pd.read_parquet(ROOT / "data" / dataset / "questions.parquet")
        retrieval = pd.read_parquet(
            ROOT / "data" / dataset / "retrieval" / NEURAL_FILES[method]
        )
        retrieval["source_row"] = (
            retrieval.query_id.astype(str).str.rsplit("-", n=1).str[-1].astype("int64")
        )
        payload = retrieval.drop(columns=[
            column for column in ("dataset", "query_id", "question", "golden_answer")
            if column in retrieval
        ])
        frame = questions.merge(payload, on="source_row", how="left", validate="one_to_one")
        if frame.excerpt_1.isna().all():
            raise ValueError(f"{dataset}/{method}: neural retrieval alignment failed")
        frame["retrieval_method"] = method
        frame["embedding_model"] = (
            "intfloat/e5-small-v2" if method == "e5"
            else "openai/embedding-model-unspecified"
        )
        frame["retrieval_error"] = ""
        frame["top_k"] = 4
        return frame
    if dataset == "contractnli":
        root = ROOT / "data/contractnli/retrieval"
        if method == "sturdy":
            paths = sorted(root.glob("sturdy-a4-r2.part-*.parquet"))
            if not paths:
                raise FileNotFoundError("missing ContractNLI Sturdy retrieval partitions")
            frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        else:
            frame = pd.read_parquet(root / "bm25.parquet")
    else:
        frame = pd.read_parquet(ROOT / "data" / dataset / "retrieval_results.parquet")
        frame = frame[frame.retrieval_method.astype(str).eq(method)].copy()
    require_unique(frame, ["query_id"], f"{dataset}/{method} retrieval")
    return frame


def retrieved_items(frame: pd.DataFrame) -> pd.Series:
    """Represent ranked retrievals as a native Parquet list-of-struct column."""
    records: list[list[dict[str, object]]] = []
    for row in frame.itertuples(index=False):
        items = []
        for rank in range(1, 5):
            text = getattr(row, f"excerpt_{rank}", "")
            if (pd.isna(text) or not str(text).strip()
                    or str(text).strip().lower() == "nan"):
                continue
            score = getattr(row, f"excerpt_{rank}_score", None)
            items.append({
                "rank": rank,
                "text": str(text),
                "unit_id": clean_string(getattr(row, f"excerpt_{rank}_unit_id", "")),
                "doc_id": clean_string(getattr(row, f"excerpt_{rank}_doc_id", "")),
                "score": None if score is None or pd.isna(score) else float(score),
            })
        records.append(items)
    return pd.Series(records, index=frame.index, dtype=object)


def clean_string(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def reconstructed_context(frame: pd.DataFrame) -> pd.Series:
    # This matches scripts/generate_answers.py exactly for BioASQ and FinQA.
    return frame.apply(
        lambda row: "\n\n".join(
            str(row[f"excerpt_{rank}"])
            for rank in range(1, 5)
            if (str(row[f"excerpt_{rank}"]).strip()
                and str(row[f"excerpt_{rank}"]).strip().lower() != "nan")
        ),
        axis=1,
    )


def contract_input(method: str) -> pd.DataFrame:
    condition = "sturdy-a4-r2-ranked-windows" if method == "sturdy" else "bm25-top4"
    path = RUNS_ROOT / "contractnli" / "input" / f"{condition}.parquet"
    frame = pd.read_parquet(path)
    require_unique(frame, ["query_id"], f"ContractNLI {condition} input")
    return frame


def full_doc_flags(dataset: str, frame: pd.DataFrame) -> pd.Series:
    documents = pd.read_parquet(ROOT / "data" / dataset / "documents.parquet")
    text_by_doc = (
        documents.groupby(documents.doc_id.astype(str)).text.apply(" ".join).to_dict()
    )

    def contains(row: pd.Series) -> bool:
        context = " ".join(
            text_by_doc.get(str(doc_id), "")
            for doc_id in json.loads(row.allowed_doc_ids_json)
        )
        gold = normalize(
            row.reference_evidence if dataset == "contractnli" else row.golden_answer
        )
        return bool(gold and gold in normalize(context))

    return frame.apply(contains, axis=1)


def add_exact_flags(dataset: str, method: str, source: pd.DataFrame,
                    result: pd.DataFrame) -> None:
    if dataset not in EXACT_DATASETS:
        return
    gold_column = "reference_evidence" if dataset == "contractnli" else "golden_answer"
    for rank in range(1, 5):
        target = f"golden_answer_exact_in_top_{rank}"
        if dataset == "contractnli" and method == "sturdy":
            contexts = source[f"containment_context_{rank}"].astype(str)
            result[target] = [
                bool(normalize(gold) and normalize(gold) in normalize(context))
                for gold, context in zip(result[gold_column], contexts)
            ]
        elif target in source:
            result[target] = source[target].astype(bool).to_numpy()
        else:
            contexts = source.apply(
                lambda row: " ".join(str(row[f"excerpt_{i}"])
                                     for i in range(1, rank + 1)),
                axis=1,
            )
            result[target] = [
                bool(normalize(gold) and normalize(gold) in normalize(context))
                for gold, context in zip(result[gold_column], contexts)
            ]
    full_doc = "golden_answer_exact_in_full_doc"
    if full_doc in source:
        result[full_doc] = source[full_doc].astype(bool).to_numpy()
    else:
        result[full_doc] = full_doc_flags(dataset, result).to_numpy()


def copy_if_present(source: pd.DataFrame, result: pd.DataFrame,
                    columns: Iterable[str]) -> None:
    for column in columns:
        if column in source:
            result[column] = source[column].to_numpy()


def base_frame(dataset: str, method: str) -> pd.DataFrame:
    source = load_retrieval(dataset, method).sort_values("source_row").reset_index(drop=True)
    result = pd.DataFrame(index=source.index)
    copy_if_present(source, result, (
        "dataset", "query_id", "split", "source_row", "question", "golden_answer",
        "allowed_doc_ids_json", "doc_id", "choice", "passage_ids", "doc_index",
        "context_hash", "retrieval_error", "retrieved_count", "top_k", "index_id",
        "index_name", "retrieval_level", "semantic_search_weight",
        "semantic_search_cutoff", "candidate_scope", "bm25_k1", "bm25_b",
        "embedding_model",
    ))
    result["retrieval_method"] = method
    result["retrieved_items"] = retrieved_items(source)

    if dataset == "contractnli" and method in {"sturdy", "bm25"}:
        inputs = contract_input(method).sort_values("source_row").reset_index(drop=True)
        if not inputs.query_id.astype(str).equals(result.query_id.astype(str)):
            raise ValueError(f"contractnli/{method}: retrieval and generation inputs differ")
        # ContractNLI labels are the answer target; the retrieval file's golden answer
        # is the supporting reference span used by exact containment.
        result["reference_evidence"] = result["golden_answer"]
        result["golden_answer"] = inputs["golden_answer"].to_numpy()
        result["retrieved_context"] = inputs["retrieved_context"].to_numpy()
        copy_if_present(inputs, result, (
            "selection_rank", "selection_version", "selection_key",
            "retrieval_context_sha256",
        ))
    elif dataset == "contractnli":
        result["reference_evidence"] = result["golden_answer"]
        result["golden_answer"] = result["choice"]
        result["retrieved_context"] = reconstructed_context(source)
    else:
        result["retrieved_context"] = reconstructed_context(source)

    add_exact_flags(dataset, method, source, result)
    require_unique(result, ["query_id"], f"{dataset}/{method} base")
    return result


GENERATION_FIELDS = (
    "generator_model", "generator_reasoning_effort", "prompt_version",
    "generator_prompt_version", "generator_system_prompt_sha256",
    "generator_user_prompt_sha256", "api_backend", "generated_answer",
    "generation_error", "response_id", "input_tokens", "output_tokens", "total_tokens",
    "response_model", "response_provider", "generation_sha256", "cohort_sha256",
)

JUDGMENT_FIELDS = (
    "judge_model", "judge_reasoning_effort", "judge_prompt_version",
    "judge_mode", "api_backend", "judge_fallback",
    "judge_system_prompt_sha256", "judge_user_prompt_sha256", "judge_correct",
    "score", "rationale", "verdict", "reason", "judge_error",
    "judge_response_id", "judge_input_tokens", "judge_output_tokens",
    "judge_total_tokens", "judge_response_model", "judge_response_provider",
)


def generation_paths(dataset: str, method: str) -> list[Path]:
    if dataset == "contractnli":
        condition = "sturdy-a4-r2-ranked-windows" if method == "sturdy" else "bm25-top4"
        root = RUNS_ROOT / "contractnli" / condition / "generations"
    else:
        root = RUNS_ROOT / dataset / "generations"
    return sorted(root.glob("*.parquet"))


def judgment_paths(dataset: str, method: str) -> list[Path]:
    if dataset == "contractnli":
        condition = "sturdy-a4-r2-ranked-windows" if method == "sturdy" else "bm25-top4"
        root = RUNS_ROOT / "contractnli" / condition / "judgments"
    else:
        root = RUNS_ROOT / dataset / "judgments"
    return sorted(root.glob("*.parquet"))


def method_rows(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "retrieval_method" in frame and method in set(frame.retrieval_method.astype(str)):
        frame = frame[frame.retrieval_method.astype(str).eq(method)].copy()
    require_unique(frame, ["query_id"], str(path.relative_to(ROOT)))
    return frame


def merge_namespaced(result: pd.DataFrame, source: pd.DataFrame, prefix: str,
                     fields: Iterable[str], label: str) -> pd.DataFrame:
    if set(source.query_id.astype(str)) != set(result.query_id.astype(str)):
        raise ValueError(f"{label}: query IDs differ from retrieval rows")
    available = [field for field in fields if field in source]
    addition = source[["query_id", *available]].copy()
    addition = addition.rename(columns={field: f"{prefix}__{field}" for field in available})
    merged = result.merge(addition, on="query_id", how="left", validate="one_to_one")
    if len(merged) != len(result):
        raise ValueError(f"{label}: row count changed while merging")
    return merged


def add_outputs(dataset: str, method: str, result: pd.DataFrame) -> pd.DataFrame:
    if method in NEURAL_FILES:
        return result
    for path in generation_paths(dataset, method):
        frame = method_rows(path, method)
        if len(frame) != len(result):
            raise ValueError(f"{path.relative_to(ROOT)}: expected {len(result)} rows")
        models = frame.generator_model.astype(str).unique()
        if len(models) != 1:
            raise ValueError(f"{path.relative_to(ROOT)}: expected one generator model")
        prefix = f"generation_{slug(models[0])}"
        result = merge_namespaced(result, frame, prefix, GENERATION_FIELDS, str(path))

    for path in judgment_paths(dataset, method):
        frame = method_rows(path, method)
        if len(frame) != len(result):
            raise ValueError(f"{path.relative_to(ROOT)}: expected {len(result)} rows")
        generators = frame.generator_model.astype(str).unique()
        judges = frame.judge_model.astype(str).unique()
        if len(generators) != 1 or len(judges) != 1:
            raise ValueError(f"{path.relative_to(ROOT)}: expected one generator and judge")
        prefix = f"judgment_{slug(generators[0])}__{slug(judges[0])}"
        result = merge_namespaced(result, frame, prefix, JUDGMENT_FIELDS, str(path))
    return result


def build(dataset: str, method: str, output_root: Path) -> Path:
    result = add_outputs(dataset, method, base_frame(dataset, method))
    result = result.sort_values("source_row").reset_index(drop=True)
    target = output_root / dataset / f"{method}.parquet"
    atomic_parquet(result, target)
    print(f"{target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}: "
          f"{len(result):,} rows, {len(result.columns):,} columns")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for dataset in args.datasets:
        methods = args.methods or METHODS_BY_DATASET[dataset]
        invalid = set(methods) - set(METHODS_BY_DATASET[dataset])
        if invalid:
            raise ValueError(f"{dataset}: unsupported methods {sorted(invalid)}")
        for method in methods:
            build(dataset, method, args.output_root)


if __name__ == "__main__":
    main()
