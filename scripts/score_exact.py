#!/usr/bin/env python3
"""Recompute normalized exact-answer containment from packaged retrievals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag_eval.common import normalize  # noqa: E402


def neural(dataset: str, name: str, questions: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data" / dataset / "retrieval" / f"{name}.parquet")
    frame["source_row"] = (
        frame.query_id.astype(str).str.rsplit("-", n=1).str[-1].astype("int64")
    )
    aligned = questions[["source_row", "question", "golden_answer"]].merge(
        frame,
        on="source_row",
        how="left",
        suffixes=("_expected", ""),
        validate="one_to_one",
    )
    if len(aligned) != len(questions) or aligned.query_id.isna().any():
        raise ValueError(f"{dataset}/{name}: expected {len(questions)} aligned rows")
    if not aligned.question.eq(aligned.question_expected).all():
        raise ValueError(f"{dataset}/{name}: question identity mismatch")
    if not aligned.golden_answer.eq(aligned.golden_answer_expected).all():
        raise ValueError(f"{dataset}/{name}: golden-answer identity mismatch")
    return aligned.drop(columns=["question_expected", "golden_answer_expected"])


def frames(dataset: str, questions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if dataset == "contractnli":
        retrieval = ROOT / "data/contractnli/retrieval"
        return {
            "sturdy": pd.concat([
                pd.read_parquet(path) for path in sorted(retrieval.glob("sturdy-a4-r2.part-*.parquet"))
            ], ignore_index=True),
            "bm25": pd.read_parquet(retrieval / "bm25.parquet"),
            "intfloat/e5-small-v2": neural(dataset, "e5-small-v2", questions),
            "openai/embedding-model-unspecified": neural(
                dataset, "openai-embedding-model-unspecified", questions
            ),
        }
    classic = pd.read_parquet(ROOT / "data/bioasq/retrieval_results.parquet")
    return {
        "sturdy": classic[classic.retrieval_method.astype(str).eq("sturdy")],
        "bm25": classic[classic.retrieval_method.astype(str).eq("bm25")],
        "intfloat/e5-small-v2": neural(dataset, "e5-small-v2", questions),
        "openai/embedding-model-unspecified": neural(
            dataset, "openai-embedding-model-unspecified", questions
        ),
    }


def full_doc_hits(dataset: str, questions: pd.DataFrame) -> int:
    documents = pd.read_parquet(ROOT / "data" / dataset / "documents.parquet")
    text_by_doc = documents.groupby(documents.doc_id.astype(str)).text.apply(" ".join).to_dict()
    hits = 0
    for row in questions.itertuples(index=False):
        context = " ".join(text_by_doc.get(str(doc_id), "") for doc_id in json.loads(row.allowed_doc_ids_json))
        gold = normalize(row.golden_answer)
        hits += bool(gold and gold in normalize(context))
    return hits


def score(dataset: str) -> pd.DataFrame:
    questions = pd.read_parquet(ROOT / "data" / dataset / "questions.parquet")
    ceiling = full_doc_hits(dataset, questions)
    rows = []
    for method, frame in frames(dataset, questions).items():
        if len(frame) != len(questions):
            raise ValueError(f"{dataset}/{method}: expected {len(questions)} rows")
        counts = []
        for rank in range(1, 5):
            if method == "sturdy" and dataset == "contractnli":
                contexts = frame[f"containment_context_{rank}"].astype(str)
            else:
                contexts = frame.apply(
                    lambda row: " ".join(str(row[f"excerpt_{i}"]) for i in range(1, rank + 1)), axis=1
                )
            counts.append(sum(
                bool(normalize(gold) and normalize(gold) in normalize(context))
                for gold, context in zip(frame.golden_answer, contexts)
            ))
        rows.append({
            "dataset": dataset, "method": method, "rows": len(frame),
            **{f"top_{i}_count": counts[i - 1] for i in range(1, 5)},
            "full_doc_count": ceiling,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("contractnli", "bioasq"))
    args = parser.parse_args()
    print(score(args.dataset).to_string(index=False))


if __name__ == "__main__":
    main()
