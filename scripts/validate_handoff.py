#!/usr/bin/env python3
"""Validate packaged counts, joins, methods, qrels, and cumulative score curves."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, config


def validate(dataset):
    expected = config()[dataset]["expected_questions"]
    directory = ROOT / "data" / dataset
    questions = pd.read_parquet(directory / "questions.parquet")
    if len(questions) != expected or questions.query_id.duplicated().any():
        raise ValueError(f"{dataset}: invalid questions parquet")
    results = pd.read_parquet(directory / "retrieval_results.parquet")
    methods = set(results.retrieval_method.astype(str))
    if methods != {"sturdy", "bm25"}:
        raise ValueError(f"{dataset}: expected sturdy and bm25, found {methods}")
    if len(results) != expected * len(methods):
        raise ValueError(f"{dataset}: expected {expected * len(methods):,} result rows")
    if results.duplicated(["query_id", "retrieval_method"]).any():
        raise ValueError(f"{dataset}: duplicate query/method result rows")
    counts = results.groupby("retrieval_method").query_id.nunique().to_dict()
    if any(count != expected for count in counts.values()):
        raise ValueError(f"{dataset}: incomplete result methods: {counts}")
    expected_ids = set(questions.query_id.astype(str))
    for method, group in results.groupby("retrieval_method"):
        if set(group.query_id.astype(str)) != expected_ids:
            raise ValueError(f"{dataset}/{method}: query IDs do not align")
    required_result = ["question", "golden_answer", "retrieval_error", "retrieved_count"]
    required_result.extend(f"excerpt_{rank}" for rank in range(1, 5))
    required_result.extend(f"excerpt_{rank}_unit_id" for rank in range(1, 5))
    required_result.extend(f"excerpt_{rank}_doc_id" for rank in range(1, 5))
    missing = [column for column in required_result if column not in results]
    if missing:
        raise ValueError(f"{dataset}: result columns missing: {', '.join(missing)}")
    if results[required_result].isna().any().any():
        raise ValueError(f"{dataset}: nulls in required result columns")

    allowed_by_query = {
        str(row.query_id): set(map(str, json.loads(row.allowed_doc_ids_json)))
        for row in questions.itertuples(index=False)
    }
    corpus = pd.read_parquet(directory / "corpus.parquet")
    corpus_units = set(corpus.unit_id.astype(str))
    for row in results.itertuples(index=False):
        excerpt_count = 0
        for rank in range(1, 5):
            unit_id = str(getattr(row, f"excerpt_{rank}_unit_id"))
            doc_id = str(getattr(row, f"excerpt_{rank}_doc_id"))
            text = str(getattr(row, f"excerpt_{rank}"))
            if not unit_id and not doc_id and not text:
                continue
            excerpt_count += 1
            if unit_id not in corpus_units:
                raise ValueError(f"{dataset}: result references unknown unit {unit_id}")
            if doc_id not in allowed_by_query[str(row.query_id)]:
                raise ValueError(
                    f"{dataset}: result doc {doc_id} violates query filter {row.query_id}")
        if excerpt_count != int(row.retrieved_count):
            raise ValueError(f"{dataset}: retrieved_count does not match populated excerpts")
    if dataset != "finqa":
        qrels = pd.read_parquet(directory / "retrieval_qrels.parquet")
        if not set(qrels.query_id.astype(str)) <= expected_ids:
            raise ValueError(f"{dataset}: qrels contain unknown query IDs")
        if not set(qrels.unit_id.astype(str)) <= corpus_units:
            raise ValueError(f"{dataset}: qrels contain unknown retrieval units")
        for row in qrels.itertuples(index=False):
            if str(row.doc_id) not in allowed_by_query[str(row.query_id)]:
                raise ValueError(f"{dataset}: qrel violates supplied-document filter")
        for rank in range(1, 4):
            lower = results[f"golden_answer_exact_in_top_{rank}"]
            upper = results[f"golden_answer_exact_in_top_{rank + 1}"]
            if (lower & ~upper).any():
                raise ValueError(f"{dataset}: answer-presence curve is not cumulative")
    print(f"{dataset}: valid; {expected:,} questions x {len(methods)} methods")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=config())
    args = parser.parse_args()
    for dataset in args.datasets:
        validate(dataset)


if __name__ == "__main__":
    main()
