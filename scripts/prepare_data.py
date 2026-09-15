#!/usr/bin/env python3
"""Build stable question and document parquets from the canonical RAG inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config


DEFAULT_SOURCE = Path(
    "/Users/kian/ML/misc/rag_approaches_by_document_genre/v3/training-output"
)
DEFAULT_EVALUATION = Path(
    "/Users/kian/ML/clojure/sturdy-search-lab/artifacts/evaluation-full-handoff"
)


def questions(dataset, source_root, evaluation_root):
    stem = {"contractnli": "contractnli", "bioasq": "bioasq", "finqa": "finqa"}[dataset]
    qa = pd.read_parquet(source_root / f"{stem}_qa.parquet").reset_index(names="source_row")
    eval_key = "contract" if dataset == "contractnli" else dataset
    query = pd.read_parquet(evaluation_root / eval_key / "queries.parquet")
    filters = pd.read_parquet(evaluation_root / eval_key / "qrels.parquet")
    allowed = (filters.groupby("query_id", sort=False).doc_id
               .apply(lambda values: json.dumps(list(dict.fromkeys(values.astype(str)))))
               .rename("allowed_doc_ids_json").reset_index())
    frame = query.merge(qa, on="source_row", how="left", validate="one_to_one")
    frame = frame.merge(allowed, on="query_id", how="left", validate="one_to_one")
    frame = frame.rename(columns={"query": "evaluation_question", "answer": "golden_answer"})
    if "question" in frame and not frame.question.fillna("").eq(frame.evaluation_question).all():
        raise ValueError(f"{dataset}: source and evaluation questions do not align")
    frame["question"] = frame.pop("evaluation_question")
    first = ["query_id", "split", "source_row", "question", "golden_answer",
             "allowed_doc_ids_json"]
    return frame[first + [column for column in frame if column not in first and column != "index"]]


def documents(dataset, source_root):
    if dataset == "contractnli":
        frame = pd.read_parquet(source_root / "contractnli_docs.parquet").copy()
        frame["unit_id"] = frame.doc_id.astype(str)
        frame = frame.rename(columns={"context": "text"})
    elif dataset == "bioasq":
        frame = pd.read_parquet(source_root / "bioasq_docs_parsed.parquet").copy()
        frame["unit_id"] = frame.doc_id.astype(str)
        frame = frame.rename(columns={"doc": "text"})
    else:
        frame = pd.read_parquet(source_root / "finqa_docs.parquet").copy()
        frame["doc_id"] = frame.doc_index.astype(str)
        frame["unit_id"] = frame.doc_id
        frame = frame.rename(columns={"context": "text"})
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=config(), default=list(config()))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    for dataset in args.datasets:
        q = questions(dataset, args.source, args.evaluation)
        expected = config()[dataset]["expected_questions"]
        if len(q) != expected:
            raise ValueError(f"{dataset}: expected {expected:,} questions, found {len(q):,}")
        atomic_parquet(q, ROOT / "data" / dataset / "questions.parquet")
        docs = documents(dataset, args.source)
        atomic_parquet(docs, ROOT / "data" / dataset / "documents.parquet")
        if dataset == "bioasq":
            corpus = docs[["unit_id", "doc_id", "text"]].copy()
            corpus.insert(0, "retrieval_level", "doc")
            atomic_parquet(corpus, ROOT / "data" / dataset / "corpus.parquet")
        print(f"{dataset}: {len(q):,} questions", flush=True)


if __name__ == "__main__":
    main()
