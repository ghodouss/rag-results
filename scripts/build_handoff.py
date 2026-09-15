#!/usr/bin/env python3
"""Combine retrieval methods, score them, and write one result parquet per dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import (
    ROOT, atomic_parquet, config, load_questions, normalize,
    relaxed_bigram_match, validate_result,
)


def excerpt_unit_id(excerpt):
    if excerpt.get("unit_id"):
        return str(excerpt["unit_id"])
    values = [str(excerpt["doc_id"])]
    for key in ("paragraph_idx", "subparagraph_idx"):
        if excerpt.get(key) is not None:
            values.append(str(excerpt[key]))
    return "|".join(values)


def retrieval_metrics(frame, qrels, cutoffs=(1, 2, 3, 4)):
    relevant = {str(q): set(g.unit_id.astype(str))
                for q, g in qrels.groupby("query_id", sort=False)}
    totals = {f"recall@{k}": 0.0 for k in cutoffs}
    totals.update({f"hit@{k}": 0.0 for k in cutoffs})
    used = 0
    for row in frame.itertuples(index=False):
        gold = relevant.get(str(row.query_id), set())
        if not gold:
            continue
        ranked = [excerpt_unit_id(value)
                  for value in json.loads(row.retrieved_excerpts_json)]
        for cutoff in cutoffs:
            found = len(set(ranked[:cutoff]) & gold)
            totals[f"recall@{cutoff}"] += found / len(gold)
            totals[f"hit@{cutoff}"] += float(found > 0)
        used += 1
    return {key: value / used for key, value in totals.items()} | {"evaluable_questions": used}


def full_document_text(questions, documents, dataset):
    if dataset == "contractnli":
        lookup = dict(zip(documents.doc_id.astype(str), documents.text.astype(str)))
        return pd.Series([
            "\n\n".join(lookup.get(doc, "") for doc in json.loads(ids))
            for ids in questions.allowed_doc_ids_json
        ], index=questions.query_id.astype(str))
    if dataset == "bioasq":
        lookup = dict(zip(documents.doc_id.astype(str), documents.text.astype(str)))
        return pd.Series([
            "\n\n".join(lookup.get(doc, "") for doc in json.loads(ids))
            for ids in questions.allowed_doc_ids_json
        ], index=questions.query_id.astype(str))
    return pd.Series(dtype=str)


def answer_presence(frame, questions, documents, dataset):
    if dataset == "finqa":
        return frame
    answer = dict(zip(questions.query_id.astype(str), questions.golden_answer.map(normalize)))
    full = full_document_text(questions, documents, dataset).map(normalize).to_dict()
    top_hits = {rank: [] for rank in range(1, 5)}
    relaxed_hits = {rank: [] for rank in range(1, 5)}
    full_hits = []
    for row in frame.itertuples(index=False):
        qid = str(row.query_id)
        gold = answer.get(qid, "")
        excerpts = json.loads(row.retrieved_excerpts_json)
        texts = [normalize(item.get("text", "")) for item in excerpts]
        for rank in range(1, 5):
            canonical_column = f"containment_context_{rank}"
            context = (
                normalize(getattr(row, canonical_column))
                if hasattr(row, canonical_column)
                else " ".join(texts[:rank])
            )
            top_hits[rank].append(bool(gold and gold in context))
            relaxed_hits[rank].append(relaxed_bigram_match(gold, context))
        full_hits.append(bool(gold and gold in full.get(qid, "")))
    for rank, values in top_hits.items():
        frame[f"golden_answer_exact_in_top_{rank}"] = values
        frame[f"golden_answer_relaxed_bigram_in_top_{rank}"] = relaxed_hits[rank]
    frame["golden_answer_exact_in_full_doc"] = full_hits
    return frame


def build(dataset, methods, allow_partial=False):
    questions = load_questions(dataset)
    frames = []
    for method in methods:
        path = ROOT / "artifacts" / dataset / f"{method}_top4.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        validate_result(frame, dataset, allow_partial=allow_partial)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    result = answer_presence(result, questions,
                             pd.read_parquet(ROOT / "data" / dataset / "documents.parquet"),
                             dataset)

    summary = []
    qrels_path = ROOT / "data" / dataset / "retrieval_qrels.parquet"
    qrels = pd.read_parquet(qrels_path) if qrels_path.exists() else None
    for method, group in result.groupby("retrieval_method", sort=True):
        row = {"dataset": dataset, "retrieval_method": method,
               "questions": int(group.query_id.nunique()),
               "retrieval_errors": int(group.retrieval_error.fillna("").ne("").sum())}
        if qrels is not None:
            row.update(retrieval_metrics(group, qrels))
        if dataset != "finqa":
            row.update({f"golden_answer_exact_in_top_{rank}": float(
                group[f"golden_answer_exact_in_top_{rank}"].mean()) for rank in range(1, 5)})
            row.update({f"golden_answer_relaxed_bigram_in_top_{rank}": float(
                group[f"golden_answer_relaxed_bigram_in_top_{rank}"].mean())
                for rank in range(1, 5)})
            row["golden_answer_exact_in_full_doc"] = float(
                group.golden_answer_exact_in_full_doc.mean())
        summary.append(row)

    # The intermediate method files retain combined context and structured JSON.
    # Store excerpt text only once in the distributable parquet to keep the full
    # datasets comfortably below common repository file-size limits.
    transient_context = [
        column for column in result
        if column in {"retrieved_context", "retrieved_excerpts_json"}
        or column.startswith("containment_context_")
    ]
    distributable = result.drop(columns=transient_context)
    target = ROOT / "data" / dataset / "retrieval_results.parquet"
    atomic_parquet(distributable, target)
    result_dir = ROOT / "results" / dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(summary).to_csv(result_dir / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=config())
    parser.add_argument("--methods", nargs="+", choices=("sturdy", "bm25"),
                        default=("sturdy", "bm25"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    build(args.dataset, args.methods, args.allow_partial)


if __name__ == "__main__":
    main()
