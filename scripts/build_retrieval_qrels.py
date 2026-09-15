#!/usr/bin/env python3
"""Build retrieval relevance labels at each dataset's evaluated unit level."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config, load_questions, normalize


DEFAULT_RAW = Path("/Users/kian/ML/clojure/sturdy-search-lab/artifacts/raw")


def contractnli():
    questions = load_questions("contractnli")
    corpus = pd.read_parquet(ROOT / "data/contractnli/corpus.parquet")
    by_doc = {str(doc): group for doc, group in corpus.groupby("doc_id")}
    rows = []
    for question in questions.itertuples(index=False):
        answer = normalize(question.golden_answer)
        for doc_id in json.loads(question.allowed_doc_ids_json):
            for unit in by_doc.get(str(doc_id), pd.DataFrame()).itertuples(index=False):
                text = normalize(unit.text)
                if len(text.split()) >= 4 and (text in answer or answer in text):
                    rows.append({"query_id": str(question.query_id),
                                 "unit_id": str(unit.unit_id), "doc_id": str(doc_id),
                                 "relevance": 1, "evidence": question.golden_answer})
    return pd.DataFrame(rows).drop_duplicates(["query_id", "unit_id"])


def bioasq(raw_root):
    questions = load_questions("bioasq")
    corpus = pd.read_parquet(ROOT / "data/bioasq/corpus.parquet")
    text_by_doc = dict(zip(corpus.doc_id.astype(str), corpus.text.map(normalize)))
    documents = pd.read_parquet(ROOT / "data/bioasq/documents.parquet")
    passage_to_doc = dict(zip(documents.passage_id.astype(str), documents.doc_id.astype(str)))
    primary = []
    with (raw_root / "bioasq-dev.jsonl").open() as source:
        primary.extend(json.loads(line) for line in source if line.strip())
    upstream = list(primary)
    for filename in ("bioasq-eval.jsonl",):
        with (raw_root / filename).open() as source:
            upstream.extend(json.loads(line) for line in source if line.strip())
    by_question = {}
    for row in upstream:
        by_question.setdefault(normalize(row["question"]), []).append(row)
    rows = []
    for question in questions.itertuples(index=False):
        source_row = int(question.source_row)
        source = primary[source_row] if source_row < len(primary) else None
        if source is None or normalize(source["question"]) != normalize(question.question):
            candidates = by_question.get(normalize(question.question), [])
            if len(candidates) != 1:
                raise ValueError(
                    f"BioASQ source row {source_row} does not align and question has "
                    f"{len(candidates)} normalized matches")
            source = candidates[0]
        allowed = set(map(str, json.loads(question.allowed_doc_ids_json)))
        for snippet in source.get("snippets", []):
            passage_id = snippet["document"].rstrip("/").rsplit("/", 1)[-1]
            doc_id = passage_to_doc.get(str(passage_id))
            evidence = normalize(snippet.get("text", ""))
            if doc_id is not None and doc_id not in allowed:
                raise ValueError(
                    f"BioASQ source evidence {passage_id} maps outside the supplied "
                    f"filter for query {question.query_id}")
            if doc_id is not None and evidence and evidence in text_by_doc.get(doc_id, ""):
                rows.append({"query_id": str(question.query_id), "unit_id": doc_id,
                             "doc_id": doc_id, "relevance": 1,
                             "evidence": snippet["text"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (frame.groupby(["query_id", "unit_id", "doc_id", "relevance"], as_index=False)
            .agg(evidence=("evidence", "first"), evidence_count=("evidence", "size")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("contractnli", "bioasq"))
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    frame = contractnli() if args.dataset == "contractnli" else bioasq(args.raw)
    target = ROOT / "data" / args.dataset / "retrieval_qrels.parquet"
    atomic_parquet(frame, target)
    print(f"{args.dataset}: {len(frame):,} qrels, "
          f"{frame.query_id.nunique():,} evaluable questions -> {target}")


if __name__ == "__main__":
    main()
