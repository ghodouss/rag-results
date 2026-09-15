#!/usr/bin/env python3
"""Regenerate a complete filtered Sturdy top-four result parquet.

Index IDs are fixed in config/datasets.json. Retrieval is serial by default and
checkpointed after every 25 questions. It requires sturdy-stats SDK credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config, load_questions, validate_result


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def load_index_class():
    sdk = os.environ.get("STURDY_STATS_SDK_PATH")
    if sdk:
        sys.path.insert(0, sdk)
    try:
        from sturdystats import Index
    except ImportError as error:
        raise SystemExit("install sturdystats or set STURDY_STATS_SDK_PATH") from error
    return Index


def retrieve(index, row, cfg, retries):
    allowed = json.loads(row["allowed_doc_ids_json"])
    values = ",".join(sql_string(value) for value in allowed)
    filter_expression = f"CAST(doc_id AS VARCHAR) IN ({values})"
    for attempt in range(retries):
        try:
            result = index.search(
                level=cfg["retrieval_level"], search_query=str(row["question"]),
                filter=filter_expression, semantic_search_weight=cfg["semantic_search_weight"],
                semantic_search_cutoff=cfg["semantic_search_cutoff"], limit=cfg["top_k"])
            excerpts = []
            if result is not None:
                for rank, item in enumerate(result.head(cfg["top_k"]).itertuples(), 1):
                    scores = item.scores if isinstance(getattr(item, "scores", None), dict) else {}
                    excerpt = {"rank": rank, "doc_id": str(item.doc_id),
                               "text": str(item.text),
                               "score": scores.get("search_score")}
                    indices = []
                    for column in ("paragraph_idx", "subparagraph_idx"):
                        if hasattr(item, column) and pd.notna(getattr(item, column)):
                            excerpt[column] = int(getattr(item, column))
                            indices.append(str(int(getattr(item, column))))
                    excerpt["unit_id"] = "|".join([str(item.doc_id), *indices])
                    excerpts.append(excerpt)
            output = dict(row)
            output.update(retrieval_method="sturdy", index_id=cfg["index_id"],
                          index_name=cfg["index_name"],
                          retrieval_level=cfg["retrieval_level"], top_k=cfg["top_k"],
                          semantic_search_weight=cfg["semantic_search_weight"],
                          semantic_search_cutoff=cfg["semantic_search_cutoff"],
                          retrieved_count=len(excerpts),
                          retrieved_context="\n\n".join(x["text"] for x in excerpts),
                          retrieved_excerpts_json=json.dumps(excerpts, ensure_ascii=False),
                          retrieval_error="")
            for rank in range(1, cfg["top_k"] + 1):
                excerpt = excerpts[rank - 1] if len(excerpts) >= rank else {}
                output[f"excerpt_{rank}"] = excerpt.get("text", "")
                output[f"excerpt_{rank}_unit_id"] = excerpt.get("unit_id", "")
                output[f"excerpt_{rank}_doc_id"] = excerpt.get("doc_id", "")
                output[f"excerpt_{rank}_score"] = excerpt.get("score")
            return output
        except Exception as error:
            if attempt + 1 == retries:
                return dict(row, retrieval_method="sturdy", retrieval_error=type(error).__name__,
                            retrieved_count=0, retrieved_context="",
                            retrieved_excerpts_json="[]",
                            **{f"excerpt_{rank}": "" for rank in range(1, 5)})
            time.sleep(min(30, 2 ** attempt + random.random()))


def run(dataset, retries=6, checkpoint_every=25):
    cfg = config()[dataset]
    questions = load_questions(dataset)
    target = ROOT / "artifacts" / dataset / "sturdy_top4.parquet"
    checkpoint = target.with_name(".sturdy_top4.checkpoint.parquet")
    records = pd.read_parquet(checkpoint).to_dict("records") if checkpoint.exists() else []
    records = [row for row in records if not row.get("retrieval_error")]
    done = {str(row["query_id"]) for row in records}
    Index = load_index_class()
    index = Index(id=cfg["index_id"], org_id=os.environ["STURDY_ORG_ID"],
                  api_key=os.environ["STURDY_API_KEY"])
    pending = [row._asdict() for row in questions.itertuples(index=False)
               if str(row.query_id) not in done]
    print(f"{dataset}: {len(records):,} done, {len(pending):,} pending", flush=True)
    for number, row in enumerate(pending, 1):
        records.append(retrieve(index, row, cfg, retries))
        if number % checkpoint_every == 0 or number == len(pending):
            atomic_parquet(pd.DataFrame(records), checkpoint)
            print(f"  {len(records):,}/{len(questions):,}", flush=True)
    output = pd.DataFrame(records).sort_values("source_row").reset_index(drop=True)
    failures = output.retrieval_error.fillna("").ne("")
    if failures.any():
        raise SystemExit(f"{int(failures.sum())} failures remain; rerun to retry")
    validate_result(output, dataset)
    atomic_parquet(output, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=config())
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
