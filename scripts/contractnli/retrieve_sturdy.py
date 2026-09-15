#!/usr/bin/env python3
"""Retrieve four ranked Sturdy sentence anchors with two neighbors per side."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rag_eval.common import atomic_parquet, config, load_questions  # noqa: E402


def sql_string(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def unit_id(unit: dict) -> str:
    locations = [unit.get(name) for name in (
        "paragraph_idx", "subparagraph_idx", "sentence_idx"
    )]
    return "|".join([str(unit.get("doc_id", "")), *(
        "" if value is None else str(int(value)) for value in locations
    )])


def ranked_windows(result: pd.DataFrame) -> list[dict]:
    """Preserve anchor rank and remove overlap from later expanded windows."""
    seen: set[str] = set()
    windows = []
    for rank, anchor in enumerate(result.head(4).to_dict("records"), 1):
        units = anchor.get("context")
        if not isinstance(units, list) or not units:
            units = [anchor]
        fresh = []
        for unit in units:
            key = unit_id(unit)
            if key in seen:
                continue
            seen.add(key)
            fresh.append(unit)
        scores = anchor.get("scores") if isinstance(anchor.get("scores"), dict) else {}
        windows.append({
            "rank": rank,
            "doc_id": str(anchor.get("doc_id", "")),
            "unit_id": unit_id(anchor),
            "score": scores.get("search_score"),
            "text": "".join(str(unit.get("text", "")) for unit in fresh).strip(),
            "units": fresh,
        })
    return windows


def retrieve(index, row: dict, retries: int) -> dict:
    allowed = json.loads(row["allowed_doc_ids_json"])
    values = ",".join(sql_string(value) for value in allowed)
    filter_expression = f"CAST(doc_id AS VARCHAR) IN ({values})"
    for attempt in range(retries):
        try:
            result = index.search(
                level="sentence", search_query=str(row["question"]),
                filter=filter_expression, semantic_search_weight=0.3,
                semantic_search_cutoff=0.1, limit=4, context=2,
            )
            windows = ranked_windows(result if result is not None else pd.DataFrame())
            output = dict(row)
            output.update(
                retrieval_method="sturdy", retrieval_level="sentence",
                context_radius=2, top_k=4, retrieved_count=len(windows),
                retrieved_context="\n\n".join(w["text"] for w in windows if w["text"]),
                retrieved_excerpts_json=json.dumps(windows, ensure_ascii=False),
                retrieval_error="",
            )
            for rank in range(1, 5):
                window = windows[rank - 1] if len(windows) >= rank else {}
                for suffix, key in (("", "text"), ("_unit_id", "unit_id"),
                                    ("_doc_id", "doc_id"), ("_score", "score")):
                    output[f"excerpt_{rank}{suffix}"] = window.get(key, "")
            return output
        except Exception as error:
            if attempt + 1 == retries:
                return dict(row, retrieval_method="sturdy", retrieval_level="sentence",
                            context_radius=2, top_k=4, retrieved_count=0,
                            retrieved_context="", retrieved_excerpts_json="[]",
                            retrieval_error=type(error).__name__)
            time.sleep(min(30, 2 ** attempt + random.random()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/contractnli/sturdy.parquet")
    args = parser.parse_args()
    sdk_path = os.environ.get("STURDY_STATS_SDK_PATH")
    if sdk_path:
        sys.path.insert(0, sdk_path)
    from sturdystats import Index
    cfg = config()["contractnli"]
    questions = load_questions("contractnli")
    checkpoint = args.output.with_name(".sturdy.checkpoint.parquet")
    records = pd.read_parquet(checkpoint).to_dict("records") if checkpoint.exists() else []
    records = [row for row in records if not row.get("retrieval_error")]
    done = {str(row["query_id"]) for row in records}
    index = Index(id=cfg["index_id"], org_id=os.environ["STURDY_ORG_ID"],
                  api_key=os.environ["STURDY_API_KEY"])
    pending = [row._asdict() for row in questions.itertuples(index=False)
               if str(row.query_id) not in done]
    for number, row in enumerate(pending, 1):
        records.append(retrieve(index, row, args.retries))
        if number % args.checkpoint_every == 0 or number == len(pending):
            atomic_parquet(pd.DataFrame(records), checkpoint)
    result = pd.DataFrame(records).sort_values("source_row").reset_index(drop=True)
    errors = result.retrieval_error.fillna("").ne("")
    if len(result) != len(questions) or errors.any():
        raise SystemExit(f"incomplete: {len(result)}/{len(questions)}, {int(errors.sum())} errors")
    atomic_parquet(result, args.output)


if __name__ == "__main__":
    main()
