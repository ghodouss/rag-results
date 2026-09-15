#!/usr/bin/env python3
"""Export the indexed retrieval units needed by the offline BM25 baseline."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT, atomic_parquet, config


SELECT = {
    "contractnli": ("subparagraph",
                    "CAST(doc_id AS VARCHAR) doc_id, paragraph_idx, "
                    "subparagraph_idx, text FROM subparagraph"),
    "bioasq": ("doc", "CAST(doc_id AS VARCHAR) doc_id, text FROM doc"),
    "finqa": ("paragraph",
              "CAST(doc_id AS VARCHAR) doc_id, paragraph_idx, text FROM paragraph"),
}


def load_index_class():
    sdk = os.environ.get("STURDY_STATS_SDK_PATH")
    if sdk:
        sys.path.insert(0, sdk)
    try:
        from sturdystats import Index
    except ImportError as error:
        raise SystemExit("install sturdystats or set STURDY_STATS_SDK_PATH") from error
    return Index


def run(dataset):
    cfg = config()[dataset]
    Index = load_index_class()
    index = Index(id=cfg["index_id"],
                  org_id=os.environ["STURDY_ORG_ID"],
                  api_key=os.environ["STURDY_API_KEY"])
    level, select = SELECT[dataset]
    frame = index.sql("SELECT " + select)
    if dataset == "contractnli":
        frame["unit_id"] = (frame.doc_id.astype(str) + "|" +
                            frame.paragraph_idx.astype(str) + "|" +
                            frame.subparagraph_idx.astype(str))
    elif dataset == "finqa":
        frame["unit_id"] = frame.doc_id.astype(str) + "|" + frame.paragraph_idx.astype(str)
    else:
        frame["unit_id"] = frame.doc_id.astype(str)
    frame.insert(0, "retrieval_level", level)
    target = ROOT / "data" / dataset / "corpus.parquet"
    atomic_parquet(frame, target)
    print(f"{dataset}: {len(frame):,} indexed {level} units -> {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=config())
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
