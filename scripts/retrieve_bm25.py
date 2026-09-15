#!/usr/bin/env python3
"""Produce a filtered top-four BM25 handoff parquet from packaged data."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.bm25 import rank_bm25
from rag_eval.common import ROOT, atomic_parquet, config, load_questions, validate_result


def run(dataset: str) -> Path:
    questions = load_questions(dataset)
    corpus_path = ROOT / "data" / dataset / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"missing {corpus_path}; the BM25 corpus must preserve indexed unit boundaries")
    output = rank_bm25(questions, pd.read_parquet(corpus_path), top_k=4)
    validate_result(output, dataset)
    target = ROOT / "artifacts" / dataset / "bm25_top4.parquet"
    atomic_parquet(output, target)
    print(f"{dataset}: {len(output):,} BM25 results -> {target}")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=config())
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
