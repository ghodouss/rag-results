#!/usr/bin/env python3
"""Run the full ContractNLI dual-label Sturdy-versus-BM25 matrix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_contractnli_full_prompt_matrix as matrix
import run_contractnli_prompt_pilot as pilot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_eval.common import ROOT


RESULTS_ROOT = ROOT / "artifacts" / "e2e" / "contractnli"
GENERATOR_PROMPT_VERSION = "supplied-generic-dual-label-v1"
GENERATOR_SUFFIX = (
    "Answer True/Entailment or False/Contradiction, followed by a brief explanation."
)
GENERATOR_SYSTEM_PROMPT = pilot.BASELINE_SYSTEM_PROMPT + "\n\n" + GENERATOR_SUFFIX
MATRIX_VERSION = "contractnli-full-dual-label-sturdy-bm25-6173-v1"


def write_generation_summary(frame: pd.DataFrame, target: Path) -> None:
    """Write public generation metadata without deterministic label scoring."""
    summary = {
        "generator_model": str(frame.generator_model.iloc[0]),
        "generator_prompt_version": str(frame.generator_prompt_version.iloc[0]),
        "questions": int(len(frame)),
        "generation_errors": int(frame.generation_error.fillna("").ne("").sum()),
    }
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(target.with_suffix(".summary.csv"), index=False)


def write_judgment_summary(frame: pd.DataFrame, target: Path) -> dict:
    """Write the supplied binary-judge metric without deterministic label scoring."""
    summary = {
        "generator_model": str(frame.generator_model.iloc[0]),
        "generator_prompt_version": str(frame.generator_prompt_version.iloc[0]),
        "judge_model": str(frame.judge_model.iloc[0]),
        "judge_prompt_version": str(frame.judge_prompt_version.iloc[0]),
        "questions": int(len(frame)),
        "judge_errors": int(frame.judge_error.fillna("").ne("").sum()),
        "correct": int(frame.judge_correct.fillna(False).sum()),
        "accuracy": float(frame.judge_correct.fillna(False).mean()),
    }
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(target.with_suffix(".summary.csv"), index=False)
    return summary


def configure() -> None:
    matrix.RESULTS_ROOT = RESULTS_ROOT
    matrix.GENERATOR_PROMPT_VERSION = GENERATOR_PROMPT_VERSION
    matrix.GENERATOR_SYSTEM_PROMPT = GENERATOR_SYSTEM_PROMPT
    matrix.MATRIX_VERSION = MATRIX_VERSION
    pilot.write_generation_summary = write_generation_summary
    pilot.write_judgment_summary = write_judgment_summary


def main() -> None:
    configure()
    matrix.main()


if __name__ == "__main__":
    main()
