#!/usr/bin/env python3
"""Validate packaged identities, API outputs, summaries, sizes, and manifest."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 49 * 1024 * 1024


def load_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "write_manifest", ROOT / "scripts/write_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def require_frame(path: Path, rows: int, error_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"missing required artifact: {path.relative_to(ROOT)}")
    frame = pd.read_parquet(path)
    if len(frame) != rows:
        raise ValueError(f"{path.relative_to(ROOT)}: expected {rows} rows, found {len(frame)}")
    for column in error_columns:
        if column not in frame or frame[column].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError(f"{path.relative_to(ROOT)}: nonempty or missing {column}")
    return frame


def validate_contract_e2e() -> None:
    root = ROOT / "results/e2e/contractnli"
    inputs = {
        condition: require_frame(root / "input" / f"{condition}.parquet", 6173)
        for condition in ("sturdy-a4-r2-ranked-windows", "bm25-top4")
    }
    ids = set(inputs["sturdy-a4-r2-ranked-windows"].query_id.astype(str))
    if ids != set(inputs["bm25-top4"].query_id.astype(str)) or len(ids) != 6173:
        raise ValueError("ContractNLI inputs are not uniquely paired")
    for condition in inputs:
        generations = sorted((root / condition / "generations").glob("*.parquet"))
        judgments = sorted((root / condition / "judgments").glob("*.parquet"))
        if len(generations) != 2 or len(judgments) != 4:
            raise ValueError(f"{condition}: expected 2 generations and 4 judgments")
        for path in generations:
            frame = require_frame(path, 6173, ("generation_error",))
            if set(frame.query_id.astype(str)) != ids:
                raise ValueError(f"{path.relative_to(ROOT)}: query IDs differ")
        for path in judgments:
            frame = require_frame(path, 6173, ("generation_error", "judge_error"))
            if set(frame.query_id.astype(str)) != ids:
                raise ValueError(f"{path.relative_to(ROOT)}: query IDs differ")
    public_text = "\n".join(
        path.read_text() for path in root.rglob("*.summary.json")
    ) + (root / "paired/summary.json").read_text()
    if "deterministic" in public_text or "label_accuracy" in public_text:
        raise ValueError("ContractNLI public summaries expose deterministic label metrics")


def validate_bioasq_finqa_e2e() -> None:
    for dataset, questions in (("bioasq", 4387), ("finqa", 6251)):
        root = ROOT / "results/e2e" / dataset
        require_frame(ROOT / "data" / dataset / "questions.parquet", questions)
        for path in (root / "generations").glob("*.parquet"):
            require_frame(path, questions * 2, ("generation_error",))
        for path in (root / "judgments").glob("*.parquet"):
            require_frame(path, questions * 2, ("generation_error", "judge_error"))


def validate_containment() -> None:
    contract = ROOT / "data/contractnli/retrieval"
    r2_parts = sorted(contract.glob("sturdy-a4-r2.part-*.parquet"))
    if len(r2_parts) != 2 or sum(pq.read_metadata(path).num_rows for path in r2_parts) != 6173:
        raise ValueError("ContractNLI Sturdy +/-2 partitions are incomplete")
    require_frame(contract / "bm25.parquet", 6173)
    for dataset, rows in (("contractnli", 6173), ("bioasq", 4387)):
        questions = pd.read_parquet(
            ROOT / "data" / dataset / "questions.parquet",
            columns=["query_id", "question", "golden_answer"],
        )
        question_ids = set(questions.query_id.astype(str))
        identity = set(zip(
            questions.question.astype(str), questions.golden_answer.astype(str)
        ))
        for name in ("e5-small-v2.parquet", "openai-embedding-model-unspecified.parquet"):
            frame = pd.read_parquet(ROOT / "data" / dataset / "retrieval" / name)
            neural_identity = set(zip(
                frame.question.astype(str), frame.golden_answer.astype(str)
            ))
            if not identity.issubset(neural_identity):
                raise ValueError(f"{dataset}/{name}: missing packaged question identities")
        if len(question_ids) != rows:
            raise ValueError(f"{dataset}: question count mismatch")
    summary = pd.read_csv(ROOT / "results/containment/summary.csv")
    if len(summary) != 8 or set(summary.dataset) != {"contractnli", "bioasq"}:
        raise ValueError("containment summary has unexpected coverage")


def validate_manifest_and_sizes() -> None:
    module = load_manifest_module()
    files = module.included_files()
    oversized = [path.relative_to(ROOT) for path in files if path.stat().st_size > MAX_BYTES]
    if oversized:
        raise ValueError(f"files exceed 49 MiB: {oversized}")
    entries = []
    for path in files:
        entry = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": module.sha256(path),
        }
        if path.suffix == ".parquet":
            metadata = pq.read_metadata(path)
            entry.update(rows=metadata.num_rows, columns=metadata.num_columns)
        entries.append(entry)
    if json.loads((ROOT / "MANIFEST.json").read_text()) != {"files": entries}:
        raise ValueError("MANIFEST.json does not match packaged files")


def validate() -> None:
    validate_contract_e2e()
    validate_bioasq_finqa_e2e()
    validate_containment()
    validate_manifest_and_sizes()
    print("valid package: E2E, containment, manifest, and 49 MiB limit")


if __name__ == "__main__":
    validate()
