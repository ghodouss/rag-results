from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def config() -> dict:
    return json.loads((ROOT / "config/datasets.json").read_text())


def normalize(value) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    value = unicodedata.normalize("NFKD", str(value)).lower()
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", value))


def tokenize(value) -> list[str]:
    return normalize(value).split()


def relaxed_bigram_match(golden, evidence, max_missing: int = 1) -> bool:
    """Return whether evidence contains all but ``max_missing`` gold bigrams."""
    gold = tokenize(golden)
    observed = tokenize(evidence)
    if len(gold) < 2:
        return bool(gold and gold[0] in observed)
    required = Counter(zip(gold, gold[1:]))
    available = Counter(zip(observed, observed[1:]))
    missing = sum(max(0, count - available[pair])
                  for pair, count in required.items())
    return missing <= max_missing


def atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)


def load_questions(dataset: str) -> pd.DataFrame:
    path = ROOT / "data" / dataset / "questions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/prepare_data.py")
    return pd.read_parquet(path)


def validate_result(frame: pd.DataFrame, dataset: str, allow_partial: bool = False) -> None:
    required = {"query_id", "question", "golden_answer", "retrieval_method",
                "retrieved_context", "retrieved_excerpts_json"}
    required.update(f"excerpt_{rank}" for rank in range(1, 5))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"result missing columns: {', '.join(missing)}")
    expected = int(config()[dataset]["expected_questions"])
    if allow_partial and not 0 < len(frame) <= expected:
        raise ValueError(
            f"{dataset}: partial result must contain 1--{expected:,} rows, "
            f"found {len(frame):,}")
    if not allow_partial and len(frame) != expected:
        raise ValueError(f"{dataset}: expected {expected:,} rows, found {len(frame):,}")
    if frame.query_id.astype(str).duplicated().any():
        raise ValueError(f"{dataset}: duplicate query_id values")
    if frame[list(required)].isna().any().any():
        columns = frame[list(required)].columns[frame[list(required)].isna().any()].tolist()
        raise ValueError(f"{dataset}: null values in required columns: {', '.join(columns)}")
    for row_number, value in enumerate(frame.retrieved_excerpts_json):
        try:
            excerpts = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{dataset}: invalid retrieved_excerpts_json at row {row_number}") from error
        if not isinstance(excerpts, list) or len(excerpts) > 4:
            raise ValueError(
                f"{dataset}: expected at most four structured excerpts at row {row_number}")
