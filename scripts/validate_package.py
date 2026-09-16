#!/usr/bin/env python3
"""Validate the twelve canonical result Parquets and the package manifest."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 49 * 1024 * 1024
ROWS = {"contractnli": 6173, "bioasq": 4387, "finqa": 6251}
METHODS = {
    "contractnli": ("sturdy", "bm25", "e5", "openai"),
    "bioasq": ("sturdy", "bm25", "e5", "openai"),
    "finqa": ("sturdy", "bm25", "e5", "openai"),
}
RETRIEVAL_ERRORS = {("bioasq", "sturdy"): 23}

EXACT_COUNTS = {
    ("contractnli", "sturdy"): (2938, 3322, 3489, 3577, 3852),
    ("contractnli", "bm25"): (1994, 2635, 2922, 3094, 3852),
    ("contractnli", "e5"): (1582, 2202, 2582, 2828, 3852),
    ("contractnli", "openai"): (1954, 2466, 2690, 2806, 3852),
    ("bioasq", "sturdy"): (238, 319, 394, 435, 570),
    ("bioasq", "bm25"): (256, 349, 404, 443, 570),
    ("bioasq", "e5"): (230, 334, 399, 435, 570),
    ("bioasq", "openai"): (232, 340, 392, 423, 570),
}

JUDGMENT_COUNTS = {
    ("contractnli", "e5"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__judge_correct": 4699,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 25364,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__judge_correct": 5021,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 26414,
    },
    ("contractnli", "openai"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__judge_correct": 4693,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 25236,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__judge_correct": 5032,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 26487,
    },
    ("contractnli", "sturdy"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__judge_correct": 5080,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score": 27039,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 26867,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__judge_correct": 5221,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4_5__score": 27323,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 27296,
    },
    ("contractnli", "bm25"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__judge_correct": 4812,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score": 25979,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 25820,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__judge_correct": 5021,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4_5__score": 26554,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 26455,
    },
    ("bioasq", "sturdy"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 3137,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score": 15590,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 15412,
        "judgment_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 3454,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4_5__score": 17432,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4__score": 17373,
        "judgment_gpt_4o_mini__gpt_4o_mini__verdict": 3255,
        "judgment_gpt_5_6_luna__gpt_5_6_luna__verdict": 3717,
    },
    ("bioasq", "bm25"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 3153,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score": 15634,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 15484,
        "judgment_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 3462,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4_5__score": 17474,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4__score": 17422,
        "judgment_gpt_4o_mini__gpt_4o_mini__verdict": 3266,
        "judgment_gpt_5_6_luna__gpt_5_6_luna__verdict": 3695,
    },
    ("bioasq", "e5"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 3076,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 15167,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 3396,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 17193,
    },
    ("bioasq", "openai"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 3069,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 15215,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 3395,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 17232,
    },
    ("finqa", "sturdy"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 2796,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score": 19761,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 19493,
        "judgment_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 2946,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4_5__score": 21372,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4__score": 21163,
        "judgment_gpt_4o_mini__gpt_4o_mini__verdict": 2561,
        "judgment_gpt_5_6_luna__gpt_5_6_luna__verdict": 4727,
    },
    ("finqa", "bm25"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 2745,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score": 19525,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 19224,
        "judgment_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 2953,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4_5__score": 21436,
        "judgment_gpt_4o_mini__anthropic_claude_sonnet_4__score": 21137,
        "judgment_gpt_4o_mini__gpt_4o_mini__verdict": 2544,
        "judgment_gpt_5_6_luna__gpt_5_6_luna__verdict": 4748,
    },
    ("finqa", "e5"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 2712,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 19063,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 2896,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 20700,
    },
    ("finqa", "openai"): {
        "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict": 2724,
        "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score": 19188,
        "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__verdict": 2974,
        "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score": 20984,
    },
}


def load_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "write_manifest", ROOT / "scripts/write_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_results() -> None:
    expected_paths = {
        ROOT / "results" / dataset / f"{method}.parquet"
        for dataset in ROWS for method in METHODS[dataset]
    }
    actual_paths = set((ROOT / "results").rglob("*.parquet"))
    if actual_paths != expected_paths:
        raise ValueError("results/ must contain exactly the twelve canonical Parquets")

    for dataset, rows in ROWS.items():
        frames = {}
        for method in METHODS[dataset]:
            path = ROOT / "results" / dataset / f"{method}.parquet"
            frame = pd.read_parquet(path)
            frames[method] = frame
            if len(frame) != rows or frame.query_id.astype(str).nunique() != rows:
                raise ValueError(f"{dataset}/{method}: incomplete or duplicate questions")
            if set(frame.retrieval_method.astype(str)) != {method}:
                raise ValueError(f"{dataset}/{method}: retrieval method mismatch")
            if frame.question.fillna("").astype(str).str.strip().eq("").any():
                raise ValueError(f"{dataset}/{method}: empty question")
            if "retrieved_items" not in frame or "retrieved_context" not in frame:
                raise ValueError(f"{dataset}/{method}: retrieval data missing")
            if any(
                str(item.get("text", "")).strip().lower() == "nan"
                for items in frame.retrieved_items for item in items
            ):
                raise ValueError(f"{dataset}/{method}: NaN retrieval sentinel exposed")
            errors = int(frame.retrieval_error.fillna("").astype(str).str.strip().ne("").sum())
            if errors != RETRIEVAL_ERRORS.get((dataset, method), 0):
                raise ValueError(f"{dataset}/{method}: unexpected retrieval error count")
            forbidden = {"parsed_label", "label_correct", "label_parse_error"}
            if any(any(name in column for name in forbidden) for column in frame):
                raise ValueError(f"{dataset}/{method}: deterministic label metric exposed")
            for column in frame:
                if column.endswith(("__generation_error", "__judge_error")):
                    if frame[column].fillna("").astype(str).str.strip().ne("").any():
                        raise ValueError(f"{dataset}/{method}: nonempty {column}")
                elif column.endswith("__generated_answer"):
                    if frame[column].fillna("").astype(str).str.strip().eq("").any():
                        raise ValueError(f"{dataset}/{method}: empty {column}")
                elif column.endswith("__judge_correct"):
                    if frame[column].isna().any():
                        raise ValueError(f"{dataset}/{method}: missing {column}")
                elif column.endswith("__score"):
                    prefix = column.removesuffix("__score")
                    mode_column = prefix + "__judge_mode"
                    if (
                        mode_column in frame
                        and set(frame[mode_column].dropna().astype(str)) == {"binary"}
                    ):
                        continue
                    fallback_column = prefix + "__judge_fallback"
                    missing = frame[column].isna()
                    allowed_missing = (
                        frame[fallback_column].fillna("").astype(str).ne("")
                        if fallback_column in frame else False
                    )
                    if (missing & ~allowed_missing).any() or not frame.loc[~missing, column].between(1, 5).all():
                        raise ValueError(f"{dataset}/{method}: invalid {column}")
                elif column.endswith("__verdict"):
                    mode_column = column.removesuffix("__verdict") + "__judge_mode"
                    if (
                        mode_column in frame
                        and set(frame[mode_column].dropna().astype(str)) == {"scale-1-5"}
                    ):
                        continue
                    if not set(frame[column].astype(str)) <= {"CORRECT", "INCORRECT"}:
                        raise ValueError(f"{dataset}/{method}: invalid {column}")

            if dataset != "finqa":
                columns = [f"golden_answer_exact_in_top_{rank}" for rank in range(1, 5)]
                columns.append("golden_answer_exact_in_full_doc")
                counts = tuple(int(frame[column].sum()) for column in columns)
                if counts != EXACT_COUNTS[(dataset, method)]:
                    raise ValueError(f"{dataset}/{method}: exact counts changed: {counts}")

            for column, expected in JUDGMENT_COUNTS.get((dataset, method), {}).items():
                if column.endswith("__verdict"):
                    actual = int(frame[column].eq("CORRECT").sum())
                else:
                    actual = int(frame[column].sum())
                if actual != expected:
                    raise ValueError(
                        f"{dataset}/{method}: {column} expected {expected}, found {actual}"
                    )

        expected_ids = set(frames["sturdy"].query_id.astype(str))
        for method, frame in frames.items():
            if set(frame.query_id.astype(str)) != expected_ids:
                raise ValueError(f"{dataset}: {method} question IDs differ")


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
    validate_results()
    validate_manifest_and_sizes()
    print("valid package: twelve canonical result Parquets, metrics, manifest, and sizes")


if __name__ == "__main__":
    validate()
