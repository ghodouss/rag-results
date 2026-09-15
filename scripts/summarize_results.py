#!/usr/bin/env python3
"""Print the published metrics directly from the ten canonical Parquets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(dataset: str, method: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "results" / dataset / f"{method}.parquet")


def exact(dataset: str) -> pd.DataFrame:
    rows = []
    for method in ("sturdy", "bm25", "e5", "openai"):
        frame = load(dataset, method)
        row = {"method": method}
        possible = int(frame["golden_answer_exact_in_full_doc"].sum())
        if not possible:
            raise ValueError(f"{dataset}/{method}: no full-document exact matches")
        for rank in range(1, 5):
            row[f"top_{rank}"] = float(
                frame[f"golden_answer_exact_in_top_{rank}"].sum() / possible
            )
        rows.append(row)
    return pd.DataFrame(rows)


def e2e(dataset: str) -> pd.DataFrame:
    rows = []
    for method in ("sturdy", "bm25"):
        frame = load(dataset, method)
        if dataset == "contractnli":
            cells = (
                ("Gemini 2.5 Flash", "Claude Haiku 4.5", "correct",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__judge_correct"),
                ("Gemini 2.5 Flash", "Claude Sonnet 4.5", "mean_score",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score"),
                ("GPT-4o mini", "Claude Haiku 4.5", "correct",
                 "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__judge_correct"),
                ("GPT-4o mini", "Claude Sonnet 4.5", "mean_score",
                 "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4_5__score"),
            )
            for generator, judge, metric, column in cells:
                rows.append({"method": method, "generator": generator,
                             "judge": judge, "metric": metric,
                             "value": float(frame[column].mean())})
        else:
            cells = (
                ("Gemini 2.5 Flash", "Claude Haiku 4.5", "correct",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict"),
                ("Gemini 2.5 Flash", "Claude Sonnet 4.5", "mean_score",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score"),
            )
            for generator, judge, metric, column in cells:
                values = frame[column]
                value = (
                    float(values.eq("CORRECT").mean())
                    if metric == "correct" else float(values.mean())
                )
                rows.append({"method": method, "generator": generator,
                             "judge": judge, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def main() -> None:
    for dataset in ("contractnli", "bioasq", "finqa"):
        print(f"\n{dataset}")
        if dataset != "finqa":
            ceiling = float(load(dataset, "sturdy")[
                "golden_answer_exact_in_full_doc"
            ].mean())
            print(f"\ndocuments_containing_golden_answer_exactly {ceiling:.6f}")
            print("\nexact")
            print(exact(dataset).to_string(index=False))
        print("\ne2e")
        print(e2e(dataset).to_string(index=False))


if __name__ == "__main__":
    main()
