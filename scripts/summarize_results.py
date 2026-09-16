#!/usr/bin/env python3
"""Print the published metrics directly from the twelve canonical Parquets."""
from __future__ import annotations

import json
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
    for method in ("sturdy", "bm25", "e5", "openai"):
        frame = load(dataset, method)
        if dataset == "contractnli":
            cells = (
                ("Gemini 2.5 Flash", "Claude Haiku 4.5", "correct",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__judge_correct"),
                ("Gemini 2.5 Flash", "Claude Sonnet 4.5", "mean_score",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score"),
                ("Gemini 2.5 Flash", "Claude Sonnet 4", "mean_score",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score"),
                ("GPT-4o mini", "Claude Haiku 4.5", "correct",
                 "judgment_openai_gpt_4o_mini__anthropic_claude_haiku_4_5__judge_correct"),
                ("GPT-4o mini", "Claude Sonnet 4.5", "mean_score",
                 "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4_5__score"),
                ("GPT-4o mini", "Claude Sonnet 4", "mean_score",
                 "judgment_openai_gpt_4o_mini__anthropic_claude_sonnet_4__score"),
            )
            for generator, judge, metric, column in cells:
                if column not in frame:
                    continue
                rows.append({"method": method, "generator": generator,
                             "judge": judge, "metric": metric,
                             "value": float(frame[column].mean())})
        else:
            gpt_slug = (
                "openai_gpt_4o_mini" if method in {"e5", "openai"}
                else "gpt_4o_mini"
            )
            cells = (
                ("Gemini 2.5 Flash", "Claude Haiku 4.5", "correct",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_haiku_4_5__verdict"),
                ("Gemini 2.5 Flash", "Claude Sonnet 4.5", "mean_score",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4_5__score"),
                ("Gemini 2.5 Flash", "Claude Sonnet 4", "mean_score",
                 "judgment_google_gemini_2_5_flash__anthropic_claude_sonnet_4__score"),
                ("GPT-4o mini", "Claude Haiku 4.5", "correct",
                 f"judgment_{gpt_slug}__anthropic_claude_haiku_4_5__verdict"),
                ("GPT-4o mini", "Claude Sonnet 4.5", "mean_score",
                 f"judgment_{gpt_slug}__anthropic_claude_sonnet_4_5__score"),
                ("GPT-4o mini", "Claude Sonnet 4", "mean_score",
                 f"judgment_{gpt_slug}__anthropic_claude_sonnet_4__score"),
            )
            for generator, judge, metric, column in cells:
                if column not in frame:
                    continue
                values = frame[column]
                value = (
                    float(values.eq("CORRECT").mean())
                    if metric == "correct" else float(values.mean())
                )
                rows.append({"method": method, "generator": generator,
                             "judge": judge, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def combined_e2e() -> pd.DataFrame:
    rows = []
    judge_slugs = {
        "Claude Haiku 4.5": "anthropic_claude_haiku_4_5",
        "Claude Sonnet 4.5": "anthropic_claude_sonnet_4_5",
        "Claude Sonnet 4": "anthropic_claude_sonnet_4",
    }
    generator_prefixes = {
        "Gemini 2.5 Flash": ("judgment_google_gemini_2_5_flash__",),
        "GPT-4o mini": (
            "judgment_gpt_4o_mini__", "judgment_openai_gpt_4o_mini__",
        ),
    }
    for dataset, questions in (("contractnli", 6173), ("bioasq", 4387), ("finqa", 6251)):
        metrics = e2e(dataset)
        for (generator, judge, metric), group in metrics.groupby(
            ["generator", "judge", "metric"], sort=False
        ):
            row = {
                "dataset": dataset, "generator": generator, "judge": judge,
                "metric": "binary_accuracy" if metric == "correct" else "mean_score_1_to_5",
                **{method: None for method in ("sturdy", "bm25", "e5", "openai")},
                "questions": questions, "errors": 0, "ignored_scores": 0,
            }
            for item in group.itertuples(index=False):
                row[str(item.method)] = float(item.value)
            judge_slug = judge_slugs[judge]
            for method in ("sturdy", "bm25", "e5", "openai"):
                if row[method] is None:
                    continue
                frame = load(dataset, method)
                matching = [
                    column for column in frame
                    if column.startswith(generator_prefixes[generator])
                    and f"__{judge_slug}__" in column
                ]
                for column in matching:
                    if column.endswith("__judge_error"):
                        row["errors"] += int(
                            frame[column].fillna("").astype(str).str.strip().ne("").sum()
                        )
                    elif column.endswith("__judge_fallback"):
                        row["ignored_scores"] += int(
                            frame[column].fillna("").astype(str)
                            .eq("content_filter_score_excluded").sum()
                        )
            rows.append(row)
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
    combined = combined_e2e()
    target = ROOT / "artifacts/e2e/summary.csv"
    combined.to_csv(target, index=False, na_rep="")
    records = combined.astype(object).where(pd.notna(combined), None).to_dict("records")
    target.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nwrote combined E2E summary -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
