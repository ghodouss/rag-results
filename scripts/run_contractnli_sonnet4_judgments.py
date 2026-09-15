#!/usr/bin/env python3
"""Add Sonnet 4 scale judgments to the completed ContractNLI generations."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import run_contractnli_full_prompt_matrix as matrix


JUDGE_MODEL = "anthropic/claude-sonnet-4"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "artifacts/e2e/contractnli"


async def main_async(
    conditions: tuple[str, ...], generator_models: tuple[str, ...],
    concurrency: int, checkpoint_every: int,
) -> None:
    # The reusable scale runner derives paths, provenance fields, and summaries
    # from this module-level model identifier.
    matrix.SONNET_MODEL = JUDGE_MODEL
    matrix.GENERATOR_PROMPT_VERSION = "supplied-generic-dual-label-v1"
    jobs = []
    for condition in conditions:
        for generator_model in generator_models:
            generation = matrix.generation_path(
                OUTPUT_ROOT, condition, generator_model
            )
            target = matrix.sonnet_path(OUTPUT_ROOT, condition, generator_model)
            jobs.append(matrix.run_scale_judgment(
                generation, target, concurrency, checkpoint_every
            ))
    await asyncio.gather(*jobs)
    if set(conditions) == set(matrix.CONDITIONS) and set(generator_models) == set(matrix.GENERATOR_MODELS):
        paired_json = OUTPUT_ROOT / "paired/summary.json"
        existing = json.loads(paired_json.read_text()) if paired_json.exists() else []
        additional = matrix.write_paired_summaries(OUTPUT_ROOT, generator_models)
        combined = {
            (row["generator_model"], row["judge_model"]): row
            for row in [*existing, *additional]
        }
        judge_order = {
            matrix.HAIKU_MODEL: 0,
            "anthropic/claude-sonnet-4.5": 1,
            JUDGE_MODEL: 2,
        }
        rows = sorted(
            combined.values(),
            key=lambda row: (
                generator_models.index(row["generator_model"]),
                judge_order.get(row["judge_model"], 99),
            ),
        )
        matrix.pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "paired/summary.csv", index=False)
        paired_json.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions", nargs="+", choices=tuple(matrix.CONDITIONS),
        default=tuple(matrix.CONDITIONS),
    )
    parser.add_argument(
        "--generator-models", nargs="+", choices=matrix.GENERATOR_MODELS,
        default=matrix.GENERATOR_MODELS,
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    args = parser.parse_args()
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")
    asyncio.run(main_async(
        tuple(args.conditions), tuple(args.generator_models),
        args.concurrency, args.checkpoint_every,
    ))


if __name__ == "__main__":
    main()
