#!/usr/bin/env python3
"""Run every additional Sonnet 4 scale-judgment cell in parallel."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUDGE_MODEL = "anthropic/claude-sonnet-4"
GENERATOR_MODELS = ("google/gemini-2.5-flash", "gpt-4o-mini")
DATASETS = ("bioasq", "finqa")


def execute(command: list[str]) -> None:
    print("starting:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    args = parser.parse_args()
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")

    common = [
        "--reasoning-effort", "none",
        "--concurrency", str(args.concurrency),
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    jobs = [
        [
            sys.executable, "scripts/run_contractnli_sonnet4_judgments.py",
            "--conditions", condition,
            "--generator-models", generator_model,
            "--concurrency", str(args.concurrency),
            "--checkpoint-every", str(args.checkpoint_every),
        ]
        for condition in (
            "sturdy-a4-r2-ranked-windows", "bm25-top4",
        )
        for generator_model in (
            "google/gemini-2.5-flash", "openai/gpt-4o-mini",
        )
    ]
    jobs.extend(
        [
            sys.executable, f"scripts/{dataset}/judge.py",
            "--generator-model", generator_model,
            "--judge-model", JUDGE_MODEL,
            "--judge-mode", "scale-1-5",
            *common,
        ]
        for dataset in DATASETS
        for generator_model in GENERATOR_MODELS
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(execute, command) for command in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
