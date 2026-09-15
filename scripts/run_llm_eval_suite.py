#!/usr/bin/env python3
"""Run the requested same-model generation and judging matrix."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("gpt-4o-mini", "gpt-5.6-luna")
DEFAULT_DATASETS = ("contractnli", "bioasq", "finqa")


def execute(command: list[str]) -> None:
    print("starting:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def stage(script: str, jobs: list[list[str]], parallel_jobs: int) -> None:
    print(f"\n{script}: {len(jobs)} jobs", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        futures = [pool.submit(execute, command) for command in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=("contractnli", "bioasq", "finqa"))
    parser.add_argument("--concurrency-per-job", type=int, default=8)
    parser.add_argument("--parallel-jobs", type=int, default=4)
    args = parser.parse_args()

    generation_jobs = [
        [sys.executable, "scripts/generate_answers.py", dataset,
         "--model", model, "--reasoning-effort", "low",
         "--concurrency", str(args.concurrency_per_job)]
        for model in args.models for dataset in args.datasets
    ]
    stage("generation", generation_jobs, args.parallel_jobs)

    judging_jobs = [
        [sys.executable, "scripts/judge_answers.py", dataset,
         "--generator-model", model, "--judge-model", model,
         "--reasoning-effort", "low",
         "--concurrency", str(args.concurrency_per_job)]
        for model in args.models for dataset in args.datasets
    ]
    stage("judging", judging_jobs, args.parallel_jobs)


if __name__ == "__main__":
    main()
