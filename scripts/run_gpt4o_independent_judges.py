#!/usr/bin/env python3
"""Judge the completed GPT-4o-mini BioASQ and FinQA generations."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_MODEL = "gpt-4o-mini"
DATASETS = ("bioasq", "finqa")
JUDGES = {
    "anthropic/claude-haiku-4.5": "binary",
    "anthropic/claude-sonnet-4.5": "scale-1-5",
}


def execute(command: list[str]) -> None:
    print("starting:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=2048)
    args = parser.parse_args()
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")

    jobs = [
        [
            sys.executable, f"scripts/{dataset}/judge.py",
            "--generator-model", GENERATOR_MODEL,
            "--judge-model", judge_model,
            "--judge-mode", judge_mode,
            "--reasoning-effort", "none",
            "--concurrency", str(args.concurrency),
            "--checkpoint-every", str(args.checkpoint_every),
        ]
        for dataset in DATASETS
        for judge_model, judge_mode in JUDGES.items()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(execute, command) for command in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
