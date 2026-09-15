#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.argv = [sys.argv[0], "finqa", *sys.argv[1:]]
runpy.run_path(str(ROOT / "scripts/judge_answers.py"), run_name="__main__")
