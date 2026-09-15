#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.argv = [sys.argv[0], "bioasq", *sys.argv[1:]]
runpy.run_path(str(ROOT / "scripts/score_exact.py"), run_name="__main__")
