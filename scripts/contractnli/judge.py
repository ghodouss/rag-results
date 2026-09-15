#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.argv = [sys.argv[0], "--stage", "judge", *sys.argv[1:]]
runpy.run_path(str(ROOT / "scripts/run_contractnli_full_dual_label_matrix.py"), run_name="__main__")
