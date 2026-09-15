#!/usr/bin/env python3
"""Write deterministic hashes, sizes, and row counts for handoff artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXCLUDED_PARTS = {
    ".git", ".venv", ".local-archive", "__pycache__", ".checkpoints",
    ".pytest_cache",
}
EXCLUDED_NAMES = {"MANIFEST.json", ".DS_Store", ".Rhistory"}


def included_files() -> list[Path]:
    return [
        path for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the existing manifest without rewriting it")
    args = parser.parse_args()
    entries = []
    for path in included_files():
        entry = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix == ".parquet":
            metadata = pq.read_metadata(path)
            entry["rows"] = metadata.num_rows
            entry["columns"] = metadata.num_columns
        entries.append(entry)
    payload = {"files": entries}
    target = ROOT / "MANIFEST.json"
    if args.check:
        if not target.exists() or json.loads(target.read_text()) != payload:
            raise SystemExit("manifest does not match the distributed artifacts")
        print(f"verified {len(entries)} entries in {target}")
        return
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} entries -> {target}")


if __name__ == "__main__":
    sys.exit(main())
