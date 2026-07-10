#!/usr/bin/env python3
"""Write a quick quality report for an SBML GEM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gemiz.quality import summarize_model_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GEM model quality.")
    parser.add_argument("model", type=Path, help="SBML model path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <model>.quality.json.",
    )
    parser.add_argument(
        "--blocked",
        action="store_true",
        help="Also run blocked-reaction analysis. This can be slow.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")

    report = summarize_model_file(args.model, include_blocked=args.blocked)
    output = args.output or args.model.with_suffix(args.model.suffix + ".quality.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    growth = report["growth"]
    counts = report["counts"]
    print(f"Model: {args.model}")
    print(f"  reactions:   {counts['reactions']}")
    print(f"  metabolites: {counts['metabolites']}")
    print(f"  genes:       {counts['genes']}")
    print(f"  growth:      {growth['growth_rate']} ({growth['status']})")
    print(f"  report:      {output}")


if __name__ == "__main__":
    main()
