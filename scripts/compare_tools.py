#!/usr/bin/env python3
"""Compare F1 scores across GEM reconstruction tools.

Reads benchmark result JSONs produced by benchmark.py and prints
a cross-tool comparison table.

Usage
-----
    python scripts/compare_tools.py \\
        data/comparison/gemiz_results.json \\
        data/comparison/carveme_results.json

    # Or with a glob:
    python scripts/compare_tools.py data/comparison/*.json

Output
------
    Tool       E.coli  B.sub  S.cer  M.tub  S.typ  Mean
    ─────────────────────────────────────────────────────
    gemiz      0.951   0.909  0.941  0.951  0.941  0.939
    carveme    0.921   0.874  0.903  0.912  0.899  0.902
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Short column labels used in the table header
_SHORT_LABELS: dict[str, str] = {
    "ecoli":         "E.coli",
    "bsubtilis":     "B.sub",
    "scerevisiae":   "S.cer",
    "mtuberculosis": "M.tub",
    "styphimurium":  "S.typ",
    "paeruginosa":   "P.aer",
    "pputida":       "P.put",
}

_MISSING = "  —  "   # shown when a tool has no result for an organism


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_result(path: Path) -> dict:
    """Load and validate a benchmark JSON file."""
    with open(path) as f:
        data = json.load(f)
    if "tool" not in data or "organisms" not in data:
        raise ValueError(
            f"{path}: missing 'tool' or 'organisms' key. "
            "Was this produced by benchmark.py?"
        )
    return data


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def _f1_map(data: dict) -> dict[str, float]:
    """Return {organism: f1} from one benchmark result dict."""
    return {row["organism"]: row["f1"] for row in data["organisms"]}


def build_table(
    datasets: list[dict],
    metric: str = "f1",
) -> tuple[list[str], list[str], list[list[str | None]]]:
    """Return (tools, organisms, rows) for the comparison table.

    *rows* is a list-of-lists; None means no data for that cell.
    """
    tools = [d["tool"] for d in datasets]

    # Organism order: union of all organisms, preserving first-seen order
    seen: dict[str, None] = {}
    for d in datasets:
        for row in d["organisms"]:
            seen[row["organism"]] = None
    organisms = list(seen)

    rows: list[list[str | None]] = []
    for d in datasets:
        scores = {row["organism"]: row[metric] for row in d["organisms"]}
        rows.append([scores.get(org) for org in organisms])

    return tools, organisms, rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_comparison(
    datasets: list[dict],
    metric: str = "f1",
    metric_label: str = "F1",
) -> None:
    tools, organisms, rows = build_table(datasets, metric)

    col_w = 7   # width for each organism column
    tool_w = max(10, max(len(t) for t in tools) + 1)

    # Header
    labels = [_SHORT_LABELS.get(org, org[:6]) for org in organisms]
    header = f"{'Tool':<{tool_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels) + f"{'Mean':>{col_w}}"
    sep = "\u2500" * len(header)

    print(f"\n{metric_label} comparison")
    print(header)
    print(sep)

    for tool, score_row in zip(tools, rows):
        valid = [s for s in score_row if s is not None]
        mean_str = f"{sum(valid)/len(valid):.3f}" if valid else _MISSING.strip()

        cells = "".join(
            f"{s:.3f}".rjust(col_w) if s is not None else "—".rjust(col_w)
            for s in score_row
        )
        print(f"{tool:<{tool_w}}{cells}{mean_str:>{col_w}}")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-tool F1 comparison table from benchmark.py JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        metavar="RESULT_JSON",
        help="One or more benchmark result JSON files.",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric to display (default: f1).",
    )
    args = parser.parse_args()

    datasets: list[dict] = []
    for path in args.results:
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        try:
            datasets.append(load_result(path))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"SKIP: {path}: {exc}", file=sys.stderr)

    if not datasets:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    metric_labels = {"f1": "F1", "precision": "Precision", "recall": "Recall"}
    print_comparison(datasets, metric=args.metric,
                     metric_label=metric_labels[args.metric])


if __name__ == "__main__":
    main()
