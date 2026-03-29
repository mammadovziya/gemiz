#!/usr/bin/env python3
"""Compare gemiz output models against gold-standard reference models.

Usage
-----
    python scripts/benchmark.py \\
        --organisms ecoli bsubtilis scerevisiae mtuberculosis styphimurium \\
        --output data/benchmark_results.json

Paths
-----
  gemiz model     : data/test_outputs/{org}_model.xml
  gold standard   : data/organisms/{org}/gold_standard.xml
  ecoli exception : data/universal/iML1515.xml  (gold standard)

Metrics
-------
  TP        reactions in both gemiz and gold standard
  FP        reactions in gemiz only
  FN        reactions in gold standard only
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  F1        = 2 * precision * recall / (precision + recall)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: dict[str, str] = {
    "ecoli":         "E. coli K-12",
    "bsubtilis":     "B. subtilis 168",
    "scerevisiae":   "S. cerevisiae",
    "mtuberculosis": "M. tuberculosis",
    "styphimurium":  "S. typhimurium",
    "paeruginosa":   "P. aeruginosa",
    "pputida":       "P. putida",
}

# Override gold-standard path for specific organisms
_GOLD_OVERRIDES: dict[str, str] = {
    "ecoli": "data/universal/iML1515.xml",
}

LOW_PRECISION_WARN = 0.3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class OrgResult(NamedTuple):
    organism: str
    display_name: str
    gemiz_reactions: int
    gold_reactions: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    growth_rate: float
    warning: str | None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _model_paths(org: str) -> tuple[Path, Path]:
    gemiz_path = Path(f"data/test_outputs/{org}_model.xml")
    gold_path = Path(
        _GOLD_OVERRIDES[org] if org in _GOLD_OVERRIDES
        else f"data/organisms/{org}/gold_standard.xml"
    )
    return gemiz_path, gold_path


def _compute_metrics(
    gemiz_ids: set[str],
    gold_ids: set[str],
) -> tuple[int, int, int, float, float, float]:
    tp = len(gemiz_ids & gold_ids)
    fp = len(gemiz_ids - gold_ids)
    fn = len(gold_ids - gemiz_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return tp, fp, fn, precision, recall, f1


def run_organism(org: str) -> OrgResult:
    """Load models and compute all metrics for one organism."""
    import cobra

    display = _DISPLAY_NAMES.get(org, org)
    gemiz_path, gold_path = _model_paths(org)

    if not gemiz_path.exists():
        raise FileNotFoundError(f"gemiz model not found: {gemiz_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"gold standard not found: {gold_path}")

    print(f"  Loading gemiz model  : {gemiz_path}")
    gemiz = cobra.io.read_sbml_model(str(gemiz_path))

    print(f"  Loading gold standard: {gold_path}")
    gold  = cobra.io.read_sbml_model(str(gold_path))

    # Growth rate via FBA
    sol = gemiz.optimize()
    growth_rate = sol.objective_value if sol.status == "optimal" else 0.0

    # Reaction-level metrics (exact ID match)
    gemiz_ids = {r.id for r in gemiz.reactions}
    gold_ids  = {r.id for r in gold.reactions}
    tp, fp, fn, precision, recall, f1 = _compute_metrics(gemiz_ids, gold_ids)

    warning = None
    if precision < LOW_PRECISION_WARN:
        warning = (
            f"WARNING: Low precision for {org} - possible reaction ID "
            f"namespace mismatch between gemiz and gold standard"
        )

    return OrgResult(
        organism=org,
        display_name=display,
        gemiz_reactions=len(gemiz.reactions),
        gold_reactions=len(gold.reactions),
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        growth_rate=growth_rate,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_FMT = " {:<20} {:>9}  {:>7}  {:>6}  {:>7}  {:>9}"
_HEADER = _FMT.format("Organism", "Precision", "Recall", "F1", "Growth", "Reactions")
_SEP = "═" * len(_HEADER)


def print_table(results: list[OrgResult]) -> None:
    print(f"\n{_SEP}")
    print(_HEADER)
    print(_SEP)

    for r in results:
        print(_FMT.format(
            r.display_name,
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.f1:.3f}",
            f"{r.growth_rate:.3f}",
            str(r.gemiz_reactions),
        ))

    print(_SEP)

    if results:
        n = len(results)
        mean_p = sum(r.precision for r in results) / n
        mean_r = sum(r.recall   for r in results) / n
        mean_f = sum(r.f1       for r in results) / n
        print(_FMT.format(
            "Mean",
            f"{mean_p:.3f}",
            f"{mean_r:.3f}",
            f"{mean_f:.3f}",
            "", "",
        ))

    print(f"{_SEP}\n")


def save_json(
    results: list[OrgResult],
    errors: list[tuple[str, str]],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    data = {
        "organisms": [
            {
                "organism":        r.organism,
                "display_name":    r.display_name,
                "gemiz_reactions": r.gemiz_reactions,
                "gold_reactions":  r.gold_reactions,
                "tp":              r.tp,
                "fp":              r.fp,
                "fn":              r.fn,
                "precision":       round(r.precision,    4),
                "recall":          round(r.recall,       4),
                "f1":              round(r.f1,           4),
                "growth_rate":     round(r.growth_rate,  4),
            }
            for r in results
        ],
        "errors": [
            {"organism": org, "error": err}
            for org, err in errors
        ],
        "summary": {
            "n_organisms":    n,
            "mean_precision": round(sum(r.precision for r in results) / n, 4) if n else 0,
            "mean_recall":    round(sum(r.recall    for r in results) / n, 4) if n else 0,
            "mean_f1":        round(sum(r.f1        for r in results) / n, 4) if n else 0,
        },
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark gemiz models against gold-standard references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--organisms",
        nargs="+",
        default=["ecoli"],
        metavar="ORG",
        help="Organism names to benchmark (space-separated).",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark_results.json",
        help="Path to write JSON results (default: data/benchmark_results.json).",
    )
    args = parser.parse_args()

    results: list[OrgResult] = []
    errors:  list[tuple[str, str]] = []

    for org in args.organisms:
        print(f"\n[{org}]")
        try:
            result = run_organism(org)
            results.append(result)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            errors.append((org, str(exc)))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append((org, str(exc)))

    if not results:
        print("\nNo results — all organisms were skipped or errored.")
        sys.exit(1)

    # Print warnings before the table so they're visible
    for r in results:
        if r.warning:
            print(f"\n{r.warning}")

    print_table(results)
    save_json(results, errors, Path(args.output))


if __name__ == "__main__":
    main()
