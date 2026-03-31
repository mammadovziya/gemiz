#!/usr/bin/env python3
"""Compare tool output models against gold-standard reference models.

Usage
-----
    python scripts/benchmark.py \\
        --tool gemiz \\
        --model-dir data/comparison/gemiz \\
        --organisms ecoli bsubtilis scerevisiae mtuberculosis styphimurium \\
        --output data/comparison/gemiz_results.json

    python scripts/benchmark.py \\
        --tool carveme \\
        --model-dir data/comparison/carveme \\
        --organisms ecoli bsubtilis scerevisiae mtuberculosis styphimurium \\
        --output data/comparison/carveme_results.json

Model filename pattern : {model_dir}/{org}_model.xml
Gold standard          : data/organisms/{org}/gold_standard.xml
E. coli exception      : data/universal/iML1515.xml

Metrics
-------
  TP        reactions in both tool model and gold standard
  FP        reactions in tool model only
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

_GOLD_OVERRIDES: dict[str, str] = {
    "ecoli": "data/universal/iML1515.xml",
}

LOW_PRECISION_WARN = 0.3
DEFAULT_MODEL_DIR  = "data/test_outputs"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class OrgResult(NamedTuple):
    organism: str
    display_name: str
    model_reactions: int
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

def _model_paths(org: str, model_dir: str) -> tuple[Path, Path]:
    model_path = Path(model_dir) / f"{org}_model.xml"
    gold_path = Path(
        _GOLD_OVERRIDES[org] if org in _GOLD_OVERRIDES
        else f"data/organisms/{org}/gold_standard.xml"
    )
    return model_path, gold_path


def _compute_metrics(
    model_ids: set[str],
    gold_ids: set[str],
) -> tuple[int, int, int, float, float, float]:
    tp = len(model_ids & gold_ids)
    fp = len(model_ids - gold_ids)
    fn = len(gold_ids - model_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return tp, fp, fn, precision, recall, f1


def run_organism(org: str, tool: str, model_dir: str) -> OrgResult:
    """Load models and compute all metrics for one organism."""
    import cobra

    display = _DISPLAY_NAMES.get(org, org)
    model_path, gold_path = _model_paths(org, model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"{tool} model not found: {model_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"gold standard not found: {gold_path}")

    print(f"  Loading {tool} model  : {model_path}")
    model = cobra.io.read_sbml_model(str(model_path))

    print(f"  Loading gold standard : {gold_path}")
    gold  = cobra.io.read_sbml_model(str(gold_path))

    sol = model.optimize()
    growth_rate = sol.objective_value if sol.status == "optimal" else 0.0

    model_ids = {r.id for r in model.reactions}
    gold_ids  = {r.id for r in gold.reactions}
    tp, fp, fn, precision, recall, f1 = _compute_metrics(model_ids, gold_ids)

    warning = None
    if precision < LOW_PRECISION_WARN:
        warning = (
            f"WARNING: Low precision for {org} - possible reaction ID "
            f"namespace mismatch between {tool} and gold standard"
        )

    return OrgResult(
        organism=org,
        display_name=display,
        model_reactions=len(model.reactions),
        gold_reactions=len(gold.reactions),
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        growth_rate=growth_rate,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_FMT    = " {:<20} {:>9}  {:>7}  {:>6}  {:>7}  {:>9}"
_HEADER = _FMT.format("Organism", "Precision", "Recall", "F1", "Growth", "Reactions")
_SEP    = "\u2550" * len(_HEADER)


def print_table(results: list[OrgResult], tool: str) -> None:
    print(f"\n{_SEP}")
    print(f" Tool: {tool}")
    print(_HEADER)
    print(_SEP)

    for r in results:
        print(_FMT.format(
            r.display_name,
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.f1:.3f}",
            f"{r.growth_rate:.3f}",
            str(r.model_reactions),
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
    tool: str,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    data = {
        "tool": tool,
        "organisms": [
            {
                "organism":        r.organism,
                "display_name":    r.display_name,
                "model_reactions": r.model_reactions,
                "gold_reactions":  r.gold_reactions,
                "tp":              r.tp,
                "fp":              r.fp,
                "fn":              r.fn,
                "precision":       round(r.precision,   4),
                "recall":          round(r.recall,      4),
                "f1":              round(r.f1,          4),
                "growth_rate":     round(r.growth_rate, 4),
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
        description="Benchmark tool models against gold-standard references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tool",
        default="gemiz",
        help="Tool name label stored in the output JSON (default: gemiz).",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing {{org}}_model.xml files "
             f"(default: {DEFAULT_MODEL_DIR}).",
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
            result = run_organism(org, tool=args.tool, model_dir=args.model_dir)
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

    for r in results:
        if r.warning:
            print(f"\n{r.warning}")

    print_table(results, tool=args.tool)
    save_json(results, errors, tool=args.tool, output=Path(args.output))


if __name__ == "__main__":
    main()
