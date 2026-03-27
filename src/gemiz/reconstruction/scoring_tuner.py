"""Approach B: Tune scoring thresholds on E. coli validation set.

Finds the optimal HIGH_CONF and LOW_CONF thresholds by grid-searching
combinations and evaluating against the known iML1515 model as ground truth.

Usage
-----
    pytest -m slow tests/test_scoring.py::test_scoring_tuner -v -s

Takes ~20 minutes (tests 25 threshold combinations).
Run once before writing the paper.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import cobra

from gemiz.reconstruction.scoring import compute_reaction_scores


def tune_thresholds(
    mmseqs_hits: dict[str, list[dict]],
    esmc_hits: dict[str, list[dict]],
    universal_model: cobra.Model,
    reference_model: cobra.Model,
    feature_table_path: str | Path | None = None,
    reference_faa_path: str | Path | None = None,
) -> dict:
    """Grid search over (high_conf, low_conf) threshold pairs.

    Parameters
    ----------
    mmseqs_hits, esmc_hits:
        Alignment results from Steps 2–3.
    universal_model:
        The model whose reactions will be scored.
    reference_model:
        Gold-standard curated model (e.g. iML1515). Reactions present
        in this model are the positive set.

    Returns
    -------
    dict
        ``{"best": {"high_conf": ..., "low_conf": ..., "f1": ...},
           "all_results": [...]}``
    """
    high_conf_values = [40, 45, 50, 55, 60]
    low_conf_values  = [20, 25, 30, 35, 40]

    reference_rxns = {r.id for r in reference_model.reactions}

    results = []
    best: dict = {"f1": 0, "high_conf": 50, "low_conf": 30}

    n_combos = sum(
        1 for h, l in itertools.product(high_conf_values, low_conf_values) if l < h
    )
    print(f"Tuning thresholds on E. coli validation set...")
    print(f"Testing {n_combos} combinations\n")

    for high_conf, low_conf in itertools.product(high_conf_values, low_conf_values):
        if low_conf >= high_conf:
            continue

        scores = compute_reaction_scores(
            universal_model, mmseqs_hits, esmc_hits,
            high_conf=high_conf, low_conf=low_conf,
            feature_table_path=feature_table_path,
            reference_faa_path=reference_faa_path,
        )

        predicted = {r for r, s in scores.items() if s > 0}

        tp = len(predicted & reference_rxns)
        fp = len(predicted - reference_rxns)
        fn = len(reference_rxns - predicted)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        result = {
            "high_conf": high_conf,
            "low_conf":  low_conf,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
        }
        results.append(result)

        if f1 > best["f1"]:
            best = {"f1": round(f1, 3), "high_conf": high_conf, "low_conf": low_conf}

        print(
            f"  high={high_conf} low={low_conf}: "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

    # summary table
    print("\n── Results Table ──────────────────────────────")
    print(f"{'high_conf':>10} {'low_conf':>9} {'precision':>10} {'recall':>7} {'F1':>6}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True)[:10]:
        marker = " <- best" if (
            r["high_conf"] == best["high_conf"] and r["low_conf"] == best["low_conf"]
        ) else ""
        print(
            f"{r['high_conf']:>10} {r['low_conf']:>9} "
            f"{r['precision']:>10.3f} {r['recall']:>7.3f} "
            f"{r['f1']:>6.3f}{marker}"
        )

    print(
        f"\nBest: high_conf={best['high_conf']}, "
        f"low_conf={best['low_conf']}, F1={best['f1']:.3f}"
    )
    return {"best": best, "all_results": results}
