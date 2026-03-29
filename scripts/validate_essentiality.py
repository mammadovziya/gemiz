#!/usr/bin/env python3
"""Gene essentiality validation for gemiz models.

Compares single-gene deletion (SGD) predictions against the Keio
collection (Baba et al. 2006, Nature Methods) for E. coli K-12.

Usage
-----
    python scripts/validate_essentiality.py \\
        --model data/test_outputs/ecoli_model.xml \\
        --organism ecoli \\
        --output data/validation/ecoli_essentiality.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Reference data: Keio collection essential genes (Baba et al. 2006)
# ---------------------------------------------------------------------------

_KEIO_ESSENTIAL: frozenset[str] = frozenset({
    "acpP", "acpS", "adk",
    "atpA", "atpB", "atpC", "atpD", "atpE", "atpF", "atpG", "atpH",
    "coaA", "coaD", "coaE",
    "dnaA", "dnaB", "dnaC", "dnaE", "dnaG", "dnaI", "dnaJ", "dnaK", "dnaX",
    "era",
    "fabA", "fabB", "fabD", "fabG", "fabH", "fabI", "fbaA",
    "folA", "folC", "folD", "folE", "folK", "folP",
    "frr",
    "gapA",
    "glmM", "glmS", "glmU",
    "glnS", "gltX",
    "grpE",
    "gyrA", "gyrB",
    "hisS", "holB", "holC", "holD", "holE", "htpG",
    "infA", "infB", "infC",
    "ileS",
    "leuS", "ligA",
    "lpxA", "lpxB", "lpxC", "lpxD", "lpxH", "lpxK", "lspA",
    "map", "metK", "miaA", "minD", "minE", "msbA",
    "murA", "murB", "murC", "murD", "murE", "murF", "murG", "murI",
    "nadD", "nadE",
    "nusA", "nusB", "nusG",
    "pgsA", "plsB", "plsC",
    "prfB", "prsA",
    "pyrG", "pyrH",
    "rho",
    "rplA", "rplB", "rplC", "rplD", "rplE", "rplF", "rplI", "rplJ",
    "rplK", "rplL", "rplM", "rplN", "rplO", "rplP", "rplQ", "rplR",
    "rplS", "rplT", "rplU", "rplV", "rplW", "rplX", "rplY",
    "rpmA", "rpmB", "rpmC", "rpmD", "rpmE", "rpmF", "rpmG", "rpmH",
    "rpoA", "rpoB", "rpoC", "rpoD", "rpoE",
    "rpsA", "rpsB", "rpsC", "rpsD", "rpsE", "rpsF", "rpsG", "rpsH",
    "rpsI", "rpsJ", "rpsK", "rpsL", "rpsM", "rpsN", "rpsO", "rpsP",
    "rpsQ", "rpsR", "rpsS", "rpsT", "rpsU",
    "secA", "secB", "secD", "secE", "secF", "secY",
    "serS", "ssb",
    "thiL", "thrS", "tilS", "trpS", "tsf", "tyrS",
    "ubiA", "ubiB", "ubiC", "ubiD", "ubiE", "ubiF", "ubiG", "ubiH", "ubiX",
    "valS",
    "zipA",
})

# Reference data keyed by organism name
_REFERENCE_DATA: dict[str, dict] = {
    "ecoli": {
        "essential_genes": _KEIO_ESSENTIAL,
        "source": "Keio collection (Baba et al. 2006, Nature Methods)",
        "display_name": "E. coli K-12",
    },
}

_DISPLAY_NAMES: dict[str, str] = {
    "ecoli":         "E. coli K-12",
    "bsubtilis":     "B. subtilis 168",
    "scerevisiae":   "S. cerevisiae",
    "mtuberculosis": "M. tuberculosis",
    "styphimurium":  "S. typhimurium",
    "paeruginosa":   "P. aeruginosa",
}

ESSENTIAL_THRESHOLD = 0.01   # growth < 1 % of wildtype -> essential


# ---------------------------------------------------------------------------
# Single-gene deletion
# ---------------------------------------------------------------------------

def run_single_gene_deletion(
    model: "cobra.Model",
    threads: int = 1,
) -> dict[str, float]:
    """Return {gene_id: growth_rate} for every gene in the model.

    Uses COBRApy's optimised single_gene_deletion which batches all
    knockouts into one solver session per gene.
    """
    from cobra.flux_analysis import single_gene_deletion

    print(f"  Running {len(model.genes)} single-gene knockouts "
          f"(threads={threads}) ...")

    results = single_gene_deletion(model, processes=threads)

    # DataFrame has integer index; 'ids' column contains strings like "{b0870}"
    gene_growth: dict[str, float] = {}
    for _, row in results.iterrows():
        gid = list(row["ids"])[0]
        g = row["growth"]
        gene_growth[gid] = float(g) if g is not None and not math.isnan(g) else 0.0

    return gene_growth


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def validate(
    model_path: Path,
    organism: str,
    threads: int,
) -> dict:
    """Run full essentiality validation; return result dict."""
    import cobra

    display_name = _DISPLAY_NAMES.get(organism, organism)

    # ---- load model ----
    print(f"\nLoading {model_path} ...")
    model = cobra.io.read_sbml_model(str(model_path))
    print(f"  {len(model.reactions)} reactions  |  "
          f"{len(model.metabolites)} metabolites  |  "
          f"{len(model.genes)} genes")

    # ---- wildtype growth ----
    wt_sol = model.optimize()
    if wt_sol.status != "optimal":
        print(f"  WARNING: wildtype FBA returned status={wt_sol.status}")
    wt_growth = wt_sol.objective_value if wt_sol.status == "optimal" else 0.0
    threshold = ESSENTIAL_THRESHOLD * wt_growth
    print(f"  Wildtype growth: {wt_growth:.4f} h\u207b\u00b9")
    print(f"  Essential threshold (1%): {threshold:.6f} h\u207b\u00b9")

    # ---- single-gene deletions ----
    gene_growth = run_single_gene_deletion(model, threads=threads)

    # ---- classify ----
    predicted_essential: set[str] = set()
    predicted_nonessential: set[str] = set()
    for gid, g in gene_growth.items():
        if g < threshold:
            predicted_essential.add(gid)
        else:
            predicted_nonessential.add(gid)

    n_genes = len(model.genes)
    n_ess = len(predicted_essential)
    n_non = len(predicted_nonessential)

    # ---- comparison against reference ----
    ref = _REFERENCE_DATA.get(organism)
    comparison: dict | None = None

    if ref is not None:
        ref_essential: frozenset[str] = ref["essential_genes"]

        # Map Keio gene names -> b-numbers present in the model
        name_to_bid = {g.name: g.id for g in model.genes if g.name}

        keio_in_model: list[str] = []   # b-numbers of matched Keio genes
        not_in_model:  list[str] = []   # Keio gene names absent from model

        for name in ref_essential:
            bid = name_to_bid.get(name)
            if bid is not None:
                keio_in_model.append(bid)
            else:
                not_in_model.append(name)

        keio_essential_bids = set(keio_in_model)

        # Classify every model gene
        tp_genes: list[str] = []
        fp_genes: list[str] = []
        tn_genes: list[str] = []
        fn_genes: list[str] = []

        for gene in model.genes:
            bid = gene.id
            is_pred_essential  = bid in predicted_essential
            is_known_essential = bid in keio_essential_bids

            if is_known_essential and is_pred_essential:
                tp_genes.append(gene.name or bid)
            elif is_known_essential and not is_pred_essential:
                fn_genes.append(gene.name or bid)
            elif not is_known_essential and is_pred_essential:
                fp_genes.append(gene.name or bid)
            else:
                tn_genes.append(gene.name or bid)

        tp = len(tp_genes)
        fp = len(fp_genes)
        tn = len(tn_genes)
        fn = len(fn_genes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        mcc = _mcc(tp, fp, tn, fn)

        comparison = {
            "source":        ref["source"],
            "ref_essential": len(ref_essential),
            "not_in_model":  len(not_in_model),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision":     round(precision, 4),
            "recall":        round(recall,    4),
            "f1":            round(f1,        4),
            "mcc":           round(mcc,       4),
            "tp_genes":      sorted(tp_genes),
            "fp_genes":      sorted(fp_genes),
            "tn_genes":      sorted(tn_genes),
            "fn_genes":      sorted(fn_genes),
            "not_in_model_genes": sorted(not_in_model),
        }

    return {
        "organism":        organism,
        "display_name":    display_name,
        "model_path":      str(model_path),
        "wt_growth":       round(wt_growth, 6),
        "threshold":       round(threshold, 8),
        "n_genes":         n_genes,
        "n_essential":     n_ess,
        "n_nonessential":  n_non,
        "essential_genes": sorted(predicted_essential),
        "comparison":      comparison,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    org = result["display_name"]
    w = max(45, len(org) + 8)
    thick = "\u2550" * w
    thin  = "\u2500" * w

    print(f"\nGene Essentiality Validation ({org})")
    print(thick)

    wt   = result["wt_growth"]
    n    = result["n_genes"]
    n_e  = result["n_essential"]
    n_ne = result["n_nonessential"]

    print(f"  Wildtype growth:         {wt:.4f} h\u207b\u00b9")
    print(f"  Genes tested:            {n}")
    print(f"  Predicted essential:     {n_e}  ({100*n_e/n:.1f}%)" if n else "")
    print(f"  Predicted non-essential: {n_ne}  ({100*n_ne/n:.1f}%)" if n else "")

    cmp = result.get("comparison")
    if cmp:
        print(f"\n  vs {cmp['source']}:")
        print(f"  {thin}")
        print(f"  Reference essential genes:  {cmp['ref_essential']}")
        if cmp["not_in_model"]:
            print(f"  Not in model (skipped):     {cmp['not_in_model']}"
                  f"  (different gene IDs)")
        print()
        print(f"  True positives:   {cmp['tp']:4d}  (correctly predicted essential)")
        print(f"  False positives:  {cmp['fp']:4d}  (predicted essential, actually not)")
        print(f"  True negatives:   {cmp['tn']:4d}  (correctly predicted non-essential)")
        print(f"  False negatives:  {cmp['fn']:4d}  (missed essential genes)")
        print()
        print(f"  Precision:  {cmp['precision']:.3f}")
        print(f"  Recall:     {cmp['recall']:.3f}")
        print(f"  F1:         {cmp['f1']:.3f}")
        print(f"  MCC:        {cmp['mcc']:.3f}  (Matthews correlation coefficient)")
    else:
        print(f"\n  No reference data available for organism '{result['organism']}'.")
        print(f"  (Add reference essential genes to _REFERENCE_DATA in this script.)")

    print(thick)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate gene essentiality predictions from a gemiz model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to gemiz output model (.xml).",
    )
    parser.add_argument(
        "--organism",
        required=True,
        help="Organism name (e.g. ecoli, bsubtilis).",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="JSON output path (default: data/validation/{organism}_essentiality.json).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Parallel workers for single-gene deletion (default: 1).",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    output = args.output or Path(
        f"data/validation/{args.organism}_essentiality.json"
    )

    result = validate(
        model_path=args.model,
        organism=args.organism,
        threads=args.threads,
    )

    print_report(result)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Remove per-gene lists from summary but keep them in JSON
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results saved to {output}")


if __name__ == "__main__":
    main()
