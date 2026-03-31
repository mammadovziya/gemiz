#!/usr/bin/env python3
"""Build cross-organism SBML templates for leave-one-out benchmarking.

For each target organism, merges the gold-standard models of all *other*
organisms into a single universal template.  The resulting template can be
used as the ``--template`` argument to ``gemiz carve`` to test how well
gemiz reconstructs an organism's model without using its own gold standard.

Usage
-----
    python scripts/build_cross_templates.py

    # Skip organisms whose gold standard is missing:
    python scripts/build_cross_templates.py --skip-missing

Output
------
    data/universal/cross_template_{org}.xml   (one per organism)
    data/organisms/{org}/config.json          (template key updated)
    ecoli uses data/universal/cross_template_ecoli.xml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Organism registry
# ---------------------------------------------------------------------------

ORGANISMS: dict[str, str] = {
    "ecoli":         "data/universal/iML1515.xml",
    "bsubtilis":     "data/organisms/bsubtilis/gold_standard.xml",
    "scerevisiae":   "data/organisms/scerevisiae/gold_standard.xml",
    "mtuberculosis": "data/organisms/mtuberculosis/gold_standard.xml",
    "styphimurium":  "data/organisms/styphimurium/gold_standard.xml",
}

OUTPUT_DIR = Path("data/universal")

# Config paths for non-ecoli organisms (ecoli config lives elsewhere)
CONFIG_PATHS: dict[str, Path] = {
    org: Path(f"data/organisms/{org}/config.json")
    for org in ORGANISMS
    if org != "ecoli"
}


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_models(models: "list[cobra.Model]", target_org: str) -> "cobra.Model":
    """Merge a list of COBRApy models into one.

    Reactions are deduplicated by ID.  The first model supplies the base
    (stoichiometry, bounds, GPR).  Reactions from subsequent models are
    appended only when their ID is not already present.
    """
    import cobra

    base = models[0].copy()
    base.id = f"cross_template_{target_org}"
    base.name = f"Cross-organism template (excl. {target_org})"

    existing_ids = {r.id for r in base.reactions}

    for m in models[1:]:
        new_rxns = [r.copy() for r in m.reactions if r.id not in existing_ids]
        if new_rxns:
            base.add_reactions(new_rxns)
            existing_ids.update(r.id for r in new_rxns)

    return base


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def update_config(org: str, template_path: Path) -> None:
    """Write the template key into data/organisms/{org}/config.json.

    Creates a minimal config if the file does not yet exist.
    """
    config_path = CONFIG_PATHS.get(org)
    if config_path is None:
        return  # ecoli — no config to update

    config: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    config["template"] = str(template_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"    config updated: {config_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-organism SBML templates (leave-one-out).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip organisms whose gold-standard file is missing "
             "instead of exiting with an error.",
    )
    args = parser.parse_args()

    import cobra

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- pre-flight: check which models are available ----
    available: dict[str, Path] = {}
    for org, path_str in ORGANISMS.items():
        p = Path(path_str)
        if p.exists():
            available[org] = p
        else:
            msg = f"  MISSING: {p}  ({org})"
            if args.skip_missing:
                print(msg + " — skipping")
            else:
                print(msg)
                print("Run setup_organism.py or use --skip-missing.")
                raise SystemExit(1)

    if len(available) < 2:
        print("Need at least 2 available models to build any cross-template.")
        raise SystemExit(1)

    # ---- load all available models once ----
    print("\nLoading gold-standard models...")
    loaded: dict[str, "cobra.Model"] = {}
    for org, path in available.items():
        print(f"  {org:<15} {path}")
        loaded[org] = cobra.io.read_sbml_model(str(path))
        print(f"             -> {len(loaded[org].reactions):,} reactions  "
              f"|  {len(loaded[org].genes):,} genes")

    # ---- build one cross-template per available organism ----
    print()
    for target_org in list(available):
        sources = [org for org in available if org != target_org]

        if not sources:
            print(f"  {target_org}: no other models to merge — skipping.")
            continue

        source_models = [loaded[org] for org in sources]
        source_labels = ", ".join(sources)

        print(f"[{target_org}]  merging: {source_labels}")

        merged = merge_models(source_models, target_org)
        n_merged = len(merged.reactions)

        # Overlap stats vs target's own gold standard
        target_ids  = {r.id for r in loaded[target_org].reactions}
        overlap     = len({r.id for r in merged.reactions} & target_ids)
        overlap_pct = 100 * overlap / len(target_ids) if target_ids else 0.0

        print(f"    {n_merged:,} reactions from {len(sources)} organisms  "
              f"|  {overlap:,}/{len(target_ids):,} overlap with {target_org} "
              f"gold standard ({overlap_pct:.1f}%)")

        out_path = OUTPUT_DIR / f"cross_template_{target_org}.xml"
        cobra.io.write_sbml_model(merged, str(out_path))
        size_kb = out_path.stat().st_size / 1024
        print(f"    saved -> {out_path}  ({size_kb:.0f} KB)")

        update_config(target_org, out_path)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
