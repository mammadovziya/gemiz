"""Step 4 — Reaction gapfilling via HiGHS MILP.

Strategy
--------
1. Build a draft model from DIAMOND hits (reactions with gene associations).
2. Identify blocked objectives (growth, ATP maintenance).
3. Solve a MILP to find the minimal set of reactions from the universal model
   that restore feasibility — using HiGHS via COBRApy's solver interface.

HiGHS is configured as COBRApy's default solver in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def run(
    *,
    hits: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    db: str,
    solver: str,
    verbose: bool,
) -> "cobra.Model":  # type: ignore[name-defined]  # noqa: F821
    """Build draft GEM, gapfill, and return a feasible COBRApy Model."""
    import cobra  # type: ignore[import-untyped]
    from cobra.flux_analysis import gapfill as cobra_gapfill

    # ---- 1. Build draft model from alignment hits -------------------------
    draft = _build_draft_model(hits, db=db, verbose=verbose)

    if verbose:
        print(
            f"[gapfill] Draft model: "
            f"{len(draft.reactions)} rxns, {len(draft.metabolites)} mets"
        )

    # ---- 2. Load universal model (template for gapfilling) ---------------
    universal = _load_universal_model(db=db)

    # ---- 3. Configure solver to HiGHS ------------------------------------
    cobra.Configuration().solver = solver  # "highs" or "glpk"

    # ---- 4. MILP gapfilling ----------------------------------------------
    solution = cobra_gapfill(draft, universal, demand_reactions=False)
    reactions_to_add = solution[0]  # first (minimal) solution

    if verbose:
        print(f"[gapfill] Adding {len(reactions_to_add)} reactions to close gaps")

    for rxn in reactions_to_add:
        draft.add_reactions([rxn])

    # ---- 5. Sanity check --------------------------------------------------
    sol = draft.optimize()
    if sol.status != "optimal":
        raise RuntimeError(
            f"Gapfilled model is still infeasible (status={sol.status}). "
            "Try --db both or check your reference databases."
        )

    if verbose:
        print(f"[gapfill] Objective value after gapfilling: {sol.objective_value:.4f}")

    return draft


# ---------------------------------------------------------------------------
# Internal helpers (stubs — to be implemented per-database)
# ---------------------------------------------------------------------------

def _build_draft_model(hits: pd.DataFrame, *, db: str, verbose: bool) -> "cobra.Model":  # noqa: F821
    """Convert DIAMOND hits into a COBRApy draft model."""
    import cobra  # type: ignore[import-untyped]

    model = cobra.Model("gemiz_draft")
    # TODO: map hit sseqid → BiGG / ModelSEED reactions using DB lookup tables
    #       and populate model.reactions with gene-protein-reaction associations
    if verbose:
        print(f"[gapfill] Draft model built from {len(hits)} alignment hits")
    return model


def _load_universal_model(*, db: str) -> "cobra.Model":  # noqa: F821
    """Load the universal reaction database as a COBRApy Model."""
    import cobra  # type: ignore[import-untyped]

    # TODO: load pre-built universal model from data/databases/
    #       For now return an empty model so the pipeline skeleton runs.
    return cobra.Model("universal")
