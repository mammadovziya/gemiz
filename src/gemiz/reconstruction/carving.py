"""Step 5 -- MILP-based model carving using HiGHS.

Given a universal metabolic model and per-reaction confidence scores from
Step 4, select the optimal subset of reactions that:

  1. Maximizes total evidence score
  2. Maintains mass balance (steady-state constraint S*v = 0)
  3. Ensures the organism can grow (biomass flux >= threshold)

Uses HiGHS (free, open-source MILP solver) via the highspy Python bindings.

MILP formulation
-----------------
  maximize   sum(score_i * y_i)

  subject to:
    S * v = 0                    (steady state, mass balance)
    lb_i * y_i  <=  v_i          (lower indicator)
    v_i  <=  ub_i * y_i          (upper indicator)
    y_i in {0, 1}                (binary inclusion)
    v_biomass >= min_growth       (growth requirement)

  When y_i = 0: v_i = 0  (reaction excluded, no flux)
  When y_i = 1: lb_i <= v_i <= ub_i  (reaction available)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

INF = 1e30


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def carve_model(
    universal_model: "cobra.Model",
    reaction_scores: dict[str, float],
    min_growth: float = 0.1,
    epsilon: float = 0.001,
    bigM: float = 1000.0,
) -> "cobra.Model":
    """Carve an organism-specific model from a universal template.

    Parameters
    ----------
    universal_model
        Template model with all candidate reactions (e.g. iML1515).
    reaction_scores
        ``{reaction_id: score}`` from ``compute_reaction_scores()``.
    min_growth
        Minimum required biomass flux (h^-1).
    epsilon
        Small flux for indicator constraints.
    bigM
        Big-M constant for indicator constraints.

    Returns
    -------
    cobra.Model
        Carved organism-specific model.
    """
    print("[gemiz] Setting up MILP carving problem...")
    milp_data = setup_milp(universal_model, reaction_scores, min_growth,
                           epsilon, bigM)

    n = milp_data["n_reactions"]
    m = milp_data["n_metabolites"]
    scores = milp_data["scores"]
    n_pos = int(np.sum(scores > 0))
    n_neg = int(np.sum(scores < 0))
    n_neu = n - n_pos - n_neg

    print(f"[gemiz]   Reactions: {n}")
    print(f"[gemiz]   Metabolites: {m}")
    print(f"[gemiz]   Binary variables: {n}")
    print(f"[gemiz]   Positive-score reactions: {n_pos}")
    print(f"[gemiz]   Negative-score reactions: {n_neg}")
    print(f"[gemiz]   Neutral reactions: {n_neu}")

    # ---- solve ----
    print("[gemiz] Solving with HiGHS...")
    result = solve_highs_milp(milp_data, time_limit=300.0)

    if result["status"] == "infeasible":
        print("[gemiz] WARNING: Infeasible with min_growth="
              f"{min_growth}, relaxing to 0.01...")
        milp_data["min_growth"] = 0.01
        result = solve_highs_milp(milp_data, time_limit=300.0)

    if result["status"] == "infeasible":
        print("[gemiz] WARNING: Still infeasible. "
              "Using all positive-score reactions as fallback.")
        result["active_reactions"] = [
            i for i in range(n) if scores[i] > 0
        ]
        result["status"] = "fallback"
        result["objective"] = float(np.sum(scores[scores > 0]))

    print(f"[gemiz] MILP solved in {result['solve_time']:.1f}s")
    print(f"[gemiz]   Status: {result['status'].title()}")
    print(f"[gemiz]   Objective: {result['objective']:.2f}")

    # ---- extract model ----
    carved = extract_carved_model(
        universal_model, result["active_reactions"], milp_data["rxn_ids"],
    )

    n_kept = len(carved.reactions)
    print(f"[gemiz] Carving complete:")
    print(f"[gemiz]   Reactions kept: {n_kept}")
    print(f"[gemiz]   Reactions removed: {n - n_kept}")
    print(f"[gemiz]   Metabolites: {len(carved.metabolites)}")
    print(f"[gemiz]   Genes: {len(carved.genes)}")

    return carved


# ---------------------------------------------------------------------------
# MILP construction
# ---------------------------------------------------------------------------

def setup_milp(
    model: "cobra.Model",
    reaction_scores: dict[str, float],
    min_growth: float,
    epsilon: float,
    bigM: float,
) -> dict:
    """Build MILP problem data from a COBRA model and reaction scores.

    Returns a dict consumed by :func:`solve_highs_milp`.
    """
    n = len(model.reactions)
    m = len(model.metabolites)

    rxn_ids = [r.id for r in model.reactions]

    # ---- stoichiometric matrix (dense, m x n) ----
    met_index = {met.id: i for i, met in enumerate(model.metabolites)}
    S = np.zeros((m, n), dtype=np.float64)
    for j, rxn in enumerate(model.reactions):
        for met, coeff in rxn.metabolites.items():
            S[met_index[met.id], j] = coeff

    lb = np.array([r.lower_bound for r in model.reactions], dtype=np.float64)
    ub = np.array([r.upper_bound for r in model.reactions], dtype=np.float64)
    scores = np.array(
        [reaction_scores.get(r.id, 0.0) for r in model.reactions],
        dtype=np.float64,
    )

    # ---- biomass reaction (objective) ----
    biomass_idx = None
    for i, rxn in enumerate(model.reactions):
        if rxn.objective_coefficient != 0:
            biomass_idx = i
            break

    if biomass_idx is not None:
        print(f"[gemiz]   Biomass reaction: {rxn_ids[biomass_idx]} "
              f"(index {biomass_idx})")
    else:
        print("[gemiz]   WARNING: No biomass reaction found in objective")

    return {
        "S": S,
        "lb": lb,
        "ub": ub,
        "scores": scores,
        "biomass_idx": biomass_idx,
        "n_reactions": n,
        "n_metabolites": m,
        "rxn_ids": rxn_ids,
        "min_growth": min_growth,
        "epsilon": epsilon,
        "bigM": bigM,
    }


# ---------------------------------------------------------------------------
# HiGHS solver
# ---------------------------------------------------------------------------

def solve_highs_milp(
    milp_data: dict,
    time_limit: float = 300.0,
) -> dict:
    """Solve the carving MILP using HiGHS via highspy.

    Variable layout (total = 2n)::

        v[0 .. n-1]   continuous flux
        y[n .. 2n-1]  binary inclusion

    Constraints::

        S * v = 0                          (mass balance)
        v_i  <=  ub_i * y_i               (upper indicator)
        v_i  >=  lb_i * y_i               (lower indicator)
        v_biomass >= min_growth            (growth)
    """
    import highspy

    n = milp_data["n_reactions"]
    m = milp_data["n_metabolites"]
    S = milp_data["S"]
    lb = milp_data["lb"]
    ub = milp_data["ub"]
    scores = milp_data["scores"]
    biomass_idx = milp_data["biomass_idx"]
    min_growth = milp_data["min_growth"]

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    # ── variables ──────────────────────────────────────────────────────
    # v[0..n-1]: continuous flux (bounds relaxed to include 0)
    for i in range(n):
        h.addVar(min(float(lb[i]), 0.0), max(float(ub[i]), 0.0))

    # y[n..2n-1]: binary inclusion
    for i in range(n):
        h.addVar(0.0, 1.0)
        h.changeColIntegrality(n + i, highspy.HighsVarType.kInteger)

    # ── objective: maximize sum(score_i * y_i) ────────────────────────
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    for i in range(n):
        h.changeColCost(n + i, float(scores[i]))

    # ── constraint 1: mass balance  S * v = 0  (m rows) ──────────────
    for i in range(m):
        nz_idx: list[int] = []
        nz_val: list[float] = []
        for j in range(n):
            if S[i, j] != 0.0:
                nz_idx.append(j)
                nz_val.append(float(S[i, j]))
        if nz_idx:
            h.addRow(0.0, 0.0, len(nz_idx), nz_idx, nz_val)

    # ── constraint 2: upper indicator  v_i <= ub_i * y_i ─────────────
    #   v_i - ub_i * y_i <= 0
    for i in range(n):
        if ub[i] > 0:
            h.addRow(-INF, 0.0, 2, [i, n + i], [1.0, -float(ub[i])])

    # ── constraint 3: lower indicator  v_i >= lb_i * y_i ─────────────
    #   v_i - lb_i * y_i >= 0
    for i in range(n):
        if lb[i] < 0:
            h.addRow(0.0, INF, 2, [i, n + i], [1.0, -float(lb[i])])
        elif lb[i] > 0:
            # Rare: forced-flux reactions. Need indicator so v=0 when y=0.
            h.addRow(0.0, INF, 2, [i, n + i], [1.0, -float(lb[i])])

    # ── constraint 4: growth requirement ──────────────────────────────
    if biomass_idx is not None:
        h.addRow(float(min_growth), INF, 1, [biomass_idx], [1.0])

    # ── solve ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t0

    ms = h.getModelStatus()

    if ms == highspy.HighsModelStatus.kOptimal:
        status = "optimal"
    elif ms == highspy.HighsModelStatus.kInfeasible:
        status = "infeasible"
    elif ms == highspy.HighsModelStatus.kTimeLimit:
        status = "timeout"
    elif ms == highspy.HighsModelStatus.kObjectiveBound:
        status = "optimal"
    else:
        status = f"error ({ms})"

    # ── extract solution ──────────────────────────────────────────────
    if status in ("optimal", "timeout"):
        sol = h.getSolution()
        cv = sol.col_value
        y_vals = [cv[n + i] for i in range(n)]
        active = [i for i in range(n) if y_vals[i] > 0.5]
        obj_val = sum(float(scores[i]) * y_vals[i] for i in range(n))
    else:
        active = []
        obj_val = 0.0

    return {
        "status": status,
        "objective": obj_val,
        "active_reactions": active,
        "solve_time": solve_time,
    }


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------

def extract_carved_model(
    universal_model: "cobra.Model",
    active_reaction_indices: list[int],
    rxn_ids: list[str],
) -> "cobra.Model":
    """Build organism-specific model by removing inactive reactions."""
    model = universal_model.copy()
    active_ids = {rxn_ids[i] for i in active_reaction_indices}

    to_remove = [r for r in model.reactions if r.id not in active_ids]
    model.remove_reactions(to_remove, remove_orphans=True)

    model.id = "gemiz_carved"
    model.name = "GEM carved by gemiz"

    return model


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_model(model: "cobra.Model") -> dict:
    """Run sanity checks on a carved model."""
    print("[gemiz] Verifying carved model...")

    warnings: list[str] = []
    n_rxns = len(model.reactions)
    n_mets = len(model.metabolites)
    n_genes = len(model.genes)

    if n_rxns < 100:
        warnings.append(f"Very few reactions ({n_rxns})")

    # biomass
    has_biomass = any(r.objective_coefficient != 0 for r in model.reactions)
    if not has_biomass:
        warnings.append("No biomass reaction in objective")

    # FBA
    can_grow = False
    growth_rate = 0.0
    try:
        sol = model.optimize()
        if sol.status == "optimal":
            growth_rate = sol.objective_value
            can_grow = growth_rate > 1e-6
        else:
            warnings.append(f"FBA status: {sol.status}")
    except Exception as e:
        warnings.append(f"FBA failed: {e}")

    # orphan metabolites
    orphans = sum(1 for met in model.metabolites if len(met.reactions) == 0)
    if orphans:
        warnings.append(f"{orphans} orphan metabolites")

    grow_mark = "YES" if can_grow else "NO"
    print(f"[gemiz]   Growth rate: {growth_rate:.4f} h^-1  {grow_mark}")
    print(f"[gemiz]   Reactions: {n_rxns}")
    print(f"[gemiz]   Metabolites: {n_mets}")
    print(f"[gemiz]   Genes: {n_genes}")
    for w in warnings:
        print(f"[gemiz]   WARNING: {w}")

    return {
        "can_grow": can_grow,
        "growth_rate": growth_rate,
        "n_reactions": n_rxns,
        "n_metabolites": n_mets,
        "n_genes": n_genes,
        "warnings": warnings,
    }
