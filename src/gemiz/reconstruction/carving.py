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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cobra

INF = 1e30
SMALL_GAPFILL_COPY_LIMIT = 25


def model_growth_rate(model: "cobra.Model") -> float:
    """Return the objective value using COBRApy's fast scalar optimizer."""
    try:
        value = model.slim_optimize(error_value=0.0)
    except Exception:
        return 0.0
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _model_grows(model: "cobra.Model", threshold: float = 1e-6) -> bool:
    return model_growth_rate(model) > threshold


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def carve_model(
    universal_model: "cobra.Model",
    reaction_scores: dict[str, float],
    min_growth: float = 0.1,
    epsilon: float = 0.001,
    bigM: float = 1000.0,
    neutral_penalty: float = -0.01,
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
    neutral_penalty
        Small MILP objective penalty for zero-evidence reactions. This keeps
        required spontaneous/transport reactions available through growth
        constraints, while discouraging arbitrary free extras.

    Returns
    -------
    cobra.Model
        Carved organism-specific model.
    """
    print("[gemiz] Setting up MILP carving problem...")
    milp_data = setup_milp(universal_model, reaction_scores, min_growth,
                           epsilon, bigM, neutral_penalty)

    n = milp_data["n_reactions"]
    m = milp_data["n_metabolites"]
    scores = milp_data["scores"]
    raw_scores = milp_data["raw_scores"]
    n_pos = int(np.sum(raw_scores > 0))
    n_neg = int(np.sum(raw_scores < 0))
    n_neu = n - n_pos - n_neg

    print(f"[gemiz]   Reactions: {n}")
    print(f"[gemiz]   Metabolites: {m}")
    print(f"[gemiz]   Binary variables: {n}")
    print(f"[gemiz]   Positive-score reactions: {n_pos}")
    print(f"[gemiz]   Negative-score reactions: {n_neg}")
    print(f"[gemiz]   Neutral reactions: {n_neu}")
    if neutral_penalty != 0.0 and n_neu:
        print(f"[gemiz]   Neutral-reaction penalty: {neutral_penalty:g}")

    # ---- detect universal mode (multiple biomass candidates) ----
    biomass_candidates = milp_data.get("biomass_candidates", [])
    universal_mode = len(biomass_candidates) > 1

    if universal_mode:
        # Universal mode: do NOT enforce biomass constraint during carving.
        # We don't know which biomass is correct for an unknown organism.
        # Let scoring drive selection, then test growth post-carving.
        print("[gemiz] Universal mode: no biomass constraint during MILP "
              "(will test growth post-carving)")
        milp_data["enforce_biomass"] = False
    else:
        milp_data["enforce_biomass"] = True

    # ---- solve ----
    print("[gemiz] Solving with HiGHS...")
    result = solve_highs_milp(milp_data, time_limit=300.0)

    if result["status"] == "infeasible" and milp_data["enforce_biomass"]:
        print("[gemiz] WARNING: Infeasible with min_growth="
              f"{min_growth}, relaxing to 0.001...")
        milp_data["min_growth"] = 0.001
        result = solve_highs_milp(milp_data, time_limit=300.0)

    if result["status"] == "infeasible" and milp_data["enforce_biomass"]:
        print("[gemiz] WARNING: Still infeasible. "
              "Removing biomass constraint entirely...")
        milp_data["enforce_biomass"] = False
        result = solve_highs_milp(milp_data, time_limit=300.0)

    if result["status"] == "infeasible":
        print("[gemiz] WARNING: MILP infeasible even without biomass constraint.")
        print("[gemiz] WARNING: Falling back to all positive-score reactions. "
              "Model quality may be poor.")
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
    print("[gemiz] Carving complete:")
    print(f"[gemiz]   Reactions kept: {n_kept}")
    print(f"[gemiz]   Reactions removed: {n - n_kept}")
    print(f"[gemiz]   Metabolites: {len(carved.metabolites)}")
    print(f"[gemiz]   Genes: {len(carved.genes)}")

    # ---- select best biomass (universal template mode) ----
    biomass_candidates = milp_data.get("biomass_candidates", [])
    if len(biomass_candidates) > 1:
        carved = _select_biomass(carved, milp_data["rxn_ids"],
                                 biomass_candidates)

    return carved


# ---------------------------------------------------------------------------
# Biomass selection (universal template mode)
# ---------------------------------------------------------------------------

def _select_biomass(
    carved: "cobra.Model",
    rxn_ids: list[str],
    biomass_candidate_indices: list[int],
) -> "cobra.Model":
    """Try each candidate biomass reaction and keep the best one.

    Filters candidates strictly: only reactions with 'biomass' in their ID
    are considered (avoids false positives like PFK being tagged as biomass).
    Rejects biologically impossible growth rates (> 10 h^-1).
    """
    candidate_ids = [rxn_ids[i] for i in biomass_candidate_indices]

    # Filter to candidates that survived carving
    present = []
    for rid in candidate_ids:
        try:
            carved.reactions.get_by_id(rid)
            present.append(rid)
        except KeyError:
            continue

    if not present:
        print("[gemiz]   No candidate biomass reactions survived carving.")
        return carved

    # Strict filter: only reactions with 'biomass' in the ID
    biomass_filtered = [rid for rid in present if "biomass" in rid.lower()]

    # Fallback: try 'growth' in the ID
    if not biomass_filtered:
        biomass_filtered = [rid for rid in present if "growth" in rid.lower()]

    if not biomass_filtered:
        print(f"[gemiz]   WARNING: {len(present)} candidates tagged as biomass "
              f"but none have 'biomass' or 'growth' in ID. Skipping.")
        return carved

    print(f"[gemiz] Selecting biomass from {len(biomass_filtered)} candidates "
          f"(filtered from {len(present)} tagged)...")

    best_id: str | None = None

    for rid in biomass_filtered:
        rxn = carved.reactions.get_by_id(rid)
        notes = rxn.notes or {}
        source = notes.get("gemiz_biomass", "?")

        with carved:
            carved.objective = rid
            gr = model_growth_rate(carved)

        # Reject biologically impossible growth (> 10 h^-1)
        if gr > 10.0:
            print(f"[gemiz]   Rejected: {rid} (from {source}) "
                  f"growth={gr:.4f} h^-1 (biologically impossible)")
            continue

        if 0 < gr < 10.0:
            best_id = rid
            print(f"[gemiz]   Selected: {rid} (from {source}) "
                  f"growth={gr:.4f} h^-1")
            break

        print(f"[gemiz]   Tried: {rid} (from {source}) growth={gr:.4f}")

    if best_id is not None:
        carved.objective = best_id
    else:
        print("[gemiz]   WARNING: No candidate biomass produced valid growth (0-10 h^-1).")

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
    neutral_penalty: float = -0.01,
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
    raw_scores = np.array(
        [reaction_scores.get(r.id, 0.0) for r in model.reactions],
        dtype=np.float64,
    )
    scores = raw_scores.copy()
    if neutral_penalty != 0.0:
        scores[raw_scores == 0.0] = neutral_penalty

    # ---- biomass reaction(s) ----
    # Primary: reaction with non-zero objective coefficient
    biomass_idx = None
    for i, rxn in enumerate(model.reactions):
        if rxn.objective_coefficient != 0:
            biomass_idx = i
            break

    # Collect candidate biomass reactions from notes (universal template)
    # E. coli biomass first (most common, most likely to work)
    biomass_candidates: list[int] = []
    if biomass_idx is None:
        ecoli_bio: list[int] = []
        other_bio: list[int] = []
        for i, rxn in enumerate(model.reactions):
            notes = rxn.notes or {}
            if "gemiz_biomass" in notes:
                src = notes.get("gemiz_biomass", "")
                if src.startswith("iML1515") or src.startswith("iJO1366"):
                    ecoli_bio.append(i)
                else:
                    other_bio.append(i)
        biomass_candidates = ecoli_bio + other_bio
        if biomass_candidates:
            biomass_idx = biomass_candidates[0]  # default for single-biomass path

    # Last fallback: reaction with 'biomass' in the id
    if biomass_idx is None:
        for i, rxn in enumerate(model.reactions):
            if "biomass" in rxn.id.lower():
                biomass_idx = i
                break

    if biomass_idx is not None:
        n_cand = len(biomass_candidates)
        extra = f" (+{n_cand - 1} candidates)" if n_cand > 1 else ""
        print(f"[gemiz]   Biomass reaction: {rxn_ids[biomass_idx]} "
              f"(index {biomass_idx}){extra}")
    else:
        print("[gemiz]   WARNING: No biomass reaction found in objective")

    return {
        "S": S,
        "lb": lb,
        "ub": ub,
        "scores": scores,
        "raw_scores": raw_scores,
        "biomass_idx": biomass_idx,
        "biomass_candidates": biomass_candidates,
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
    enforce_biomass = milp_data.get("enforce_biomass", True)

    if enforce_biomass and biomass_idx is not None:
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
# Gap-filling (Step 5.5)
# ---------------------------------------------------------------------------

def gapfill_model(
    carved_model: "cobra.Model",
    template_model: "cobra.Model",
    reaction_scores: "dict[str, float] | None" = None,
    timeout: float = 60.0,
) -> "tuple[cobra.Model, list[str]]":
    """Gap-fill a carved model to restore biomass growth.

    Tries COBRApy's MILP-based gapfill first (finds minimum reaction set).
    Falls back to greedy backward-elimination if that exceeds *timeout* seconds.

    Parameters
    ----------
    carved_model
        Output of :func:`carve_model` that cannot grow.
    template_model
        Universal template containing all candidate reactions.
    reaction_scores
        Per-reaction scores from Step 4, used to prioritise candidates in the
        greedy fallback (higher-score reactions are tried last for removal,
        so they stay in the model).
    timeout
        Seconds before abandoning COBRApy MILP gapfill and switching to greedy.

    Returns
    -------
    (filled_model, added_reaction_ids)
    """
    if _model_grows(carved_model):
        return carved_model.copy(), []

    carved_ids = {r.id for r in carved_model.reactions}
    candidates = [r for r in template_model.reactions if r.id not in carved_ids]

    if not candidates:
        print("[gemiz]   No candidate reactions available for gap-filling.")
        return carved_model.copy(), []

    added_ids: list[str] | None = []
    if len(candidates) > 1000:
        print(f"[gemiz]   {len(candidates)} candidate reactions. "
              "Trying fast prioritized gapfill first...")
        added_ids = _fast_forward_gapfill(
            carved_model, template_model, candidates, reaction_scores or {}
        )

    if not added_ids:
        print(f"[gemiz]   {len(candidates)} candidate reactions. "
              f"Trying MILP gapfill (timeout {timeout:.0f}s)...")
        added_ids = _try_cobra_gapfill(carved_model, template_model, timeout)

    if added_ids is None:
        print("[gemiz]   MILP gapfill timed out — switching to greedy heuristic...")
        added_ids = _fast_forward_gapfill(
            carved_model, template_model, candidates, reaction_scores or {}
        )
        if not added_ids:
            added_ids = _greedy_gapfill(
                carved_model, template_model, candidates, reaction_scores or {}
            )
    elif not added_ids:
        print("[gemiz]   MILP gapfill found no solution — switching to greedy heuristic...")
        added_ids = _fast_forward_gapfill(
            carved_model, template_model, candidates, reaction_scores or {}
        )
        if not added_ids:
            added_ids = _greedy_gapfill(
                carved_model, template_model, candidates, reaction_scores or {}
            )
    else:
        print(f"[gemiz]   MILP gapfill found {len(added_ids)} reactions to add.")

    if len(added_ids) <= SMALL_GAPFILL_COPY_LIMIT:
        try:
            filled = _augment_model_with_added_reactions(
                carved_model, template_model, added_ids
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[gemiz]   Fast gapfill model build failed: {exc}")
            filled_ids = carved_ids | set(added_ids)
            filled = _subset_template_model(template_model, filled_ids, carved_model)
    else:
        filled_ids = carved_ids | set(added_ids)
        filled = _subset_template_model(template_model, filled_ids, carved_model)

    return filled, added_ids


def _try_cobra_gapfill(
    carved_model: "cobra.Model",
    template_model: "cobra.Model",
    timeout: float,
) -> "list[str] | None":
    """Run COBRApy gapfill in a daemon thread; return None if it times out."""
    import threading

    result: list = [None]

    def _run() -> None:
        try:
            from cobra.flux_analysis import gapfill
            solutions = gapfill(carved_model, template_model,
                                demand_reactions=False)
            if solutions and solutions[0]:
                result[0] = [r.id for r in solutions[0]]
            else:
                result[0] = []
        except Exception as exc:  # noqa: BLE001
            print(f"[gemiz]   COBRApy gapfill error: {exc}")
            result[0] = []

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None  # timed out
    return result[0]


def _gapfill_candidate_key(
    reaction: "cobra.Reaction",
    reaction_scores: "dict[str, float]",
    objective_metabolites: "set[str] | None" = None,
) -> tuple[int, float, str]:
    """Sort likely one-reaction gapfill fixes before broad internals."""
    rid = reaction.id.lower()
    is_objective_precursor = bool(
        objective_metabolites
        and rid.startswith(("sink_", "dm_", "sk_"))
        and any(met.id in objective_metabolites for met in reaction.metabolites)
    )
    if is_objective_precursor:
        group = 0
    elif rid.startswith(("sink_", "dm_", "sk_")):
        group = 1
    elif rid.startswith("ex_"):
        group = 2
    elif rid.endswith(("tex", "texi", "tpp")) or "transport" in reaction.name.lower():
        group = 3
    else:
        group = 4
    return (group, -reaction_scores.get(reaction.id, 0.0), reaction.id)


def _objective_metabolite_ids(model: "cobra.Model") -> set[str]:
    """Return metabolites touched by the current objective reaction(s)."""
    metabolites: set[str] = set()
    for rxn in model.reactions:
        if rxn.objective_coefficient != 0:
            metabolites.update(met.id for met in rxn.metabolites)
    return metabolites


def _try_single_reaction_gapfill(
    carved_model: "cobra.Model",
    candidates: "list[cobra.Reaction]",
    reaction_scores: "dict[str, float]",
    max_candidates: int = 300,
) -> "list[str]":
    """Find a one-reaction gapfill fix without copying the full template."""
    objective_metabolites = _objective_metabolite_ids(carved_model)
    ranked = sorted(
        candidates,
        key=lambda reaction: _gapfill_candidate_key(
            reaction, reaction_scores, objective_metabolites
        ),
    )

    test_model = carved_model.copy()
    for rxn in ranked[:max_candidates]:
        with test_model:
            _add_template_reaction(test_model, rxn)
            if _model_grows(test_model):
                print(f"[gemiz]   Single-reaction gapfill: {rxn.id}")
                return [rxn.id]

    return []


def _fast_forward_gapfill(
    carved_model: "cobra.Model",
    template_model: "cobra.Model",
    candidates: "list[cobra.Reaction]",
    reaction_scores: "dict[str, float]",
    max_candidates: int = 250,
) -> "list[str]":
    """Try prioritized small candidate sets before the full greedy fallback."""
    carved_ids = {r.id for r in carved_model.reactions}
    single = _try_single_reaction_gapfill(
        carved_model, candidates, reaction_scores, max_candidates=max_candidates
    )
    if single:
        return single

    objective_metabolites = _objective_metabolite_ids(carved_model)
    ranked = sorted(
        candidates,
        key=lambda reaction: _gapfill_candidate_key(
            reaction, reaction_scores, objective_metabolites
        ),
    )

    tried_sets: set[frozenset[str]] = set()
    candidate_sets: list[list["cobra.Reaction"]] = []
    for max_group in (0, 1, 2):
        group = [
            rxn for rxn in ranked
            if _gapfill_candidate_key(
                rxn, reaction_scores, objective_metabolites
            )[0] <= max_group
        ]
        if group:
            candidate_sets.append(group[:max_candidates])
    candidate_sets.append(ranked[:max_candidates])

    for selected in candidate_sets:
        selected_ids = frozenset(rxn.id for rxn in selected)
        if not selected_ids or selected_ids in tried_sets:
            continue
        tried_sets.add(selected_ids)

        test_model = _subset_template_model(
            template_model,
            carved_ids | set(selected_ids),
            carved_model,
        )

        if _model_grows(test_model):
            for rxn in selected:
                try:
                    test_rxn = test_model.reactions.get_by_id(rxn.id)
                except KeyError:
                    continue
                with test_model:
                    test_model.remove_reactions([test_rxn], remove_orphans=False)
                    still_grows = _model_grows(test_model)
                if still_grows:
                    test_model.remove_reactions(
                        [test_model.reactions.get_by_id(rxn.id)],
                        remove_orphans=False,
                    )

            added = [
                r.id for r in test_model.reactions
                if r.id not in carved_ids
            ]
            print(f"[gemiz]   Fast gapfill restored growth with "
                  f"{len(added)} reaction(s).")
            return added

    return []


def _greedy_gapfill(
    carved_model: "cobra.Model",
    template_model: "cobra.Model",
    candidates: "list[cobra.Reaction]",
    reaction_scores: "dict[str, float]",
) -> "list[str]":
    """Greedy backward-elimination gap-filling.

    1. Starts from a full template copy.
    2. Confirms the augmented model can grow.
    3. Removes candidates one by one in ascending score order (least
       important first); keeps a reaction only when its removal kills growth.
    """
    try:
        test_model = _subset_template_model(
            template_model,
            {r.id for r in template_model.reactions},
            carved_model,
        )
    except RecursionError as exc:
        print(f"[gemiz]   Greedy gapfill setup failed: {exc}")
        return []

    if not _model_grows(test_model):
        print("[gemiz]   WARNING: model cannot grow even with full template. "
              "Gap-filling cannot help.")
        return []

    # Sort ascending by score — try removing low-score (less important) ones first
    candidates_sorted = sorted(
        candidates,
        key=lambda r: reaction_scores.get(r.id, 0.0),
    )

    for rxn in candidates_sorted:
        # Check the reaction is still in the model (may have been permanently removed)
        try:
            test_rxn = test_model.reactions.get_by_id(rxn.id)
        except KeyError:
            continue

        # Temporarily remove and test
        with test_model:
            test_model.remove_reactions([test_rxn], remove_orphans=False)
            still_grows = _model_grows(test_model)

        if still_grows:
            # Dispensable — permanently remove it
            test_model.remove_reactions(
                [test_model.reactions.get_by_id(rxn.id)],
                remove_orphans=False,
            )

    # Whatever remains (beyond the original carved reactions) is the minimal set
    carved_ids = {r.id for r in carved_model.reactions}
    return [r.id for r in test_model.reactions if r.id not in carved_ids]


def _clone_metabolite_for_model(
    metabolite: "cobra.Metabolite",
    model: "cobra.Model",
) -> "cobra.Metabolite":
    """Return the model-local metabolite with matching metadata."""
    try:
        return model.metabolites.get_by_id(metabolite.id)
    except KeyError:
        pass

    import cobra

    cloned = cobra.Metabolite(
        metabolite.id,
        formula=metabolite.formula,
        name=metabolite.name,
        charge=metabolite.charge,
        compartment=metabolite.compartment,
    )
    cloned.annotation = dict(metabolite.annotation or {})
    cloned.notes = dict(metabolite.notes or {})
    return cloned


def _add_template_reaction(
    model: "cobra.Model",
    template_reaction: "cobra.Reaction",
) -> None:
    """Add one template reaction without deep-copying the template model."""
    import cobra

    try:
        model.reactions.get_by_id(template_reaction.id)
        return
    except KeyError:
        pass

    reaction = cobra.Reaction(template_reaction.id)
    reaction.name = template_reaction.name
    reaction.lower_bound = template_reaction.lower_bound
    reaction.upper_bound = template_reaction.upper_bound
    reaction.subsystem = template_reaction.subsystem
    reaction.annotation = dict(template_reaction.annotation or {})
    reaction.notes = dict(template_reaction.notes or {})
    reaction.gene_reaction_rule = template_reaction.gene_reaction_rule

    reaction.add_metabolites({
        _clone_metabolite_for_model(met, model): coeff
        for met, coeff in template_reaction.metabolites.items()
    })
    model.add_reactions([reaction])


def _augment_model_with_added_reactions(
    carved_model: "cobra.Model",
    template_model: "cobra.Model",
    added_ids: "list[str]",
) -> "cobra.Model":
    """Copy the carved model and add a small number of template reactions."""
    filled = carved_model.copy()
    for rid in added_ids:
        _add_template_reaction(filled, template_model.reactions.get_by_id(rid))
    return filled


def _subset_template_model(
    template_model: "cobra.Model",
    keep_ids: "set[str]",
    objective_source: "cobra.Model",
) -> "cobra.Model":
    """Copy a template and keep only selected reactions."""
    model = template_model.copy()
    to_remove = [r for r in model.reactions if r.id not in keep_ids]
    if to_remove:
        model.remove_reactions(to_remove, remove_orphans=True)

    model.id = objective_source.id
    model.name = objective_source.name
    _copy_objective(model, objective_source)
    return model


def _copy_objective(
    target_model: "cobra.Model",
    source_model: "cobra.Model",
) -> None:
    """Preserve the selected biomass objective on a rebuilt model."""
    objective = {
        rxn.id: rxn.objective_coefficient
        for rxn in source_model.reactions
        if rxn.objective_coefficient != 0
    }
    if not objective:
        return

    present = {}
    for rid, coefficient in objective.items():
        try:
            present[target_model.reactions.get_by_id(rid)] = coefficient
        except KeyError:
            continue

    if present:
        target_model.objective = present


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
    growth_rate = model_growth_rate(model)
    can_grow = growth_rate > 1e-6
    if not can_grow:
        warnings.append("FBA objective is zero or infeasible")

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
