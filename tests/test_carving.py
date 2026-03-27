"""Tests for Step 5 -- HiGHS MILP Carving.

All tests require iML1515 + reaction scores from Step 4.
Tests are skipped if data files are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemiz.reconstruction.carving import (
    carve_model,
    extract_carved_model,
    setup_milp,
    solve_highs_milp,
    verify_model,
)
from gemiz.reconstruction.scoring import load_universal_model

MODEL_PATH  = Path("data/universal/iML1515.xml")
SCORES_PATH = Path("data/test_outputs/ecoli_reaction_scores.json")


# ---------------------------------------------------------------------------
# Fixtures (shared across all tests, computed once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iml1515():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return load_universal_model(str(MODEL_PATH))


@pytest.fixture(scope="module")
def reaction_scores():
    if not SCORES_PATH.exists():
        pytest.skip(f"Scores not found: {SCORES_PATH} "
                     "(run test_scoring tests on WSL2 first)")
    with open(SCORES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def milp_data(iml1515, reaction_scores):
    return setup_milp(iml1515, reaction_scores, min_growth=0.1,
                      epsilon=0.001, bigM=1000.0)


@pytest.fixture(scope="module")
def milp_result(milp_data):
    return solve_highs_milp(milp_data, time_limit=300.0)


@pytest.fixture(scope="module")
def carved(iml1515, milp_result, milp_data):
    return extract_carved_model(
        iml1515, milp_result["active_reactions"], milp_data["rxn_ids"],
    )


# ---------------------------------------------------------------------------
# Test 1 -- setup_milp
# ---------------------------------------------------------------------------

def test_setup_milp(milp_data):
    """Verify MILP problem dimensions and score distribution."""
    n = milp_data["n_reactions"]
    m = milp_data["n_metabolites"]
    scores = milp_data["scores"]

    n_pos = int((scores > 0).sum())
    n_neg = int((scores < 0).sum())
    n_neu = n - n_pos - n_neg

    print(f"\n  Reactions:          {n}")
    print(f"  Metabolites:        {m}")
    print(f"  Score range:        [{scores.min():.1f}, {scores.max():.1f}]")
    print(f"  Positive reactions: {n_pos}")
    print(f"  Negative reactions: {n_neg}")
    print(f"  Neutral reactions:  {n_neu}")

    assert n == 2712
    assert m == 1877
    assert milp_data["biomass_idx"] is not None
    assert n_pos > 1000
    assert milp_data["S"].shape == (m, n)


# ---------------------------------------------------------------------------
# Test 2 -- solve_highs_milp
# ---------------------------------------------------------------------------

def test_solve_highs_milp(milp_result):
    """Verify HiGHS finds an optimal solution."""
    n_active = len(milp_result["active_reactions"])

    print(f"\n  Status:           {milp_result['status'].title()}")
    print(f"  Solve time:       {milp_result['solve_time']:.1f}s")
    print(f"  Objective value:  {milp_result['objective']:.2f}")
    print(f"  Active reactions: {n_active}")

    assert milp_result["status"] == "optimal"
    assert n_active > 500
    assert n_active < 2500
    assert milp_result["objective"] > 0


# ---------------------------------------------------------------------------
# Test 3 -- carve_model (full pipeline)
# ---------------------------------------------------------------------------

def test_carve_model(iml1515, reaction_scores):
    """Full carving pipeline: setup -> solve -> extract."""
    model = carve_model(iml1515, reaction_scores)

    n_orig = len(iml1515.reactions)
    print(f"\n  Reactions:   {len(model.reactions)} (was {n_orig})")
    print(f"  Metabolites: {len(model.metabolites)} (was {len(iml1515.metabolites)})")
    print(f"  Genes:       {len(model.genes)} (was {len(iml1515.genes)})")

    assert len(model.reactions) > 500
    assert len(model.reactions) < 2500
    assert len(model.metabolites) > 200
    assert len(model.genes) > 100


# ---------------------------------------------------------------------------
# Test 4 -- verify_model
# ---------------------------------------------------------------------------

def test_verify_model(carved):
    """Verify the carved model can grow."""
    result = verify_model(carved)

    print(f"\n  Growth rate: {result['growth_rate']:.4f} h^-1")
    print(f"  Can grow:    {result['can_grow']}")

    assert result["can_grow"], "Carved model cannot grow!"
    assert result["growth_rate"] > 0.01


# ---------------------------------------------------------------------------
# Test 5 -- compare to iML1515
# ---------------------------------------------------------------------------

def test_compare_to_iml1515(iml1515, carved):
    """Benchmark carved model against the curated iML1515."""
    carved_ids = {r.id for r in carved.reactions}
    ref_ids = {r.id for r in iml1515.reactions}

    in_both = carved_ids & ref_ids
    gemiz_only = carved_ids - ref_ids
    iml_only = ref_ids - carved_ids

    precision = len(in_both) / len(carved_ids) if carved_ids else 0
    recall = len(in_both) / len(ref_ids) if ref_ids else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0)

    print(f"\n  Reactions in both:    {len(in_both)}")
    print(f"  In gemiz only:        {len(gemiz_only)} (false positives)")
    print(f"  In iML1515 only:      {len(iml_only)} (false negatives)")
    print(f"  Precision:            {precision:.3f}")
    print(f"  Recall:               {recall:.3f}")
    print(f"  F1:                   {f1:.3f}")

    # Save benchmark
    benchmark = {
        "reactions_in_both": len(in_both),
        "gemiz_only": len(gemiz_only),
        "iml1515_only": len(iml_only),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }
    out = Path("data/test_outputs")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ecoli_benchmark.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"  Saved to {out / 'ecoli_benchmark.json'}")

    assert f1 > 0.5, f"F1 too low: {f1:.3f}"


# ---------------------------------------------------------------------------
# Test 6 -- save and reload model
# ---------------------------------------------------------------------------

def test_save_model(carved):
    """Save carved model to SBML and verify it reloads correctly."""
    import cobra

    out = Path("data/test_outputs")
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "ecoli_model.xml"

    cobra.io.write_sbml_model(carved, str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved to: {model_path}")
    print(f"  File size: {size_mb:.1f} MB")

    # Reload and verify
    reloaded = cobra.io.read_sbml_model(str(model_path))
    print(f"  Reloaded reactions:   {len(reloaded.reactions)}")
    print(f"  Reloaded metabolites: {len(reloaded.metabolites)}")
    print(f"  Reloaded genes:       {len(reloaded.genes)}")

    assert len(reloaded.reactions) == len(carved.reactions)
    assert len(reloaded.metabolites) == len(carved.metabolites)
    assert len(reloaded.genes) == len(carved.genes)

    # Verify it can still grow after save/load round-trip
    sol = reloaded.optimize()
    assert sol.status == "optimal"
    print(f"  Growth after reload:  {sol.objective_value:.4f} h^-1")
