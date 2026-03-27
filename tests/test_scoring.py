"""Tests for Step 4 — Reaction Scoring.

Tests 1-3 are pure unit tests (no data downloads, run anywhere).
Tests 4-5 need the iML1515 model + MMseqs2 alignment output (Linux/WSL2 only).
Test 6 (tuner) is marked slow and runs only with ``pytest -m slow``.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pytest

from gemiz.reconstruction.scoring import (
    HIGH_CONF_THRESHOLD,
    LOW_CONF_THRESHOLD,
    NO_EVIDENCE_SCORE,
    build_protein_score_map,
    compute_reaction_scores,
    diagnose_id_mapping,
    evaluate_gpr_rule,
    extract_gpr_associations,
    load_universal_model,
    merge_protein_scores,
    parse_reference_id_map,
)

MODEL_PATH     = Path("data/universal/iML1515.xml")
GENOME         = Path("data/genomes/ecoli_k12.fna")
REFERENCE      = Path("data/reference/iML1515_proteins.faa")
FEATURE_TABLE  = Path("data/reference/ecoli_feature_table.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — merge_protein_scores (pure unit test)
# ──────────────────────────────────────────────────────────────────────────────

def test_merge_protein_scores():
    """Verify the adaptive weighting formula across all regimes."""
    # Above high threshold -> MMseqs2 only (identity/100)
    s = merge_protein_scores(98.5, 0.94)
    print(f"\n  identity=98.5%, esmc=0.94 -> {s:.4f}  (expected 0.985)")
    assert abs(s - 0.985) < 1e-6

    # Exactly at high threshold
    s = merge_protein_scores(50.0, 0.80)
    print(f"  identity=50.0%, esmc=0.80 -> {s:.4f}  (expected 0.500)")
    assert abs(s - 0.500) < 1e-6

    # Twilight zone (40% identity) — blended
    s = merge_protein_scores(40.0, 0.90)
    # t = (40-30)/(50-30) = 0.5  -> w_mmseqs=0.7  w_esmc=0.3
    # 0.7*0.4 + 0.3*0.9 = 0.28 + 0.27 = 0.55
    print(f"  identity=40.0%, esmc=0.90 -> {s:.4f}  (expected 0.550)")
    assert abs(s - 0.55) < 1e-3

    # Below low threshold (20% identity)
    s = merge_protein_scores(20.0, 0.85)
    expected = 0.4 * 0.2 + 0.6 * 0.85  # 0.08 + 0.51 = 0.59
    print(f"  identity=20.0%, esmc=0.85 -> {s:.4f}  (expected {expected:.3f})")
    assert abs(s - expected) < 1e-3

    # No MMseqs2 hit at all
    s = merge_protein_scores(0.0, 0.94)
    print(f"  identity= 0.0%, esmc=0.94 -> {s:.4f}  (expected 0.940)")
    assert abs(s - 0.94) < 1e-6

    # Both zero
    s = merge_protein_scores(0.0, 0.0)
    print(f"  identity= 0.0%, esmc=0.00 -> {s:.4f}  (expected 0.000)")
    assert s == 0.0

    print("  All merge_protein_scores assertions pass")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — extract_gpr_associations from iML1515
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def iml1515():
    """Load iML1515 once per module."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return load_universal_model(str(MODEL_PATH))


def test_extract_gpr_associations(iml1515):
    """Extract GPR rules; verify counts by type."""
    assoc = extract_gpr_associations(iml1515)

    types = {}
    for a in assoc.values():
        types[a["type"]] = types.get(a["type"], 0) + 1

    print(f"\n  Total reactions: {len(assoc)}")
    for t in ("single", "isozyme", "complex", "mixed", "no_gpr"):
        print(f"  {t:10s}: {types.get(t, 0):>5}")

    # Show examples
    print(f"\n  Examples:")
    for rxn_id, a in list(assoc.items())[:5]:
        print(f"    {rxn_id}: [{a['type']}] {a['rule'][:60]}")

    assert len(assoc) == len(iml1515.reactions)
    assert types.get("no_gpr", 0) > 0, "Expected some no-GPR reactions"
    assert sum(types.get(t, 0) for t in ("single", "isozyme", "complex", "mixed")) > 1000


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — evaluate_gpr_rule (pure unit test)
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluate_gpr_rule():
    """GPR evaluation: OR->max, AND->min, mixed, missing, empty."""
    scores = {
        "b3916": 0.95,
        "b1723": 0.12,
        "b3731": 0.88,
        "b3732": 0.43,
        "b3733": 0.91,
    }

    # OR logic -> max
    s = evaluate_gpr_rule("b3916 or b1723", scores, "isozyme")
    print(f"\n  OR:  b3916 or b1723           = {s:.2f}  (expected 0.95)")
    assert abs(s - 0.95) < 1e-6

    # AND logic -> min
    s = evaluate_gpr_rule("b3731 and b3732 and b3733", scores, "complex")
    print(f"  AND: b3731 and b3732 and b3733 = {s:.2f}  (expected 0.43)")
    assert abs(s - 0.43) < 1e-6

    # Mixed: (b3731 AND b3732) OR b3916
    s = evaluate_gpr_rule("(b3731 and b3732) or b3916", scores, "mixed")
    # min(0.88,0.43)=0.43 then max(0.43,0.95)=0.95
    print(f"  MIX: (b3731 and b3732) or b3916 = {s:.2f}  (expected 0.95)")
    assert abs(s - 0.95) < 1e-6

    # Missing gene -> -1.0
    s = evaluate_gpr_rule("b3916 and b9999", scores, "complex")
    print(f"  MISS: b3916 and b9999         = {s:.2f}  (expected -1.0)")
    assert abs(s - (-1.0)) < 1e-6

    # Empty GPR -> 0.0 (spontaneous)
    s = evaluate_gpr_rule("", scores, "no_gpr")
    print(f"  EMPTY: (spontaneous)          = {s:.2f}  (expected  0.0)")
    assert s == 0.0

    print("  All GPR evaluation assertions pass")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — build_protein_score_map with real alignment data
# ──────────────────────────────────────────────────────────────────────────────

_NEEDS_LINUX = platform.system() == "Windows"


@pytest.mark.skipif(_NEEDS_LINUX, reason="MMseqs2 requires WSL2 on Windows")
def test_build_protein_score_map(iml1515, tmp_path):
    """Build protein score map from real MMseqs2 alignment."""
    from gemiz.pipeline.alignment import align_proteins, parse_alignment
    from gemiz.pipeline.prodigal import call_genes

    assert GENOME.exists()
    assert REFERENCE.exists()

    faa = call_genes(str(GENOME), str(tmp_path / "genes"))
    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5, threads=4,
    )
    mmseqs_hits = parse_alignment(tsv)

    assoc = extract_gpr_associations(iml1515)

    # Run ID mapping diagnostic before scoring
    ft = str(FEATURE_TABLE) if FEATURE_TABLE.exists() else None
    id_map = parse_reference_id_map(feature_table_path=ft)
    diagnose_id_mapping(mmseqs_hits, assoc, id_map)

    scores = build_protein_score_map(
        mmseqs_hits, {}, assoc,
        feature_table_path=ft,
    )

    # Check score range
    vals = list(scores.values())
    print(f"\n  Scored genes     : {len(scores)}")
    print(f"  Score range      : [{min(vals):.3f}, {max(vals):.3f}]")
    print(f"  Mean score       : {sum(vals)/len(vals):.3f}")

    assert all(0 <= v <= 1.0 for v in vals)
    assert len(scores) > 100


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — compute_reaction_scores full integration
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(_NEEDS_LINUX, reason="MMseqs2 requires WSL2 on Windows")
def test_compute_reaction_scores(iml1515, tmp_path):
    """Full pipeline: gene calling -> alignment -> reaction scoring."""
    from gemiz.pipeline.alignment import align_proteins, parse_alignment
    from gemiz.pipeline.prodigal import call_genes

    assert GENOME.exists()
    assert REFERENCE.exists()

    t0 = time.perf_counter()

    faa = call_genes(str(GENOME), str(tmp_path / "genes"))
    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5, threads=4,
    )
    mmseqs_hits = parse_alignment(tsv)

    ft = str(FEATURE_TABLE) if FEATURE_TABLE.exists() else None
    scores = compute_reaction_scores(
        iml1515, mmseqs_hits, {},
        high_conf=HIGH_CONF_THRESHOLD, low_conf=LOW_CONF_THRESHOLD,
        feature_table_path=ft,
    )

    elapsed = time.perf_counter() - t0

    # Top 10 highest
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 highest scoring reactions:")
    for rxn_id, s in sorted_scores[:10]:
        rxn = iml1515.reactions.get_by_id(rxn_id)
        print(f"    {rxn_id:20s}  {s:+.4f}  ({rxn.name[:40]})")

    # 5 penalised
    penalised = [(r, s) for r, s in sorted_scores if s < 0]
    print(f"\n  5 penalised reactions:")
    for rxn_id, s in penalised[:5]:
        rxn = iml1515.reactions.get_by_id(rxn_id)
        print(f"    {rxn_id:20s}  {s:+.4f}  ({rxn.name[:40]})")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    out = Path("data/test_outputs")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ecoli_reaction_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  Saved to {out / 'ecoli_reaction_scores.json'}")

    assert len(scores) == len(iml1515.reactions)
    assert any(s > 0.7 for s in scores.values())
    assert any(s < 0 for s in scores.values())


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — scoring tuner (slow, run explicitly with pytest -m slow)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(_NEEDS_LINUX, reason="MMseqs2 requires WSL2 on Windows")
def test_scoring_tuner(iml1515, tmp_path):
    """Grid search over thresholds using iML1515 as gold standard."""
    from gemiz.pipeline.alignment import align_proteins, parse_alignment
    from gemiz.pipeline.prodigal import call_genes
    from gemiz.reconstruction.scoring_tuner import tune_thresholds

    faa = call_genes(str(GENOME), str(tmp_path / "genes"))
    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5, threads=4,
    )
    mmseqs_hits = parse_alignment(tsv)

    ft = str(FEATURE_TABLE) if FEATURE_TABLE.exists() else None
    result = tune_thresholds(
        mmseqs_hits=mmseqs_hits,
        esmc_hits={},
        universal_model=iml1515,
        reference_model=iml1515,
        feature_table_path=ft,
    )

    out = Path("data/test_outputs")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_thresholds.json", "w") as f:
        json.dump(result["best"], f, indent=2)
    print(f"\n  Saved to {out / 'best_thresholds.json'}")

    assert result["best"]["f1"] > 0.5, f"Best F1 too low: {result['best']['f1']}"
