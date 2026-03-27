"""Tests for Step 6 -- Full reconstruction pipeline.

The integration test requires:
  - E. coli genome (data/genomes/ecoli_k12.fna)
  - iML1515 model (data/universal/iML1515.xml)
  - Reference proteins (data/reference/iML1515_proteins.faa)
  - Feature table (data/reference/ecoli_feature_table.txt)
  - MMseqs2 binary (Linux/WSL2 only)

Skipped on Windows (MMseqs2 requires WSL2).
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

GENOME        = Path("data/genomes/ecoli_k12.fna")
MODEL_PATH    = Path("data/universal/iML1515.xml")
REFERENCE     = Path("data/reference/iML1515_proteins.faa")
FEATURE_TABLE = Path("data/reference/ecoli_feature_table.txt")
OUTPUT        = Path("data/test_outputs/ecoli_pipeline_model.xml")

_NEEDS_LINUX = platform.system() == "Windows"


@pytest.mark.skipif(_NEEDS_LINUX, reason="MMseqs2 requires WSL2 on Windows")
def test_full_pipeline():
    """Full end-to-end: .fna -> .xml in one function call."""
    import cobra

    from gemiz.reconstruction.pipeline import run_full_pipeline

    assert GENOME.exists(), f"Genome not found: {GENOME}"
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    assert REFERENCE.exists(), f"Reference not found: {REFERENCE}"

    result = run_full_pipeline(
        genome_fna=str(GENOME),
        output_xml=str(OUTPUT),
        universal_model_path=str(MODEL_PATH),
        reference_faa_path=str(REFERENCE),
        feature_table_path=str(FEATURE_TABLE) if FEATURE_TABLE.exists() else None,
        high_conf=50.0,
        low_conf=30.0,
        min_growth=0.1,
        use_esm=False,
        threads=4,
    )

    # ---- verify output file exists ----
    assert OUTPUT.exists(), f"Output not created: {OUTPUT}"

    # ---- verify model loads ----
    model = cobra.io.read_sbml_model(str(OUTPUT))

    # ---- verify model is functional ----
    assert len(model.reactions) > 500, f"Too few reactions: {len(model.reactions)}"
    sol = model.optimize()
    assert sol.status == "optimal", f"FBA failed: {sol.status}"
    assert sol.objective_value > 0.01, f"Growth too low: {sol.objective_value}"

    # ---- print full summary ----
    print(f"\n  Full pipeline complete")
    print(f"  Reactions:   {len(model.reactions)}")
    print(f"  Metabolites: {len(model.metabolites)}")
    print(f"  Genes:       {len(model.genes)}")
    print(f"  Growth:      {sol.objective_value:.4f} h^-1")
    print(f"  Time:        {result['total_time']:.1f}s")
    print(f"  Output:      {OUTPUT}")

    # ---- timing breakdown ----
    print(f"\n  Timing breakdown:")
    for i in range(1, 7):
        key = f"step{i}_time"
        if key in result:
            print(f"    Step {i}: {result[key]:.1f}s")

    # ---- save pipeline results ----
    out_dir = Path("data/test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_proteins": result.get("n_proteins"),
        "n_high_conf": result.get("n_high_conf"),
        "n_low_conf": result.get("n_low_conf"),
        "n_scored_positive": result.get("n_scored_positive"),
        "n_scored_negative": result.get("n_scored_negative"),
        "n_reactions": result["n_reactions"],
        "n_metabolites": result["n_metabolites"],
        "n_genes": result["n_genes"],
        "growth_rate": result["growth_rate"],
        "total_time": round(result["total_time"], 1),
    }
    with open(out_dir / "ecoli_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to {out_dir / 'ecoli_pipeline_summary.json'}")


def test_cli_carve_help():
    """Verify the carve CLI command is registered and shows help."""
    from click.testing import CliRunner

    from gemiz.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["carve", "--help"])
    assert result.exit_code == 0
    assert "Reconstruct a GEM" in result.output
    assert "--output" in result.output
    assert "--template" in result.output
    assert "--no-esm" in result.output
    print(f"\n  CLI help output:\n{result.output}")
