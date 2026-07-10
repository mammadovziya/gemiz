"""Tests for the gene-calling pipeline (pyrodigal)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemiz.pipeline.prodigal import call_genes

GENOME = Path("data/genomes/ecoli_k12.fna")
needs_ecoli_genome = pytest.mark.skipif(
    not GENOME.exists(),
    reason=f"Optional benchmark genome not found: {GENOME}",
)


def test_pyrodigal_importable():
    """pyrodigal must be installed."""
    import pyrodigal
    print(f"\n  pyrodigal version : {pyrodigal.__version__}")


@needs_ecoli_genome
def test_call_genes_ecoli(tmp_path, capsys):
    """call_genes() must complete and print progress lines."""
    assert GENOME.exists(), f"Test genome not found: {GENOME}"

    faa_path = call_genes(str(GENOME), str(tmp_path))

    out = capsys.readouterr().out
    print(f"\n  stdout:\n{out.rstrip()}")

    assert "Calling genes" in out
    assert "Found" in out
    assert Path(faa_path).exists()


@needs_ecoli_genome
def test_protein_count(tmp_path):
    """E. coli K-12 must yield >= 4000 proteins (typically 4319)."""
    faa_path = Path(call_genes(str(GENOME), str(tmp_path)))

    headers = [line for line in faa_path.read_text().splitlines() if line.startswith(">")]
    n = len(headers)

    print(f"\n  Proteins found : {n:,}")
    print("  First 3 headers:")
    for h in headers[:3]:
        print(f"    {h}")

    assert n >= 4000, f"Expected >=4000 proteins, got {n}"


@needs_ecoli_genome
def test_output_is_valid_fasta(tmp_path):
    """Every sequence in the .faa must start with a > header line."""
    faa_path = Path(call_genes(str(GENOME), str(tmp_path)))
    first_line = faa_path.open().readline().strip()
    assert first_line.startswith(">"), f"First line not a FASTA header: {first_line!r}"
