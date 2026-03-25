"""Tests for the MMseqs2 alignment pipeline.

MMseqs2 is not available on native Windows (requires Cygwin).
All alignment tests are automatically skipped on Windows with a
message pointing to WSL2.

On Linux and macOS these tests run fully.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import pytest

# Skip the whole module on native Windows
if platform.system() == "Windows":
    pytest.skip(
        "MMseqs2 requires WSL2 on Windows.\n"
        "Run: wsl --install  then run gemiz inside WSL.",
        allow_module_level=True,
    )

from gemiz.utils.binaries import check_cpu_features, verify_mmseqs
from gemiz.pipeline.alignment import (
    align_proteins,
    build_mmseqs_db,
    classify_proteins,
    parse_alignment,
)
from gemiz.pipeline.prodigal import call_genes

GENOME    = Path("data/genomes/ecoli_k12.fna")
REFERENCE = Path("data/reference/iML1515_proteins.faa")


# ---------------------------------------------------------------------------
# Test 1 — verify_mmseqs() + CPU features
# ---------------------------------------------------------------------------

def test_verify_mmseqs():
    """Bundled MMseqs2 binary must run and report a version."""
    info = verify_mmseqs()
    cpu  = check_cpu_features()

    avx2_str  = "supported" if cpu.get("avx2")  else "not supported"
    sse41_str = "supported" if cpu.get("sse41") else "not supported"

    print(f"\n  MMseqs2 version : {info.get('version', '?')}")
    print(f"  Binary path     : {info.get('path', '?')}")
    print(f"  CPU AVX2        : {avx2_str}")
    print(f"  CPU SSE4.1      : {sse41_str}")

    assert info["ok"], f"MMseqs2 verification failed: {info.get('error')}"
    assert info["version"], "Empty version string"


# ---------------------------------------------------------------------------
# Test 2 — build_mmseqs_db()
# ---------------------------------------------------------------------------

def test_build_mmseqs_db(tmp_path):
    """build_mmseqs_db() must create MMseqs2 database files."""
    assert REFERENCE.exists(), f"Reference not found: {REFERENCE}"

    db_prefix = build_mmseqs_db(str(REFERENCE), str(tmp_path / "refdb"))

    db_files = list(Path(tmp_path / "refdb").glob("db*"))
    print(f"\n  DB prefix : {db_prefix}")
    print(f"  DB files  : {[f.name for f in sorted(db_files)]}")

    assert len(db_files) >= 2, "Expected at least 2 database files"


# ---------------------------------------------------------------------------
# Test 3 — align_proteins()
# ---------------------------------------------------------------------------

def test_align_proteins(tmp_path):
    """align_proteins() must produce a non-empty TSV result file."""
    assert GENOME.exists(),    f"Genome not found: {GENOME}"
    assert REFERENCE.exists(), f"Reference not found: {REFERENCE}"

    t0 = time.perf_counter()

    faa = call_genes(str(GENOME), str(tmp_path / "genes"))
    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5,
        threads=4,
    )

    elapsed = time.perf_counter() - t0

    lines = [l for l in Path(tsv).read_text().splitlines() if l.strip()]
    print(f"\n  Result TSV   : {tsv}")
    print(f"  Total hits   : {len(lines):,}")
    print(f"  Time taken   : {elapsed:.1f}s")

    assert Path(tsv).exists()
    assert len(lines) > 0, "Alignment produced no hits"


# ---------------------------------------------------------------------------
# Test 4 — parse_alignment()
# ---------------------------------------------------------------------------

def test_parse_alignment(tmp_path):
    """parse_alignment() must return a correctly structured dict."""
    faa = call_genes(str(GENOME), str(tmp_path / "genes"))
    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5,
        threads=4,
    )

    hits = parse_alignment(tsv)
    first_id  = next(iter(hits))
    top_hits  = hits[first_id]

    print(f"\n  Proteins with hits : {len(hits):,}")
    print(f"\n  Top 3 hits for {first_id!r}:")
    for h in top_hits[:3]:
        print(
            f"    ref={h['ref_id']!r:40s}  "
            f"id={h['identity']:5.1f}%  "
            f"e={h['evalue']:.2e}  "
            f"bits={h['bitscore']:.0f}  "
            f"cov={h['coverage']:.1f}%"
        )

    assert isinstance(hits, dict)
    assert isinstance(top_hits, list)
    required_keys = {"ref_id", "identity", "evalue", "bitscore", "coverage"}
    assert required_keys <= top_hits[0].keys()
    if len(top_hits) > 1:
        assert top_hits[0]["bitscore"] >= top_hits[1]["bitscore"]


# ---------------------------------------------------------------------------
# Test 5 — classify_proteins() — full summary
# ---------------------------------------------------------------------------

def test_classify_proteins(tmp_path):
    """Full pipeline: gene calling -> alignment -> classification."""
    t0 = time.perf_counter()

    faa = call_genes(str(GENOME), str(tmp_path / "genes"))

    all_proteins = [
        line[1:].split()[0]
        for line in Path(faa).read_text().splitlines()
        if line.startswith(">")
    ]

    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5,
        threads=4,
    )
    alignment = parse_alignment(tsv)
    result    = classify_proteins(all_proteins, alignment)
    stats     = result["stats"]
    elapsed   = time.perf_counter() - t0

    mmseqs_version = verify_mmseqs().get("version", "?")
    cpu = check_cpu_features()

    print(f"\n  MMseqs2 version:          {mmseqs_version}")
    print(f"  CPU AVX2:                 {'supported' if cpu.get('avx2') else 'not supported'}")
    print(f"  Total proteins:           {stats['total']:,}")
    print(f"  High confidence hits:     {stats['high_confidence']:,} ({stats['high_confidence_pct']}%)")
    print(f"  Low confidence -> ESM C:  {stats['low_confidence']:,} ({stats['low_confidence_pct']}%)")
    print(f"  Time taken:               {elapsed:.1f}s")

    assert stats["total"] == len(all_proteins)
    assert stats["high_confidence"] + stats["low_confidence"] == stats["total"]
    assert stats["high_confidence"] > 0
    assert 0.0 <= stats["high_confidence_pct"] <= 100.0
