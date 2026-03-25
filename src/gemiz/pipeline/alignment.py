"""Step 2 — Protein alignment via bundled MMseqs2.

No external installation needed. MMseqs2 is shipped inside the package.
On Linux the correct binary (AVX2 or SSE4.1) is selected automatically.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from gemiz.utils.binaries import get_mmseqs_path

# Output columns requested from MMseqs2
_FORMAT = "query,target,pident,alnlen,evalue,bits,qcov"
_COLS   = ["query", "target", "pident", "alnlen", "evalue", "bits", "qcov"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_proteins(
    query_faa: str,
    reference_db: str,
    output_dir: str,
    threads: int = 4,
    sensitivity: float = 7.5,
    max_hits: int = 5,
    min_identity: float = 30.0,
) -> str:
    """Align proteins in *query_faa* against *reference_db* using MMseqs2.

    Parameters
    ----------
    query_faa:
        Path to the protein FASTA produced by ``call_genes()``.
    reference_db:
        Path to an MMseqs2 sequence database (built by ``build_mmseqs_db()``)
        **or** a plain ``.faa`` file (MMseqs2 easy-search handles both).
    output_dir:
        Directory where ``results.tsv`` will be written.
    threads:
        CPU threads for MMseqs2.
    sensitivity:
        MMseqs2 sensitivity preset.
        4.0 = fast, 7.5 = balanced (default), 9.5 = most sensitive.
    max_hits:
        Maximum hits per query sequence.
    min_identity:
        Minimum % sequence identity to report.

    Returns
    -------
    str
        Absolute path to the alignment TSV.
    """
    mmseqs = get_mmseqs_path()

    query   = Path(query_faa).resolve()
    ref     = Path(reference_db).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_tsv = out_dir / "results.tsv"

    # Count query proteins for the progress line
    n_query = sum(1 for l in query.open() if l.startswith(">"))
    print(f"[gemiz] Aligning {n_query:,} proteins (MMseqs2, sensitivity={sensitivity})...")

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            str(mmseqs), "easy-search",
            str(query),
            str(ref),
            str(result_tsv),
            tmp,                             # MMseqs2 scratch dir
            "--format-output", _FORMAT,
            "-s",              str(sensitivity),
            "--max-seqs",      str(max_hits),
            "--min-seq-id",    str(min_identity / 100),
            "--threads",       str(threads),
            "-v",              "0",          # suppress MMseqs2 progress bars
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"[gemiz] MMseqs2 failed (exit {result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )

    # Count unique queried proteins that got at least one hit
    hits_tsv  = result_tsv.read_text().splitlines()
    hit_prots = {line.split("\t")[0] for line in hits_tsv if line.strip()}
    n_hits    = len(hit_prots)

    print(
        f"[gemiz] Found hits for {n_hits:,}/{n_query:,} proteins "
        f"({100*n_hits/n_query:.1f}%)  [{elapsed:.1f}s]"
    )
    return str(result_tsv)


def build_mmseqs_db(faa_path: str, db_dir: str) -> str:
    """Build an MMseqs2 sequence database from a protein FASTA.

    Parameters
    ----------
    faa_path:
        Input protein FASTA (.faa).
    db_dir:
        Output directory for the MMseqs2 database files.

    Returns
    -------
    str
        Path to the database prefix (pass as *reference_db* to
        ``align_proteins()``).
    """
    mmseqs = get_mmseqs_path()

    db_path = Path(db_dir).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    db_prefix = str(db_path / "db")

    print(f"[gemiz] Building MMseqs2 database from {Path(faa_path).name}...")
    cmd = [str(mmseqs), "createdb", str(Path(faa_path).resolve()), db_prefix, "-v", "0"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[gemiz] MMseqs2 createdb failed:\n{result.stderr[-1000:]}"
        )

    db_files = list(Path(db_dir).glob("db*"))
    print(f"[gemiz] Database ready ({len(db_files)} files at {db_prefix})")
    return db_prefix


def parse_alignment(tsv_path: str) -> dict[str, list[dict]]:
    """Parse MMseqs2 easy-search TSV output into a structured dict.

    Parameters
    ----------
    tsv_path:
        Path to the TSV produced by ``align_proteins()``.

    Returns
    -------
    dict
        ``{protein_id: [{"ref_id": ..., "identity": ..., "evalue": ...,
                          "bitscore": ..., "coverage": ...}, ...]}``

        Hits within each protein are sorted by bitscore descending.
    """
    import pandas as pd

    path = Path(tsv_path)
    if not path.exists() or path.stat().st_size == 0:
        return {}

    df = pd.read_csv(path, sep="\t", header=None, names=_COLS)

    results: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        hit = {
            "ref_id":   row["target"],
            "identity": float(row["pident"]),
            "evalue":   float(row["evalue"]),
            "bitscore": float(row["bits"]),
            "coverage": round(float(row["qcov"]) * 100, 1),  # fraction -> %
        }
        results.setdefault(row["query"], []).append(hit)

    # Sort each protein's hits by bitscore descending
    for hits in results.values():
        hits.sort(key=lambda h: h["bitscore"], reverse=True)

    return results


def classify_proteins(
    all_proteins: list[str],
    alignment: dict[str, list[dict]],
    high_confidence_threshold: float = 50.0,
) -> dict:
    """Split proteins into high-confidence and low-confidence bins.

    High confidence
        Best hit identity >= *high_confidence_threshold* → use alignment score.
    Low confidence
        No hit, or best hit identity < threshold → send to ESM C embeddings.

    Parameters
    ----------
    all_proteins:
        List of all protein IDs from the .faa (order preserved).
    alignment:
        Output of ``parse_alignment()``.
    high_confidence_threshold:
        Minimum % identity for a high-confidence call (default 50%).

    Returns
    -------
    dict ::

        {
          "high_confidence": {"prot_001": [hits...], ...},
          "low_confidence":  ["prot_045", ...],
          "stats": {
            "total":                4319,
            "high_confidence":      2847,
            "low_confidence":       1472,
            "high_confidence_pct":  66.0,
            "low_confidence_pct":   34.0,
          }
        }
    """
    high: dict[str, list[dict]] = {}
    low:  list[str] = []

    for prot_id in all_proteins:
        hits = alignment.get(prot_id, [])
        if hits and hits[0]["identity"] >= high_confidence_threshold:
            high[prot_id] = hits
        else:
            low.append(prot_id)

    total = len(all_proteins)
    stats = {
        "total":               total,
        "high_confidence":     len(high),
        "low_confidence":      len(low),
        "high_confidence_pct": round(100 * len(high) / total, 1) if total else 0.0,
        "low_confidence_pct":  round(100 * len(low)  / total, 1) if total else 0.0,
    }

    return {"high_confidence": high, "low_confidence": low, "stats": stats}
