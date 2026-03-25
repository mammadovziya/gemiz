"""Step 1 — Gene calling via pyrodigal (pure Python).

Install
-------
    pip install gemiz[full]   # includes pyrodigal
    pip install pyrodigal     # standalone
"""

from __future__ import annotations

from pathlib import Path


def call_genes(fna_path: str, output_dir: str) -> str:
    """Call protein-coding genes in a genome FASTA.

    Uses pyrodigal — a pure-Python/Cython port of Prodigal.
    No external binaries required.

    Parameters
    ----------
    fna_path:
        Path to input genome (.fna / .fa / .fasta).
    output_dir:
        Directory where the output .faa will be written.

    Returns
    -------
    str
        Absolute path to the output protein FASTA (.faa).

    Raises
    ------
    ImportError
        If pyrodigal is not installed.
    """
    try:
        import pyrodigal
    except ImportError:
        raise ImportError(
            "\n[gemiz] pyrodigal is not installed.\n"
            "\n"
            "    pip install gemiz[full]\n"
            "    # or: pip install pyrodigal\n"
        )

    from Bio import SeqIO

    fna = Path(fna_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    faa_path = out_dir / f"{fna.stem}.faa"

    records = list(SeqIO.parse(fna, "fasta"))
    is_complete = len(records) == 1
    genome_type = "complete" if is_complete else "draft"

    print(f"[gemiz] Calling genes in {fna.name} ({genome_type} genome) using pyrodigal...")

    orf_finder = pyrodigal.GeneFinder(meta=not is_complete)
    if is_complete:
        orf_finder.train(*(str(r.seq) for r in records))

    protein_count = 0
    with faa_path.open("w") as fh:
        for record in records:
            genes = orf_finder.find_genes(str(record.seq))
            genes.write_translations(fh, sequence_id=record.id)
            protein_count += len(genes)

    print(f"[gemiz] Found {protein_count:,} proteins")
    return str(faa_path)
