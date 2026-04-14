"""NCBI genome downloading utilities."""

from __future__ import annotations

import gzip
import re
import shutil
from pathlib import Path

import requests

_NCBI_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def _assembly_ftp_dir(accession: str) -> str:
    """Convert GCF_000009045.1 -> FTP directory URL."""
    prefix = accession[:3]
    digits = accession[4:].split(".")[0]
    p1, p2, p3 = digits[0:3], digits[3:6], digits[6:9]
    return f"{_NCBI_FTP}/{prefix}/{p1}/{p2}/{p3}"


def _find_assembly_dir(accession: str) -> str:
    """Resolve the full FTP path including the assembly directory name."""
    parent = _assembly_ftp_dir(accession)
    r = requests.get(parent, timeout=30)
    r.raise_for_status()

    pattern = re.compile(rf'href="({re.escape(accession)}[^"]*)"')
    matches = pattern.findall(r.text)
    if not matches:
        raise RuntimeError(
            f"Could not find assembly directory for {accession} at {parent}\n"
            "Check that the accession is correct (e.g. GCF_000005845.2)."
        )
    asm_dir = matches[0].rstrip("/")
    return f"{parent}/{asm_dir}"


def download_assembly(accession: str, dest: Path) -> Path:
    """Download the genomic FASTA from NCBI.

    Args:
        accession: NCBI RefSeq accession (e.g., 'GCF_000005845.2').
        dest: Directory to save the uncompressed FASTA file.

    Returns:
        Path to the downloaded and decompressed .fna file.
    """
    asm_url = _find_assembly_dir(accession)
    asm_name = asm_url.rsplit("/", 1)[-1]
    
    fna_gz_name = f"{asm_name}_genomic.fna.gz"
    url = f"{asm_url}/{fna_gz_name}"
    
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{accession}_genomic.fna"
    gz_path = dest / fna_gz_name
    
    # Download
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    
    with open(gz_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)

    # Decompress
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    
    return out_path
