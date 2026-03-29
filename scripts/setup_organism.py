#!/usr/bin/env python3
"""Download and set up reference data for a target organism.

Usage
-----
    python scripts/setup_organism.py ecoli \\
        --ncbi-assembly GCF_000005845.2 \\
        --bigg-model iML1515

    python scripts/setup_organism.py bsubtilis \\
        --ncbi-assembly GCF_000009045.1 \\
        --bigg-model iYO844

    python scripts/setup_organism.py paeruginosa \\
        --ncbi-assembly GCF_000006765.1 \\
        --gold-standard path/to/iMO1056.xml \\
        --skip-bigg

Output
------
    data/organisms/<name>/
        proteins.faa            reference proteome from NCBI
        feature_table.txt       NCBI feature table (accession -> locus_tag)
        gold_standard.xml       BiGG model for benchmarking
        esm_db/                 pre-built FAISS index (optional, needs GPU)
            reference.faiss
            reference_ids.json
            reference_embeddings.npz
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# NCBI FTP helpers
# ---------------------------------------------------------------------------

_NCBI_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def _assembly_ftp_dir(accession: str) -> str:
    """Convert GCF_000009045.1 -> FTP directory URL."""
    prefix = accession[:3]          # GCF
    digits = accession[4:].split(".")[0]  # 000009045
    p1, p2, p3 = digits[0:3], digits[3:6], digits[6:9]
    return f"{_NCBI_FTP}/{prefix}/{p1}/{p2}/{p3}"


def _find_assembly_dir(accession: str) -> str:
    """Resolve the full FTP path including the assembly directory name.

    NCBI FTP stores assemblies under a directory whose name is
    ``<accession>_<asm_name>``.  We list the parent to find it.
    """
    parent = _assembly_ftp_dir(accession)
    print(f"  Resolving assembly directory for {accession} ...")
    r = requests.get(parent, timeout=30)
    r.raise_for_status()

    # The FTP-over-HTTPS listing is an HTML page with links.
    # Look for a link whose text starts with the accession.
    import re
    pattern = re.compile(
        rf'href="({re.escape(accession)}[^"]*)"'
    )
    matches = pattern.findall(r.text)
    if not matches:
        raise RuntimeError(
            f"Could not find assembly directory for {accession} at {parent}\n"
            "Check that the accession is correct (e.g. GCF_000009045.1)."
        )
    asm_dir = matches[0].rstrip("/")
    return f"{parent}/{asm_dir}"


def download_proteins(accession: str, dest: Path) -> Path:
    """Download the annotated protein FASTA from NCBI."""
    asm_url = _find_assembly_dir(accession)
    asm_name = asm_url.rsplit("/", 1)[-1]

    faa_gz_name = f"{asm_name}_protein.faa.gz"
    url = f"{asm_url}/{faa_gz_name}"

    out_path = dest / "proteins.faa"
    gz_path = dest / faa_gz_name

    print(f"  Downloading {faa_gz_name} ...")
    _download_file(url, gz_path)

    print(f"  Decompressing -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    n = sum(1 for line in open(out_path) if line.startswith(">"))
    print(f"  {n:,} proteins")
    return out_path


def download_feature_table(accession: str, dest: Path) -> Path:
    """Download the NCBI feature table."""
    asm_url = _find_assembly_dir(accession)
    asm_name = asm_url.rsplit("/", 1)[-1]

    ft_gz_name = f"{asm_name}_feature_table.txt.gz"
    url = f"{asm_url}/{ft_gz_name}"

    out_path = dest / "feature_table.txt"
    gz_path = dest / ft_gz_name

    print(f"  Downloading {ft_gz_name} ...")
    _download_file(url, gz_path)

    print(f"  Decompressing -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    # count CDS rows
    n_cds = sum(
        1 for line in open(out_path, encoding="utf-8")
        if line.startswith("CDS\t")
    )
    print(f"  {n_cds:,} CDS entries")
    return out_path


# ---------------------------------------------------------------------------
# BiGG model download
# ---------------------------------------------------------------------------

def download_bigg_model(model_id: str, dest: Path) -> Path:
    """Download a COBRA model from the BiGG database.

    Tries multiple URLs in order (BiGG has changed endpoints over time):
      1. https://bigg.ucsd.edu/static/models/{model_id}.xml.gz
      2. https://bigg.ucsd.edu/static/models/{model_id}.json
      3. http://bigg.ucsd.edu/api/v2/models/{model_id}/download
    """
    out_path = dest / "gold_standard.xml"

    urls = [
        (f"https://bigg.ucsd.edu/static/models/{model_id}.xml.gz", "xml.gz"),
        (f"https://bigg.ucsd.edu/static/models/{model_id}.json", "json"),
        (f"http://bigg.ucsd.edu/api/v2/models/{model_id}/download", "xml.gz"),
    ]

    print(f"  Downloading {model_id} from BiGG ...")

    for url, fmt in urls:
        print(f"    Trying {url} ...")
        try:
            tmp_path = dest / f"{model_id}_tmp.{fmt}"
            _download_file(url, tmp_path)
        except (FileNotFoundError, requests.HTTPError) as e:
            print(f"    -> {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            continue

        if fmt == "xml.gz":
            print(f"    Decompressing -> {out_path.name}")
            with gzip.open(tmp_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            tmp_path.unlink()
        elif fmt == "json":
            # BiGG JSON format — convert to SBML via COBRApy
            import cobra
            print(f"    Converting JSON -> SBML ...")
            m = cobra.io.load_json_model(str(tmp_path))
            cobra.io.write_sbml_model(m, str(out_path))
            tmp_path.unlink()

        # Validate the downloaded model
        import cobra
        m = cobra.io.read_sbml_model(str(out_path))
        print(f"  {model_id}: {len(m.reactions)} reactions, "
              f"{len(m.metabolites)} metabolites, {len(m.genes)} genes")
        return out_path

    raise RuntimeError(
        f"Could not download {model_id} from any BiGG URL.\n"
        "Use --gold-standard to provide a local model file instead."
    )


def copy_gold_standard(source: Path, dest: Path) -> Path:
    """Copy a local SBML model as the gold standard."""
    out_path = dest / "gold_standard.xml"
    print(f"  Copying {source} -> {out_path.name}")
    shutil.copy2(source, out_path)

    import cobra
    m = cobra.io.read_sbml_model(str(out_path))
    print(f"  {source.stem}: {len(m.reactions)} reactions, "
          f"{len(m.metabolites)} metabolites, {len(m.genes)} genes")
    return out_path


# ---------------------------------------------------------------------------
# ESM C reference DB
# ---------------------------------------------------------------------------

def build_esm_db(proteins_faa: Path, dest: Path) -> Path:
    """Generate the FAISS reference database from a protein FASTA."""
    db_dir = dest / "esm_db"

    # Check if already built
    if ((db_dir / "reference.faiss").exists()
            and (db_dir / "reference_ids.json").exists()
            and (db_dir / "reference_embeddings.npz").exists()):
        print(f"  ESM C database already exists at {db_dir}")
        return db_dir

    from gemiz.embedding.database import generate_reference_db

    print(f"  Generating ESM C reference database ...")
    generate_reference_db(str(proteins_faa), str(db_dir))
    print(f"  Saved to {db_dir}")
    return db_dir


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download a file with a progress indicator."""
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code == 404:
        raise FileNotFoundError(f"Not found: {url}")
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                      f"({pct:.0f}%)", end="", flush=True)
    if total:
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and set up reference data for an organism.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "name",
        help="Short organism name (e.g. ecoli, bsubtilis, pputida).",
    )
    parser.add_argument(
        "--ncbi-assembly",
        required=True,
        help="NCBI RefSeq assembly accession (e.g. GCF_000009045.1).",
    )
    parser.add_argument(
        "--bigg-model",
        default=None,
        help="BiGG model ID for gold-standard benchmarking (e.g. iYO844). "
             "Optional; skipped if not provided.",
    )
    parser.add_argument(
        "--gold-standard",
        default=None,
        help="Path to a local SBML model (.xml) to use as gold standard "
             "instead of downloading from BiGG.",
    )
    parser.add_argument(
        "--skip-bigg",
        action="store_true",
        help="Skip BiGG model download entirely.",
    )
    parser.add_argument(
        "--skip-esm",
        action="store_true",
        help="Skip ESM C database generation (requires torch + GPU).",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Output directory (default: data/organisms/<name>).",
    )

    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else Path("data/organisms") / args.name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nSetting up reference data for: {args.name}")
    print(f"  Output:   {dest}")
    print(f"  Assembly: {args.ncbi_assembly}")
    if args.gold_standard:
        print(f"  Gold std: {args.gold_standard} (local)")
    elif args.bigg_model:
        print(f"  BiGG:     {args.bigg_model}")
    print()

    t0 = time.perf_counter()

    # Step 1: proteins
    print("[1/5] Downloading annotated proteins from NCBI ...")
    proteins_faa = download_proteins(args.ncbi_assembly, dest)

    # Step 2: feature table
    print("\n[2/5] Downloading feature table from NCBI ...")
    feature_table = download_feature_table(args.ncbi_assembly, dest)

    # Step 3: Gold standard model (optional)
    if args.gold_standard:
        print(f"\n[3/5] Copying local gold-standard model ...")
        gs_path = Path(args.gold_standard)
        if not gs_path.exists():
            print(f"  ERROR: file not found: {gs_path}", file=sys.stderr)
            sys.exit(1)
        copy_gold_standard(gs_path, dest)
    elif args.bigg_model and not args.skip_bigg:
        print(f"\n[3/5] Downloading gold-standard model from BiGG ...")
        download_bigg_model(args.bigg_model, dest)
    else:
        print("\n[3/5] No gold-standard model specified, skipping.")

    # Step 4: ESM C database (optional)
    if not args.skip_esm:
        print("\n[4/5] Building ESM C reference database ...")
        try:
            build_esm_db(proteins_faa, dest)
        except ImportError as e:
            print(f"  Skipping: {e}")
            print("  Install gemiz[embeddings] for ESM C support.")
    else:
        print("\n[4/5] Skipping ESM C database (--skip-esm).")

    # Step 5: Generate config.json
    print("\n[5/5] Generating config.json ...")
    config = {
        "name": args.name,
        "proteins": str(dest / "proteins.faa"),
        "feature_table": str(dest / "feature_table.txt"),
    }
    if (dest / "gold_standard.xml").exists():
        config["template"] = str(dest / "gold_standard.xml")
    if (dest / "esm_db").exists():
        config["esm_db"] = str(dest / "esm_db")

    # Detect locus tag underscore format from feature table
    ft_path = dest / "feature_table.txt"
    if ft_path.exists():
        import re
        has_underscore = False
        with open(ft_path, encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 17 and cols[0] == "CDS":
                    tag = cols[16].strip()
                    if tag and re.match(r"^[A-Za-z]+_\d+$", tag):
                        has_underscore = True
                        break
        config["locus_tag_strip_underscore"] = has_underscore

    config_path = dest / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {config_path}")

    elapsed = time.perf_counter() - t0

    # Summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nFiles at {dest}/")
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            if size > 1e6:
                label = f"{size / 1e6:.1f} MB"
            elif size > 1e3:
                label = f"{size / 1e3:.0f} KB"
            else:
                label = f"{size} B"
            print(f"  {p.relative_to(dest)}  ({label})")

    print(f"\nTo reconstruct a model:")
    print(f"  gemiz carve <genome>.fna \\")
    print(f"    --reference {dest}/proteins.faa \\")
    print(f"    --feature-table {dest}/feature_table.txt \\")
    if (dest / "esm_db").exists():
        print(f"    --esm-db {dest}/esm_db \\")
    print(f"    -o <output>.xml")


if __name__ == "__main__":
    main()
