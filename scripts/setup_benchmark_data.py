#!/usr/bin/env python3
"""Download the small public assets needed for local gemiz benchmarks.

The full universal database can take hours to build. This script prepares the
E. coli smoke/benchmark fixture used by tests and examples:

  data/genomes/ecoli_k12.fna
  data/reference/iML1515_proteins.faa
  data/reference/ecoli_feature_table.txt
  data/universal/iML1515.xml

All sources are public: NCBI RefSeq for genome/proteins/feature table and BiGG
for the curated iML1515 model.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ECOLI_ASSEMBLY = "GCF_000005845.2"
BIGG_MODEL = "iML1515"
NCBI_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all"

GENOME_OUT = DATA / "genomes" / "ecoli_k12.fna"
PROTEINS_OUT = DATA / "reference" / "iML1515_proteins.faa"
FEATURE_TABLE_OUT = DATA / "reference" / "ecoli_feature_table.txt"
MODEL_OUT = DATA / "universal" / "iML1515.xml"
MANIFEST_OUT = DATA / "benchmark_manifest.json"


def _assembly_parent(accession: str) -> str:
    prefix = accession[:3]
    digits = accession[4:].split(".")[0]
    return f"{NCBI_FTP}/{prefix}/{digits[0:3]}/{digits[3:6]}/{digits[6:9]}"


def _find_assembly_dir(accession: str) -> str:
    parent = _assembly_parent(accession)
    response = requests.get(parent, timeout=30)
    response.raise_for_status()
    pattern = re.compile(rf'href="({re.escape(accession)}[^"]*)"')
    matches = pattern.findall(response.text)
    if not matches:
        raise RuntimeError(f"No NCBI assembly directory for {accession} at {parent}")
    return f"{parent}/{matches[0].rstrip('/')}"


def _download(url: str, dest: Path, *, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  cached: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", "0"))
    done = 0
    with tmp.open("wb") as f:
        for chunk in response.iter_content(1024 * 256):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r  {dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
    if total:
        print()
    tmp.replace(dest)


def _download_gzip_to_file(url: str, dest: Path, *, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  cached: {dest}")
        return

    gz_path = dest.with_suffix(dest.suffix + ".gz")
    _download(url, gz_path, force=True)
    print(f"  decompress: {gz_path.name} -> {dest.name}")
    with gzip.open(gz_path, "rb") as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)
    gz_path.unlink()


def download_ncbi_assets(*, force: bool = False) -> dict[str, str]:
    print(f"\n[NCBI] {ECOLI_ASSEMBLY}")
    asm_dir = _find_assembly_dir(ECOLI_ASSEMBLY)
    asm_name = asm_dir.rsplit("/", 1)[-1]

    assets = {
        "genome": (f"{asm_dir}/{asm_name}_genomic.fna.gz", GENOME_OUT),
        "proteins": (f"{asm_dir}/{asm_name}_protein.faa.gz", PROTEINS_OUT),
        "feature_table": (
            f"{asm_dir}/{asm_name}_feature_table.txt.gz",
            FEATURE_TABLE_OUT,
        ),
    }

    for _name, (url, dest) in assets.items():
        _download_gzip_to_file(url, dest, force=force)

    return {name: str(dest) for name, (_url, dest) in assets.items()}


def download_bigg_model(*, force: bool = False) -> str:
    print(f"\n[BiGG] {BIGG_MODEL}")
    if MODEL_OUT.exists() and not force:
        print(f"  cached: {MODEL_OUT}")
        return str(MODEL_OUT)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://bigg.ucsd.edu/static/models/{BIGG_MODEL}.xml.gz",
        f"http://bigg.ucsd.edu/static/models/{BIGG_MODEL}.xml.gz",
    ]
    last_error: Exception | None = None
    for url in urls:
        try:
            _download_gzip_to_file(url, MODEL_OUT, force=True)
            return str(MODEL_OUT)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  failed: {url} ({exc})")

    raise RuntimeError(f"Could not download {BIGG_MODEL}: {last_error}")


def write_manifest(paths: dict[str, str]) -> None:
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "organism": "ecoli",
        "assembly": ECOLI_ASSEMBLY,
        "bigg_model": BIGG_MODEL,
        "paths": paths,
    }
    with MANIFEST_OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nManifest: {MANIFEST_OUT}")


def main() -> None:
    force = "--force" in sys.argv
    paths = download_ncbi_assets(force=force)
    paths["gold_standard"] = download_bigg_model(force=force)
    write_manifest(paths)

    print("\nReady:")
    print("  pytest -q")
    print("  python scripts/benchmark_competitors.py --organism ecoli "
          "--genome data/genomes/ecoli_k12.fna "
          "--gold-standard data/universal/iML1515.xml --tools gemiz")


if __name__ == "__main__":
    main()
