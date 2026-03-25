"""Database downloader — BiGG and ModelSEED protein sequences."""

from __future__ import annotations

from pathlib import Path

import requests
from rich.console import Console
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Public URLs
# ---------------------------------------------------------------------------
_SOURCES: dict[str, dict[str, str]] = {
    "bigg": {
        # BiGG universal model (SBML) — used to extract protein sequences
        "universal_model": (
            "http://bigg.ucsd.edu/static/namespace/bigg_models_reactions.txt"
        ),
        # Pre-built FASTA of BiGG metabolite-linked sequences (community mirror)
        # TODO: build DIAMOND db from NCBI RefSeq proteins mapped to BiGG reactions
    },
    "modelseed": {
        # ModelSEED biochemistry GitHub release
        "reactions": (
            "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
            "master/Biochemistry/reactions.tsv"
        ),
        "compounds": (
            "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
            "master/Biochemistry/compounds.tsv"
        ),
    },
}


def download(*, db: str, dest: Path, console: Console) -> None:
    """Download reference database files to *dest*."""
    targets = list(_SOURCES.keys()) if db == "all" else [db]

    for name in targets:
        console.print(f"[bold]Downloading[/] [cyan]{name}[/] database …")
        sources = _SOURCES.get(name, {})
        for label, url in sources.items():
            out = dest / f"{name}_{label}"
            if out.exists():
                console.print(f"  [yellow]skip[/] {out.name} (already exists)")
                continue
            _download_file(url, out, console=console)
        console.print(f"  [green]✓[/] {name} done")


def _download_file(url: str, dest: Path, *, console: Console) -> None:
    """Stream-download *url* → *dest* with a progress bar."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    console.print(f"  → {dest.name}")

    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
