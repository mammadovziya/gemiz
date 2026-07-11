#!/usr/bin/env python3
"""Import open CarveMe reference assets for gemiz universal mode.

CarveMe ships an open BiGG bacterial universe, BiGG proteins, and GPR mappings.
This script converts those packaged assets into the local gemiz universal layout:

  data/universal/carveme_bacteria.xml
  data/universal/db/universal_proteins.faa
  data/universal/db/universal_gpr.csv
  data/universal/db/mmseqs_db/db*

The generated files are intentionally ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSAL_DIR = REPO_ROOT / "data" / "universal"
DB_DIR = UNIVERSAL_DIR / "db"

TEMPLATE_OUT = UNIVERSAL_DIR / "carveme_bacteria.xml"
FAA_OUT = DB_DIR / "universal_proteins.faa"
GPR_OUT = DB_DIR / "universal_gpr.csv"
MMSEQS_DIR = DB_DIR / "mmseqs_db"
MANIFEST_OUT = DB_DIR / "carveme_import_manifest.json"
MEDIA_OUT = UNIVERSAL_DIR / "media_db.tsv"


def _carveme_generated_dir() -> Path:
    try:
        import carveme
    except ImportError as exc:
        raise SystemExit(
            "CarveMe is not installed in this environment. "
            "Install it first, then rerun this importer."
        ) from exc

    return Path(carveme.__file__).resolve().parent / "data" / "generated"


def _copy_if_needed(src: Path, dst: Path, *, force: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return False
    shutil.copy2(src, dst)
    return True


def _gunzip_if_needed(src: Path, dst: Path, *, force: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return False
    with gzip.open(src, "rb") as compressed, dst.open("wb") as out:
        shutil.copyfileobj(compressed, out)
    return True


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix):] if value.startswith(prefix) else value


def _normalise_reaction_id(value: str) -> str:
    return _strip_prefix(value.strip(), "R_")


def _normalise_gene_id(model_id: str, value: str) -> str:
    gene = _strip_prefix(value.strip(), "G_")
    return f"{model_id}.{gene}"


def _protein_clause(model_id: str, protein_id: str) -> str:
    """Convert a CarveMe protein-complex id to a boolean GPR clause."""
    protein = _strip_prefix(protein_id.strip(), "P_")
    subunits = [
        _normalise_gene_id(model_id, subunit)
        for subunit in protein.split("+")
        if subunit.strip()
    ]
    if not subunits:
        return ""
    if len(subunits) == 1:
        return subunits[0]
    return "(" + " and ".join(subunits) + ")"


def _convert_gprs(src_gz: Path, dst_csv: Path, *, force: bool) -> dict[str, Any]:
    """Convert CarveMe bigg_gprs.csv.gz to gemiz universal_gpr.csv."""
    if dst_csv.exists() and not force:
        return {"status": "cached", "path": str(dst_csv)}

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    rules: dict[tuple[str, str], set[str]] = {}
    source_rows = 0

    with gzip.open(src_gz, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_rows += 1
            model_id = row["model"].strip()
            reaction_id = _normalise_reaction_id(row["reaction"])
            clause = _protein_clause(model_id, row["protein"])
            if clause:
                rules.setdefault((reaction_id, model_id), set()).add(clause)

    with dst_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bigg_id", "reaction_id", "gpr"],
        )
        writer.writeheader()
        for (reaction_id, model_id), clauses in sorted(rules.items()):
            writer.writerow({
                "bigg_id": model_id,
                "reaction_id": reaction_id,
                "gpr": " or ".join(sorted(clauses)),
            })

    reaction_ids = {reaction_id for reaction_id, _model_id in rules}
    model_ids = {model_id for _reaction_id, model_id in rules}
    return {
        "status": "written",
        "path": str(dst_csv),
        "source_rows": source_rows,
        "rules": len(rules),
        "reactions": len(reaction_ids),
        "source_models": len(model_ids),
    }


def _count_fasta_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def _template_stats(path: Path) -> dict[str, Any]:
    import cobra

    model = cobra.io.read_sbml_model(str(path))
    objectives = [
        rxn.id for rxn in model.reactions
        if rxn.objective_coefficient != 0
    ]
    return {
        "reactions": len(model.reactions),
        "metabolites": len(model.metabolites),
        "genes": len(model.genes),
        "objective_reactions": objectives,
    }


def _build_mmseqs_db() -> dict[str, Any]:
    from gemiz.pipeline.alignment import build_mmseqs_db

    prefix = build_mmseqs_db(str(FAA_OUT), str(MMSEQS_DIR))
    return {
        "status": "written",
        "prefix": prefix,
        "files": sorted(p.name for p in MMSEQS_DIR.glob("db*")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import CarveMe's open bacterial universe into gemiz universal mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--skip-mmseqs",
        action="store_true",
        help="Do not build the optional MMseqs2 target database.",
    )
    args = parser.parse_args()

    generated = _carveme_generated_dir()
    template_src = generated / "universe_bacteria.xml.gz"
    proteins_src = generated / "bigg_proteins.faa"
    gpr_src = generated / "bigg_gprs.csv.gz"
    media_src = generated.parent / "input" / "media_db.tsv"

    missing = [
        path for path in (template_src, proteins_src, gpr_src)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "CarveMe installation is missing required generated assets:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    UNIVERSAL_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print("[gemiz] Importing CarveMe universal assets...")
    template_written = _gunzip_if_needed(template_src, TEMPLATE_OUT, force=args.force)
    proteins_written = _copy_if_needed(proteins_src, FAA_OUT, force=args.force)
    media_written = (
        _copy_if_needed(media_src, MEDIA_OUT, force=args.force)
        if media_src.exists()
        else False
    )
    gpr_stats = _convert_gprs(gpr_src, GPR_OUT, force=args.force)

    manifest: dict[str, Any] = {
        "source": str(generated),
        "outputs": {
            "template": str(TEMPLATE_OUT),
            "proteins": str(FAA_OUT),
            "gpr": str(GPR_OUT),
            "media": str(MEDIA_OUT) if MEDIA_OUT.exists() else None,
        },
        "written": {
            "template": template_written,
            "proteins": proteins_written,
            "media": media_written,
        },
        "template": _template_stats(TEMPLATE_OUT),
        "proteins": {
            "records": _count_fasta_records(FAA_OUT),
        },
        "gpr": gpr_stats | {
            "rows": _count_csv_rows(GPR_OUT),
        },
    }

    if not args.skip_mmseqs:
        try:
            manifest["mmseqs"] = _build_mmseqs_db()
        except Exception as exc:  # noqa: BLE001
            manifest["mmseqs"] = {"status": "error", "error": str(exc)}
            print(f"[gemiz] WARNING: MMseqs2 DB build failed: {exc}")
    else:
        manifest["mmseqs"] = {"status": "skipped"}

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_OUT.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nImport complete")
    print("===============")
    print(f"  Template: {manifest['template']['reactions']:,} reactions")
    print(f"  Proteins: {manifest['proteins']['records']:,} sequences")
    print(f"  GPR rows: {manifest['gpr']['rows']:,}")
    print(f"  Manifest: {MANIFEST_OUT}")
    if manifest["mmseqs"].get("status") == "written":
        print(f"  MMseqs2:  {manifest['mmseqs']['prefix']}")


if __name__ == "__main__":
    main()
