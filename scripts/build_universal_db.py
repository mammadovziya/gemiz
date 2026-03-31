#!/usr/bin/env python3
"""Build gemiz universal reference database from all BiGG models.

Downloads proteins + feature tables for every BiGG model that has a
genome reference, maps GPRs to protein sequences, and produces:

  data/universal/db/
    universal_proteins.faa    all proteins, namespaced headers
    universal_gpr.csv         reaction → protein mappings
    universal_template.xml    merged SBML with namespaced GPRs
    build_log.json            per-model success/failure log
    mmseqs_db/                MMseqs2 sequence database

Usage
-----
    python scripts/build_universal_db.py           # build all models
    python scripts/build_universal_db.py --only iML1515 iYO844
    python scripts/build_universal_db.py --force   # re-download everything
    python scripts/build_universal_db.py --skip-mmseqs
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("data")
DB_DIR      = DATA_ROOT / "universal" / "db"
ORG_CACHE   = DATA_ROOT / "universal" / "organisms"
MMSEQS_DIR  = DB_DIR / "mmseqs_db"

FAA_OUT     = DB_DIR / "universal_proteins.faa"
CSV_OUT     = DB_DIR / "universal_gpr.csv"
XML_OUT     = DB_DIR / "universal_template.xml"
LOG_OUT     = DB_DIR / "build_log.json"

_NCBI_FTP   = "https://ftp.ncbi.nlm.nih.gov/genomes/all"

# ---------------------------------------------------------------------------
# BiGG API
# ---------------------------------------------------------------------------

_BIGG_API = "http://bigg.ucsd.edu/api/v2"


def fetch_bigg_model_list() -> list[dict]:
    """Return list of {bigg_id, organism, ...} dicts from the BiGG API."""
    print("Fetching BiGG model list ...")
    r = requests.get(f"{_BIGG_API}/models", timeout=30)
    r.raise_for_status()
    models = r.json().get("results", [])
    print(f"  {len(models)} models listed")
    return models


def fetch_bigg_model_detail(bigg_id: str) -> dict:
    """Return model detail including genome_ref_string."""
    r = requests.get(f"{_BIGG_API}/models/{bigg_id}", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Accession resolution
# ---------------------------------------------------------------------------

_NC_GCF_CACHE_PATH = DATA_ROOT / "universal" / "nc_to_gcf_cache.json"
_nc_gcf_cache: dict[str, str] = {}


def _load_nc_gcf_cache() -> None:
    """Load the NC_ -> GCF cache from disk into _nc_gcf_cache."""
    global _nc_gcf_cache
    if _NC_GCF_CACHE_PATH.exists():
        with open(_NC_GCF_CACHE_PATH) as f:
            _nc_gcf_cache = json.load(f)


def _save_nc_gcf_cache() -> None:
    """Persist _nc_gcf_cache to disk."""
    _NC_GCF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_NC_GCF_CACHE_PATH, "w") as f:
        json.dump(_nc_gcf_cache, f, indent=2)


def resolve_gcf_accession(genome_ref: str) -> str | None:
    """Convert a BiGG genome_ref_string to a GCF_ assembly accession.

    Handles two formats:
      "ncbi_assembly:GCF_000146045.2"  -> strip prefix, return GCF directly
      "ncbi_accession:NC_000913.3"     -> convert via NCBI eutils elink
    """
    if not genome_ref:
        return None

    # Format A: direct GCF reference anywhere in the string
    m = re.search(r"GCF_\d+\.\d+", genome_ref)
    if m:
        return m.group(0)

    # Format B: ncbi_accession:NC_XXXXXX.X  (or NZ_, NW_, …)
    m2 = re.search(r"ncbi_accession:(\S+)", genome_ref)
    if m2:
        nc_acc = m2.group(1).strip()
        return _nc_to_gcf_eutils(nc_acc)

    # Fallback: any bare NC_/NZ_ token
    m3 = re.search(r"(N[CZWT]_\d+(?:\.\d+)?)", genome_ref)
    if m3:
        return _nc_to_gcf_eutils(m3.group(1))

    return None


def _nc_to_gcf_eutils(nc_acc: str) -> str | None:
    """Convert a nucleotide accession (NC_/NZ_) to a GCF assembly accession.

    Three-step eutils chain:
      1. efetch nuccore docsum -> numeric UID
      2. elink nuccore -> assembly -> assembly UID
      3. esummary assembly -> assemblyaccession (GCF_...)

    Results are cached in data/universal/nc_to_gcf_cache.json.
    """
    cache_key = nc_acc
    if cache_key in _nc_gcf_cache:
        return _nc_gcf_cache[cache_key]

    try:
        # Step 1: efetch nuccore docsum -> numeric UID
        r1 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "nuccore", "id": nc_acc,
                    "rettype": "docsum", "retmode": "json"},
            timeout=20,
        )
        r1.raise_for_status()
        data1 = r1.json()
        uids = data1["result"]["uids"]
        if not uids:
            return None
        nuccore_uid = uids[0]

        # Step 2: elink nuccore -> assembly
        r2 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            params={"dbfrom": "nuccore", "db": "assembly",
                    "id": nuccore_uid, "retmode": "json"},
            timeout=20,
        )
        r2.raise_for_status()
        data2 = r2.json()
        links = (data2.get("linksets", [{}])[0]
                      .get("linksetdbs", [{}])[0]
                      .get("links", []))
        if not links:
            return None
        assembly_id = links[0]

        # Step 3: esummary assembly -> GCF accession
        r3 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "assembly", "id": assembly_id, "retmode": "json"},
            timeout=20,
        )
        r3.raise_for_status()
        data3 = r3.json()
        gcf = data3["result"][str(assembly_id)]["assemblyaccession"]

        _nc_gcf_cache[cache_key] = gcf
        _save_nc_gcf_cache()
        return gcf

    except Exception as e:
        print(f"    WARNING: NC->GCF lookup failed for {nc_acc}: {e}")
        return None


# ---------------------------------------------------------------------------
# NCBI FTP helpers (inline, no import from setup_organism)
# ---------------------------------------------------------------------------

def _assembly_ftp_dir(accession: str) -> str:
    prefix = accession[:3]
    digits = accession[4:].split(".")[0]
    p1, p2, p3 = digits[0:3], digits[3:6], digits[6:9]
    return f"{_NCBI_FTP}/{prefix}/{p1}/{p2}/{p3}"


def _find_assembly_dir(accession: str) -> str:
    parent = _assembly_ftp_dir(accession)
    r = requests.get(parent, timeout=30)
    r.raise_for_status()
    pattern = re.compile(rf'href="({re.escape(accession)}[^"]*)"')
    matches = pattern.findall(r.text)
    if not matches:
        raise RuntimeError(f"No assembly directory for {accession} at {parent}")
    return f"{parent}/{matches[0].rstrip('/')}"


def _download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code == 404:
        raise FileNotFoundError(f"404: {url}")
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r      {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
    if total:
        print()


def _download_ncbi_file(accession: str, dest: Path, suffix: str, out_name: str) -> Path:
    """Download a named file from an NCBI assembly FTP directory."""
    asm_url  = _find_assembly_dir(accession)
    asm_name = asm_url.rsplit("/", 1)[-1]
    gz_name  = f"{asm_name}_{suffix}.gz"
    gz_path  = dest / gz_name
    out_path = dest / out_name

    _download_file(f"{asm_url}/{gz_name}", gz_path)
    print(f"      Decompressing ...", end=" ", flush=True)
    with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    gz_path.unlink()
    print("done")
    return out_path


# ---------------------------------------------------------------------------
# BiGG model download
# ---------------------------------------------------------------------------

def _download_bigg_model(bigg_id: str, dest: Path) -> Path:
    """Return a local SBML model for *bigg_id*, downloading if necessary.

    Checks these cache locations before downloading:
      1. data/universal/{bigg_id}.xml          (existing universal models)
      2. data/organisms/{bigg_id}/gold_standard.xml
      3. data/universal/organisms/{bigg_id}/model.xml  (this script's cache)

    BiGG static files are served over plain HTTP only.
    """
    out_path = dest / "model.xml"

    # ---- local cache check ----
    candidates = [
        DATA_ROOT / "universal" / f"{bigg_id}.xml",
        DATA_ROOT / "organisms" / bigg_id / "gold_standard.xml",
        out_path,
    ]
    for cached in candidates:
        if cached.exists():
            print(f"      Using cached model: {cached}")
            if cached != out_path:
                import shutil as _shutil
                _shutil.copy2(cached, out_path)
            return out_path

    # ---- download (HTTP only — BiGG does not accept HTTPS for static files) ----
    urls = [
        (f"http://bigg.ucsd.edu/static/models/{bigg_id}.xml.gz", "gz"),
        (f"http://bigg.ucsd.edu/static/models/{bigg_id}.json",   "json"),
    ]
    for url, fmt in urls:
        tmp = dest / f"_tmp.{fmt}"
        try:
            _download_file(url, tmp)
            if fmt == "gz":
                with gzip.open(tmp, "rb") as fin, open(out_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                tmp.unlink()
            else:
                import cobra
                m = cobra.io.load_json_model(str(tmp))
                cobra.io.write_sbml_model(m, str(out_path))
                tmp.unlink()
            return out_path
        except Exception:
            if tmp.exists():
                tmp.unlink()
            continue
    raise RuntimeError(f"Could not download BiGG model {bigg_id}")


# ---------------------------------------------------------------------------
# Feature table / protein FASTA parsing
# ---------------------------------------------------------------------------

class _FTRow:
    """One CDS row from an NCBI feature table."""
    __slots__ = ("accession", "locus_tag", "symbol", "name", "old_locus_tags")

    def __init__(self, cols: list[str]) -> None:
        self.accession   = cols[10].strip() if len(cols) > 10 else ""
        self.locus_tag   = cols[16].strip() if len(cols) > 16 else ""
        self.symbol      = cols[14].strip() if len(cols) > 14 else ""
        self.name        = cols[13].strip() if len(cols) > 13 else ""

        # old_locus_tag lives in the attributes column (col 19)
        # Format: "old_locus_tag=TM0006" or "old_locus_tag=TM0006,TM_0006"
        self.old_locus_tags: list[str] = []
        attrs = cols[19].strip() if len(cols) > 19 else ""
        if attrs:
            for part in attrs.split(";"):
                part = part.strip()
                if part.startswith("old_locus_tag="):
                    raw = part[len("old_locus_tag="):]
                    self.old_locus_tags = [t.strip() for t in raw.split(",") if t.strip()]


def _parse_feature_table_full(ft_path: Path) -> list[_FTRow]:
    """Parse all CDS rows from an NCBI feature table."""
    rows: list[_FTRow] = []
    with open(ft_path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 11 or cols[0] != "CDS":
                continue
            row = _FTRow(cols)
            if row.accession:
                rows.append(row)
    return rows


def _build_id_map(
    ft_rows: list[_FTRow],
    model_gene_ids: set[str],
) -> dict[str, str]:
    """Build {accession: gene_id} trying multiple feature table columns.

    Tries columns in order of specificity and returns the first mapping
    that resolves >10% of model gene IDs:
      1. locus_tag          (e.g. b0001, BSU_00010)
      2. old_locus_tag      (pre-RefSeq nomenclature)
      3. symbol             (e.g. thrL, glk)
      4. Union of all above (last resort)
    """
    # Expand model gene IDs to include underscore variants
    expanded_ids: set[str] = set()
    for gid in model_gene_ids:
        expanded_ids.update(_tag_variants(gid))

    def _try_column(
        extractor: "callable",
    ) -> dict[str, str]:
        acc_map: dict[str, str] = {}
        for row in ft_rows:
            tags = extractor(row)
            for tag in tags:
                for v in _tag_variants(tag):
                    if v in expanded_ids:
                        acc_map[row.accession] = tag
                        break
                if row.accession in acc_map:
                    break
        return acc_map

    def _match_rate(acc_map: dict[str, str]) -> float:
        if not model_gene_ids:
            return 0.0
        mapped_tags: set[str] = set()
        for tag in acc_map.values():
            mapped_tags.update(_tag_variants(tag))
        return len(mapped_tags & expanded_ids) / len(model_gene_ids)

    # Strategy 1: locus_tag
    m1 = _try_column(lambda r: [r.locus_tag] if r.locus_tag else [])
    if _match_rate(m1) > 0.1:
        return m1

    # Strategy 2: old_locus_tag
    m2 = _try_column(lambda r: r.old_locus_tags)
    if _match_rate(m2) > 0.1:
        return m2

    # Strategy 3: symbol (gene common name like thrL, glk)
    m3 = _try_column(lambda r: [r.symbol] if r.symbol else [])
    if _match_rate(m3) > 0.1:
        return m3

    # Strategy 4: union of all — take whatever matches
    merged: dict[str, str] = {}
    for row in ft_rows:
        candidates = []
        if row.locus_tag:
            candidates.append(row.locus_tag)
        candidates.extend(row.old_locus_tags)
        if row.symbol:
            candidates.append(row.symbol)
        for tag in candidates:
            for v in _tag_variants(tag):
                if v in expanded_ids:
                    merged[row.accession] = tag
                    break
            if row.accession in merged:
                break

    return merged


def _tag_variants(tag: str) -> list[str]:
    m = re.match(r"^([A-Za-z]+)_(\d+)$", tag)
    if m:
        return [tag, m.group(1) + m.group(2)]
    m2 = re.match(r"^([A-Za-z]+)(\d+)$", tag)
    if m2:
        return [tag, f"{m2.group(1)}_{m2.group(2)}"]
    return [tag]


def _parse_proteins_faa(faa_path: Path) -> dict[str, str]:
    """Return {accession: sequence}."""
    seqs: dict[str, str] = {}
    cur: str | None = None
    parts: list[str] = []
    with open(faa_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs[cur] = "".join(parts)
                cur = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
        if cur:
            seqs[cur] = "".join(parts)
    return seqs


# ---------------------------------------------------------------------------
# GPR gene extraction
# ---------------------------------------------------------------------------

_GPR_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-:]+")


def _extract_gpr_genes(gpr: str) -> list[str]:
    if not gpr.strip():
        return []
    tokens = _GPR_TOKEN_RE.findall(gpr)
    keywords = {"and", "or", "not", "AND", "OR", "NOT"}
    return list(dict.fromkeys(t for t in tokens if t not in keywords))




# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def process_model(
    bigg_id: str,
    gcf_acc: str,
    org_dir: Path,
    force: bool,
) -> dict:
    """
    Download and process one BiGG model.
    Returns a result dict with keys: proteins, gpr_rows, model.
    Raises on unrecoverable error.
    """
    import cobra

    org_dir.mkdir(parents=True, exist_ok=True)

    # ---- BiGG model ----
    model_path = org_dir / "model.xml"
    if not model_path.exists() or force:
        print(f"      Downloading BiGG model ...", flush=True)
        _download_bigg_model(bigg_id, org_dir)
    else:
        print(f"      Model cached")

    model = cobra.io.read_sbml_model(str(model_path))
    n_rxns = len(model.reactions)
    n_rxns_gpr = sum(1 for r in model.reactions if r.gene_reaction_rule.strip())
    print(f"      {n_rxns:,} reactions  |  {n_rxns_gpr:,} with GPR")

    # ---- NCBI proteins ----
    faa_path = org_dir / "proteins.faa"
    if not faa_path.exists() or force:
        print(f"      Downloading proteins ({gcf_acc}) ...", flush=True)
        _download_ncbi_file(gcf_acc, org_dir, "protein.faa", "proteins.faa")
    else:
        print(f"      Proteins cached")

    n_prot = sum(1 for l in open(faa_path) if l.startswith(">"))
    print(f"      {n_prot:,} proteins")

    # ---- Feature table ----
    ft_path = org_dir / "feature_table.txt"
    if not ft_path.exists() or force:
        print(f"      Downloading feature table ...", flush=True)
        _download_ncbi_file(gcf_acc, org_dir, "feature_table.txt", "feature_table.txt")
    else:
        print(f"      Feature table cached")

    # ---- Build locus_tag -> sequence map ----
    ft_rows = _parse_feature_table_full(ft_path)
    acc_to_seq = _parse_proteins_faa(faa_path)

    # Collect all gene identifiers the model uses (gene.id + gene.name)
    model_gene_ids: set[str] = set()
    for gene in model.genes:
        model_gene_ids.add(gene.id)
        if gene.name:
            model_gene_ids.add(gene.name)

    acc_to_tag = _build_id_map(ft_rows, model_gene_ids)
    print(f"      ID map: {len(acc_to_tag):,} accession -> gene entries")

    tag_to_seq: dict[str, str] = {}
    tag_to_acc: dict[str, str] = {}
    for acc, tag in acc_to_tag.items():
        if acc in acc_to_seq:
            seq = acc_to_seq[acc]
            for v in _tag_variants(tag):
                tag_to_seq[v] = seq
                tag_to_acc[v] = acc

    # Also index by gene.name for models that use b-numbers as gene.id
    name_to_tag: dict[str, str] = {}
    for gene in model.genes:
        if gene.name:
            for v in _tag_variants(gene.name):
                if v in tag_to_seq:
                    name_to_tag[gene.id] = gene.name
                    break
        if gene.id not in name_to_tag:
            for v in _tag_variants(gene.id):
                if v in tag_to_seq:
                    name_to_tag[gene.id] = gene.id
                    break

    # ---- Map GPRs + collect proteins ----
    proteins: list[tuple[str, str, str, str]] = []  # (bigg_id, tag, acc, seq)
    gpr_rows: list[dict] = []
    seen_tags: set[str] = set()
    mapped_genes = 0
    total_genes = len(model.genes)

    for rxn in model.reactions:
        gpr_raw = rxn.gene_reaction_rule.strip()
        gene_ids = _extract_gpr_genes(gpr_raw)

        ns_gpr = gpr_raw

        gpr_rows.append({
            "bigg_id":     bigg_id,
            "reaction_id": rxn.id,
            "gpr":         ns_gpr,
            "lb":          rxn.lower_bound,
            "ub":          rxn.upper_bound,
        })

        for gid in gene_ids:
            canonical = name_to_tag.get(gid, gid)
            for v in _tag_variants(canonical):
                if v in tag_to_seq:
                    if v not in seen_tags:
                        seen_tags.add(v)
                        proteins.append((
                            bigg_id, v,
                            tag_to_acc.get(v, "unknown"),
                            tag_to_seq[v],
                        ))
                    break

    for gene in model.genes:
        canonical = name_to_tag.get(gene.id, gene.id)
        for v in _tag_variants(canonical):
            if v in tag_to_seq:
                mapped_genes += 1
                break

    pct = 100 * mapped_genes / total_genes if total_genes else 0
    warn = "  ⚠ low mapping" if pct < 50 and total_genes > 10 else ""
    print(f"      Genes mapped: {mapped_genes}/{total_genes} ({pct:.1f}%){warn}")
    print(f"      Proteins written: {len(proteins):,}")

    return {
        "bigg_id":    bigg_id,
        "gcf_acc":    gcf_acc,
        "n_reactions": n_rxns,
        "n_gpr":      n_rxns_gpr,
        "n_proteins_written": len(proteins),
        "mapped_pct": round(pct, 1),
        "proteins":   proteins,
        "gpr_rows":   gpr_rows,
        "model":      model,
    }


# ---------------------------------------------------------------------------
# Universal template assembly
# ---------------------------------------------------------------------------

def build_template(model_results: list[dict]) -> "cobra.Model":
    """Merge all reactions into one COBRApy model.

    Biomass reactions from each source model are tagged via ``rxn.notes``
    so that :func:`carving.setup_milp` can discover candidate biomass
    reactions when the merged model has no single objective set.

    Tags stored:
      ``rxn.notes["gemiz_biomass"]``   = source BiGG model ID
      ``rxn.notes["gemiz_organism"]``  = organism short name
    """
    import cobra

    universal = cobra.Model("universal_template")
    universal.name = "gemiz universal template"

    rxn_gprs: dict[str, list[str]] = {}
    rxn_obj:  dict[str, "cobra.Reaction"] = {}
    # {rxn_id: (bigg_id, organism_name)} — first model that contributed it
    biomass_source: dict[str, tuple[str, str]] = {}

    for res in model_results:
        model   = res["model"]
        bigg_id = res["bigg_id"]
        org_name = res.get("organism", bigg_id)

        # Identify biomass reaction: objective first, then name fallback
        for rxn in model.reactions:
            if rxn.objective_coefficient != 0:
                if rxn.id not in biomass_source:
                    biomass_source[rxn.id] = (bigg_id, org_name)

        # Fallback: look for 'biomass' in the reaction id
        if not any(
            rxn.objective_coefficient != 0 for rxn in model.reactions
        ):
            for rxn in model.reactions:
                if "biomass" in rxn.id.lower():
                    if rxn.id not in biomass_source:
                        biomass_source[rxn.id] = (bigg_id, org_name)
                    break

        for rxn in model.reactions:
            gpr = rxn.gene_reaction_rule.strip()

            if rxn.id not in rxn_obj:
                rxn_copy = rxn.copy()
                rxn_copy.gene_reaction_rule = gpr
                rxn_obj[rxn.id] = rxn_copy
                rxn_gprs[rxn.id] = [gpr] if gpr else []
            elif gpr:
                rxn_gprs[rxn.id].append(gpr)

    # Collapse GPRs and tag biomass reactions
    for rxn_id, rxn in rxn_obj.items():
        gprs = rxn_gprs.get(rxn_id, [])
        if len(gprs) > 1:
            rxn.gene_reaction_rule = " or ".join(f"({g})" for g in gprs)
        elif gprs:
            rxn.gene_reaction_rule = gprs[0]

        if rxn_id in biomass_source:
            src_bigg, src_org = biomass_source[rxn_id]
            if not rxn.notes:
                rxn.notes = {}
            rxn.notes["gemiz_biomass"]  = src_bigg
            rxn.notes["gemiz_organism"] = src_org

    n_bio = len(biomass_source)
    print(f"    {n_bio} biomass reactions tagged from {len(model_results)} models")

    universal.add_reactions(list(rxn_obj.values()))
    return universal


# ---------------------------------------------------------------------------
# MMseqs2 DB
# ---------------------------------------------------------------------------

def build_mmseqs_db(faa_path: Path) -> None:
    MMSEQS_DIR.mkdir(parents=True, exist_ok=True)
    db_prefix = str(MMSEQS_DIR / "db")

    try:
        from gemiz.utils.binaries import get_mmseqs_path
        mmseqs = str(get_mmseqs_path())
    except Exception:
        mmseqs = "mmseqs"

    cmd = [mmseqs, "createdb", str(faa_path), db_prefix]
    print(f"\nBuilding MMseqs2 DB ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: mmseqs createdb failed")
        for line in (result.stderr or "").splitlines()[:10]:
            print(f"    {line}")
        raise RuntimeError("MMseqs2 DB build failed")

    n = sum(1 for l in open(faa_path) if l.startswith(">"))
    print(f"  MMseqs2 DB ready: {n:,} sequences  ->  {db_prefix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gemiz universal reference database from all BiGG models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="BIGG_ID",
        default=None,
        help="Process only these BiGG model IDs (for testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all files even if cached.",
    )
    parser.add_argument(
        "--skip-mmseqs",
        action="store_true",
        help="Skip MMseqs2 DB build at the end.",
    )
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    ORG_CACHE.mkdir(parents=True, exist_ok=True)

    _load_nc_gcf_cache()
    if _nc_gcf_cache:
        print(f"Loaded NC->GCF cache: {len(_nc_gcf_cache)} entries")

    # ---- Step 1: Fetch BiGG model list ----
    try:
        all_models = fetch_bigg_model_list()
    except Exception as exc:
        print(f"ERROR: Could not fetch BiGG model list: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Step 2: Fetch details + resolve accessions ----
    print("\nResolving genome accessions ...")
    candidates: list[tuple[str, str, str]] = []  # (bigg_id, organism, gcf)

    only_set = set(args.only) if args.only else None

    for entry in all_models:
        bigg_id  = entry.get("bigg_id", "")
        organism = entry.get("organism", "")

        if only_set and bigg_id not in only_set:
            continue

        try:
            detail       = fetch_bigg_model_detail(bigg_id)
            genome_ref   = detail.get("genome_ref_string", "")
            gcf          = resolve_gcf_accession(genome_ref)
            time.sleep(0.1)   # be polite to NCBI/BiGG APIs
        except Exception as exc:
            print(f"  SKIP {bigg_id}: {exc}")
            continue

        if gcf:
            candidates.append((bigg_id, organism, gcf))
            print(f"  {bigg_id:<20}  {gcf}  ({organism[:40]})")
        else:
            print(f"  SKIP {bigg_id}: no genome accession (ref={genome_ref!r})")

    print(f"\n{len(candidates)} models with resolvable genome accessions")

    if not candidates:
        print("Nothing to process.")
        sys.exit(1)

    # ---- Steps 3-6: Per-model processing ----
    build_log: list[dict] = []
    model_results: list[dict] = []
    n_ok = 0
    n_fail = 0

    # Open output files in append mode so we can stream results
    faa_out   = open(FAA_OUT,  "w", encoding="utf-8")
    gpr_writer_handle = open(CSV_OUT, "w", newline="", encoding="utf-8")
    gpr_writer = csv.DictWriter(
        gpr_writer_handle,
        fieldnames=["bigg_id", "reaction_id", "gpr", "lb", "ub"],
    )
    gpr_writer.writeheader()

    try:
        for idx, (bigg_id, organism, gcf) in enumerate(candidates, 1):
            label = f"[{idx:>3}/{len(candidates)}] {bigg_id:<12}  {organism[:40]}"
            print(f"\n{label}")

            org_dir = ORG_CACHE / bigg_id
            t0 = time.perf_counter()

            try:
                res = process_model(bigg_id, gcf, org_dir, force=args.force)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"      FAILED: {exc}")
                build_log.append({
                    "bigg_id": bigg_id, "organism": organism, "gcf": gcf,
                    "status": "error", "error": str(exc),
                    "elapsed_s": round(elapsed, 1),
                })
                n_fail += 1
                continue

            elapsed = time.perf_counter() - t0

            # Stream proteins to FAA
            for (org_name, tag, acc, seq) in res["proteins"]:
                faa_out.write(f">{org_name}|{tag}|{acc}\n")
                for i in range(0, len(seq), 60):
                    faa_out.write(seq[i:i+60] + "\n")

            # Stream GPR rows to CSV
            gpr_writer.writerows(res["gpr_rows"])

            build_log.append({
                "bigg_id":   bigg_id,
                "organism":  organism,
                "gcf":       gcf,
                "status":    "ok",
                "n_reactions":         res["n_reactions"],
                "n_gpr":               res["n_gpr"],
                "n_proteins_written":  res["n_proteins_written"],
                "mapped_pct":          res["mapped_pct"],
                "elapsed_s":           round(elapsed, 1),
            })
            model_results.append(res)
            n_ok += 1

            # Flush periodically
            if n_ok % 5 == 0:
                faa_out.flush()
                gpr_writer_handle.flush()

    finally:
        faa_out.close()
        gpr_writer_handle.close()

    # ---- Step 7: Universal template SBML ----
    if model_results:
        print(f"\nBuilding universal template SBML ...")
        template = build_template(model_results)
        import cobra
        cobra.io.write_sbml_model(template, str(XML_OUT))
        size_mb = XML_OUT.stat().st_size / 1e6
        print(f"  {len(template.reactions):,} unique reactions  ->  {XML_OUT}  ({size_mb:.1f} MB)")
    else:
        print("\nNo models succeeded — skipping template build.")

    # ---- Write build log ----
    with open(LOG_OUT, "w") as f:
        json.dump({"models": build_log}, f, indent=2)
    print(f"Build log: {LOG_OUT}")

    # ---- Step 8: MMseqs2 DB ----
    if not args.skip_mmseqs and FAA_OUT.exists():
        try:
            build_mmseqs_db(FAA_OUT)
        except RuntimeError:
            print("  WARNING: MMseqs2 DB build failed (non-fatal).")
    elif args.skip_mmseqs:
        print("\nSkipping MMseqs2 DB (--skip-mmseqs).")

    # ---- Summary ----
    total_proteins = sum(1 for l in open(FAA_OUT) if l.startswith(">")) if FAA_OUT.exists() else 0
    total_rxns = len(template.reactions) if model_results else 0

    print(f"\n{'='*50}")
    print(f"  === Build complete ===")
    print(f"  Models processed:  {n_ok:>4}/{len(candidates)}")
    print(f"  Models failed:     {n_fail:>4}/{len(candidates)}")
    print(f"  Total proteins:    {total_proteins:>8,}")
    print(f"  Unique reactions:  {total_rxns:>8,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
