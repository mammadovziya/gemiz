#!/usr/bin/env python3
"""ESM C degradation test: does ESM rescue reactions lost by weak MMseqs2 alignment?

Method
------
1. Run (or load cached) MMseqs2 alignment of E. coli proteins against iML1515.
2. Split proteins into "high-confidence" (identity >= 50%) and the rest.
3. For each degradation level D in {10, 30, 50, 70, 90}%:
     - Randomly mask D% of high-confidence hits (set identity = 0).
     - Score reactions using:
         A. Random baseline: assign Uniform[0,1] scores to masked proteins.
         B. ESM C: use FAISS similarity to score masked proteins.
     - Carve a model (MILP or score-threshold) and measure reaction recall
       vs iML1515 gold standard.
4. Print a table: Coverage% | Random recall | ESM C recall | Delta

Cached files (reused if present)
---------------------------------
  MMseqs2 TSV : <work_dir>/alignment/hits.tsv
  ESM embeddings: <work_dir>/esm_embeddings.npz
  ESM ref DB  : data/reference/iml1515_esm_db/

Run
---
  python scripts/esm_degradation_test.py [--genome data/genomes/ecoli_k12.fna]

"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_GENOME   = "data/GCF_000005845.2_genomic.fna"
REFERENCE_FAA    = "data/reference/iML1515_proteins.faa"
FEATURE_TABLE    = "data/reference/ecoli_feature_table.txt"
GOLD_STANDARD    = "data/universal/iML1515.xml"
WORK_DIR         = Path("data/.gemiz_degradation_test")
ESM_DB_DIR       = Path("data/reference/iml1515_esm_db")

DEGRADATION_LEVELS = [10, 30, 50, 70, 90]   # % of high-conf hits to mask
HIGH_CONF_THRESHOLD = 50.0                   # identity % for "high confidence"
N_RANDOM_SEEDS      = 3                      # average random baseline over N seeds


# ---------------------------------------------------------------------------
# Step 1 — Gene calling (cached)
# ---------------------------------------------------------------------------

def ensure_genes(genome_fna: str) -> str:
    """Call genes with pyrodigal, returning path to .faa. Cached."""
    from gemiz.pipeline.prodigal import call_genes

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    faa_path = call_genes(genome_fna, str(WORK_DIR / "genes"))
    n = sum(1 for l in open(faa_path) if l.startswith(">"))
    print(f"  Genes: {n:,} proteins ({faa_path})")
    return faa_path


# ---------------------------------------------------------------------------
# Step 2 — MMseqs2 alignment (cached)
# ---------------------------------------------------------------------------

def ensure_mmseqs_hits(faa_path: str, threads: int) -> dict:
    """Run MMseqs2 against iML1515, return parsed hits dict. Cached."""
    from gemiz.pipeline.alignment import align_proteins, parse_alignment

    tsv_path = WORK_DIR / "alignment" / "hits.tsv"
    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        print(f"  MMseqs2: using cached {tsv_path}")
    else:
        print(f"  MMseqs2: aligning against {REFERENCE_FAA}...")
        t0 = time.perf_counter()
        align_proteins(
            query_faa=faa_path,
            reference_db=REFERENCE_FAA,
            output_dir=str(WORK_DIR / "alignment"),
            threads=threads,
            sensitivity=7.5,
        )
        print(f"    done in {time.perf_counter()-t0:.1f}s")

    hits = parse_alignment(str(tsv_path))
    n_hits = sum(len(v) for v in hits.values())
    print(f"  MMseqs2 hits: {len(hits):,} query proteins, {n_hits:,} total hits")
    return hits


# ---------------------------------------------------------------------------
# Step 3 — ESM C embeddings (cached)
# ---------------------------------------------------------------------------

def ensure_esm_embeddings(faa_path: str, low_conf_ids: list[str]) -> tuple[list, object]:
    """Embed low-confidence proteins with ESM C 600M. Cached."""
    from gemiz.embedding.esm import embed_proteins, load_embeddings

    npz_path = WORK_DIR / "esm_embeddings.npz"
    if npz_path.exists():
        print(f"  ESM C: using cached {npz_path}")
    else:
        print(f"  ESM C: embedding {len(low_conf_ids):,} proteins...")
        t0 = time.perf_counter()
        seqs = _read_fasta_subset(faa_path, set(low_conf_ids))
        embed_proteins(seqs, output_path=str(npz_path))
        print(f"    done in {time.perf_counter()-t0:.1f}s")

    ids, matrix = load_embeddings(str(npz_path))
    print(f"  ESM embeddings: {len(ids):,} proteins, shape {matrix.shape}")
    return ids, matrix


def ensure_esm_db() -> tuple[Path, Path]:
    """Build ESM reference DB from iML1515_proteins.faa. Cached."""
    faiss_path = ESM_DB_DIR / "reference.faiss"
    ids_path   = ESM_DB_DIR / "reference_ids.json"

    if faiss_path.exists() and ids_path.exists():
        print(f"  ESM DB: using cached {ESM_DB_DIR}")
    else:
        from gemiz.embedding.database import generate_reference_db
        print(f"  ESM DB: building from {REFERENCE_FAA}...")
        t0 = time.perf_counter()
        ESM_DB_DIR.mkdir(parents=True, exist_ok=True)
        generate_reference_db(REFERENCE_FAA, str(ESM_DB_DIR))
        print(f"    done in {time.perf_counter()-t0:.1f}s")

    return faiss_path, ids_path


# ---------------------------------------------------------------------------
# Step 4 — Scoring helpers
# ---------------------------------------------------------------------------

def _split_confidence(hits: dict) -> tuple[list[str], list[str]]:
    """Split query protein IDs into high / low confidence groups."""
    high, low = [], []
    for pid, hit_list in hits.items():
        best_identity = max((h["identity"] for h in hit_list), default=0.0)
        if best_identity >= HIGH_CONF_THRESHOLD:
            high.append(pid)
        else:
            low.append(pid)
    return high, low


def _degrade_hits(
    hits: dict,
    high_conf_ids: list[str],
    pct: float,
    seed: int,
) -> tuple[dict, list[str]]:
    """Zero out `pct`% of high-confidence hits. Returns degraded hits + masked IDs."""
    rng = random.Random(seed)
    n_mask = int(len(high_conf_ids) * pct / 100)
    masked = set(rng.sample(high_conf_ids, n_mask))

    degraded: dict = {}
    for pid, hit_list in hits.items():
        if pid in masked:
            degraded[pid] = []           # erase all hits for this protein
        else:
            degraded[pid] = hit_list
    return degraded, sorted(masked)


def _random_esmc_hits(masked_ids: list[str], seed: int) -> dict:
    """Return fake ESM C hits with random similarity scores."""
    rng = random.Random(seed + 1000)
    # We can't return actual ref_ids without knowing the mapping,
    # so we assign scores via protein_scores instead (handled in scoring path).
    # Return empty — caller injects random scores directly.
    return {}


def _compute_protein_scores_random(masked_ids: list[str], seed: int) -> dict[str, float]:
    """Random baseline: uniform [0,1] scores for masked proteins."""
    rng = random.Random(seed + 2000)
    return {pid: rng.random() for pid in masked_ids}


def _compute_protein_scores_esmc(
    masked_ids: list[str],
    emb_ids: list[str],
    emb_matrix: object,
    faiss_path: Path,
    ids_path: Path,
    id_map: dict[str, str],
) -> dict[str, float]:
    """ESM C scores for masked proteins via FAISS search."""
    import numpy as np
    from gemiz.embedding.database import search_similar

    # Filter to masked IDs that have embeddings
    emb_id_set = set(emb_ids)
    subset_ids = [pid for pid in masked_ids if pid in emb_id_set]
    if not subset_ids:
        return {}

    # Extract embedding rows for this subset
    id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
    indices = [id_to_idx[pid] for pid in subset_ids]
    sub_matrix = np.array(emb_matrix)[indices]

    esmc_raw = search_similar(subset_ids, sub_matrix,
                              str(faiss_path), str(ids_path))

    # Translate ref accessions to locus_tags, take best per tag
    scores: dict[str, float] = {}
    for pid, hit_list in esmc_raw.items():
        for h in hit_list:
            gene_id = id_map.get(h["ref_id"], h["ref_id"])
            if h["similarity"] > scores.get(gene_id, 0.0):
                scores[gene_id] = h["similarity"]

    return scores


# ---------------------------------------------------------------------------
# Step 5 — Reaction recall
# ---------------------------------------------------------------------------

def _score_reactions(
    model,
    hits: dict,
    extra_protein_scores: dict[str, float],
    id_map: dict[str, str],
) -> dict[str, float]:
    """Score reactions using MMseqs2 hits + any extra per-gene scores."""
    from gemiz.reconstruction.scoring import (
        evaluate_gpr_rule,
        extract_gpr_associations,
        merge_protein_scores,
        NO_EVIDENCE_SCORE,
    )

    gpr_assoc = extract_gpr_associations(model)

    # Build gene score map from MMseqs2 hits
    gene_scores: dict[str, float] = {}
    for _pid, hit_list in hits.items():
        for h in hit_list:
            gene_id = id_map.get(h["ref_id"], h["ref_id"])
            score = merge_protein_scores(h["identity"], 0.0)
            if score > gene_scores.get(gene_id, 0.0):
                gene_scores[gene_id] = score

    # Merge extra scores (random or ESM C) — use max
    for gene_id, s in extra_protein_scores.items():
        if s > gene_scores.get(gene_id, 0.0):
            gene_scores[gene_id] = s

    reaction_scores: dict[str, float] = {}
    for rxn_id, assoc in gpr_assoc.items():
        reaction_scores[rxn_id] = evaluate_gpr_rule(
            assoc["rule"], gene_scores, assoc["type"]
        )
    return reaction_scores


def _recall_at_threshold(
    reaction_scores: dict[str, float],
    gold_ids: set[str],
    threshold: float = 0.0,
) -> float:
    """Fraction of gold reactions with score > threshold."""
    selected = {rid for rid, s in reaction_scores.items() if s > threshold}
    tp = len(selected & gold_ids)
    return tp / len(gold_ids) if gold_ids else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_fasta_subset(faa_path: str, keep_ids: set[str]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur_id: str | None = None
    cur_seq: list[str] = []
    with open(faa_path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id and cur_id in keep_ids:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id and cur_id in keep_ids:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


def _load_id_map(feature_table: str) -> dict[str, str]:
    from gemiz.reconstruction.scoring import parse_reference_id_map
    return parse_reference_id_map(feature_table_path=feature_table)


def _load_gold_ids(gold_path: str) -> set[str]:
    import cobra
    gold = cobra.io.read_sbml_model(gold_path)
    return {r.id for r in gold.reactions
            if not r.id.startswith("EX_") and not r.id.startswith("DM_")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome", default=DEFAULT_GENOME,
                        help=f"Genome FASTA (default: {DEFAULT_GENOME})")
    parser.add_argument("--threads", type=int, default=4,
                        help="MMseqs2 threads (default: 4)")
    parser.add_argument("--no-esm", action="store_true",
                        help="Skip ESM C (random baseline only)")
    args = parser.parse_args()

    print("=" * 60)
    print("ESM C degradation test")
    print("=" * 60)

    # -- Setup --
    import cobra

    print("\n[1/5] Calling genes...")
    faa_path = ensure_genes(args.genome)

    print("\n[2/5] Running MMseqs2 alignment...")
    full_hits = ensure_mmseqs_hits(faa_path, args.threads)

    high_conf_ids, low_conf_ids = _split_confidence(full_hits)
    print(f"  High-confidence proteins: {len(high_conf_ids):,} (>= {HIGH_CONF_THRESHOLD}% identity)")
    print(f"  Low-confidence proteins:  {len(low_conf_ids):,}")

    print("\n[3/5] Loading template model and gold standard...")
    template = cobra.io.read_sbml_model(GOLD_STANDARD)
    gold_ids  = _load_gold_ids(GOLD_STANDARD)
    id_map    = _load_id_map(FEATURE_TABLE)
    print(f"  Template: {len(template.reactions)} reactions")
    print(f"  Gold (metabolic only): {len(gold_ids)} reactions")

    # Baseline recall with full MMseqs2 (no degradation)
    full_scores = _score_reactions(template, full_hits, {}, id_map)
    baseline_recall = _recall_at_threshold(full_scores, gold_ids)
    print(f"  Baseline recall (full MMseqs2, threshold>0): {baseline_recall:.3f}")

    # -- ESM C setup (skip if --no-esm) --
    emb_ids, emb_matrix, faiss_path, ids_path = None, None, None, None
    if not args.no_esm:
        print("\n[4/5] Preparing ESM C embeddings...")
        # Embed all proteins that aren't high-confidence
        # (we'll need embeddings for the masked ones at each degradation level)
        # Simplest: embed all proteins that could be masked = high_conf_ids
        all_to_embed = high_conf_ids + low_conf_ids
        emb_ids, emb_matrix = ensure_esm_embeddings(faa_path, all_to_embed)
        faiss_path, ids_path = ensure_esm_db()
    else:
        print("\n[4/5] ESM C disabled (--no-esm), skipping embeddings.")

    # -- Degradation sweep --
    print("\n[5/5] Running degradation sweep...")

    col_w = 14
    header = (
        f"  {'Coverage%':>10}  "
        f"{'MMseqs2':>{col_w}}  "
        f"{'Random':>{col_w}}  "
        + (f"{'ESM C':>{col_w}}  {'Delta (ESM-Rand)':>{col_w}}" if not args.no_esm else "")
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    rows = []
    for pct in DEGRADATION_LEVELS:
        coverage = 100 - pct   # % of high-conf hits retained

        # Average over multiple random seeds
        mmseqs_recalls = []
        random_recalls = []
        esmc_recalls   = []

        for seed in range(N_RANDOM_SEEDS):
            degraded_hits, masked_ids = _degrade_hits(full_hits, high_conf_ids, pct, seed)

            # A — MMseqs2 only (no rescue of masked proteins)
            sc_mmseqs = _score_reactions(template, degraded_hits, {}, id_map)
            mmseqs_recalls.append(_recall_at_threshold(sc_mmseqs, gold_ids))

            # B — Random rescue
            rand_scores = _compute_protein_scores_random(masked_ids, seed)
            # Map protein IDs → gene IDs via id_map
            rand_gene_scores = {id_map.get(pid, pid): s
                                for pid, s in rand_scores.items()}
            sc_rand = _score_reactions(template, degraded_hits, rand_gene_scores, id_map)
            random_recalls.append(_recall_at_threshold(sc_rand, gold_ids))

            # C — ESM C rescue
            if not args.no_esm and emb_ids is not None:
                esmc_gene_scores = _compute_protein_scores_esmc(
                    masked_ids, emb_ids, emb_matrix,
                    faiss_path, ids_path, id_map,
                )
                sc_esmc = _score_reactions(template, degraded_hits, esmc_gene_scores, id_map)
                esmc_recalls.append(_recall_at_threshold(sc_esmc, gold_ids))

        mmseqs_r = sum(mmseqs_recalls) / len(mmseqs_recalls)
        random_r = sum(random_recalls) / len(random_recalls)
        esmc_r   = sum(esmc_recalls)   / len(esmc_recalls) if esmc_recalls else None
        delta    = (esmc_r - random_r) if esmc_r is not None else None

        rows.append((coverage, mmseqs_r, random_r, esmc_r, delta))

        line = (
            f"  {coverage:>10}%  "
            f"{mmseqs_r:>{col_w}.3f}  "
            f"{random_r:>{col_w}.3f}  "
        )
        if not args.no_esm:
            esmc_str  = f"{esmc_r:.3f}" if esmc_r is not None else "n/a"
            delta_str = (f"+{delta:.3f}" if delta and delta >= 0 else f"{delta:.3f}") if delta is not None else "n/a"
            line += f"{esmc_str:>{col_w}}  {delta_str:>{col_w}}"
        print(line)

    print(sep)
    print(f"  Baseline (100% coverage, full MMseqs2): {baseline_recall:.3f}")
    print(sep)

    if not args.no_esm and rows:
        # Summary: at which degradation level ESM C first beats random by > 1%
        improvements = [(cov, d) for cov, _, _, _, d in rows if d is not None and d > 0.01]
        if improvements:
            first_cov, first_d = improvements[-1]   # lowest coverage where ESM helps
            print(f"\n  ESM C outperforms random baseline by >1% at coverage <= {first_cov}%")
        else:
            print("\n  ESM C did not outperform random baseline by >1% at any coverage level")


if __name__ == "__main__":
    main()
