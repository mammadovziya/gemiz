"""Step 6 -- Full reconstruction pipeline.

Wires Steps 1-5 into a single function call::

    genome.fna -> pyrodigal -> MMseqs2 -> [ESM C] -> scoring -> MILP -> model.xml

Called by the CLI ``gemiz carve`` command.
"""

from __future__ import annotations

import time
from pathlib import Path


def run_full_pipeline(
    genome_fna: str,
    output_xml: str,
    universal_model_path: str,
    reference_faa_path: str,
    feature_table_path: str | None = None,
    esm_db_path: str | None = None,
    high_conf: float = 50.0,
    low_conf: float = 30.0,
    min_growth: float = 0.1,
    use_esm: bool = False,
    threads: int = 4,
    sensitivity: float = 7.5,
) -> dict:
    """Run the full GEM reconstruction pipeline.

    Parameters
    ----------
    genome_fna
        Path to the input genome FASTA (.fna).
    output_xml
        Path where the carved SBML model will be written.
    universal_model_path
        Path to the universal/template model (.xml).
    reference_faa_path
        Path to reference protein FASTA (.faa) for MMseqs2 alignment.
    feature_table_path
        Path to NCBI feature table for accession -> locus_tag mapping.
    esm_db_path
        Path to a pre-built ESM C reference database directory containing
        ``reference.faiss`` and ``reference_ids.json``.  When provided the
        pipeline skips the expensive index-generation step.  When *None*,
        the index is generated into the per-run work directory.
    high_conf
        Identity %% above which MMseqs2 is fully trusted.
    low_conf
        Identity %% below which sequence alignment is unreliable.
    min_growth
        Minimum biomass flux required for the carved model.
    use_esm
        Whether to use ESM C embeddings for low-confidence proteins.
    threads
        CPU threads for MMseqs2.
    sensitivity
        MMseqs2 sensitivity (4.0=fast, 7.5=balanced, 9.5=sensitive).

    Returns
    -------
    dict
        Summary with timing and model statistics.
    """
    results: dict = {}
    t_total = time.perf_counter()

    genome = Path(genome_fna)
    output = Path(output_xml)
    output.parent.mkdir(parents=True, exist_ok=True)

    # work directory next to output
    work_dir = output.parent / f".gemiz_{genome.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Gene calling (pyrodigal)
    # ------------------------------------------------------------------
    print("\n[1/6] Calling genes...")
    t0 = time.perf_counter()

    from gemiz.pipeline.prodigal import call_genes

    faa_path = call_genes(str(genome), str(work_dir / "genes"))

    # count proteins
    n_proteins = sum(1 for line in open(faa_path) if line.startswith(">"))
    elapsed = time.perf_counter() - t0

    print(f"      pyrodigal -> {n_proteins:,} proteins ({elapsed:.1f}s)")
    results["n_proteins"] = n_proteins
    results["step1_time"] = elapsed

    # ------------------------------------------------------------------
    # Step 2: Protein alignment (MMseqs2)
    # ------------------------------------------------------------------
    print("\n[2/6] Aligning proteins...")
    t0 = time.perf_counter()

    from gemiz.pipeline.alignment import (
        align_proteins,
        classify_proteins,
        parse_alignment,
    )

    tsv_path = align_proteins(
        query_faa=faa_path,
        reference_db=reference_faa_path,
        output_dir=str(work_dir / "alignment"),
        threads=threads,
        sensitivity=sensitivity,
    )
    mmseqs_hits = parse_alignment(tsv_path)

    # classify into high/low confidence
    all_prot_ids = [
        line[1:].split()[0]
        for line in open(faa_path)
        if line.startswith(">")
    ]
    classified = classify_proteins(all_prot_ids, mmseqs_hits, high_conf)
    stats = classified["stats"]
    elapsed = time.perf_counter() - t0

    print(f"      MMseqs2 -> {stats['high_confidence']:,} high confidence "
          f"({stats['high_confidence_pct']:.1f}%)")
    print(f"                 {stats['low_confidence']:,} low confidence "
          f"-> {'ESM C' if use_esm else 'skipped'}")
    print(f"      ({elapsed:.1f}s)")
    results["n_high_conf"] = stats["high_confidence"]
    results["n_low_conf"] = stats["low_confidence"]
    results["step2_time"] = elapsed

    # ------------------------------------------------------------------
    # Step 3: ESM C embeddings (optional)
    # ------------------------------------------------------------------
    esmc_hits: dict = {}

    if use_esm and classified["low_confidence"]:
        print("\n[3/6] Embedding low-confidence proteins...")
        t0 = time.perf_counter()

        try:
            from gemiz.embedding.esm import embed_proteins, load_embeddings
            from gemiz.embedding.database import search_similar

            # Read sequences for low-confidence proteins
            low_conf_seqs = _read_fasta_subset(
                faa_path, set(classified["low_confidence"])
            )

            if low_conf_seqs:
                emb_path = embed_proteins(
                    low_conf_seqs,
                    output_path=str(work_dir / "esm_embeddings.npz"),
                )
                emb_ids, emb_matrix = load_embeddings(emb_path)

                # Resolve reference FAISS index
                if esm_db_path is not None:
                    ref_db_dir = Path(esm_db_path)
                else:
                    ref_db_dir = work_dir / "esm_ref_db"

                faiss_path = ref_db_dir / "reference.faiss"
                ids_path = ref_db_dir / "reference_ids.json"

                if faiss_path.exists() and ids_path.exists():
                    print(f"      Using cached ESM C database: {ref_db_dir}")
                else:
                    from gemiz.embedding.database import generate_reference_db
                    print(f"      Building ESM C reference database...")
                    generate_reference_db(
                        reference_faa_path, str(ref_db_dir)
                    )

                esmc_hits = search_similar(
                    emb_ids, emb_matrix,
                    str(faiss_path), str(ids_path),
                )

            n_rescued = sum(1 for h in esmc_hits.values() if h)
            elapsed = time.perf_counter() - t0
            print(f"      ESM C 600M -> {n_rescued}/{len(low_conf_seqs)} "
                  f"rescued ({elapsed:.1f}s)")
            results["n_esm_rescued"] = n_rescued
            results["step3_time"] = elapsed

        except ImportError:
            elapsed = time.perf_counter() - t0
            print("      ESM C not available (install gemiz[embeddings]). "
                  "Skipping.")
            results["n_esm_rescued"] = 0
            results["step3_time"] = elapsed
    else:
        if use_esm:
            print("\n[3/6] No low-confidence proteins to embed. Skipping.")
        else:
            print("\n[3/6] ESM C disabled (--no-esm). Skipping.")
        results["n_esm_rescued"] = 0
        results["step3_time"] = 0.0

    # ------------------------------------------------------------------
    # Step 4: Reaction scoring
    # ------------------------------------------------------------------
    print("\n[4/6] Computing reaction scores...")
    t0 = time.perf_counter()

    from gemiz.reconstruction.scoring import (
        compute_reaction_scores,
        load_universal_model,
    )

    universal = load_universal_model(universal_model_path)

    reaction_scores = compute_reaction_scores(
        universal,
        mmseqs_hits,
        esmc_hits,
        high_conf=high_conf,
        low_conf=low_conf,
        feature_table_path=feature_table_path,
        reference_faa_path=reference_faa_path,
    )

    n_evidence = sum(1 for s in reaction_scores.values() if s > 0)
    n_no_evidence = sum(1 for s in reaction_scores.values() if s < 0)
    n_spontaneous = sum(1 for s in reaction_scores.values() if s == 0.0)
    elapsed = time.perf_counter() - t0

    print(f"      {n_evidence:,} reactions with evidence")
    print(f"      {n_no_evidence:,} reactions with no evidence")
    print(f"      {n_spontaneous:,} spontaneous reactions ({elapsed:.1f}s)")
    results["n_scored_positive"] = n_evidence
    results["n_scored_negative"] = n_no_evidence
    results["n_spontaneous"] = n_spontaneous
    results["step4_time"] = elapsed

    # ------------------------------------------------------------------
    # Step 5: MILP carving (HiGHS)
    # ------------------------------------------------------------------
    print("\n[5/6] Carving model...")
    t0 = time.perf_counter()

    from gemiz.reconstruction.carving import carve_model, verify_model

    carved = carve_model(universal, reaction_scores, min_growth=min_growth)
    elapsed = time.perf_counter() - t0

    print(f"      HiGHS MILP -> {len(carved.reactions)} reactions "
          f"selected ({elapsed:.1f}s)")
    results["step5_time"] = elapsed

    # ------------------------------------------------------------------
    # Step 6: Save model
    # ------------------------------------------------------------------
    print("\n[6/6] Saving model...")
    t0 = time.perf_counter()

    import cobra

    cobra.io.write_sbml_model(carved, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    elapsed = time.perf_counter() - t0

    print(f"      {output.name} ({size_mb:.1f} MB)")
    results["step6_time"] = elapsed
    results["output_size_mb"] = size_mb

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    verification = verify_model(carved)
    results["model"] = carved
    results["verification"] = verification
    results["n_reactions"] = len(carved.reactions)
    results["n_metabolites"] = len(carved.metabolites)
    results["n_genes"] = len(carved.genes)
    results["growth_rate"] = verification["growth_rate"]
    results["can_grow"] = verification["can_grow"]
    results["total_time"] = time.perf_counter() - t_total

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_fasta_subset(
    faa_path: str, keep_ids: set[str]
) -> dict[str, str]:
    """Read a FASTA file and return only sequences whose IDs are in keep_ids."""
    seqs: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with open(faa_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_id is not None and current_id in keep_ids:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        # last sequence
        if current_id is not None and current_id in keep_ids:
            seqs[current_id] = "".join(current_seq)

    return seqs
