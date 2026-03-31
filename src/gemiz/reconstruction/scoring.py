"""Step 4 — Reaction scoring.

Translates per-protein alignment/embedding evidence into per-reaction
confidence scores that feed directly into the HiGHS MILP objective
in Step 5.

Score range
-----------
  (0, 1]   genetic evidence found (higher → stronger)
  0.0      spontaneous / no GPR  (neutral in MILP)
 -1.0      enzyme reaction with no evidence (penalised)

Thresholds based on Rost (1999) "twilight zone of sequence alignment":
  ≥ 50 % identity → same function very likely
  30–50 %         → twilight zone, blend sequence + structure signal
  < 30 %          → sequence alignment unreliable, rely on ESM C
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import cobra

# Established bioinformatics thresholds
HIGH_CONF_THRESHOLD = 50.0   # above: trust MMseqs2 fully
LOW_CONF_THRESHOLD  = 30.0   # below: sequence alignment unreliable
NO_EVIDENCE_SCORE   = -1.0   # penalty in MILP for unsupported enzyme rxns


# ---------------------------------------------------------------------------
# Score merging
# ---------------------------------------------------------------------------

def merge_protein_scores(
    mmseqs_identity: float,
    esmc_similarity: float,
    high_conf: float = HIGH_CONF_THRESHOLD,
    low_conf: float = LOW_CONF_THRESHOLD,
) -> float:
    """Adaptive weighted combination of MMseqs2 and ESM C scores.

    Weights interpolate smoothly through the twilight zone (30–50 %).

    ≥ 50 % identity   → trust MMseqs2 fully  (identity / 100)
    30–50 %           → blend (weight ramps linearly)
    > 0 but < 30 %    → ESM C dominates (0.6) / MMseqs2 (0.4)
    0 (no hit)        → ESM C only

    All inputs and outputs in [0, 1].
    """
    identity_norm = mmseqs_identity / 100.0

    if mmseqs_identity >= high_conf:
        return identity_norm

    if mmseqs_identity >= low_conf:
        # linear ramp: at low_conf→0.4 mmseqs weight, at high_conf→1.0
        t = (mmseqs_identity - low_conf) / (high_conf - low_conf)
        w_mmseqs = 0.4 + 0.6 * t
        w_esmc   = 1.0 - w_mmseqs
        return w_mmseqs * identity_norm + w_esmc * esmc_similarity

    if mmseqs_identity > 0:
        return 0.4 * identity_norm + 0.6 * esmc_similarity

    # no MMseqs2 hit at all
    return esmc_similarity


# ---------------------------------------------------------------------------
# Universal model loading
# ---------------------------------------------------------------------------

def load_universal_model(model_path: str) -> cobra.Model:
    """Load an SBML metabolic model (universal template or curated).

    Works with any COBRApy-readable SBML (.xml) file.
    """
    print("[gemiz] Loading universal model...")
    model = cobra.io.read_sbml_model(model_path)
    print(
        f"[gemiz] Universal model: {len(model.reactions)} reactions, "
        f"{len(model.metabolites)} metabolites, "
        f"{len(model.genes)} genes"
    )
    return model


# ---------------------------------------------------------------------------
# GPR extraction
# ---------------------------------------------------------------------------

_GENE_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def extract_gpr_associations(model: cobra.Model) -> dict[str, dict]:
    """Extract Gene-Protein-Reaction associations from a COBRA model.

    Returns
    -------
    dict ::

        {reaction_id: {"type": ..., "genes": [...], "rule": "..."}}

    where *type* is one of ``"single"``, ``"isozyme"``, ``"complex"``,
    ``"mixed"``, or ``"no_gpr"``.
    """
    print("[gemiz] Extracting GPR associations...")
    assoc: dict[str, dict] = {}

    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        genes = sorted(g.id for g in rxn.genes)

        if not rule:
            assoc[rxn.id] = {"type": "no_gpr", "genes": [], "rule": ""}
        elif len(genes) == 1:
            assoc[rxn.id] = {"type": "single", "genes": genes, "rule": rule}
        elif " and " in rule and " or " not in rule:
            assoc[rxn.id] = {"type": "complex", "genes": genes, "rule": rule}
        elif " or " in rule and " and " not in rule:
            assoc[rxn.id] = {"type": "isozyme", "genes": genes, "rule": rule}
        else:
            assoc[rxn.id] = {"type": "mixed", "genes": genes, "rule": rule}

    # counts
    types = {}
    for a in assoc.values():
        types[a["type"]] = types.get(a["type"], 0) + 1

    enzyme = len(assoc) - types.get("no_gpr", 0)
    no_gpr = types.get("no_gpr", 0)
    print(f"[gemiz] Found: {enzyme} enzyme reactions, {no_gpr} no-GPR reactions")
    for t in ("single", "isozyme", "complex", "mixed"):
        if types.get(t):
            print(f"[gemiz]   {t:8s}: {types[t]}")

    return assoc


# ---------------------------------------------------------------------------
# Reference ID mapping (NCBI accession -> locus_tag)
# ---------------------------------------------------------------------------

_LOCUS_UNDERSCORE_RE = re.compile(r"^([A-Za-z]+)_(\d+)$")


def _locus_tag_variant(tag: str) -> str | None:
    """Return the underscore-variant of a locus tag, or None if no variant exists.

    ``BSU_00010`` -> ``BSU00010`` and vice versa.
    """
    m = _LOCUS_UNDERSCORE_RE.match(tag)
    if m:
        return m.group(1) + m.group(2)
    m2 = re.match(r"^([A-Za-z]+)(\d+)$", tag)
    if m2:
        return f"{m2.group(1)}_{m2.group(2)}"
    return None


def parse_reference_id_map(
    feature_table_path: str | Path | None = None,
    reference_faa_path: str | Path | None = None,
) -> dict[str, str]:
    """Build a mapping from NCBI protein accession to locus_tag.

    Strategy (in order):
    1. Parse NCBI feature table (tab-separated, CDS rows have both
       ``product_accession`` and ``locus_tag``).
    2. Parse reference FASTA headers for ``[locus_tag=...]`` patterns.
    3. Return empty dict (caller falls back to using accessions as-is).

    Parameters
    ----------
    feature_table_path:
        Path to NCBI feature table (e.g. ``ecoli_feature_table.txt``).
    reference_faa_path:
        Path to reference protein FASTA (``.faa``). Headers may contain
        ``[locus_tag=b0001]`` annotations.

    Returns
    -------
    dict
        ``{"NP_414542.1": "b0001", ...}``
    """
    id_map: dict[str, str] = {}

    # Strategy 0: universal proteins FAA with namespaced headers
    # Header format: >org|locus_tag|accession  (built by build_universal_db.py)
    if reference_faa_path is not None:
        faa = Path(reference_faa_path)
        if faa.exists():
            with open(faa, encoding="utf-8") as f:
                for line in f:
                    if not line.startswith(">"):
                        continue
                    header = line[1:].split()[0]  # e.g. iML1515|b1779|NP_416370.1
                    parts = header.split("|")
                    if len(parts) == 3:
                        _org, locus_tag, accession = parts
                        if accession and locus_tag:
                            id_map[accession] = locus_tag
            if id_map:
                print(f"[gemiz] ID map: {len(id_map)} accession -> locus_tag "
                      f"entries from universal FAA headers")
                return id_map

    # Strategy 1: NCBI feature table
    if feature_table_path is not None:
        ft = Path(feature_table_path)
        if ft.exists():
            with open(ft, encoding="utf-8") as f:
                f.readline()  # skip header
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) < 17:
                        continue
                    feature = cols[0]
                    if feature != "CDS":
                        continue
                    product_acc = cols[10].strip()
                    locus_tag = cols[16].strip()
                    if product_acc and locus_tag:
                        id_map[product_acc] = locus_tag
            if id_map:
                print(f"[gemiz] ID map: {len(id_map)} accession -> locus_tag "
                      f"entries from feature table")
                return id_map

    # Strategy 2: parse FASTA headers for [locus_tag=...] pattern
    if reference_faa_path is not None:
        faa = Path(reference_faa_path)
        if faa.exists():
            locus_re = re.compile(r"\[locus_tag=([^\]]+)\]")
            with open(faa, encoding="utf-8") as f:
                for line in f:
                    if not line.startswith(">"):
                        continue
                    acc = line[1:].split()[0]  # e.g. NP_414542.1
                    m = locus_re.search(line)
                    if m:
                        id_map[acc] = m.group(1)
            if id_map:
                print(f"[gemiz] ID map: {len(id_map)} accession -> locus_tag "
                      f"entries from FASTA headers")
                return id_map

    if not id_map:
        print("[gemiz] WARNING: No ID mapping found. Accessions will be used as-is.")

    return id_map


def diagnose_id_mapping(
    mmseqs_hits: dict[str, list[dict]],
    gpr_associations: dict[str, dict],
    id_map: dict[str, str] | None = None,
    reference_faa_path: str | Path | None = None,
) -> None:
    """Print diagnostic information about ID overlap between alignments and GPR.

    Helps debug the common mismatch between NCBI accessions (``NP_414676.1``)
    in MMseqs2 output and locus tags (``b4025``) in GPR rules.
    """
    universal_mode = (
        reference_faa_path is not None
        and "universal" in str(reference_faa_path)
    )

    # Collect ref IDs from MMseqs2 (resolved to gene IDs)
    mmseqs_ref_ids: set[str] = set()
    mmseqs_raw_ids: set[str] = set()
    for hits in mmseqs_hits.values():
        for h in hits:
            mmseqs_raw_ids.add(h["ref_id"])
            mmseqs_ref_ids.add(
                _resolve_ref_id(h["ref_id"], id_map or {}, universal_mode)
            )

    # Collect gene IDs from GPR
    gpr_genes: set[str] = set()
    for a in gpr_associations.values():
        gpr_genes.update(a["genes"])

    # Direct overlap (after resolution)
    direct_overlap = mmseqs_ref_ids & gpr_genes

    print(f"\n[gemiz] === ID Mapping Diagnostic ===")
    if universal_mode:
        print(f"[gemiz]   Mode: universal (pipe-delimited ref IDs)")
    print(f"[gemiz]   MMseqs2 ref IDs:    {len(mmseqs_raw_ids)}")
    print(f"[gemiz]   Resolved gene IDs:  {len(mmseqs_ref_ids)}")
    print(f"[gemiz]   GPR gene IDs:       {len(gpr_genes)}")
    print(f"[gemiz]   Overlap:            {len(direct_overlap)} "
          f"({100*len(direct_overlap)/max(len(gpr_genes),1):.1f}% of GPR genes)")

    # Show sample IDs
    sample_resolved = sorted(mmseqs_ref_ids)[:5]
    sample_gpr = sorted(gpr_genes)[:5]
    print(f"[gemiz]   Sample resolved IDs: {sample_resolved}")
    print(f"[gemiz]   Sample GPR IDs:      {sample_gpr}")

    print(f"[gemiz] ================================\n")


# ---------------------------------------------------------------------------
# Protein score map
# ---------------------------------------------------------------------------

def _resolve_ref_id(rid: str, id_map: dict[str, str], universal_mode: bool) -> str:
    """Resolve a MMseqs2/ESM C ref_id to a GPR-compatible gene ID.

    In universal mode, ref_ids are pipe-delimited: ``org|locus_tag|accession``.
    Extract the locus_tag directly.  Otherwise, fall back to id_map lookup.
    """
    if universal_mode:
        parts = rid.split("|")
        if len(parts) == 3:
            return parts[1]  # locus_tag
    return id_map.get(rid, rid)


def build_protein_score_map(
    mmseqs_hits: dict[str, list[dict]],
    esmc_hits: dict[str, list[dict]],
    gpr_associations: dict[str, dict],
    high_conf: float = HIGH_CONF_THRESHOLD,
    low_conf: float = LOW_CONF_THRESHOLD,
    feature_table_path: str | Path | None = None,
    reference_faa_path: str | Path | None = None,
) -> dict[str, float]:
    """Score every reference gene that appears in any GPR rule.

    For each reference gene, find the best genome-protein that matched it
    (via MMseqs2 or ESM C) and compute a merged score.

    Parameters
    ----------
    mmseqs_hits:
        ``{genome_protein: [{"ref_id": ..., "identity": ..., ...}, ...]}``
    esmc_hits:
        ``{genome_protein: [{"ref_id": ..., "similarity": ...}, ...]}``
    gpr_associations:
        Output of ``extract_gpr_associations()``.
    feature_table_path:
        Path to NCBI feature table for accession -> locus_tag mapping.
    reference_faa_path:
        Path to reference FASTA (fallback for ID mapping from headers).

    Returns
    -------
    dict
        ``{gene_id: score}``  (score in [0, 1] or missing -> will default to
        ``NO_EVIDENCE_SCORE`` during GPR evaluation).
    """
    print("[gemiz] Building protein score map...")

    # Detect universal mode: ref FAA built by build_universal_db.py
    # has pipe-delimited headers (org|locus_tag|accession)
    universal_mode = False
    if reference_faa_path is not None:
        universal_mode = "universal" in str(reference_faa_path)

    # Load ID mapping (NCBI accession -> locus_tag) — used in non-universal mode
    id_map = parse_reference_id_map(feature_table_path, reference_faa_path)

    if universal_mode:
        print("[gemiz] Universal mode: parsing locus tags from pipe-delimited ref IDs")

    # Collect all genes that appear in any GPR rule
    all_gpr_genes: set[str] = set()
    for a in gpr_associations.values():
        all_gpr_genes.update(a["genes"])

    # Build reverse index: gene_id -> best (identity, similarity) from any
    # genome protein that hit it.
    ref_best_identity:   dict[str, float] = {}
    ref_best_similarity: dict[str, float] = {}

    for _pid, hits in mmseqs_hits.items():
        for h in hits:
            gene_id = _resolve_ref_id(h["ref_id"], id_map, universal_mode)
            if h["identity"] > ref_best_identity.get(gene_id, 0.0):
                ref_best_identity[gene_id] = h["identity"]

    for _pid, hits in esmc_hits.items():
        for h in hits:
            gene_id = _resolve_ref_id(h["ref_id"], id_map, universal_mode)
            if h["similarity"] > ref_best_similarity.get(gene_id, 0.0):
                ref_best_similarity[gene_id] = h["similarity"]

    # Score each GPR gene
    scores: dict[str, float] = {}
    n_high = n_blend = n_esmc = n_none = 0

    for gene_id in all_gpr_genes:
        identity   = ref_best_identity.get(gene_id, 0.0)
        similarity = ref_best_similarity.get(gene_id, 0.0)

        # Locus tag variant fallback: iYO844 uses BSU00010 but the
        # NCBI feature table lists BSU_00010.  Try the other form.
        if identity == 0 and similarity == 0:
            variant = _locus_tag_variant(gene_id)
            if variant:
                identity   = ref_best_identity.get(variant, 0.0)
                similarity = ref_best_similarity.get(variant, 0.0)

        if identity > 0 or similarity > 0:
            s = merge_protein_scores(identity, similarity, high_conf, low_conf)
            scores[gene_id] = s
            if identity >= high_conf:
                n_high += 1
            elif identity >= low_conf:
                n_blend += 1
            else:
                n_esmc += 1
        else:
            n_none += 1
            # leave absent -> GPR evaluator will use NO_EVIDENCE_SCORE

    total = len(all_gpr_genes)
    print(f"[gemiz] Reference proteins scored: {total}")
    if total:
        print(f"[gemiz]   High confidence (MMseqs2):  {n_high:>5} ({100*n_high/total:.1f}%)")
        print(f"[gemiz]   Blended (twilight zone):    {n_blend:>5} ({100*n_blend/total:.1f}%)")
        print(f"[gemiz]   ESM C only:                 {n_esmc:>5} ({100*n_esmc/total:.1f}%)")
        print(f"[gemiz]   No evidence:                {n_none:>5} ({100*n_none/total:.1f}%)")

    return scores


# ---------------------------------------------------------------------------
# GPR evaluation — recursive descent parser
# ---------------------------------------------------------------------------
#
#   Grammar
#   -------
#   expr      → or_expr
#   or_expr   → and_expr ('or' and_expr)*
#   and_expr  → atom ('and' atom)*
#   atom      → GENE_ID | '(' expr ')'
#
#   OR  → max(children)   isozymes: reaction works if ANY gene is present
#   AND → min(children)   complex:  ALL subunits needed, weakest link wins

class _GPRParser:
    _TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

    def __init__(self, text: str, scores: dict[str, float]) -> None:
        self.tokens = self._TOKEN_RE.findall(text)
        self.pos = 0
        self.scores = scores

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> float:
        if not self.tokens:
            return 0.0
        return self._or_expr()

    def _or_expr(self) -> float:
        val = self._and_expr()
        while self._peek() == "or":
            self._consume()
            val = max(val, self._and_expr())
        return val

    def _and_expr(self) -> float:
        val = self._atom()
        while self._peek() == "and":
            self._consume()
            val = min(val, self._atom())
        return val

    def _atom(self) -> float:
        if self._peek() == "(":
            self._consume()
            val = self._or_expr()
            self._consume()   # ')'
            return val
        gene = self._consume()
        return self.scores.get(gene, NO_EVIDENCE_SCORE)


def evaluate_gpr_rule(
    rule: str,
    protein_scores: dict[str, float],
    gpr_type: str = "",
) -> float:
    """Evaluate a GPR boolean expression to a numeric score.

    OR  (isozymes) → ``max``   reaction works if ANY protein present
    AND (complex)  → ``min``   needs ALL subunits
    Empty GPR      → ``0.0``   spontaneous / neutral
    Missing gene   → ``-1.0``
    """
    if not rule or not rule.strip():
        return 0.0
    return _GPRParser(rule, protein_scores).parse()


# ---------------------------------------------------------------------------
# Universal mode — direct protein→reaction scoring via universal_gpr.csv
# ---------------------------------------------------------------------------

_UNIVERSAL_GPR_CSV = Path("data/universal/db/universal_gpr.csv")

_GPR_GENE_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def _load_universal_gpr(
    csv_path: Path,
) -> dict[str, list[str]]:
    """Load universal_gpr.csv and build reaction_id -> [locus_tags].

    Parses the ``gpr`` column to extract gene IDs, collecting all genes
    across all organisms that are linked to each reaction.
    """
    reaction_proteins: dict[str, set[str]] = {}

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rxn_id = row["reaction_id"]
            gpr = row.get("gpr", "").strip()
            if not gpr:
                continue
            # Extract gene IDs from GPR rule (skip 'and', 'or', parens)
            genes = [
                tok for tok in _GPR_GENE_RE.findall(gpr)
                if tok not in ("and", "or")
            ]
            if rxn_id not in reaction_proteins:
                reaction_proteins[rxn_id] = set()
            reaction_proteins[rxn_id].update(genes)

    return {rid: sorted(genes) for rid, genes in reaction_proteins.items()}


def _build_protein_scores_from_hits(
    mmseqs_hits: dict[str, list[dict]],
    esmc_hits: dict[str, list[dict]],
    high_conf: float,
    low_conf: float,
) -> dict[str, float]:
    """Build locus_tag -> score from MMseqs2/ESM C hits.

    Parses pipe-delimited target IDs (``org|locus_tag|accession``) to
    extract the locus_tag.  Takes the best score per locus_tag.
    """
    best_identity:   dict[str, float] = {}
    best_similarity: dict[str, float] = {}

    for _pid, hits in mmseqs_hits.items():
        for h in hits:
            parts = h["ref_id"].split("|")
            tag = parts[1] if len(parts) == 3 else h["ref_id"]
            if h["identity"] > best_identity.get(tag, 0.0):
                best_identity[tag] = h["identity"]

    for _pid, hits in esmc_hits.items():
        for h in hits:
            parts = h["ref_id"].split("|")
            tag = parts[1] if len(parts) == 3 else h["ref_id"]
            if h["similarity"] > best_similarity.get(tag, 0.0):
                best_similarity[tag] = h["similarity"]

    # Merge into final scores
    all_tags = set(best_identity) | set(best_similarity)
    scores: dict[str, float] = {}
    for tag in all_tags:
        identity = best_identity.get(tag, 0.0)
        similarity = best_similarity.get(tag, 0.0)
        scores[tag] = merge_protein_scores(identity, similarity,
                                           high_conf, low_conf)

    return scores


def _score_reactions_universal(
    universal_model: cobra.Model,
    mmseqs_hits: dict[str, list[dict]],
    esmc_hits: dict[str, list[dict]],
    high_conf: float,
    low_conf: float,
) -> dict[str, float]:
    """Universal mode scoring: protein→reaction via universal_gpr.csv.

    Does NOT use the template model's GPRs (CarveMe universe has 0 genes).
    Instead, loads the GPR mappings from the CSV built by build_universal_db.py
    and scores each reaction by the best-matching protein.
    """
    csv_path = _UNIVERSAL_GPR_CSV
    if not csv_path.exists():
        print(f"[gemiz] WARNING: {csv_path} not found. "
              f"Falling back to model-based scoring.")
        return {}

    # 1 — reaction → [locus_tags] from CSV
    reaction_proteins = _load_universal_gpr(csv_path)
    print(f"[gemiz] Loaded universal_gpr.csv: "
          f"{len(reaction_proteins)} reactions with GPR data")

    # 2 — locus_tag → score from hits
    protein_scores = _build_protein_scores_from_hits(
        mmseqs_hits, esmc_hits, high_conf, low_conf,
    )
    print(f"[gemiz] Protein scores: {len(protein_scores)} locus tags scored")

    # 3 — score each reaction in the template model
    model_rxn_ids = {r.id for r in universal_model.reactions}
    reaction_scores: dict[str, float] = {}

    n_with_evidence = 0
    n_penalized = 0
    n_neutral = 0

    for rxn_id in model_rxn_ids:
        genes = reaction_proteins.get(rxn_id)

        if genes is None:
            # Reaction not in CSV — neutral (transport, exchange, etc.)
            reaction_scores[rxn_id] = 0.0
            n_neutral += 1
            continue

        # Find best protein score among all genes linked to this reaction
        gene_scores = [protein_scores.get(g, 0.0) for g in genes]
        best = max(gene_scores) if gene_scores else 0.0

        if best > 0:
            reaction_scores[rxn_id] = best
            n_with_evidence += 1
        else:
            reaction_scores[rxn_id] = NO_EVIDENCE_SCORE
            n_penalized += 1

    return reaction_scores


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_reaction_scores(
    universal_model: cobra.Model,
    mmseqs_hits: dict[str, list[dict]],
    esmc_hits: dict[str, list[dict]],
    high_conf: float = HIGH_CONF_THRESHOLD,
    low_conf: float = LOW_CONF_THRESHOLD,
    feature_table_path: str | Path | None = None,
    reference_faa_path: str | Path | None = None,
) -> dict[str, float]:
    """Compute a confidence score for every reaction in the model.

    In universal mode (detected when reference FAA path contains 'universal'),
    uses direct protein→reaction mapping via universal_gpr.csv, bypassing
    the template model's GPRs entirely.

    In organism-specific mode, uses the template model's GPR rules with
    ID mapping through feature tables or FAA headers.

    Returns
    -------
    dict
        ``{reaction_id: score}``  where score in [-1.0, 1.0].
    """
    print("[gemiz] Computing reaction scores...")

    # Detect universal mode
    universal_mode = (
        reference_faa_path is not None
        and "universal" in str(reference_faa_path)
    )

    if universal_mode:
        print("[gemiz] Universal mode: scoring via universal_gpr.csv")
        reaction_scores = _score_reactions_universal(
            universal_model, mmseqs_hits, esmc_hits,
            high_conf, low_conf,
        )
        if reaction_scores:
            _print_score_summary(reaction_scores)
            return reaction_scores
        print("[gemiz] Universal scoring failed, falling back to GPR-based scoring")

    # --- Organism-specific mode (GPR-based) ---

    # 1 -- GPR associations
    gpr_assoc = extract_gpr_associations(universal_model)

    # 2 -- protein score map
    protein_scores = build_protein_score_map(
        mmseqs_hits, esmc_hits, gpr_assoc, high_conf, low_conf,
        feature_table_path=feature_table_path,
        reference_faa_path=reference_faa_path,
    )

    # 3 — evaluate every reaction
    reaction_scores = {}
    for rxn_id, assoc in gpr_assoc.items():
        reaction_scores[rxn_id] = evaluate_gpr_rule(
            assoc["rule"], protein_scores, assoc["type"],
        )

    _print_score_summary(reaction_scores)
    return reaction_scores


def _print_score_summary(reaction_scores: dict[str, float]) -> None:
    """Print scoring summary statistics."""
    n = len(reaction_scores)
    vals = list(reaction_scores.values())
    strong   = sum(1 for s in vals if s >  0.7)
    moderate = sum(1 for s in vals if 0.3 < s <= 0.7)
    weak     = sum(1 for s in vals if 0.0 < s <= 0.3)
    penalty  = sum(1 for s in vals if s == NO_EVIDENCE_SCORE)
    spontan  = sum(1 for s in vals if s == 0.0)

    print(f"[gemiz] Reactions scored: {n}")
    if n:
        print(f"[gemiz]   Strong evidence  (score > 0.7):  {strong:>4} ({100*strong/n:.1f}%)")
        print(f"[gemiz]   Moderate evidence (0.3-0.7):     {moderate:>4} ({100*moderate/n:.1f}%)")
        print(f"[gemiz]   Weak evidence    (0.0-0.3):      {weak:>4} ({100*weak/n:.1f}%)")
        print(f"[gemiz]   No evidence      (score = -1.0): {penalty:>4} ({100*penalty/n:.1f}%)")
        print(f"[gemiz]   Spontaneous      (score = 0.0):  {spontan:>4} ({100*spontan/n:.1f}%)")
        if vals:
            print(
                f"[gemiz] Score range: min={min(vals):.3f}, "
                f"max={max(vals):.3f}, mean={sum(vals)/n:.3f}"
            )
