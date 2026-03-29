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

import re
from pathlib import Path
from typing import Optional

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
) -> None:
    """Print diagnostic information about ID overlap between alignments and GPR.

    Helps debug the common mismatch between NCBI accessions (``NP_414676.1``)
    in MMseqs2 output and locus tags (``b4025``) in GPR rules.
    """
    # Collect ref IDs from MMseqs2
    mmseqs_ref_ids: set[str] = set()
    for hits in mmseqs_hits.values():
        for h in hits:
            mmseqs_ref_ids.add(h["ref_id"])

    # Collect gene IDs from GPR
    gpr_genes: set[str] = set()
    for a in gpr_associations.values():
        gpr_genes.update(a["genes"])

    # Direct overlap
    direct_overlap = mmseqs_ref_ids & gpr_genes

    print(f"\n[gemiz] === ID Mapping Diagnostic ===")
    print(f"[gemiz]   MMseqs2 ref IDs:    {len(mmseqs_ref_ids)}")
    print(f"[gemiz]   GPR gene IDs:       {len(gpr_genes)}")
    print(f"[gemiz]   Direct overlap:     {len(direct_overlap)}")

    # Show sample IDs
    sample_mmseqs = sorted(mmseqs_ref_ids)[:5]
    sample_gpr = sorted(gpr_genes)[:5]
    print(f"[gemiz]   Sample MMseqs2 IDs: {sample_mmseqs}")
    print(f"[gemiz]   Sample GPR IDs:     {sample_gpr}")

    if id_map:
        # Translate MMseqs2 IDs and check overlap
        translated = {id_map.get(rid, rid) for rid in mmseqs_ref_ids}
        mapped_overlap = translated & gpr_genes
        print(f"[gemiz]   After ID mapping:   {len(mapped_overlap)} overlap "
              f"({100*len(mapped_overlap)/max(len(gpr_genes),1):.1f}% of GPR genes)")
    else:
        print(f"[gemiz]   No ID map provided.")

    print(f"[gemiz] ================================\n")


# ---------------------------------------------------------------------------
# Protein score map
# ---------------------------------------------------------------------------

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

    # Load ID mapping (NCBI accession -> locus_tag)
    id_map = parse_reference_id_map(feature_table_path, reference_faa_path)

    # Collect all genes that appear in any GPR rule
    all_gpr_genes: set[str] = set()
    for a in gpr_associations.values():
        all_gpr_genes.update(a["genes"])

    # Build reverse index: gene_id -> best (identity, similarity) from any
    # genome protein that hit it.  Translate ref IDs via id_map.
    ref_best_identity:   dict[str, float] = {}
    ref_best_similarity: dict[str, float] = {}

    for _pid, hits in mmseqs_hits.items():
        for h in hits:
            rid = h["ref_id"]
            gene_id = id_map.get(rid, rid)  # translate accession -> locus_tag
            if h["identity"] > ref_best_identity.get(gene_id, 0.0):
                ref_best_identity[gene_id] = h["identity"]

    for _pid, hits in esmc_hits.items():
        for h in hits:
            rid = h["ref_id"]
            gene_id = id_map.get(rid, rid)  # translate accession -> locus_tag
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

    Steps:
    1. Extract GPR associations
    2. Build per-gene protein score map
    3. Evaluate each reaction's GPR rule
    4. Print summary

    Returns
    -------
    dict
        ``{reaction_id: score}``  where score in [-1.0, 1.0].
    """
    print("[gemiz] Computing reaction scores...")

    # 1 -- GPR associations
    gpr_assoc = extract_gpr_associations(universal_model)

    # 2 -- protein score map
    protein_scores = build_protein_score_map(
        mmseqs_hits, esmc_hits, gpr_assoc, high_conf, low_conf,
        feature_table_path=feature_table_path,
        reference_faa_path=reference_faa_path,
    )

    # 3 — evaluate every reaction
    reaction_scores: dict[str, float] = {}
    for rxn_id, assoc in gpr_assoc.items():
        reaction_scores[rxn_id] = evaluate_gpr_rule(
            assoc["rule"], protein_scores, assoc["type"],
        )

    # 4 — summary
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

    return reaction_scores
