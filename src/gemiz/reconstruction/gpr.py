"""GPR (Gene-Protein-Reaction) rule parsing and mapping utilities.

A GPR rule is a boolean expression that encodes which gene products are
required for a metabolic reaction to occur:

    "b0001"                        single gene
    "b0001 or b0002"               isozymes  (OR  → max)
    "b0001 and b0002"              complex   (AND → min)
    "(b0001 and b0002) or b0003"   mixed
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Universal model loading
# ---------------------------------------------------------------------------

def load_universal_model_gprs(universal_model_path: str) -> dict[str, str]:
    """Load GPR rules from a CarveMe-style universal SBML model.

    Parameters
    ----------
    universal_model_path:
        Path to the decompressed ``.xml`` SBML file
        (e.g. ``universe_bacteria.xml``).

    Returns
    -------
    dict
        ``{reaction_id: gpr_string}``
        Spontaneous reactions have an empty GPR string ``""``.
    """
    import cobra

    model = cobra.io.read_sbml_model(universal_model_path)

    gpr_table: dict[str, str] = {}
    for rxn in model.reactions:
        gpr_table[rxn.id] = rxn.gene_reaction_rule
    return gpr_table


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def build_protein_to_reaction_map(gpr_table: dict[str, str]) -> dict[str, list[str]]:
    """Invert a GPR table: gene_id → [reaction_ids].

    Useful for quickly finding which reactions are affected when a protein
    gains evidence.
    """
    gene_re = re.compile(r"[A-Za-z0-9_.:-]+")
    mapping: dict[str, list[str]] = {}
    for rxn_id, gpr in gpr_table.items():
        if not gpr:
            continue
        genes = {g for g in gene_re.findall(gpr) if g not in ("and", "or")}
        for g in genes:
            mapping.setdefault(g, []).append(rxn_id)
    return mapping


def build_gene_to_protein_map(
    mmseqs_hits: dict[str, list[dict]],
    esmc_hits: dict[str, list[dict]],
) -> dict[str, list[str]]:
    """Map each genome protein to the reference proteins it matched.

    Merges targets from both MMseqs2 and ESM C hit tables (deduped).

    Returns
    -------
    dict
        ``{"NC_000913.3_1": ["NP_414676.1", ...], ...}``
    """
    mapping: dict[str, list[str]] = {}
    for pid, hits in mmseqs_hits.items():
        refs = {h["ref_id"] for h in hits}
        mapping.setdefault(pid, []).extend(refs)
    for pid, hits in esmc_hits.items():
        refs = {h["ref_id"] for h in hits}
        existing = set(mapping.get(pid, []))
        mapping.setdefault(pid, []).extend(refs - existing)
    return mapping


def build_ref_to_model_gene_map(
    feature_table_path: str,
) -> dict[str, str]:
    """Parse an NCBI feature table to build ref_accession → locus_tag map.

    The feature table is tab-separated with columns including
    ``product_accession`` (e.g. ``NP_414542.1``) and ``locus_tag``
    (e.g. ``b3702``).

    Parameters
    ----------
    feature_table_path:
        Path to the decompressed NCBI feature table (``.txt``).

    Returns
    -------
    dict
        ``{"NP_414542.1": "b3702", ...}``
    """
    import pandas as pd

    df = pd.read_csv(feature_table_path, sep="\t", dtype=str, low_memory=False)

    # Find the columns (names vary slightly between NCBI releases)
    acc_col = None
    tag_col = None
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "_")
        if "product_accession" in cl:
            acc_col = c
        elif cl == "locus_tag":
            tag_col = c

    if acc_col is None or tag_col is None:
        raise ValueError(
            f"Could not find product_accession / locus_tag columns in {feature_table_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    subset = df[[acc_col, tag_col]].dropna()
    return dict(zip(subset[acc_col], subset[tag_col]))


def extract_gpr_genes(gpr_string: str) -> set[str]:
    """Extract all gene identifiers from a GPR string."""
    if not gpr_string or not gpr_string.strip():
        return set()
    gene_re = re.compile(r"[A-Za-z0-9_.:-]+")
    return {g for g in gene_re.findall(gpr_string) if g not in ("and", "or")}


# ---------------------------------------------------------------------------
# GPR evaluation — recursive descent parser
# ---------------------------------------------------------------------------
#
#   Grammar
#   -------
#   expr      → or_expr
#   or_expr   → and_expr ('or' and_expr)*
#   and_expr  → atom     ('and' atom)*
#   atom      → GENE_ID  | '(' expr ')'
#
#   OR  → max(children)   — reaction fires if ANY isozyme present
#   AND → min(children)   — reaction needs ALL subunits

class _GPRParser:
    """Recursive-descent parser/evaluator for GPR boolean expressions."""

    _TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

    def __init__(self, text: str, scores: dict[str, float]) -> None:
        self.tokens = self._TOKEN_RE.findall(text)
        self.pos = 0
        self.scores = scores

    # -- helpers --

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    # -- grammar --

    def parse(self) -> float:
        if not self.tokens:
            return 0.0          # empty GPR → spontaneous
        return self._or_expr()

    def _or_expr(self) -> float:
        result = self._and_expr()
        while self._peek() == "or":
            self._consume()
            result = max(result, self._and_expr())
        return result

    def _and_expr(self) -> float:
        result = self._atom()
        while self._peek() == "and":
            self._consume()
            result = min(result, self._atom())
        return result

    def _atom(self) -> float:
        tok = self._peek()
        if tok == "(":
            self._consume()             # eat '('
            result = self._or_expr()
            self._consume()             # eat ')'
            return result
        gene_id = self._consume()
        return self.scores.get(gene_id, -1.0)


def resolve_gpr(gpr_string: str, protein_scores: dict[str, float]) -> float:
    """Evaluate a GPR expression to a numeric score.

    OR  (isozymes) → **max** of children (reaction works if ANY present)
    AND (complex)  → **min** of children (needs ALL subunits)

    Missing genes default to ``-1.0``.
    Empty GPR (spontaneous) returns ``0.0``.
    """
    if not gpr_string or not gpr_string.strip():
        return 0.0
    return _GPRParser(gpr_string, protein_scores).parse()
