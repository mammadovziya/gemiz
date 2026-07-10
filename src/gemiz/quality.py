"""Model quality summaries for reconstructed GEMs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cobra


def _is_boundary_like(rxn_id: str) -> bool:
    norm = rxn_id[2:] if rxn_id.startswith("R_") else rxn_id
    return norm.startswith(("EX_", "DM_", "SK_"))


def _safe_growth(model: "cobra.Model") -> dict[str, Any]:
    try:
        sol = model.optimize()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc), "growth_rate": 0.0}

    growth = sol.objective_value if sol.status == "optimal" else 0.0
    return {
        "status": sol.status,
        "growth_rate": round(float(growth or 0.0), 6),
        "can_grow": bool(sol.status == "optimal" and (growth or 0.0) > 1e-6),
    }


def _blocked_reactions(model: "cobra.Model", enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"computed": False, "reason": "disabled"}

    try:
        from cobra.flux_analysis import find_blocked_reactions

        blocked = find_blocked_reactions(model, open_exchanges=True)
    except Exception as exc:  # noqa: BLE001
        return {"computed": False, "error": str(exc)}

    internal = [rid for rid in blocked if not _is_boundary_like(rid)]
    return {
        "computed": True,
        "count": len(blocked),
        "internal_count": len(internal),
        "fraction": round(len(blocked) / len(model.reactions), 4) if model.reactions else 0.0,
        "sample": internal[:25],
    }


def summarize_model(
    model: "cobra.Model",
    *,
    include_blocked: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable model quality summary."""
    reactions = list(model.reactions)
    metabolites = list(model.metabolites)
    genes = list(model.genes)

    boundary = [rxn for rxn in reactions if _is_boundary_like(rxn.id)]
    gpr_reactions = [rxn for rxn in reactions if rxn.gene_reaction_rule.strip()]
    orphan_mets = [met.id for met in metabolites if len(met.reactions) == 0]
    formulas = [met for met in metabolites if getattr(met, "formula", None)]
    charges = [met for met in metabolites if getattr(met, "charge", None) is not None]

    return {
        "id": model.id,
        "name": model.name,
        "counts": {
            "reactions": len(reactions),
            "metabolites": len(metabolites),
            "genes": len(genes),
            "boundary_reactions": len(boundary),
            "internal_reactions": len(reactions) - len(boundary),
        },
        "growth": _safe_growth(model),
        "annotations": {
            "metabolites_with_formula": len(formulas),
            "metabolites_with_formula_fraction": (
                round(len(formulas) / len(metabolites), 4) if metabolites else 0.0
            ),
            "metabolites_with_charge": len(charges),
            "metabolites_with_charge_fraction": (
                round(len(charges) / len(metabolites), 4) if metabolites else 0.0
            ),
        },
        "gpr": {
            "reactions_with_gpr": len(gpr_reactions),
            "reaction_gpr_fraction": (
                round(len(gpr_reactions) / len(reactions), 4) if reactions else 0.0
            ),
        },
        "connectivity": {
            "orphan_metabolites": len(orphan_mets),
            "orphan_metabolite_sample": orphan_mets[:25],
        },
        "blocked_reactions": _blocked_reactions(model, include_blocked),
    }


def summarize_model_file(
    model_path: str | Path,
    *,
    include_blocked: bool = False,
) -> dict[str, Any]:
    """Load an SBML model and return :func:`summarize_model` output."""
    import cobra

    path = Path(model_path)
    model = cobra.io.read_sbml_model(str(path))
    summary = summarize_model(model, include_blocked=include_blocked)
    summary["path"] = str(path)
    return summary
