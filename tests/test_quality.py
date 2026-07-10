"""Tests for lightweight model quality summaries."""

from __future__ import annotations


def test_quality_summary_toy_model():
    from cobra import Metabolite, Model, Reaction

    from gemiz.quality import summarize_model

    model = Model("toy")
    a = Metabolite("a_c", compartment="c")
    b = Metabolite("b_c", compartment="c")
    a.formula = "C"
    b.charge = -1

    rxn = Reaction("R_TOY")
    rxn.add_metabolites({a: -1, b: 1})
    rxn.gene_reaction_rule = "geneA"
    model.add_reactions([rxn])
    model.objective = rxn

    report = summarize_model(model)

    assert report["id"] == "toy"
    assert report["counts"]["reactions"] == 1
    assert report["counts"]["metabolites"] == 2
    assert report["counts"]["genes"] == 1
    assert report["gpr"]["reactions_with_gpr"] == 1
    assert report["annotations"]["metabolites_with_formula"] == 1
    assert report["annotations"]["metabolites_with_charge"] == 1
