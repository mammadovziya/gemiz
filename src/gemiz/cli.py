"""gemiz CLI -- entry point for all sub-commands."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# Default data paths (relative to cwd; overridable via options)
_DEFAULT_MODEL     = "data/universal/iML1515.xml"
_DEFAULT_REF_FAA   = "data/reference/iML1515_proteins.faa"
_DEFAULT_FEAT_TBL  = "data/reference/ecoli_feature_table.txt"

# Universal DB paths (built by scripts/build_universal_db.py)
_UNIVERSAL_TEMPLATE = "data/universal/db/universal_template.xml"
_UNIVERSAL_FAA      = "data/universal/db/universal_proteins.faa"
_UNIVERSAL_MMSEQS   = "data/universal/db/mmseqs_db"

# CarveMe bacteria universe (preferred template for universal mode)
_CARVEME_UNIVERSE   = "data/universal/carveme_bacteria.xml"

# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="gemiz")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """gemiz -- reconstruct genome-scale metabolic models from raw .fna files.

    \b
    Typical usage:
        gemiz carve genome.fna -o model.xml
        gemiz carve genome.fna --no-esm --threads 8
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# carve  (primary command)
# ---------------------------------------------------------------------------

@main.command()
@click.argument("genome", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output .xml path (default: <genome_stem>_model.xml).",
)
@click.option(
    "--organism",
    default=None,
    help="Organism name (loads config from data/organisms/<name>/config.json). "
         "Run setup_organism.py first to create the config.",
)
@click.option(
    "--template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=f"Universal model .xml (default: {_DEFAULT_MODEL}).",
)
@click.option(
    "--reference",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=f"Reference proteins .faa (default: {_DEFAULT_REF_FAA}).",
)
@click.option(
    "--feature-table",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=f"NCBI feature table for ID mapping (default: {_DEFAULT_FEAT_TBL}).",
)
@click.option(
    "--threads", "-t",
    type=int,
    default=4,
    show_default=True,
    help="CPU threads for MMseqs2.",
)
@click.option(
    "--esm-db",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Pre-built ESM C reference DB directory (skips index generation).",
)
@click.option(
    "--no-esm", is_flag=True, default=False,
    help="Skip ESM C embeddings (faster but less accurate).",
)
@click.option(
    "--high-conf",
    type=float,
    default=50.0,
    show_default=True,
    help="Identity %% above which MMseqs2 is fully trusted.",
)
@click.option(
    "--low-conf",
    type=float,
    default=30.0,
    show_default=True,
    help="Identity %% below which sequence alignment is unreliable.",
)
@click.option(
    "--min-growth",
    type=float,
    default=0.1,
    show_default=True,
    help="Minimum biomass flux (h^-1) required.",
)
@click.option(
    "--sensitivity",
    type=float,
    default=7.5,
    show_default=True,
    help="MMseqs2 sensitivity (4=fast, 7.5=balanced, 9.5=sensitive).",
)
@click.option(
    "--media",
    type=click.Choice(["M9", "LB", "M9[glyc]", "M9[-O2]", "LB[-O2]"], case_sensitive=True),
    default="M9",
    show_default=True,
    help="Growth medium for universal mode (constrains exchange reactions).",
)
@click.pass_context
def carve(
    ctx: click.Context,
    genome: Path,
    output: Path | None,
    organism: str | None,
    template: Path | None,
    reference: Path | None,
    feature_table: Path | None,
    esm_db: Path | None,
    threads: int,
    no_esm: bool,
    high_conf: float,
    low_conf: float,
    min_growth: float,
    sensitivity: float,
    media: str,
) -> None:
    """Reconstruct a GEM from a raw genome FASTA file.

    Full pipeline:

    \b
      1. Gene calling        -- pyrodigal
      2. Protein alignment   -- MMseqs2
      3. Protein embeddings  -- ESM C 600M  (skip with --no-esm)
      4. Reaction scoring    -- adaptive weighting
      5. MILP carving        -- HiGHS
      6. Model export        -- SBML (.xml)
    """
    # ---- load organism config if specified ----
    org_config: dict = {}
    if organism is not None:
        config_path = Path(f"data/organisms/{organism}/config.json")
        if not config_path.exists():
            console.print(f"[bold red]Error:[/] organism config not found: {config_path}")
            console.print(f"Run: python scripts/setup_organism.py {organism} --ncbi-assembly <accession>")
            sys.exit(1)
        with open(config_path) as f:
            org_config = json.load(f)
        console.print(f"  Organism:   [green]{organism}[/] (loaded {config_path})")

    # ---- resolve defaults (CLI flags override organism config) ----
    if output is None:
        output = genome.parent / f"{genome.stem}_model.xml"

    # Detect universal mode: no --organism and no explicit --template/--reference
    _univ_faa      = Path(_UNIVERSAL_FAA)
    _univ_mmseqs   = Path(_UNIVERSAL_MMSEQS)
    _carveme_univ  = Path(_CARVEME_UNIVERSE)
    _univ_template = Path(_UNIVERSAL_TEMPLATE)
    _universal_mode = (
        organism is None
        and template is None
        and reference is None
        and _univ_faa.exists()
    )

    if _universal_mode:
        console.print(f"  Mode:       [cyan]universal[/] "
                      f"(data/universal/db/)")
        # Prefer CarveMe bacteria universe (5532 rxns, proper Growth objective)
        if template is None:
            if _carveme_univ.exists():
                template = _carveme_univ
            elif _univ_template.exists():
                template = _univ_template
            else:
                console.print("[bold red]Error:[/] No universal template found. "
                              f"Expected {_carveme_univ} or {_univ_template}")
                sys.exit(1)
        if reference is None:
            reference = _univ_faa
        # feature_table left as None — scoring uses universal_gpr.csv directly
        if esm_db is None and _univ_mmseqs.exists():
            esm_db = _univ_mmseqs
    else:
        if template is None:
            if "template" in org_config:
                template = Path(org_config["template"])
            else:
                template = Path(_DEFAULT_MODEL)
        if reference is None:
            if "proteins" in org_config:
                reference = Path(org_config["proteins"])
            else:
                reference = Path(_DEFAULT_REF_FAA)
        if feature_table is None:
            if "feature_table" in org_config:
                ft = Path(org_config["feature_table"])
                feature_table = ft if ft.exists() else None
            else:
                ft = Path(_DEFAULT_FEAT_TBL)
                feature_table = ft if ft.exists() else None
        if esm_db is None and "esm_db" in org_config:
            db = Path(org_config["esm_db"])
            if db.exists():
                esm_db = db

    # ---- validate inputs ----
    if not template.exists():
        console.print(f"[bold red]Error:[/] template model not found: {template}")
        console.print("Download iML1515 or specify --template path.")
        sys.exit(1)
    if not reference.exists():
        console.print(f"[bold red]Error:[/] reference proteins not found: {reference}")
        sys.exit(1)

    # ---- header ----
    console.print()
    console.print(Rule("[bold cyan]gemiz v0.1.0[/]", style="cyan"))
    console.print(f"  Genome:     [green]{genome}[/]")
    console.print(f"  Output:     [green]{output}[/]")
    console.print(f"  Template:   [yellow]{template}[/]")
    console.print(f"  Reference:  [yellow]{reference}[/]")
    console.print(f"  Threads:    [yellow]{threads}[/]")
    esm_label = "disabled" if no_esm else f"enabled{f' (db: {esm_db})' if esm_db else ''}"
    console.print(f"  ESM C:      [yellow]{esm_label}[/]")
    console.print(f"  Min growth: [yellow]{min_growth}[/]")
    if _universal_mode:
        console.print(f"  Media:      [yellow]{media}[/]")
    console.print(Rule(style="cyan"))

    # ---- run pipeline ----
    from gemiz.reconstruction.pipeline import run_full_pipeline

    result = run_full_pipeline(
        genome_fna=str(genome),
        output_xml=str(output),
        universal_model_path=str(template),
        reference_faa_path=str(reference),
        feature_table_path=str(feature_table) if feature_table else None,
        esm_db_path=str(esm_db) if esm_db else None,
        high_conf=high_conf,
        low_conf=low_conf,
        min_growth=min_growth,
        use_esm=not no_esm,
        threads=threads,
        sensitivity=sensitivity,
        media=media if _universal_mode else None,
    )

    # ---- summary ----
    console.print()
    console.print(Rule(style="cyan"))
    total = result["total_time"]
    console.print(f"  [bold green]Done in {total:.1f}s[/]")
    console.print()
    console.print("  Model summary:")
    console.print(f"    Reactions:   [cyan]{result['n_reactions']}[/]")
    console.print(f"    Metabolites: [cyan]{result['n_metabolites']}[/]")
    console.print(f"    Genes:       [cyan]{result['n_genes']}[/]")
    gr = result["growth_rate"]
    grow_mark = "[green]YES[/]" if result["can_grow"] else "[red]NO[/]"
    console.print(f"    Growth rate: [cyan]{gr:.4f}[/] h^-1  {grow_mark}")
    console.print(Rule(style="cyan"))
    console.print()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(model: Path) -> None:
    """Print summary statistics for an existing GEM (.xml)."""
    import cobra  # type: ignore[import-untyped]

    console.print(f"[bold]Loading[/] {model} …")
    gem = cobra.io.read_sbml_model(str(model))

    console.print(Panel.fit(
        f"  reactions  : [cyan]{len(gem.reactions)}[/]\n"
        f"  metabolites: [cyan]{len(gem.metabolites)}[/]\n"
        f"  genes      : [cyan]{len(gem.genes)}[/]\n"
        f"  objective  : [cyan]{gem.objective.to_json()['expression']}[/]",
        title=f"[bold]{gem.id or model.stem}[/]",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(model: Path) -> None:
    """Run basic feasibility check on a GEM using the HiGHS solver."""
    import cobra  # type: ignore[import-untyped]

    console.print(f"[bold]Validating[/] {model} …")
    gem = cobra.io.read_sbml_model(str(model))
    gem.solver = "glpk"  # cobra bundles glpk; highs integration via cobra.Configuration

    solution = gem.optimize()
    if solution.status == "optimal":
        console.print(
            f"[bold green]PASS[/]  objective = {solution.objective_value:.4f}"
        )
    else:
        console.print(f"[bold red]FAIL[/]  status = {solution.status}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# download-db
# ---------------------------------------------------------------------------

@main.command("download-db")
@click.option(
    "--db",
    type=click.Choice(["bigg", "modelseed", "all"], case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path("data/databases"),
    show_default=True,
    help="Directory to store downloaded databases.",
)
def download_db(db: str, dest: Path) -> None:
    """Download reference databases (BiGG, ModelSEED) for reaction mapping."""
    from gemiz.db import downloader

    dest.mkdir(parents=True, exist_ok=True)
    downloader.download(db=db, dest=dest, console=console)


if __name__ == "__main__":
    main()
