"""gemiz CLI — entry point for all sub-commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="gemiz")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """gemiz — reconstruct genome-scale metabolic models from raw .fna files.

    \b
    Typical usage:
        gemiz reconstruct genome.fna --output model.xml
        gemiz reconstruct genome.fna --db bigg --solver highs
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

@main.command()
@click.argument("genome", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    show_default=True,
    help="Output .xml path (default: <genome_stem>.xml next to input).",
)
@click.option(
    "--db",
    type=click.Choice(["bigg", "modelseed", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Reference database(s) to use.",
)
@click.option(
    "--solver",
    type=click.Choice(["highs", "glpk"], case_sensitive=False),
    default="highs",
    show_default=True,
    help="LP/MILP solver. HiGHS is recommended.",
)
@click.option(
    "--threads", "-t",
    type=int,
    default=1,
    show_default=True,
    help="Number of CPU threads for Prodigal / DIAMOND.",
)
@click.option(
    "--no-esm", is_flag=True, default=False,
    help="Skip ESM C embeddings (faster but less accurate).",
)
@click.pass_context
def reconstruct(
    ctx: click.Context,
    genome: Path,
    output: Path | None,
    db: str,
    solver: str,
    threads: int,
    no_esm: bool,
) -> None:
    """Reconstruct a GEM from a raw genome FASTA file (GENOME).

    Steps executed:
    \b
      1. Gene calling        — Prodigal
      2. Sequence alignment  — DIAMOND vs BiGG / ModelSEED
      3. Protein embeddings  — ESM C 600M  (skipped with --no-esm)
      4. Reaction gapfilling — HiGHS MILP
      5. Model export        — SBML / COBRApy
    """
    verbose: bool = ctx.obj["verbose"]

    if output is None:
        output = genome.with_suffix(".xml")

    console.print(
        Panel.fit(
            f"[bold cyan]gemiz reconstruct[/]\n"
            f"  genome : [green]{genome}[/]\n"
            f"  output : [green]{output}[/]\n"
            f"  db     : [yellow]{db}[/]\n"
            f"  solver : [yellow]{solver}[/]\n"
            f"  threads: [yellow]{threads}[/]\n"
            f"  ESM C  : [yellow]{'disabled' if no_esm else 'enabled'}[/]",
            title="[bold]gemiz[/]",
            border_style="cyan",
        )
    )

    from gemiz.pipeline import run_pipeline  # lazy import keeps CLI startup fast

    run_pipeline(
        genome=genome,
        output=output,
        db=db,
        solver=solver,
        threads=threads,
        use_esm=not no_esm,
        verbose=verbose,
    )


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
