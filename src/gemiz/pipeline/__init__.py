"""Pipeline orchestrator — wires together all reconstruction steps."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def run_pipeline(
    *,
    genome: Path,
    output: Path,
    db: str,
    solver: str,
    threads: int,
    use_esm: bool,
    verbose: bool,
) -> None:
    """Run the full GEM reconstruction pipeline.

    Steps
    -----
    1. Gene calling  (Prodigal)
    2. Alignment     (DIAMOND)
    3. Embeddings    (ESM C 600M)   — optional
    4. Gapfilling    (HiGHS MILP)
    5. Export        (COBRApy → SBML)
    """
    steps = [
        ("Gene calling (Prodigal)", _step_prodigal),
        ("Sequence alignment (DIAMOND)", _step_diamond),
    ]
    if use_esm:
        steps.append(("Protein embeddings (ESM C)", _step_esm))
    steps += [
        ("Reaction gapfilling (HiGHS)", _step_gapfill),
        ("Model export (SBML)", _step_export),
    ]

    ctx: dict = {
        "genome": genome,
        "output": output,
        "db": db,
        "solver": solver,
        "threads": threads,
        "verbose": verbose,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for label, fn in steps:
            task = progress.add_task(label, total=None)
            fn(ctx, progress, task)
            progress.update(task, description=f"[green]✓[/] {label}", completed=1, total=1)

    console.print(f"\n[bold green]Done![/] Model written to [cyan]{output}[/]")


# ---------------------------------------------------------------------------
# Step stubs — each will be fleshed out in its own module
# ---------------------------------------------------------------------------

def _step_prodigal(ctx: dict, progress, task) -> None:  # noqa: ANN001
    from gemiz.pipeline import prodigal
    ctx["proteins_faa"] = prodigal.run(
        genome=ctx["genome"],
        threads=ctx["threads"],
        verbose=ctx["verbose"],
    )


def _step_diamond(ctx: dict, progress, task) -> None:  # noqa: ANN001
    from gemiz.pipeline import diamond
    ctx["hits"] = diamond.run(
        proteins=ctx["proteins_faa"],
        db=ctx["db"],
        threads=ctx["threads"],
        verbose=ctx["verbose"],
    )


def _step_esm(ctx: dict, progress, task) -> None:  # noqa: ANN001
    from gemiz.embeddings import esmc
    ctx["embeddings"] = esmc.embed(
        proteins=ctx["proteins_faa"],
        verbose=ctx["verbose"],
    )


def _step_gapfill(ctx: dict, progress, task) -> None:  # noqa: ANN001
    from gemiz.solver import gapfill
    ctx["model"] = gapfill.run(
        hits=ctx["hits"],
        embeddings=ctx.get("embeddings"),
        db=ctx["db"],
        solver=ctx["solver"],
        verbose=ctx["verbose"],
    )


def _step_export(ctx: dict, progress, task) -> None:  # noqa: ANN001
    from gemiz.io import sbml
    sbml.write(model=ctx["model"], path=ctx["output"])
