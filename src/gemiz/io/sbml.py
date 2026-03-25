"""Step 5 — Export COBRApy model to SBML Level 3 (.xml)."""

from __future__ import annotations

from pathlib import Path


def write(*, model: "cobra.Model", path: Path) -> None:  # noqa: F821
    """Write *model* to SBML Level 3 at *path*."""
    import cobra  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.write_sbml_model(model, str(path))


def read(path: Path) -> "cobra.Model":  # noqa: F821
    """Read an SBML model from *path* and return a COBRApy Model."""
    import cobra  # type: ignore[import-untyped]

    return cobra.io.read_sbml_model(str(path))
