"""Step 3 — Protein embeddings via ESM C 600M.

ESM C (Cambrian) is EvolutionaryScale's open, MIT-licensed protein language
model.  We use the 600M parameter variant which fits on an 8 GB GPU.

Install:
    pip install esm   (EvolutionaryScale SDK)
    # or: pip install fair-esm  (Meta's original ESM)

References
----------
- https://github.com/evolutionaryscale/esm
- Hayes et al. 2024 "Simulating 500 million years of evolution with a language model"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def embed(*, proteins: Path, verbose: bool) -> np.ndarray:
    """Generate ESM C embeddings for all proteins in *proteins* (.faa).

    Returns
    -------
    np.ndarray
        Shape (N, 1152) float32 array — one row per protein sequence.
        (1152 = ESM C 600M hidden dim)
    """
    sequences = _load_sequences(proteins)
    if verbose:
        print(f"[esmc] Embedding {len(sequences)} sequences with ESM C 600M …")

    try:
        return _embed_esmc(sequences, verbose=verbose)
    except ImportError:
        # Graceful fallback to random embeddings during development
        import warnings
        warnings.warn(
            "ESM SDK not installed — using random embeddings (dev mode only).\n"
            "Install with: pip install esm",
            stacklevel=2,
        )
        return np.random.randn(len(sequences), 1152).astype(np.float32)


def _load_sequences(faa: Path) -> list[str]:
    from Bio import SeqIO  # type: ignore[import-untyped]
    return [str(r.seq) for r in SeqIO.parse(str(faa), "fasta")]


def _embed_esmc(sequences: list[str], *, verbose: bool) -> np.ndarray:
    """Run ESM C 600M inference; returns (N, 1152) float32."""
    import torch
    from esm.models.esmc import ESMC  # type: ignore[import-untyped]
    from esm.sdk.api import ESMProtein, LogitsConfig  # type: ignore[import-untyped]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[esmc] Using device: {device}")

    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for seq in sequences:
            protein = ESMProtein(sequence=seq)
            output = model.encode(protein)
            # Mean-pool over residue dimension → (1152,)
            emb = output.embeddings.mean(dim=1).squeeze(0).cpu().float().numpy()
            embeddings.append(emb)

    return np.vstack(embeddings)
