"""Step 3 — ESM C 600M protein embeddings.

Embeds low-confidence proteins (those that MMseqs2 could not match above
the identity threshold) into a 1152-dimensional vector space.

Hardware requirements
---------------------
  RTX 3070 8 GB VRAM — fits comfortably (~4 GB for the model)
  Apple M-series      — MPS backend
  CPU                 — works, just slower

Key rules (from benchmarking)
-----------------------------
  1. Use float32, not bfloat16 — avoids precision issues on consumer GPUs.
  2. Process ONE sequence at a time — avoids padding artifacts and OOM on
     long sequences.
  3. Mean-pool per-residue embeddings → one 1152-dim vector per protein.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MODEL_NAME = "esmc_600m"
EMBEDDING_DIM = 1152

_model_cache: tuple | None = None
_device_cache: str | None = None


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Detect the best available device for ESM C inference.

    Returns ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[gemiz] ESM C device: CUDA ({name}, {vram:.1f}GB)")
        _device_cache = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[gemiz] ESM C device: Apple Metal (M-series)")
        _device_cache = "mps"
    else:
        print("[gemiz] ESM C device: CPU")
        _device_cache = "cpu"

    return _device_cache


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: Optional[str] = None) -> tuple:
    """Load ESM C 600M and return ``(model, device_str)``.

    The model is cached after the first call.  Subsequent calls return
    the cached reference instantly.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if device is None:
        device = get_device()

    print("[gemiz] Loading ESM C 600M...")
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(MODEL_NAME, device=torch.device(device))
    model.eval()

    if device == "cuda":
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[gemiz] ESM C ready ({used:.1f}GB / {total:.1f}GB VRAM)")
    else:
        print("[gemiz] ESM C ready")

    _model_cache = (model, device)
    return _model_cache


def unload_model() -> None:
    """Free the cached model and reclaim GPU memory."""
    global _model_cache, _device_cache
    if _model_cache is not None:
        model, device = _model_cache
        del model
        _model_cache = None
        if device == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single-sequence embedding
# ---------------------------------------------------------------------------

def embed_sequence(
    sequence: str,
    model: object,
    device: str,
) -> np.ndarray:
    """Embed ONE protein sequence → 1152-dim float32 vector.

    Steps
    -----
    1. Wrap in ESMProtein.
    2. Tokenise via ``model.encode``.
    3. Forward pass through ESM C.
    4. Mean-pool per-residue embeddings (rule 3).
    5. Return as float32 numpy array, shape ``(1152,)``.
    """
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)

    output = model.logits(protein_tensor, LogitsConfig(return_embeddings=True))

    # output.embeddings shape: [1, seq_len, 1152]
    embedding = output.embeddings.float().squeeze(0)   # [seq_len, 1152]
    embedding = embedding.mean(dim=0)                   # [1152]

    return embedding.cpu().numpy()


# ---------------------------------------------------------------------------
# Batch (one-at-a-time) embedding
# ---------------------------------------------------------------------------

def embed_proteins(
    sequences: dict[str, str],
    output_path: str,
    device: Optional[str] = None,
) -> str:
    """Embed multiple proteins, one at a time, and save to ``.npz``.

    Parameters
    ----------
    sequences:
        ``{protein_id: amino_acid_sequence}``.
    output_path:
        Destination ``.npz`` file.
    device:
        Override device (``"cuda"`` / ``"mps"`` / ``"cpu"``).

    Saves
    -----
    ``ids``  : ``str`` array of protein IDs (same order).
    ``matrix``: ``float32`` array, shape ``(n_proteins, 1152)``.

    Returns
    -------
    str
        Absolute path to the saved ``.npz`` file.
    """
    model, dev = load_model(device)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ids = list(sequences.keys())
    n = len(ids)
    matrix = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    print(f"[gemiz] Embedding {n} proteins with ESM C 600M...")
    start = time.time()

    for i, pid in enumerate(ids):
        matrix[i] = embed_sequence(sequences[pid], model, dev)

        # progress every 10 proteins or on last
        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            mins, secs = divmod(int(remaining), 60)
            pct = (i + 1) / n * 100
            print(
                f"[gemiz] {i + 1}/{n} ({pct:.1f}%) "
                f"| ~{mins}m {secs}s remaining"
            )

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)

    np.savez(out, ids=np.array(ids), matrix=matrix)
    print(f"[gemiz] Embedding complete: {n} proteins in {mins}m {secs}s")
    print(f"[gemiz] Saved to {out}")
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Loading saved embeddings
# ---------------------------------------------------------------------------

def load_embeddings(npz_path: str) -> tuple[list[str], np.ndarray]:
    """Load embeddings from a ``.npz`` file.

    Returns ``(ids_list, matrix)`` where matrix has shape
    ``(n_proteins, 1152)`` and dtype ``float32``.
    """
    data = np.load(npz_path, allow_pickle=True)
    return list(data["ids"]), data["matrix"]
