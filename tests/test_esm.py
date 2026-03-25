"""Tests for ESM C 600M embedding pipeline.

Requires:
  pip install gemiz[embeddings]
  A HuggingFace token with access to EvolutionaryScale models:
    1. Register at https://forge.evolutionaryscale.ai
    2. Accept the ESM C license
    3. Run: huggingface-cli login

On native Windows these tests are skipped (CUDA/WSL2 required).
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level availability check
# ---------------------------------------------------------------------------

_skip_reason: str | None = None

if platform.system() == "Windows":
    _skip_reason = "ESM C requires CUDA (WSL2) or MPS (Mac). Run inside WSL on Windows."

if _skip_reason is None:
    try:
        import torch
    except ImportError:
        _skip_reason = "torch not installed. Run: pip install gemiz[embeddings]"

if _skip_reason is None:
    try:
        import esm  # noqa: F401
    except ImportError:
        _skip_reason = (
            "esm package not installed. Run: pip install gemiz[embeddings]\n"
            "Then get a token at https://forge.evolutionaryscale.ai"
        )

if _skip_reason is None:
    try:
        import faiss  # noqa: F401
    except ImportError:
        _skip_reason = "faiss not installed. Run: pip install faiss-cpu"

if _skip_reason is not None:
    pytest.skip(_skip_reason, allow_module_level=True)

# --- imports that depend on the above guard ---
from gemiz.embedding.esm import (
    EMBEDDING_DIM,
    embed_proteins,
    embed_sequence,
    get_device,
    load_embeddings,
    load_model,
)
from gemiz.embedding.database import (
    build_faiss_index,
    generate_reference_db,
    search_similar,
)

GENOME    = Path("data/genomes/ecoli_k12.fna")
REFERENCE = Path("data/reference/iML1515_proteins.faa")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_faa(faa_path: Path, limit: int | None = None) -> dict[str, str]:
    """Read a .faa FASTA file → {id: sequence}."""
    from Bio import SeqIO
    seqs: dict[str, str] = {}
    for rec in SeqIO.parse(faa_path, "fasta"):
        seqs[rec.id] = str(rec.seq)
        if limit and len(seqs) >= limit:
            break
    return seqs


def _vram_status() -> str:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.2f}GB / {total:.1f}GB VRAM"
    return "N/A"


# ---------------------------------------------------------------------------
# Test 1 — availability check
# ---------------------------------------------------------------------------

def test_check_available():
    """ESM C 600M dependencies must be importable."""
    import esm  # noqa: F401
    import faiss  # noqa: F401
    print(f"\n  ESM C 600M available")
    print(f"  torch : {torch.__version__}")
    print(f"  CUDA  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM  : {vram:.1f}GB")


# ---------------------------------------------------------------------------
# Test 2 — device detection
# ---------------------------------------------------------------------------

def test_get_device():
    """get_device() must return a valid device string."""
    device = get_device()
    assert device in ("cuda", "mps", "cpu")
    print(f"\n  Selected device: {device}")
    if device == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {_vram_status()}")


# ---------------------------------------------------------------------------
# Test 3 — model loading + caching
# ---------------------------------------------------------------------------

def test_load_model():
    """load_model() must return (model, device) and cache on second call."""
    print(f"\n  VRAM before load: {_vram_status()}")

    t0 = time.perf_counter()
    model, device = load_model()
    t_first = time.perf_counter() - t0

    print(f"  VRAM after load : {_vram_status()}")
    print(f"  First load      : {t_first:.1f}s")

    assert model is not None
    assert device in ("cuda", "mps", "cpu")

    # Second call must be instant (cached)
    t0 = time.perf_counter()
    model2, _ = load_model()
    t_cached = time.perf_counter() - t0

    print(f"  Cached load     : {t_cached:.4f}s")
    assert model2 is model, "Second load_model() should return cached instance"
    assert t_cached < 0.1, "Cached load took too long — model was reloaded"


# ---------------------------------------------------------------------------
# Test 4 — single sequence embedding
# ---------------------------------------------------------------------------

def test_embed_sequence():
    """embed_sequence() must return a float32 array of shape (1152,)."""
    seqs = _read_faa(REFERENCE, limit=1)
    pid, seq = next(iter(seqs.items()))

    model, device = load_model()

    t0 = time.perf_counter()
    vec = embed_sequence(seq, model, device)
    elapsed = time.perf_counter() - t0

    print(f"\n  Protein        : {pid}")
    print(f"  Sequence length: {len(seq)} aa")
    print(f"  Embedding shape: {vec.shape}")
    print(f"  dtype          : {vec.dtype}")
    print(f"  First 5 values : {vec[:5]}")
    print(f"  L2 norm        : {np.linalg.norm(vec):.4f}")
    print(f"  Time           : {elapsed:.3f}s")
    print(f"  VRAM           : {_vram_status()}")

    assert vec.shape == (EMBEDDING_DIM,), f"Expected (1152,), got {vec.shape}"
    assert vec.dtype == np.float32
    assert not np.any(np.isnan(vec)), "NaN in embedding"
    assert np.linalg.norm(vec) > 0, "Zero-norm embedding"


# ---------------------------------------------------------------------------
# Test 5 — embed 10 proteins
# ---------------------------------------------------------------------------

def test_embed_proteins_10(tmp_path):
    """embed_proteins() on 10 sequences must produce (10, 1152) .npz."""
    seqs = _read_faa(REFERENCE, limit=10)
    npz = str(tmp_path / "test_10.npz")

    t0 = time.perf_counter()
    path = embed_proteins(seqs, npz)
    elapsed = time.perf_counter() - t0

    ids, matrix = load_embeddings(path)

    print(f"\n  Proteins : {len(ids)}")
    print(f"  Shape    : {matrix.shape}")
    print(f"  dtype    : {matrix.dtype}")
    print(f"  Time     : {elapsed:.1f}s  ({elapsed / 10:.2f}s/protein)")
    print(f"  File size: {Path(path).stat().st_size / 1024:.0f} KB")
    print(f"  VRAM     : {_vram_status()}")

    assert matrix.shape == (10, EMBEDDING_DIM)
    assert matrix.dtype == np.float32
    assert len(ids) == 10
    assert not np.any(np.isnan(matrix))


# ---------------------------------------------------------------------------
# Test 6 — generate full iML1515 reference DB
# ---------------------------------------------------------------------------

_REFDB_DIR = Path("data/reference/iml1515_db")


def test_generate_reference_db():
    """generate_reference_db() must embed all iML1515 proteins + build FAISS."""
    assert REFERENCE.exists(), f"Reference .faa not found: {REFERENCE}"

    t0 = time.perf_counter()
    result_dir = generate_reference_db(str(REFERENCE), str(_REFDB_DIR))
    elapsed = time.perf_counter() - t0

    npz   = _REFDB_DIR / "reference_embeddings.npz"
    faiss_file = _REFDB_DIR / "reference.faiss"
    ids_json   = _REFDB_DIR / "reference_ids.json"

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Output dir     : {result_dir}")
    print(f"  embeddings.npz : {npz.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  reference.faiss: {faiss_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  reference_ids  : {ids_json.stat().st_size / 1024:.0f} KB")
    print(f"  Time           : {mins}m {secs}s")
    print(f"  VRAM           : {_vram_status()}")

    assert npz.exists()
    assert faiss_file.exists()
    assert ids_json.exists()

    ids, matrix = load_embeddings(str(npz))
    print(f"  Proteins       : {len(ids)}")
    print(f"  Matrix shape   : {matrix.shape}")
    assert len(ids) > 1000, f"Expected >1000 proteins, got {len(ids)}"
    assert matrix.shape[1] == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Test 7 — FAISS index stats
# ---------------------------------------------------------------------------

def test_build_faiss_index():
    """The FAISS index must have correct dimensions and entry count."""
    import faiss as faiss_mod
    import json

    faiss_file = _REFDB_DIR / "reference.faiss"
    ids_json   = _REFDB_DIR / "reference_ids.json"

    assert faiss_file.exists(), "Run test_generate_reference_db first"

    index = faiss_mod.read_index(str(faiss_file))
    with open(ids_json) as fh:
        id_map = json.load(fh)

    print(f"\n  Index dimension : {index.d}")
    print(f"  Index size      : {index.ntotal}")
    print(f"  ID map entries  : {len(id_map)}")
    print(f"  FAISS file size : {faiss_file.stat().st_size / 1024 / 1024:.1f} MB")

    assert index.d == EMBEDDING_DIM
    assert index.ntotal == len(id_map)
    assert index.ntotal > 1000


# ---------------------------------------------------------------------------
# Test 8 — full pipeline: embed low-confidence → search
# ---------------------------------------------------------------------------

def test_search_similar_full_pipeline(tmp_path):
    """End-to-end: embed query proteins → FAISS search → summary."""
    from gemiz.pipeline.prodigal import call_genes
    from gemiz.pipeline.alignment import align_proteins, parse_alignment, classify_proteins

    faiss_file = _REFDB_DIR / "reference.faiss"
    ids_json   = _REFDB_DIR / "reference_ids.json"
    assert faiss_file.exists(), "Run test_generate_reference_db first"

    # --- Step 1: gene calling ---
    faa = call_genes(str(GENOME), str(tmp_path / "genes"))

    # --- Step 2: MMseqs2 alignment + classify ---
    all_proteins_seqs = _read_faa(Path(faa))
    all_ids = list(all_proteins_seqs.keys())

    tsv = align_proteins(
        query_faa=faa,
        reference_db=str(REFERENCE),
        output_dir=str(tmp_path / "alignment"),
        sensitivity=7.5,
        threads=4,
    )
    aln = parse_alignment(tsv)
    classified = classify_proteins(all_ids, aln)
    low_conf_ids = classified["low_confidence"]

    print(f"\n  Low confidence proteins from MMseqs2: {len(low_conf_ids)}")

    # --- Step 3: embed low-confidence proteins ---
    low_seqs = {pid: all_proteins_seqs[pid] for pid in low_conf_ids}
    npz = str(tmp_path / "low_conf_embeddings.npz")

    t0 = time.perf_counter()
    embed_proteins(low_seqs, npz)
    embed_time = time.perf_counter() - t0

    lc_ids, lc_matrix = load_embeddings(npz)

    # --- Step 4: FAISS search ---
    results = search_similar(
        query_ids=lc_ids,
        query_embeddings=lc_matrix,
        faiss_index_path=str(faiss_file),
        ids_json_path=str(ids_json),
        top_k=5,
        min_similarity=0.7,
    )

    matched   = sum(1 for h in results.values() if h)
    unmatched = sum(1 for h in results.values() if not h)

    print(f"\n  --- ESM C Search Summary ---")
    print(f"  Searched:               {len(lc_ids)} proteins")
    print(f"  Found match (>=0.70):   {matched} ({100 * matched / len(lc_ids):.1f}%)")
    print(f"  No match found:         {unmatched} ({100 * unmatched / len(lc_ids):.1f}%)")
    print(f"  Embedding time:         {embed_time:.1f}s")
    print(f"  VRAM:                   {_vram_status()}")

    # Print top 3 results for first 5 matched proteins
    shown = 0
    print(f"\n  --- Sample Matches ---")
    for qid, hits in results.items():
        if hits and shown < 5:
            print(f"  {qid}:")
            for rank, h in enumerate(hits[:3], 1):
                print(f"    {rank}. {h['ref_id']}  similarity={h['similarity']:.4f}")
            shown += 1

    assert len(results) == len(lc_ids)
