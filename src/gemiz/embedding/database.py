"""FAISS-based reference database for ESM C embeddings.

The reference database maps ESM C embedding vectors to known metabolic
protein IDs.  At query time we find the nearest neighbours via cosine
similarity (``IndexFlatIP`` on L2-normalised vectors).

Workflow
--------
  generate_reference_db()   — ONE-TIME: embed reference proteins, build index
  download_reference_db()   — download pre-computed index from HuggingFace
  search_similar()          — query the index with new embeddings
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

HUGGINGFACE_REPO = "gemiz-team/gemiz-reference-db"
LOCAL_CACHE = Path.home() / ".gemiz" / "reference_db"


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings_npz: str,
    output_dir: str,
) -> tuple[str, str]:
    """Build a FAISS ``IndexFlatIP`` from saved embeddings.

    Vectors are L2-normalised so that inner-product == cosine similarity.

    Saves
    -----
    ``reference.faiss``      — the FAISS index file.
    ``reference_ids.json``   — ``{index: protein_id}`` mapping.

    Returns
    -------
    (faiss_path, ids_json_path)
    """
    import faiss

    from gemiz.embedding.esm import EMBEDDING_DIM, load_embeddings

    ids, matrix = load_embeddings(embeddings_npz)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[gemiz] Building FAISS index "
        f"({len(ids)} proteins, dim={EMBEDDING_DIM})..."
    )

    vectors = matrix.astype(np.float32).copy()
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    faiss_path = out / "reference.faiss"
    ids_path = out / "reference_ids.json"

    faiss.write_index(index, str(faiss_path))
    with ids_path.open("w") as fh:
        json.dump({str(i): pid for i, pid in enumerate(ids)}, fh)

    print("[gemiz] Index ready. ~2ms per query.")
    return str(faiss_path), str(ids_path)


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def search_similar(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    faiss_index_path: str,
    ids_json_path: str,
    top_k: int = 5,
    min_similarity: float = 0.7,
) -> dict[str, list[dict]]:
    """Find the most similar reference proteins for each query embedding.

    Only hits above *min_similarity* are returned.  Proteins below the
    threshold get an empty list (truly novel — no reference match).

    Returns
    -------
    dict
        ``{query_id: [{"ref_id": str, "similarity": float}, ...]}``
    """
    import faiss

    print(f"[gemiz] Searching {len(query_ids)} proteins against reference...")
    t0 = time.perf_counter()

    index = faiss.read_index(faiss_index_path)
    with open(ids_json_path) as fh:
        id_map = json.load(fh)

    vectors = query_embeddings.astype(np.float32).copy()
    faiss.normalize_L2(vectors)

    similarities, indices = index.search(vectors, top_k)

    results: dict[str, list[dict]] = {}
    for i, qid in enumerate(query_ids):
        hits = []
        for sim, idx in zip(similarities[i], indices[i]):
            if idx != -1 and float(sim) >= min_similarity:
                hits.append({
                    "ref_id": id_map[str(idx)],
                    "similarity": round(float(sim), 4),
                })
        results[qid] = hits

    elapsed = time.perf_counter() - t0
    matched = sum(1 for h in results.values() if h)
    print(f"[gemiz] Search complete in {elapsed:.1f}s ({matched}/{len(query_ids)} matched)")
    return results


# ---------------------------------------------------------------------------
# Reference database generation (one-time developer job)
# ---------------------------------------------------------------------------

def generate_reference_db(
    proteins_faa: str,
    output_dir: str,
    device: Optional[str] = None,
) -> str:
    """Embed all reference proteins and build a FAISS index.

    This is a ONE-TIME operation, typically run by the developer:

    - iML1515 (~1 500 proteins):    ~5–8 min on RTX 3070
    - full BiGG+ModelSEED (~530 k): ~14 h overnight

    Returns the path to *output_dir*.
    """
    from Bio import SeqIO

    from gemiz.embedding.esm import embed_proteins

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sequences: dict[str, str] = {}
    for record in SeqIO.parse(proteins_faa, "fasta"):
        sequences[record.id] = str(record.seq)

    print(f"[gemiz] Generating reference DB: {len(sequences)} proteins")

    npz_path = str(out / "reference_embeddings.npz")
    embed_proteins(sequences, npz_path, device)

    build_faiss_index(npz_path, str(out))

    print(f"[gemiz] Reference DB ready at {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Pre-built database download
# ---------------------------------------------------------------------------

def download_reference_db(
    output_dir: Optional[str] = None,
) -> str:
    """Download the pre-computed reference DB from HuggingFace.

    Skips the download if all files are already cached locally.

    Returns the path to the cache directory.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = Path(output_dir) if output_dir else LOCAL_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "reference_embeddings.npz",
        "reference.faiss",
        "reference_ids.json",
    ]

    if all((cache_dir / f).exists() for f in files):
        print(f"[gemiz] Reference DB already cached at {cache_dir}")
        return str(cache_dir)

    print("[gemiz] Downloading reference database...")
    for fname in files:
        hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename=fname,
            local_dir=str(cache_dir),
            repo_type="dataset",
        )

    print(f"[gemiz] Reference DB cached at {cache_dir}")
    return str(cache_dir)
