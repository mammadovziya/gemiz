# gemiz

**gemiz** reconstructs genome-scale metabolic models (GEMs) from raw genome FASTA files — no paid software required.

It replaces CarveMe's DIAMOND + Gurobi stack with:

| Step | Tool | Replaces |
|---|---|---|
| Gene calling | [pyrodigal](https://github.com/althonos/pyrodigal) | Prodigal |
| Sequence alignment | [MMseqs2](https://github.com/soedinglab/MMseqs2) | DIAMOND |
| Structure similarity | [ESM C 600M](https://github.com/evolutionaryscale/esm) | — |
| MILP solver | [HiGHS](https://highs.dev) | Gurobi |
| Reaction template | CarveMe bacteria universe (5 532 reactions) | — |

## Quick start

```bash
# 1. Install
pip install gemiz[embeddings]          # with ESM C support
# or: pip install gemiz                 # MMseqs2 only, faster

# 2. Download MMseqs2 binary (not bundled in git, included in PyPI wheels)
python scripts/download_mmseqs.py

# 3. Build the universal protein database (~2–4 h, one-time)
python scripts/build_universal_db.py

# 4. Reconstruct a GEM
gemiz carve genome.fna -o model.xml
```

That's it. `gemiz` auto-detects universal mode when `data/universal/db/` exists.

## Installation

**Python 3.11+** required.

```bash
git clone https://github.com/your-org/gemiz
cd gemiz
pip install -e ".[embeddings,dev]"
python scripts/download_mmseqs.py
```

### Optional: GPU acceleration

```bash
pip install gemiz[gpu]      # swaps faiss-cpu for faiss-gpu
```

## Usage

### Universal mode (recommended — no organism config needed)

```bash
# First time: build the universal database
python scripts/build_universal_db.py   # downloads all 108 BiGG models

# Reconstruct any bacterial genome
gemiz carve genome.fna -o model.xml

# From NCBI accession
gemiz carve --refseq GCF_000005845.2 -o ecoli_model.xml

# Faster (skip ESM C embeddings)
gemiz carve genome.fna --no-esm -o model.xml

# Different growth medium
gemiz carve genome.fna --media LB -o model.xml
```

### Organism-specific mode

```bash
# Set up reference data for a known organism
python scripts/setup_organism.py ecoli \
    --ncbi-assembly GCF_000005845.2 \
    --gold-standard data/universal/iML1515.xml

# Reconstruct using the organism's own reference
gemiz carve genome.fna --organism ecoli -o model.xml
```

### Other commands

```bash
gemiz info   model.xml          # print reaction/metabolite/gene counts
gemiz validate model.xml        # run FBA feasibility check
```

## Pipeline

```
genome.fna
  │
  ├─[1] pyrodigal           → proteins.faa         (gene calling)
  ├─[2] MMseqs2             → alignment hits        (sequence similarity)
  ├─[3] ESM C 600M          → embeddings + FAISS    (structure similarity, optional)
  ├─[3.5] Media constraints → closes off-media EX_  (universal mode)
  ├─[4] Reaction scoring    → score per reaction     (GPR evaluation)
  ├─[5] HiGHS MILP          → reaction subset        (carving)
  ├─[5.5] Gap-filling       → restore growth         (if needed)
  └─[6] SBML export         → model.xml
```

### Scoring

Each reaction gets a score in `[-1.0, 1.0]` based on its GPR rule:

| Score | Meaning |
|---|---|
| `> 0.7` | Strong evidence (≥50% sequence identity) |
| `0.3–0.7` | Blended MMseqs2 + ESM C (twilight zone) |
| `0.0–0.3` | Weak evidence (ESM C only) |
| `0.0` | Spontaneous / no GPR |
| `-1.0` | No evidence (penalised in MILP) |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/build_universal_db.py` | Build universal protein DB from all 108 BiGG models |
| `scripts/setup_organism.py` | Set up reference data for a specific organism |
| `scripts/download_mmseqs.py` | Download MMseqs2 binary for the current platform |
| `scripts/benchmark.py` | Compare tool models against gold-standard references |
| `scripts/validate_essentiality.py` | Gene essentiality vs Keio collection (E. coli) |
| `scripts/compare_tools.py` | Cross-tool F1 comparison table |
| `scripts/build_cross_templates.py` | Leave-one-out cross-organism templates |
| `scripts/esm_degradation_test.py` | Test ESM C recovery at degraded alignment coverage |

## Data

Large files are **not committed** to this repository. After cloning:

```
data/
  universal/
    media_db.tsv          ← committed (CarveMe media definitions)
  reference/
    ecoli_feature_table.txt  ← committed (NCBI, ~2 MB)
```

Files generated at runtime (gitignored):

```
data/universal/
  carveme_bacteria.xml    ← CarveMe bacteria universe (14 MB)
  iML1515.xml             ← E. coli gold standard (11 MB)
  db/                     ← universal_proteins.faa, universal_gpr.csv, mmseqs_db/

src/gemiz/bin/mmseqs/
  mmseqs-linux-avx2       ← downloaded by scripts/download_mmseqs.py
  mmseqs-linux-sse41
  mmseqs-linux-arm64
  mmseqs-mac-universal
```

## Benchmarks

CarveMe bacteria universe template vs iML1515 (E. coli K-12):

| Metric | Value |
|---|---|
| Precision | 0.784 |
| Recall | 0.774 |
| F1 | 0.779 |

(Exchange reactions excluded from comparison.)

## Requirements

- Python ≥ 3.11
- Linux or macOS (Windows: use WSL2)
- MMseqs2 binary (`python scripts/download_mmseqs.py`)
- ~4 GB RAM for universal mode; ~16 GB for ESM C embeddings

## License

MIT
