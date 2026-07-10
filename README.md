# gemiz

Experimental genome-scale metabolic model reconstruction for bacteria.

`gemiz` is an alpha-stage research project for reconstructing draft
genome-scale metabolic models (GEMs) from bacterial genome FASTA files. It
combines gene calling, protein homology search, optional protein language model
embeddings, reaction scoring, COBRApy model handling, MILP-based model carving,
gap filling, and SBML export.

The project is under active development. It is useful for experimentation,
benchmarking, and methods development, but it is not yet a polished
production-grade replacement for mature GEM reconstruction tools.

Keywords: genome-scale metabolic model reconstruction, GEM reconstruction,
constraint-based modeling, COBRApy, systems biology, metabolic network
reconstruction, bacterial genomes, BiGG models, iML1515, MMseqs2, ESM-C,
protein embeddings, FAISS, HiGHS, SBML, flux balance analysis.

## Project Status

| Area | Current status |
| --- | --- |
| Package maturity | Alpha research prototype (`0.1.0`) |
| Main target | Bacterial GEM reconstruction |
| Eukaryotes | Not supported yet |
| Installation | Source install only; not presented as a PyPI release |
| Core pipeline | Implemented, but requires local reference data and binaries |
| ESM-C support | Optional experimental scoring signal |
| Benchmarking | Scripts exist; results should be regenerated locally before making claims |
| Windows support | Development works on Windows, but reconstruction benchmarks should run in WSL2/Linux |

## What gemiz Does Today

- Calls protein-coding genes from bacterial genomes with `pyrodigal`.
- Aligns predicted proteins against reference proteins with MMseqs2.
- Optionally embeds low-confidence proteins with ESM-C 600M and searches a FAISS reference index.
- Converts protein-level evidence into reaction-level scores through GPR rules and reference mappings.
- Uses COBRApy models as templates and exports SBML models.
- Uses HiGHS/COBRApy-based optimization for reaction selection and gap filling.
- Includes scripts for preparing E. coli benchmark data, inspecting model quality, and comparing reconstructed models with gold-standard SBML models.

## What gemiz Does Not Claim Yet

- It does not claim to reconstruct high-quality models for every bacterium.
- It does not yet support eukaryotic reconstruction.
- It does not claim to outperform CarveMe, ModelSEED, gapseq, or other tools without running the benchmark workflow on the same input data and hardware.
- It does not claim that ESM-C improves reconstruction accuracy until the ablation benchmark has been run and reported.
- It does not ship large reference databases in git.
- It is not currently documented as a one-command PyPI installation.

## Why This Project Exists

Most GEM reconstruction pipelines rely heavily on sequence homology and curated
reaction templates. `gemiz` explores whether protein language model embeddings
can add useful signal when homology is weak, especially for proteins in the
sequence-identity twilight zone.

The working hypothesis is:

> Homology search should stay the primary evidence source for high-confidence
> enzyme assignments, while protein embeddings may help rank or rescue
> lower-confidence gene-reaction mappings.

This is a research direction, not yet a validated conclusion.

## Pipeline Overview

```text
genome.fna
  |
  +-- [1] pyrodigal gene calling
  |        -> proteins.faa
  |
  +-- [2] MMseqs2 protein alignment
  |        -> sequence similarity hits
  |
  +-- [3] optional ESM-C 600M embeddings
  |        -> embedding similarity hits through FAISS
  |
  +-- [4] reaction scoring
  |        -> score per reaction from GPR evidence
  |
  +-- [5] model carving / reaction selection
  |        -> draft organism model
  |
  +-- [6] gap filling and FBA checks
  |        -> growth-feasible model when possible
  |
  +-- [7] SBML export
           -> model.xml
```

## Methods

### Gene Calling

`gemiz` uses `pyrodigal`, a Python/Cython implementation of Prodigal, to predict
protein-coding genes from input genome FASTA files.

### Homology Search

Protein alignment is performed with MMseqs2. High-identity matches are treated
as the strongest source of evidence for gene-reaction mapping.

### Optional Embedding Signal

For low-confidence proteins, `gemiz` can use ESM-C 600M embeddings and FAISS
nearest-neighbor search against a reference embedding database. This path is
optional because it requires additional dependencies and may require GPU memory
for practical runtimes.

### Reaction Scoring

Reaction scores are derived from protein evidence and gene-protein-reaction
(GPR) rules:

| Score range | Interpretation |
| --- | --- |
| `> 0.7` | Strong evidence |
| `0.3` to `0.7` | Moderate or blended evidence |
| `0.0` to `0.3` | Weak evidence |
| `0.0` | No GPR / spontaneous / neutral |
| `-1.0` | Enzyme reaction with no supporting evidence |

The thresholds are experimental and should be tuned with validation data before
being used for biological conclusions.

## Repository Layout

```text
src/gemiz/
  cli.py                         command-line interface
  pipeline/                      gene calling and alignment helpers
  embedding/                     ESM-C and FAISS embedding utilities
  reconstruction/                scoring, carving, gap filling, full pipeline
  quality.py                     model quality summaries
  io/                            SBML helpers
  utils/                         binary resolution helpers

scripts/
  setup_benchmark_data.py        download E. coli benchmark fixture
  download_mmseqs.py             download MMseqs2 binary
  build_universal_db.py          build local universal reference database
  benchmark.py                   compare models to gold standards
  benchmark_competitors.py       run explicit tool comparison workflow
  model_quality.py               summarize SBML model quality
  validate_essentiality.py       experimental essentiality validation

tests/
  test_*.py                      unit, smoke, and optional integration tests
```

## Installation From Source

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/mammadovziya/gemiz.git
cd gemiz
python -m pip install -e ".[dev]"
```

Optional embedding dependencies:

```bash
python -m pip install -e ".[embeddings,dev]"
```

On native Windows, MMseqs2 is not self-contained. Use WSL2/Linux for full
reconstruction and benchmark runs.

## Prepare Local Data

Large biological assets are intentionally not committed to git.

For the smaller E. coli benchmark fixture:

```bash
python scripts/setup_benchmark_data.py
```

This downloads public data into:

```text
data/genomes/ecoli_k12.fna
data/reference/iML1515_proteins.faa
data/reference/ecoli_feature_table.txt
data/universal/iML1515.xml
```

For MMseqs2:

```bash
python scripts/download_mmseqs.py
```

For the larger universal database:

```bash
python scripts/build_universal_db.py
```

The universal database build downloads and processes many public models and can
take hours. It should be treated as a reproducible local build step, not as data
that is already present after cloning.

## Basic Usage

After installing dependencies and preparing reference data:

```bash
gemiz carve data/genomes/ecoli_k12.fna \
  --template data/universal/iML1515.xml \
  --reference data/reference/iML1515_proteins.faa \
  --feature-table data/reference/ecoli_feature_table.txt \
  --no-esm \
  -o data/test_outputs/ecoli_model.xml
```

Inspect a generated model:

```bash
gemiz info data/test_outputs/ecoli_model.xml
gemiz validate data/test_outputs/ecoli_model.xml
python scripts/model_quality.py data/test_outputs/ecoli_model.xml
```

## Benchmarking

Do not copy benchmark numbers from this README into papers, posters, resumes, or
claims. Generate them from the current code and data.

Prepare the E. coli fixture:

```bash
python scripts/setup_benchmark_data.py
```

Run a gemiz-only benchmark:

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz
```

If competitor tools are installed, include them explicitly:

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz carveme gapseq
```

The benchmark summary records commands, runtime, model statistics, growth
status, precision, recall, and F1 where possible.

## Testing

Run the lightweight test suite:

```bash
pytest -q
```

Some tests are skipped unless optional benchmark data, MMseqs2 binaries, ESM-C
dependencies, or WSL2/Linux are available. This is intentional: large public
reference files are downloaded locally instead of committed to the repository.

## Development Roadmap

- Make the source install and benchmark setup reproducible from a fresh clone.
- Keep optional data-dependent tests skipped until fixtures are downloaded.
- Add clear benchmark artifacts for E. coli reconstruction against iML1515.
- Add ablation studies comparing homology-only scoring with homology plus ESM-C.
- Improve reaction evidence sidecars and model quality reports.
- Document limitations around templates, GPR mappings, biomass reactions, media constraints, and gap filling.
- Clean stale DIAMOND/Gurobi skeleton references from older modules.
- Expand validation beyond E. coli only after the first benchmark path is stable.

## Suggested GitHub Repository Metadata

Description:

```text
Alpha Python toolkit for bacterial genome-scale metabolic model reconstruction using pyrodigal, MMseqs2, COBRApy, optional ESM-C protein embeddings, FAISS, and HiGHS.
```

Topics:

```text
bioinformatics
systems-biology
genome-scale-metabolic-models
metabolic-modeling
constraint-based-modeling
cobrapy
flux-balance-analysis
sbml
bigg-models
bacterial-genomics
protein-embeddings
esm-c
mmseqs2
faiss
highs
python
```

## Citation and Responsible Use

This repository is a research prototype. Reconstructed GEMs should be treated as
draft models and inspected before biological interpretation. Any performance or
accuracy claim should include the exact input genome, reference model, database
version, tool versions, hardware, command line, and benchmark output.

## License

MIT
