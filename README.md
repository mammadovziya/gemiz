# Gemiz

Gemiz is an alpha Python toolkit for FASTA-to-SBML bacterial genome-scale metabolic model (GEM) reconstruction.

```text
genome.fna -> pyrodigal -> MMseqs2 -> reaction scoring -> COBRApy/HiGHS -> SBML
```

Gemiz outputs a draft SBML model plus evidence and quality artifacts for inspection and benchmarking.

## Status

- Alpha research code.
- Bacterial GEM reconstruction only; eukaryotes are not supported.
- Public reconstruction is homology-first. ESM-C support is experimental/private R&D, not a validated public feature.
- Large biological databases are downloaded or built locally, not committed to git.

## Install

Python 3.11+ is recommended. Use Linux or WSL2 for full MMseqs2 workflows.

```bash
git clone https://github.com/mammadovziya/gemiz.git
cd gemiz
python -m pip install -e ".[dev]"
python scripts/download_mmseqs.py
```

## Quickstart

```bash
python scripts/setup_benchmark_data.py
python scripts/import_carveme_assets.py

gemiz carve data/genomes/ecoli_k12.fna \
  --no-esm \
  --threads 4 \
  -o data/test_outputs/ecoli_model.xml

gemiz info data/test_outputs/ecoli_model.xml
gemiz validate data/test_outputs/ecoli_model.xml
```

## Benchmark

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz carveme
```

Report hardware, database versions, tool versions, commands, and artifacts with any speed or accuracy claim.

## Test

```bash
pytest -q
```

## Roadmap

- Project genes/GPRs into universal SBML output.
- Improve runtime in universal bacterial mode.
- Add a multi-organism benchmark panel.
- Add CI, docs, releases, and package distribution.

## License

MIT
