# Gemiz

Open-source bacterial genome-scale metabolic model reconstruction from raw genome FASTA files.

Gemiz is a Python toolkit for reconstructing draft genome-scale metabolic models (GEMs) from bacterial genomes using free and open-source software. It combines gene calling, protein homology search, reaction evidence scoring, COBRApy model handling, HiGHS optimization, gap filling, SBML export, model-quality checks, and reproducible benchmarking against tools such as CarveMe, gapseq, ModelSEED, and curated BiGG models.

Gemiz is designed for researchers, bioinformaticians, systems biologists, metabolic engineers, and computational biology teams who need a transparent, scriptable, no-paid-software workflow for bacterial GEM reconstruction.

Keywords: genome-scale metabolic model reconstruction, GEM reconstruction, bacterial metabolic model reconstruction, metabolic network reconstruction, constraint-based modeling, flux balance analysis, FBA, COBRApy, SBML, BiGG models, ModelSEED, CarveMe alternative, gapseq alternative, pyrodigal, MMseqs2, HiGHS, systems biology, bacterial genomics, metabolic engineering, bioinformatics.

## Status

Gemiz is an alpha-stage research project. The open-source repository contains the core reconstruction and benchmarking pipeline. The ESM-C protein-language-model enhancement is being developed privately and is not part of the currently available public feature set. Public Gemiz should be treated as a transparent homology-first GEM reconstruction toolkit with experimental hooks for future embedding-assisted scoring.

| Area | Public status |
| --- | --- |
| Raw genome FASTA input | Available |
| Bacterial gene calling | Available |
| MMseqs2 homology search | Available |
| Universal bacterial template import | Available |
| COBRApy/HiGHS model carving | Available |
| Gap filling | Available |
| SBML export | Available |
| Model quality reports | Available |
| Competitor benchmarking | Available |
| ESM-C enhanced scoring | Private R&D / not publicly released |
| Eukaryotic GEM reconstruction | Not supported yet |

## Why Gemiz

Most automated GEM reconstruction tools are difficult to inspect, hard to benchmark fairly, or depend on large external databases and hidden assumptions. Gemiz focuses on a reproducible open workflow:

- Raw bacterial genome FASTA in, draft SBML GEM out.
- No paid solver required; optimization uses HiGHS and COBRApy.
- MMseqs2-based protein homology search for fast evidence collection.
- Universal bacterial reconstruction mode using open CarveMe/BiGG assets.
- Per-reaction evidence sidecar JSON for auditability.
- Reproducible benchmarking against gold-standard SBML models.
- Explicit comparison scripts for Gemiz, CarveMe, gapseq, and external models.

Gemiz is not presented as a black box. Every major step has code, command-line output, and saved artifacts that can be inspected.

## Available Features

### Genome To GEM Pipeline

Gemiz can reconstruct a draft bacterial metabolic model from a raw genome FASTA:

```text
genome.fna
  -> pyrodigal gene calling
  -> proteins.faa
  -> MMseqs2 homology search
  -> reaction evidence scoring
  -> HiGHS MILP model carving
  -> gap filling and FBA validation
  -> SBML model.xml
  -> evidence JSON sidecar
```

### Universal Bacterial Reconstruction

The public pipeline can import open CarveMe/BiGG bacterial assets into Gemiz's local universal database layout:

```bash
python scripts/import_carveme_assets.py
```

This creates ignored local data files:

```text
data/universal/carveme_bacteria.xml
data/universal/db/universal_proteins.faa
data/universal/db/universal_gpr.csv
data/universal/db/mmseqs_db/
```

After that, Gemiz can run in universal bacterial mode without using an organism's own gold-standard model as the reconstruction template.

### Evidence-Aware Reaction Selection

Gemiz scores reactions from protein evidence and GPR rules. Reactions with strong sequence evidence are preferred, unsupported enzyme reactions are penalized, and neutral no-GPR reactions receive a small penalty so the model does not keep arbitrary free reactions unless they are needed for growth.

### Gap Filling

Gemiz first attempts COBRApy gap filling, then uses a prioritized open fallback for common single-reaction growth fixes such as sink, demand, exchange, and transport reactions. This avoids paid solvers and reduces failure cases in large universal templates.

### Benchmarking

Gemiz includes a benchmark runner that records:

- command line
- runtime
- growth status and growth rate
- reaction precision, recall, and F1 against a gold-standard SBML model
- model quality summary
- skipped/error status for missing competitor tools

Example:

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz carveme
```

Benchmark claims should always be regenerated on the same hardware, same input genome, same databases, and same tool versions.

## ESM-C Roadmap

ESM-C support is a private development track. The goal is to improve reconstruction when homology is weak by using protein language model embeddings as an additional signal for low-confidence proteins.

Current public repository:

- Homology-first reconstruction is available.
- Universal bacterial reconstruction is available.
- Benchmarking and model-quality tooling are available.
- ESM-C enhanced production scoring is not publicly released yet.

Planned private ESM-C direction:

- Embed low-confidence proteins.
- Search against a reference embedding index.
- Blend homology and embedding evidence.
- Run ablation studies comparing homology-only reconstruction with homology plus ESM-C.
- Release only after accuracy, runtime, and reproducibility are validated.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/mammadovziya/gemiz.git
cd gemiz
python -m pip install -e ".[dev]"
```

On Windows, use WSL2/Linux for full reconstruction and benchmarks because MMseqs2 is Linux-oriented in this project workflow.

Download or verify the bundled MMseqs2 binary:

```bash
python scripts/download_mmseqs.py
```

## Prepare Data

Large biological assets are not committed to git.

For the small E. coli benchmark fixture:

```bash
python scripts/setup_benchmark_data.py
```

For broad bacterial universal mode:

```bash
python scripts/import_carveme_assets.py
```

For a larger from-source universal database rebuilt from public BiGG/NCBI data:

```bash
python scripts/build_universal_db.py
```

The full from-source build can take hours and should be treated as a reproducible local data build step.

## Usage

Run Gemiz in universal bacterial mode after importing the open assets:

```bash
gemiz carve data/genomes/ecoli_k12.fna \
  --no-esm \
  --threads 4 \
  -o data/test_outputs/ecoli_model.xml
```

Run with explicit template/reference files:

```bash
gemiz carve data/genomes/ecoli_k12.fna \
  --template data/universal/iML1515.xml \
  --reference data/reference/iML1515_proteins.faa \
  --feature-table data/reference/ecoli_feature_table.txt \
  --no-esm \
  -o data/test_outputs/ecoli_model.xml
```

Inspect a model:

```bash
gemiz info data/test_outputs/ecoli_model.xml
gemiz validate data/test_outputs/ecoli_model.xml
python scripts/model_quality.py data/test_outputs/ecoli_model.xml
```

## Benchmarking Against Competitors

Gemiz is built to be compared, not merely advertised. Use the benchmark runner before making claims about accuracy or speed.

Gemiz only:

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz
```

Gemiz and CarveMe:

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz carveme
```

Gemiz and gapseq, if gapseq is installed:

```bash
python scripts/benchmark_competitors.py \
  --organism ecoli \
  --genome data/genomes/ecoli_k12.fna \
  --gold-standard data/universal/iML1515.xml \
  --tools gemiz gapseq \
  --gapseq-command-template "gapseq ... {genome} ... {model}"
```

## Repository Layout

```text
src/gemiz/
  cli.py                         command-line interface
  pipeline/                      gene calling and alignment
  reconstruction/                scoring, carving, gap filling, full pipeline
  quality.py                     model quality summaries
  io/                            SBML helpers
  utils/                         binary resolution helpers

scripts/
  setup_benchmark_data.py        download E. coli benchmark fixture
  download_mmseqs.py             download MMseqs2 binary
  import_carveme_assets.py       import open CarveMe/BiGG universal assets
  build_universal_db.py          build universal database from public sources
  benchmark_competitors.py       compare Gemiz with CarveMe, gapseq, ModelSEED
  model_quality.py               summarize SBML model quality
  validate_essentiality.py       experimental essentiality validation

tests/
  test_*.py                      unit, smoke, and optional integration tests
```

## Testing

```bash
pytest -q
```

Some tests are skipped unless optional local data has been downloaded. This is intentional; large biological databases are generated locally instead of committed to the repository.

## What Gemiz Does Not Claim Yet

- It is not a finished production-grade GEM reconstruction platform.
- It does not yet support eukaryotes.
- It does not publicly ship the private ESM-C enhanced reconstruction system.
- It does not guarantee high-quality models for every bacterial genome.
- It does not make universal performance claims without benchmark artifacts.
- It does not replace manual curation for publishable metabolic models.

## Roadmap

- Project query-gene GPR rules into universal SBML outputs.
- Reduce runtime in universal bacterial mode.
- Add more organism benchmarks beyond E. coli.
- Add gapseq benchmark examples once an installation command is available.
- Improve media handling and growth-condition reporting.
- Continue private ESM-C development and release only after validated ablation benchmarks.

## Suggested GitHub Metadata

Description:

```text
Open-source bacterial genome-scale metabolic model reconstruction from FASTA using pyrodigal, MMseqs2, COBRApy, HiGHS, SBML, and reproducible benchmarking.
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
modelseed
carveme-alternative
gapseq-alternative
bacterial-genomics
metabolic-engineering
mmseqs2
highs
python
```

## Responsible Use

Gemiz produces draft metabolic models. Always inspect reconstructed GEMs before biological interpretation, publication, strain design, or downstream simulation. Report input genomes, database versions, command lines, hardware, tool versions, and benchmark artifacts with any accuracy or speed claim.

## License

MIT
