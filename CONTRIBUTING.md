# Contributing

Gemiz is alpha research software. Keep changes small, reproducible, and tied to benchmark artifacts when they affect reconstruction quality or runtime.

## Local Setup

```bash
python -m pip install -e ".[dev]"
python scripts/download_mmseqs.py
```

Use Linux or WSL2 for full reconstruction because the MMseqs2 workflow is Linux-oriented.

## Checks

```bash
python -m ruff check .
python -m pytest tests/test_cli.py tests/test_quality.py -q
```

Large biological assets are generated locally and should not be committed.

## Benchmark Claims

For any accuracy or speed claim, include the genome, database versions, competitor versions, command lines, hardware, outputs, and quality summaries.

