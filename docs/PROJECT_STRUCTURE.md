# Project Structure

```text
src/gemiz/          Python package and CLI
scripts/            Data setup, benchmarking, and research utilities
tests/              Unit, smoke, and optional integration tests
data/               Small tracked fixtures plus ignored generated assets
docs/               Project notes and contributor-facing documentation
```

Local competitor checkouts such as `carveme/` and `gapseq/` are ignored. Use the benchmark scripts to call those tools, but do not vendor their repositories into Gemiz.

Generated files that should stay out of git include downloaded genomes, SBML templates, MMseqs2 databases, model outputs, benchmark results, caches, virtual environments, and editor/agent-local settings.

