#!/usr/bin/env python3
"""Run time/accuracy benchmarks for gemiz and competitor tools.

This script is intentionally explicit: it records skipped tools, commands,
runtime, model statistics, and reaction-overlap metrics against one gold
standard model. Use it to build the evidence needed before claiming gemiz is
faster or more accurate than competitors.

Examples
--------
    python scripts/benchmark_competitors.py \
        --organism ecoli \
        --genome data/genomes/ecoli_k12.fna \
        --gold-standard data/universal/iML1515.xml \
        --tools gemiz carveme

    python scripts/benchmark_competitors.py \
        --organism ecoli \
        --genome data/genomes/ecoli_k12.fna \
        --gold-standard data/universal/iML1515.xml \
        --tools gemiz gapseq \
        --gapseq-command-template "gapseq ... {genome} ... {model}"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gemiz.quality import summarize_model_file  # noqa: E402


def _env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SRC_ROOT) if not current else f"{SRC_ROOT}{os.pathsep}{current}"
    )
    return env


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:] if text else ""


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    runtime = time.perf_counter() - start
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "runtime_seconds": round(runtime, 3),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def _same_file(a: Path | None, b: Path | None) -> bool:
    """Best-effort file identity check that tolerates missing files."""
    if a is None or b is None:
        return False
    try:
        return a.resolve() == b.resolve() or a.samefile(b)
    except OSError:
        return a.resolve() == b.resolve()


def _infer_gemiz_template(args: argparse.Namespace) -> Path | None:
    """Infer the template gemiz will use when --gemiz-template is omitted."""
    if args.gemiz_template is not None:
        return args.gemiz_template

    # Mirrors the default branch in gemiz.cli: universal mode is active only
    # when the universal protein DB exists and no explicit template/reference
    # was supplied.
    if args.gemiz_reference is None and (REPO_ROOT / "data/universal/db/universal_proteins.faa").exists():
        for candidate in (
            REPO_ROOT / "data/universal/carveme_bacteria.xml",
            REPO_ROOT / "data/universal/db/universal_template.xml",
        ):
            if candidate.exists():
                return candidate

    default_template = REPO_ROOT / "data/universal/iML1515.xml"
    return default_template if default_template.exists() else None


def _remove_stale_outputs(model_path: Path, *, genome: Path) -> None:
    """Remove files that could make a failed run look successful."""
    candidates = [
        model_path,
        model_path.with_suffix(model_path.suffix + ".evidence.json"),
    ]
    for path in candidates:
        if path.exists():
            path.unlink()

    work_dir = model_path.parent / f".gemiz_{genome.stem}"
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _normalise_id(rxn_id: str) -> str:
    return rxn_id[2:] if rxn_id.startswith("R_") else rxn_id


def _is_exchange(rxn_id: str) -> bool:
    norm = _normalise_id(rxn_id)
    return norm.startswith("EX_") or norm.startswith("DM_") or norm.startswith("SK_")


def _model_stats(model_path: Path) -> dict[str, Any]:
    import cobra

    model = cobra.io.read_sbml_model(str(model_path))
    sol = model.optimize()
    growth = sol.objective_value if sol.status == "optimal" else 0.0
    return {
        "reactions": len(model.reactions),
        "metabolites": len(model.metabolites),
        "genes": len(model.genes),
        "growth_status": sol.status,
        "growth_rate": round(float(growth or 0.0), 6),
    }


def _reaction_metrics(model_path: Path, gold_path: Path) -> dict[str, Any]:
    import cobra

    model = cobra.io.read_sbml_model(str(model_path))
    gold = cobra.io.read_sbml_model(str(gold_path))

    model_ids = {
        _normalise_id(r.id)
        for r in model.reactions
        if not _is_exchange(r.id)
    }
    gold_ids = {
        _normalise_id(r.id)
        for r in gold.reactions
        if not _is_exchange(r.id)
    }

    tp = len(model_ids & gold_ids)
    fp = len(model_ids - gold_ids)
    fn = len(gold_ids - model_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _finish_result(
    result: dict[str, Any],
    *,
    model_path: Path,
    gold_path: Path | None,
) -> dict[str, Any]:
    result["model_path"] = str(model_path)
    if result.get("returncode") not in (None, 0):
        result["status"] = "error"
    elif model_path.exists():
        result["status"] = "success"
        result["model_stats"] = _model_stats(model_path)
        result["quality"] = summarize_model_file(model_path, include_blocked=False)
        if gold_path is not None and gold_path.exists():
            result["accuracy"] = _reaction_metrics(model_path, gold_path)
    elif result.get("returncode") == 0:
        result["status"] = "error"
        result["error"] = f"Command succeeded but model was not created: {model_path}"
    else:
        result["status"] = "error"
    return result


def run_gemiz(args: argparse.Namespace, tool_dir: Path) -> dict[str, Any]:
    if platform.system() == "Windows":
        return {
            "status": "skipped",
            "reason": "gemiz reconstruction uses MMseqs2; run this benchmark inside WSL2/Linux.",
        }

    model_path = tool_dir / f"{args.organism}_model.xml"
    _remove_stale_outputs(model_path, genome=args.genome)

    cmd = [
        sys.executable,
        "-m",
        "gemiz.cli",
        "carve",
        str(args.genome),
        "-o",
        str(model_path),
        "--threads",
        str(args.threads),
    ]
    if args.gemiz_template:
        cmd.extend(["--template", str(args.gemiz_template)])
    if args.gemiz_reference:
        cmd.extend(["--reference", str(args.gemiz_reference)])
    if args.gemiz_feature_table:
        cmd.extend(["--feature-table", str(args.gemiz_feature_table)])
    if args.gemiz_media:
        cmd.extend(["--media", args.gemiz_media])
    if args.gemiz_no_esm:
        cmd.append("--no-esm")
    if args.gemiz_extra_args:
        cmd.extend(shlex.split(args.gemiz_extra_args))

    result = _run_command(cmd, cwd=REPO_ROOT, timeout=args.timeout)
    effective_template = _infer_gemiz_template(args)
    result["benchmark_context"] = {
        "template": str(effective_template) if effective_template else "gemiz default",
        "reference": str(args.gemiz_reference) if args.gemiz_reference else "gemiz default",
        "feature_table": (
            str(args.gemiz_feature_table) if args.gemiz_feature_table else "gemiz default"
        ),
        "media": args.gemiz_media or "gemiz default",
        "template_equals_gold_standard": _same_file(effective_template, args.gold_standard),
    }
    return _finish_result(result, model_path=model_path, gold_path=args.gold_standard)


def run_carveme(args: argparse.Namespace, tool_dir: Path) -> dict[str, Any]:
    carve_cmd = shutil.which(args.carveme_command)
    if carve_cmd is None:
        return {
            "status": "skipped",
            "reason": f"CarveMe command not found: {args.carveme_command}",
        }

    from gemiz.pipeline.prodigal import call_genes

    model_path = tool_dir / f"{args.organism}_model.xml"
    _remove_stale_outputs(model_path, genome=args.genome)
    protein_dir = tool_dir / "proteins"
    protein_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    proteins = call_genes(str(args.genome), str(protein_dir))
    cmd = [carve_cmd, proteins, "-o", str(model_path)]
    if args.carveme_extra_args:
        cmd.extend(shlex.split(args.carveme_extra_args))
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    runtime = time.perf_counter() - start
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "runtime_seconds": round(runtime, 3),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "preprocessing": "pyrodigal protein FASTA from raw genome",
    }
    return _finish_result(result, model_path=model_path, gold_path=args.gold_standard)


def run_gapseq(args: argparse.Namespace, tool_dir: Path) -> dict[str, Any]:
    if not args.gapseq_command_template:
        return {
            "status": "skipped",
            "reason": "Provide --gapseq-command-template for this installation.",
        }

    model_path = tool_dir / f"{args.organism}_model.xml"
    tool_dir.mkdir(parents=True, exist_ok=True)
    tokens = {
        "genome": str(args.genome),
        "out_dir": str(tool_dir),
        "model": str(model_path),
        "threads": str(args.threads),
        "organism": args.organism,
    }
    command = args.gapseq_command_template.format(**tokens)
    result = _run_command(shlex.split(command), cwd=REPO_ROOT, timeout=args.timeout)
    return _finish_result(result, model_path=model_path, gold_path=args.gold_standard)


def run_external_model(
    args: argparse.Namespace,
    tool_dir: Path,
    *,
    tool_name: str,
    model_path: Path | None,
) -> dict[str, Any]:
    if model_path is None:
        return {
            "status": "skipped",
            "reason": f"Provide --{tool_name}-model to include this tool.",
        }
    dest = tool_dir / f"{args.organism}_model.xml"
    tool_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, dest)
    return _finish_result(
        {"status": "success", "runtime_seconds": None, "source_model": str(model_path)},
        model_path=dest,
        gold_path=args.gold_standard,
    )


def print_summary(results: dict[str, Any]) -> None:
    print("\nBenchmark summary")
    print("=================")
    for tool, row in results["tools"].items():
        status = row.get("status")
        runtime = row.get("runtime_seconds")
        f1 = row.get("accuracy", {}).get("f1")
        growth = row.get("model_stats", {}).get("growth_rate")
        bits = [
            f"{tool:<10}",
            f"status={status}",
            f"time={runtime}s" if runtime is not None else "time=n/a",
            f"f1={f1}" if f1 is not None else "f1=n/a",
            f"growth={growth}" if growth is not None else "growth=n/a",
        ]
        if status == "skipped":
            bits.append(f"reason={row.get('reason')}")
        print("  " + "  ".join(bits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gemiz against competitors on one genome.",
    )
    parser.add_argument("--organism", required=True, help="Organism label, e.g. ecoli.")
    parser.add_argument("--genome", required=True, type=Path, help="Raw genome FASTA.")
    parser.add_argument(
        "--gold-standard",
        type=Path,
        default=None,
        help="Gold-standard SBML model for reaction-overlap accuracy metrics.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["gemiz", "carveme"],
        choices=["gemiz", "carveme", "gapseq", "modelseed"],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/comparison"),
        help="Directory for models and benchmark_summary.json.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--gemiz-no-esm", action="store_true")
    parser.add_argument(
        "--gemiz-template",
        type=Path,
        default=None,
        help="Explicit template SBML for gemiz. Use this for fair cross-template tests.",
    )
    parser.add_argument(
        "--gemiz-reference",
        type=Path,
        default=None,
        help="Explicit reference protein FASTA for gemiz.",
    )
    parser.add_argument(
        "--gemiz-feature-table",
        type=Path,
        default=None,
        help="Explicit NCBI feature table for gemiz ID mapping.",
    )
    parser.add_argument(
        "--gemiz-media",
        default=None,
        help="Media passed to gemiz, e.g. M9, LB, or none.",
    )
    parser.add_argument(
        "--gemiz-extra-args",
        default=None,
        help="Extra arguments appended to gemiz carve, e.g. '--media M9'.",
    )
    parser.add_argument("--carveme-command", default="carve")
    parser.add_argument(
        "--carveme-extra-args",
        default=None,
        help="Extra arguments appended to carve, e.g. '-i M9 -g M9'.",
    )
    parser.add_argument(
        "--gapseq-command-template",
        default=None,
        help="Command template with {genome}, {out_dir}, {model}, {threads}.",
    )
    parser.add_argument(
        "--modelseed-model",
        type=Path,
        default=None,
        help="Existing ModelSEED/KBase SBML output to include in comparison.",
    )
    args = parser.parse_args()

    args.genome = args.genome.resolve()
    if args.gold_standard is not None:
        args.gold_standard = args.gold_standard.resolve()
    if args.gemiz_template is not None:
        args.gemiz_template = args.gemiz_template.resolve()
    if args.gemiz_reference is not None:
        args.gemiz_reference = args.gemiz_reference.resolve()
    if args.gemiz_feature_table is not None:
        args.gemiz_feature_table = args.gemiz_feature_table.resolve()
    if args.modelseed_model is not None:
        args.modelseed_model = args.modelseed_model.resolve()

    if not args.genome.exists():
        parser.error(f"Genome not found: {args.genome}")
    if args.gold_standard is not None and not args.gold_standard.exists():
        parser.error(f"Gold standard not found: {args.gold_standard}")
    if args.gemiz_template is not None and not args.gemiz_template.exists():
        parser.error(f"gemiz template not found: {args.gemiz_template}")
    if args.gemiz_reference is not None and not args.gemiz_reference.exists():
        parser.error(f"gemiz reference not found: {args.gemiz_reference}")
    if args.gemiz_feature_table is not None and not args.gemiz_feature_table.exists():
        parser.error(f"gemiz feature table not found: {args.gemiz_feature_table}")
    if args.modelseed_model is not None and not args.modelseed_model.exists():
        parser.error(f"ModelSEED model not found: {args.modelseed_model}")

    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "gemiz": run_gemiz,
        "carveme": run_carveme,
        "gapseq": run_gapseq,
    }
    results: dict[str, Any] = {
        "organism": args.organism,
        "genome": str(args.genome),
        "gold_standard": str(args.gold_standard) if args.gold_standard else None,
        "tools": {},
    }

    for tool in args.tools:
        tool_dir = args.out_dir / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{tool}]")
        try:
            if tool == "modelseed":
                row = run_external_model(
                    args,
                    tool_dir,
                    tool_name="modelseed",
                    model_path=args.modelseed_model,
                )
            else:
                row = runners[tool](args, tool_dir)
        except subprocess.TimeoutExpired as exc:
            row = {"status": "error", "error": f"Timed out: {exc}"}
        except Exception as exc:  # noqa: BLE001
            row = {"status": "error", "error": str(exc)}
        results["tools"][tool] = row

    summary_path = args.out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
