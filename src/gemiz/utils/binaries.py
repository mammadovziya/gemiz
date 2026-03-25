"""Bundled binary resolver — MMseqs2.

MMseqs2 binaries are shipped inside the package so that
``pip install gemiz`` works out of the box on Linux and macOS.

Bundled files
-------------
  bin/mmseqs/mmseqs-linux-avx2      Linux x86_64, AVX2 (modern CPUs, fastest)
  bin/mmseqs/mmseqs-linux-sse41     Linux x86_64, SSE4.1 (older CPU fallback)
  bin/mmseqs/mmseqs-linux-arm64     Linux ARM64 (Raspberry Pi, AWS Graviton, etc.)
  bin/mmseqs/mmseqs-mac-universal   macOS universal (Intel + Apple Silicon)

Windows note
------------
The official MMseqs2 Windows build requires Cygwin (cygz.dll) and is not
self-contained. Windows users should run gemiz inside WSL2:

    wsl --install           # one-time setup
    pip install gemiz       # inside WSL terminal
    gemiz reconstruct ...

WSL detection: if Python is running *inside* WSL, platform.system() already
returns "Linux" so the Linux binary is picked automatically. No special
handling is needed.

MMseqs2 release: 18-8cc5c
"""

from __future__ import annotations

import platform
import stat
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_binary_dir() -> Path:
    """Return the absolute path to ``src/gemiz/bin/``."""
    return Path(__file__).parent.parent / "bin"


# ---------------------------------------------------------------------------
# CPU feature detection
# ---------------------------------------------------------------------------

def check_cpu_features() -> dict[str, bool]:
    """Return a dict of supported CPU instruction sets.

    Returns
    -------
    dict
        ``{"avx2": True/False, "sse41": True/False}``

    Reads ``/proc/cpuinfo`` on Linux; ``sysctl`` on macOS.
    Always returns ``True`` for both on Windows (CPUID not read, but
    the Windows binary handles detection internally).
    """
    system = platform.system().lower()

    if system == "linux":
        try:
            flags = Path("/proc/cpuinfo").read_text()
            # Each flag is a space-separated word in the "flags" line
            flag_set = {w for line in flags.splitlines()
                        if line.startswith("flags")
                        for w in line.split()}
            return {
                "avx2":  "avx2"  in flag_set,
                "sse41": "sse4_1" in flag_set or "sse4.1" in flag_set,
            }
        except OSError:
            return {"avx2": False, "sse41": True}

    elif system == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features",
                 "machdep.cpu.leaf7_features"],
                text=True, stderr=subprocess.DEVNULL,
            )
            upper = out.upper()
            return {
                "avx2":  "AVX2"  in upper,
                "sse41": "SSE4_1" in upper or "SSE4.1" in upper,
            }
        except Exception:
            return {"avx2": True, "sse41": True}

    else:  # Windows — binary handles its own dispatch
        return {"avx2": True, "sse41": True}


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def get_mmseqs_path() -> Path:
    """Return the correct MMseqs2 binary for the current platform.

    Selection logic
    ---------------
    Linux x86_64 : AVX2 supported → avx2 binary (fastest)
                   AVX2 absent    → sse41 binary (compatible)
    Linux arm64  : arm64 binary
    macOS        : universal binary (Intel + Apple Silicon)
    Windows      : native win64 binary (no WSL required)

    Raises
    ------
    FileNotFoundError
        If the expected binary is missing from the package (packaging bug).
    """
    system  = platform.system().lower()
    machine = platform.machine().lower()
    bin_dir = get_binary_dir() / "mmseqs"

    if system == "linux":
        if machine in ("arm64", "aarch64"):
            binary = bin_dir / "mmseqs-linux-arm64"
        else:
            # x86_64: prefer AVX2, fall back to SSE4.1
            features = check_cpu_features()
            if features["avx2"]:
                binary = bin_dir / "mmseqs-linux-avx2"
            else:
                binary = bin_dir / "mmseqs-linux-sse41"

    elif system == "darwin":
        binary = bin_dir / "mmseqs-mac-universal"

    elif system == "windows":
        raise OSError(
            "\n[gemiz] MMseqs2 does not have a self-contained Windows binary.\n"
            "\n"
            "Please run gemiz inside WSL2 (Windows Subsystem for Linux):\n"
            "\n"
            "    wsl --install           # one-time setup (PowerShell, admin)\n"
            "    # open a WSL terminal, then:\n"
            "    pip install gemiz\n"
            "    gemiz reconstruct genome.fna\n"
            "\n"
            "WSL2 is free, fast, and gives you a full Linux environment.\n"
            "If you are already inside WSL, platform.system() returns 'Linux'\n"
            "and gemiz works automatically.\n"
        )

    else:
        raise FileNotFoundError(
            f"[gemiz] Unsupported OS: {platform.system()!r}\n"
            "Supported: Linux, macOS, Windows"
        )

    if not binary.exists():
        raise FileNotFoundError(
            f"\n[gemiz] Bundled MMseqs2 binary not found: {binary}\n"
            "This is a packaging bug — please open an issue:\n"
            "https://github.com/gemiz/gemiz/issues"
        )

    _ensure_executable(binary)
    return binary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_mmseqs() -> dict:
    """Run ``mmseqs version`` and return a status dict.

    Returns
    -------
    dict
        ``{"ok": True, "version": "18-8cc5c", "path": "..."}``
        or ``{"ok": False, "error": "...", "path": None}``
    """
    try:
        binary = get_mmseqs_path()
        proc = subprocess.run(
            [str(binary), "version"],
            capture_output=True, text=True, timeout=10,
        )
        version = (proc.stdout + proc.stderr).strip().splitlines()[0]
        return {"ok": True, "version": version, "path": str(binary)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "path": None}


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _ensure_executable(path: Path) -> None:
    """Set the executable bit on Unix (no-op on Windows)."""
    if sys.platform != "win32":
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
