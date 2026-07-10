#!/usr/bin/env python3
"""Download MMseqs2 binaries into src/gemiz/bin/mmseqs/.

Run once after cloning:
    python scripts/download_mmseqs.py

MMseqs2 release: 15-6f452 (matches versions tested with gemiz)
Downloads the correct binary for the current platform, or all binaries
when --all is passed (needed for building a distributable wheel).
"""
from __future__ import annotations
import argparse
import io
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
from pathlib import Path

RELEASE = "15-6f452"
BASE_URL = (
    f"https://github.com/soedinglab/MMseqs2/releases/download/{RELEASE}"
)
BINARIES = {
    "mmseqs-linux-avx2":    f"{BASE_URL}/mmseqs-linux-avx2.tar.gz",
    "mmseqs-linux-sse41":   f"{BASE_URL}/mmseqs-linux-sse41.tar.gz",
    "mmseqs-linux-arm64":   f"{BASE_URL}/mmseqs-linux-arm64.tar.gz",
    "mmseqs-mac-universal": f"{BASE_URL}/mmseqs-osx-universal.tar.gz",
}
BIN_DIR = Path(__file__).parent.parent / "src" / "gemiz" / "bin" / "mmseqs"


def _platform_binary() -> str:
    system  = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        return "mmseqs-mac-universal"
    if system == "linux":
        if machine in ("arm64", "aarch64"):
            return "mmseqs-linux-arm64"
        return "mmseqs-linux-avx2"
    print("Windows: run gemiz inside WSL2.", file=sys.stderr)
    sys.exit(1)


def _download(name: str, url: str) -> None:
    dest = BIN_DIR / name
    if dest.exists():
        if dest.is_file():
            print(f"  {name}: already present, skipping")
            return
        print(f"  {name}: removing invalid directory from previous download")
        shutil.rmtree(dest)
    print(f"  {name}: downloading from {url}")
    with urllib.request.urlopen(url) as resp:
        data = io.BytesIO(resp.read())
    with tarfile.open(fileobj=data) as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and Path(m.name).name == "mmseqs"
        ]
        if not members:
            raise RuntimeError(f"No executable named 'mmseqs' found in {url}")
        src = tf.extractfile(members[0])
        if src is None:
            raise RuntimeError(f"Could not extract mmseqs from {url}")
        with dest.open("wb") as out:
            shutil.copyfileobj(src, out)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  {name}: OK ({dest.stat().st_size // 1024 // 1024} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="Download all platform binaries (for wheel builds).")
    args = parser.parse_args()

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    targets = list(BINARIES.items()) if args.all else [(_platform_binary(),
                   BINARIES[_platform_binary()])]
    for name, url in targets:
        _download(name, url)


if __name__ == "__main__":
    main()
