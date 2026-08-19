#!/usr/bin/env python3
"""Render a wrfout and its ``nc_rewrite`` rewrite, compare the PNGs byte
for byte.

ACCEPTANCE (3) for the Rust NetCDF writer.  The other two acceptances ask
whether the file is correct as data; this one asks whether the PRODUCT is
unchanged, which is the only question a user of the tape actually asks.
The instrument is ``rw_wrfbatch``, the production renderer, driven exactly
as ``gpuwm render --engine rust`` drives it -- the real binary, not a
mock, and the render law's renderer rather than a matplotlib stand-in.

An identical PNG is a strong claim: the renderer reads the tape through
its own importer, derives severe/thermodynamic fields from it, quantises
to a colortable and encodes.  Every one of those steps is sensitive to a
single changed float, so byte-identical output over a whole product set
means the values that reached the renderer were the same values.

usage: nc_rewrite_render_parity.py ORIGINAL REWRITE --rw-wrfbatch EXE
                                   --work DIR [--width N] [--height N]
                                   [--products all|SLUGS]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render(exe: str, source: pathlib.Path, work: pathlib.Path, tag: str,
           width: int, height: int, products: str) -> pathlib.Path:
    store = work / f"store_{tag}"
    out = work / f"out_{tag}"
    for folder in (store, out):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    command = [
        exe,
        "--store-root", str(store),
        "--out-dir", str(out),
        "--products", products,
        "--frames", "all",
        "--width", str(width),
        "--height", str(height),
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"rw_wrfbatch failed on {source} (exit {result.returncode})")
    finished = [line for line in result.stdout.splitlines() if line.startswith("FINISHED")]
    print(f"{tag}: {finished[-1] if finished else 'no FINISHED line'}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original")
    parser.add_argument("rewrite")
    parser.add_argument("--rw-wrfbatch", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--products", default="all")
    args = parser.parse_args()

    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    out_a = render(args.rw_wrfbatch, pathlib.Path(args.original), work, "original",
                   args.width, args.height, args.products)
    out_b = render(args.rw_wrfbatch, pathlib.Path(args.rewrite), work, "rewrite",
                   args.width, args.height, args.products)

    names_a = sorted(p.name for p in out_a.glob("*.png"))
    names_b = sorted(p.name for p in out_b.glob("*.png"))
    if not names_a:
        raise SystemExit("the original rendered no PNGs; nothing to compare")

    problems: list[str] = []
    if names_a != names_b:
        missing = sorted(set(names_a) - set(names_b))
        extra = sorted(set(names_b) - set(names_a))
        problems.append(f"product set differs: missing={missing[:8]} extra={extra[:8]}")

    identical = 0
    for name in names_a:
        if name not in set(names_b):
            continue
        left = sha256(out_a / name)
        right = sha256(out_b / name)
        if left == right:
            identical += 1
        else:
            problems.append(f"{name}: {left[:16]} vs {right[:16]}")

    print(f"COMPARED pngs={len(names_a)} identical={identical}")
    if problems:
        print(f"RENDER PARITY FAIL differences={len(problems)}")
        for problem in problems[:40]:
            print(f"  - {problem}")
        return 1
    print(f"RENDER PARITY PASS all {identical} PNG(s) are byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
