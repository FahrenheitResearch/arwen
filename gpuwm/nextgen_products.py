"""NextGen product door; Rust owns numeric processing and rendering."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gpuwm.rustwx import find_renderer, renderer_env


NEXTGEN_PRODUCTS_ABI = "gpuwm.nextgen-products.v1 node-map HAND terrain-depth ensemble-depth hydrograph component-series streamflow-map"


def run_nextgen_products(request: Path, *, renderer: Path | None = None,
                         timeout: int = 600) -> Path:
    """Launch the native contract; Python never loads field values."""
    binary = renderer or find_renderer()
    if binary is None:
        raise RuntimeError("NextGen products require the rw_wrfbatch renderer")
    environment = renderer_env()
    probe = subprocess.run([str(binary), "--nextgen-products-abi"],
                           env=environment, text=True, capture_output=True, timeout=15)
    if probe.returncode or probe.stdout.strip() != NEXTGEN_PRODUCTS_ABI:
        raise RuntimeError("rw_wrfbatch needs a rebuild with this source tree's NextGen products ABI")
    result = subprocess.run([str(binary), "--nextgen-products", str(Path(request).resolve())],
                            env=environment, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "NextGen product renderer failed")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    lines = [line.split("\t", 1)[1] for line in result.stdout.splitlines()
             if line.startswith("NEXTGEN_RECEIPT\t")]
    if len(lines) != 1 or not Path(lines[0]).is_file():
        raise RuntimeError("NextGen renderer did not produce its receipt")
    return Path(lines[0])


def _run(args):
    try:
        print(run_nextgen_products(args.request))
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"NextGen products: {error}", file=sys.stderr)
        return 1


def register_cli(commands):
    parser = commands.add_parser("products", help="render NextGen gauge and flood products")
    parser.add_argument("--request", type=Path, required=True)
    parser.set_defaults(func=_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render NextGen products through Rust")
    parser.add_argument("--request", type=Path, required=True)
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
