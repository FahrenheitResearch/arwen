"""``gpuwm spectral-op``: Level-2 operator research and evidence front door."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adaptive import BandObservation, fit_hyperdiffusion
from .benchmark import run_benchmark
from .pins import registration
from .receipt import validate_receipt
from .response import hyperdiffusion_response
from .transfer import Hyperdiffusion


def _pins(_args) -> int:
    print(json.dumps(registration(), indent=2, sort_keys=True))
    return 0


def _benchmark(args) -> int:
    result = run_benchmark(
        nx=args.nx, ny=args.ny, levels=args.levels, dx_m=args.dx_m,
        dy_m=args.dy_m, backend=args.backend, repeats=args.repeats)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def _check(args) -> int:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    value = validate_receipt(receipt)
    print(value["receipt_sha256"])
    return 0


def _calibrate(args) -> int:
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    rows = raw.get("bands") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("calibration input must be a list or an object with a bands list")
    observations = [BandObservation(
        wavelength_m=float(row["wavelength_m"]),
        power_ratio=float(row["power_ratio"]),
        weight=float(row.get("weight", 1.0))) for row in rows]
    result = fit_hyperdiffusion(
        observations, dt_s=args.dt_s,
        protect_wavelength_m=args.protect_wavelength_m)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0



def _response(args) -> int:
    spec = Hyperdiffusion(
        order=args.order, reference_wavelength_m=args.reference_wavelength_m,
        e_fold_time_s=args.e_fold_time_s,
        protect_wavelength_m=args.protect_wavelength_m,
        maximum_damping_fraction=args.maximum_damping_fraction)
    if args.wavelength_m:
        wavelengths = args.wavelength_m
    else:
        import numpy as np
        wavelengths = np.geomspace(args.minimum_wavelength_m,
                                   args.maximum_wavelength_m,
                                   args.samples).tolist()
    result = {
        "schema": "gpuwm.spectral-transfer-response/v1",
        "spec": {
            "order": spec.order,
            "reference_wavelength_m": spec.reference_wavelength_m,
            "e_fold_time_s": spec.e_fold_time_s,
            "protect_wavelength_m": spec.protect_wavelength_m,
            "maximum_damping_fraction": spec.maximum_damping_fraction,
        },
        "dt_s": args.dt_s,
        "rows": hyperdiffusion_response(spec, dt_s=args.dt_s,
                                         wavelengths_m=wavelengths),
    }
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0

def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "spectral-op",
        help="Level-2 regional spectral numerical operators and evidence")
    commands = parser.add_subparsers(dest="spectral_op_command", required=True)
    pins = commands.add_parser("pins", help="print the immutable arithmetic identity")
    pins.set_defaults(func=_pins)
    bench = commands.add_parser("benchmark", help="run analytic scalar/vector/elliptic controls")
    bench.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    bench.add_argument("--nx", type=int, default=512)
    bench.add_argument("--ny", type=int, default=384)
    bench.add_argument("--levels", type=int, default=32)
    bench.add_argument("--dx-m", type=float, default=3000.0)
    bench.add_argument("--dy-m", type=float, default=3000.0)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--output", type=Path, default=None)
    bench.set_defaults(func=_benchmark)
    response = commands.add_parser("response", help="print the exact wavelength response")
    response.add_argument("--order", type=int, default=3)
    response.add_argument("--reference-wavelength-m", type=float, required=True)
    response.add_argument("--e-fold-time-s", type=float, required=True)
    response.add_argument("--protect-wavelength-m", type=float, default=None)
    response.add_argument("--maximum-damping-fraction", type=float, default=1.0)
    response.add_argument("--dt-s", type=float, required=True)
    response.add_argument("--wavelength-m", action="append", type=float, default=[])
    response.add_argument("--minimum-wavelength-m", type=float, default=3000.0)
    response.add_argument("--maximum-wavelength-m", type=float, default=3000000.0)
    response.add_argument("--samples", type=int, default=64)
    response.add_argument("--output", type=Path, default=None)
    response.set_defaults(func=_response)
    check = commands.add_parser("check", help="validate a hash-bound step receipt")
    check.add_argument("receipt", type=Path)
    check.set_defaults(func=_check)
    calibrate = commands.add_parser(
        "calibrate", help="fit a damping-only proposal from Level-1 band power ratios")
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--dt-s", type=float, required=True)
    calibrate.add_argument("--protect-wavelength-m", type=float, default=None)
    calibrate.set_defaults(func=_calibrate)


__all__ = ["register_cli"]
