"""Recapture a per-card GPU pin for THIS card, with its readings.

The pins ``tests/_card_pins.py`` registers besides the phase-2 step capture
(which ``tools/recapture_phase2_pin.py`` owns):

    coriolis_map_sina0_pin         one launch_coriolis_curvature call on the
                                   seed-1974 inputs of tests/test_coriolis_map.py
    advection_periodic_regression  the four flux-divergence launchers on the
                                   original file's inputs (tests/test_openbc.py)
    diff6_base_4d2ce99             tests/test_diff6_boundary_face.py's own
                                   generate_base_capture, dual run, verified

Run on the card the pin is graded on:

    python tools/recapture_card_pins.py --pin NAME                 # readings only
    python tools/recapture_card_pins.py --pin NAME --write --readings out.json

Without ``--write`` nothing is touched.  With it, this card's file (the
``PINS`` row for this device; ``--new-card FILE`` when there is none) is
replaced by the current tree's answer on this card, and the readings JSON
carries, per entry, the maximum absolute difference, the RMS of the entry
it replaced, their ratio and the count of differing words, against this
card's committed file when it has one, else against the reference (the
reference card's file, or the original capture), plus the card, its
compute capability, the driver and the commit.  That is what the ledger
beside the test must then quote.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from tools.recapture_phase2_pin import readings  # noqa: E402


def _load_test_module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"card_pin_{name}", ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_coriolis_map_sina0_pin(_target: Path) -> dict[str, np.ndarray]:
    import cupy as cp
    from gpuwm.core.dycore import launch_coriolis_curvature
    from gpuwm.core.grid import make_vertical_coord
    test = _load_test_module("test_coriolis_map")
    d = test._random_rotational_inputs(seed=1974)
    nz, ny, nxp1 = d["u"].shape
    nx = nxp1 - 1
    coord = make_vertical_coord(nz, stretch=2.0)
    dev = {k: cp.asarray(v, cp.float32) for k, v in d.items()}
    ru_t = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv_t = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw_t = cp.zeros((nz + 1, ny, nx), cp.float32)
    launch_coriolis_curvature(
        dev["ru"], dev["rv"], dev["u"], dev["v"], dev["w"], dev["mut"],
        dev["msft"], dev["msfu"], dev["msfv"], dev["f"], dev["e"],
        cp.asarray(coord.c1f, cp.float32), cp.asarray(coord.c2f, cp.float32),
        cp.asarray(coord.fnm, cp.float32), cp.asarray(coord.fnp, cp.float32),
        200.0, 250.0, ru_t, rv_t, rw_t)
    return {"ru_t": cp.asnumpy(ru_t), "rv_t": cp.asnumpy(rv_t),
            "rw_t": cp.asnumpy(rw_t)}


def capture_advection_periodic_regression(_target: Path) -> dict[str, np.ndarray]:
    import cupy as cp
    import _card_pins
    from gpuwm.core.advection import (launch_flux_div_scalar, launch_flux_div_u,
                                      launch_flux_div_v, launch_flux_div_w)
    from gpuwm.core.grid import make_vertical_coord
    with np.load(_card_pins.original_path("advection_periodic_regression")) as ref:
        inputs = {k: ref[k].copy() for k in ("q", "u", "v", "w", "ru", "rv", "rw")}
        shapes = {k: ref[k].shape for k in ("tend_scalar", "tend_u", "tend_v", "tend_w")}
    nz = inputs["q"].shape[0]
    coord = make_vertical_coord(nz)
    dev = {k: cp.asarray(v) for k, v in inputs.items()}
    out = dict(inputs)
    for field, launcher, key in ((dev["q"], launch_flux_div_scalar, "tend_scalar"),
                                 (dev["u"], launch_flux_div_u, "tend_u"),
                                 (dev["v"], launch_flux_div_v, "tend_v"),
                                 (dev["w"], launch_flux_div_w, "tend_w")):
        tend = cp.zeros(shapes[key], cp.float32)
        launcher(field, dev["ru"], dev["rv"], dev["rw"], tend, coord, 100.0, 100.0)
        out[key] = cp.asnumpy(tend)
    return out


def capture_diff6_base_4d2ce99(target: Path) -> dict[str, np.ndarray]:
    test = _load_test_module("test_diff6_boundary_face")
    dual = str(target) + ".dualrun.npz"
    first = test.generate_base_capture(str(target))
    test.generate_base_capture(dual)
    with np.load(dual) as second:
        for key in first:
            if key == "provenance":
                continue
            if not np.array_equal(first[key], second[key]):
                raise SystemExit(f"diff6 dual run differs on {key}; the card is not deterministic here")
    Path(dual).unlink()
    return {k: np.asarray(v) for k, v in first.items()}


CAPTURES = {
    "coriolis_map_sina0_pin": capture_coriolis_map_sina0_pin,
    "advection_periodic_regression": capture_advection_periodic_regression,
    "diff6_base_4d2ce99": capture_diff6_base_4d2ce99,
}


def _card() -> dict:
    import cupy as cp
    import _card_pins

    def _query(command):
        try:
            done = subprocess.run(command, capture_output=True, text=True)
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    return {"card": _card_pins.device_name(),
            "compute_capability": _card_pins.device_compute_capability(),
            "driver": _query(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"]),
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            **_nvrtc(),
            "commit": _query(["git", "-C", str(ROOT), "rev-parse", "HEAD"])}


def _nvrtc() -> dict:
    """The NVRTC build that compiles this process's kernels, the key the
    per-compiler pin rows use (gpuwm.certify.compile_platform): a pin is a
    property of the compiled image, so the capture records the compiler
    beside the card."""
    try:
        from gpuwm.certify.compile_platform import compile_platform_fingerprint
        fingerprint = compile_platform_fingerprint()
    except Exception as error:  # a tree without the fingerprint module
        return {"nvrtc": f"unresolved ({type(error).__name__})"}
    return {"nvrtc": fingerprint["nvrtc_build"],
            "nvrtc_build_id": fingerprint["nvrtc_build_id"],
            "nvrtc_library_sha256": fingerprint["nvrtc_library_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin", required=True, choices=sorted(CAPTURES))
    parser.add_argument("--write", action="store_true",
                        help="replace this card's capture with the current tree's answer")
    parser.add_argument("--readings", type=Path, help="write the per-entry readings as JSON")
    parser.add_argument("--against", metavar="CARD",
                        help="take the readings against CARD's committed file")
    parser.add_argument("--new-card", metavar="FILE", type=str,
                        help="this card has no PINS row: with --write, create tests/data/FILE")
    args = parser.parse_args()
    import _card_pins
    card = _card()
    name, pin = card["card"], args.pin
    own = _card_pins.path(pin, name)
    if args.against is not None:
        old_path = _card_pins.path(pin, args.against)
        if old_path is None:
            parser.error(f"--against {args.against!r} has no committed file for {pin}")
    else:
        old_path = own or _card_pins.reference_path(pin)
    print(f"pin {pin}: card {name!r} (compute capability {card['compute_capability']}, "
          f"driver {card['driver']}), commit {card['commit']}")
    print(f"readings against {old_path.relative_to(ROOT)}"
          + ("" if old_path == own else " (this card has no committed file)"))
    target = own or _card_pins.declared_path(pin, name)
    if args.write and target is None:
        if not args.new_card:
            print(f"no committed capture and no PINS row for {name!r}; pass --new-card FILE",
                  file=sys.stderr)
            return 2
        target = _card_pins.PIN_DIR / args.new_card
    # The capture lands in a scratch file first (diff6's generator writes its
    # own path), so a readings-only run never touches the committed file.
    scratch = _card_pins.PIN_DIR / f".{pin}.capture.npz"
    with np.load(old_path) as data:
        old = {k: data[k] for k in data.files}
    new = CAPTURES[pin](scratch)
    if scratch.exists() and not args.write:
        scratch.unlink()
    comparable_old = {k: v for k, v in old.items() if v.dtype.kind == "f"}
    comparable_new = {k: v for k, v in new.items() if v.dtype.kind == "f"}
    report = readings(comparable_old, comparable_new)
    moved = [k for k, v in report.items() if v.get("status") != "held"]
    for key in sorted(report):
        v = report[key]
        if v.get("status") in ("moved", "held"):
            print(f"{key:22s} {v['status']:5s} max_abs={v['max_abs_difference']:.3e} "
                  f"rms_old={v['rms_of_old_entry']:.3e} ratio="
                  f"{(v['max_abs_over_rms'] or 0.0):.3e} "
                  f"differing={v['differing_words']}/{v['words']}")
        else:
            print(f"{key:22s} {v}")
    print(f"{len(moved)} of {len(report)} entries moved")
    if args.readings is not None:
        args.readings.parent.mkdir(parents=True, exist_ok=True)
        args.readings.write_text(json.dumps(
            {"pin": pin, "file": str((target or old_path).relative_to(ROOT)),
             "against": str(old_path.relative_to(ROOT)), **card,
             "entries": report, "moved": len(moved), "total": len(report)},
            indent=2) + "\n", encoding="utf-8")
    if args.write:
        if scratch.exists():                     # diff6's generator wrote the scratch
            scratch.replace(target)
        else:
            np.savez(target, **new)
        print(f"wrote {target}")
        if name not in _card_pins.PINS[pin]:
            print(f"add to tests/_card_pins.py PINS[{pin!r}]: {name!r}: {target.name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
