"""Dual-write verification for the wrfinput/wrfbdy writer flip.

2.5.0 flips :mod:`gpuwm.wrf_direct` onto the Rust classic writer BY
DEFAULT, so every WRF handoff file gpuwm produces is written by
``tools/rustwx/crates/netcdf-writer`` instead of netCDF4.  The
verification that justifies the flip is the same shape the history
tape's flip used (``tools/wrfout_dual_write.py``) and is used from two
places:

* ``tests/test_wrf_direct_writer.py`` runs it on small extents against
  the real frozen contract on every test run;
* this file's CLI runs the FULL-SIZE version -- the contract's own
  199 x 199 x 49 grid, or whatever ``--nx/--ny/--nz`` say -- once,
  manually, with a receipt::

      python tools/wrfinput_dual_write.py --workdir DIR \
          [--nx 199 --ny 199 --nz 49] [--receipt OUT.json]

The claim being verified, for one identical set of fields declared
through BOTH engines:

1. STRUCTURE: same variables in the same definition order, same
   dimension extents, same global attribute inventory with values equal
   in type as well as in number, same per-variable attributes;
2. VALUES: every variable's payload bitwise-equal, read raw with
   masking and scaling off;
3. VERIFICATION: each file passes its OWN engine's read-back --
   the Rust sweep for the classic container, netCDF4 for the HDF5 one.

The CONTAINER is expected to differ and is reported, not compared: the
Python engine writes the frozen contract's HDF5 ``NETCDF4``, the Rust
engine writes CDF-2 ``NETCDF3_64BIT_OFFSET`` -- which is what stock
WRF's own ``real.exe`` writes at ``io_form_input = 2``.  So is the SIZE:
the classic container carries no compression, and the receipt records
both so the cost is stated rather than discovered.

The fields are synthetic (a seeded generator), on purpose: what is being
compared is two writers handed identical arrays, so where the arrays
came from is not part of the claim, and a synthetic set makes the run
reproducible on any box with no prepared cache in reach.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import netCDF4

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpuwm import wrf_direct  # noqa: E402

RECEIPT_SCHEMA = "gpuwm-wrfinput-dual-write-receipt-v1"

#: A projection receipt shaped like the geometry one an export carries.
#: Values are arbitrary and identical for both engines; they exist so the
#: global-attribute half of the comparison has real content.
_GEOMETRY = {
    "center_lat": 39.5, "center_lon": -84.25,
    "truelat1": 30.0, "truelat2": 60.0,
    "ref_lat": 39.5, "stand_lon": -84.25,
    "map_proj": "lambert",
}

_LOGICAL_TO_WRF = {"u": "U", "v": "V", "phi": "PH", "theta": "T",
                   "mu": "MU", "qv": "QVAPOR"}
_SIDE_TO_SUFFIX = {"west": "XS", "east": "XE", "south": "YS", "north": "YE"}


class _BoundaryCache:
    """The slice of ``PreparedCache`` that ``_wrfbdy_fields`` asks for.

    Shapes are DERIVED from the frozen contract and the transpose in
    ``_lbc_to_wrf`` is inverted, rather than the staggering being
    restated here: a fixture that puts PH on ``bottom_top`` instead of
    ``bottom_top_stag`` tests its own guess.
    """

    def __init__(self, contract, dimensions):
        self._target = {
            item["name"]: tuple(int(dimensions[dim])
                                for dim in item["dimensions"][1:])
            for item in contract["variables"]
        }

    def array(self, key):
        _lbc, index, logical, side, kind = key.split("/")
        target = self._target[
            f"{_LOGICAL_TO_WRF[logical]}_B{_SIDE_TO_SUFFIX[side]}"]
        if logical == "mu":
            shape = ((1, target[1], target[0]) if side in {"west", "east"}
                     else (1, target[0], target[1]))
        elif side in {"west", "east"}:
            shape = (target[1], target[2], target[0])
        else:
            shape = (target[1], target[0], target[2])
        seed = (int(index) * 7 + len(kind)) % 97
        count = int(np.prod(shape))
        return (np.arange(count, dtype=np.float64).reshape(shape)
                + seed) * 1e-2


def _fields(nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1974)
    mass = rng.normal(290.0, 5.0, (nz, ny, nx))
    surface = rng.normal(1000.0, 50.0, (ny, nx))
    return {
        "T": mass,
        "QVAPOR": np.abs(mass) * 1e-5,
        "U": rng.normal(0.0, 8.0, (nz, ny, nx + 1)),
        "V": rng.normal(0.0, 8.0, (nz, ny + 1, nx)),
        "MU": surface,
        "HGT": np.abs(surface) * 0.3,
        "TSK": surface * 0.28 + 10.0,
    }


def _updates(nx, ny, nz, valid_time):
    return wrf_direct._global_updates(
        valid_time=valid_time, nx=nx, ny=ny, nz=nz,
        dx=3000.0, dy=3000.0, dt=15.0, geometry=_GEOMETRY)


def write_pair(workdir: Path, engine: str, *, nx: int, ny: int, nz: int,
               soil_layers: int, boundary_times: int) -> dict:
    """Write both handoff files on one engine, verify each, time it."""

    workdir.mkdir(parents=True, exist_ok=True)
    bundle = wrf_direct._load_contract()
    valid_time = datetime(2026, 8, 18, 0)
    updates = _updates(nx, ny, nz, valid_time)
    row: dict[str, object] = {"engine": engine}

    contract = bundle["wrfinput"]
    dimensions = wrf_direct._dimensions(
        contract, nx=nx, ny=ny, nz=nz, num_soil_layers=soil_layers)
    path = workdir / f"wrfinput_d01.{engine}"
    started = time.perf_counter()
    wrf_direct._write_wrfinput(
        path, contract, dimensions, updates, _fields(nx, ny, nz),
        wrf_direct._date_text(valid_time), engine)
    row["wrfinput_write_seconds"] = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    receipt = wrf_direct._validate_file(
        path, contract, nx=nx, ny=ny, nz=nz, num_soil_layers=soil_layers,
        expected_global_attributes=wrf_direct._domain_global_attributes(
            updates),
        engine=engine)
    row["wrfinput_verify_seconds"] = round(time.perf_counter() - started, 3)
    row["wrfinput_bytes"] = receipt["bytes"]
    row["wrfinput_sha256"] = receipt["sha256"]
    row["wrfinput_path"] = str(path)

    contract = bundle["wrfbdy"]
    dimensions = wrf_direct._dimensions(
        contract, nx=nx, ny=ny, nz=nz, num_soil_layers=soil_layers)
    times = [datetime(2026, 8, 18, hour) for hour in range(boundary_times)]
    bdy_path = workdir / f"wrfbdy_d01.{engine}"
    started = time.perf_counter()
    wrf_direct._write_wrfbdy(
        bdy_path, contract, dimensions, updates,
        _BoundaryCache(contract, dimensions), times, 3600, engine)
    row["wrfbdy_write_seconds"] = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    receipt = wrf_direct._validate_file(
        bdy_path, contract, nx=nx, ny=ny, nz=nz,
        num_soil_layers=soil_layers, engine=engine)
    row["wrfbdy_verify_seconds"] = round(time.perf_counter() - started, 3)
    row["wrfbdy_bytes"] = receipt["bytes"]
    row["wrfbdy_sha256"] = receipt["sha256"]
    row["wrfbdy_path"] = str(bdy_path)
    return row


def compare(left: Path, right: Path) -> list[str]:
    """Every difference a reader could see, or an empty list."""

    deltas: list[str] = []
    with netCDF4.Dataset(left) as one, netCDF4.Dataset(right) as two:
        if list(one.variables) != list(two.variables):
            deltas.append("variable inventory or definition order differs")
        if {name: len(dim) for name, dim in one.dimensions.items()} != \
                {name: len(dim) for name, dim in two.dimensions.items()}:
            deltas.append("dimension extents differ")
        if sorted(one.ncattrs()) != sorted(two.ncattrs()):
            deltas.append("global attribute inventory differs")
        for name in one.ncattrs():
            got, want = one.getncattr(name), two.getncattr(name)
            if np.asarray(got).dtype != np.asarray(want).dtype:
                deltas.append(f"global attribute {name}: dtype differs")
            elif not np.array_equal(np.asarray(got), np.asarray(want)):
                deltas.append(f"global attribute {name}: value differs")
        for name, variable in one.variables.items():
            if name not in two.variables:
                continue
            other = two.variables[name]
            if variable.dimensions != other.dimensions:
                deltas.append(f"{name}: dimensions differ")
            if variable.dtype != other.dtype:
                deltas.append(f"{name}: dtype differs")
            if variable.ncattrs() != other.ncattrs():
                deltas.append(f"{name}: attribute inventory differs")
            for attribute in variable.ncattrs():
                if attribute in other.ncattrs() and \
                        variable.getncattr(attribute) != \
                        other.getncattr(attribute):
                    deltas.append(f"{name}.{attribute}: differs")
            variable.set_auto_maskandscale(False)
            other.set_auto_maskandscale(False)
            if np.asarray(variable[:]).tobytes() != \
                    np.asarray(other[:]).tobytes():
                deltas.append(f"{name}: VALUES differ")
    return deltas


def container_of(path: Path) -> str:
    with netCDF4.Dataset(path) as dataset:
        return str(dataset.file_format)


def run(workdir: Path, *, nx: int, ny: int, nz: int, soil_layers: int,
        boundary_times: int) -> dict:
    rows = [write_pair(workdir, engine, nx=nx, ny=ny, nz=nz,
                       soil_layers=soil_layers,
                       boundary_times=boundary_times)
            for engine in ("rust", "python")]
    by_engine = {row["engine"]: row for row in rows}
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "grid": {"nx": nx, "ny": ny, "nz": nz,
                 "soil_layers": soil_layers,
                 "boundary_times": boundary_times},
        "engines": rows,
        "containers": {
            engine: {
                "wrfinput": container_of(Path(row["wrfinput_path"])),
                "wrfbdy": container_of(Path(row["wrfbdy_path"])),
            }
            for engine, row in by_engine.items()
        },
        "deltas": {
            "wrfinput": compare(Path(by_engine["rust"]["wrfinput_path"]),
                                Path(by_engine["python"]["wrfinput_path"])),
            "wrfbdy": compare(Path(by_engine["rust"]["wrfbdy_path"]),
                              Path(by_engine["python"]["wrfbdy_path"])),
        },
    }
    receipt["verdict"] = ("PASS" if not any(receipt["deltas"].values())
                          else "DIFFERENCES")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=199)
    parser.add_argument("--ny", type=int, default=199)
    parser.add_argument(
        "--nz", type=int, default=49,
        help="the frozen contract's base-profile prototypes are 49 levels "
             "long, so a different nz needs zero-shaped prototypes to "
             "resize onto (49 is the value that always works)")
    parser.add_argument("--soil-layers", type=int, default=4)
    parser.add_argument("--boundary-times", type=int, default=5)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)

    receipt = run(args.workdir, nx=args.nx, ny=args.ny, nz=args.nz,
                  soil_layers=args.soil_layers,
                  boundary_times=args.boundary_times)
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered + "\n", encoding="utf-8")
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
