"""Generate observation-remap parity goldens by RUNNING the real Python.

Every golden here is the output of ``gpuwm.verify.obs.regrid`` -- the
scipy ``cKDTree`` plan build and the numpy ``add.at`` apply -- on REAL
staged observation and model bytes:

* the 2024-05-21 case's real model grid,
  ``cache/obsbattery/battery/model_grid_case20240521.npz`` (400x480
  float64 lat/lon off the real Lambert domain builder), and
* the real MRMS MergedReflectivityQCComposite pack for the same case,
  ``cache/obsbattery/packs/mrms_20240521T210037.obspack`` with its
  geometry pack ``mrms_grid.obspack`` -- real reflectivity values and
  the real coverage mask, decoded by the shipped reader.

The full MRMS CONUS mosaic is 3500x7000 float64, which is 196 MB of
latitude alone, so the goldens carry a real contiguous WINDOW of it
around the case domain, decimated to a spacing that keeps the file set
near the lane-2 precedent's size.  Decimating a real grid leaves real
coordinates and real values in a real geometry; it is not synthesis.
Two synthetic sets follow it, both there to pin behaviour real data
cannot exhibit on demand: the distance-bound edge and the tie rule.

Three probes run BEFORE any golden is written, and the script refuses if
any of them fails, because each one is an assumption the Rust port's
bit-identical claim rests on:

``deg2rad``      numpy's ufunc is one multiply by the double nearest
                 pi/180, which is what Rust's ``PI / 180.0`` folds to.
``trig``         numpy's float64 ``sin``/``cos`` are bit-identical to
                 the platform libm CPython calls, which is the same
                 libm Rust's ``f64::sin``/``cos`` lower to.
``bound``        scipy accepts a neighbour on the STRICT squared
                 predicate ``d2 < bound*bound``.

The tie probe does NOT gate: it MEASURES, and writes its measurement
into the manifest, because scipy's tie-breaking is the one behaviour
this port deliberately does not reproduce (see ``src/kdtree.rs``).

Regenerate with::

    python tools/rustwx/crates/obs-regrid/golden/gen_regrid_goldens.py

Array container ("GWARR1"), identical to the lane-2 static goldens:
magic 8 bytes ``b"GWARR1\\x00\\x00"``, dtype u8 (0=f64, 1=f32, 2=i64,
3=u8/bool), ndim u8, dims as u64 LE, payload LE.  Floats in the JSON
manifest are hex bit patterns, so a reader cannot lose a bit to a
decimal round-trip.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

from gpuwm.obs import obspack  # noqa: E402
from gpuwm.verify.obs import regrid  # noqa: E402

OUT = HERE / "cases"

#: Real staged bytes, in the repository's gitignored object cache.
#: Absent on a box that has not fetched the case, which the script says
#: out loud rather than substituting synthesis.
#:
#: Repo-relative, not one box's absolute path, for two reasons that both
#: bite: the path was wrong on every machine but the author's, and it
#: also reached the COMMITTED manifest below, where
#: ``work/build_release_snapshot.py``'s machine-path scan fails the
#: release build on it.  ``GPUWM_OBSBATTERY_CACHE`` points the generator
#: at a cache outside this checkout -- a worktree borrowing the main
#: clone's staged case, say -- without either spelling being written down.
CACHE = Path(os.environ.get("GPUWM_OBSBATTERY_CACHE")
             or REPO / "cache" / "obsbattery")
MODEL_GRID = CACHE / "battery" / "model_grid_case20240521.npz"
OBS_GRID_PACK = CACHE / "packs" / "mrms_grid.obspack"
OBS_FIELD_PACK = CACHE / "packs" / "mrms_20240521T210037.obspack"


def recorded_path(path: Path) -> str:
    """How an input path is spelled in the committed manifest.

    Repo-relative with forward slashes when the file is inside the
    checkout; otherwise folded to the environment variable that pointed
    at it.  Never one machine's absolute path: the release snapshot
    builder refuses to build a tree carrying one, and the manifest is a
    tracked file that ships with the crate.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        pass
    try:
        tail = resolved.relative_to(Path(CACHE).resolve()).as_posix()
    except ValueError:
        return f"<outside the checkout>/{resolved.name}"
    return f"$GPUWM_OBSBATTERY_CACHE/{tail}"


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------

_CODES = {"float64": 0, "float32": 1, "int64": 2, "uint8": 3, "bool": 3}


def write_arr(path: Path, array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    code = _CODES[str(array.dtype)]
    header = b"GWARR1\x00\x00" + bytes((code, array.ndim))
    header += b"".join(struct.pack("<Q", int(d)) for d in array.shape)
    payload = header + array.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def hexf(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


# --------------------------------------------------------------------------
# probes: the assumptions the bit-identical claim rests on
# --------------------------------------------------------------------------

def probe_deg2rad() -> dict[str, object]:
    sample = np.linspace(-179.9, 179.9, 100_003)
    lhs = np.deg2rad(sample)
    rhs = sample * (math.pi / 180.0)
    same = bool(np.array_equal(lhs.view(np.int64), rhs.view(np.int64)))
    if not same:
        raise SystemExit(
            "numpy's deg2rad is not one multiply by the double nearest "
            "pi/180 on this build, so the Rust port's unit vectors cannot "
            "be bit-identical; the port needs a different transcription "
            "before these goldens mean anything")
    return {"bitwise_identical": True, "samples": int(sample.size),
            "constant": hexf(math.pi / 180.0)}


def probe_trig() -> dict[str, object]:
    sample = np.deg2rad(np.linspace(-90.0, 90.0, 40_001))
    report: dict[str, object] = {"samples": int(sample.size)}
    for name, ufunc, scalar in (("sin", np.sin, math.sin),
                                ("cos", np.cos, math.cos)):
        got = ufunc(sample)
        reference = np.array([scalar(value) for value in sample])
        same = bool(np.array_equal(got.view(np.int64),
                                   reference.view(np.int64)))
        if not same:
            raise SystemExit(
                f"numpy's float64 {name} is not the platform libm on this "
                f"build, so the Rust port cannot be bit-identical through "
                f"the unit-vector step; measure the gap and declare a "
                f"tolerance before writing goldens")
        report[name] = {"bitwise_identical_to_libm": True}
    return report


def probe_bound() -> dict[str, object]:
    """scipy's acceptance predicate, measured rather than assumed."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(11)
    squared_agree = 0
    linear_agree = 0
    inclusive_agree = 0
    trials = 0
    for _ in range(2000):
        a = rng.normal(size=(1, 3))
        b = rng.normal(size=(1, 3))
        d = (a - b)[0]
        d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        distance = math.sqrt(d2)
        for bound in (np.nextafter(distance, 0.0), distance,
                      np.nextafter(distance, np.inf)):
            got = bool(np.isfinite(cKDTree(a).query(
                b, k=1, distance_upper_bound=bound)[0][0]))
            trials += 1
            squared_agree += int(got == (d2 < bound * bound))
            linear_agree += int(got == (distance < bound))
            inclusive_agree += int(got == (d2 <= bound * bound))
    if squared_agree != trials:
        raise SystemExit(
            "scipy's distance_upper_bound predicate is not `d2 < "
            "bound*bound` on this build; the Rust port reproduces that "
            "exact predicate and would diverge at the bound")
    return {
        "trials": trials,
        "strict_squared_agreements": squared_agree,
        "linear_agreements": linear_agree,
        "inclusive_squared_agreements": inclusive_agree,
        "note": "the linear and inclusive counts are the reason the port "
                "squares the bound and compares strictly; both spellings "
                "disagree with scipy near the boundary",
    }


def probe_ties() -> dict[str, object]:
    """The one behaviour the port deliberately does NOT reproduce."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(3)
    lowest = 0
    other = 0
    for _ in range(400):
        count = int(rng.integers(3, 40))
        points = rng.normal(size=(count, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        slots = rng.choice(count, size=int(rng.integers(2, min(6, count) + 1)),
                           replace=False)
        points[slots] = points[slots[0]]
        _, index = cKDTree(points).query(points[slots[0]][None, :], k=1,
                                         distance_upper_bound=10.0)
        if int(index[0]) == int(slots.min()):
            lowest += 1
        else:
            other += 1
    return {
        "trials": lowest + other,
        "scipy_returned_lowest_index": lowest,
        "scipy_returned_some_other_tied_index": other,
        "verdict": "unspecified -- scipy returns whichever tied point the "
                   "traversal reached first, so there is no rule to be "
                   "bit-exact to; obs-regrid DEFINES lowest-flat-index-wins",
    }


# --------------------------------------------------------------------------
# real inputs
# --------------------------------------------------------------------------

def load_real_inputs():
    missing = [p for p in (MODEL_GRID, OBS_GRID_PACK, OBS_FIELD_PACK)
               if not p.is_file()]
    if missing:
        raise SystemExit(
            "the real case bytes are not staged on this box:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nFetch them with `python -m tools.obs_battery_fetch` (the "
              "2024-05-21 case) and re-run; this script will NOT substitute "
              "synthetic grids for the real ones.")
    grid = np.load(MODEL_GRID)
    model_lat = np.asarray(grid["latitude"], dtype=np.float64)
    model_lon = np.asarray(grid["longitude"], dtype=np.float64)

    geometry = obspack.read_pack(OBS_GRID_PACK)
    obs_lat = np.asarray(geometry.arrays["latitude"], dtype=np.float64)
    obs_lon = np.asarray(geometry.arrays["longitude"], dtype=np.float64)
    field = obspack.read_pack(OBS_FIELD_PACK)
    obs_values = np.asarray(field.arrays["values"], dtype=np.float64)
    obs_valid = np.asarray(field.arrays["valid"]).astype(bool)
    return (model_lat, model_lon, obs_lat, obs_lon, obs_values, obs_valid,
            field.meta, geometry.meta)


def window(model_lat, model_lon, obs_lat, obs_lon, obs_values, obs_valid,
           *, model_step: int, obs_step: int, margin_deg: float):
    """A real contiguous window of both grids around the case domain."""
    model_lat = model_lat[::model_step, ::model_step]
    model_lon = model_lon[::model_step, ::model_step]
    lat_lo = float(model_lat.min()) - margin_deg
    lat_hi = float(model_lat.max()) + margin_deg
    lon_lo = float(model_lon.min()) - margin_deg
    lon_hi = float(model_lon.max()) + margin_deg

    rows = np.flatnonzero((obs_lat[:, 0] >= lat_lo) & (obs_lat[:, 0] <= lat_hi))
    cols = np.flatnonzero((obs_lon[0, :] >= lon_lo) & (obs_lon[0, :] <= lon_hi))
    if rows.size == 0 or cols.size == 0:
        raise SystemExit(
            "the case domain does not intersect the observation mosaic; the "
            "staged pair is not the pair this golden set describes")
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1, obs_step)
    col_slice = slice(int(cols[0]), int(cols[-1]) + 1, obs_step)
    return (model_lat, model_lon,
            obs_lat[row_slice, col_slice], obs_lon[row_slice, col_slice],
            obs_values[row_slice, col_slice], obs_valid[row_slice, col_slice])


# --------------------------------------------------------------------------
# one golden case
# --------------------------------------------------------------------------

def emit_case(name: str, *, source_latitude, source_longitude,
              destination_latitude, destination_longitude, values, valid,
              method: str, max_distance_m: float,
              provenance: dict[str, object]) -> dict[str, object]:
    plan = regrid.build_plan(
        source_latitude=source_latitude, source_longitude=source_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude, method=method,
        max_distance_m=max_distance_m)
    out_values, out_valid = regrid.apply_plan(plan, values, valid)

    directory = OUT / name
    files = {
        "source_latitude": write_arr(directory / "source_latitude.bin",
                                     np.asarray(source_latitude, np.float64)),
        "source_longitude": write_arr(directory / "source_longitude.bin",
                                      np.asarray(source_longitude, np.float64)),
        "destination_latitude": write_arr(
            directory / "destination_latitude.bin",
            np.asarray(destination_latitude, np.float64)),
        "destination_longitude": write_arr(
            directory / "destination_longitude.bin",
            np.asarray(destination_longitude, np.float64)),
        "values": write_arr(directory / "values.bin",
                            np.asarray(values, np.float64)),
        "valid": write_arr(directory / "valid.bin",
                           np.asarray(valid, bool)),
        "source_index": write_arr(directory / "source_index.bin",
                                  np.asarray(plan.source_index, np.int64)),
        "reachable": write_arr(directory / "reachable.bin",
                               np.asarray(plan.reachable, bool)),
        "out_values": write_arr(directory / "out_values.bin", out_values),
        "out_valid": write_arr(directory / "out_valid.bin", out_valid),
    }
    return {
        "name": name,
        "method": method,
        "max_distance_m": hexf(max_distance_m),
        "max_used_distance_m": hexf(plan.max_used_distance_m),
        "source_shape": list(plan.source_shape),
        "destination_shape": list(plan.destination_shape),
        "record": {
            key: (hexf(value) if isinstance(value, float) else value)
            for key, value in plan.record().items()
        },
        "reachable_destination_cells": int(np.count_nonzero(plan.reachable)),
        "valid_destination_cells": int(np.count_nonzero(out_valid)),
        "files": files,
        "provenance": provenance,
    }


def main() -> None:
    probes = {
        "deg2rad": probe_deg2rad(),
        "trig": probe_trig(),
        "bound": probe_bound(),
        "ties": probe_ties(),
    }
    print("probes passed:", json.dumps(probes["bound"])[:120], "...")

    (model_lat, model_lon, obs_lat, obs_lon, obs_values, obs_valid,
     field_meta, geometry_meta) = load_real_inputs()
    full_shapes = {
        "model_grid": list(model_lat.shape),
        "observation_grid": list(obs_lat.shape),
    }
    (model_lat, model_lon, obs_lat, obs_lon,
     obs_values, obs_valid) = window(
        model_lat, model_lon, obs_lat, obs_lon, obs_values, obs_valid,
        model_step=4, obs_step=12, margin_deg=0.25)
    print("windowed model", model_lat.shape, "observation", obs_lat.shape)

    real_provenance = {
        "model_grid": recorded_path(MODEL_GRID),
        "observation_geometry_pack": recorded_path(OBS_GRID_PACK),
        "observation_field_pack": recorded_path(OBS_FIELD_PACK),
        "observation_sha256": obspack.sha256_of(OBS_FIELD_PACK),
        "observation_quantity": field_meta.get("quantity"),
        "observation_valid_time": field_meta.get("valid_time"),
        "full_shapes": full_shapes,
        "window": {"model_step": 4, "observation_step": 12,
                   "margin_deg": 0.25},
    }

    cases = []
    # 1. The battery's reflectivity leg: real observation onto the real
    #    model grid, `nearest`, at the registered bound.
    cases.append(emit_case(
        "real_nearest_obs_to_model",
        source_latitude=obs_lat, source_longitude=obs_lon,
        destination_latitude=model_lat, destination_longitude=model_lon,
        values=obs_values, valid=obs_valid,
        method=regrid.NEAREST, max_distance_m=12_000.0,
        provenance=real_provenance))
    # 2. The qualification leg: the same real pair under the OTHER
    #    registered operator, which is the reverse assignment.
    cases.append(emit_case(
        "real_cell_average_obs_to_model",
        source_latitude=obs_lat, source_longitude=obs_lon,
        destination_latitude=model_lat, destination_longitude=model_lon,
        values=obs_values, valid=obs_valid,
        method=regrid.CELL_AVERAGE, max_distance_m=12_000.0,
        provenance=real_provenance))
    # 3. The precipitation leg's direction: model onto the observation
    #    grid, which is the pair battery.py builds for Stage-IV.
    model_field = np.asarray(
        np.hypot(model_lat - float(model_lat.mean()),
                 model_lon - float(model_lon.mean())), dtype=np.float64)
    cases.append(emit_case(
        "real_nearest_model_to_obs",
        source_latitude=model_lat, source_longitude=model_lon,
        destination_latitude=obs_lat, destination_longitude=obs_lon,
        values=model_field, valid=np.ones(model_lat.shape, dtype=bool),
        method=regrid.NEAREST, max_distance_m=12_000.0,
        provenance=real_provenance))
    # 4. The bound doing its job on real geometry: a bound tight enough
    #    that a large part of the destination is unreachable rather than
    #    borrowed from far away.
    cases.append(emit_case(
        "real_nearest_tight_bound",
        source_latitude=obs_lat, source_longitude=obs_lon,
        destination_latitude=model_lat, destination_longitude=model_lon,
        values=obs_values, valid=obs_valid,
        method=regrid.NEAREST, max_distance_m=1_500.0,
        provenance=real_provenance))

    # 5. Synthetic: the distance bound's edge, where the strict squared
    #    predicate is the whole answer.  Real grids do not land on a
    #    bound to the ULP on demand.
    #
    # The two metre bounds are found by WALKING the float64 grid until
    # the answer actually flips, not by assuming one `nextafter` does
    # it: `chord_from_arc` and `arc_from_chord` are not bitwise inverses
    # (a sine and an arcsine round independently), so a single ULP of
    # metres frequently lands on the same chord and the pair would have
    # pinned nothing.  The flip is asserted before the goldens are
    # written, because a bound-edge case that does not flip is a test
    # that passes on any implementation.
    edge_source_lat = np.array([[0.0, 0.0]])
    edge_source_lon = np.array([[0.0, 1.0]])
    edge_destination_lat = np.array([[0.0]])
    edge_destination_lon = np.array([[0.25]])

    def edge_reaches(bound_m: float) -> bool:
        plan = regrid.build_plan(
            source_latitude=edge_source_lat,
            source_longitude=edge_source_lon,
            destination_latitude=edge_destination_lat,
            destination_longitude=edge_destination_lon,
            method=regrid.NEAREST, max_distance_m=float(bound_m))
        return bool(plan.reachable[0, 0])

    exact = regrid.arc_from_chord(float(np.sqrt(
        ((regrid.unit_vectors(edge_source_lat, edge_source_lon)[0]
          - regrid.unit_vectors(edge_destination_lat,
                                edge_destination_lon)[0]) ** 2).sum())))
    below = exact
    for _ in range(4096):
        if not edge_reaches(below):
            break
        below = np.nextafter(below, 0.0)
    else:
        raise SystemExit("could not find a bound the observation is outside")
    above = below
    for _ in range(4096):
        if edge_reaches(above):
            break
        above = np.nextafter(above, np.inf)
    else:
        raise SystemExit("could not find a bound the observation is inside")
    if edge_reaches(below) or not edge_reaches(above):
        raise SystemExit("the bound-edge pair does not straddle the flip")
    print(f"bound edge flips between {below!r} and {above!r} metres "
          f"({int(above.view(np.int64) - below.view(np.int64))} ULP apart)")
    for label, bound in (("below", below), ("above", above)):
        cases.append(emit_case(
            f"synthetic_bound_edge_{label}",
            source_latitude=edge_source_lat,
            source_longitude=edge_source_lon,
            destination_latitude=edge_destination_lat,
            destination_longitude=edge_destination_lon,
            values=np.array([[3.5, 9.5]]), valid=np.array([[True, True]]),
            method=regrid.NEAREST, max_distance_m=float(bound),
            provenance={"kind": "synthetic",
                        "why": "a real grid does not land one ULP from the "
                               "bound on demand; this pins the predicate",
                        "flip": f"reachable={label == 'above'}"}))

    manifest = {
        "generated_by": "tools/rustwx/crates/obs-regrid/golden/"
                        "gen_regrid_goldens.py",
        "reference": "gpuwm.verify.obs.regrid (scipy.spatial.cKDTree + "
                     "numpy.add.at), run on real staged bytes",
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "python": sys.version.split()[0],
        "probes": probes,
        "divergence": {
            "what": "nearest-neighbour tie-breaking",
            "python": "scipy returns whichever exactly-tied source point "
                      "the k-d tree traversal reached first",
            "rust": "the lowest flat source index wins",
            "why": "the battery's premise is that every arm is remapped by "
                   "the identical integer array, so no score can differ "
                   "because a neighbour search broke a tie differently; a "
                   "tie broken by traversal order is exactly that defect, "
                   "one rebuild away",
            "measured": probes["ties"],
            "control": "synthetic_tie_control below is generated by the "
                       "PERTURBED grid, where the tie is broken by real "
                       "geometry and both engines must agree exactly",
        },
        "cases": cases,
    }

    # 6. The perturbed control for the documented divergence.  Two source
    #    cells at the same place is a tie neither engine has to resolve
    #    the same way; move one by a hair and the answer becomes a fact
    #    of the geometry, which both engines MUST get right.
    tie_lat = np.array([[10.0, 10.0]])
    tie_lon = np.array([[20.0, 20.0]])
    perturbed_lon = np.array([[20.0, np.nextafter(20.0, 21.0)]])
    manifest["cases"].append(emit_case(
        "synthetic_tie_degenerate",
        source_latitude=tie_lat, source_longitude=tie_lon,
        destination_latitude=np.array([[10.0]]),
        destination_longitude=np.array([[20.0]]),
        values=np.array([[1.0, 2.0]]), valid=np.array([[True, True]]),
        method=regrid.NEAREST, max_distance_m=50_000.0,
        provenance={"kind": "synthetic",
                    "why": "the exact tie itself; obs-regrid must answer "
                           "source index 0 by its own rule, and scipy's "
                           "answer here is NOT the contract"}))
    manifest["cases"].append(emit_case(
        "synthetic_tie_control",
        source_latitude=tie_lat, source_longitude=perturbed_lon,
        destination_latitude=np.array([[10.0]]),
        destination_longitude=np.array([[20.0]]),
        values=np.array([[1.0, 2.0]]), valid=np.array([[True, True]]),
        method=regrid.NEAREST, max_distance_m=50_000.0,
        provenance={"kind": "synthetic-control",
                    "why": "the perturbed twin of the degenerate case: one "
                           "ULP of longitude makes the nearest neighbour a "
                           "fact of the geometry, and both engines must "
                           "agree bit for bit"}))

    OUT.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicitly: `tools/rustwx/crates/** -text` stores this
    # tree byte for byte, so a Windows generator writing CRLF would
    # commit CRLF into a tree the repository keeps on LF.
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        newline="\n")
    total = sum(path.stat().st_size for path in OUT.rglob("*.bin"))
    print(f"wrote {len(manifest['cases'])} cases, "
          f"{total / 1_048_576:.2f} MiB of arrays, to {OUT}")


if __name__ == "__main__":
    main()
