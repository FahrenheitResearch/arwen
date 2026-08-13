"""A/B one LETKF analysis between solve devices, on identical inputs.

WHY THIS EXISTS.  ``tools/da_cycle_prepared.py`` defaults ``--solve-device``
to ``host``, so a bare cycling run analyses with numpy while a purpose-built
device eigensolver (:mod:`gpuwm.core.jacobi_eigh`) sits behind an opt-in
flag.  Committed cycle reports put the analysis at 1317 to 1477 seconds a
leg against an 8434-second cycle -- 65% of the wall clock -- so the choice of
default is the single largest scheduling decision in the DA path.  Changing
a default on that evidence alone would be changing it on an *inference*: no
receipt on disk holds the same leg solved both ways, and the reports that
measure the host arm were produced by driver scripts that hard-coded
``solve_device="host"`` and never ran the other one.  This tool produces the
missing measurement.

WHAT IT MEASURES, and why each part is here.

* **Wall per stage, both arms.**  ``gpuwm.da.letkf.analyze`` now splits its
  own clock three ways (setup, chunk loop, finish) and
  ``assimilate_radar_grid`` records the device arm's staging and unstaging
  separately.  The split matters because the two arms can differ for
  opposite reasons: the localisation weighting of phase 1 runs at EVERY
  gridpoint while the eigensolve runs only at the active ones, so on a
  radar-sparse domain a device arm can win the solve and still lose the
  leg to a transfer.  A single wall number cannot tell those apart.

* **Analysis-field deltas.**  Per field, max-abs and max-rel between the
  two arms' increments, plus the same statistics against the ensemble-mean
  increment magnitude, so "agrees to rounding" is a measured claim and not
  a hope.  The two arms are NOT expected to agree bitwise -- different
  eigensolvers, different summation orders -- and this is where that is
  quantified rather than assumed.

* **GPU contention, both interleavings.**  During a real DA cycle the card
  is integrating the ensemble.  A device analysis therefore does not get an
  idle GPU, and the honest question is not "which arm is faster alone" but
  "which arrangement finishes the leg sooner".  ``--interleave`` measures
  both: ``serial`` runs the ensemble-shaped load to completion and then the
  analysis (what a cycle does today), ``concurrent`` runs them together
  (what a host analysis makes possible, and what a device analysis has to
  share a card for).  The load is work-conserving across interleavings --
  the same iteration count either way -- so the two totals are comparable.

WHAT IT REFUSES TO PRETEND.  A bundle records whether it is ``real`` (the
analysis inputs of an actual DA leg, dumped by
``tools/da_cycle_prepared.py --dump-analysis-bundle``) or ``synthetic``
(built here from a twin-experiment construction).  That word is copied into
every receipt and into the verdict line.  A synthetic bundle proves the
instrument works and gives a shape-comparable ratio; it does not settle a
default, and the receipt says so in words.

USAGE

    # 1. get a bundle.  From a real leg, in the DA driver:
    #      tools/da_cycle_prepared.py ... --dump-analysis-bundle DIR
    #    or build one here, for instrument checks and shape studies:
    python -m tools.da_solve_ab make-bundle --out BUNDLE \\
        --nz 6 --ny 40 --nx 40 --members 10 --radars 2

    # 2. run the A/B
    python -m tools.da_solve_ab run --bundle BUNDLE --out receipt.json \\
        --device host --device cuda --repeat 3 \\
        --interleave alone --interleave serial --interleave concurrent

Every arm runs in a FRESH SUBPROCESS.  That is not tidiness: a device
context created by one arm changes what the next arm's allocator and the
chunk sizer see, and the sizer reads the card's free memory.  Two arms in
one process do not measure two arms.

EXPERIMENTAL.  Nothing in the forecast or DA path imports this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: Schemas.  The bundle is the replayable input set; the receipt is the
#: comparison.  Both are versioned because a later revision that changes
#: what is measured must not be silently readable as this one.
BUNDLE_SCHEMA = "gpuwm-da.solve-ab-bundle.v1"
RECEIPT_SCHEMA = "gpuwm-da.solve-ab-receipt.v1"
ARM_SCHEMA = "gpuwm-da.solve-ab-arm.v1"

#: The two words a bundle may use about itself.  ``real`` means the arrays
#: came off an actual DA leg; ``synthetic`` means this module built them.
BUNDLE_KINDS = ("real", "synthetic")

#: How an arm may be interleaved with the ensemble-shaped GPU load.
INTERLEAVINGS = ("alone", "serial", "concurrent")

#: Solve devices this harness will drive.  ``auto`` is deliberately absent:
#: an A/B names the arm it ran, and "whatever the box chose" is not an arm.
DEVICES = ("host", "cuda")

#: Relative agreement at or below which the two arms are called equivalent
#: to rounding.  Anchored on the module's own measured figure for the two
#: eigensolvers, which ``gpuwm/da/radar_assimilation.py`` records as
#: agreeing to ~1e-11 relative, with an order of margin for the different
#: summation orders a device arm also introduces.
AGREEMENT_TOLERANCE = 1.0e-9


# ---------------------------------------------------------------------------
# digests and small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, *, block: int = 1 << 20) -> str:
    """Streamed digest.  Never ``read_bytes()`` -- these files are large."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    """The tree whose ``gpuwm`` an arm subprocess must import.

    Derived from the package this process actually imported, never from
    the current working directory: a subprocess that picks up a different
    checkout is measuring a different filter, and the receipt would name
    this one.
    """

    import gpuwm                                          # noqa: PLC0415

    return Path(gpuwm.__file__).resolve().parent.parent


def _jsonable(value):
    """Make numpy scalars and paths survive ``json.dump``."""

    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# the bundle: a replayable set of analysis inputs
# ---------------------------------------------------------------------------

def write_manifest(bundle: Path, manifest: dict) -> Path:
    path = bundle / "bundle.json"
    path.write_text(json.dumps(_jsonable(manifest), indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_manifest(bundle: Path) -> dict:
    path = Path(bundle) / "bundle.json"
    if not path.is_file():
        raise SystemExit(
            f"{bundle} carries no bundle.json.  A solve A/B bundle is the "
            "analysis inputs of one DA leg: the member checkpoints, the "
            "observation file, and the grid they were placed on.  Dump one "
            "from a real leg with tools/da_cycle_prepared.py "
            "--dump-analysis-bundle, or build a synthetic one with "
            "`make-bundle`.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema != BUNDLE_SCHEMA:
        raise SystemExit(f"{path}: schema {schema!r}, expected "
                         f"{BUNDLE_SCHEMA!r}")
    kind = manifest.get("kind")
    if kind not in BUNDLE_KINDS:
        raise SystemExit(f"{path}: kind {kind!r}, expected one of "
                         f"{BUNDLE_KINDS}")
    return manifest


def verify_bundle(bundle: Path, manifest: dict) -> dict:
    """Re-digest every input file.  Both arms must read the same bytes.

    This is the whole premise of the comparison, so it is checked rather
    than assumed: a bundle whose observation file was regenerated between
    the two arms would produce a difference that looks numerical and is
    not.
    """

    bundle = Path(bundle)
    checked = {}
    entries = [("observations", manifest["observations"])]
    entries += [(f"member_{entry['index']:03d}", entry)
                for entry in manifest["members"]]
    if manifest["grid"].get("path"):
        entries.append(("grid", manifest["grid"]))
    for label, entry in entries:
        path = bundle / entry["path"]
        if not path.is_file():
            raise SystemExit(f"bundle {bundle}: {label} is missing at "
                             f"{path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise SystemExit(
                f"bundle {bundle}: {label} at {path} digests {actual}, the "
                f"manifest says {entry['sha256']}.  The two arms would not "
                "be reading the same inputs; refusing to compare them.")
        checked[label] = actual
    return checked


def load_grid(bundle: Path, manifest: dict):
    """Rebuild the :class:`~gpuwm.obs.target_grid.TargetGrid`.

    Two forms, because a real bundle and a synthetic one honestly have
    different provenance.  ``wrfout`` re-reads the history file the
    observations were gridded against, which is what a real leg has.
    ``projection_npz`` carries the projection parameters plus the
    coordinate arrays, which is what a constructed grid has and what a
    real leg does NOT, so the two are never confusable.
    """

    from gpuwm.obs.target_grid import TargetGrid          # noqa: PLC0415

    bundle = Path(bundle)
    spec = manifest["grid"]
    source = spec["source"]
    if source == "wrfout":
        grid = TargetGrid.from_wrfout(bundle / spec["path"],
                                      frame=int(spec.get("frame", 0)),
                                      name=spec.get("name"))
    elif source == "projection_npz":
        from gpuwm.static.projection import (             # noqa: PLC0415
            projection_class)

        arrays = np.load(bundle / spec["path"], allow_pickle=False)
        params = spec["projection"]
        projection = projection_class(params["map_proj"])(
            ref_lat=float(params["ref_lat"]),
            ref_lon=float(params["ref_lon"]),
            truelat1=float(params["truelat1"]),
            truelat2=float(params["truelat2"]),
            stand_lon=float(params["stand_lon"]),
            dx=float(params["dx"]), dy=float(params["dy"]),
            e_we=int(params["e_we"]), e_sn=int(params["e_sn"]))
        grid = TargetGrid.from_projection(
            projection, z_w=np.asarray(arrays["z_w"]),
            terrain_m=np.asarray(arrays["terrain_m"]),
            name=spec.get("name", "solve-ab"),
            source=spec.get("provenance", "solve-ab bundle"))
    else:
        raise SystemExit(f"bundle grid source {source!r} is not one this "
                         "harness can rebuild ('wrfout', 'projection_npz')")
    expected = spec.get("identity_sha256")
    if expected:
        grid.require_identity(expected)
    return grid


def serialize_config(cfg) -> dict:
    """A :class:`RadarAssimilationConfig` as JSON, minus the treatment.

    ``solve_device`` is deliberately dropped: it is the one field the A/B
    varies, and a bundle that carried it would let a careless replay run
    both arms on whatever the dumping run happened to use.
    """

    import dataclasses                                    # noqa: PLC0415

    from gpuwm.da.letkf import Localization               # noqa: PLC0415

    out = {}
    for spec in dataclasses.fields(cfg):
        if spec.name == "solve_device":
            continue
        value = getattr(cfg, spec.name)
        if value is None:
            continue
        if isinstance(value, Localization):
            out[spec.name] = {"horizontal_m": float(value.horizontal_m),
                              "vertical_m": float(value.vertical_m)}
        elif isinstance(value, (tuple, list)):
            out[spec.name] = list(value)
        else:
            out[spec.name] = value
    return out


def build_config(manifest: dict, device: str):
    """The :class:`RadarAssimilationConfig` both arms share, bar the device.

    Everything except ``solve_device`` comes out of the manifest, so the
    two arms differ in exactly one field.  That is the A/B's treatment and
    it is the only difference the harness is willing to introduce.
    """

    import dataclasses                                    # noqa: PLC0415

    from gpuwm.da.letkf import Localization               # noqa: PLC0415
    from gpuwm.da.radar_assimilation import (             # noqa: PLC0415
        RadarAssimilationConfig)

    localization_fields = {
        spec.name for spec in dataclasses.fields(RadarAssimilationConfig)
        if "Localization" in str(spec.type)}
    spec = dict(manifest["config"])
    spec.pop("solve_device", None)
    kwargs = {}
    for name, value in spec.items():
        if name in localization_fields and value is not None:
            kwargs[name] = Localization(**value)
        elif name == "analysis_fields":
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    kwargs["solve_device"] = device
    return RadarAssimilationConfig(**kwargs)


def dump_real_bundle(out, *, checkpoints, obs_path, grid_wrfout, grid, cfg,
                     note: str = "", extra: dict | None = None) -> dict:
    """Copy one real leg's analysis inputs into a replayable bundle.

    Called from the DA driver at the analysis seam, where every input
    exists and is still on disk: the member checkpoints it just staged,
    the observation file for that leg, and the history file the
    observations were gridded against.  Copied rather than referenced,
    because a stage directory is usually a tmpfs the next leg overwrites
    and a bundle that points at deleted arrays is not a bundle.

    This is the only way a solve-device A/B can be run against a REAL
    analysis without also paying for the forecast legs that produced it,
    which is what makes the comparison affordable enough to repeat.
    """

    import shutil                                         # noqa: PLC0415

    out = Path(out)
    (out / "members").mkdir(parents=True, exist_ok=True)

    obs_path = Path(obs_path)
    obs_copy = out / "obs-radar-grid.nc"
    shutil.copy2(obs_path, obs_copy)

    grid_wrfout = Path(grid_wrfout)
    grid_copy = out / "grid-wrfout.nc"
    shutil.copy2(grid_wrfout, grid_copy)

    members = []
    for index in sorted(int(k) for k in checkpoints):
        source = Path(checkpoints[index])
        target = out / "members" / f"member_{index:03d}.npz"
        shutil.copy2(source, target)
        members.append({"index": int(index),
                        "path": f"members/member_{index:03d}.npz",
                        "sha256": sha256_file(target),
                        "origin": str(source)})

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "kind": "real",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note or ("The analysis inputs of a real DA leg, copied at "
                         "the analysis seam."),
        "grid": {
            "source": "wrfout",
            "path": "grid-wrfout.nc",
            "frame": 0,
            "name": grid.name,
            "identity_sha256": grid.identity_sha256(),
            "origin": str(grid_wrfout),
            "sha256": sha256_file(grid_copy),
        },
        "observations": {"path": "obs-radar-grid.nc",
                         "sha256": sha256_file(obs_copy),
                         "origin": str(obs_path)},
        "members": members,
        "shape": {"nz": int(grid.nz), "ny": int(grid.ny),
                  "nx": int(grid.nx), "members": len(members)},
        "config": serialize_config(cfg),
    }
    if extra:
        manifest["origin"] = extra
    write_manifest(out, manifest)
    return manifest


# ---------------------------------------------------------------------------
# synthetic bundle construction
# ---------------------------------------------------------------------------
#
# A twin-experiment: an ensemble centre, R independent correlated draws
# around it as members, and one more independent draw as the truth the
# radars observe.  Prior-mean error and ensemble spread then match by
# construction, which is the setting in which the filter's own gate on
# this shape is a fair demand.  It is the same construction
# tests/test_radar_assimilation.py builds its world from, parameterised,
# and it goes through the radar lane's OWN writer and the real checkpoint
# layout -- never a dict pretending to be a file.

def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        for axis in (1, 2):
            field = (np.roll(field, 1, axis) + field
                     + np.roll(field, -1, axis)) / 3.0
    return field


def _correlated(rng, sigma, shape):
    field = _smooth(rng.normal(0.0, 1.0, shape))
    return sigma * field / field.std()


def make_bundle(out: Path, *, nz: int, ny: int, nx: int, members: int,
                radars: int, dx_m: float, top_m: float, margin: int,
                horizontal_loc_m: float, vertical_loc_m: float,
                obs_err_ms: float, thin_cells: int, rtps_alpha: float,
                memory_budget_mib: float, seed: int,
                ref_lat: float, ref_lon: float) -> dict:
    """Write a synthetic but structurally real bundle at the named shape."""

    from gpuwm.da import obsop                            # noqa: PLC0415
    from gpuwm.da.radar_assimilation import (             # noqa: PLC0415
        grid_rotation, mass_to_u_faces, mass_to_v_faces, mass_to_w_faces)
    from gpuwm.obs.radar_grid import write_radar_grid     # noqa: PLC0415
    from gpuwm.obs.superob import (                       # noqa: PLC0415
        GriddedObservations, SuperobParams)
    from gpuwm.obs.target_grid import TargetGrid          # noqa: PLC0415
    from gpuwm.static.lambert import LambertGrid          # noqa: PLC0415

    if radars < 1:
        raise SystemExit("a radar-grid product needs at least one radar")
    if 2 * margin >= min(ny, nx):
        raise SystemExit(
            f"--margin {margin} leaves no observed interior in a "
            f"{ny}x{nx} domain; the observed band is "
            f"{ny - 2 * margin}x{nx - 2 * margin}")

    out = Path(out)
    (out / "members").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)

    projection = LambertGrid(
        ref_lat=ref_lat, ref_lon=ref_lon, truelat1=ref_lat - 2.0,
        truelat2=ref_lat + 2.0, stand_lon=ref_lon, dx=dx_m, dy=dx_m,
        e_we=nx + 1, e_sn=ny + 1)
    # Terrain with real relief, so the vertical localisation metric is
    # exercised on columns that differ.  A flat grid would let a
    # precomputed offset table stand in for the per-column distance the
    # filter actually computes, and would under-measure phase 1.
    terrain = 300.0 + 400.0 * _correlated(rng, 1.0, (1, ny, nx))[0]
    column = np.linspace(0.0, top_m, nz + 1)
    z_w = terrain[None, :, :] + column[:, None, None] * (
        1.0 - terrain[None, :, :] / (top_m + terrain.max() + 1.0))
    z_w = np.ascontiguousarray(z_w, dtype=np.float64)
    grid = TargetGrid.from_projection(projection, z_w=z_w,
                                      terrain_m=terrain, name="solve-ab",
                                      source="solve-ab synthetic bundle")

    centre = {
        "u": 8.0 + _smooth(rng.normal(0.0, 3.0, shape)),
        "v": -2.0 + _smooth(rng.normal(0.0, 3.0, shape)),
        "w": _smooth(rng.normal(0.0, 0.5, shape)),
        "thp": _smooth(rng.normal(0.0, 1.0, shape)),
    }
    truth = {name: values + _correlated(rng, sigma, shape)
             for (name, values), sigma
             in zip(centre.items(), (1.5, 1.5, 0.3, 0.5))}

    stagger = {"u": mass_to_u_faces, "v": mass_to_v_faces,
               "w": mass_to_w_faces}
    member_entries = []
    for index in range(members):
        fields = {}
        for name, sigma in (("u", 1.5), ("v", 1.5), ("w", 0.3)):
            fields[name] = stagger[name](
                centre[name] + _correlated(rng, sigma, shape))
        fields["thp"] = centre["thp"] + _correlated(rng, 0.5, shape)
        fields["p"] = np.broadcast_to(
            np.linspace(9.0e4, 4.0e4, nz)[:, None, None], shape).copy()
        path = out / "members" / f"member_{index:03d}.npz"
        np.savez(path, **{f"state/{name}": np.asarray(values, np.float32)
                          for name, values in fields.items()})
        member_entries.append({"index": index,
                               "path": f"members/member_{index:03d}.npz",
                               "sha256": sha256_file(path)})

    # -- observations, through the radar lane's own writer -----------------
    sites = []
    for slot in range(radars):
        # Sites walk the rim, so their beams cross the interior at
        # genuinely different angles: co-located radars would make the
        # velocity operator rank-deficient and the filter's conditioning
        # unrepresentative of a real multi-radar analysis.
        frac = (slot + 1) / (radars + 1)
        if slot % 2 == 0:
            j, i = int(frac * (ny - 1)), 1
        else:
            j, i = 1, int(frac * (nx - 1))
        sites.append(obsop.RadarSite(
            latitude_deg=float(grid.lat[j, i]),
            longitude_deg=float(grid.lon[j, i]),
            altitude_m=350.0 + 10.0 * slot,
            name=f"R{slot:03d}"))

    geometry = obsop.GridGeometry.from_target_grid(grid)
    interior = np.zeros(shape, bool)
    interior[:, margin:ny - margin, margin:nx - margin] = True
    sina, cosa = grid_rotation(grid)
    u_e, v_n = obsop.earth_relative_winds(truth["u"], truth["v"], sina, cosa)

    east = np.zeros((radars,) + shape)
    north = np.zeros((radars,) + shape)
    up = np.zeros((radars,) + shape)
    vr_obs = np.zeros((radars,) + shape)
    for slot, site in enumerate(sites):
        beam = obsop.beam_geometry(geometry, site)
        e, n, u = (np.broadcast_to(np.asarray(c, np.float64), shape).copy()
                   for c in beam.unit_vector_enu())
        east[slot], north[slot], up[slot] = e, n, u
        vr_obs[slot] = (u_e * e + v_n * n + truth["w"] * u
                        + rng.normal(0.0, obs_err_ms, shape))

    vr_mask = np.broadcast_to(interior, (radars,) + shape).astype(np.int8)
    zeros = np.zeros(shape, np.float32)
    observations = GriddedObservations(
        z_obs=zeros, z_mask=np.zeros(shape, np.int8), z_err=zeros,
        z_max=zeros, z_mean=zeros, z_count=np.zeros(shape, np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), obs_err_ms,
                        0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((radars,) + shape, np.int32),
        vr_beam_east=east.astype(np.float32),
        vr_beam_north=north.astype(np.float32),
        vr_beam_up=up.astype(np.float32),
        radars=[{"id": site.name, "lat_deg": site.latitude_deg,
                 "lon_deg": site.longitude_deg, "alt_m": site.altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"} for site in sites],
        counts=[], provenance=[])
    obs_path = out / "obs-radar-grid.nc"
    if obs_path.exists():
        obs_path.unlink()
    write_radar_grid(obs_path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)

    grid_npz = out / "grid.npz"
    np.savez(grid_npz, z_w=grid.z_w, terrain_m=grid.terrain_m)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "kind": "synthetic",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Twin-experiment construction, not a real leg.  Structure "
                 "is real -- the radar lane's own writer, the restart "
                 "checkpoint layout, terrain-following levels -- but the "
                 "atmosphere and the observations were generated here.  A "
                 "verdict from this bundle is about the harness and about "
                 "the shape, never about a case."),
        "seed": seed,
        "grid": {
            "source": "projection_npz",
            "path": "grid.npz",
            "name": grid.name,
            "identity_sha256": grid.identity_sha256(),
            "provenance": grid.source,
            "projection": {
                "map_proj": projection.map_proj,
                "ref_lat": ref_lat, "ref_lon": ref_lon,
                "truelat1": ref_lat - 2.0, "truelat2": ref_lat + 2.0,
                "stand_lon": ref_lon, "dx": dx_m, "dy": dx_m,
                "e_we": nx + 1, "e_sn": ny + 1,
            },
            "sha256": sha256_file(grid_npz),
        },
        "observations": {"path": "obs-radar-grid.nc",
                         "sha256": sha256_file(obs_path)},
        "members": member_entries,
        "shape": {"nz": nz, "ny": ny, "nx": nx, "members": members,
                  "radars": radars, "observed_columns":
                      int((ny - 2 * margin) * (nx - 2 * margin))},
        "config": {
            "localization": {"horizontal_m": horizontal_loc_m,
                             "vertical_m": vertical_loc_m},
            "analysis_fields": ["u", "v"],
            "rtps_alpha": rtps_alpha,
            "velocity": True,
            "velocity_thinning_cells": thin_cells,
            "memory_budget_mib": memory_budget_mib,
        },
    }
    write_manifest(out, manifest)
    return manifest


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------

def run_arm(bundle: Path, device: str, *, out_json: Path,
            out_npz: Path | None) -> dict:
    """Solve the bundle's analysis once, on ``device``, and record it."""

    t0 = time.perf_counter()
    from gpuwm.da.letkf import LetkfDiagnostics           # noqa: PLC0415
    from gpuwm.da.radar_assimilation import (             # noqa: PLC0415
        assimilate_radar_grid)
    t_import = time.perf_counter() - t0

    bundle = Path(bundle)
    manifest = read_manifest(bundle)
    t1 = time.perf_counter()
    digests = verify_bundle(bundle, manifest)
    grid = load_grid(bundle, manifest)
    cfg = build_config(manifest, device)
    checkpoints = {int(entry["index"]): bundle / entry["path"]
                   for entry in manifest["members"]}
    t_load = time.perf_counter() - t1

    diagnostics = LetkfDiagnostics()
    t2 = time.perf_counter()
    increments, provenance = assimilate_radar_grid(
        checkpoints, bundle / manifest["observations"]["path"], grid, cfg,
        diagnostics=diagnostics)
    t_assimilate = time.perf_counter() - t2

    if out_npz is not None:
        payload = {}
        for index in sorted(increments):
            for name, values in increments[index].items():
                payload[f"{index:03d}/{name}"] = np.asarray(values)
        np.savez(out_npz, **payload)

    record = {
        "schema": ARM_SCHEMA,
        "device": device,
        "bundle": str(bundle),
        "bundle_kind": manifest["kind"],
        "bundle_digests": digests,
        "grid_identity_sha256": grid.identity_sha256(),
        "seconds": {
            "import": round(t_import, 3),
            "load_and_verify": round(t_load, 3),
            "assimilate": round(t_assimilate, 3),
            "letkf_setup": round(float(diagnostics.setup_seconds), 3),
            "letkf_solve": round(float(diagnostics.solve_seconds), 3),
            "letkf_finish": round(float(diagnostics.finish_seconds), 3),
            "letkf_weights": round(float(diagnostics.weights_seconds), 3),
            "letkf_transform": round(
                float(diagnostics.transform_seconds), 3),
            "stage_to_device": provenance.get("solve_stage_seconds", 0.0),
            "unstage_from_device":
                provenance.get("solve_unstage_seconds", 0.0),
        },
        # The counts that say the two arms saw the same observations.  A
        # difference here is a plumbing fault wearing a numerical costume,
        # and the comparison checks it before it reports any delta.
        "filter": provenance["filter"],
        "innovations": provenance["innovations"],
        "velocity_thinning": provenance.get("velocity_thinning"),
        "increments_npz": str(out_npz) if out_npz else None,
        "platform": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "node": platform.node(),
        },
    }
    try:
        import cupy                                       # noqa: PLC0415

        record["platform"]["cupy"] = cupy.__version__
        if device == "cuda":
            props = cupy.cuda.runtime.getDeviceProperties(
                cupy.cuda.runtime.getDevice())
            record["platform"]["device_name"] = props["name"].decode()
    except Exception as exc:                               # pragma: no cover
        record["platform"]["cupy"] = f"unavailable: {type(exc).__name__}"

    Path(out_json).write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# the ensemble-shaped GPU load
# ---------------------------------------------------------------------------
#
# During a DA cycle the card is integrating the ensemble.  Reproducing
# that here without a prepared case means standing in for it, and the
# stand-in is labelled in every receipt so nobody reads it as the real
# forecast: it is a documented occupancy generator, work-conserving across
# interleavings, not a member forecast.  An operator with a real case
# should pass --load-command instead, and the receipt records which ran.

def run_load(iters: int, mib: float) -> dict:
    """Occupy a card with a fixed amount of work, then exit.

    Fixed ITERATION COUNT rather than fixed duration, because the two
    interleavings are only comparable if the load does the same work in
    both.  A duration-based load would hand the concurrent arrangement a
    smaller job and manufacture the win it was meant to test.

    Exits on its own.  Nothing in this harness terminates a process it
    started.
    """

    import cupy as cp                                     # noqa: PLC0415

    side = max(256, int((mib * (1 << 20) / 8 / 3) ** 0.5))
    a = cp.random.random((side, side), dtype=cp.float64)
    b = cp.random.random((side, side), dtype=cp.float64)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a = cp.matmul(a, b)
        a /= cp.linalg.norm(a)
    cp.cuda.runtime.deviceSynchronize()
    return {"kind": "synthetic-occupancy", "iters": iters,
            "matrix_side": side,
            "seconds": round(time.perf_counter() - t0, 3)}


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare_increments(reference: Path, candidate: Path) -> dict:
    """Per-field agreement between two arms' increments.

    Reported three ways on purpose.  ``max_abs`` is the raw difference;
    ``max_rel_to_field`` normalises it by the largest increment the
    reference arm produced for that field, which is what "agrees to
    rounding" has to mean when the increments themselves span orders of
    magnitude; ``rms`` is the whole-field summary that a max cannot give.
    A comparison that reported only a max would call one unlucky
    gridpoint a disagreement, and one that reported only an RMS would hide
    it.
    """

    with np.load(reference, allow_pickle=False) as ref, \
            np.load(candidate, allow_pickle=False) as cand:
        keys = sorted(set(ref.files))
        if keys != sorted(set(cand.files)):
            raise SystemExit(
                "the two arms returned different increment keys: "
                f"{sorted(set(ref.files)) } vs {sorted(set(cand.files))}")
        per_field: dict[str, dict] = {}
        for key in keys:
            member, _, field = key.partition("/")
            a = np.asarray(ref[key], dtype=np.float64)
            b = np.asarray(cand[key], dtype=np.float64)
            if a.shape != b.shape:
                raise SystemExit(f"{key}: shapes {a.shape} vs {b.shape}")
            delta = np.abs(b - a)
            scale = float(np.max(np.abs(a)))
            entry = per_field.setdefault(field, {
                "max_abs": 0.0, "reference_max_abs": 0.0,
                "sum_sq": 0.0, "count": 0, "bitwise_identical": True,
                "members": 0})
            entry["max_abs"] = max(entry["max_abs"], float(delta.max()))
            entry["reference_max_abs"] = max(entry["reference_max_abs"],
                                             scale)
            entry["sum_sq"] += float(np.sum(delta * delta))
            entry["count"] += int(delta.size)
            entry["members"] += 1
            if entry["bitwise_identical"] and not np.array_equal(a, b):
                entry["bitwise_identical"] = False

    out = {}
    for field, entry in per_field.items():
        scale = entry["reference_max_abs"]
        out[field] = {
            "members": entry["members"],
            "max_abs": entry["max_abs"],
            "reference_max_abs": scale,
            "max_rel_to_field": (entry["max_abs"] / scale if scale > 0.0
                                 else 0.0),
            "rms": (entry["sum_sq"] / entry["count"]) ** 0.5,
            "bitwise_identical": entry["bitwise_identical"],
        }
    return out


def compare_counts(reference: dict, candidate: dict) -> dict:
    """Did the two arms see the same observations and the same points?

    The A/B-arms law: an exact-zero delta can mean "the treatment agrees"
    or "the treatment never ran", and these counts are what separates
    them.  ``active_points`` and the innovation counts are properties of
    the OBSERVATIONS and the localisation, not of the arithmetic, so they
    must match exactly across arms.  ``eigensolver`` must DIFFER when a
    host arm is compared with a device one -- the host arm cannot reach
    the project kernel -- and a receipt where it does not is a receipt
    where the device arm silently fell back.
    """

    ref_filter = reference["filter"]
    cand_filter = candidate["filter"]
    mismatches = {}
    for key in ("active_points", "total_points", "max_local_obs",
                "members", "chunk_points_initial"):
        if key in ref_filter and ref_filter.get(key) != cand_filter.get(key):
            mismatches[key] = [ref_filter.get(key), cand_filter.get(key)]
    # ``innovations`` is a LIST of per-batch entries, one per observation
    # type the filter actually solved.  Both the names and the counts have
    # to match: a run that dropped a batch would otherwise pass a
    # comparison that only looked at totals.
    def _counts(record):
        return {entry["name"]: entry["observations"]
                for entry in record["innovations"]}

    ref_counts = _counts(reference)
    cand_counts = _counts(candidate)
    if ref_counts != cand_counts:
        mismatches["innovation_counts"] = [ref_counts, cand_counts]
    return {
        "same_inputs": not mismatches,
        "mismatches": mismatches,
        "eigensolver": [ref_filter.get("eigensolver"),
                        cand_filter.get("eigensolver")],
        "active_points": ref_filter.get("active_points"),
        "total_points": ref_filter.get("total_points"),
        "chunk_points": [ref_filter.get("chunk_points"),
                         cand_filter.get("chunk_points")],
        "chunk_oom_shrinks": [ref_filter.get("chunk_oom_shrinks"),
                              cand_filter.get("chunk_oom_shrinks")],
    }


# ---------------------------------------------------------------------------
# the run driver
# ---------------------------------------------------------------------------

def _arm_command(bundle: Path, device: str, out_json: Path,
                 out_npz: Path | None) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "arm",
           "--bundle", str(bundle), "--device", device,
           "--out", str(out_json)]
    if out_npz is not None:
        cmd += ["--increments", str(out_npz)]
    return cmd


def _child_env() -> dict:
    env = dict(os.environ)
    root = str(_repo_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (root if not existing
                         else root + os.pathsep + existing)
    # A device arm needs the card.  The harness never sets or clears the
    # local-GPU authorisation itself -- that is a state the operator owns
    # -- it only refuses to smuggle a host-only marker into a cuda arm.
    return env


def _spawn(cmd: list[str], log: Path, env: dict):
    handle = open(log, "wb")
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                            env=env), handle


def run_ab(bundle: Path, out: Path, *, devices: list[str],
           interleavings: list[str], repeat: int, load_iters: int,
           load_mib: float, load_command: str | None,
           work: Path) -> dict:
    """Run every (device, interleaving) cell ``repeat`` times, then judge."""

    bundle = Path(bundle)
    manifest = read_manifest(bundle)
    verify_bundle(bundle, manifest)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    env = _child_env()

    load_cmd = None
    if set(interleavings) - {"alone"}:
        if load_command:
            load_cmd = load_command
            load_kind = "operator-supplied"
        else:
            load_cmd = " ".join([
                subprocess.list2cmdline([sys.executable]),
                subprocess.list2cmdline([str(Path(__file__).resolve())]),
                "load", "--iters", str(load_iters),
                "--mib", str(load_mib)])
            load_kind = "synthetic-occupancy"
    else:
        load_kind = None

    cells: list[dict] = []
    for device in devices:
        for interleaving in interleavings:
            for trial in range(repeat):
                tag = f"{device}-{interleaving}-{trial}"
                arm_json = work / f"arm-{tag}.json"
                arm_npz = work / f"incr-{tag}.npz"
                arm_log = work / f"arm-{tag}.log"
                load_log = work / f"load-{tag}.log"
                cmd = _arm_command(bundle, device, arm_json, arm_npz)

                t_start = time.perf_counter()
                load_seconds = None
                if interleaving == "alone":
                    proc, handle = _spawn(cmd, arm_log, env)
                    rc = proc.wait()
                    handle.close()
                    load_rc = None
                elif interleaving == "serial":
                    lproc, lhandle = _spawn(
                        ["cmd", "/c", load_cmd] if os.name == "nt"
                        else ["sh", "-c", load_cmd], load_log, env)
                    t_load0 = time.perf_counter()
                    load_rc = lproc.wait()
                    lhandle.close()
                    load_seconds = time.perf_counter() - t_load0
                    proc, handle = _spawn(cmd, arm_log, env)
                    rc = proc.wait()
                    handle.close()
                else:                                     # concurrent
                    lproc, lhandle = _spawn(
                        ["cmd", "/c", load_cmd] if os.name == "nt"
                        else ["sh", "-c", load_cmd], load_log, env)
                    t_load0 = time.perf_counter()
                    proc, handle = _spawn(cmd, arm_log, env)
                    rc = proc.wait()
                    handle.close()
                    load_rc = lproc.wait()
                    lhandle.close()
                    load_seconds = time.perf_counter() - t_load0
                end_to_end = time.perf_counter() - t_start

                cell = {
                    "device": device,
                    "interleaving": interleaving,
                    "trial": trial,
                    "returncode": rc,
                    "load_returncode": load_rc,
                    "end_to_end_seconds": round(end_to_end, 3),
                    "load_seconds": (None if load_seconds is None
                                     else round(load_seconds, 3)),
                    "arm_json": str(arm_json),
                    "log": str(arm_log),
                }
                if rc == 0 and arm_json.is_file():
                    cell["arm"] = json.loads(
                        arm_json.read_text(encoding="utf-8"))
                    cell["increments_npz"] = str(arm_npz)
                else:
                    cell["failure_tail"] = _tail(arm_log)
                cells.append(cell)

    receipt = _judge(bundle, manifest, cells, devices, interleavings,
                     load_kind, load_cmd, repeat)
    Path(out).write_text(
        json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return receipt


def _tail(path: Path, lines: int = 40) -> list[str]:
    if not Path(path).is_file():
        return []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[-lines:]


def _stat(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (ordered[mid] if len(ordered) % 2
              else 0.5 * (ordered[mid - 1] + ordered[mid]))
    return {"n": len(ordered), "min": round(ordered[0], 3),
            "median": round(median, 3), "max": round(ordered[-1], 3)}


def _judge(bundle: Path, manifest: dict, cells: list[dict],
           devices: list[str], interleavings: list[str],
           load_kind: str | None, load_cmd: str | None,
           repeat: int) -> dict:
    ok = [c for c in cells if c.get("arm")]
    failed = [c for c in cells if not c.get("arm")]

    timings: dict[str, dict] = {}
    for device in devices:
        for interleaving in interleavings:
            subset = [c for c in ok if c["device"] == device
                      and c["interleaving"] == interleaving]
            if not subset:
                continue
            key = f"{device}/{interleaving}"
            timings[key] = {
                "end_to_end_seconds": _stat(
                    [c["end_to_end_seconds"] for c in subset]),
                "assimilate_seconds": _stat(
                    [c["arm"]["seconds"]["assimilate"] for c in subset]),
                "letkf_setup_seconds": _stat(
                    [c["arm"]["seconds"]["letkf_setup"] for c in subset]),
                "letkf_solve_seconds": _stat(
                    [c["arm"]["seconds"]["letkf_solve"] for c in subset]),
                "letkf_finish_seconds": _stat(
                    [c["arm"]["seconds"]["letkf_finish"] for c in subset]),
                "letkf_weights_seconds": _stat(
                    [c["arm"]["seconds"]["letkf_weights"] for c in subset]),
                "letkf_transform_seconds": _stat(
                    [c["arm"]["seconds"]["letkf_transform"]
                     for c in subset]),
                "stage_seconds": _stat(
                    [c["arm"]["seconds"]["stage_to_device"]
                     for c in subset]),
                "unstage_seconds": _stat(
                    [c["arm"]["seconds"]["unstage_from_device"]
                     for c in subset]),
                "load_seconds": _stat(
                    [c["load_seconds"] for c in subset
                     if c["load_seconds"] is not None]),
            }

    # -- speedups, per interleaving, host as the incumbent -----------------
    speedups = {}
    for interleaving in interleavings:
        host = timings.get(f"host/{interleaving}")
        cuda = timings.get(f"cuda/{interleaving}")
        if not host or not cuda:
            continue
        entry = {}
        for stage in ("end_to_end_seconds", "assimilate_seconds",
                      "letkf_solve_seconds"):
            h = host[stage].get("median")
            c = cuda[stage].get("median")
            entry[stage] = (None if not h or not c or c <= 0.0
                            else round(h / c, 3))
        speedups[interleaving] = entry

    # -- agreement --------------------------------------------------------
    agreement = None
    reference = next((c for c in ok if c["device"] == "host"), None)
    candidate = next((c for c in ok if c["device"] == "cuda"), None)
    if reference and candidate:
        agreement = {
            "counts": compare_counts(reference["arm"], candidate["arm"]),
            "fields": compare_increments(Path(reference["increments_npz"]),
                                         Path(candidate["increments_npz"])),
        }

    findings = []
    verdict = "INCONCLUSIVE"
    if failed:
        findings.append(
            f"{len(failed)} of {len(cells)} cells did not produce an arm "
            "record; see failure_tail on each")
    if agreement is None:
        findings.append(
            "no host/cuda pair completed, so no agreement statistic "
            "exists and no default can be decided from this receipt")
    else:
        counts = agreement["counts"]
        if not counts["same_inputs"]:
            findings.append(
                "the two arms did not see the same observations or the "
                f"same active points: {counts['mismatches']}.  The timing "
                "comparison is void")
            verdict = "VOID"
        worst_rel = max((f["max_rel_to_field"]
                         for f in agreement["fields"].values()), default=0.0)
        all_bitwise = all(f["bitwise_identical"]
                          for f in agreement["fields"].values())
        if all_bitwise:
            # Exact zero across a treatment boundary is the signature of a
            # treatment that never ran, not of two eigensolvers agreeing.
            findings.append(
                "every increment is BITWISE identical between the two "
                "arms.  Two different eigensolvers on two different "
                "devices do not agree to the byte; read this as the cuda "
                "arm having silently run the host path, and check "
                "filter.eigensolver in both arm records before believing "
                "any speedup here")
            verdict = "VOID"
        elif worst_rel > AGREEMENT_TOLERANCE:
            findings.append(
                f"worst relative field disagreement {worst_rel:.3e} "
                f"exceeds the {AGREEMENT_TOLERANCE:.1e} tolerance; the "
                "arms are not equivalent to rounding and the faster one "
                "is not simply the same analysis")
            verdict = "DISAGREE"
        elif verdict == "INCONCLUSIVE":
            verdict = "AGREE"

    if manifest["kind"] != "real" and verdict == "AGREE":
        findings.append(
            "this bundle is SYNTHETIC.  The harness and the agreement "
            "statistic are proven; the speedup is a property of this "
            "shape, and a default flip wants the same receipt off a real "
            "leg's dumped bundle")

    return {
        "schema": RECEIPT_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle": {"path": str(bundle), "kind": manifest["kind"],
                   "shape": manifest.get("shape"),
                   "note": manifest.get("note"),
                   "config": manifest["config"]},
        "devices": devices,
        "interleavings": interleavings,
        "repeat": repeat,
        "load": {"kind": load_kind, "command": load_cmd},
        "timings": timings,
        "speedup_host_over_cuda": speedups,
        "agreement": agreement,
        "verdict": verdict,
        "findings": findings,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="da_solve_ab",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser(
        "make-bundle",
        help="build a synthetic bundle at a named shape (labelled "
             "synthetic in every receipt it produces)")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--nz", type=int, default=6)
    make.add_argument("--ny", type=int, default=40)
    make.add_argument("--nx", type=int, default=40)
    make.add_argument("--members", type=int, default=10)
    make.add_argument("--radars", type=int, default=2)
    make.add_argument("--dx-m", type=float, default=3000.0)
    make.add_argument("--top-m", type=float, default=12000.0)
    make.add_argument("--margin", type=int, default=7,
                      help="unobserved rim, in columns; the observed "
                           "interior is what drives active_points")
    make.add_argument("--horizontal-loc-m", type=float, default=15000.0)
    make.add_argument("--vertical-loc-m", type=float, default=4000.0)
    make.add_argument("--obs-err-ms", type=float, default=1.0)
    make.add_argument("--thin-cells", type=int, default=2)
    make.add_argument("--rtps-alpha", type=float, default=0.9)
    make.add_argument("--memory-budget-mib", type=float, default=6144.0)
    make.add_argument("--seed", type=int, default=20260812)
    make.add_argument("--ref-lat", type=float, default=35.0)
    make.add_argument("--ref-lon", type=float, default=-97.0)

    arm = sub.add_parser("arm", help="internal: solve one arm and exit")
    arm.add_argument("--bundle", type=Path, required=True)
    arm.add_argument("--device", choices=DEVICES, required=True)
    arm.add_argument("--out", type=Path, required=True)
    arm.add_argument("--increments", type=Path, default=None)

    load = sub.add_parser(
        "load", help="internal: the ensemble-shaped GPU occupancy stand-in")
    load.add_argument("--iters", type=int, default=200)
    load.add_argument("--mib", type=float, default=512.0)

    run = sub.add_parser("run", help="run the A/B and write the receipt")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", action="append", dest="devices",
                     choices=DEVICES, default=None,
                     help="repeatable; default host and cuda")
    run.add_argument("--interleave", action="append", dest="interleavings",
                     choices=INTERLEAVINGS, default=None,
                     help="repeatable; default alone only")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--work", type=Path, default=None,
                     help="scratch for per-arm records and increments "
                          "(default: <out>.work)")
    run.add_argument("--load-iters", type=int, default=200)
    run.add_argument("--load-mib", type=float, default=512.0)
    run.add_argument(
        "--load-command", default=None,
        help="a SELF-TERMINATING command standing in for the ensemble "
             "integration.  Give the real member forecast here when a "
             "prepared case is available; the receipt records which ran.  "
             "Nothing in this harness kills a process it started, so a "
             "command that does not exit will hang the run")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "make-bundle":
        manifest = make_bundle(
            args.out, nz=args.nz, ny=args.ny, nx=args.nx,
            members=args.members, radars=args.radars, dx_m=args.dx_m,
            top_m=args.top_m, margin=args.margin,
            horizontal_loc_m=args.horizontal_loc_m,
            vertical_loc_m=args.vertical_loc_m, obs_err_ms=args.obs_err_ms,
            thin_cells=args.thin_cells, rtps_alpha=args.rtps_alpha,
            memory_budget_mib=args.memory_budget_mib, seed=args.seed,
            ref_lat=args.ref_lat, ref_lon=args.ref_lon)
        print(json.dumps({"bundle": str(args.out),
                          "kind": manifest["kind"],
                          "shape": manifest["shape"]}, indent=2))
        return 0

    if args.command == "arm":
        record = run_arm(args.bundle, args.device, out_json=args.out,
                         out_npz=args.increments)
        print(json.dumps(record["seconds"], indent=2))
        return 0

    if args.command == "load":
        print(json.dumps(run_load(args.iters, args.mib)))
        return 0

    devices = args.devices or list(DEVICES)
    interleavings = args.interleavings or ["alone"]
    work = args.work or Path(str(args.out) + ".work")
    receipt = run_ab(args.bundle, args.out, devices=devices,
                     interleavings=interleavings, repeat=args.repeat,
                     load_iters=args.load_iters, load_mib=args.load_mib,
                     load_command=args.load_command, work=work)
    print(json.dumps({"verdict": receipt["verdict"],
                      "bundle_kind": receipt["bundle"]["kind"],
                      "timings": receipt["timings"],
                      "speedup_host_over_cuda":
                          receipt["speedup_host_over_cuda"],
                      "findings": receipt["findings"]}, indent=2))
    # A void or disagreeing A/B is a failed measurement, and the exit code
    # says so: a queued run whose verdict nobody reads must not look green.
    return 0 if receipt["verdict"] == "AGREE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
