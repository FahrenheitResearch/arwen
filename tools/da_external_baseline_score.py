"""Score an external gridded forecast on the nowcast's own frames and metric.

``tools/da_sweep_score.py`` grades one of *our* cycled runs: its members and
its control already sit on the verification grid, so the only question is the
metric.  Grading somebody else's model -- an operational deterministic run on
its own projection -- adds a second question, and it is the one that decides
whether the comparison means anything: **how does the external field get onto
the verification grid, and how much does that map smooth it?**

FSS at a fixed threshold is a *binary-area* statistic.  Anything that pulls
peaks down pulls area across the threshold with it, so a regrid that smooths
is a regrid that scores.  This module therefore refuses to pick an
interpolation for you and hide the choice:

* it scores every method it is given (``--method`` is repeatable) and writes
  them all into the receipt, so the sensitivity to the map is visible rather
  than asserted;
* the default primary is ``nearest``, the only method with *no* amplitude
  mixing at all -- each verification cell takes one donor value unchanged, so
  the external model's reflectivity distribution is preserved exactly and the
  regrid cannot manufacture or destroy threshold-crossing area;
* ``--roundtrip-control`` measures the smoothing directly instead of arguing
  about it: our own ensemble mean is pushed onto the source grid and pulled
  back with the same operator, then re-scored.  That is *twice* the smoothing
  the external field receives one-way, so the FSS change it produces is a
  hard upper bound on the regrid's contribution to any margin.

The metric itself is not re-implemented.  Constants, the FSS and the structure
diagnostics all come from :mod:`tools.da_sweep_score`, which in turn follows
the renderer, so a number produced here lands on the same axis as the
published gallery numbers or the run refuses to proceed.

The structure block matters more here than anywhere else, because the
comparison it appears in is the one that invites the wrong reading.  An
external deterministic model is a single field; our ensemble mean is an
average of ten, and averaging is a spatial filter.  Comparing the mean's
object count against a deterministic model's object count is therefore not a
comparison between two models -- it is a measurement of the averaging.  Every
frame here carries the members' own structure alongside the mean's, and the
receipt says which is which.

The target grid is not typed in either.  It is *reconstructed* from the
verification file's own projection attributes and then checked against the
``XLAT``/``XLONG`` those files carry; a reconstruction that misses the stored
coordinates by more than ``--grid-tolerance-m`` aborts, because a silently
wrong target projection would corrupt every number downstream while looking
perfectly healthy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from gpuwm.static.lambert import LambertGrid
from gpuwm.verify.field_metrics import fss_distance
from tools.da_sweep_score import (ENSEMBLE_MEAN_STRUCTURE_WARNING,
                                  load_composite, member_names,
                                  metric_constants, structure_block,
                                  structure_means)

#: Methods this module knows how to apply, in ascending order of mixing.
METHODS = ("nearest", "bilinear", "parabolic")

#: Frame files are named ``*valid<YYYYMMDDTHHMMSS>Z*.npy``; the valid time is
#: read out of the name so a directory can hold several cycles at once.
VALID_IN_NAME = re.compile(r"valid(\d{8}T\d{6})Z")

_R_EARTH_DEG_M = np.pi * 6_371_000.0 / 180.0


# --------------------------------------------------------------------------
# target grid: reconstructed, then proved against the file's own coordinates
# --------------------------------------------------------------------------

def target_grid_from_verification(path: Path) -> tuple[LambertGrid, dict]:
    """Rebuild the verification grid from a radar-grid file's attributes.

    Returns the grid and a receipt recording the maximum separation between
    the reconstruction and the ``XLAT``/``XLONG`` stored in the file.  The
    caller decides whether that separation is acceptable.
    """

    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        proj = int(ds.getncattr("MAP_PROJ"))
        if proj != 1:
            raise SystemExit(
                f"{path.name}: MAP_PROJ {proj} is not Lambert conformal; "
                "this module only reconstructs Lambert targets")
        lat = np.asarray(ds["XLAT"][:], float)
        lon = np.asarray(ds["XLONG"][:], float)
        attrs = {k: ds.getncattr(k) for k in
                 ("TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
                  "DX", "DY")}

    ny, nx = lat.shape
    grid = LambertGrid(
        ref_lat=float(attrs["CEN_LAT"]), ref_lon=float(attrs["CEN_LON"]),
        truelat1=float(attrs["TRUELAT1"]), truelat2=float(attrs["TRUELAT2"]),
        stand_lon=float(attrs["STAND_LON"]),
        dx=float(attrs["DX"]), dy=float(attrs["DY"]),
        e_we=nx + 1, e_sn=ny + 1)

    i = np.arange(1, nx + 1, dtype=np.float64)
    j = np.arange(1, ny + 1, dtype=np.float64)
    xx, yy = np.meshgrid(i, j)
    rebuilt_lat, rebuilt_lon = grid.ij_to_latlon(xx, yy)

    # Great-circle separation is overkill at these distances; the local
    # tangent-plane bound is exact enough to catch a wrong projection by
    # many orders of magnitude.
    d_north = (rebuilt_lat - lat) * _R_EARTH_DEG_M
    d_east = ((rebuilt_lon - lon) * _R_EARTH_DEG_M
              * np.cos(np.radians(lat)))
    separation = np.hypot(d_north, d_east)

    receipt = {
        "source_file": path.name,
        "attributes": {k: float(v) for k, v in attrs.items()},
        "shape": [int(ny), int(nx)],
        "max_separation_from_stored_coordinates_m":
            round(float(separation.max()), 4),
        "mean_separation_from_stored_coordinates_m":
            round(float(separation.mean()), 4),
        "note": "stored XLAT/XLONG are float32; a residual of order 0.5 m at "
                "these latitudes is the storage precision, not a projection "
                "disagreement",
    }
    return grid, (rebuilt_lat, rebuilt_lon), receipt


# --------------------------------------------------------------------------
# the map, and the mixing it does
# --------------------------------------------------------------------------

def source_coordinates(source_grid, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Zero-based fractional source-array coordinates for target points.

    ``LambertGrid`` coordinates are one-based mass-point indices, and the
    source arrays are stored with element ``[0, 0]`` at one-based ``(1, 1)``,
    so the conversion is a shift of exactly one.  This mirrors
    ``gpuwm.ingest.hrrr._projected_index_geometry``.
    """

    x, y = source_grid.latlon_to_ij(lat, lon)
    return (np.asarray(x, np.float64) - 1.0,
            np.asarray(y, np.float64) - 1.0)


def _clip_index(index: np.ndarray, size: int) -> np.ndarray:
    return np.clip(index, 0, size - 1)


def regrid(field: np.ndarray, x: np.ndarray, y: np.ndarray, *,
           method: str) -> np.ndarray:
    """Sample ``field`` at fractional coordinates ``(x, y)``.

    ``nearest`` mixes nothing.  ``bilinear`` mixes four donors.  ``parabolic``
    is WPS's overlapping-parabolic stencil over sixteen donors, evaluated with
    the repo's own :func:`gpuwm.ingest.hrrr._wps_oned_cpu` so this module owns
    no interpolation mathematics of its own.
    """

    if field.ndim != 2:
        raise ValueError("source field must be 2-D")
    ny, nx = field.shape
    if method == "nearest":
        ix = _clip_index(np.floor(x + 0.5).astype(np.int64), nx)
        iy = _clip_index(np.floor(y + 0.5).astype(np.int64), ny)
        return np.asarray(field[iy, ix], dtype=np.float64)

    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy
    if method == "bilinear":
        i0 = _clip_index(ix, nx)
        i1 = _clip_index(ix + 1, nx)
        j0 = _clip_index(iy, ny)
        j1 = _clip_index(iy + 1, ny)
        lower = (1.0 - fx) * field[j0, i0] + fx * field[j0, i1]
        upper = (1.0 - fx) * field[j1, i0] + fx * field[j1, i1]
        return np.asarray((1.0 - fy) * lower + fy * upper, dtype=np.float64)
    if method != "parabolic":
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    from gpuwm.ingest.hrrr import _wps_oned_cpu

    fx32 = np.asarray(fx, np.float32)
    fy32 = np.asarray(fy, np.float32)
    xs = [_clip_index(ix + offset, nx) for offset in (-1, 0, 1, 2)]
    ys = [_clip_index(iy + offset, ny) for offset in (-1, 0, 1, 2)]
    # WPS's oned has a zero-valued special case; the ingest path substitutes
    # a tiny sentinel so a legitimate zero does not take the wrong branch,
    # and restores it afterwards.  Mirrored here for the same reason.
    tiny = np.float32(1.0e-20)
    rows = []
    for jy in ys:
        values = [np.where(field[jy, jx].astype(np.float32) == np.float32(0.0),
                           tiny, field[jy, jx].astype(np.float32))
                  for jx in xs]
        rows.append(_wps_oned_cpu(fx32, *values))
    out = _wps_oned_cpu(fy32, *rows)
    return np.asarray(np.where(out == tiny, np.float32(0.0), out),
                      dtype=np.float64)


def mixing_report(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                  threshold: float) -> dict:
    """What the map does to amplitude, method by method, on one frame.

    ``nearest`` is the reference: its statistics *are* the source field's
    statistics restricted to the target footprint, because every value is a
    donor value carried across unchanged.  The other methods are reported as
    departures from it.
    """

    base = regrid(field, x, y, method="nearest")
    out = {"footprint_cells": int(base.size),
           "nearest": _amplitude(base, threshold)}
    for method in ("bilinear", "parabolic"):
        mapped = regrid(field, x, y, method=method)
        stats = _amplitude(mapped, threshold)
        stats["max_shift_dbz_vs_nearest"] = round(
            float(np.abs(mapped - base).max()), 4)
        stats["rms_shift_dbz_vs_nearest"] = round(
            float(np.sqrt(np.mean((mapped - base) ** 2))), 4)
        stats["threshold_cells_delta_vs_nearest"] = int(
            (mapped >= threshold).sum() - (base >= threshold).sum())
        out[method] = stats
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    out["fractional_offset"] = {
        "mean_abs_distance_from_donor_cells":
            round(float(np.mean(np.hypot(np.minimum(fx, 1 - fx),
                                         np.minimum(fy, 1 - fy)))), 4),
        "note": "0 would mean the two grids coincide and every method is the "
                "identity; 0.5 in both axes is the worst case for mixing",
    }
    return out


def _amplitude(field: np.ndarray, threshold: float) -> dict:
    return {
        "max_dbz": round(float(field.max()), 4),
        "p999_dbz": round(float(np.percentile(field, 99.9)), 4),
        "p99_dbz": round(float(np.percentile(field, 99.0)), 4),
        "cells_ge_threshold": int((field >= threshold).sum()),
        "cells_ge_35dbz": int((field >= 35.0).sum()),
    }


# --------------------------------------------------------------------------
# observations, and the scoring itself
# --------------------------------------------------------------------------

def observation_frame(path: Path, const: dict) -> dict:
    """The verifier's own truth construction, unchanged."""

    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        z = np.asarray(ds["z_obs"][:], float)
        zmask = np.asarray(ds["z_mask"][:]).astype(bool)
        valid = ds.getncattr("valid_time")
    echo2d = zmask.any(axis=0)
    comp = np.where(zmask, z, -np.inf).max(axis=0)
    comp = np.where(np.isfinite(comp), comp, const["MISSING_OBS_FILL_DBZ"])
    return {"valid_time": valid, "composite": comp, "echo2d": echo2d,
            "cols_gt35": int(((z * zmask).max(axis=0) >= 35.0).sum())}


def score_field(field: np.ndarray, obs: dict, const: dict,
                half_width: int) -> dict:
    return {
        "cols_gt35_in_echo": int((field >= 35.0)[obs["echo2d"]].sum()),
        "fss": round(1.0 - fss_distance(
            field, obs["composite"], threshold=const["FSS_THRESHOLD_DBZ"],
            half_width=half_width), 4),
    }


def frame_valid_key(path: Path) -> str:
    match = VALID_IN_NAME.search(path.name)
    if not match:
        raise SystemExit(f"{path.name}: no valid<YYYYMMDDTHHMMSS>Z in name")
    stamp = match.group(1)
    return (f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
            f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_external_baseline_score",
        description=__doc__.splitlines()[0])
    parser.add_argument("--external-dir", type=Path, required=True,
                        help="directory of source-grid .npy frames named "
                             "*valid<YYYYMMDDTHHMMSS>Z*.npy")
    parser.add_argument("--external-glob", required=True,
                        help="pattern selecting one cycle's frames")
    parser.add_argument("--valid-times", required=True,
                        help="comma-separated nominal valid times "
                             "(YYYY-MM-DDTHH:MM:SSZ) of the scored frames, in "
                             "the order the observation files sort.  The "
                             "external frame for each one is selected by "
                             "name, so the pairing is asserted in the receipt "
                             "rather than inferred from sort order")
    parser.add_argument("--external-label", required=True,
                        help="what the external model is, carried into the "
                             "receipt and the figure")
    parser.add_argument("--source-grid", default="hrrr",
                        choices=("hrrr",),
                        help="projection of the external frames")
    parser.add_argument("--composites", type=Path,
                        help="our run's leg composites.  Omit to score the "
                             "external model alone -- the mode a lead-matched "
                             "error-growth curve needs, because there the "
                             "external frames verify at valid times our free "
                             "forecast never covers and pairing them with our "
                             "legs would record a false correspondence")
    parser.add_argument("--obs-dir", type=Path, required=True)
    parser.add_argument("--obs-glob", default="*verify*.nc")
    parser.add_argument("--first-free-leg", type=int, required=True)
    parser.add_argument("--dx-km", type=float, required=True)
    parser.add_argument("--method", action="append", choices=METHODS,
                        help="repeatable; default is every method")
    parser.add_argument("--primary-method", default="nearest",
                        choices=METHODS,
                        help="the method the headline number comes from "
                             "(default nearest: no amplitude mixing)")
    parser.add_argument("--grid-tolerance-m", type=float, default=5.0,
                        help="abort if the reconstructed target grid misses "
                             "the stored coordinates by more than this")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also sweep neighborhood and threshold.  A "
                             "margin that survives only at the one box size "
                             "and the one threshold the gallery happens to "
                             "use is a margin found by metric shopping, and "
                             "the receipt should show which it is")
    parser.add_argument("--roundtrip-control", action="store_true",
                        help="also push our ensemble mean onto the source "
                             "grid and pull it back, to bound the regrid's "
                             "own contribution to the score")
    parser.add_argument("--single-member",
                        help="member name for the deterministic-vs-"
                             "deterministic line (default: the first member)")
    parser.add_argument("--no-structure", action="store_true",
                        help="skip the object/spectrum/distribution block.  "
                             "It is on by default: a comparison that reports "
                             "only FSS cannot say whether the higher-scoring "
                             "field is the more realistic one")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-regridded", type=Path,
                        help="write the primary-method regridded frames here "
                             "for the renderer")
    args = parser.parse_args(argv)

    methods = tuple(dict.fromkeys(args.method or METHODS))
    if args.primary_method not in methods:
        methods = (args.primary_method,) + methods

    const, const_source = metric_constants()
    half_width = max(1, round(const["FSS_BOX_KM"] / 2.0 / args.dx_km))
    threshold = const["FSS_THRESHOLD_DBZ"]

    obs_files = sorted(args.obs_dir.glob(args.obs_glob))
    if not obs_files:
        raise SystemExit(f"no observation files matched {args.obs_glob}")
    wanted = [t.strip() for t in args.valid_times.split(",") if t.strip()]
    if len(wanted) != len(obs_files):
        raise SystemExit(
            f"{len(wanted)} valid times against {len(obs_files)} observation "
            "files; a frame-matched score needs one of each")
    available: dict[str, list[Path]] = {}
    for path in sorted(args.external_dir.glob(args.external_glob)):
        available.setdefault(frame_valid_key(path), []).append(path)
    external_files = []
    for stamp in wanted:
        matches = available.get(stamp, [])
        if len(matches) != 1:
            raise SystemExit(
                f"{len(matches)} external frames valid {stamp} matched "
                f"{args.external_glob}; exactly one is required")
        external_files.append(matches[0])

    grid, (lat, lon), grid_receipt = target_grid_from_verification(
        obs_files[0])
    if grid_receipt["max_separation_from_stored_coordinates_m"] > \
            args.grid_tolerance_m:
        raise SystemExit(
            "reconstructed target grid misses the stored coordinates by "
            f"{grid_receipt['max_separation_from_stored_coordinates_m']} m "
            f"(> {args.grid_tolerance_m} m); refusing to score on a grid "
            "that is not provably the verification grid")

    if args.source_grid == "hrrr":
        from gpuwm.ingest.hrrr import hrrr_source_grid
        source_grid = hrrr_source_grid()
    x, y = source_coordinates(source_grid, lat, lon)

    if args.composites is None:
        if args.roundtrip_control:
            raise SystemExit("--roundtrip-control regrids OUR field and so "
                             "needs --composites")
        members, single = [], None
    else:
        members = member_names(args.composites, args.first_free_leg)
        if not members:
            raise SystemExit("no member composites found")
        single = args.single_member or members[0]
        if single not in members:
            raise SystemExit(f"member {single!r} not among {members}")

    saved: dict[str, str] = {}
    if args.save_regridded:
        args.save_regridded.mkdir(parents=True, exist_ok=True)

    frames = []
    for offset, (obs_path, ext_path, nominal) in enumerate(
            zip(obs_files, external_files, wanted)):
        leg = args.first_free_leg + offset
        obs = observation_frame(obs_path, const)
        source = np.load(ext_path)
        if source.ndim != 2:
            raise SystemExit(f"{ext_path.name}: expected a 2-D source field")
        if (x.min() < 1 or y.min() < 1
                or x.max() > source.shape[1] - 3
                or y.max() > source.shape[0] - 3):
            raise SystemExit(
                f"{ext_path.name}: target footprint reaches the source array "
                "edge; the sixteen-point stencil has no halo there")

        ensemble = None
        member_fields: dict[str, np.ndarray] = {}
        if members:
            member_fields = {name: load_composite(args.composites, leg, name)
                             for name in members}
            ensemble = np.mean(list(member_fields.values()), axis=0)
            if ensemble.shape != obs["composite"].shape:
                raise SystemExit(f"leg {leg}: composite and observation grids "
                                 "differ; these are not the same case")

        external = {}
        primary_field = None
        for method in methods:
            mapped = regrid(source, x, y, method=method)
            external[method] = score_field(mapped, obs, const, half_width)
            if method == args.primary_method:
                primary_field = mapped
            if method == args.primary_method and args.save_regridded:
                out = (args.save_regridded
                       / f"{args.external_label}_leg{leg:02d}_"
                         f"{args.primary_method}.npy")
                np.save(out, mapped)
                saved[obs["valid_time"]] = str(out)

        row = {
            "leg": leg,
            "nominal_valid_time": nominal,
            "observation_file": obs_path.name,
            "external_file": ext_path.name,
            "external_valid_time": frame_valid_key(ext_path),
            "obs_valid_time": obs["valid_time"],
            "obs_cols_gt35": obs["cols_gt35"],
            "external": external,
            "regrid_mixing": mixing_report(source, x, y, threshold),
        }
        control = None
        if members:
            control = load_composite(args.composites, leg, "control")
            row["ensemble_mean"] = score_field(ensemble, obs, const,
                                               half_width)
            row[f"member_{single}"] = score_field(
                member_fields[single], obs, const, half_width)
            row["control"] = score_field(control, obs, const, half_width)
            row["per_member"] = {
                name: score_field(field, obs, const, half_width)["fss"]
                for name, field in member_fields.items()}

        if not args.no_structure:
            # The external field is the one that HAS to be in here: without
            # it the receipt can say our mean is smooth but not whether the
            # model it is being compared with is any less so.
            extra = {"external": primary_field}
            if control is not None:
                extra["control"] = control
            row["structure"] = structure_block(
                observed=obs["composite"], dx_km=args.dx_km,
                ensemble_mean=ensemble, members=member_fields or None,
                extra=extra)

        if args.sensitivity:
            fields = {"external": regrid(source, x, y,
                                         method=args.primary_method)}
            if members:
                fields["ensemble_mean"] = ensemble
                fields[f"member_{single}"] = load_composite(
                    args.composites, leg, single)
                fields["control"] = load_composite(args.composites, leg,
                                                   "control")
            row["sensitivity"] = _sensitivity(fields, obs, args.dx_km)

        if args.roundtrip_control:
            row["roundtrip_control"] = _roundtrip(
                ensemble, grid, source_grid, source.shape, x, y,
                obs, const, half_width, args.primary_method)

        frames.append(row)

    payload = {
        "schema": "gpuwm-da.external-baseline-score.v1",
        "external_label": args.external_label,
        "dx_km": args.dx_km,
        "constants_source": const_source,
        "constants": const,
        "metric_statement": {
            "fss": "Roberts & Lean (2008) fractions skill score, computed by "
                   "gpuwm.verify.field_metrics.fss_distance",
            "both_fields_smoothed": True,
            "both_fields_smoothed_note":
                "the TRUTH field is smoothed by the same neighborhood as the "
                "forecast (the generous formulation); a binary-truth FSS on "
                "the same data would be substantially lower and the two are "
                "not comparable",
            "neighborhood": "square box, SIDE length "
                            f"{(2 * half_width + 1) * args.dx_km:g} km "
                            f"({2 * half_width + 1} cells across); NOT a "
                            "radius",
            "threshold_dbz": threshold,
            "ensemble_mean_is_not_deterministic":
                "the ensemble-mean column is an average over members and is "
                "therefore spatially smoother than any single field; an "
                "external deterministic model receives no such averaging, so "
                "the like-for-like line is the single-member column",
            "structure": "object counts (35 dBZ), radial power spectrum and "
                         "in-echo dBZ quantiles, all from "
                         "tools.da_sweep_score; they are diagnostics, not a "
                         "score, and nothing here ranks on them",
            "structure_on_the_mean_is_misleading":
                ENSEMBLE_MEAN_STRUCTURE_WARNING,
        },
        "target_grid": grid_receipt,
        "source_grid": args.source_grid,
        "regrid": {
            "methods_scored": list(methods),
            "primary_method": args.primary_method,
            "primary_method_rationale":
                "nearest-donor sampling mixes no amplitudes, so the external "
                "model's reflectivity distribution reaches the metric intact "
                "and the regrid cannot move threshold-crossing area",
            "implementation":
                "target lat/lon -> source projection coordinates via the "
                "repo's own LambertGrid.latlon_to_ij; the parabolic stencil "
                "is gpuwm.ingest.hrrr._wps_oned_cpu, unmodified",
        },
        "members": members,
        "single_member": single,
        "composites": str(args.composites),
        "obs_dir": str(args.obs_dir),
        "external_dir": str(args.external_dir),
        "frames": frames,
        "means": _means(frames, methods, single),
    }
    if not args.no_structure:
        payload["structure_means"] = structure_means(
            frames, extra_labels=("external",) + (("control",) if members
                                                  else ()))
    if saved:
        payload["regridded_frames"] = saved

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    means = payload["means"]
    if members:
        print(f"{'valid':>22}  {'obs>=35':>7}  {'ens':>6}  {'mem':>6}  "
              f"{'ext':>6}  {'ctrl':>6}")
        for row in frames:
            print(f"{row['nominal_valid_time']:>22}  "
                  f"{row['obs_cols_gt35']:>7}  "
                  f"{row['ensemble_mean']['fss']:>6.4f}  "
                  f"{row[f'member_{single}']['fss']:>6.4f}  "
                  f"{row['external'][args.primary_method]['fss']:>6.4f}  "
                  f"{row['control']['fss']:>6.4f}")
        print(f"{'MEAN':>22}  {'':>7}  {means['ensemble_mean_fss']:>6.4f}  "
              f"{means['single_member_fss']:>6.4f}  "
              f"{means['external_fss'][args.primary_method]:>6.4f}  "
              f"{means['control_fss']:>6.4f}")
    else:
        print(f"{'valid':>22}  {'obs>=35':>7}  {'ext':>6}   (external only)")
        for row in frames:
            print(f"{row['nominal_valid_time']:>22}  "
                  f"{row['obs_cols_gt35']:>7}  "
                  f"{row['external'][args.primary_method]['fss']:>6.4f}")
        print(f"{'MEAN':>22}  {'':>7}  "
              f"{means['external_fss'][args.primary_method]:>6.4f}")
    print(f"[metric constants from {const_source}; "
          f"{2 * half_width + 1}-cell box; primary regrid "
          f"{args.primary_method}]")
    if not args.no_structure:
        _print_structure(payload["structure_means"], args.external_label)
    return 0


def _print_structure(means: dict, external_label: str) -> None:
    """The structure summary, with the mean and the members kept apart."""

    def row(name: str, block: dict | None) -> None:
        if not block:
            return
        print(f"{name:>28}  "
              f"objects {_cell(block.get('objects.count'), '5.1f')}  "
              f"nn-sep {_cell(block.get('objects.mean_nearest_neighbor_km'), '6.1f')} km  "
              f"2-4dx power {_cell(block.get('spectrum.power_ratio_2_4dx'), '6.3f')}  "
              f"P99 {_cell(block.get('distribution.p99_dbz'), '5.1f')} dBZ")

    print("\nSTRUCTURE (diagnostic, not a score; frame means)")
    row("OBSERVED", means.get("observed"))
    row("ours: ensemble mean", means.get("ensemble_mean"))
    members = means.get("members") or {}
    if members:
        for key, label in (("objects.count", "ours: members, objects"),
                           ("spectrum.power_ratio_2_4dx",
                            "ours: members, 2-4dx power")):
            spread = members.get(key)
            if spread:
                print(f"{label:>28}  min {spread['min']}  "
                      f"median {spread['median']}  max {spread['max']}")
    row(external_label, means.get("external"))
    row("control", means.get("control"))
    print("  " + ENSEMBLE_MEAN_STRUCTURE_WARNING)


def _cell(value, fmt: str) -> str:
    return format(value, fmt) if value is not None else "    --"


def _roundtrip(field, target_grid, source_grid, source_shape, x, y,
               obs, const, half_width, method) -> dict:
    """Our own field, pushed to the source grid and pulled back.

    The external model crosses the projection boundary once.  This crosses it
    twice, so whatever it costs is strictly more than the external model
    pays -- which makes it an upper bound rather than an estimate.
    """

    ny, nx = source_shape
    # Source cell centres, in target coordinates.
    si = np.arange(nx, dtype=np.float64) + 1.0
    sj = np.arange(ny, dtype=np.float64) + 1.0
    # Only the neighbourhood of the target footprint is needed, and building
    # the whole source grid's coordinates is wasteful at CONUS size.
    i0, i1 = int(np.floor(x.min())) - 4, int(np.ceil(x.max())) + 5
    j0, j1 = int(np.floor(y.min())) - 4, int(np.ceil(y.max())) + 5
    ii, jj = np.meshgrid(si[i0:i1], sj[j0:j1])
    slat, slon = source_grid.ij_to_latlon(ii, jj)
    tx, ty = target_grid.latlon_to_ij(slat, slon)
    tx -= 1.0
    ty -= 1.0
    inside = ((tx >= 1) & (ty >= 1)
              & (tx <= field.shape[1] - 3) & (ty <= field.shape[0] - 3))
    pushed = np.full(ii.shape, const["MISSING_OBS_FILL_DBZ"], dtype=np.float64)
    if inside.any():
        pushed[inside] = regrid(field, tx[inside], ty[inside], method=method)
    pulled = regrid(pushed, x - i0, y - j0, method=method)
    scored = score_field(pulled, obs, const, half_width)
    direct = score_field(field, obs, const, half_width)
    return {
        "method": method,
        "note": "our ensemble mean after a source-grid round trip; two "
                "regrids, versus the one the external field receives",
        "fss_direct": direct["fss"],
        "fss_after_roundtrip": scored["fss"],
        "fss_change": round(scored["fss"] - direct["fss"], 4),
        "cols_gt35_direct": direct["cols_gt35_in_echo"],
        "cols_gt35_after_roundtrip": scored["cols_gt35_in_echo"],
    }


#: Boxes and thresholds the sensitivity sweep visits.  The 9-cell box and
#: the 30 dBZ threshold are the gallery's; the rest are there to show whether
#: a margin depends on them.
SENSITIVITY_HALF_WIDTHS = (1, 2, 4, 8, 16)
SENSITIVITY_THRESHOLDS = (25.0, 30.0, 35.0, 40.0, 45.0)


def _sensitivity(fields: dict, obs: dict, dx_km: float) -> dict:
    out = {}
    for half_width in SENSITIVITY_HALF_WIDTHS:
        box = f"{(2 * half_width + 1) * dx_km:g}km"
        for threshold in SENSITIVITY_THRESHOLDS:
            key = f"{box}_{threshold:g}dBZ"
            out[key] = {
                name: round(1.0 - fss_distance(
                    field, obs["composite"], threshold=threshold,
                    half_width=half_width), 4)
                for name, field in fields.items()}
    return out


def _means(frames, methods, single) -> dict:
    def mean(values):
        return round(float(np.mean(values)), 4)

    out = {
        "external_fss": {m: mean([f["external"][m]["fss"] for f in frames])
                         for m in methods},
    }
    if single is not None:
        out["ensemble_mean_fss"] = mean([f["ensemble_mean"]["fss"]
                                         for f in frames])
        out["single_member_fss"] = mean([f[f"member_{single}"]["fss"]
                                         for f in frames])
        out["control_fss"] = mean([f["control"]["fss"] for f in frames])
        per_member = [[row["per_member"][name] for row in frames]
                      for name in frames[0]["per_member"]]
        member_means = [float(np.mean(series)) for series in per_member]
        out["member_fss_mean_over_members"] = round(
            float(np.mean(member_means)), 4)
        out["member_fss_worst_member"] = round(min(member_means), 4)
        out["member_fss_best_member"] = round(max(member_means), 4)
    if "roundtrip_control" in frames[0]:
        out["roundtrip_fss_change"] = mean(
            [f["roundtrip_control"]["fss_change"] for f in frames])
    if "sensitivity" in frames[0]:
        out["sensitivity"] = {
            key: {name: mean([f["sensitivity"][key][name] for f in frames])
                  for name in frames[0]["sensitivity"][key]}
            for key in frames[0]["sensitivity"]}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
