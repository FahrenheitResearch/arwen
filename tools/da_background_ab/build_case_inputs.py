"""Derive the HRRR arm's case inputs FROM the GFS arm's own authority.

The A/B this belongs to asks one question -- does a convection-allowing
background improve WaH's skill -- and the answer is worth nothing unless
the two arms differ in the background and in nothing else.  The GFS arm
reuses a prepared case that already exists.  The HRRR arm has to build
one, and every number that describes its grid, its vertical ladder and
its physics is an opportunity to differ by accident.

So none of those numbers is typed here.  They are READ from the GFS
case's own ``experiment.toml`` and ``namelist.wps`` and re-emitted in
the two documents the native HRRR preparation consumes:

* a ``gpuwm-hrrr.target-domain.v1`` specification
  (:class:`gpuwm.ingest.hrrr_target.HrrrTargetDomain`), which fixes the
  horizontal grid, the projection, the time step and the boundary zones;
* a WRF ``namelist.input``, which is where the native HRRR route reads
  its vertical grid from
  (:func:`gpuwm.vertical_contract.explicit_vertical_from_wrf_namelist`)
  and where it validates the physics profile.

Both are then proved, not asserted: :func:`verify_equivalence` renders
the experiment tables the HRRR route would build from them, runs them
through the ordinary ``build_experiment`` front door, and compares the
resulting prepared-domain identity against the one baked into the GFS
prepared cache with the SAME comparator the front door uses
(:func:`gpuwm.ingest.prepared_cache.compare_prepared_domain_config`).
A field that differs is printed with both values.  Nothing is
"close enough": the prepared cache compares this document by strict
equality, so a difference here is a refusal at run time, hours later,
after the fetch has been paid for.

The vertical grid is checked separately and deliberately.  ``eta_levels``
and ``p_top`` are NOT part of the prepared-domain identity, so two cases
can pass the domain comparison while sitting on different ladders -- and
a skill comparison between two different vertical grids measures the
ladder as much as the background.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm.experiment import load_experiment                    # noqa: E402
from gpuwm.ingest.hrrr_target import HrrrTargetDomain           # noqa: E402
from gpuwm.namelist_import import parse_namelist                # noqa: E402

#: Schema of the equivalence receipt this module writes.
EQUIVALENCE_SCHEMA = "gpuwm-da.background-ab-case-equivalence.v1"

#: Largest tolerated disagreement between the two arms' eta ladders.
#: Zero, expressed as a float comparison, because the ladder is COPIED:
#: any nonzero difference means the copy did not happen.
ETA_TOLERANCE = 0.0


def target_from_gfs_authority(
        experiment_toml: Path, wps_namelist: Path, *, name: str
) -> HrrrTargetDomain:
    """The HRRR target domain that reproduces the GFS case's grid.

    Every field comes from the GFS authority.  ``e_we``/``e_sn`` in the
    WPS namelist are cross-checked against ``nx``/``ny`` in the
    experiment config rather than one being trusted: they are two
    independent statements of the same grid, and the case is only
    coherent when they agree.
    """

    experiment = load_experiment(experiment_toml)
    run = experiment.root.run
    projection = experiment.projection
    wps = parse_namelist(wps_namelist)
    geogrid = wps.get("geogrid", {})

    def _one(values, label):
        if not values:
            raise ValueError(f"{wps_namelist}: &geogrid has no {label}")
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(
                f"{wps_namelist}: {label} is not uniform across domains "
                f"({values}); this A/B is single-domain")
        return first

    e_we = int(_one(geogrid.get("e_we", []), "e_we"))
    e_sn = int(_one(geogrid.get("e_sn", []), "e_sn"))
    if (e_we - 1, e_sn - 1) != (int(run.nx), int(run.ny)):
        raise ValueError(
            f"grid disagreement between the two authorities: namelist.wps "
            f"says e_we/e_sn {e_we}/{e_sn} (mass {e_we - 1}x{e_sn - 1}) and "
            f"experiment.toml says nx/ny {run.nx}x{run.ny}.  These are the "
            "same case's two descriptions of one grid; refusing to guess "
            "which is meant")

    # The exact rational clock, decomposed the way the target-domain
    # spec (and WRF's registry) spells it: whole seconds + a proper
    # remainder.  ``run.dt`` is the float32-chained IMAGE of the clock;
    # ``dt_exact`` is the rational itself, so a 1.5 km case's 7.5 s
    # arrives as 7 + 1/2 with nothing truncated and nothing rounded.
    from fractions import Fraction

    dt = Fraction(experiment.dt_exact(run.grid_id))
    dt_whole = dt.numerator // dt.denominator
    dt_rem = dt - dt_whole

    return HrrrTargetDomain(
        name=name,
        map_proj="lambert",
        nx=int(run.nx), ny=int(run.ny), nz=int(run.nz),
        dx_m=float(run.dx), dy_m=float(run.dy),
        ref_lat=float(projection.ref_lat), ref_lon=float(projection.ref_lon),
        truelat1=float(projection.truelat1),
        truelat2=float(projection.truelat2),
        stand_lon=float(projection.stand_lon),
        time_step_seconds=int(dt_whole),
        time_step_fract_num=dt_rem.numerator,
        time_step_fract_den=dt_rem.denominator,
        spec_bdy_width=int(experiment.spec_bdy_width),
        spec_zone=int(run.spec_zone), relax_zone=int(run.relax_zone))


def namelist_input_for(target: HrrrTargetDomain, experiment_toml: Path, *,
                       valid_time: datetime, run_seconds: int) -> str:
    """The WRF ``namelist.input`` the native HRRR route reads.

    Rendered by ``tools/write_hrrr_stock_wrf_namelist.render_namelist`` --
    the shipped writer, not a copy of it -- with the vertical grid taken
    from the GFS authority so the two arms cannot land on different
    ladders.
    """

    from tools.write_hrrr_stock_wrf_namelist import render_namelist

    experiment = load_experiment(experiment_toml)
    vertical = experiment.vertical
    return render_namelist(
        target=target, eta=list(vertical.eta_levels), valid_time=valid_time,
        run_seconds=int(run_seconds), p_top=float(vertical.p_top),
        hybrid_opt=int(vertical.hybrid_opt), etac=float(vertical.etac))


def _hrrr_domain_identity(target: HrrrTargetDomain, namelist_input: Path, *,
                          run_seconds: float, start_time: datetime,
                          physics_profile: str,
                          history_interval_seconds: float) -> dict:
    """The prepared-domain identity the HRRR route would bake in.

    Built through the route's own table builder and the ordinary
    ``build_experiment`` front door, so this is the identity that will
    actually be written, not a model of it.
    """

    from gpuwm.experiment import build_experiment
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity
    from gpuwm.vertical_contract import explicit_vertical_from_wrf_namelist
    from tools.hrrr_single_domain_benchmark import _experiment_tables

    vertical = explicit_vertical_from_wrf_namelist(
        namelist_input, expected_nz=target.nz,
        context="native HRRR initializer")
    tables, resolved = _experiment_tables(
        vertical, run_seconds=float(run_seconds), start_time=start_time,
        target=target, physics_profile=physics_profile,
        history_interval_seconds=float(history_interval_seconds))
    experiment = build_experiment(
        tables, f"programmatic:native-HRRR:{resolved.identity_sha256()}")
    return prepared_domain_config_identity(experiment.root), experiment


def verify_equivalence(*, gfs_prepared_cache_header: Path,
                       gfs_experiment_toml: Path,
                       target: HrrrTargetDomain, namelist_input: Path,
                       run_seconds: float, start_time: datetime,
                       physics_profile: str,
                       history_interval_seconds: float) -> dict:
    """Compare the two arms' prepared-domain identities and eta ladders."""

    from gpuwm.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config,
        undelayed_identity_defaults)

    cached = json.loads(
        gfs_prepared_cache_header.read_text(encoding="utf-8"))
    gfs_identity = cached["identity"]["domain_config"]
    hrrr_identity, hrrr_experiment = _hrrr_domain_identity(
        target, namelist_input, run_seconds=run_seconds,
        start_time=start_time, physics_profile=physics_profile,
        history_interval_seconds=history_interval_seconds)

    tolerated, differing = compare_prepared_domain_config(
        effective_prepared_domain_config(gfs_identity),
        effective_prepared_domain_config(hrrr_identity),
        not_in_use=undelayed_identity_defaults(hrrr_experiment))

    def _at(document, path):
        node = document
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return "<absent>"
            node = node[part]
        return node

    gfs_experiment = load_experiment(gfs_experiment_toml)
    gfs_eta = [float(value) for value in gfs_experiment.vertical.eta_levels]
    hrrr_eta = [float(value)
                for value in hrrr_experiment.vertical.eta_levels]
    eta_max_delta = (
        max(abs(a - b) for a, b in zip(gfs_eta, hrrr_eta))
        if len(gfs_eta) == len(hrrr_eta) else math.inf)
    p_top_delta = abs(float(gfs_experiment.vertical.p_top)
                      - float(hrrr_experiment.vertical.p_top))

    return {
        "schema": EQUIVALENCE_SCHEMA,
        "domain_identity": {
            "equal": not differing,
            "tolerated_fields": tolerated,
            "differing_fields": [
                {"path": path,
                 "gfs": _at(gfs_identity, path),
                 "hrrr": _at(hrrr_identity, path)}
                for path in differing],
            "comparator": ("gpuwm.ingest.prepared_cache."
                           "compare_prepared_domain_config, the same "
                           "function the prepared-forecast front door "
                           "uses"),
        },
        "vertical_grid": {
            "equal": (len(gfs_eta) == len(hrrr_eta)
                      and eta_max_delta <= ETA_TOLERANCE
                      and p_top_delta <= ETA_TOLERANCE),
            "levels": len(gfs_eta),
            "eta_max_abs_delta": eta_max_delta,
            "p_top_abs_delta": p_top_delta,
            "why_checked_separately": (
                "eta_levels and p_top are not in the prepared-domain "
                "identity, so two cases can pass the domain comparison on "
                "different ladders; a skill difference across different "
                "ladders is not a background result"),
        },
        "target_domain": target.to_payload(),
        "target_domain_identity_sha256": target.identity_sha256(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.build_case_inputs",
        description=__doc__.splitlines()[0])
    parser.add_argument("--gfs-authority", type=Path, required=True,
                        help="the GFS arm's authority directory "
                             "(experiment.toml + namelist.wps)")
    parser.add_argument("--gfs-prepared-root", type=Path, required=True,
                        help="the GFS arm's prepared root, for the cache "
                             "header whose identity is the thing to match")
    parser.add_argument("--name", required=True,
                        help="target-domain name; the A/B uses one name "
                             "for both HRRR arms because the GRID is the "
                             "same and only the source cycle differs")
    parser.add_argument("--valid-time", required=True,
                        help="model init, YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--run-seconds", type=float, required=True)
    parser.add_argument("--history-interval-seconds", type=float,
                        required=True)
    parser.add_argument("--physics-profile", required=True)
    parser.add_argument("--out-domain-spec", type=Path, required=True)
    parser.add_argument("--out-namelist-input", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    valid_time = datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
    experiment_toml = args.gfs_authority / "experiment.toml"
    wps_namelist = args.gfs_authority / "namelist.wps"

    target = target_from_gfs_authority(
        experiment_toml, wps_namelist, name=args.name)
    args.out_domain_spec.parent.mkdir(parents=True, exist_ok=True)
    args.out_domain_spec.write_text(
        json.dumps(target.to_payload(), indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    args.out_namelist_input.parent.mkdir(parents=True, exist_ok=True)
    args.out_namelist_input.write_text(
        namelist_input_for(target, experiment_toml, valid_time=valid_time,
                           run_seconds=int(args.run_seconds)),
        encoding="utf-8", newline="\n")

    receipt = verify_equivalence(
        gfs_prepared_cache_header=(
            args.gfs_prepared_root / "prepared-cache" / "header.json"),
        gfs_experiment_toml=experiment_toml,
        target=target, namelist_input=args.out_namelist_input,
        run_seconds=args.run_seconds, start_time=valid_time,
        physics_profile=args.physics_profile,
        history_interval_seconds=args.history_interval_seconds)
    receipt["inputs"] = {
        "gfs_authority": str(args.gfs_authority),
        "gfs_prepared_root": str(args.gfs_prepared_root),
        "domain_spec": str(args.out_domain_spec),
        "namelist_input": str(args.out_namelist_input),
        "valid_time": args.valid_time,
        "run_seconds": args.run_seconds,
        "history_interval_seconds": args.history_interval_seconds,
        "physics_profile": args.physics_profile,
    }
    args.out_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.out_receipt.write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    domain_ok = receipt["domain_identity"]["equal"]
    vertical_ok = receipt["vertical_grid"]["equal"]
    print(f"target domain      {target.nx}x{target.ny}x{target.nz} "
          f"dx {target.dx_m:g} m dt {float(target.time_step_exact):g} s "
          f"({target.identity_sha256()[:16]})")
    print(f"domain identity    {'EQUAL' if domain_ok else 'DIFFERS'}"
          + (f" (tolerated: {receipt['domain_identity']['tolerated_fields']})"
             if receipt["domain_identity"]["tolerated_fields"] else ""))
    for entry in receipt["domain_identity"]["differing_fields"]:
        print(f"    {entry['path']}: gfs={entry['gfs']!r} "
              f"hrrr={entry['hrrr']!r}")
    print(f"vertical grid      "
          f"{'EQUAL' if vertical_ok else 'DIFFERS'} "
          f"({receipt['vertical_grid']['levels']} levels, max |d eta| "
          f"{receipt['vertical_grid']['eta_max_abs_delta']:g}, "
          f"|d p_top| {receipt['vertical_grid']['p_top_abs_delta']:g})")
    if not (domain_ok and vertical_ok):
        print("\nREFUSED: the HRRR arm would not be the same experiment as "
              "the GFS arm.  A skill difference between these two cases "
              "would not be attributable to the background.")
        return 1
    print(f"\nreceipt {args.out_receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
