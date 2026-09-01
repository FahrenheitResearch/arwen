"""EXPERIMENTAL: cycling radar DA over a prepared single-domain case.

Forecast legs on the real GPU dycore, a real LETKF analysis between
them, real NEXRAD volumes -- the loop the ensemble engine cannot host
today and that the v1.2 assembly notes call the engine gap: the engine
refuses domain trees and loads only the ``[case_data]`` route, while the
prepared-tree runners are deterministic and hash-bind their own run
bounds.  Until that gap closes this driver is how a prepared case gets
cycled, and it is deliberately written as the reference implementation
the engine work can absorb rather than as a one-off script.

What it does, per leg:

1. wires the model exactly as
   :func:`gpuwm.prepared_single_domain_forecast.run_prepared_forecast`
   does -- same preflight, same prepared-cache restore, same physics
   initialisation, same ``execute_experiment`` -- but per trajectory and
   with the clock placed at the leg boundary;
2. integrates one leg on the GPU, with the state-health validator on;
3. snapshots the serialised prognostic state
   (:data:`gpuwm.state_serialization_contract.STATE_SERIALIZED_ATTRS`)
   to the host, which is what joins one leg to the next;
4. analyses the members against a ``gpuwm-obs.radar-grid.v1`` file
   through :mod:`gpuwm.da.radar_assimilation`, optionally adding
   :mod:`gpuwm.da.hotstart`'s reflectivity nudge, and applies the result
   through :func:`gpuwm.ensemble.increments.apply_increments` with
   ``refresh_diagnostics`` after it -- the shipped appliers, not a
   private write path.

A control trajectory runs beside the members and is never analysed, so
every number the report carries has a no-DA counterpart taken through
the same code.

**Deliberate simplifications, stated rather than buried.**

* Physics driver state (surface fields, accumulators) is re-initialised
  at every leg for EVERY trajectory including the control, because the
  host snapshot carries the serialised atmosphere only.  Member-vs-
  control comparisons are therefore like-for-like; continuity of surface
  accumulators across a leg boundary is NOT claimed, and a leg boundary
  is not a bit-exact restart.  ``gpuwm.io.restart`` is the bit-exact
  path and it is not reachable from a prepared-cache run today (that is
  the same engine gap).
* Members share the deterministic run's lateral boundary conditions --
  the perturbation lane's documented setting, and the reason ensemble
  spread is suppressed near the rim.
* The observation grid is supplied per leg and is bound to the
  observation file by digest inside the adapter; a member's own column
  heights differ from that grid by the member's own perturbation, which
  is representativeness error rather than a binding error.

**Cycling past the end of one process.**  ``--save-ensemble`` writes the
leg boundary -- host snapshots plus the analysis increments the next leg
has still to apply -- through :mod:`tools.da_ensemble_state`, and
``--resume-ensemble`` starts from one.  That is what a continuous
nowcast needs: the next radar volume does not exist when this one is
assimilated, so the ensemble has to survive the wait without being
re-initialised.  The generation is written at the end of the last
OBSERVED leg, so trailing free legs stay a branch off the cycle rather
than becoming it, and the boundary-data horizon of the prepared case is
a refusal rather than a surprise inside the integrator.

Nothing here is on a default route.  EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import shutil
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

try:                                    # python -m tools.da_cycle_prepared
    from tools import da_ensemble_state as ens_state
    from tools import da_solve_ab as ab_bundle
except ImportError:                     # python tools/da_cycle_prepared.py
    import da_ensemble_state as ens_state
    import da_solve_ab as ab_bundle

#: Name of the unassimilated trajectory in every report.
CONTROL = "control"

#: Schema of the report this driver writes.
REPORT_SCHEMA = "gpuwm-da.prepared-cycle-report.v1"

#: The physics-receipt fields that are registry VOCABULARY rather than
#: physics.  A prepared authority written by an older tree names its
#: maturity tier and registry digest in that tree's vocabulary; every
#: selector, component and profile id can still match exactly.
#: ``--tolerate-physics-vocabulary-drift`` allows a mismatch confined to
#: these two fields, records it in the report, and refuses any other
#: difference.  Without the flag the preflight is strict.
PHYSICS_VOCABULARY_FIELDS = ("maturity", "registry_sha256")


def jump_clock(clock, start_seconds: float, dt: float) -> None:
    """Place a freshly built clock at a leg boundary.

    A leg after the first restores a host snapshot into a state the
    prepared cache just rebuilt, so its clock starts at zero while the
    trajectory it carries is already ``start_seconds`` old.  Three things
    have to move together.

    ``ticks`` and ``step_count`` are the integer calendar and are exact:
    the tick lattice is what every alarm evaluates on, so setting seconds
    would be setting a derived quantity.

    ``dtbc_fp32`` is WRF's REAL boundary-tendency accumulator, and it is
    NOT derivable in closed form.  It recurs as
    ``fl32(dtbc + dt)`` once per step and resets at every external-LBC
    seam (``gpuwm.core.clock.DomainClock.prepare_step`` /
    :meth:`mark_force`), so its value carries the accumulated FP32
    rounding of every step since the last seam.  ``steps_since_seam * dt``
    is a different number in the last bits, and the boundary relaxation
    consumes this one.  So it is REPLAYED with the same recurrence rather
    than computed -- at most one boundary interval of iterations, and
    bit-exact against a clock that stepped there.
    """
    steps = int(round(start_seconds / dt))
    clock.ticks = steps * clock.spec.step_ticks
    clock.step_count = steps
    interval = clock.spec.lbc_interval_ticks
    since = steps
    if interval is not None:
        per_seam = interval // clock.spec.step_ticks
        since = steps % per_seam
        # A leg boundary that lands exactly ON a seam has a FULL interval
        # accumulated, not zero.  The reset is top-of-step work
        # (``lbc_reset_due`` -> ``mark_force``), so the integrator applies
        # it to this clock on its own first step; zeroing it here would
        # hand the integrator a state it never produces, and the two only
        # happen to agree because the reset lands next.  Reproducing what
        # the integrator would have HAD is the rule -- a clock this driver
        # places must be indistinguishable from one that stepped there,
        # and ``tests/test_da_cycle_prepared.py`` compares them bit for
        # bit at exactly this instant.
        if since == 0 and steps > 0:
            since = per_seam
    value = np.float32(0.0)
    for _ in range(since):
        value = np.float32(value + clock.spec.dt_fp32)
    clock.dtbc_fp32 = value


def to_host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value))



def _write_composite_wrfout(npz_path, refl_colmax, grid, cfg,
                            elapsed_seconds: float, exp, *, label: str):
    """A real wrfout beside a composite ``.npz``; the path, or ``None``.

    Why a writer and not a Rust ``.npz`` reader: ``.npz`` carries no
    geolocation contract, so a reader for it would be a per-format adapter
    -- exactly what the arbitrary-acceptance rule forbids -- while this
    lane already holds the grid the composite was computed on and a wrfout
    is the container that states it.

    Additive and non-fatal.  The cycle's product is the ``.npz``; a
    failure to write the second copy is REPORTED and the forecast
    continues, because a rendering convenience must not be able to kill a
    DA cycle.
    """

    import datetime

    from gpuwm.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         snapshot_wrfout_path,
                                         write_surface_wrfout)
    from gpuwm.io.wrfout import wrf_global_attrs

    try:
        lat, lon = grid.latlon_mass()
        snapshot = {
            "XLAT": np.asarray(lat, np.float32),
            "XLONG": np.asarray(lon, np.float32),
            "REFL_COMPOSITE": np.asarray(refl_colmax, np.float32),
        }
        start = getattr(exp, "start_time", None)
        if not isinstance(start, datetime.datetime):
            start = datetime.datetime(1970, 1, 1)
        stamp = (start + datetime.timedelta(seconds=float(elapsed_seconds))
                 ).strftime("%Y-%m-%d_%H:%M:%S")
        attrs = wrf_global_attrs(grid, start, dt=float(cfg.dt))
        return write_surface_wrfout(
            snapshot_wrfout_path(npz_path), snapshot, time_str=stamp,
            dx=float(cfg.dx), dy=float(cfg.dy), global_attrs=attrs,
            title=f"gpuwm DA composite ({label})")
    except (SurfaceSnapshotRefusal, AttributeError, TypeError,
            ValueError, OSError) as problem:
        print(f"    composite wrfout skipped for {label}: {problem}")
        return None

def main() -> int:
    # Deferred like every other gpuwm import in this driver: the module
    # has to be importable by a bare `python tools/da_cycle_prepared.py
    # --help` from a checkout, and the package lands on sys.path only
    # once the process is actually running the tool.
    from gpuwm.da import background

    parser = argparse.ArgumentParser(
        description="cycling radar DA over a prepared single-domain case")
    # -- the prepared authority (all required: this driver reproduces the
    #    front door's own binding rather than inventing a looser one) ----
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--authority-dir", type=Path, default=None,
                        help="defaults to <prepared-root>/../authority")
    parser.add_argument(
        "--source", default=background.DEFAULT_BACKGROUND_SOURCE,
        choices=sorted(background.BACKGROUND_SOURCES),
        help=("which background the prepared case was built on.  This is "
              "the SELECTION recorded in the report, not a switch that "
              "changes how the case is read: the prepared root already "
              "IS one source's case, and naming a different one here is "
              "refused at the front door.  "
              f"{background.DEFAULT_BACKGROUND_SOURCE} is the default "
              "and its behaviour is unchanged"))
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prepared-content-sha256", required=True)
    parser.add_argument("--physics-profile", required=True)
    parser.add_argument("--run-seconds", type=float, required=True,
                        help="the hash-bound experiment's own run_seconds")
    parser.add_argument("--history-interval-seconds", type=float,
                        required=True)
    parser.add_argument(
        "--tolerate-physics-vocabulary-drift", action="store_true",
        help=("accept a prepared physics receipt that differs from this "
              f"tree only in {list(PHYSICS_VOCABULARY_FIELDS)} -- registry "
              "vocabulary, not physics.  Any other difference is still a "
              "refusal, and the divergence is recorded in the report"))
    # -- the cycle ------------------------------------------------------
    parser.add_argument("--leg-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--final-leg-seconds", type=float, default=None,
        help=("duration of the LAST leg only (default: --leg-seconds). "
              "The last leg's analysis is computed and never applied, so "
              "a longer final leg is a free forecast from the last "
              "applied analysis, verified against the final obs file at "
              "its own end -- cycling at one cadence and verifying at a "
              "longer lead without pretending the driver can restart"))
    parser.add_argument(
        "--free-leg-seconds", type=float, default=None,
        help=("duration of each FREE leg (default: --leg-seconds). A "
              "nowcast that cycles on the radar's own volume times has "
              "an observed-leg length set by the data and a forecast "
              "length set by what is worth looking at; this separates "
              "them instead of making one impersonate the other"))
    parser.add_argument(
        "--free-legs", type=int, default=0,
        help=("number of trailing legs run with NO observations: no "
              "analysis, no verification, composites still saved.  When "
              "nonzero, every obs leg's analysis IS applied (the free "
              "legs are the forecast running past the last observation, "
              "whose verification frames do not exist yet -- receipts "
              "and renders must say so).  Zero keeps the legacy "
              "final-leg-is-verification rule unchanged"))
    parser.add_argument(
        "--save-composites", action="store_true",
        help=("write each trajectory's column-max H_Z(x) at every leg "
              "end to <out>/composites/legNN_<name>.npz (float32 "
              "(ny, nx)).  A quicklook diagnostic, deliberately tiny; "
              "wrfout history for cycled arms remains its own package"))
    parser.add_argument("--members", type=int, default=10)
    # -- the fine nest over the free forecast ---------------------------
    #
    # Off unless asked for.  When on, it applies to the FREE legs only:
    # the parent carries the ensemble and the assimilation, and the nest
    # is the detailed picture of the forecast that runs past the
    # observations.  Everything about the child except these keys is
    # DERIVED -- dx, dy and dt come off the parent through the ratio
    # chain and are never typed here.
    parser.add_argument(
        "--nest-ratio", type=int, default=3,
        help=("parent-to-child refinement ratio, applied to BOTH space "
              "and time (WRF's SINT assumes a square ratio).  3 off a "
              "3 km parent is a 1 km nest at a 5 s step"))
    parser.add_argument(
        "--nest-half-width-km", type=float, default=None,
        help=("half-width of the nest in kilometres, centred in the "
              "parent.  Turning this on is what enables the nest; "
              "mutually exclusive with --nest-nx/--nest-ny"))
    parser.add_argument("--nest-nx", type=int, default=None,
                        help="child extent in CHILD cells (with --nest-ny)")
    parser.add_argument("--nest-ny", type=int, default=None,
                        help="child extent in CHILD cells (with --nest-nx)")
    parser.add_argument("--nest-i-parent-start", type=int, default=None,
                        help="1-based parent cell of the child's origin "
                             "(default: centred)")
    parser.add_argument("--nest-j-parent-start", type=int, default=None)
    parser.add_argument(
        "--nest-members", type=int, default=None,
        help=("how many ensemble members carry a nest, beside the "
              "control (default 0: the control only).  The parent always "
              "carries the FULL ensemble -- the nest is deliberately "
              "cheap, and nest cost scales as ratio^3 per covered parent "
              "cell TIMES this number"))
    parser.add_argument(
        "--nest-history-interval-s", type=float, default=None,
        help="child history cadence (default: the parent's)")
    parser.add_argument(
        "--nest-acknowledge", action="append", default=[],
        help=("acknowledge a nested-domain admissibility refusal by id, "
              "e.g. nested-forecast:sub-gray-zone-pbl"))
    parser.add_argument(
        "--obs", type=Path, action="append", default=[],
        help=("one gpuwm-obs.radar-grid.v1 file per leg, in leg order.  "
              "The last leg's file is read for verification only -- its "
              "analysis is computed and reported but never applied, so "
              "the final state is a forecast nobody has corrected"))
    parser.add_argument(
        "--grid-wrfout", type=Path, action="append", default=[],
        help=("the wrfout whose georeference each leg's observations were "
              "gridded onto, in leg order; one per --obs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, default=None,
                        help="where per-leg member checkpoints are staged "
                             "(a tmpfs is a good choice); default is a "
                             "temporary directory under --out")
    # -- carrying the ensemble between processes ------------------------
    # A leg boundary inside one process is snapshots + unapplied
    # increments; these two flags are that same boundary on disk, so a
    # continuous nowcast can assimilate an observation that did not
    # exist when the previous cycle ran.  tools/da_ensemble_state.py
    # documents the format and does the identity checking.
    parser.add_argument(
        "--resume-ensemble", type=Path, default=None,
        help=("resume from an ensemble generation written by "
              "--save-ensemble instead of perturbing a fresh ensemble. "
              "Leg 0 restores those snapshots, applies the increments "
              "the generation carried, and starts at the generation's "
              "own elapsed seconds"))
    parser.add_argument(
        "--save-ensemble", type=Path, default=None,
        help=("write the ensemble generation at the end of the LAST "
              "OBSERVED leg -- before any free legs, because free legs "
              "are a branch off the cycle and must never become the "
              "cycle.  Requires at least one --obs"))
    parser.add_argument(
        "--leg-number-offset", type=int, default=0,
        help=("absolute leg number of this run's first leg, so composite "
              "and increment filenames from consecutive resumed runs do "
              "not collide (default 0)"))
    # -- the ensemble and the analysis ----------------------------------
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--wind-sigma-ms", type=float, default=1.5)
    parser.add_argument("--length-scale-km", type=float, default=150.0)
    parser.add_argument("--horizontal-loc-m", type=float, default=36000.0)
    parser.add_argument("--vertical-loc-m", type=float, default=4000.0)
    parser.add_argument("--rtps-alpha", type=float, default=0.9)
    parser.add_argument("--relaxation", default="rtps",
                        choices=("rtps", "rtpp"),
                        help="which posterior relaxation --rtps-alpha "
                             "drives; see gpuwm.da.letkf.RELAXATION_MODES")
    parser.add_argument("--thin-cells", type=int, default=2)
    parser.add_argument("--err-inflation", type=float, default=1.0)
    parser.add_argument("--memory-budget-mib", type=float, default=6144.0)
    parser.add_argument(
        "--solve-device", default="auto", choices=("auto", "host", "cuda"),
        help=("where the LETKF analysis solves.  'auto' (default) takes "
              "the card when this process can reach one and numpy when it "
              "cannot, and the receipt records which and why.  'host' and "
              "'cuda' are honoured verbatim: pin host to reproduce a "
              "receipt banked on numpy, pin cuda to make a missing card "
              "an error instead of a silent 20x slower run"))
    parser.add_argument(
        "--dump-analysis-bundle", type=Path, default=None,
        help=("copy each leg's ANALYSIS INPUTS -- the staged member "
              "checkpoints, that leg's observation file and the history "
              "file its observations were gridded onto -- into "
              "DIR/leg_NNN, as a bundle tools/da_solve_ab.py can replay.  "
              "That is how the solve-device A/B gets a real leg to "
              "compare on without re-running the forecast that produced "
              "it.  Copies, because the stage directory is usually a "
              "tmpfs the next leg overwrites; budget one ensemble of "
              "checkpoints plus one observation file per leg"))
    parser.add_argument("--no-hotstart", action="store_true")
    # -- the moisture / hydrometeor half ---------------------------------
    # All default to zero amplitude, so a caller that does not ask for
    # them gets the wind-only ensemble this driver has always built and
    # the wind-only analysis that goes with it.  The switch that changes
    # the science is --hydrometeors, and it is explicit.
    parser.add_argument(
        "--hydrometeors", action="store_true",
        help="perturb the scheme's moisture and hydrometeor state and "
             "analyse it, instead of u and v alone.  Hydrometeor species "
             "are scaled MULTIPLICATIVELY together with their number and "
             "volume moments, which preserves the drop size distribution "
             "exactly, cannot produce a negative mixing ratio, and cannot "
             "break a moment pair")
    parser.add_argument("--theta-sigma-k", type=float, default=0.5)
    parser.add_argument("--qv-log-sigma", type=float, default=0.05,
                        help="FRACTIONAL, not kg/kg")
    parser.add_argument("--hydro-log-sigma", type=float, default=0.7,
                        help="FRACTIONAL, per species, applied to every "
                             "moment of that species")
    parser.add_argument("--thermo-length-scale-km", type=float, default=60.0)
    parser.add_argument("--thermo-vertical-levels", type=float, default=3.0)
    parser.add_argument("--clip-sigmas", type=float, default=2.5)
    parser.add_argument("--reflectivity-analysis", action="store_true",
                        help="assimilate the merged reflectivity batch "
                             "beside the velocity batches; requires "
                             "--hydrometeors, since reflectivity against "
                             "a wind-only state vector analyses nothing")
    parser.add_argument("--z-thin-cells", type=int, default=2)
    parser.add_argument("--z-err-inflation", type=float, default=1.0)
    parser.add_argument("--clear-air-analysis", action="store_true",
                        help="assimilate clear-air 'zero' observations: "
                             "cells the radar measured and found free of "
                             "significant echo. Suppresses spurious "
                             "convection. Requires --hydrometeors and an "
                             "observation file built with a clear-air "
                             "assessment; a file without one is refused "
                             "rather than having zeroes inferred from its "
                             "echo mask")
    parser.add_argument("--z0-thin-cells", type=int, default=4,
                        help="clear air is the majority of any volume and "
                             "is smooth, so it starves the filter's rank "
                             "faster than echo does")
    parser.add_argument("--z0-err-inflation", type=float, default=1.0)
    # -- the satellite half ----------------------------------------------
    parser.add_argument(
        "--goes-cwp", type=Path, action="append", default=[],
        help="one gpuwm-obs.goes-grid.v1 file per leg, in leg order, "
             "assimilated as a cloud-water-path batch beside the radar "
             "ones. Build them with tools/obs_goes_grid_build.py. Legs "
             "past the end of this list assimilate radar only. Requires "
             "--hydrometeors and --cwp-vertical-loc-m")
    parser.add_argument("--cwp-thin-cells", type=int, default=2)
    parser.add_argument("--cwp-err-inflation", type=float, default=1.0)
    parser.add_argument("--cwp-horizontal-loc-m", type=float, default=None,
                        help="horizontal localisation for the CWP batch; "
                             "defaults to --horizontal-loc-m")
    parser.add_argument(
        "--cwp-vertical-loc-m", type=float, default=None,
        help="vertical localisation for the CWP batch, in metres. "
             "REQUIRED with --goes-cwp and deliberately has no default: "
             "CWP is a COLUMN INTEGRAL carried at one level, so this "
             "radius is what decides whether the observation acts on the "
             "column it integrated or on a slab. The radar default "
             "(4 km) would assimilate a whole-column measurement as a "
             "4 km-tall one, and in particular would stop a clear-sky "
             "zero from removing model cloud at other heights")
    parser.add_argument(
        "--cwp-ice-species", default="qc,qi,qs",
        help="comma-separated condensate integrated for an ice/mixed "
             "observation. The default is the model's own optical "
             "condensate (gpuwm/core/rrtmgp.py:1097-1098). "
             "'qc,qi' is docs/obs-goes-cwp-operator-spec.md's v1 rule; "
             "which is right is a scoreboard question")
    parser.add_argument("--positivity-policy", default=None,
                        choices=("clip", "reject", "none"),
                        help="required once a non-negative field is "
                             "analysed; gpuwm.da.positivity documents "
                             "what each choice costs")
    # -- surface observations (default OFF) -------------------------------
    # METAR/ASOS through the rw_asos seam (gpuwm-obs.asos-surface.v1).
    # A quantity is enabled by stating its error standard deviation; the
    # record is hourly-matched by decoder design, so most sub-hourly
    # cycles legitimately see zero fresh surface reports and each report
    # enters exactly one analysis (the nearest).  gpuwm/da/obs_surface.py
    # documents what the v1 seam can and cannot express.
    parser.add_argument(
        "--surface-obs", type=Path, default=None,
        help="one gpuwm-obs.asos-surface.v1 record covering the whole "
             "run; reports are routed to the analysis nearest their "
             "valid time.  OFF unless given")
    parser.add_argument(
        "--sfc-t2-sigma-k", type=float, default=None,
        help="assimilate 2 m temperature with this error stddev (K); "
             "representativeness, not instrument precision -- WoFS-like "
             "practice is 1.5-2.5 K at storm-scale grids")
    parser.add_argument(
        "--sfc-wspd-sigma-ms", type=float, default=None,
        help="assimilate 10 m wind SPEED (the v1 seam carries no "
             "direction) with this error stddev (m/s); H is "
             "hypot(u10, v10) of the member diagnostics")
    parser.add_argument("--sfc-err-inflation", type=float, default=1.0)
    parser.add_argument(
        "--sfc-elev-max-diff-m", type=float, default=200.0,
        help="refuse stations whose table elevation differs from the "
             "model terrain at their gridpoint by more than this")
    parser.add_argument(
        "--sfc-max-age-s", type=float, default=900.0,
        help="refuse reports older (or newer) than this at the analysis")
    parser.add_argument(
        "--sfc-horizontal-loc-m", type=float, default=None,
        help="surface localization override; both --sfc-*-loc-m or "
             "neither (default: the run's --horizontal/vertical-loc-m)")
    parser.add_argument("--sfc-vertical-loc-m", type=float, default=None)
    args = parser.parse_args()
    if args.surface_obs is not None:
        if args.sfc_t2_sigma_k is None and args.sfc_wspd_sigma_ms is None:
            parser.error(
                "--surface-obs was given but neither --sfc-t2-sigma-k "
                "nor --sfc-wspd-sigma-ms states an error; a quantity is "
                "enabled by stating its sigma, and there is no default "
                "sigma on purpose")
        if not args.obs:
            parser.error(
                "--surface-obs joins the radar analysis legs; a "
                "surface-only cycle has no analysis times to join and "
                "is not wired")
    elif (args.sfc_t2_sigma_k is not None
          or args.sfc_wspd_sigma_ms is not None
          or args.sfc_horizontal_loc_m is not None
          or args.sfc_vertical_loc_m is not None):
        parser.error(
            "surface flags were given without --surface-obs; they would "
            "silently assimilate nothing")
    if (args.sfc_horizontal_loc_m is None) != (
            args.sfc_vertical_loc_m is None):
        parser.error(
            "--sfc-horizontal-loc-m and --sfc-vertical-loc-m come "
            "together; a localisation is a lens, not a line")
    if args.reflectivity_analysis and not args.hydrometeors:
        parser.error(
            "--reflectivity-analysis needs --hydrometeors: reflectivity "
            "constrains condensate, and against a u/v state vector every "
            "dBZ increment would come from wind-hydrometeor sampling "
            "covariance in a rank-(R-1) ensemble, which is noise")
    if args.clear_air_analysis and not args.hydrometeors:
        parser.error(
            "--clear-air-analysis needs --hydrometeors: a clear-air zero "
            "says condensate is absent, and against a u/v state vector "
            "there is no condensate for it to act on")
    if args.goes_cwp and not args.hydrometeors:
        parser.error(
            "--goes-cwp needs --hydrometeors: cloud water path IS the "
            "column condensate, and against a u/v state vector every CWP "
            "increment would come from wind-condensate sampling covariance "
            "in a rank-(R-1) ensemble, which is noise")
    if args.goes_cwp and args.cwp_vertical_loc_m is None:
        parser.error(
            "--goes-cwp needs an explicit --cwp-vertical-loc-m. CWP is a "
            "column integral carried at one model level, so its vertical "
            "localisation radius is not a tuning detail: it is what decides "
            "whether the observation acts on the column it actually "
            "integrated. Inheriting the radar radius would silently "
            "assimilate a whole-column measurement as a 4 km-tall one, and "
            "would stop an obs-clear zero from removing model cloud that "
            "sits outside that slab")
    if args.goes_cwp and len(args.goes_cwp) > len(args.obs):
        parser.error(
            f"--goes-cwp was given {len(args.goes_cwp)} file(s) but --obs "
            f"has {len(args.obs)}; satellite files are matched to legs by "
            "position and a leg with no radar file runs no analysis at all")
    if args.hydrometeors and args.positivity_policy is None:
        parser.error(
            "--hydrometeors analyses physically non-negative fields and "
            "--positivity-policy is unstated. clip / reject / none are not "
            "equivalent: clipping at zero ADDS mass and is biased wetward, "
            "rejecting conserves the background and invents gradients, and "
            "none lets the microphysics meet the negatives")

    nest_requested = (args.nest_half_width_km is not None
                      or args.nest_nx is not None
                      or args.nest_ny is not None)
    if nest_requested and args.free_legs <= 0:
        parser.error(
            "--nest-* needs --free-legs > 0: the nest runs over the FREE "
            "forecast legs, which is the whole point of it -- the parent "
            "carries the assimilation and the nest carries the detail of "
            "the forecast that runs past the observations")
    if args.nest_members is not None and not nest_requested:
        parser.error("--nest-members without a nest extent "
                     "(--nest-half-width-km or --nest-nx/--nest-ny)")
    if (args.nest_members is not None
            and args.nest_members > args.members):
        parser.error(
            f"--nest-members {args.nest_members} exceeds --members "
            f"{args.members}: the nest is a subset of the parent ensemble")

    # The nested lane still carried the older, stricter form of this rule
    # ("--obs is required", full stop).  The cycling nowcast deliberately
    # relaxed it so a run can be a pure free forecast off a resumed
    # generation, which is how the auto-cycle daemon extends a nowcast past
    # the observations it has.  The relaxed rule is the newer intent and it
    # is a superset, so it wins; the nest refusals above are additive.
    if not args.obs and args.free_legs <= 0:
        parser.error(
            "--obs is required (one observation file per leg). A run "
            "with no observations at all is only meaningful as a free "
            "forecast, which needs --free-legs and, to start from "
            "anything but the prepared background, --resume-ensemble")
    if args.save_ensemble is not None and not args.obs:
        parser.error(
            "--save-ensemble writes the generation at the end of the "
            "last OBSERVED leg, and this run has no --obs: a free-legs "
            "branch is a forecast off the cycle, never the cycle itself")
    if args.leg_number_offset < 0:
        parser.error("--leg-number-offset must be >= 0")
    if len(args.grid_wrfout) != len(args.obs):
        parser.error(
            f"--grid-wrfout was given {len(args.grid_wrfout)} time(s) and "
            f"--obs {len(args.obs)}: every leg's observations are bound to "
            "the georeference they were gridded onto, and pairing them by "
            "position is the caller's statement of which is which")
    legs = len(args.obs) + args.free_legs
    authority = (args.authority_dir if args.authority_dir is not None
                 else args.prepared_root.parent / "authority")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_root = Path(args.stage_dir) if args.stage_dir else out / "stage"
    report: dict = {"schema": REPORT_SCHEMA, "stability": "experimental",
                    "args": {key: str(value) for key, value
                             in vars(args).items()}, "legs": []}
    t_total = time.time()

    # ---- imports (CuPy present; model env untouched) -----------------------
    import cupy as cp

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, execute_experiment)
    from gpuwm.da import moments, nested_forecast, obsop, perturb
    from gpuwm.da.hotstart import HotStartConfig, hotstart_increments
    from gpuwm.da.letkf import Localization
    from gpuwm.da.obs_radar import read_document
    from gpuwm.da.obs_surface import (SurfaceObsConfig,
                                      surface_to_gridded_obs)
    from gpuwm.da.obsop_cwp import CwpComposition, checkpoint_cwp_provider
    from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                             assimilate_radar_grid,
                                             grid_rotation,
                                             member_earth_winds)
    from gpuwm.da import treatment

    # Which streams this invocation CLAIMS. Derived from the flags once,
    # here, so the proof below is checking the same set the driver acted
    # on. Radial velocity is unconditional: every leg reads a radar grid
    # and the velocity batches are what a cycle is built around.
    enabled_obs_kinds = ["radial_velocity"]
    if args.reflectivity_analysis:
        enabled_obs_kinds.append("reflectivity")
    if args.clear_air_analysis:
        enabled_obs_kinds.append("clear_air_reflectivity")
    if args.surface_obs is not None:
        enabled_obs_kinds.append("surface")
    if args.goes_cwp:
        enabled_obs_kinds.append("cloud_water_path")
    enabled_obs_kinds = tuple(enabled_obs_kinds)
    from gpuwm.ensemble.increments import apply_increments
    from gpuwm.ensemble.member import refresh_diagnostics
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.runtime import declared_constant_glw
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    # The prepared authority was written by the tree that ran the 24 h
    # forecast on this node; between that tree and this branch the physics
    # REGISTRY VOCABULARY moved (maturity "model-validated" ->
    # "wrf-matched-run", and with it the registry digest).  Every physics
    # selector, component and profile id is identical -- verified below
    # field by field -- so this driver tolerates exactly those two
    # vocabulary fields, refuses anything else, and records the
    # divergence in its report.  Driver-side accommodation only; the
    # repo's preflight stays strict.
    import gpuwm.prepared_single_domain_forecast as psdf
    _orig_check = psdf._validate_front_door_physics_proof
    vocabulary_divergence: dict = {}

    def _tolerant_check(proof, *, source, profile, cfg):
        try:
            return _orig_check(proof, source=source, profile=profile,
                               cfg=cfg)
        except ValueError as error:
            if (not args.tolerate_physics_vocabulary_drift
                    or "physics selection differs" not in str(error)):
                raise
            selected = dict(proof["physics"])
            expected = dict(psdf.validate_single_domain_physics_profile(
                profile, config=cfg,
                expert_acknowledgements=tuple(
                    selected["acknowledgements"]),
                acknowledgement_provenance=selected[
                    "acknowledgement_provenance"]))
            diverged = {}
            for key in PHYSICS_VOCABULARY_FIELDS:
                if selected.get(key) != expected.get(key):
                    diverged[key] = {"proof": selected.pop(key, None),
                                     "branch": expected.pop(key, None)}
            if selected != expected:
                raise
            vocabulary_divergence.update(diverged)
            print(f"physics proof vocabulary divergence tolerated: "
                  f"{diverged}", flush=True)
            return dict(proof["physics"])

    psdf._validate_front_door_physics_proof = _tolerant_check

    inputs = preflight_prepared_forecast(
        source=args.source, prepared_root=args.prepared_root,
        proof_sha256=args.proof_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        prepared_content_sha256=args.prepared_content_sha256,
        experiment_config=authority / "experiment.toml",
        wps_namelist=authority / "namelist.wps",
        physics_profile=args.physics_profile,
        run_seconds=args.run_seconds,
        history_interval_seconds=args.history_interval_seconds)
    psdf._validate_front_door_physics_proof = _orig_check
    report["physics_proof_vocabulary_divergence"] = vocabulary_divergence
    # The serialization contract this tree carries, recorded beside the
    # per-trajectory snapshot_fields lists: the audit is the DIFF of the
    # two (contract minus snapshot = fields this configuration does not
    # allocate; snapshot minus contract = impossible by construction).
    report["state_serialized_attrs"] = list(STATE_SERIALIZED_ATTRS)
    exp = inputs.experiment
    cfg = exp.root.run
    dt = float(cfg.dt)
    leg_seconds = float(args.leg_seconds)
    final_leg_seconds = (float(args.final_leg_seconds)
                         if args.final_leg_seconds is not None
                         else leg_seconds)
    free_leg_seconds = (float(args.free_leg_seconds)
                        if args.free_leg_seconds is not None else None)
    n_obs = len(args.obs)

    def leg_length(index: int) -> float:
        """How long leg ``index`` runs for.

        A free leg with its own declared length uses it; otherwise the
        rule is the one this driver has always had -- every leg is
        ``--leg-seconds`` except the last, which may be longer.
        """

        if free_leg_seconds is not None and index >= n_obs:
            return free_leg_seconds
        return final_leg_seconds if index == legs - 1 else leg_seconds
    print(f"preflight OK: {cfg.nx}x{cfg.ny}x{cfg.nz} dt={dt} "
          f"mp={cfg.mp_physics} hyps={cfg.hypsometric_opt}", flush=True)

    perturb_fields = [
        {"name": "u", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
        {"name": "v", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
    ]
    perturb_species: list[dict] = []
    if args.hydrometeors:
        # The species the SCHEME advances, never a list typed here.  qv
        # is every scheme's mass_only field and has no number moment, so
        # it goes through the field path with the multiplicative mode;
        # the rest go through the species path, which scales each one's
        # mass, number and volume moments by a single common factor.
        scheme = moments.scheme_moments(int(cfg.mp_physics))
        perturb_fields += [
            {"name": "theta", "amplitude": args.theta_sigma_k,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels},
            {"name": "qv", "amplitude": args.qv_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "mode": "lognormal", "clip_sigmas": args.clip_sigmas},
        ]
        perturb_species = [
            {"mass_field": name, "amplitude": args.hydro_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "clip_sigmas": args.clip_sigmas,
             "threshold_kg_kg": scheme.q_threshold}
            for name in scheme.mass_fields
            if name in perturb.SUPPORTED_SPECIES]
        report["perturbation"] = {
            "scheme": scheme.name, "mp_physics": scheme.mp_physics,
            "species": [spec["mass_field"] for spec in perturb_species]}
    cfg_perturb = perturb.PerturbationConfig.from_mapping({
        "dx_km": float(cfg.dx) / 1000.0, "dy_km": float(cfg.dy) / 1000.0,
        "rim_width": 5,
        "fields": perturb_fields,
        "species": perturb_species,
    })
    hot_cfg = HotStartConfig()

    # ---- how each trajectory's background was built --------------------
    # Planned BEFORE any GPU work, from the perturbation configuration
    # this run actually assembled, so an ensemble that would be N
    # identical copies of the control refuses here instead of consuming
    # a card for an hour and reporting zero spread.  The result is one
    # record per trajectory, and it goes into the report verbatim: a
    # skill comparison can then attribute a difference to the background
    # rather than guess at it.
    try:
        member_plan = background.plan_member_backgrounds(
            control_name=CONTROL, members=int(args.members),
            seed=int(args.seed), perturbed_fields=perturb_fields,
            perturbed_species=perturb_species)
    except background.BackgroundError as error:
        # A refusal a caller can act on, in this driver's own idiom --
        # the boundary-horizon refusal below reads the same way.
        raise SystemExit(str(error)) from None
    report["background"] = background.background_receipt(
        source=args.source, cycle=None, members=member_plan,
        prepared_content_sha256=str(args.prepared_content_sha256),
        notes={
            "prepared_root": str(args.prepared_root),
            "forcing_hours": list(inputs.forcing_hours),
            "initial_valid_time": exp.start_time.isoformat(),
            "run_seconds": float(args.run_seconds),
            # Taken from the hash-bound proof, not from a flag: these
            # are what the preparation actually fetched, so a reader can
            # tell how old the first guess was without trusting the
            # command line that started this process.
            "source_cycle": inputs.proof.get("source_cycle"),
            "source_forecast_hours": inputs.proof.get(
                "source_forecast_hours"),
            # Stated rather than implied: the members share this case's
            # lateral boundary conditions, so the rim taper is what keeps
            # the perturbation legal and spread decays toward the rim by
            # construction (gpuwm/da/perturb.py documents the whole list).
            "shared_lateral_boundaries": True,
        })
    print("background: " + json.dumps({
        "source": report["background"]["source"],
        "initial_hydrometeors": report["background"]["initial_hydrometeors"],
        "members": report["background"]["ensemble"]["member_count"],
        "construction": report["background"]["ensemble"]["construction"],
    }, sort_keys=True), flush=True)

    trajectories = [CONTROL] + list(range(args.members))
    setup_arrays: dict | None = None
    snapshots: dict = {name: None for name in trajectories}
    pending: dict = {name: None for name in trajectories}
    hot_pending: dict = {}

    # ---- the ensemble this run starts from ---------------------------------
    # Either a fresh perturbed ensemble off the prepared background (leg 0
    # perturbs), or a generation a previous process wrote (leg 0 restores
    # and applies that generation's unapplied analysis).  The identity is
    # what makes the second safe: everything that changes the meaning of
    # the stored arrays is compared before a single array is read.
    identity = ens_state.EnsembleIdentity(
        members=int(args.members), nx=int(cfg.nx), ny=int(cfg.ny),
        nz=int(cfg.nz), dt_s=dt, mp_physics=int(cfg.mp_physics),
        physics_profile=str(args.physics_profile),
        prepared_content_sha256=str(args.prepared_content_sha256))
    resumed_from: dict | None = None
    base_seconds = 0.0
    if args.resume_ensemble is not None:
        snapshots, pending, resumed_from = ens_state.read_generation(
            args.resume_ensemble, identity)
        base_seconds = float(resumed_from["elapsed_seconds"])
        report["resumed_from"] = {
            "directory": str(args.resume_ensemble),
            "written": resumed_from["written"],
            "elapsed_seconds": base_seconds,
            "leg_number": resumed_from["leg_number"],
            "valid_time": resumed_from.get("valid_time"),
            "trajectories_with_unapplied_analysis": sorted(
                key for key, entry
                in resumed_from["trajectories"].items()
                if entry.get("pending")),
        }
        print(f"resumed ensemble from {args.resume_ensemble} at "
              f"{base_seconds:.0f} s elapsed "
              f"(leg {resumed_from['leg_number']})", flush=True)
    report["leg_number_offset"] = int(args.leg_number_offset)
    report["base_elapsed_seconds"] = base_seconds

    # The prepared case's lateral boundary conditions only cover the
    # hash-bound run length.  Integrating past it would read boundary
    # data that does not exist, so it is a refusal here rather than a
    # surprise inside the integrator -- a resumed daemon hits this edge
    # eventually by construction, and the honest answer is a new case.
    span_end = base_seconds + sum(leg_length(i) for i in range(legs))
    if span_end > float(args.run_seconds) + 1e-6:
        raise SystemExit(
            f"this run would integrate to {span_end:.0f} s but the "
            f"prepared case is bound to {float(args.run_seconds):.0f} s "
            "of boundary data; shorten the legs or prepare a case on a "
            "newer background")
    #: The absolute leg number of a within-run leg index.
    def leg_number(index: int) -> int:
        return int(args.leg_number_offset) + index

    #: The last leg that carries an observation; the ensemble generation
    #: is written there and nowhere else.
    save_at_leg = len(args.obs) - 1 if args.save_ensemble else None
    #: Per-member leg-end dBZ, diagnosed on the device from the state
    #: that was snapshotted.  This is H_Z(x) for the filter.
    member_dbz: dict = {}
    #: Per-member leg-end 2m/10m diagnostics off the live physics driver
    #: -- the surface H(x), snapshotted at the SAME point as member_dbz
    #: and never at leg start, where the prepared-cache cycle's
    #: re-initialised driver state is spin-up.
    member_sfc: dict = {}
    thb_host = None

    # ---- the fine nest over the free forecast ------------------------
    #
    # The child is DERIVED here, once, so an inadmissible nest is a
    # preflight refusal rather than a crash six legs in.  Which
    # trajectories carry it is a separate, explicit decision: the parent
    # carries the whole ensemble and the nest is deliberately cheap.
    nest_geometry = None
    nest_child_dc = None
    nest_trajectories: tuple = ()
    nest_first_leg = len(args.obs)
    if nest_requested:
        nest_members = (nested_forecast.DEFAULT_NEST_MEMBERS
                        if args.nest_members is None
                        else int(args.nest_members))
        nest_geometry = nested_forecast.NestGeometry(
            ratio=int(args.nest_ratio),
            nx=args.nest_nx, ny=args.nest_ny,
            half_width_km=args.nest_half_width_km,
            i_parent_start=args.nest_i_parent_start,
            j_parent_start=args.nest_j_parent_start,
            history_interval_s=args.nest_history_interval_s,
            members=nest_members)
        nest_child_dc = nested_forecast.nest_domain_config(
            exp, nest_geometry,
            acknowledgements=tuple(args.nest_acknowledge))
        nest_admissibility = nested_forecast.validate_nest_admissibility(
            nest_child_dc.run, parent_run=cfg,
            acknowledgements=tuple(args.nest_acknowledge))
        nest_trajectories = tuple(
            [CONTROL] + list(range(nest_members)))
        report["nest"] = nested_forecast.nested_forecast_receipt(
            geometry=nest_geometry,
            exp=nested_forecast.nested_experiment(exp, nest_child_dc),
            child_dc=nest_child_dc, admissibility=nest_admissibility,
            land_receipt={
                "terrain_policy": nested_forecast.TERRAIN_POLICY,
                "land_policy": nested_forecast.LAND_POLICY},
            legs=list(range(nest_first_leg, legs)),
            nest_members=nest_members)
        report["nest"]["trajectories"] = [str(name)
                                          for name in nest_trajectories]
        child_run = nest_child_dc.run
        print(f"nest: d{nest_child_dc.grid_id:02d} {child_run.nx}x"
              f"{child_run.ny}x{child_run.nz} dx={child_run.dx:g} "
              f"dt={child_run.dt:g} over legs "
              f"{nest_first_leg}..{legs - 1} for "
              f"{len(nest_trajectories)} trajector"
              f"{'y' if len(nest_trajectories) == 1 else 'ies'}", flush=True)

    #: Host snapshots of the child's state, one per nesting trajectory.
    #: The nest is built from the parent on its FIRST leg and restored
    #: from here afterwards, so the fine-scale structure it develops
    #: survives a leg boundary instead of being flattened back to a
    #: parent interpolation every fifteen minutes.
    nest_snapshots: dict = {name: None for name in nest_trajectories}

    def nests_this_leg(leg: int, name) -> bool:
        return (nest_child_dc is not None and leg >= nest_first_leg
                and name in nest_trajectories)

    # ---- model wiring, per member-leg ---------------------------------------

    def wire(run_seconds_total: float, *, child_dc=None):
        """Build one leg's parent model, and the clocks the child needs.

        With ``child_dc`` the clock and the schedule are the TWO-domain
        ones, but the child itself is NOT built here.  It is derived from
        the parent's state by SINT, and on every leg after the first that
        state does not exist yet: the host snapshot has still to be
        restored and the analysis increment applied.  Building the child
        before that would hand the nest a pre-analysis parent, which is
        the very failure this whole design exists to avoid.
        :func:`assemble` closes the model once the child is real.
        """
        exp_leg = dataclasses.replace(exp,
                                      run_seconds=float(run_seconds_total))
        if child_dc is not None:
            exp_leg = nested_forecast.nested_experiment(exp_leg, child_dc)
        restored = restore_prepared_cache(
            inputs.prepared_cache_path, expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static)
        driver = initialize_prepared_physics(
            restored.initial_result, cfg, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid,
            exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp))
        tick = resolve_clock(
            exp_leg, lbc_interval_s=float(inputs.boundary_interval_seconds))
        schedule = build_schedule(exp_leg, tick)
        clocks = tick.clocks()
        node = DomainNode(exp.root, inputs.grid,
                          restored.initial_result.state, clocks[1],
                          None, [], None)
        return SimpleNamespace(node=node, restored=restored, driver=driver,
                               clocks=clocks, schedule=schedule,
                               child_dc=child_dc)

    def assemble(wired, *, child_node=None):
        """Turn the wired pieces into the ExperimentState the executor runs."""
        nodes = {1: wired.node}
        if child_node is not None:
            nodes[child_node.cfg.grid_id] = child_node
        model = ExperimentState(wired.node, MappingProxyType(nodes),
                                wired.schedule, None, "da-cycle-shakedown")
        model._runtime_status = ModelRuntimeStatus()
        model._resumed = False
        model._resume_committed_history_grid_ids = frozenset()
        # The shared scratch arena hands every domain a prefix VIEW of one
        # backing buffer, on the premise that the schedule steps exactly
        # one domain at a time; the shared dycore workspace is the same
        # bargain.  Both stay None on this route -- with a nest attached
        # that is no longer a memory question but a correctness one, and
        # it is asserted rather than assumed.
        model._scratch_arena = None
        model._dycore_state_workspace = None
        if model._scratch_arena is not None \
                or model._dycore_state_workspace is not None:
            raise RuntimeError(
                "the DA cycling path requires per-domain scratch: a "
                "shared arena would let parent and child write the same "
                "bytes with no exception raised")
        model._io_manager = None
        model._last_checkpoint = None
        model._prepared_by_grid_id = MappingProxyType({
            1: SimpleNamespace(static_fields=inputs.static,
                               geog_selection=None,
                               initial_result=wired.restored.initial_result)})
        return model

    def teardown(*objects) -> None:
        for obj in objects:
            del obj
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # ---- the legs ------------------------------------------------------------

    leg_starts = []
    _cursor = base_seconds
    for _index in range(legs):
        leg_starts.append(_cursor)
        _cursor += leg_length(_index)

    # ---- surface observations (rw_asos seam), default OFF -------------------
    surface_cfg = None
    surface_schedule = None
    if args.surface_obs is not None:
        from datetime import datetime as _datetime  # noqa: PLC0415
        from datetime import timedelta as _timedelta  # noqa: PLC0415

        if not isinstance(exp.start_time, _datetime):
            raise SystemExit(
                "--surface-obs needs the experiment's absolute start time "
                f"to route reports to analyses, and exp.start_time is "
                f"{type(exp.start_time).__name__}")
        surface_localization = None
        if args.sfc_horizontal_loc_m is not None:
            surface_localization = Localization(
                horizontal_m=float(args.sfc_horizontal_loc_m),
                vertical_m=float(args.sfc_vertical_loc_m))
        surface_cfg = SurfaceObsConfig(
            temperature_error_k=args.sfc_t2_sigma_k,
            wind_speed_error_ms=args.sfc_wspd_sigma_ms,
            error_inflation=float(args.sfc_err_inflation),
            elevation_max_diff_m=float(args.sfc_elev_max_diff_m),
            max_age_seconds=float(args.sfc_max_age_s),
            temperature_localization=surface_localization,
            wind_localization=surface_localization)
        #: One analysis time per OBSERVED leg -- the same t_end the radar
        #: analysis runs at.  The adapter routes each hourly report to
        #: exactly one of these.
        surface_schedule = [
            exp.start_time + _timedelta(seconds=leg_starts[i]
                                        + leg_length(i))
            for i in range(len(args.obs))]
        report["surface_observations"] = {
            "record": str(args.surface_obs),
            "t2_sigma_k": args.sfc_t2_sigma_k,
            "wspd_sigma_ms": args.sfc_wspd_sigma_ms,
            "analysis_times": [t.isoformat() for t in surface_schedule],
        }

    for leg in range(legs):
        t_start = leg_starts[leg]
        t_end = t_start + leg_length(leg)
        leg_record: dict = {"leg": leg_number(leg), "leg_in_run": leg,
                            "start_s": t_start, "end_s": t_end,
                            "trajectories": {}}
        member_dbz.clear()
        member_sfc.clear()
        has_obs = leg < len(args.obs)
        goes_path = None
        if leg < len(args.goes_cwp):
            goes_path = Path(args.goes_cwp[leg])
            if not goes_path.is_file():
                raise FileNotFoundError(
                    f"leg {leg}: no GOES CWP file at {goes_path}")
        if has_obs:
            obs_path = Path(args.obs[leg])
            if not obs_path.is_file():
                raise FileNotFoundError(
                    f"leg {leg}: no observation file at {obs_path}")
            # Without --free-legs: the last leg's analysis is computed and
            # reported but NEVER applied -- it is the verification the run
            # is judged by, and applying it would leave the final state
            # corrected by the very observations used to score it.
            # With --free-legs: EVERY obs leg's analysis is applied, and
            # the trailing free legs are the forecast that runs past the
            # observations -- which is exactly why they cannot verify
            # anything: their verification frames do not exist yet.  The
            # override is this explicit flag condition, never a silent
            # change to the legacy rule.
            # --save-ensemble means this run is one link in a chain that
            # keeps cycling, so its last analysis is applied for the
            # same reason a free forecast's is: the run does not END at
            # this observation, and leaving the analysis uncomputed-into
            # the carried state would silently drop one cycle.
            if args.free_legs > 0 or args.save_ensemble is not None:
                analysis_due = True
                verification_only = False
            else:
                analysis_due = leg < legs - 1
                verification_only = leg == legs - 1

            grid_h = TargetGrid.from_wrfout(Path(args.grid_wrfout[leg]))
            document = read_document(obs_path, expected_grid=grid_h) \
                if obs_path.is_file() else None
        else:
            obs_path = None
            grid_h = None
            document = None
            analysis_due = False
            verification_only = False
        z_obs_cp = z_mask_cp = None
        if document is not None and not args.no_hotstart:
            z_obs_cp = cp.asarray(np.asarray(
                document["variables"]["z_obs"], np.float32))
            z_mask_cp = cp.asarray(np.asarray(
                document["variables"]["z_mask"]).astype(bool))

        for name in trajectories:
            t_leg = time.time()
            nested_leg = nests_this_leg(leg, name)
            wired = wire(t_end,
                         child_dc=nest_child_dc if nested_leg else None)
            node, restored, driver = wired.node, wired.restored, wired.driver
            state = node.state
            if setup_arrays is None:
                # The eta coordinate arrays and the base column mass a
                # checkpoint does not serialize (STATE_SETUP_ARRAYS in
                # gpuwm/state_serialization_contract.py). The CWP operator
                # integrates in the model's own mass measure and cannot
                # rebuild them from the npz, so they are captured here off
                # a live state, once, exactly as the reflectivity provider
                # would need thb.
                setup_arrays = {
                    "c1h": to_host(state.c1h).astype(np.float64),
                    "c2h": to_host(state.c2h).astype(np.float64),
                    "dnw": to_host(state.dnw).astype(np.float64),
                    "mub2d": to_host(state.mub2d).astype(np.float64),
                }
            # `resumed` drives model._resumed below, which stops the
            # resumed leg from rewriting history the previous process
            # already committed.  `resumed_from is None` is the separate
            # and more dangerous condition: a generation carried in from
            # --resume-ensemble is ALREADY perturbed, and perturbing it
            # again at leg 0 would throw away the analysis it carried with
            # no error raised anywhere.  Both guards are required; they are
            # not the same question.
            resumed = False
            if leg == 0 and resumed_from is None:
                if name != CONTROL:
                    prov = perturb.apply_perturbations(
                        state, args.seed + int(name), cfg_perturb)
                    refresh_diagnostics(
                        state, hypsometric_opt=cfg.hypsometric_opt)
            else:
                jump_clock(node.clock, t_start, dt)
                resumed = True
                for field, host in snapshots[name].items():
                    getattr(state, field)[...] = cp.asarray(
                        host, dtype=getattr(state, field).dtype)
                if pending[name]:
                    receipt = apply_increments(
                        state, pending[name], mp_physics=cfg.mp_physics)
                    leg_record["trajectories"].setdefault(str(name), {})[
                        "apply_fields"] = receipt["field_count"]
                    refresh_diagnostics(
                        state, hypsometric_opt=cfg.hypsometric_opt)
            health = StateHealthValidator(state).validate(
                phase=f"leg{leg}.{name}")
            if not health.ok:
                raise FloatingPointError(
                    f"leg {leg} {name}: pre-leg health failed: "
                    f"{vars(health)}")

            # -- the fine nest, built from the state above -------------------
            #
            # Deliberately AFTER the snapshot restore, the increment
            # application and the health gate: the parent state the child
            # is derived from has to be the ANALYSED one, or the nest
            # would inherit a forecast nobody corrected and the whole
            # exercise would be worth less than interpolating the output.
            child_node = child_driver = None
            if nested_leg:
                child_node, child_driver, land_receipt = \
                    nested_forecast.build_nested_child(
                        node, nest_child_dc,
                        static=inputs.static, surface=restored.surface,
                        landuse_identity=inputs.landuse_identity,
                        valid_time=exp.start_time,
                        clock=wired.clocks[nest_child_dc.grid_id],
                        parent_driver=driver,
                        constant_glw_wm2=declared_constant_glw(exp))
                child_state = child_node.state
                nest_entry = leg_record["trajectories"].setdefault(
                    str(name), {}).setdefault("nest", {})
                nest_entry["grid_id"] = nest_child_dc.grid_id
                if nest_snapshots.get(name) is None:
                    nest_entry["initialization"] = "parent-live-state-sint"
                else:
                    # A later nested leg: the child already has fine
                    # structure of its own, and flattening it back to a
                    # parent interpolation every leg boundary would throw
                    # away exactly what the nest is for.  The SINT above
                    # still runs -- it is what builds the state object and
                    # its base state -- and is then overwritten field by
                    # field, in the same shape the parent's own leg
                    # handoff uses.
                    nest_entry["initialization"] = "child-host-snapshot"
                    for field, host in nest_snapshots[name].items():
                        getattr(child_state, field)[...] = cp.asarray(
                            host, dtype=getattr(child_state, field).dtype)
                    refresh_diagnostics(
                        child_state,
                        hypsometric_opt=nest_child_dc.run.hypsometric_opt)
                # The child clock is placed on the same tick lattice as the
                # parent.  Its dtbc accumulator is deliberately left at
                # zero: a child's boundary clock resets at EVERY nested
                # force, the executor refuses a child STEP before the
                # leg's first parent STEP -> FORCE, so no child step can
                # read a value the force has not just written.
                child_node.clock.ticks = int(
                    round(t_start / float(nest_child_dc.run.dt))
                ) * child_node.clock.spec.step_ticks
                child_node.clock.step_count = int(
                    round(t_start / float(nest_child_dc.run.dt)))
                child_health = StateHealthValidator(child_state).validate(
                    phase=f"leg{leg}.{name}.nest")
                if not child_health.ok:
                    raise FloatingPointError(
                        f"leg {leg} {name}: nest pre-leg health failed: "
                        f"{vars(child_health)}")

            model = assemble(wired, child_node=child_node)
            if resumed:
                model._resumed = True
                model._resume_committed_history_grid_ids = frozenset(
                    model.nodes_by_grid_id)

            execute_experiment(model, history_handler=None,
                               progress_callback=None, validate_state=True,
                               skip_feedback_path=True,
                               pool_trim_per_period=True)
            cp.cuda.Stream.null.synchronize()

            thb_live = getattr(state, "thb", None)
            thb_snapshot = to_host(thb_live) if thb_live is not None \
                else None

            # -- leg-end diagnostics on the live device state ---------------
            refl = obsop.simulated_reflectivity(state, cfg)
            refl_host = to_host(refl).astype(np.float32)
            if name != CONTROL:
                member_dbz[int(name)] = refl_host.astype(np.float64)
            if surface_cfg is not None and name != CONTROL:
                # Leg-END surface diagnostics off the live driver: the
                # 2m/10m fields the surface layer diagnosed for the very
                # state that was snapshotted.  Leg START would hand the
                # filter re-initialised spin-up (documented simplification
                # of this driver), which is why the snapshot lives here
                # beside member_dbz and nowhere else.
                absent = [key for key in ("t2", "u10", "v10")
                          if key not in driver.fields]
                if absent:
                    raise RuntimeError(
                        f"leg {leg} {name}: --surface-obs needs the "
                        f"driver's {absent} diagnostics and this physics "
                        "profile does not allocate them")
                member_sfc[int(name)] = {
                    key: to_host(driver.fields[key]).astype(np.float64)
                    for key in ("t2", "u10", "v10")}
            if args.save_composites:
                comp_dir = out / "composites"
                comp_dir.mkdir(parents=True, exist_ok=True)
                composite_npz = (comp_dir /
                                 f"leg{leg_number(leg):02d}_{name}.npz")
                np.savez_compressed(
                    composite_npz,
                    refl_colmax=refl_host.max(axis=0),
                    elapsed_seconds=np.float64(
                        node.clock.elapsed_seconds))
                # ...and the same composite as a real wrfout beside it, so
                # `gpuwm render --engine rust` -- the renderer the render
                # law names -- can draw this member.  The npz is what the
                # cycle's own analysis reads and is untouched; this is a
                # second copy in the format a different tool reads.
                _write_composite_wrfout(
                    composite_npz, refl_host.max(axis=0), inputs.grid, cfg,
                    node.clock.elapsed_seconds, exp,
                    label=f"leg {leg_number(leg):02d} member {name}")
            # -- the nest's own leg-end product ------------------------------
            #
            # Written under a d02 name beside the parent's rather than
            # replacing it: the point of the exercise is a fine view OF the
            # parent's forecast, and a reader has to be able to see both.
            refl_nest = None
            if child_node is not None:
                refl_nest = obsop.simulated_reflectivity(
                    child_node.state, nest_child_dc.run)
                refl_nest_host = to_host(refl_nest).astype(np.float32)
                nest_entry["refl_max_dbz"] = float(refl_nest_host.max())
                nest_entry["elapsed_seconds"] = float(
                    child_node.clock.elapsed_seconds)
                nest_entry["step_count"] = int(child_node.clock.step_count)
                if args.save_composites:
                    np.savez_compressed(
                        comp_dir / f"leg{leg:02d}_{name}_d"
                                   f"{nest_child_dc.grid_id:02d}.npz",
                        refl_colmax=refl_nest_host.max(axis=0),
                        elapsed_seconds=np.float64(
                            child_node.clock.elapsed_seconds),
                        dx_m=np.float64(nest_child_dc.run.dx),
                        i_parent_start=np.int32(
                            nest_child_dc.i_parent_start),
                        j_parent_start=np.int32(
                            nest_child_dc.j_parent_start),
                        parent_grid_ratio=np.int32(
                            nest_child_dc.parent_grid_ratio))
                nest_snapshot = {}
                for field in STATE_SERIALIZED_ATTRS:
                    value = getattr(child_node.state, field, None)
                    if value is not None:
                        nest_snapshot[field] = to_host(value)
                nest_snapshots[name] = nest_snapshot
                nest_entry["snapshot_fields"] = sorted(nest_snapshot)
            entry = leg_record["trajectories"].setdefault(str(name), {})
            entry["wall_seconds"] = round(time.time() - t_leg, 1)
            entry["elapsed_seconds"] = float(node.clock.elapsed_seconds)
            if document is not None:
                z_mask = np.asarray(
                    document["variables"]["z_mask"]).astype(bool)
                z_obs = np.asarray(document["variables"]["z_obs"],
                                   np.float64)
                inside = z_mask
                model_in_mask = refl_host[inside].astype(np.float64)
                entry["z_obs_space"] = {
                    "points": int(inside.sum()),
                    "obs_mean_dbz": float(z_obs[inside].mean()),
                    "model_mean_dbz": float(model_in_mask.mean()),
                    "model_max_dbz": float(model_in_mask.max()),
                    "model_cols_gt35_in_echo": int(
                        (refl_host.max(axis=0) >= 35.0)[
                            z_mask.any(axis=0)].sum()),
                    "obs_cols_gt35": int(
                        ((z_obs * z_mask).max(axis=0) >= 35.0).sum()),
                    "innovation_mean_dbz": float(
                        (z_obs[inside] - model_in_mask).mean()),
                }
            if (analysis_due and name != CONTROL
                    and not args.no_hotstart):
                increments_hot, hot_prov = hotstart_increments(
                    state, z_obs_cp, z_mask_cp, hot_cfg,
                    simulated_dbz=refl)
                hot_pending[name] = {
                    field: to_host(values).astype(np.float32)
                    for field, values in increments_hot.items()}
                entry["hotstart"] = {
                    key: hot_prov[key] for key in hot_prov
                    if isinstance(hot_prov[key], (int, float, str))}

            snapshot = {}
            for field in STATE_SERIALIZED_ATTRS:
                value = getattr(state, field, None)
                if value is not None:
                    snapshot[field] = to_host(value)
            snapshots[name] = snapshot
            # The contract tuple grows over time (tke, nwfa/nifa pairs,
            # e_sgs since this driver was written), and the None-skip
            # above silently carries whatever the configuration
            # allocates.  Print what was ACTUALLY snapshotted into the
            # report, per trajectory, so a grown inventory is an audit
            # line rather than a surprise -- and so a member whose list
            # ever differs from the control's is visible in the receipt.
            entry["snapshot_fields"] = sorted(snapshot)
            if thb_host is None and thb_snapshot is not None:
                thb_host = thb_snapshot
            pending[name] = None

            teardown(model, node, restored, driver, state, refl,
                     wired, child_node, child_driver, refl_nest)
            print(f"leg {leg} {name}: {entry['wall_seconds']} s, "
                  f"elapsed {entry['elapsed_seconds']:.0f} s", flush=True)

        # -- analysis at t_end ------------------------------------------------
        if analysis_due or verification_only:
            shm_leg = stage_root / f"cycle_{leg_number(leg):03d}"
            if shm_leg.exists():
                shutil.rmtree(shm_leg)
            checkpoints = {}
            for index in range(args.members):
                member_dir = shm_leg / f"member_{index:03d}"
                member_dir.mkdir(parents=True)
                path = member_dir / f"gpuwmrst_d01_{int(t_end):06d}.npz"
                np.savez(path, **{f"state/{k}": v
                                  for k, v in snapshots[index].items()})
                checkpoints[index] = path

            analysis_fields = ("u", "v")
            provider = None
            if args.hydrometeors:
                # Derived from the scheme, then intersected with what the
                # checkpoints actually carry: a scheme may advance a
                # moment this configuration does not (Morrison's nc is
                # prognostic only under progn=1), and naming a field the
                # background has not got is a refusal rather than a zero.
                # Then any field the ensemble is constant in is dropped
                # by WHOLE SPECIES, so no moment pair is truncated -- the
                # filter refuses a spreadless field, correctly, and a
                # species the model has not made anywhere is a species
                # with nothing to update.
                carried = set(snapshots[0])
                candidates = tuple(
                    name for name in moments.analysis_fields(
                        int(cfg.mp_physics)) if name in carried)
                spreads = {}
                for name in candidates:
                    stack = np.stack([snapshots[i][name] for i in
                                      range(args.members)]).astype(
                                          np.float64)
                    scale = float(np.abs(stack).max())
                    widest = float(stack.std(axis=0, ddof=1).max())
                    spreads[name] = {"max_abs": scale,
                                     "max_spread": widest,
                                     "usable": widest > 1e-12 * scale}
                    del stack
                dropped = {name for name, entry in spreads.items()
                           if not entry["usable"]}
                for pair in moments.pairs_present(
                        tuple(carried), mp_physics=int(cfg.mp_physics)):
                    if dropped & set(pair.fields):
                        dropped |= set(pair.fields) & set(candidates)
                analysis_fields = tuple(name for name in candidates
                                        if name not in dropped)
                leg_record["analysis_field_selection"] = {
                    "candidates": list(candidates),
                    "dropped_for_no_ensemble_spread": sorted(dropped),
                    "spreads": spreads,
                }
                if not analysis_fields:
                    raise RuntimeError(
                        f"leg {leg}: no analysed field has ensemble "
                        f"spread ({spreads})")
                # Clear air is differenced against the same H_Z(x) echo is,
                # so it needs the same provider -- a clear-air-only cycle
                # would otherwise reach the filter with none.
                if args.reflectivity_analysis or args.clear_air_analysis:
                    def provider(index, state, _t=dict(member_dbz)):
                        """H_Z(x) from the device diagnostic this driver
                        already computed for the very state that was
                        snapshotted -- the product's own authority, and
                        the same one the leg-end diagnostics and the hot
                        start read.  The float64 column mirror would be
                        one Python call per column and a second Z
                        authority in the same cycle."""
                        return _t[int(index)]
            cwp_provider = None
            cwp_localization = None
            if goes_path is not None:
                if setup_arrays is None:
                    raise RuntimeError(
                        f"leg {leg}: the CWP operator needs c1h/c2h/dnw/"
                        "mub2d off a live state and none was captured; "
                        "this leg ran no trajectory")
                composition = CwpComposition(
                    ice=tuple(name.strip() for name
                              in args.cwp_ice_species.split(",")
                              if name.strip()),
                    clear=tuple(name.strip() for name
                                in args.cwp_ice_species.split(",")
                                if name.strip()))
                cwp_provider = checkpoint_cwp_provider(
                    cfg, composition=composition, **setup_arrays)
                cwp_localization = Localization(
                    horizontal_m=(args.cwp_horizontal_loc_m
                                  if args.cwp_horizontal_loc_m is not None
                                  else args.horizontal_loc_m),
                    vertical_m=args.cwp_vertical_loc_m)
            cfg_da = RadarAssimilationConfig(
                localization=Localization(
                    horizontal_m=args.horizontal_loc_m,
                    vertical_m=args.vertical_loc_m),
                rtps_alpha=args.rtps_alpha, relaxation=args.relaxation,
                analysis_fields=analysis_fields,
                velocity=True,
                reflectivity=bool(args.reflectivity_analysis),
                fall_speed="none",
                velocity_thinning_cells=args.thin_cells,
                velocity_error_inflation=args.err_inflation,
                reflectivity_thinning_cells=args.z_thin_cells,
                reflectivity_error_inflation=args.z_err_inflation,
                clear_air=bool(args.clear_air_analysis),
                clear_air_thinning_cells=args.z0_thin_cells,
                clear_air_error_inflation=args.z0_err_inflation,
                cwp=goes_path is not None,
                cwp_localization=cwp_localization,
                cwp_thinning_cells=args.cwp_thin_cells,
                cwp_error_inflation=args.cwp_err_inflation,
                positivity_policy=args.positivity_policy,
                mp_physics=int(cfg.mp_physics),
                solve_device=args.solve_device,
                memory_budget_mib=args.memory_budget_mib)
            # -- the replayable copy of this leg's analysis inputs -------
            # Written BEFORE the solve, from the same objects the solve is
            # about to consume, so a bundle can never describe a different
            # analysis than the one this leg ran.  Radar-only: a leg whose
            # analysis also needs a reflectivity or CWP forward operator
            # cannot be replayed from files alone (the operator needs the
            # scheme's setup state, which is not in a checkpoint), and a
            # bundle that silently dropped those batches would compare two
            # arms on an analysis neither of them performed.
            if args.dump_analysis_bundle is not None and obs_path is not None:
                if cfg_da.cwp or cfg_da.reflectivity or cfg_da.clear_air \
                        or surface_cfg is not None:
                    print(f"leg {leg}: analysis bundle NOT dumped -- this "
                          "analysis carries batches whose forward operator "
                          "is not reconstructable from files "
                          f"(reflectivity={cfg_da.reflectivity}, "
                          f"clear_air={cfg_da.clear_air}, "
                          f"cwp={cfg_da.cwp}, "
                          f"surface={surface_cfg is not None})", flush=True)
                else:
                    bundle_dir = (Path(args.dump_analysis_bundle)
                                  / f"leg_{leg_number(leg):03d}")
                    ab_manifest = ab_bundle.dump_real_bundle(
                        bundle_dir, checkpoints=checkpoints,
                        obs_path=obs_path,
                        grid_wrfout=Path(args.grid_wrfout[leg]),
                        grid=grid_h, cfg=cfg_da,
                        note=("Analysis inputs of a real cycling DA leg, "
                              "copied at the analysis seam by "
                              "tools/da_cycle_prepared.py."),
                        extra={"driver": "tools/da_cycle_prepared.py",
                               "leg": int(leg),
                               "leg_number": int(leg_number(leg)),
                               "elapsed_seconds": float(t_end),
                               "solve_device_of_the_dumping_run":
                                   args.solve_device})
                    leg_record["analysis_bundle"] = {
                        "path": str(bundle_dir),
                        "members": len(ab_manifest["members"]),
                        "grid_identity_sha256":
                            ab_manifest["grid"]["identity_sha256"]}
                    print(f"leg {leg}: analysis bundle -> {bundle_dir}",
                          flush=True)
            surface_batches = None
            surface_prov = None
            if surface_cfg is not None:
                simulated = {"t2": None, "u10": None, "v10": None}
                stacks = [member_sfc[i] for i in range(args.members)]
                if surface_cfg.temperature:
                    simulated["t2"] = np.stack(
                        [entry["t2"] for entry in stacks])
                if surface_cfg.wind_speed:
                    simulated["u10"] = np.stack(
                        [entry["u10"] for entry in stacks])
                    simulated["v10"] = np.stack(
                        [entry["v10"] for entry in stacks])
                surface_batches, surface_prov = surface_to_gridded_obs(
                    args.surface_obs, target_grid=grid_h,
                    analysis_time=surface_schedule[leg],
                    analysis_times=surface_schedule,
                    config=surface_cfg,
                    simulated_t2=simulated["t2"],
                    simulated_u10=simulated["u10"],
                    simulated_v10=simulated["v10"])
            t_solve = time.time()
            increments, prov = assimilate_radar_grid(
                checkpoints, obs_path, grid_h, cfg_da,
                reflectivity_provider=provider,
                extra_obs=surface_batches,
                extra_obs_provenance=surface_prov,
                cwp_observations=goes_path,
                cwp_provider=cwp_provider)
            leg_record["analysis"] = {
                "applied": bool(analysis_due),
                "solve_seconds": round(time.time() - t_solve, 1),
                "analysis_fields": list(analysis_fields),
                "innovations": prov["innovations"],
                "filter": prov["filter"],
                "velocity_thinning": prov["velocity_thinning"],
                "reflectivity_thinning": prov["reflectivity_thinning"],
                "moment_policy": prov["moment_policy"],
                "positivity": prov["positivity"],
                "surface": surface_prov,
                # What was actually assimilated this leg, COUNTED off the
                # batches the filter solved rather than inferred from
                # which flags were on. A leg whose satellite file held no
                # usable observation must not read the same as one that
                # had none to read, and a stream that was enabled but
                # contributed nothing must not read the same as one that
                # contributed. See gpuwm.da.treatment.
                "assimilated": treatment.cycle_record(
                    prov["innovations"],
                    adapter_provenance=prov["observations"],
                    extra_obs_provenance=prov["extra_observations"],
                    cwp_provenance=prov["cwp_observations"]),
                "goes_cwp_file": (None if goes_path is None
                                  else goes_path.name),
                "cwp_thinning": prov["cwp_thinning"],
                "cwp_error_inflation": prov["cwp_error_inflation"],
                "cwp_localization_horizontal_m": prov[
                    "cwp_localization_horizontal_m"],
                "cwp_localization_vertical_m": prov[
                    "cwp_localization_vertical_m"],
                "cwp_observations": prov["cwp_observations"],
                "cwp_composition": (
                    None if cwp_provider is None
                    else composition.to_payload()),
            }
            inc_stats = {}
            for field in analysis_fields:
                stack = np.stack([increments[i][field]
                                  for i in sorted(increments)])
                inc_stats[field] = {
                    "finite": bool(np.isfinite(stack).all()),
                    "max_abs": float(np.abs(stack).max()),
                    "rms_where_nonzero": float(np.sqrt(np.mean(
                        stack[stack != 0.0] ** 2)))
                    if np.any(stack != 0.0) else 0.0,
                    "nonzero_fraction": float(np.mean(stack != 0.0)),
                }
                if field in ("u", "v"):
                    inc_stats[field]["max_abs_ms"] = inc_stats[field][
                        "max_abs"]
            leg_record["analysis"]["increments"] = inc_stats
            # Every mass-shaped analysed field must be nonzero on exactly
            # the same gridpoints: with prior_inflation = 1 the increment
            # outside a localisation lens is bitwise zero, so two fields
            # disagreeing about where the analysis reached would mean the
            # localisation is not doing what it claims.
            support = None
            support_ok = True
            for field in analysis_fields:
                if field in ("u", "v", "w"):
                    continue
                stack = np.stack([increments[i][field]
                                  for i in sorted(increments)])
                touched = np.any(stack != 0.0, axis=0)
                if support is None:
                    support = touched
                elif not np.array_equal(support, touched):
                    support_ok = False
            leg_record["analysis"]["structural_zero"] = {
                "mass_fields_share_one_support": bool(support_ok),
                "support_points": (0 if support is None
                                   else int(support.sum())),
                "filter_active_points": int(
                    prov["filter"]["active_points"]),
            }

            # control-member Vr innovation (no filter, direct H(x)).
            control_state = snapshots[CONTROL]
            u_e, v_n, w_m = member_earth_winds(
                control_state, grid_rotation(grid_h),
                where="control snapshot")
            from gpuwm.da.obs_radar import (beam_unit_vectors,
                                            simulated_radial_velocity)
            # Every radar in the file, not radar 0.  Each antenna has its
            # own beam geometry, so a radial-velocity innovation is only
            # defined per radar; indexing [0] and calling the answer "the"
            # control innovation was harmless while every file held one
            # radar and silently wrong the moment one held twenty.  The
            # aggregate below pools the residuals, which is the same
            # statistic when there is one radar -- so single-radar
            # receipts keep the numbers they had.
            radars = list(document["radars"])
            per_radar = []
            pooled = []
            from gpuwm.obs.radar_grid import radar_plane
            for index, radar in enumerate(radars):
                # radar_plane, not variables[...][index]: on a v2 file the
                # stored plane covers only this radar's reach window, and
                # the control innovation is computed against a whole-domain
                # wind field.  On v1 it is the same array it always was.
                vr_mask = np.asarray(
                    radar_plane(document, "vr_mask", index)).astype(bool)
                points = int(vr_mask.sum())
                row = {"radar": str(radar["id"]), "points": points}
                if points:
                    vr_obs = np.asarray(
                        radar_plane(document, "vr_obs", index), np.float64)
                    sim = simulated_radial_velocity(
                        u_e, v_n, w_m, beam_unit_vectors(document, index))
                    d = vr_obs[vr_mask] - sim[vr_mask]
                    pooled.append(d)
                    row["innovation_mean_ms"] = float(d.mean())
                    row["innovation_rms_ms"] = float(
                        np.sqrt(np.mean(d ** 2)))
                per_radar.append(row)
            alld = (np.concatenate(pooled) if pooled
                    else np.zeros(0, np.float64))
            leg_record["analysis"]["control_vr"] = {
                # The count that says how much of the network this leg
                # actually saw, beside the innovation it produced.
                "radars": len(radars),
                "radars_with_points": len(pooled),
                "points": int(alld.size),
                "innovation_mean_ms": (float(alld.mean()) if alld.size
                                       else None),
                "innovation_rms_ms": (float(np.sqrt(np.mean(alld ** 2)))
                                      if alld.size else None),
                "per_radar": per_radar,
            }

            if analysis_due:
                overlap: dict[str, dict] = {}
                for index in range(args.members):
                    merged = dict(increments[index])
                    if not args.no_hotstart and index in hot_pending:
                        for field, values in hot_pending[index].items():
                            if field not in merged:
                                merged[field] = values
                                continue
                            # BOTH the filter and the insertion have an
                            # opinion about this field.  Summing double
                            # counts the same reflectivity volume, so
                            # each component's size is reported rather
                            # than one silently overwriting the other.
                            if index == 0:
                                overlap[field] = {
                                    "filter_rms": float(np.sqrt(np.mean(
                                        np.asarray(merged[field],
                                                   np.float64) ** 2))),
                                    "hotstart_rms": float(np.sqrt(np.mean(
                                        np.asarray(values,
                                                   np.float64) ** 2))),
                                }
                            merged[field] = (
                                np.asarray(merged[field], np.float64)
                                + np.asarray(values, np.float64)
                            ).astype(np.float32)
                    pending[index] = merged
                if overlap:
                    leg_record["analysis"]["hotstart_overlap"] = {
                        "fields": sorted(overlap),
                        "member_000_rms": overlap,
                        "rule": "summed; both are increments to the same "
                                "background from the same reflectivity "
                                "volume, so this double counts that "
                                "volume in the overlapping fields",
                    }
                hot_pending.clear()
            if pending.get(0):
                np.savez_compressed(
                    out / f"increments_m000_leg{leg_number(leg)}.npz",
                    **{k: v.astype(np.float32) for k, v
                       in pending[0].items()})
            shutil.rmtree(shm_leg, ignore_errors=True)

        # -- carry the cycle across the process boundary ------------------
        # At the end of the last OBSERVED leg the in-memory state is
        # exactly what the next leg would have consumed: post-leg
        # snapshots plus the analysis increments waiting to be applied.
        # Writing it here (before any free legs run) is what lets the
        # next observation -- which does not exist yet -- be assimilated
        # by a different process without re-initialising the ensemble.
        if save_at_leg is not None and leg == save_at_leg:
            generation = ens_state.write_generation(
                args.save_ensemble, identity=identity,
                elapsed_seconds=t_end, leg_number=leg_number(leg),
                snapshots=snapshots, pending=pending,
                note=("written at the end of the last observed leg; "
                      "any free legs after it are a branch and are not "
                      "part of this ensemble's history"))
            leg_record["ensemble_generation"] = {
                "directory": str(args.save_ensemble),
                "elapsed_seconds": t_end,
                "leg_number": leg_number(leg),
                "written": generation["written"],
            }
            print(f"ensemble generation written to "
                  f"{args.save_ensemble} at {t_end:.0f} s elapsed",
                  flush=True)

        report["legs"].append(leg_record)

        # ---- the treatment proof --------------------------------------
        # Counted, not claimed. An enabled stream that assimilated nothing
        # over the opening cycles stops the run here, before six hours of
        # card time produce a headline naming a stream that never touched
        # the state. The verdict is written into the record either way, so
        # a completed run carries its own proof and a stopped one carries
        # its own reason.
        analyses = [leg["analysis"]["assimilated"]
                    for leg in report["legs"]
                    if isinstance(leg.get("analysis"), dict)
                    and leg["analysis"].get("assimilated") is not None]
        try:
            report["treatment"] = treatment.verify_treatment(
                enabled_obs_kinds, analyses)
        except treatment.TreatmentNotApplied as error:
            report["treatment"] = {
                "label": "full-stack",
                "verdict": "silent_stream",
                "enabled": list(enabled_obs_kinds),
                "error": str(error),
            }
            (out / "cycle-report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            raise SystemExit(f"TREATMENT_NOT_APPLIED: {error}") from error

        (out / "cycle-report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")

    report["total_wall_seconds"] = round(time.time() - t_total, 1)
    (out / "cycle-report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("CYCLE_DRIVER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
