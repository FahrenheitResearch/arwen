#!/usr/bin/env python3
"""Score one arm of one case against the observations, and write its receipt.

The battery's scoring engine, its registration discipline and its observation
readers all shipped as libraries.  What did not ship is the one command that
puts them together, so a shakedown could be run but not scored.  This is that
command.

Two modes, because the useful thing to do before a forecast exists is
different from the useful thing to do after:

``--read-only``
    open the run directory through the mandated science core and report what
    the scorer would see -- the grid, the frames, the column-max reflectivity
    range at one valid time, the station mapping.  Contacts no observation,
    computes no score, and is how the reading half of this tool is exercised
    against a real ``wrfout`` before the case's own run exists.

full scoring
    build the case's registration, read the arm, read the observations, freeze
    the station set against the arm's own terrain and land mask, score, and
    write the score file.

The tool knows no case: the case id, the init instant, the observation
directories and the run directory are all arguments.

**Which registration a score is bound to is an argument too.**  Pass
``--registration`` and the score is scored under the committed campaign
document, whose digest travels in the score file -- and a case or an arm that
document never registered is refused, because scoring a case the rule did not
register is how a case gets added after somebody saw its numbers.  Without it
the tool builds a single-arm document from
:mod:`gpuwm.verify.obs.registration`'s registered defaults, which carries
``rule_status`` ``proposed-unratified``; that is the one-off path, and the
promotion evaluator declines to promote on it, by design.

**The registration is what the defaults are read from, not this file.**  The
leads reported beside the scored ones and the accumulation windows scored come
from the document the run is bound to -- its ``scored_lead_hours``, its
``spinup_policy``, its ``precipitation.window_hours``.  The first real
off-node scoring attempt is why: hardcoded lead and window defaults asked for
forecast hours past the end of the observation archive and for one window
where the rule registers two, and both refusals were the engine correctly
declining to score what it had not been given.  An explicit flag still
overrides; the default now conforms.  The registered coverage floor for frame
selection (amendment v2.1) is read the same way and handed to the reader, so
an upstream ingest outage at one lead is recorded as missing observations
rather than scored as a mask.

``--obs-archive-root`` is the other half of scoring somewhere other than the
box that fetched the bytes: the front doors record an absolute path in every
observation's provenance, and the promotion rule's integrity clause re-hashes
every one of them.  Given one or more roots, each source looks for its object
by name under them instead.  Given none, nothing changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpuwm.obs.sources import (AsosSurfaceSource, MrmsCompositeSource,
                               Stage4PrecipSource)
from gpuwm.verify.obs import battery, model_source, registration, stations

#: The earliest lead that can be reported at all.  Lead 0 is the initial
#: condition written back out: the t=0 history frame carries no ``REFL_10CM``,
#: so there is no forecast field there either to score or to report.  The
#: registration's spin-up policy reports the leads *before* the first scored
#: hour, and this is the floor of that range -- not a choice this tool makes,
#: but where the model's first forecast field exists.
EARLIEST_REPORTABLE_LEAD_HOURS = 1

#: Archive ids exactly as the ingest front doors stamp them into every
#: observation's provenance (see :mod:`gpuwm.obs.sources`).  They route a
#: re-hash back to the source that read the object.  These are archive
#: identities and never case ones.
MRMS_ARCHIVE_ID = "mrms"
STAGE4_ARCHIVE_ID = "stage4"
ASOS_ARCHIVE_ID = "asos"


def _packs(directory: Path, pattern: str = "*.obspack"):
    packs = sorted((directory / "packs").glob(pattern))
    geometry = directory / "packs" / "geometry.obspack"
    frames = [path for path in packs if path != geometry]
    if not frames:
        raise FileNotFoundError(f"{directory} holds no observation packs")
    if not geometry.is_file():
        raise FileNotFoundError(f"{geometry} is missing")
    return frames, geometry


def _hour_list(text: str, *, flag: str) -> list[int]:
    """A comma-separated hour list, in this CLI's own list style."""
    hours: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            hours.append(int(token))
        except ValueError:
            raise ValueError(
                f"{flag} takes whole hours; {token!r} is not one") from None
    if not hours:
        raise ValueError(f"{flag} was given no hours")
    return sorted(set(hours))


def parse_reported_lead_hours(text: str) -> list[int]:
    """An explicitly requested reported-lead set, with lead 0 refused.

    The refusal is not a taste.  The reported block is scored reflectivity
    like any other lead -- the same reader, the same observation match -- and
    at lead 0 the model has no ``REFL_10CM`` to read, because the t=0 history
    frame is the initial condition written back out.  The registration's
    spin-up policy reports the leads before the first scored hour; the
    earliest of those that carries a forecast field is
    :data:`EARLIEST_REPORTABLE_LEAD_HOURS`.
    """
    hours = _hour_list(text, flag="--reported-lead-hours")
    refused = [hour for hour in hours if hour < EARLIEST_REPORTABLE_LEAD_HOURS]
    if refused:
        raise ValueError(
            f"--reported-lead-hours {refused} is refused. The registration's "
            f"spin-up policy reports the leads BEFORE the first scored hour "
            f"and never scores them, and the earliest lead that can be "
            f"reported at all is {EARLIEST_REPORTABLE_LEAD_HOURS}: lead 0 is "
            f"the initial condition written back out, and its history frame "
            f"carries no REFL_10CM to report")
    return hours


def parse_precipitation_window_hours(text: str) -> list[int]:
    """An explicitly requested accumulation-window set."""
    windows = _hour_list(text, flag="--precipitation-window-hours")
    refused = [window for window in windows if window <= 0]
    if refused:
        raise ValueError(
            f"--precipitation-window-hours {refused} is refused: an "
            f"accumulation window is a positive number of hours")
    return windows


def registered_reported_lead_hours(document: Mapping[str, object]) -> list[int]:
    """The leads THIS registration reports and never scores.

    The document names exactly one lead policy beside ``scored_lead_hours``:
    its ``spinup_policy``, *"leads before the first scored hour are reported
    separately and never scored"*.  That sentence is the reported set, so it
    is read from the document rather than typed in here.  A hardcoded default
    asked, on the first real off-node run, for forecast leads past the end of
    the observation archive, and the scorer refused on a frame that does not
    exist -- correctly, and for a set the rule never asked for.

    Leads the document scores cannot appear here by construction (the engine
    refuses a lead that is both), and neither can lead 0.
    """
    scored = sorted(int(hour) for hour in
                    document["parameters"]["scored_lead_hours"])
    if not scored:
        raise ValueError("the registration scores no lead hours")
    return list(range(EARLIEST_REPORTABLE_LEAD_HOURS, scored[0]))


def registered_precipitation_window_hours(
        document: Mapping[str, object]) -> list[int]:
    """Every accumulation window THIS registration scores."""
    windows = [int(hour) for hour in
               document["parameters"]["precipitation"]["window_hours"]]
    if not windows:
        raise ValueError("the registration scores no accumulation window")
    return windows


def registered_frame_coverage_floor(document: Mapping[str, object]) -> float:
    """The coverage floor THIS registration selects frames under.

    Amendment v2.1 registered it; a document written before that amendment
    does not carry the pin, and the registered default applies, which is
    what the amendment says and why it is an amendment rather than a new
    parameter somebody may or may not have set.
    """
    return float(document["parameters"]["reflectivity"].get(
        "frame_coverage_floor", registration.DEFAULT_FRAME_COVERAGE_FLOOR))


def precipitation_sources(directory: Path, windows: Sequence[int]
                          ) -> dict[int, Stage4PrecipSource]:
    """One Stage-IV source per window, out of one archive directory.

    The engine scores every window the registration names and refuses when a
    source is missing for one of them -- which is right, and is what a
    single-window flag walked into: the rule registers a 1 h and a 6 h
    accumulation, both are decoded in the kit, and only the command asked for
    one of them.
    """
    built: dict[int, Stage4PrecipSource] = {}
    for window in windows:
        hours = int(window)
        frames, geometry = _packs(directory, f"*_{hours:02d}h_*.obspack")
        built[hours] = Stage4PrecipSource(frames, geometry,
                                          accumulation_hours=hours)
    return built


class ObservationRehash:
    """Re-hash every scored observation, through the source that read it.

    :func:`gpuwm.verify.obs.battery.score_case_arm` takes one callable and
    applies it to the provenance of every observation it scored -- radar,
    precipitation and surface alike -- so routing each record back to its own
    source happens here rather than there.  Routing is by the archive id the
    provenance carries; a record naming an archive this run did not read is
    still re-hashed by another source, because every source implements the
    same digest round trip and an unperformed check counts as a failed one.

    ``roots`` is what makes scoring possible anywhere but the box that
    fetched the bytes.  The front doors record an absolute path, and
    ``verify`` will look for the object by its own basename under a root
    instead -- but only when it is told one, because a search that happens on
    its own will one day find a different file of the same name.  A root may
    be given more than once: the archive keeps radar, precipitation and
    surface objects in separate directories, and one flat root cannot name
    all three.  A record is clean when a directory the operator named holds
    bytes with the *registered digest*; a same-named file that is not that
    object fails under every root, which is what makes this a relocation and
    not a widening.  With no root at all the behaviour is the old one
    exactly: the recorded path, or a raised wiring fault.
    """

    def __init__(self, sources: Mapping[str, object], *,
                 roots: Sequence[Path] = ()):
        self._sources = {str(key): value
                         for key, value in dict(sources).items()
                         if value is not None}
        if not self._sources:
            raise ValueError(
                "a re-hash needs at least one observation source to ask")
        self._roots = [Path(root) for root in roots]

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def _source_for(self, provenance):
        archive = str(getattr(provenance, "source", "") or "")
        if archive in self._sources:
            return self._sources[archive]
        return next(iter(self._sources.values()))

    def __call__(self, provenance) -> bool:
        source = self._source_for(provenance)
        if not self._roots:
            return bool(source.verify(provenance))
        found = False
        for root in self._roots:
            try:
                matched = bool(source.verify(provenance, root=root))
            except FileNotFoundError:
                continue
            found = True
            if matched:
                return True
        if found:
            return False
        uri = getattr(provenance, "uri", provenance)
        raise FileNotFoundError(
            f"cannot re-hash {uri}: it is not at the path its provenance "
            f"records, and no --obs-archive-root "
            f"{[str(root) for root in self._roots]} holds an object of that "
            f"name")


def read_only(arm) -> dict[str, object]:
    """What the scorer would see, without scoring anything."""
    grid = arm.grid()
    times = arm.valid_times()
    sample = times[len(times) // 2]
    reflectivity = np.asarray(arm.composite_reflectivity(sample),
                              dtype=np.float64)
    finite = np.isfinite(reflectivity)
    locator = arm.station_locator()
    centre = locator("grid-centre",
                     float(grid.latitude[grid.shape[0] // 2,
                                         grid.shape[1] // 2]),
                     float(grid.longitude[grid.shape[0] // 2,
                                          grid.shape[1] // 2]))
    land = arm.land_mask()
    return {
        "land_fraction": float(np.count_nonzero(land) / land.size),
        "frames": len(times),
        "first_valid_time": times[0],
        "last_valid_time": times[-1],
        "grid_shape": list(grid.shape),
        "dx_m": grid.dx_m,
        "latitude_range": [float(grid.latitude.min()),
                           float(grid.latitude.max())],
        "longitude_range": [float(grid.longitude.min()),
                            float(grid.longitude.max())],
        "terrain_available": grid.terrain_m is not None,
        "terrain_range_m": (None if grid.terrain_m is None else
                            [float(np.min(grid.terrain_m)),
                             float(np.max(grid.terrain_m))]),
        "sampled_valid_time": sample,
        "composite_reflectivity_finite_cells": int(np.count_nonzero(finite)),
        "composite_reflectivity_range_dbz": [
            float(np.min(reflectivity[finite])),
            float(np.max(reflectivity[finite]))],
        "grid_centre_station_position": {
            "x": centre.x, "y": centre.y,
            "note": ("the science core's own projection maps the grid centre "
                     "back to the centre index; an off-by-one here is a "
                     "projection disagreement, not a rounding taste")},
    }


def build_registration(*, case_id: str, init_time: str, arm_id: str,
                       evaluator_commit: str, station_table_sha256: str,
                       twin_rung: str) -> dict[str, object]:
    return registration.make_registration(
        evaluator_commit=evaluator_commit,
        reflectivity=registration.reflectivity_parameters(),
        surface=registration.surface_parameters(),
        precipitation=registration.precipitation_parameters(),
        promotion=registration.promotion_parameters(),
        cases=[{"case_id": case_id, "init_time": init_time,
                "station_table_sha256": station_table_sha256}],
        arms=[{"arm_id": arm_id}],
        twin={"rung": twin_rung,
              "perturbation": "one documented FP ULP on the wrfinput theta "
                              "field (tools/n5s/perturb_ulp.py)"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--domain", default="d01")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--arm-id")
    parser.add_argument("--init-time")
    parser.add_argument("--reflectivity-packs", type=Path)
    parser.add_argument("--precipitation-packs", type=Path, default=None,
                        help="a Stage-IV driver output directory")
    parser.add_argument("--precipitation-window-hours", default=None,
                        help="comma-separated accumulation windows; every "
                             "window the registration scores, by default")
    parser.add_argument("--asos-stations", type=Path, default=None)
    parser.add_argument("--asos-surface", type=Path, default=None)
    parser.add_argument("--boundary-width-cells", type=int, default=None)
    parser.add_argument("--reported-lead-hours", default=None,
                        help="comma-separated leads to report and never "
                             "score; the registration's own spin-up leads, "
                             "by default")
    parser.add_argument("--obs-archive-root", type=Path, action="append",
                        default=None, metavar="DIRECTORY",
                        help="a directory holding relocated archive objects, "
                             "repeatable once per directory. Without it the "
                             "re-hash reads the absolute paths the front "
                             "doors recorded, which exist only on the box "
                             "that fetched the bytes")
    parser.add_argument("--twin-rung", default="rung-1-one-ulp-theta")
    parser.add_argument("--registration", type=Path, default=None,
                        help="score against a COMMITTED registration document "
                             "(the campaign's own) rather than building a "
                             "single-arm one; the case must be in its case set")
    parser.add_argument("--evaluator-commit")
    parser.add_argument("--registration-out", type=Path, default=None)
    parser.add_argument("--score-out", type=Path, default=None)
    arguments = parser.parse_args()

    # Explicit lead and window sets are checked before anything is opened: a
    # request for a lead that cannot exist is a usage error, and it should
    # not cost a run directory read to hear so.
    explicit_reported: list[int] | None = None
    explicit_windows: list[int] | None = None
    try:
        if arguments.reported_lead_hours is not None:
            explicit_reported = parse_reported_lead_hours(
                arguments.reported_lead_hours)
        if arguments.precipitation_window_hours is not None:
            explicit_windows = parse_precipitation_window_hours(
                arguments.precipitation_window_hours)
    except ValueError as error:
        parser.error(str(error))

    arm = model_source.WrfHistorySource(arguments.run_directory,
                                        domain=arguments.domain)
    if arguments.read_only:
        record = {"schema": "gpuwm.obs-battery-arm-readback/v1",
                  "run_directory": str(arguments.run_directory),
                  "reader": arm.record(),
                  **read_only(arm)}
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    required = ["case_id", "arm_id", "init_time", "reflectivity_packs",
                "boundary_width_cells"]
    if arguments.registration is None:
        required.append("evaluator_commit")
    for name in required:
        if getattr(arguments, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required when scoring")

    # The reflectivity packs are indexed here and the source is CONSTRUCTED
    # below, once the registration that carries its coverage floor exists.
    frames, geometry = _packs(arguments.reflectivity_packs)

    # The station table is read here for its digest, which the one-off
    # registration carries; the station SET is frozen further down, once the
    # document that says which hours it must report on exists.
    surface_source = None
    station_table_sha256 = ""
    if arguments.asos_surface is not None:
        if arguments.asos_stations is None:
            parser.error("--asos-surface needs --asos-stations")
        table = json.loads(
            arguments.asos_stations.read_text(encoding="utf-8"))
        station_table_sha256 = str(table.get("content_sha256", ""))
        surface_source = AsosSurfaceSource(arguments.asos_surface)

    if arguments.registration is not None:
        # Scoring against the committed campaign registration is what binds a
        # score to the rule it was scored under: the score file carries that
        # document's own digest, so a score published beside a different
        # registration stops matching. Building a fresh single-arm document
        # is the fallback for a one-off, and it says so in its rule_status.
        document = registration.validate_registration(json.loads(
            arguments.registration.read_text(encoding="utf-8")))
        known = {str(case["case_id"])
                 for case in document["parameters"]["cases"]}
        if str(arguments.case_id) not in known:
            parser.error(
                f"{arguments.case_id!r} is not in the registration's case set "
                f"{sorted(known)}; scoring a case the rule never registered "
                f"is how a case gets added after seeing its numbers")
        arms = {str(arm["arm_id"]) for arm in document["parameters"]["arms"]}
        if str(arguments.arm_id) not in arms:
            parser.error(
                f"{arguments.arm_id!r} is not a registered arm {sorted(arms)}")
        print(f"registration {document['registration_sha256']} "
              f"({document['rule_status']}) from {arguments.registration}")
    else:
        document = build_registration(
            case_id=arguments.case_id, init_time=arguments.init_time,
            arm_id=arguments.arm_id,
            evaluator_commit=arguments.evaluator_commit,
            station_table_sha256=station_table_sha256,
            twin_rung=arguments.twin_rung)
    if arguments.registration_out is not None:
        arguments.registration_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.registration_out.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"registration {document['registration_sha256']} -> "
              f"{arguments.registration_out}")

    # Everything below reads its lead and window sets off the document above.
    scored_leads = [int(hour) for hour in
                    document["parameters"]["scored_lead_hours"]]
    reported = (explicit_reported if explicit_reported is not None
                else registered_reported_lead_hours(document))
    windows = (explicit_windows if explicit_windows is not None
               else registered_precipitation_window_hours(document))
    coverage_floor = registered_frame_coverage_floor(document)
    print(f"leads: scored {scored_leads[0]}..{scored_leads[-1]}, "
          f"reported-not-scored {reported}; "
          f"precipitation windows {windows} h; "
          f"frame coverage floor {coverage_floor}")

    reflectivity_obs = MrmsCompositeSource(
        frames, geometry, minimum_observed_fraction=coverage_floor)

    precipitation_obs = None
    if arguments.precipitation_packs is not None:
        precipitation_obs = precipitation_sources(
            arguments.precipitation_packs, windows)

    station_obs = None
    frozen = None
    if surface_source is not None:
        hours = battery.valid_times(arguments.init_time, scored_leads)
        station_obs = surface_source.observations(hours)
        grid = arm.grid()
        if grid.terrain_m is None:
            raise ValueError(
                "the arm's reader returned no terrain, and the registered "
                "station admission compares model terrain with station "
                "elevation; scoring the surface without it would drop the "
                "rule rather than apply it")
        locator = arm.station_locator()
        positions = {station.station_id: locator(station.station_id,
                                                 station.latitude,
                                                 station.longitude)
                     for station in station_obs.stations}
        region = battery.interior_mask(
            grid.shape,
            boundary_width_cells=int(arguments.boundary_width_cells),
            rim_m=registration.DEFAULT_INTERIOR_RIM_M, dx_m=grid.dx_m)
        frozen = stations.freeze_station_set(
            station_obs.stations, positions, observations=station_obs,
            valid_times=hours, interior_mask=region,
            land_mask=arm.land_mask(), terrain_m=grid.terrain_m,
            elevation_tolerance_m=registration.DEFAULT_ELEVATION_TOLERANCE_M,
            minimum_reporting_fraction=(
                registration.DEFAULT_MINIMUM_REPORTING_FRACTION),
            match_tolerance_seconds=(
                registration.DEFAULT_MATCH_TOLERANCE_SECONDS),
            maximum_screen_fraction=(
                registration.DEFAULT_MAXIMUM_SCREEN_FRACTION))
        print(f"frozen station set: {len(frozen.station_ids)} kept, "
              f"{len(frozen.drops)} dropped")

    # One re-hash over every source this run read, not the radar's alone: the
    # integrity clause covers every scored observation, and the relocation
    # root has to reach all of them or scoring stays bound to one box.  The
    # windows differ in what they read, not in how they re-hash, so one
    # Stage-IV source answers for that archive.
    observation_sources: dict[str, object] = {
        MRMS_ARCHIVE_ID: reflectivity_obs,
        STAGE4_ARCHIVE_ID: (next(iter(precipitation_obs.values()))
                            if precipitation_obs else None),
        ASOS_ARCHIVE_ID: surface_source,
    }
    rehash = ObservationRehash(observation_sources,
                               roots=arguments.obs_archive_root or ())
    if rehash.roots:
        print("re-hash under relocated archive roots: "
              + ", ".join(str(root) for root in rehash.roots))

    payload = battery.score_case_arm(
        registration=document, case_id=arguments.case_id,
        arm_id=arguments.arm_id, init_time=arguments.init_time, model=arm,
        reflectivity_obs=reflectivity_obs,
        boundary_width_cells=int(arguments.boundary_width_cells),
        station_obs=station_obs, frozen_stations=frozen,
        precipitation_obs=precipitation_obs, reported_lead_hours=reported,
        rehash=rehash)

    primary = payload["reflectivity"]["primary_scalar"]
    print(f"{arguments.case_id} / {arguments.arm_id}: S_refl = {primary:.4f}")
    if arguments.score_out is not None:
        battery.write_score_file(arguments.score_out, payload)
        print(f"score file -> {arguments.score_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
