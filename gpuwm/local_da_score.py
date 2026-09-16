"""Give every local DA run a nowcast skill number, by default.

A local DA window ends with a short forecast and a folder of pictures.  What
it did not end with was an answer to the only question a user actually has:
how close was that forecast to the radar an hour later.  The engine already
held the referee -- masked neighbourhood FSS against the MRMS composite -- and
nothing connected it to a run.  This module is that connection.

What it does, whenever a window's forecast completes:

* reads the forecast's own composite reflectivity at each registered lead
  (15, 30, 45, 60 minutes after the analysis) the forecast actually reached;
* fetches the MRMS composite nearest each of those instants into a cache
  inside the case, so a rescore refetches nothing;
* scores model against radar, and radar-persistence against radar, on the
  identical cells, and records the difference;
* writes ``nowcast-score.json`` beside the window's own receipts, and hands
  back the compact per-lead rows the window receipt, the run receipt and the
  continuous status document all carry.

**The run is never at the mercy of the score.**  Every failure mode here is
recorded, not raised: no network, no front door, an archive that has not
published the scan yet, a lead whose valid time is still in the future.  A
forecast that ran is a forecast that completed, and a scoring problem that
failed the run would make the skill number cost more than it is worth.

**Latency is the normal case, not an error.**  At real time the 60 minute
lead cannot be scored until an hour after the analysis.  So a lead is left
``pending`` with its reason and the receipt is rewritten later: the continuous
controller rescores every earlier window as each new window completes, and
``gpuwm local-da --score PLAN`` does the same on demand.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

#: The receipt one window writes, beside that window's own decisions.
RECEIPT_NAME = "nowcast-score.json"

#: What the run receipt and the status document carry: the schema, the path
#: to the full receipt, and one compact row per lead.
SUMMARY_SCHEMA = "arwen.local-da-nowcast-summary.v1"

#: What a window's compact summary says happened.  ``scored`` means at least
#: one lead carries a number; the other three mean none does and say why, in
#: the same words a lead row uses.  A summary that said ``scored`` because a
#: receipt exists would read as a score to anyone skimming a status document,
#: which is the one thing this must not do.
SUMMARY_SCORED = "scored"
SUMMARY_PENDING = "pending"
SUMMARY_MISSING_OBS = "missing-obs"
SUMMARY_UNAVAILABLE = "unavailable"
SUMMARY_STATUSES = (SUMMARY_SCORED, SUMMARY_PENDING, SUMMARY_MISSING_OBS,
                    SUMMARY_UNAVAILABLE)

_SEAM_TIME = "%Y-%m-%dT%H:%M:%S"


def _seam(value) -> str:
    """The seam's spelling of an instant, from anything the run carries."""
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        moment = datetime.fromisoformat(text)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.strftime(_SEAM_TIME)


class ForecastLeads:
    """One window's short forecast, read at the leads it reached.

    The composite is the column maximum of the stored ``REFL_10CM``, taken
    through :func:`gpuwm.verify.obs.model_source.frame_composite_reflectivity`
    so the nowcast and the free-forecast battery read a forecast the same way.

    One member is scored and the receipt names it.  That is a limitation with
    a reason rather than an oversight: neighbourhood FSS of a column maximum
    is a statistic of one realisation, and the ensemble version of this
    question is a probabilistic score, which is a different instrument.
    """

    def __init__(self, manifest_path, analysis_time: str, *,
                 lead_minutes: Sequence[int], domain: str = "d01",
                 member: int | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.analysis_time = _seam(analysis_time)
        self.domain = str(domain)
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.members_available = int(document.get("n_members", 0)
                                     or len(document.get("members", ())))
        rows = sorted(document.get("members", ()),
                      key=lambda row: int(row["index"]))
        if not rows:
            raise ValueError(f"{self.manifest_path} names no forecast members")
        if member is None:
            chosen = rows[0]
        else:
            chosen = next((row for row in rows
                           if int(row["index"]) == int(member)), None)
            if chosen is None:
                raise ValueError(
                    f"{self.manifest_path} has no member {member}; it names "
                    f"members {[int(row['index']) for row in rows]}")
        self.member = int(chosen["index"])
        root = self.manifest_path.parent / str(chosen["member_dir"])
        self._frames: dict[int, Path] = {}
        self._digests: dict[int, str] = {}
        start = datetime.strptime(self.analysis_time, _SEAM_TIME)
        wanted = {start + timedelta(minutes=int(value)): int(value)
                  for value in lead_minutes}
        from gpuwm.ensemble.wrfout_inventory import WRFOUT_INVENTORY_KEY

        for entry in chosen.get(WRFOUT_INVENTORY_KEY) or []:
            if str(entry.get("domain", self.domain)) != self.domain:
                continue
            for frame in entry.get("frames", ()):
                stamp = datetime.strptime(str(frame["valid_time"]),
                                          "%Y-%m-%d_%H:%M:%S")
                minutes = wanted.get(stamp)
                if minutes is not None:
                    self._frames[minutes] = root / str(entry["path"])
                    self._digests[minutes] = str(entry.get("sha256", ""))
        self._grid = None

    def lead_minutes(self) -> tuple[int, ...]:
        return tuple(sorted(self._frames))

    def frame_path(self, minutes: int) -> Path:
        return self._frames[int(minutes)]

    def composite(self, minutes: int) -> np.ndarray:
        from gpuwm.verify.obs.model_source import frame_composite_reflectivity

        return frame_composite_reflectivity(self.frame_path(minutes))

    def grid(self):
        """Latitude, longitude and spacing, off the forecast's own frame."""
        from gpuwm import netcdf_bridge
        from gpuwm.verify.obs.contracts import ModelGrid, normalize_longitude

        if self._grid is None:
            leads = self.lead_minutes()
            if not leads:
                raise ValueError(
                    "this forecast reached none of the registered nowcast "
                    "leads, so it has no grid to score on")
            with netcdf_bridge.open_dataset(self.frame_path(leads[0])) as data:
                latitude = np.asarray(data.variables["XLAT"][0],
                                      dtype=np.float64)
                longitude = np.asarray(data.variables["XLONG"][0],
                                       dtype=np.float64)
                dx = float(data.getncattr("DX"))
            self._grid = ModelGrid(latitude=latitude,
                                   longitude=normalize_longitude(longitude),
                                   dx_m=dx)
        return self._grid

    def record(self) -> dict[str, object]:
        from gpuwm.verify.obs import model_source

        return {
            "route": "the window's own ensemble forecast output",
            "forecast_manifest": str(self.manifest_path),
            "domain": self.domain,
            "member_scored": self.member,
            "members_available": self.members_available,
            "member_policy": (
                "the lowest-numbered member is scored and named; the other "
                "members are not, because neighbourhood FSS of a column "
                "maximum is a statistic of one realisation and the ensemble "
                "form of this question is a probabilistic score, which is a "
                "different instrument"),
            "reflectivity_variable": model_source.DEFAULT_REFLECTIVITY_VARIABLE,
            "reflectivity_reduction": model_source.COMPOSITE_REDUCTION,
            "frames": {str(minutes): {
                "path": str(self._frames[minutes]),
                "sha256": self._digests.get(minutes, "")}
                for minutes in self.lead_minutes()},
        }


class CachedCompositeSource:
    """The MRMS composite, fetched on demand into the case's own cache.

    Satisfies the seam a scorer reads: ``field(valid_time)`` answers with an
    :class:`~gpuwm.verify.obs.contracts.ObsGridField` or raises the three
    distinguishable failures the nowcast registration records separately.
    """

    def __init__(self, cache, *, match_seconds: int,
                 coverage_floor: float) -> None:
        self.cache = cache
        self.match_seconds = int(match_seconds)
        self.coverage_floor = float(coverage_floor)
        self._source = None
        self._packs: tuple[str, ...] = ()
        self._details: dict[str, dict[str, object]] = {}
        self._walk_problem: dict[str, str] = {}

    def _rebuilt(self):
        from gpuwm.obs.sources import MrmsCompositeSource

        packs = tuple(str(path) for path in self.cache.pack_paths())
        if not packs:
            raise LookupError(
                "the case's observation cache holds no composite packs")
        if self._source is None or packs != self._packs:
            self._source = MrmsCompositeSource(
                [Path(value) for value in packs], self.cache.geometry_path,
                match_seconds=self.match_seconds,
                minimum_observed_fraction=self.coverage_floor)
            self._packs = packs
        return self._source

    def field(self, valid_time: str):
        from gpuwm.obs.mrms_fetch import MrmsArchiveUnavailable
        from gpuwm.verify.obs.contracts import ObservedFractionBelowFloor
        from gpuwm.verify.obs.nowcast import ObservationsUnreachable

        walked: list[object] = []
        try:
            frame = self.cache.ensure(valid_time)
        except MrmsArchiveUnavailable as error:
            raise ObservationsUnreachable(str(error)) from error
        if frame.observed_fraction < self.coverage_floor:
            # The registered selection walks outward inside the same window
            # for a better-covered frame; it can only walk over frames that
            # are on disk.
            try:
                walked = list(self.cache.ensure_window(valid_time))
            except MrmsArchiveUnavailable as error:
                # An offline run refuses the listing outright, and so does a
                # box that cannot reach the archive.  The nearest frame came
                # back a line ago and is on disk: score it rather than throw
                # the lead away, and say on the lead that the outward walk
                # did not run.  The alternative recorded `unavailable` for a
                # scan the case already held.
                self._walk_problem[frame.valid_time] = (
                    "the poorly covered nearest scan was scored because the "
                    f"frames around it could not be listed: {error}")
        # Every frame the walk pulled is recorded, not only the one ensure()
        # picked: the registered selection can score any of them, and a lead
        # scored on a neighbour used to lose the archive identity of the
        # object it was actually scored from.
        for candidate in [frame, *walked]:
            self._details[candidate.valid_time] = {
                "archive_bucket": candidate.bucket,
                "archive_key": candidate.key,
                "archive_object_uri": candidate.object_uri,
                "archive_object_sha256": candidate.object_sha256,
                "pack_path": candidate.pack_path,
                "pack_sha256": candidate.pack_sha256,
                "observed_fraction": candidate.observed_fraction,
            }
        try:
            return self._rebuilt().field(valid_time)
        except ObservedFractionBelowFloor:
            raise
        except MrmsArchiveUnavailable as error:  # pragma: no cover - defensive
            raise ObservationsUnreachable(str(error)) from error

    def scan_detail(self, valid_time: str) -> dict[str, object]:
        detail = dict(self._details.get(str(valid_time), {}))
        problem = self._walk_problem.get(str(valid_time))
        if problem:
            detail["coverage_walk"] = problem
        return detail

    def record(self) -> dict[str, object]:
        return self.cache.record()


def boundary_width_cells(case_root) -> int | None:
    """The specified plus relaxation rows this case was configured with.

    Read off the case's own ``experiment.toml`` through the configuration
    owner rather than assumed: the registered interior mask excludes those
    rows, and a mask that excluded the DEFAULT number while the run used
    another would score cells the lateral boundary was still driving and
    never say so.  ``None`` when the case cannot be read, and the caller
    then registers the default and the receipt says which it used.
    """
    path = Path(case_root) / "experiment.toml"
    if not path.is_file():
        return None
    try:
        from gpuwm.experiment import load_experiment

        run = load_experiment(path).domains[0].run
        return int(run.spec_zone) + int(run.relax_zone)
    except (OSError, ValueError, KeyError, IndexError, AttributeError,
            ImportError):
        return None


def registration_for(plan: Mapping[str, object] | None = None, *,
                     case_root=None, boundary_width: int | None = None):
    """The registration a run scores under: the defaults, plus the run's mask.

    Only one pin depends on the case: the boundary width, which is the
    specified and relaxation rows that run was configured with.  Everything
    else -- leads, thresholds, boxes, the primary scalar, the tolerance, the
    coverage floor, the rim -- is fixed before any score is looked at.
    """
    from gpuwm.verify.obs import nowcast

    width = boundary_width
    if width is None and case_root is not None:
        width = boundary_width_cells(case_root)
    parameters = nowcast.nowcast_parameters(
        **({} if width is None else {"boundary_width_cells": int(width)}))
    return nowcast.make_nowcast_registration(
        evaluating_tree=nowcast.evaluating_tree_record(),
        parameters=parameters)


def _unavailable(analysis_time: str, reason: str, receipt_path: Path | None
                 ) -> dict[str, object]:
    return {"schema": SUMMARY_SCHEMA, "status": SUMMARY_UNAVAILABLE,
            "analysis_time": analysis_time, "reason": reason,
            "receipt_path": None if receipt_path is None else str(receipt_path),
            "leads": [], "leads_pending": [], "primary": None}


def summary_status(document: Mapping[str, object]) -> str:
    """What the window as a whole managed, in one word."""
    if document.get("leads_scored"):
        return SUMMARY_SCORED
    if document.get("leads_unavailable"):
        return SUMMARY_UNAVAILABLE
    if document.get("leads_pending"):
        return SUMMARY_PENDING
    return SUMMARY_MISSING_OBS


def summarize(document: Mapping[str, object], receipt_path) -> dict[str, object]:
    """The compact form a run receipt and a status document carry."""
    from gpuwm.verify.obs import nowcast

    return {
        "schema": SUMMARY_SCHEMA,
        "status": summary_status(document),
        "receipt_schema": document["schema"],
        "receipt_path": str(receipt_path),
        "analysis_time": document["analysis_time"],
        "primary": document["primary"]["statistic"],
        "registration_sha256": document["registration_sha256"],
        "leads": nowcast.lead_summary(document),
        "leads_scored": list(document["leads_scored"]),
        "leads_pending": list(document["leads_pending"]),
        "leads_missing_obs": list(document["leads_missing_obs"]),
        "leads_unavailable": list(document["leads_unavailable"]),
        "scored_utc": document["scored_utc"],
    }


def write_receipt(path, document: Mapping[str, object]) -> Path:
    """The receipt is rewritten in place as pending leads are filled in."""
    from gpuwm.ensemble.manifest import write_json_atomically

    write_json_atomically(Path(path), dict(document))
    return Path(path)


def read_receipt(path):
    target = Path(path)
    if not target.is_file():
        return None
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    from gpuwm.verify.obs.nowcast import NOWCAST_SCORE_SCHEMA

    return document if document.get("schema") == NOWCAST_SCORE_SCHEMA else None


def has_open_leads(document: Mapping[str, object] | None) -> bool:
    """Is there anything left to score in this receipt?"""
    if document is None:
        return True
    return bool(document.get("leads_pending")
                or document.get("leads_unavailable"))


def score_window(*, plan: Mapping[str, object], receipt_path,
                 analysis_time, forecast_manifest, cache_root,
                 case_root=None, now: datetime | None = None,
                 offline: bool = False,
                 member: int | None = None) -> dict[str, object]:
    """Score one window's forecast and write its receipt.

    Returns the compact summary.  Never raises for a scoring problem: a run
    that produced a forecast has produced a forecast, and this is a number
    about it, not a gate on it.
    """
    from gpuwm.obs import mrms_fetch
    from gpuwm.verify.obs import nowcast

    stamp = _seam(analysis_time)
    receipt_path = Path(receipt_path)
    existing = read_receipt(receipt_path)
    if existing is not None and not has_open_leads(existing):
        return summarize(existing, receipt_path)
    try:
        registration = registration_for(
            plan, case_root=case_root or Path(cache_root).parent)
        parameters = registration["parameters"]
        leads = ForecastLeads(forecast_manifest, stamp,
                              lead_minutes=parameters["lead_minutes"],
                              member=member)
        if not leads.lead_minutes():
            return _unavailable(
                stamp,
                "this forecast reached none of the registered nowcast leads, "
                "so there is nothing to score against the radar",
                receipt_path)
        grid = leads.grid()
        cache = mrms_fetch.MrmsCompositeCache(
            cache_root,
            bbox=mrms_fetch.bbox_around(grid.latitude, grid.longitude),
            window_seconds=int(parameters["obs_match_tolerance_seconds"]),
            offline=offline)
        observations = CachedCompositeSource(
            cache, match_seconds=int(parameters["obs_match_tolerance_seconds"]),
            coverage_floor=float(parameters["frame_coverage_floor"]))
        document = nowcast.score_nowcast(
            registration=registration, analysis_time=stamp, model=leads,
            observations=observations, grid=grid, now=now)
    except Exception as error:
        # Deliberately everything. A forecast that ran is a forecast that ran,
        # and a scoring fault that failed the run would make the skill number
        # cost more than it is worth. Nothing is swallowed: the exception type
        # and message go into the summary, so a reader is told what broke
        # rather than shown a window with no score and no explanation.
        return _unavailable(
            stamp, f"{type(error).__name__}: {error}", receipt_path)
    write_receipt(receipt_path, document)
    return summarize(document, receipt_path)


def _window_directories(root: Path) -> list[Path]:
    return sorted((root / "continuous").glob("window_*"))


def _recorded_manifest(execution_path: Path):
    if not execution_path.is_file():
        return None
    try:
        document = json.loads(execution_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    manifest = document.get("forecast_manifest")
    return Path(manifest) if manifest else None


def _forecast_manifest(directory: Path, case_root: Path):
    """The forecast this window's score reads, preferring the case's own copy.

    An execution receipt records an absolute path, so a case that was copied
    to another box or another directory names a manifest that either is not
    there or -- worse -- is the ORIGINAL case's, still readable on the same
    machine. Scoring the original while claiming to score the copy is a wrong
    number that looks right, so a recorded path outside this case root yields
    to the manifest sitting inside it.
    """
    candidates = [path / "ensemble-manifest.json"
                  for path in sorted(directory.glob("forecast*"))]
    local = next((path for path in candidates if path.is_file()), None)
    recorded = _recorded_manifest(directory / "execution.json")
    if recorded is not None and recorded.is_file():
        try:
            inside = recorded.resolve().is_relative_to(case_root.resolve())
        except (OSError, ValueError):
            inside = False
        if inside or local is None:
            return recorded
    return local


def cache_root_for(root: Path) -> Path:
    """Where a case keeps the scans it was scored against.

    Inside the case, so the observations travel with the run and a rescore
    reads the same bytes instead of asking the archive again.
    """
    return Path(root) / "nowcast-observations"


def score_case(plan_path, *, now: datetime | None = None,
               offline: bool = False, member: int | None = None,
               only_open: bool = True) -> dict[str, object]:
    """Score every lead of a saved plan that can be scored now.

    This is the door's body: it validates the saved plan the way a launch
    does, then walks the case's completed windows, scores the leads that are
    newly reachable, and leaves the rest pending with their reasons.
    """
    from gpuwm.local_da_runtime import read_plan

    path = Path(plan_path).resolve()
    return score_case_root(read_plan(path), path.parent, plan_path=path,
                           now=now, offline=offline, member=member,
                           only_open=only_open)


def score_case_root(plan: Mapping[str, object], root, *, plan_path=None,
                    now: datetime | None = None, offline: bool = False,
                    member: int | None = None, only_open: bool = True
                    ) -> dict[str, object]:
    """The walk itself, for a caller that already holds the reviewed plan.

    The running controller is that caller: it has the plan in memory and
    re-reading it between windows would re-hash every published file for no
    new fact.
    """
    root = Path(root)
    path = Path(plan_path) if plan_path else root / "local-da.json"
    cache = cache_root_for(root)
    windows: list[dict[str, object]] = []

    directories = _window_directories(root)
    if directories:
        for directory in directories:
            complete = directory / "complete.json"
            if not complete.is_file():
                continue
            try:
                committed = json.loads(complete.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            manifest = _forecast_manifest(directory, root)
            if manifest is None:
                continue
            receipt = directory / RECEIPT_NAME
            # Read once: the second read used to happen after the decision
            # the first one made, so a receipt that went away between them
            # reached summarize() as None.
            settled = read_receipt(receipt)
            if only_open and not has_open_leads(settled):
                windows.append(dict(window=int(committed["index"]),
                                    **summarize(settled, receipt)))
                continue
            summary = score_window(
                plan=plan, receipt_path=receipt,
                analysis_time=committed["analysis_utc"],
                forecast_manifest=manifest, cache_root=cache, case_root=root,
                now=now, offline=offline, member=member)
            windows.append(dict(window=int(committed["index"]), **summary))
    else:
        manifest = _forecast_manifest(root, root)
        receipt = root / RECEIPT_NAME
        if manifest is None:
            windows.append(dict(window=0, **_unavailable(
                _seam(plan["analysis_times"][-1]),
                "this case has no completed forecast to score", receipt)))
        else:
            summary = score_window(
                plan=plan, receipt_path=receipt,
                analysis_time=plan["analysis_times"][-1],
                forecast_manifest=manifest, cache_root=cache, case_root=root,
                now=now, offline=offline, member=member)
            windows.append(dict(window=0, **summary))

    return {
        "schema": SUMMARY_SCHEMA,
        "plan_path": str(path),
        "review_sha256": plan["review_sha256"],
        "cache_root": str(cache),
        "primary": next((row["primary"] for row in windows
                         if row.get("primary")), None),
        "windows": windows,
        "open_leads": sum(len(row.get("leads_pending", ()))
                          + len(row.get("leads_unavailable", ()))
                          for row in windows),
    }


def status_scores(root) -> dict[str, object] | None:
    """The live per-window scores a continuous status document publishes.

    Read off the receipts rather than off a snapshot taken when a window
    completed, because a later window's pass fills in an earlier window's
    pending leads and the status is supposed to be current.
    """
    root = Path(root)
    windows: list[dict[str, object]] = []
    for directory in _window_directories(root):
        document = read_receipt(directory / RECEIPT_NAME)
        if document is None:
            continue
        index = int(str(directory.name).split("_")[-1])
        windows.append(dict(window=index,
                            **summarize(document, directory / RECEIPT_NAME)))
    if not windows:
        return None
    return {
        "schema": SUMMARY_SCHEMA,
        "primary": windows[-1]["primary"],
        "windows": windows,
        "open_leads": sum(len(row["leads_pending"]) + len(row["leads_unavailable"])
                          for row in windows),
    }


__all__ = [
    "CachedCompositeSource", "ForecastLeads", "RECEIPT_NAME",
    "SUMMARY_SCHEMA", "SUMMARY_STATUSES", "boundary_width_cells",
    "cache_root_for", "has_open_leads", "read_receipt", "summary_status",
    "registration_for", "score_case", "score_case_root", "score_window",
    "status_scores", "summarize", "write_receipt",
]
