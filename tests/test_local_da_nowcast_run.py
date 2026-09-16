"""The nowcast score reaches a local DA run by default, and the door to it.

The instrument is tested in ``tests/test_obs_nowcast_score.py``.  This file
tests the wiring: that an ordinary run and a continuous window each
write a receipt with no flag and publish the summary they wrote,
that the receipt, the window's completion record and the status document all
carry the same three numbers per lead, that a lead nobody can score yet is
pending rather than zero, that an offline run leaves the run untouched and
says why, and that the door exists and is documented.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import local_da_score
from gpuwm.obs import mrms_fetch
from gpuwm.verify.obs import nowcast

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = "2026-09-10T12:15:00"
SHAPE = (48, 48)
DX_M = 3000.0


def _geometry():
    degrees_lat = DX_M / 111000.0
    degrees_lon = DX_M / (111000.0 * np.cos(np.deg2rad(35.2)))
    rows = 35.2 + (np.arange(SHAPE[0]) - SHAPE[0] / 2.0) * degrees_lat
    cols = -97.4 + (np.arange(SHAPE[1]) - SHAPE[1] / 2.0) * degrees_lon
    longitude, latitude = np.meshgrid(cols, rows)
    return latitude, longitude


def _storm(row: int, col: int) -> np.ndarray:
    rows, cols = np.indices(SHAPE)
    return np.maximum(48.0 - 6.0 * np.hypot(rows - row, cols - col),
                      -30.0).astype(np.float64)


def _grid():
    from gpuwm.verify.obs.contracts import ModelGrid

    latitude, longitude = _geometry()
    return ModelGrid(latitude=latitude, longitude=longitude, dx_m=DX_M)


class StubModel:
    def __init__(self, composites):
        self._composites = dict(composites)

    def lead_minutes(self):
        return tuple(sorted(self._composites))

    def composite(self, minutes):
        return self._composites[int(minutes)]

    def grid(self):
        return _grid()

    def record(self):
        return {"route": "stub", "member_scored": 0}


class StubObs:
    def __init__(self, frames, *, unreachable=()):
        self._frames = dict(frames)
        self._unreachable = set(unreachable)

    def field(self, valid_time):
        from gpuwm.verify.obs.contracts import ObsGridField, ObsProvenance

        if valid_time in self._unreachable:
            raise nowcast.ObservationsUnreachable(
                "the MRMS archive could not be reached: no network")
        if valid_time not in self._frames:
            raise LookupError(f"no frame at {valid_time}")
        latitude, longitude = _geometry()
        return ObsGridField(
            quantity="composite_reflectivity", valid_time=valid_time,
            values=self._frames[valid_time],
            valid=np.ones(SHAPE, dtype=bool), latitude=latitude,
            longitude=longitude, units="dBZ",
            provenance=ObsProvenance(
                source="mrms", product="MergedReflectivityQCComposite_00.50",
                uri=f"s3://noaa-mrms-pds/{valid_time}.grib2.gz",
                sha256="ab" * 32, fetched_at="2026-09-10T13:30:00"))

    def scan_detail(self, valid_time):
        return {"archive_key": f"CONUS/x/{valid_time}.grib2.gz"}

    def record(self):
        return {"route": "stub"}


def _score(tmp_path, model, observations, *, now=None):
    registration = local_da_score.registration_for(None)
    document = nowcast.score_nowcast(
        registration=registration, analysis_time=ANALYSIS, model=model,
        observations=observations, grid=_grid(),
        now=now or datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc))
    receipt = tmp_path / local_da_score.RECEIPT_NAME
    local_da_score.write_receipt(receipt, document)
    return document, receipt


# -- the receipt a window leaves behind ---------------------------------

def test_the_summary_carries_model_persistence_and_difference_per_lead(tmp_path):
    truth = _storm(24, 24)
    moved = _storm(24, 28)
    observations = StubObs({ANALYSIS: truth,
                            "2026-09-10T12:30:00": moved,
                            "2026-09-10T12:45:00": moved})
    document, receipt = _score(
        tmp_path, StubModel({15: moved, 30: _storm(24, 34)}), observations)
    summary = local_da_score.summarize(document, receipt)
    assert summary["schema"] == local_da_score.SUMMARY_SCHEMA
    assert summary["receipt_schema"] == nowcast.NOWCAST_SCORE_SCHEMA
    assert summary["receipt_path"] == str(receipt)
    assert summary["primary"].startswith("FSS at 30 dBZ")
    assert [row["lead_minutes"] for row in summary["leads"]] == [15, 30]
    for row in summary["leads"]:
        assert row["primary_fss"] is not None
        assert row["persistence_primary_fss"] is not None
        assert row["difference_primary"] == pytest.approx(
            row["primary_fss"] - row["persistence_primary_fss"])
    # The lead that put the storm where the radar found it beat the scan it
    # started from; the lead that overshot it by six cells lost to that scan.
    assert summary["leads"][0]["difference_primary"] > 0.0
    assert summary["leads"][1]["difference_primary"] < 0.0


def test_the_receipt_round_trips_and_names_the_archive_object(tmp_path):
    truth = _storm(24, 24)
    document, receipt = _score(
        tmp_path, StubModel({15: truth}),
        StubObs({ANALYSIS: truth, "2026-09-10T12:30:00": truth}))
    reread = local_da_score.read_receipt(receipt)
    assert reread == json.loads(receipt.read_text(encoding="utf-8"))
    assert reread["leads"][0]["scan"]["archive_key"].endswith(".grib2.gz")
    assert reread["leads"][0]["scan"]["sha256"] == "ab" * 32
    assert not local_da_score.has_open_leads(reread)


def test_a_receipt_with_pending_leads_is_still_open(tmp_path):
    truth = _storm(24, 24)
    document, receipt = _score(
        tmp_path, StubModel({15: truth, 60: truth}),
        StubObs({ANALYSIS: truth, "2026-09-10T12:30:00": truth}),
        now=datetime(2026, 9, 10, 12, 35, tzinfo=timezone.utc))
    assert document["leads_pending"] == [60]
    assert local_da_score.has_open_leads(document)
    summary = local_da_score.summarize(document, receipt)
    assert summary["status"] == local_da_score.SUMMARY_SCORED
    pending = summary["leads"][1]
    assert pending["status"] == nowcast.PENDING
    assert pending["primary_fss"] is None
    assert "has not arrived" in pending["reason"]


def test_a_pending_lead_is_filled_in_by_a_later_pass(tmp_path):
    truth = _storm(24, 24)
    frames = {ANALYSIS: truth, "2026-09-10T12:30:00": truth}
    early, receipt = _score(
        tmp_path, StubModel({15: truth, 60: truth}), StubObs(frames),
        now=datetime(2026, 9, 10, 12, 35, tzinfo=timezone.utc))
    assert early["leads_pending"] == [60]
    frames["2026-09-10T13:15:00"] = truth
    later, _ = _score(tmp_path, StubModel({15: truth, 60: truth}),
                      StubObs(frames))
    assert later["leads_pending"] == []
    assert later["leads_scored"] == [15, 60]
    assert local_da_score.read_receipt(receipt)["leads_scored"] == [15, 60]


# -- an offline run -----------------------------------------------------

def test_an_offline_run_records_the_reason_and_scores_nothing(tmp_path):
    truth = _storm(24, 24)
    document, receipt = _score(
        tmp_path, StubModel({15: truth, 30: truth}),
        StubObs({}, unreachable=(ANALYSIS, "2026-09-10T12:30:00",
                                 "2026-09-10T12:45:00")))
    assert document["leads_unavailable"] == [15, 30]
    assert document["primary_by_lead"] == {}
    # The window summary says unavailable, not scored: a summary that read
    # "scored" because a receipt exists would look like a score on a status
    # document that a reader only skims.
    summary = local_da_score.summarize(document, receipt)
    assert summary["status"] == local_da_score.SUMMARY_UNAVAILABLE
    for row in document["leads"]:
        assert row["status"] == nowcast.UNAVAILABLE
        assert "no network" in row["reason"]
        assert "primary_fss" not in row


def test_an_offline_cache_refuses_without_touching_the_network(tmp_path):
    cache = mrms_fetch.MrmsCompositeCache(tmp_path / "obs", offline=True)
    with pytest.raises(mrms_fetch.MrmsArchiveUnavailable, match="offline"):
        cache.ensure(ANALYSIS)
    assert not (tmp_path / "obs" / "objects").exists()


def test_score_window_never_raises_when_the_forecast_cannot_be_read(tmp_path):
    summary = local_da_score.score_window(
        plan={}, receipt_path=tmp_path / local_da_score.RECEIPT_NAME,
        analysis_time=ANALYSIS,
        forecast_manifest=tmp_path / "missing" / "ensemble-manifest.json",
        cache_root=tmp_path / "obs")
    assert summary["status"] == local_da_score.SUMMARY_UNAVAILABLE
    assert summary["leads"] == []
    assert "Error" in summary["reason"] or "error" in summary["reason"]
    assert not (tmp_path / local_da_score.RECEIPT_NAME).exists()


# -- the cache ----------------------------------------------------------

def test_a_cache_refuses_to_reuse_packs_decoded_for_another_box(tmp_path):
    root = tmp_path / "obs"
    root.mkdir()
    (root / "index.json").write_text(json.dumps({
        "schema": mrms_fetch.CACHE_SCHEMA, "bucket": mrms_fetch.DEFAULT_BUCKET,
        "region": "CONUS", "product": mrms_fetch.DEFAULT_PRODUCT,
        "bbox": "-99.0,34.0,-96.0,36.0", "frames": {}, "requests": {},
        "geometry": None}) + "\n", encoding="utf-8")
    cache = mrms_fetch.MrmsCompositeCache(root, bbox="-101.0,34.0,-96.0,36.0")
    with pytest.raises(ValueError, match="score one domain against another"):
        cache.pack_paths()


class RecordingCache(mrms_fetch.MrmsCompositeCache):
    """A cache whose only front-door calls are counted, not made."""

    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.calls: list[str] = []

    def _run(self, subcommand, arguments, *, schema):
        self.calls.append(subcommand)
        if subcommand == "nearest":
            return {"schema": schema, "offset_seconds": 40,
                    "frame": {"key": "CONUS/p/20260910/f-123040.grib2.gz",
                              "valid_time": "2026-09-10T12:30:40"}}
        if subcommand == "fetch":
            source = self.objects_dir / "f-123040.grib2.gz"
            source.write_bytes(b"not a real archive object")
            return {"schema": schema, "files": [{
                "path": str(source), "sha256": "cd" * 32,
                "fetched_at": "2026-09-15T18:00:00"}]}
        if subcommand == "decode":
            Path(arguments[arguments.index("--out") + 1]).write_bytes(b"pack")
            return {"schema": schema, "content_sha256": "ef" * 32,
                    "sentinels": {"observed_fraction": 1.0}}
        if subcommand == "grid":
            self.geometry_path.write_bytes(b"geo")
            return {"schema": schema, "content_sha256": "01" * 32,
                    "grid": {"nx": 220, "ny": 180}}
        raise AssertionError(subcommand)


def test_a_cached_request_is_answered_without_asking_the_archive(tmp_path):
    cache = RecordingCache(tmp_path / "obs", bbox="-99.0,34.0,-96.0,36.0")
    first = cache.ensure("2026-09-10T12:30:00")
    assert cache.calls == ["nearest", "fetch", "decode", "grid"]
    assert first.offset_seconds == 40.0
    assert first.object_uri == (
        "s3://noaa-mrms-pds/CONUS/p/20260910/f-123040.grib2.gz")
    assert first.object_sha256 == "cd" * 32

    reopened = RecordingCache(tmp_path / "obs", bbox="-99.0,34.0,-96.0,36.0")
    again = reopened.ensure("2026-09-10T12:30:00")
    assert reopened.calls == []
    assert again.record() == first.record()


def test_two_leads_that_resolve_to_one_frame_decode_it_once(tmp_path):
    cache = RecordingCache(tmp_path / "obs", bbox="-99.0,34.0,-96.0,36.0")
    cache.ensure("2026-09-10T12:30:00")
    cache.calls.clear()
    cache.ensure("2026-09-10T12:31:00")
    assert cache.calls == ["nearest"]
    assert len(cache.pack_paths()) == 1


def test_the_campaign_driver_drives_the_shared_library(tmp_path):
    source = (REPO_ROOT / "tools" / "obs_fetch_mrms.py").read_text(
        encoding="utf-8")
    assert "from gpuwm.obs import mrms_fetch" in source
    assert "frontdoor" not in source, (
        "the campaign driver must not resolve the front door itself; two "
        "drivers of one archive are two answers to which object was taken")


def test_the_decode_box_covers_the_grid_with_a_margin():
    latitude, longitude = _geometry()
    west, south, east, north = (
        float(value) for value in
        mrms_fetch.bbox_around(latitude, longitude).split(","))
    assert west < float(longitude.min()) and east > float(longitude.max())
    assert south < float(latitude.min()) and north > float(latitude.max())


def test_the_registration_takes_the_mask_width_from_the_case(tmp_path):
    from gpuwm.verify.obs import nowcast

    # No case to read: the default is registered and the receipt says so.
    assert local_da_score.boundary_width_cells(tmp_path) is None
    default = local_da_score.registration_for(None, case_root=tmp_path)
    assert (default["parameters"]["boundary_width_cells"]
            == nowcast.DEFAULT_BOUNDARY_WIDTH_CELLS)

    # A case that names its own rows registers those instead, and the pins
    # hash differently, so two runs with different masks cannot be compared
    # by accident.
    wider = local_da_score.registration_for(None, boundary_width=9)
    assert wider["parameters"]["boundary_width_cells"] == 9
    assert wider["registration_sha256"] != default["registration_sha256"]


# -- the door -----------------------------------------------------------

def _parser():
    import argparse

    from gpuwm.local_da import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    return parser


def test_the_score_door_takes_one_plan_and_says_what_it_does():
    parsed = _parser().parse_args(["local-da", "--score", "case/local-da.json"])
    assert parsed.score == Path("case/local-da.json")
    action = next(a for a in _parser()._subparsers._group_actions[0]
                  .choices["local-da"]._actions if a.dest == "score")
    assert action.metavar == "PLAN"
    for phrase in ("persistence", "MRMS", "15, 30, 45 and 60"):
        assert phrase in action.help


def test_the_score_door_refuses_to_share_a_command_with_a_review():
    from gpuwm.local_da import main

    parsed = _parser().parse_args(
        ["local-da", "--score", "case/local-da.json", "--dry-run"])
    assert main(parsed) == 1


def test_the_cli_reference_carries_the_score_row():
    page = (REPO_ROOT / "docs" / "public" / "CLI-OPTIONS.md").read_text(
        encoding="utf-8")
    assert "| `--score PLAN` |" in page
    row = next(line for line in page.splitlines()
               if line.startswith("| `--score PLAN` |"))
    assert "persistence" in row and "MRMS" in row


def test_the_companion_contract_publishes_the_score():
    from gpuwm.local_da import protocol_document

    contract = protocol_document()["nowcast_score"]
    assert contract["default"] is True
    assert contract["schema"] == nowcast.NOWCAST_SCORE_SCHEMA
    assert contract["summary_schema"] == local_da_score.SUMMARY_SCHEMA
    assert contract["lead_minutes"] == [15, 30, 45, 60]
    assert contract["lead_statuses"] == list(nowcast.LEAD_STATUSES)
    assert contract["score_command"] == ["local-da", "--score", "{plan_path}"]
    # A companion reading the compact row must be told which fields it has
    # and that an FSS of 1 can mean an empty box.
    assert contract["summary_lead_fields"] == [
        "lead_minutes", "valid_time", "status", "primary_fss",
        "persistence_primary_fss", "difference_primary",
        "primary_observed_base_rate", "primary_model_base_rate", "reason"]
    assert "primary_observed_base_rate 0.0" in contract["empty_box_note"]


def test_the_pages_describe_the_score_and_no_longer_deny_it():
    docs = REPO_ROOT / "docs"
    cycling = (docs / "local-da-rapid-cycling.md").read_text(encoding="utf-8")
    protocol = (docs / "local-da-companion-protocol.md").read_text(
        encoding="utf-8")
    for phrase in ("## The nowcast score", "nowcast-score.json",
                   "persistence", "`pending`", "`missing-obs`",
                   "--score"):
        assert phrase in cycling, phrase
    for phrase in ("arwen.local-da-nowcast-score.v1",
                   "arwen.local-da-nowcast-summary.v1",
                   "difference_primary", "leads_pending",
                   "primary_observed_base_rate", "nowcast_score_error",
                   "--score PLAN"):
        assert phrase in protocol, phrase
    # The sentence the receipt used to carry, retired everywhere it claimed
    # a local DA run had no skill number.
    for path in sorted((REPO_ROOT / "gpuwm").rglob("*.py")):
        assert "no forecast-skill claim" not in path.read_text(
            encoding="utf-8", errors="replace"), path
    assert "forecast-skill evaluation are not" not in cycling


# -- the score reaches a run with no flag -------------------------------
#
# Everything above drives the scorer and its helpers directly.  These four
# drive the run: delete the call at gpuwm/local_da_runtime.py:340, the one
# in ContinuousBackend.produce_window or the publication in
# Controller.step and one of them goes red.  That is the default-on claim,
# and it is the claim that most needs a pin, because the scorer swallows
# every exception by design: a broken call site cannot fail a test that
# does not look for its result.


class _AnyTimeObs(StubObs):
    """The analysis scan holds the storm where it was; every later scan has
    moved it four cells, which is where the stub forecast puts it.  So the
    model beats persistence at every lead and both numbers are real rather
    than a pair of ones off an empty box."""

    def __init__(self, analysis_time):
        super().__init__({})
        self._analysis_time = str(analysis_time)

    def field(self, valid_time):
        column = 24 if str(valid_time) == self._analysis_time else 28
        self._frames[str(valid_time)] = _storm(24, column)
        return super().field(valid_time)


def _stub_scoring(monkeypatch):
    """Put the stub model and the stub radar inside the real score_window.

    The two seams score_window builds from disk are replaced and nothing
    else is: the registration, the interior mask, the receipt write, the
    summary and every call site above them are the shipped ones.  A test
    that stubbed score_window itself would pin the name of a function
    rather than the fact that a bare run produces a number.
    """
    seen = {}

    def leads(manifest, stamp, *, lead_minutes, member=None):
        seen["stamp"] = stamp
        seen["manifest"] = Path(manifest)
        return StubModel({int(value): _storm(24, 28) for value in lead_minutes})

    def observations(cache, *, match_seconds, coverage_floor):
        seen["cache_root"] = Path(cache.root)
        return _AnyTimeObs(seen["stamp"])

    monkeypatch.setattr(local_da_score, "ForecastLeads", leads)
    monkeypatch.setattr(local_da_score, "CachedCompositeSource", observations)
    return seen


def test_an_ordinary_launch_scores_itself_and_writes_the_receipt(
        tmp_path, monkeypatch):
    from test_local_da_runtime import Backend, saved
    from gpuwm.local_da_runtime import launch

    path, plan = saved(tmp_path)
    root = path.parent
    seen = _stub_scoring(monkeypatch)
    report = launch(path, backend=Backend())

    assert report["status"] == "COMPLETE"
    receipt = root / local_da_score.RECEIPT_NAME
    assert receipt.is_file(), "a bare launch must leave the receipt behind"
    assert seen["stamp"] == local_da_score._seam(plan["analysis_times"][-1])
    assert seen["cache_root"] == local_da_score.cache_root_for(root)

    score = report["nowcast_score"]
    assert score["schema"] == local_da_score.SUMMARY_SCHEMA
    assert score["status"] == local_da_score.SUMMARY_SCORED
    assert score["receipt_path"] == str(receipt)
    assert score["receipt_schema"] == nowcast.NOWCAST_SCORE_SCHEMA
    assert score["leads_scored"] == list(nowcast.DEFAULT_LEAD_MINUTES)
    for row in score["leads"]:
        assert row["status"] == nowcast.SCORED
        assert row["primary_fss"] == pytest.approx(1.0)
        assert row["persistence_primary_fss"] < 1.0
        assert row["difference_primary"] > 0.0

    execution = json.loads((root / "execution.json").read_text(
        encoding="utf-8"))
    assert execution["nowcast_score"] == score
    assert local_da_score.read_receipt(receipt)["leads_scored"] == list(
        nowcast.DEFAULT_LEAD_MINUTES)


def test_a_launch_whose_scoring_breaks_still_completes(tmp_path, monkeypatch):
    from test_local_da_runtime import Backend, saved
    from gpuwm.local_da_runtime import launch

    path, _ = saved(tmp_path)

    def explode(*args, **kwargs):
        raise KeyError("the forecast manifest lost a key")

    monkeypatch.setattr(local_da_score, "ForecastLeads", explode)
    report = launch(path, backend=Backend())
    assert report["status"] == "COMPLETE"
    score = report["nowcast_score"]
    assert score["status"] == local_da_score.SUMMARY_UNAVAILABLE
    assert "KeyError" in score["reason"]


def _window_forecast(directory, monkeypatch):
    """A completed forecast with a frame inventory that verifies."""
    from gpuwm.ensemble import engine
    from gpuwm.ensemble.wrfout_inventory import (
        WRFOUT_INVENTORY_CONTRACT, WRFOUT_INVENTORY_KEY, file_sha256)

    ens_root = directory / "forecast"
    member_dir = ens_root / "member_000"
    member_dir.mkdir(parents=True)
    frame = member_dir / "wrfout_d01_2026-09-10_12_15_00"
    frame.write_bytes(b"one frame of output")
    entry = {"contract": WRFOUT_INVENTORY_CONTRACT, "path": frame.name,
             "domain": "d01",
             "frames": [{"index": 0, "valid_time": "2026-09-10_12:15:00"}],
             "size_bytes": frame.stat().st_size,
             "sha256": file_sha256(frame)}
    from gpuwm.ensemble.manifest import ENSEMBLE_MANIFEST_SCHEMA

    manifest = ens_root / "ensemble-manifest.json"
    manifest.write_text(json.dumps({
        "schema": ENSEMBLE_MANIFEST_SCHEMA, "status": "COMPLETE",
        "members": [{"member_dir": "member_000", "status": "DONE",
                     WRFOUT_INVENTORY_KEY: [entry]}]}), encoding="utf-8")
    forecast = SimpleNamespace(status="COMPLETE", manifest_path=manifest,
                               ens_root=ens_root)
    monkeypatch.setattr(engine, "run_ensemble", lambda *a, **k: forecast)
    return forecast


def test_a_continuous_window_writes_its_receipt_beside_its_completion(
        tmp_path, monkeypatch):
    """The shipped ContinuousBackend.produce_window, CPU forecast injected.

    Only the ensemble engine and the renderer are replaced.  The score
    call, the receipt path it is handed and the publication of the summary
    onto the window's execution record and product decision are the
    shipped ones.
    """
    from gpuwm import local_da_runtime as runtime
    from gpuwm.output_identity import file_record
    from test_local_da_runtime import saved
    from test_local_da_controller import Backend as ControllerBackend

    path, plan = saved(tmp_path)
    root = path.parent
    helper = ControllerBackend(tmp_path)
    directory = root / "continuous" / "window_000000"
    (directory / "products").mkdir(parents=True)
    (directory / "products" / "composite.png").write_bytes(b"PNG bytes")
    cycle_manifest = directory / "cycle-manifest.json"
    cycle_manifest.write_text("{}", encoding="utf-8")
    restart = directory / "analysis.npz"
    restart.write_bytes(b"analysed state")
    _window_forecast(directory, monkeypatch)
    seen = _stub_scoring(monkeypatch)

    backend = runtime.ContinuousBackend.__new__(runtime.ContinuousBackend)
    backend.plan, backend.root, backend.cfg = plan, root, helper.cfg
    backend._continuous_epoch = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    backend._active_index = 0
    backend.controller = SimpleNamespace(
        stage=lambda name, **details: None, monotonic=lambda: 1.0, started=0.0)
    backend.products = lambda forecast, product_root: {"status": "rendered"}

    decision = {"analysis": [file_record(restart)],
                "outcome": {"cycle_manifest": str(cycle_manifest),
                            "observation_usage": []}}
    products = runtime.ContinuousBackend.produce_window(
        backend, 0, directory, decision)

    receipt = directory / local_da_score.RECEIPT_NAME
    assert receipt.is_file(), (
        "the window receipt belongs beside the window's completion record")
    assert seen["stamp"] == local_da_score._seam(
        runtime._analysis_time(backend, 0))
    assert seen["cache_root"] == local_da_score.cache_root_for(root)
    score = products["nowcast_score"]
    assert score["receipt_path"] == str(receipt)
    assert score["status"] == local_da_score.SUMMARY_SCORED
    execution = json.loads((directory / "execution.json").read_text(
        encoding="utf-8"))
    assert execution["nowcast_score"] == score


def test_a_committed_window_names_its_score_and_the_status_republishes_it(
        tmp_path):
    """complete.json carries the score the window committed with, and the
    live status reads the receipts rather than that snapshot."""
    from test_local_da_controller import Backend as ControllerBackend
    from test_local_da_controller import controller

    summary = {"schema": local_da_score.SUMMARY_SCHEMA, "status": "scored",
               "receipt_path": "unused", "leads": [], "primary": "FSS"}

    class Scoring(ControllerBackend):
        rescored = False

        def produce_window(self, index, directory, decision):
            value = super().produce_window(index, directory, decision)
            value["nowcast_score"] = summary
            return value

        def rescore_pending(self):
            type(self).rescored = True
            return {}

    backend = Scoring(tmp_path)
    result = controller(tmp_path).step(backend)
    assert result["completed_windows"] == 1
    assert Scoring.rescored, (
        "the controller must ask the backend to fill in pending leads")
    completed = json.loads(
        (tmp_path / "continuous/window_000000/complete.json").read_text(
            encoding="utf-8"))
    assert completed["nowcast_score"] == summary
    status = json.loads(
        (tmp_path / "continuous/status.json").read_text(encoding="utf-8"))
    assert "nowcast_score" in status


# -- a scoring fault never fails the run --------------------------------


def _damaged_receipt(directory):
    """A receipt of the right schema with a key summarize() indexes missing.

    This is what a half-written or older-build receipt looks like to the
    reader: read_receipt() admits it, because it checks the schema tag and
    nothing else, and summarize() then indexes seven keys of it directly.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / local_da_score.RECEIPT_NAME).write_text(json.dumps({
        "schema": nowcast.NOWCAST_SCORE_SCHEMA,
        "analysis_time": ANALYSIS, "leads": []}), encoding="utf-8")


def test_a_damaged_receipt_does_not_fail_the_window_it_is_not_a_gate_on(
        tmp_path):
    """The window commits and the fault is recorded on the status.

    score_window catches every exception for this reason, and the
    controller's refresh has to keep the same rule: it walks receipts
    another process wrote, and a KeyError out of that walk used to take
    down a run over a number that is explicitly not a gate on it.
    """
    from test_local_da_controller import Backend as ControllerBackend
    from test_local_da_controller import controller

    class Damaging(ControllerBackend):
        def produce_window(self, index, directory, decision):
            value = super().produce_window(index, directory, decision)
            _damaged_receipt(directory)
            return value

    result = controller(tmp_path).step(Damaging(tmp_path))
    assert result["completed_windows"] == 1
    assert (tmp_path / "continuous/window_000000/complete.json").is_file()
    status = json.loads(
        (tmp_path / "continuous/status.json").read_text(encoding="utf-8"))
    assert status["nowcast_score"] is None
    assert status["nowcast_score_error"].startswith("KeyError")


def test_a_rescore_that_raises_any_type_is_recorded_not_propagated(tmp_path):
    from test_local_da_controller import Backend as ControllerBackend
    from test_local_da_controller import controller

    class Failing(ControllerBackend):
        def rescore_pending(self):
            raise KeyError("analysis_times")

    result = controller(tmp_path).step(Failing(tmp_path))
    assert result["completed_windows"] == 1
    status = json.loads(
        (tmp_path / "continuous/status.json").read_text(encoding="utf-8"))
    assert status["nowcast_score_error"].startswith("KeyError")


def test_the_status_door_answers_even_when_a_receipt_cannot_be_read(tmp_path):
    """gpuwm local-da --status reports whether the run is alive.

    A fault reading a score receipt must not take that answer away: the
    score is one field of the document, and the field says what broke.
    """
    from dataclasses import replace
    from gpuwm.local_da import build_plan, publish
    from gpuwm.local_da_controller import status_for_plan
    from test_local_da_plan import availability, price, request

    plan = build_plan(replace(request(), continuous_windows=2),
                      availability=availability, price=price)
    path = Path(publish(plan, tmp_path / "case")["plan_path"])
    _damaged_receipt(path.parent / "continuous" / "window_000000")

    value = status_for_plan(path)
    assert value["status"] == "NOT_STARTED"
    assert value["nowcast_score"] is None
    assert value["nowcast_score_error"].startswith("KeyError")


# -- the frame the receipt names is the frame that was scored -----------


def _frame(valid_time, *, observed_fraction, tag):
    return mrms_fetch.MrmsFrame(
        requested_valid_time="2026-09-10T12:30:00", valid_time=valid_time,
        offset_seconds=0.0, bucket=mrms_fetch.DEFAULT_BUCKET,
        key=f"CONUS/p/20260910/{tag}.grib2.gz",
        object_uri=f"s3://noaa-mrms-pds/CONUS/p/20260910/{tag}.grib2.gz",
        object_sha256=f"{tag[0]}b" * 32, fetched_at="2026-09-15T18:00:00",
        pack_path=f"/packs/{tag}.npz", pack_sha256=f"{tag[0]}c" * 32,
        observed_fraction=observed_fraction)


class _WalkingCache:
    """A cache whose nearest frame is below the floor and has neighbours."""

    def __init__(self, tmp_path, *, offline=False):
        self.root = Path(tmp_path)
        self.nearest = _frame("2026-09-10T12:30:00",
                              observed_fraction=0.2, tag="near")
        self.neighbour = _frame("2026-09-10T12:32:00",
                                observed_fraction=1.0, tag="side")
        self.offline = offline

    def ensure(self, valid_time):
        return self.nearest

    def ensure_window(self, valid_time):
        if self.offline:
            raise mrms_fetch.MrmsArchiveUnavailable(
                f"this run is offline, so the frames around {valid_time} "
                "cannot be listed")
        return [self.nearest, self.neighbour]

    def record(self):
        return {"route": "stub"}


class _Source(local_da_score.CachedCompositeSource):
    """The shipped seam with only the decoded-pack reader replaced."""

    def _rebuilt(self):
        return SimpleNamespace(field=lambda valid_time: valid_time)


def test_a_lead_scored_on_a_neighbour_keeps_that_object_s_archive_identity(
        tmp_path):
    """The walk pulls neighbours; the selection may score one of them.

    Recording only the frame ensure() picked left the scan block of that
    lead carrying the pack digest and URI from the provenance and none of
    the archive fields the cache exists to supply.
    """
    cache = _WalkingCache(tmp_path)
    source = _Source(cache, match_seconds=300, coverage_floor=0.9)
    source.field("2026-09-10T12:30:00")

    detail = source.scan_detail(cache.neighbour.valid_time)
    assert detail["archive_key"] == cache.neighbour.key
    assert detail["archive_object_uri"] == cache.neighbour.object_uri
    assert detail["archive_object_sha256"] == cache.neighbour.object_sha256
    assert detail["pack_path"] == cache.neighbour.pack_path
    assert detail["observed_fraction"] == 1.0
    # The nearest frame is still recorded; the walk adds, it does not replace.
    assert source.scan_detail(cache.nearest.valid_time)["archive_key"] == (
        cache.nearest.key)


def test_an_offline_walk_scores_the_cached_scan_and_says_the_walk_did_not_run(
        tmp_path):
    """A frame already on disk is not thrown away by an offline listing.

    ensure() answered from the cache a line earlier.  Letting the listing's
    refusal become the lead's own recorded unavailable meant an offline
    rescore reported no observations for a scan the case was holding.
    """
    cache = _WalkingCache(tmp_path, offline=True)
    source = _Source(cache, match_seconds=300, coverage_floor=0.9)
    assert source.field("2026-09-10T12:30:00") == "2026-09-10T12:30:00"

    detail = source.scan_detail(cache.nearest.valid_time)
    assert detail["archive_key"] == cache.nearest.key
    assert "offline" in detail["coverage_walk"]
    assert "could not be listed" in detail["coverage_walk"]


def test_an_unreachable_nearest_scan_is_still_unavailable(tmp_path):
    """The fall-through is for the walk only.

    When the scan itself cannot be obtained there is nothing on disk to
    score, and the lead is unavailable with the archive's own sentence.
    """
    class Unreachable(_WalkingCache):
        def ensure(self, valid_time):
            raise mrms_fetch.MrmsArchiveUnavailable(
                "this run is offline and the scan is not in the cache")

    source = _Source(Unreachable(tmp_path), match_seconds=300,
                     coverage_floor=0.9)
    with pytest.raises(nowcast.ObservationsUnreachable, match="offline"):
        source.field("2026-09-10T12:30:00")
