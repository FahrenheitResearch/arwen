"""The multi-radar build: what it refuses, and what it must record.

Site ids here are invented (``AAAA``/``BBBB``/``CCCC``) so that a test can
never become the place a real site name enters the tree.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.obs.superob import CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR
from tools import obs_radar_grid_build as build


# --------------------------------------------------------------------------
# the pure request check, which runs before anything is read or fetched
# --------------------------------------------------------------------------
def _args(**kwargs):
    base = {"site": [], "discover_sites": False}
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_repeated_sites_are_kept_in_order_and_upper_cased():
    assert build.requested_sites(
        _args(site=["aaaa", "BBBB", " cccc "])) == ["AAAA", "BBBB", "CCCC"]


def test_naming_a_site_twice_is_a_refusal_not_a_silent_collapse():
    with pytest.raises(SystemExit, match="given twice"):
        build.requested_sites(_args(site=["AAAA", "aaaa"]))


def test_naming_and_discovering_at_once_is_a_refusal():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        build.requested_sites(_args(site=["AAAA"], discover_sites=True))


def test_no_site_and_no_discovery_is_a_refusal_not_a_default():
    with pytest.raises(SystemExit, match="must not be one"):
        build.requested_sites(_args())


def test_discovery_names_nothing_up_front():
    assert build.requested_sites(_args(discover_sites=True)) == []


# --------------------------------------------------------------------------
# the whole tool, with the network and the model stubbed out
# --------------------------------------------------------------------------
SHAPE = (4, 6, 5)


class _Grid:
    nz, ny, nx = SHAPE
    lat = np.full((SHAPE[1], SHAPE[2]), 42.0)
    lon = np.full((SHAPE[1], SHAPE[2]), -94.0)

    def identity_sha256(self):
        return "grid-identity"


def _contribution(site_id, valid_time, vr_cells,
                  clear_air_source=CLEAR_AIR_SOURCE):
    """A superob contribution carrying `vr_cells` usable velocity cells."""

    zeros = np.zeros(SHAPE)
    vr_count = np.zeros(SHAPE, dtype=np.int64)
    flat = vr_count.reshape(-1)
    flat[:vr_cells] = 3
    beam = np.zeros(SHAPE)
    beam.reshape(-1)[:vr_cells] = 1.0
    counts = SimpleNamespace(
        to_payload=lambda: {"gates_considered": 100 * vr_cells})
    return SimpleNamespace(
        site_id=site_id, lat_deg=42.0, lon_deg=-94.0, alt_m=300.0,
        valid_time=valid_time, z_linear_sum=zeros.copy(),
        z_count=np.zeros(SHAPE, dtype=np.int64),
        z_max_dbz=zeros.copy(), z_sumsq_dbz=zeros.copy(),
        z_sum_dbz=zeros.copy(), vr_sum=zeros.copy(), vr_sumsq=zeros.copy(),
        vr_count=vr_count, vr_min=zeros.copy(), vr_max=zeros.copy(),
        beam_east=beam, beam_north=zeros.copy(), beam_up=zeros.copy(),
        nyquist_min=np.full(SHAPE, 25.5),
        vr_rejected=np.zeros(SHAPE, dtype=np.int64),
        z0_count=np.zeros(SHAPE, dtype=np.int64),
        clear_air_source=clear_air_source,
        counts=counts, provenance={}, fold_suspicion=[])


def _install(monkeypatch, tmp_path, *, volumes, failing=(), cells=None):
    """Stub the feed, the decoder and the writer; keep the real merge."""

    from gpuwm.obs import superob

    written = {}
    cells = cells or {}
    censored: list[bool] = []
    written["censor_flag_seen"] = censored

    def _acquire_volume(binary, *, site, out_dir, valid_time, **kwargs):
        if site in failing:
            raise build_source_error(f"{site} could not be served")
        path = tmp_path / f"{site}.volume"
        path.write_bytes(site.encode())
        return SimpleNamespace(
            path=path, filename=f"{site}.volume", valid_time=volumes[site],
            sha256=build._sha256(path), keys=(f"key/{site}",), partial=False,
            chunks=None, lag_seconds=1.0, observed_at="2026-08-05T04:30:00Z",
            offset_seconds=0.0, feed="archive-volumes",
            receipt={"bucket": "a-bucket", "site": site})

    def _read_sweep_pack(pack):
        site = pack.name.split(".")[0]
        return SimpleNamespace(
            site=SimpleNamespace(id=site, lat_deg=42.0, lon_deg=-94.0,
                                 alt_m=300.0, source="message31-vol-block"),
            valid_time=volumes[site], provenance=lambda: {},
            pack_schema="gpuwm-obs.sweep-pack.v2")

    def _superob_volume(volume, grid, *, params, clear_air_from_censor=False):
        site = volume.site.id
        # The clear-air regime the build asked for has to reach the superob
        # call, or a file would claim censor-derived zeroes it never built.
        censored.append(bool(clear_air_from_censor))
        source = (CLEAR_AIR_SOURCE_CENSOR if clear_air_from_censor
                  else CLEAR_AIR_SOURCE)
        return _contribution(site, volume.valid_time, cells.get(site, 6),
                             clear_air_source=source)

    def _write_radar_grid(path, observations, grid, *, valid_time, params,
                          provenance, overwrite):
        written["valid_time"] = valid_time
        written["provenance"] = provenance
        written["radars"] = list(observations.radars)
        return {"schema": "gpuwm-obs.radar-grid.v1", "sha256": "deadbeef"}

    monkeypatch.setattr(build, "_sha256", lambda p: f"sha-{Pathish(p)}")
    monkeypatch.setattr("gpuwm.obs.radar_source.acquire_volume",
                        _acquire_volume)
    monkeypatch.setattr("gpuwm.obs.nexrad.find_nexrad_bin",
                        lambda: tmp_path / "rw_nexrad")
    monkeypatch.setattr("gpuwm.obs.nexrad.run_decode",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("gpuwm.obs.nexrad.run_verify",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("gpuwm.obs.sweeps.read_sweep_pack", _read_sweep_pack)
    monkeypatch.setattr(superob, "superob_volume", _superob_volume)
    monkeypatch.setattr("gpuwm.obs.radar_grid.write_radar_grid",
                        _write_radar_grid)
    monkeypatch.setattr("gpuwm.obs.target_grid.TargetGrid.from_wrfout",
                        classmethod(lambda cls, path: _Grid()))
    return written


def Pathish(p):
    return getattr(p, "name", str(p))


def build_source_error(message):
    from gpuwm.obs.radar_source import RadarSourceError
    return RadarSourceError(message)


def _run(tmp_path, *extra):
    wrfout = tmp_path / "wrfout"
    wrfout.write_bytes(b"georeference")
    argv = ["--valid-time", "2026-08-05T04:30:00Z",
            "--grid-wrfout", str(wrfout),
            "--out", str(tmp_path / "obs.nc"),
            "--work-dir", str(tmp_path / "work"),
            "--source", "archive", *extra]
    return build.main(argv)


def test_three_radars_land_as_three_batches_with_their_own_volume_times(
        monkeypatch, tmp_path, capsys):
    written = _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:28:56Z",
        "BBBB": "2026-08-05T04:28:50Z",
        "CCCC": "2026-08-05T04:32:23Z"})
    assert _run(tmp_path, "--site", "AAAA", "--site", "BBBB",
                "--site", "CCCC") == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["sites"] == ["AAAA", "BBBB", "CCCC"]
    # Each radar keeps its OWN volume time; a merged product that claimed
    # one time for three antennas would be a fiction about simultaneity.
    assert [r["volume_valid_time"] for r in payload["per_radar"]] == [
        "2026-08-05T04:28:56Z", "2026-08-05T04:28:50Z",
        "2026-08-05T04:32:23Z"]
    assert [r["id"] for r in written["radars"]] == ["AAAA", "BBBB", "CCCC"]
    # The file's own valid_time is the requested ANALYSIS time once more
    # than one radar contributes.
    assert written["valid_time"] == "2026-08-05T04:30:00Z"
    # 213 s between the earliest and latest contributing volume.
    assert payload["radar_time_spread_seconds"] == pytest.approx(213.0)


def test_a_single_radar_keeps_the_flat_receipt_keys_and_its_own_time(
        monkeypatch, tmp_path, capsys):
    written = _install(monkeypatch, tmp_path,
                       volumes={"AAAA": "2026-08-05T04:28:56Z"})
    assert _run(tmp_path, "--site", "AAAA") == 0
    payload = json.loads(capsys.readouterr().out)
    # The keys every reader written against the single-radar route uses.
    for key in ("site", "volume", "volume_sha256", "counts", "antenna",
                "source"):
        assert key in payload, key
    assert payload["site"] == "AAAA"
    assert written["valid_time"] == "2026-08-05T04:28:56Z"
    assert written["provenance"]["site"] == "AAAA"


def test_a_site_that_cannot_be_served_degrades_to_the_ones_that_can(
        monkeypatch, tmp_path, capsys):
    _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:28:56Z",
        "CCCC": "2026-08-05T04:32:23Z"}, failing={"BBBB"})
    assert _run(tmp_path, "--site", "AAAA", "--site", "BBBB",
                "--site", "CCCC") == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["sites"] == ["AAAA", "CCCC"]
    # Degrading is allowed.  Degrading quietly is not: the failure is named
    # with its reason and the full request is still on the record.
    assert [f["site"] for f in payload["sites_failed"]] == ["BBBB"]
    assert "could not be served" in payload["sites_failed"][0]["reason"]
    assert payload["sites_requested"] == ["AAAA", "BBBB", "CCCC"]


def test_min_radars_turns_degradation_into_a_refusal(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, volumes={"AAAA": "2026-08-05T04:28:56Z"},
             failing={"BBBB", "CCCC"})
    with pytest.raises(SystemExit, match="--min-radars demands 2"):
        _run(tmp_path, "--site", "AAAA", "--site", "BBBB", "--site", "CCCC",
             "--min-radars", "2")


def test_every_site_failing_is_a_refusal_even_at_the_default(
        monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, volumes={}, failing={"AAAA", "BBBB"})
    with pytest.raises(SystemExit, match="0 radar"):
        _run(tmp_path, "--site", "AAAA", "--site", "BBBB")


def test_volumes_too_far_apart_in_time_are_refused(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:20:00Z", "BBBB": "2026-08-05T04:38:00Z"})
    with pytest.raises(SystemExit, match="describing two atmospheres"):
        _run(tmp_path, "--site", "AAAA", "--site", "BBBB",
             "--max-radar-time-spread-seconds", "300")


def test_a_spread_inside_the_ceiling_is_accepted_and_still_reported(
        monkeypatch, tmp_path, capsys):
    _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:29:00Z", "BBBB": "2026-08-05T04:31:00Z"})
    assert _run(tmp_path, "--site", "AAAA", "--site", "BBBB",
                "--max-radar-time-spread-seconds", "300") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["radar_time_spread_seconds"] == pytest.approx(120.0)


def test_the_overlap_between_radars_is_measured_not_assumed(
        monkeypatch, tmp_path, capsys):
    # AAAA sees the first 6 cells, BBBB the first 4: 4 cells are seen by
    # both, 2 by exactly one.
    _install(monkeypatch, tmp_path,
             volumes={"AAAA": "2026-08-05T04:29:00Z",
                      "BBBB": "2026-08-05T04:30:00Z"},
             cells={"AAAA": 6, "BBBB": 4})
    assert _run(tmp_path, "--site", "AAAA", "--site", "BBBB") == 0
    overlap = json.loads(capsys.readouterr().out)["overlap"]
    assert overlap["cells_with_any_radar"] == 6
    assert overlap["multi_radar_cells"] == 4
    assert overlap["cells_by_radar_count"] == {"1": 2, "2": 4}


def test_per_radar_provenance_reaches_the_written_file(monkeypatch,
                                                       tmp_path):
    written = _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:29:00Z", "BBBB": "2026-08-05T04:31:00Z"})
    _run(tmp_path, "--site", "AAAA", "--site", "BBBB")
    prov = written["provenance"]
    assert prov["contributing_sites"] == ["AAAA", "BBBB"]
    assert prov["site_selection"] == "named"
    assert [entry["site"] for entry in prov["per_radar"]] == ["AAAA", "BBBB"]
    # An analysis must always be attributable: every contributing volume's
    # object key and digest travels with the file.
    for entry in prov["per_radar"]:
        assert entry["volume_key"]
        assert entry["volume_sha256"]
        assert entry["antenna"]["source"] == "message31-vol-block"
        # Which decoder schema produced the pack is part of attribution:
        # the clear-air regime a file can honestly claim depends on it.
        assert entry["pack_schema"] == "gpuwm-obs.sweep-pack.v2"
    assert "radar_time_spread_policy" in prov


def test_the_clear_air_regime_is_the_one_the_build_asked_for(monkeypatch,
                                                             tmp_path,
                                                             capsys):
    """--clear-air-from-censor has to reach EVERY radar's superob call.

    A build that forwarded the flag to the decoder but not to the superob
    would write a file whose zeroes were measurement-only while its
    ``clear_air_source`` claimed the censored regime -- the DA adapter
    would then accept coverage that was never established.
    """

    written = _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:29:00Z", "BBBB": "2026-08-05T04:31:00Z"})
    assert _run(tmp_path, "--site", "AAAA", "--site", "BBBB",
                "--clear-air-from-censor") == 0
    payload = json.loads(capsys.readouterr().out)
    assert written["censor_flag_seen"] == [True, True]
    assert payload["clear_air_source"] == CLEAR_AIR_SOURCE_CENSOR
    assert written["provenance"]["clear_air_source"] == (
        CLEAR_AIR_SOURCE_CENSOR)


def test_without_the_flag_the_file_claims_only_the_measured_regime(
        monkeypatch, tmp_path, capsys):
    written = _install(monkeypatch, tmp_path, volumes={
        "AAAA": "2026-08-05T04:29:00Z", "BBBB": "2026-08-05T04:31:00Z"})
    assert _run(tmp_path, "--site", "AAAA", "--site", "BBBB") == 0
    payload = json.loads(capsys.readouterr().out)
    assert written["censor_flag_seen"] == [False, False]
    assert payload["clear_air_source"] == CLEAR_AIR_SOURCE
