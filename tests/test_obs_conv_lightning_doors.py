"""MADIS streams and the GLM door reach a user (2.9.2, lane conv-obs).

Breakage these prevent:

* ``gpuwm obs streams`` called MADIS aircraft account-gated and refused to
  fetch it, while the public MADIS files open anonymously and the engine
  already carried the decoder (``rw_madis``) that nothing could reach.
* GOES GLM flash extent density, an open lightning product, had no engine
  door at all.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.obs import frontdoor, streams


def test_no_stream_claims_madis_needs_an_account():
    assert "madis-aircraft" not in streams.STREAMS
    for name in ("madis-acars", "madis-raob", "madis-mesonet"):
        spec = streams.STREAMS[name]
        assert spec.public and not spec.account_gated, name
        assert spec.door == "rw_madis" and spec.decoder_built, name
        assert spec.latency_class == "fast", name
    assert streams.account_gated_sources() == []
    assert "need a NOAA MADIS account" not in streams.__doc__


def test_the_madis_rows_carry_their_measurement_labels():
    from gpuwm.obs.neutral_table import MEASUREMENT_TABLE

    for label in ("aircraft_level", "platform_wind", "profiler_level", "vad_level",
                  "vad_superob", "station_pressure_from_sea_level"):
        assert label in MEASUREMENT_TABLE, label


def test_a_madis_fetch_drives_the_door_fetch_then_table(monkeypatch, tmp_path):
    calls = []

    def fake_run(name, arguments):
        calls.append((name, list(arguments)))
        if arguments[0] == "fetch":
            out = Path(arguments[arguments.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            record = {"schema": "gpuwm-obs.madis-fetch.v1", "status": "READY",
                      "files": [{"name": "20261016_0700.gz"}, {"name": "20261016_0800.gz"}]}
            (out / "fetch-acars.json").write_text(json.dumps(record))
            return record
        table = Path(arguments[arguments.index("--out") + 1])
        # rw_madis nests the table facts and names its counters "counts".
        table.with_suffix(".json").write_text(json.dumps({
            "status": "READY", "table": {"path": str(table), "rows": 3, "sha256": "ab", "bytes": 9},
            "counts": {"reports_read": 5}}))
        return {}

    monkeypatch.setattr(streams, "_run_door", fake_run)
    got = streams.fetch_stream("madis-acars", "2026-10-16T07:45:00Z", "2026-10-16T08:15:00Z", tmp_path,
                               bbox="20,55,-130,-60")
    (door, fetch), (_, table) = calls
    assert door == "rw_madis"
    assert fetch[:3] == ["fetch", "--kind", "acars"]
    assert table[:3] == ["table", "--kind", "acars"]
    raw = tmp_path / "madis-acars" / "raw"
    assert table[table.index("--files") + 1:table.index("--files") + 3] == [
        str(raw / "20261016_0700.gz"), str(raw / "20261016_0800.gz")]
    assert table[table.index("--bbox") + 1] == "20,55,-130,-60"
    assert got.manifest["latency_class"] == "fast"
    assert got.manifest["table"]["rows"] == 3 and got.manifest["table"]["sha256"] == "ab"
    assert got.manifest["counters"] == {"reports_read": 5}
    assert got.table == tmp_path / "madis-acars" / "madis-acars.csv"


def test_an_hour_madis_never_had_is_a_stated_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(streams, "_run_door", lambda name, arguments: {
        "status": "EMPTY", "files": [], "hours_not_in_archive": [{"hour": "x"}, {"hour": "y"}]})
    with pytest.raises(streams.StreamEmpty, match="2 hours answered 404"):
        streams.fetch_stream("madis-raob", "2026-10-16T07:00:00Z", "2026-10-16T08:00:00Z", tmp_path)


def test_gpuwm_obs_glm_forwards_to_the_goes_door(monkeypatch, tmp_path):
    from gpuwm.obs import cli

    binary = tmp_path / "rw_goes"
    monkeypatch.setattr(frontdoor.FrontDoor, "require", lambda self: binary)
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda argv, check=False: ran.append(argv) or SimpleNamespace(returncode=0))
    assert cli._glm_main(SimpleNamespace(argv=["--time", "2026-10-16T08:15:00Z", "--center", "32.5,-97.3",
                                                "--radius-km", "420", "--out", "fed.csv"])) == 0
    assert ran[-1][:2] == [str(binary), "glm"]
    assert cli._glm_main(SimpleNamespace(argv=[])) == 0
    assert ran[-1] == [str(binary), "--help"]


def test_the_goes_door_pins_the_glm_record():
    assert "gpuwm-obs.glm-fed.v2" in frontdoor.GOES.abi_marker.split("\t")


def test_gpuwm_obs_glm_is_a_command():
    import argparse

    from gpuwm.obs import cli

    parser = argparse.ArgumentParser()
    cli.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["obs", "glm", "--time", "2026-10-16T08:15:00Z"])
    assert args.func is cli._glm_main
    assert args.argv == ["--time", "2026-10-16T08:15:00Z"]
