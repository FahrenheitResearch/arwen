"""An explicit ``--forecast-start-hour 0`` authors a manifest over f000.

``gpuwm fetch --author-front-door-manifest --forecast-start-hour 0``
named ``gfs-series-f000.tsv`` in the manifest and wrote that series only
for a nonzero lead, so the explicit analysis start was refused with
"front-door manifest inputs are missing" while the same command with
the flag omitted worked.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from gpuwm import fetch


def _fetched(tmp_path, source):
    """A completed fetch directory, as the fetch leaves it."""

    out = tmp_path / source
    out.mkdir()
    files = []
    for hour in (0, 1, 2):
        name = f"{source}.t12z.pgrb2.0p25.f{hour:03d}.subset.grib2"
        (out / name).write_bytes(f"payload f{hour:03d}".encode())
        files.append({"forecast_hour": hour, "name": name,
                      "role": f"{source}-subset"})
    # The fetch's own series, in the fetch's own format.
    (out / f"{source}-series.tsv").write_text("".join(
        f"{item['forecast_hour']}\t{item['name']}\t"
        f"{81 if item['forecast_hour'] == 0 else 96}\n" for item in files),
        encoding="utf-8")
    (out / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch.FETCH_MANIFEST_SCHEMA, "source": source,
        "cycle": "2026-09-27T12:00:00Z", "forecast_hours": [0, 1, 2],
        "files": files}), encoding="utf-8")
    roles = []
    for name in ("bridge", "namelist.wps", "experiment.toml"):
        path = tmp_path / name
        path.write_bytes(b"hashed by the manifest, never executed here")
        roles.append(path)
    return out, roles


def _author(out, roles, source, start):
    manifest, _digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=roles[0], wps_namelist=roles[1],
        experiment_config=roles[2], source=source,
        forecast_start_hour=start, progress=lambda line: None)
    return json.loads(manifest.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ["gfs", "gdas"])
@pytest.mark.parametrize("start", [0, 1])
def test_every_named_start_writes_the_series_it_names(tmp_path, source, start):
    out, roles = _fetched(tmp_path, source)
    document = _author(out, roles, source, start)
    entry = document["files"]["series"]
    assert entry["name"] == f"{source}-series-f{start:03d}.tsv"
    series = out / entry["name"]
    assert [int(line.split("\t")[0])
            for line in series.read_text().splitlines()] == list(range(start, 3))
    assert hashlib.sha256(series.read_bytes()).hexdigest() == entry["sha256"]
    assert sorted(role for role in document["files"]
                  if role.startswith("grib-")) == [
        f"grib-f{hour:03d}" for hour in range(start, 3)]


@pytest.mark.parametrize("source", ["gfs", "gdas"])
def test_an_explicit_analysis_start_is_the_whole_fetched_series(tmp_path, source):
    out, roles = _fetched(tmp_path, source)
    explicit = _author(out, roles, source, 0)
    omitted = _author(out, roles, source, None)
    assert omitted["files"]["series"]["name"] == f"{source}-series.tsv"
    # The same rows, byte for byte: f000 selects everything that was fetched.
    assert ((out / explicit["files"]["series"]["name"]).read_bytes()
            == (out / omitted["files"]["series"]["name"]).read_bytes())
    assert ({role: entry for role, entry in explicit["files"].items()
             if role != "series"}
            == {role: entry for role, entry in omitted["files"].items()
                if role != "series"})


def test_a_start_that_leaves_one_frame_is_still_refused_by_name(tmp_path):
    out, roles = _fetched(tmp_path, "gfs")
    with pytest.raises(ValueError, match="--forecast-start-hour 2 leaves 1"):
        _author(out, roles, "gfs", 2)
    assert not (out / "gfs-series-f002.tsv").exists()
