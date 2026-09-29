"""Preparations started in parallel from one GFS download keep their own
front-door manifest.

The manifest binds a preparation's own namelist, experiment and bridge by
sha256, and the front door verifies it when it starts and again when it
publishes.  Every preparation from one download used to write the one
``<download>/gfs-input-manifest.json``, so a second preparation started
from that download replaced the first one's binding while the first was
still preparing, and the first then failed with "GFS input manifest SHA
mismatch".  Both doors that author the manifest for a preparation (the
prep door with ``--source-manifest`` omitted, and the ``gpuwm go`` chain's
manifest stage) now write it beside that preparation's output root.
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime
from pathlib import Path

from gpuwm import fetch


CYCLE = "2026-08-18_18:00:00"


def _grib2_stream(count):
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * count


def _fetched_gfs_dir(tmp_path, monkeypatch):
    """One real ``gpuwm fetch --source gfs`` output, over a stubbed wire."""

    monkeypatch.setattr(
        fetch, "gfs_live_index", lambda *args, **kwargs: None)
    from tools import download_gfs_native_subset as gfs_transport

    monkeypatch.setattr(
        gfs_transport, "_download",
        lambda url, destination, **kw: destination.write_bytes(
            _grib2_stream(fetch.GFS_SUBSET_RECORD_COUNT)))
    out = tmp_path / "download"
    fetch.fetch_gfs(
        cycle=datetime(2026, 8, 18, 18), hours=(0, 3, 6),
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lambda line: None,
        derived_bar=lambda cycle, **kwargs: fetch.GFS_SUBSET_RECORD_COUNT)
    return out


def _front_door_roles(*, series, bridge, wps_namelist, experiment_config):
    """The role inventory the GFS front door verifies, built its own way."""

    from gpuwm import gfs_direct

    roles = {
        "series": Path(series),
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    for hour, path in gfs_direct._read_series(Path(series)):
        roles[f"grib-f{hour:03d}"] = path
    return roles


def test_parallel_preparations_from_one_download_each_keep_their_binding(
        tmp_path, monkeypatch):
    """Two prep doors on one download, two experiments, both authoring
    before either front door verifies: each verifies its own binding."""

    from gpuwm import gfs_direct, prep_output, source_cli

    out = _fetched_gfs_dir(tmp_path, monkeypatch)
    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"hashed, never launched")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    names = ("north", "south")
    for name in names:
        # Two experiments, so two different bindings of one download.
        (tmp_path / f"{name}.toml").write_text(
            f"[run]\n# {name}\n", encoding="utf-8")

    both_authored = threading.Barrier(len(names), timeout=120)

    def front_door(command):
        """Stands in for the launched preparation: it starts only once
        BOTH doors have authored, then runs the front door's own manifest
        verification on exactly the command its door composed."""

        both_authored.wait()

        def value(flag):
            return command[command.index(flag) + 1]

        gfs_direct._verify_input_manifest(
            Path(value("--input-manifest")), value("--input-manifest-sha256"),
            _front_door_roles(
                series=value("--series"), bridge=value("--bridge"),
                wps_namelist=value("--wps-namelist"),
                experiment_config=value("--experiment-config")))
        return 0

    monkeypatch.setattr(source_cli, "_run_native_adapter", front_door)
    # The log wrapper swaps the process's stdout, which two threads cannot
    # share; the launch it wraps is the part under test.
    monkeypatch.setattr(
        prep_output, "run_preparation", lambda args, launch: launch())

    exits: dict[str, int] = {}
    failures: dict[str, BaseException] = {}

    def prepare(name):
        try:
            exits[name] = source_cli.main([
                "--source", "gfs",
                "--gfs-series", str(out / "gfs-series.tsv"),
                "--cycle", CYCLE,
                "--wps-namelist", str(wps),
                "--experiment-config", str(tmp_path / f"{name}.toml"),
                "--geog-root", str(tmp_path / "WPS_GEOG"),
                "--output-root", str(tmp_path / name / "prepared"),
                "--bridge", str(bridge)])
        except BaseException as error:  # reported below, with its words
            failures[name] = error
            both_authored.abort()

    threads = [threading.Thread(target=prepare, args=(name,))
               for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)

    assert failures == {}, "; ".join(
        f"{name}: {type(error).__name__}: {error}"
        for name, error in failures.items())
    assert exits == {name: 0 for name in names}
    manifests = [fetch.preparation_manifest_path(tmp_path / name / "prepared")
                 for name in names]
    assert manifests[0].read_bytes() != manifests[1].read_bytes()
    assert not (out / fetch.GFS_INPUT_MANIFEST_NAME).exists(), (
        "a preparation leaves the shared download as it was fetched")


def test_go_runs_sharing_one_download_bind_their_own_manifests(
        tmp_path, monkeypatch):
    """The ``gpuwm go`` manifest stage of two runs on one download, run
    for real, each read back and digest-bound the way the chain does,
    then both verified by the front door after both stages have run."""

    from gpuwm import gfs_direct, go_cli
    from gpuwm.cli import main as cli_main

    out = _fetched_gfs_dir(tmp_path, monkeypatch)
    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"hashed, never launched")
    plans = []
    for name in ("run-a", "run-b"):
        root = tmp_path / "case" / name
        authority = root / "authority"
        authority.mkdir(parents=True)
        (authority / "namelist.wps").write_text("&share\n/\n", encoding="utf-8")
        (authority / "experiment.toml").write_text(
            f"[run]\n# {name}\n", encoding="utf-8")
        plans.append({"source": "gfs", "data": out, "root": root,
                      "authority": authority, "prepared": root / "prepared"})

    bound = []
    for plan in plans:
        command = go_cli.manifest_command(plan, bridge)
        assert command[1:4] == ["-m", "gpuwm.cli", "fetch"]
        assert cli_main(command[3:]) == 0
        manifest = go_cli.front_door_manifest(plan)
        bound.append((manifest,
                      hashlib.sha256(manifest.read_bytes()).hexdigest()))

    assert bound[0][0] != bound[1][0]
    for plan, (manifest, digest) in zip(plans, bound):
        assert manifest.parent == plan["root"]
        gfs_direct._verify_input_manifest(manifest, digest, _front_door_roles(
            series=out / "gfs-series.tsv", bridge=bridge,
            wps_namelist=plan["authority"] / "namelist.wps",
            experiment_config=plan["authority"] / "experiment.toml"))
    assert not (out / fetch.GFS_INPUT_MANIFEST_NAME).exists()
