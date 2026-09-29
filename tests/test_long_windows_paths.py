"""A forecast chain under a deep runs folder must not fail as a missing file.

Windows refuses a plain path at or past 260 characters (248 for a
directory) with ERROR_PATH_NOT_FOUND, which Python reports as
FileNotFoundError, unless LongPathsEnabled is set.  A GFS download
under ``<runs>/<run>/chain/downloads/<64-hex key>/`` failed that way on
its ``.part`` staging file, and the stages after it failed the same way
once their output folder was resolved back to the plain spelling.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.filesystem_paths import (CHAIN_DEPTH_BUDGET, WINDOWS_MAX_PATH,
                                    canonical_path, io_path, keep_spelling)

windows = pytest.mark.skipif(os.name != "nt", reason="the path limit is a Windows limit")

REPO = Path(__file__).resolve().parents[1]
EXTENDED = "\\\\?\\"


def deep_folder(base: Path, length: int) -> Path:
    """A folder under ``base`` whose plain spelling is ``length`` long."""
    folder = Path(os.path.abspath(base))
    while len(str(folder)) < length - 40:
        folder = folder / ("level-" + "x" * 30)
    return folder / ("y" * (length - len(str(folder)) - 1))


@windows
def test_a_resolved_folder_keeps_the_spelling_its_caller_chose(tmp_path):
    folder = deep_folder(tmp_path, 300)
    io_path(folder).mkdir(parents=True)
    identity = canonical_path(folder)
    # Receipts record the plain identity, as they always have ...
    assert not str(identity).startswith(EXTENDED)
    # ... and a caller that asked in the extended spelling writes through it.
    spelled = keep_spelling(io_path(folder), identity)
    assert str(spelled).startswith(EXTENDED)
    (spelled / "file.part").write_bytes(b"GRIB")
    assert (io_path(folder) / "file.part").read_bytes() == b"GRIB"
    assert keep_spelling(tmp_path, canonical_path(tmp_path)) == canonical_path(tmp_path)


@windows
def test_a_deep_output_folder_is_handed_to_the_chain_extended(tmp_path):
    from gpuwm.go_cli import _extend_outdir, chain_io_root

    short = Path("C:/runs/one")
    assert chain_io_root(short) == short
    deep = deep_folder(tmp_path, WINDOWS_MAX_PATH - CHAIN_DEPTH_BUDGET + 5)
    assert str(chain_io_root(deep)).startswith(EXTENDED)
    downloading = {"fetch": {"source": "gfs"}}
    args = SimpleNamespace(outdir=deep)
    _extend_outdir(args, tmp_path / "forecast.toml", downloading)
    assert str(args.outdir).startswith(EXTENDED)
    args = SimpleNamespace(outdir=short)
    _extend_outdir(args, tmp_path / "forecast.toml", downloading)
    assert args.outdir == short
    # A run that reuses a prepared bundle writes no request cache, so the
    # same folder has room for it.
    args = SimpleNamespace(outdir=deep, prepared_root="bundle")
    _extend_outdir(args, tmp_path / "forecast.toml", downloading)
    assert args.outdir == deep


@windows
def test_the_authority_stage_writes_under_a_deep_run_folder(tmp_path):
    from gpuwm.prepared_single_domain_forecast import (
        materialize_named_source_authorities)

    from gpuwm.go_cli import chain_io_root

    # Spelled the way the chain hands it over; the stage's own resolution
    # must not strip it back to a spelling it cannot write through.
    run = chain_io_root(deep_folder(tmp_path, 250)) / "run-20260926-000000Z_i202609260000Z"
    materialize_named_source_authorities(
        source="gfs",
        base_experiment_config=REPO / "configs" / "gfs_12km_quickstart.toml",
        base_wps_namelist=REPO / "configs" / "gfs_12km_quickstart.namelist.wps",
        physics_profile=None, output_directory=run / "authority")
    written = sorted(path.name for path in io_path(run / "authority").iterdir())
    assert "experiment.toml" in written and "namelist.wps" in written


@windows
def test_a_gfs_object_downloads_into_a_deep_folder(tmp_path, monkeypatch):
    from tools import download_gfs_native_subset as transport

    payload = b"GRIB" + b"\0" * 2048 + b"7777"

    class Reply:
        def __init__(self):
            self.left = payload

        def read(self, size=-1):
            chunk, self.left = self.left[:size], self.left[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(transport, "paced_urlopen", lambda request, timeout: Reply())
    folder = deep_folder(tmp_path, 240) / ("0" * 64)
    io_path(folder).mkdir(parents=True)
    # The folder the fetch hands its transfers, spelled as the fetch spells it.
    from gpuwm.filesystem_paths import DOWNLOAD_DEPTH_BUDGET, deep_io_path

    out = deep_io_path(folder, DOWNLOAD_DEPTH_BUDGET)
    assert str(out).startswith(EXTENDED)
    target = out / "gfs.t00z.pgrb2.0p25.f003.subset.grib2"
    transport._download("https://nomads.invalid/object", target, retries=1)
    assert io_path(target).read_bytes() == payload
    assert not [name for name in os.listdir(io_path(folder)) if name.endswith(".part")]


@windows
def test_the_domain_tree_claims_a_deep_folder_in_the_spelling_it_was_handed(tmp_path):
    from gpuwm.go_cli import chain_io_root
    from gpuwm.prepared_domain_tree_forecast import claim_output_directory

    inputs = tmp_path / "prepared"
    inputs.mkdir()
    run = chain_io_root(deep_folder(tmp_path, 250)) / "run-20260926-000000Z_tree"
    claimed = claim_output_directory(run, protected_roots=(inputs,))
    # A nested run writes its domains and pictures below this folder.
    assert str(claimed).startswith(EXTENDED)
    (claimed / "d02" / "product").mkdir(parents=True)
    (claimed / "d02" / "product" / ("f" * 40 + ".png")).write_bytes(b"PNG")
    # An output folder handed over in the extended spelling still may not sit inside the inputs.
    nested = chain_io_root(deep_folder(inputs, 250))
    with pytest.raises(ValueError, match="overlaps protected input"):
        claim_output_directory(nested, protected_roots=(inputs,))
    with pytest.raises(ValueError, match="overlaps protected input"):
        claim_output_directory(io_path(inputs) / "out", protected_roots=(inputs,))


@windows
def test_the_page_reads_a_finished_run_under_a_deep_runs_folder(tmp_path, monkeypatch):
    from gpuwm.gui.server import build_server, serve_in_thread

    from test_gui_server import FakeRunner, PNG, dead_pid, make_run, request

    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    # A page asks through its own short probe; no server is asked in a test either.
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    # The runs folder a person typed, short enough to name but too deep for the pictures below it.
    root = deep_folder(tmp_path, 205)
    server = build_server(root, port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    try:
        make_run(io_path(server.root), "done", pid=dead_pid(), end={"event": "completed"}, frames=4)
        response, body = request(server, "GET", "/api/runs")
        assert response.status == 200, body
        assert [row["id"] for row in body["runs"]] == ["done"]
        assert body["runs"][0]["status"]["state"] == "finished"
        response, index = request(server, "GET", "/api/runs/done/pictures")
        assert response.status == 200 and index["count"] == 4, index
        response, listing = request(server, "GET",
                                    "/api/runs/done/pictures/list?product=2m_temperature&domain=d01-12km")
        picture = listing["pictures"][0]["path"]
        assert len(str(root)) + len("/done/") + len(picture) >= WINDOWS_MAX_PATH
        response, raw = request(server, "GET", "/api/runs/done/files/" + picture)
        assert response.status == 200 and raw == PNG
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("source", ["gem-gdps", "icon-eu", "aigefs", "gefs", "hrrr-prs"])
def test_each_table_source_is_measured_by_its_own_object_names(source):
    """GDPS and ICON name an object in more than 80 characters; the 150-character chain budget was measured on
    GFS's 37, so a folder that fit GFS's download could still take a GDPS download past the limit."""
    from gpuwm import fetch_routes
    from gpuwm.go_cli import download_depth

    table = {"source": source, "cycle": "2026-09-20T00", "hours": 6}
    depth = download_depth(table)
    plan = fetch_routes.resolve_request(source, cycle=__import__("datetime").datetime(2026, 9, 20), hours=6)
    deepest = max(len(obj.relpath) for obj in plan.objects)
    # downloads/<64-hex key>/<object>.part, every character counted.
    assert depth >= len("downloads/") + 64 + 1 + deepest + len(".part")


@windows
def test_a_folder_that_fits_gfs_is_spelled_extended_for_a_longer_named_source(tmp_path):
    from gpuwm.filesystem_paths import CHAIN_DEPTH_BUDGET
    from gpuwm.go_cli import _extend_outdir, download_depth

    table = {"source": "gem-gdps", "cycle": "2026-09-20T00", "hours": 6}
    depth = download_depth(table)
    assert depth > CHAIN_DEPTH_BUDGET
    # Room for the GFS chain, not for this source's longest object.
    folder = deep_folder(tmp_path, WINDOWS_MAX_PATH - CHAIN_DEPTH_BUDGET - 3)
    args = SimpleNamespace(outdir=folder)
    _extend_outdir(args, tmp_path / "forecast.toml", {"fetch": table})
    assert str(args.outdir).startswith(EXTENDED)
    assert len(str(folder)) + depth >= WINDOWS_MAX_PATH
