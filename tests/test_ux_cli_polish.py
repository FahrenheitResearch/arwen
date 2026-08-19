"""CLI-polish findings from the 2.5.0 UX persona walks.

Seven measured frictions, each reproduced here the way the walk met it
-- through the real front door wherever the finding IS front-door
behaviour -- before it was fixed:

* **N9** -- ``gpuwm setup`` spent 15.5 s of its 16.2 s in unbroken
  silence while 315 MiB of Thompson tables downloaded, which is
  indistinguishable from a hang.  The table download now counts its
  bytes on stderr, and the setup wrapper captures stdout only, so the
  counter reaches the reader through it.
* **N10** -- fetch feedback was inverted: the large table route printed
  nothing at all between its opening line and its manifest, while the
  small GFS route's per-file lines block-buffered through a pipe (9.1 s
  of 9.8 silent in a log).  The route now names each object as it lands
  and counts bytes while they move, and every front door line-buffers
  its stdout so a log streams.
* **N13** -- ``prep-command.txt`` is documented "runnable as written"
  and exited 78 on the second paste: authoring refused to overwrite
  ``inputs.json`` and named no remedy.  Re-authoring byte-identical
  content is now idempotent; differing content still refuses, and the
  refusal prints a line the reader can type.
* **N15** -- the upgrade was invisible: doctor's report is shape
  identical to 2.4.1 and ``gpuwm run --help`` was byte-identical.
  Doctor now names what changed when the installed version differs from
  the version its last run here recorded, and ``run --help`` points at
  the unbundled stages and the run-folder/render layouts.
* **N16 residue** -- the ahead-of-PyPI sentence landed, and said the
  index's version twice in one sentence.
* **N17** -- ``gpuwm doctor --source`` accepted era5/gfs/hrrr only,
  three of the registry's sources, and refused ``hrrr-prs`` -- which is
  the route the walk actually used.  The choices are the registry's now.
* **N24** -- ``gpuwm --version`` was an argparse usage error; render's
  layout line mixed ``\\`` and ``/``; three consecutive no-information
  "pip extra aliases" lines read as three findings.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from gpuwm import fetch_routes, table_assets
from gpuwm.cli import main as cli_main
from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS, TableAsset

_REPO = Path(__file__).resolve().parents[1]
_CLI = (sys.executable, "-m", "gpuwm.cli")


def _door(argv, *, env=None, timeout=600):
    """One real front-door invocation, captured."""

    return subprocess.run(
        [*_CLI, *argv], capture_output=True, text=True, cwd=_REPO,
        env=env, timeout=timeout)


class _Serve(http.server.BaseHTTPRequestHandler):
    """Hand out exactly ``body`` for any path, in small writes."""

    body = b""

    def do_GET(self):                                    # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        view = memoryview(self.body)
        for start in range(0, len(view), 65536):
            self.wfile.write(view[start:start + 65536])

    def log_message(self, *_args):                       # noqa: D102
        return


#: Two MiB of a GRIB-shaped payload: bigger than the one-MiB transfer
#: chunk (so a counter has to tick more than once) and complete by the
#: route's own magic/end-marker bar, so nothing here needs the bars
#: relaxed to make a point about progress.
_BODY = b"GRIB" + bytes(2 * 1024 * 1024) + b"7777"


@pytest.fixture
def local_host(request):
    """A real HTTP server on loopback, serving one body."""

    body = getattr(request, "param", _BODY)
    handler = type("_Handler", (_Serve,), {"body": body})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", body
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# N9: the table download counts its bytes, out loud, on stderr
# ---------------------------------------------------------------------------

def test_a_table_download_counts_its_bytes_on_stderr(local_host, tmp_path,
                                                     capsys):
    """The noob's 15.5 s of silence, measured over a real socket.

    The pinned size is the total, so the counter can say a percentage
    from the first chunk without trusting a header.
    """

    base, body = local_host
    asset = TableAsset(
        filename="freezeH2O.dat", bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest())
    installed = table_assets.fetch_asset_from_url(
        tmp_path, asset, f"{base}/{asset.filename}")

    assert installed.is_file()
    said = capsys.readouterr().err
    assert asset.filename in said, said
    assert "MiB" in said, said
    assert "%" in said, said
    # The counter has to MOVE: a single line printed at the end is the
    # silence this finding is about, one line later.
    assert said.count(asset.filename) > 1, said


def test_the_setup_wrapper_captures_stdout_only(monkeypatch, capsys):
    """Why the counter is on stderr.

    ``gpuwm setup`` captures each step's stdout so it can print one
    status line per step and replay the whole text on a refusal.  A
    progress counter written to stdout would be captured with it and
    reach the reader only after the download it was reporting on had
    finished -- which is exactly the silence N9 names.
    """

    import argparse

    from gpuwm import setup_cli

    def _step(_args):
        print("captured: the step's own report")
        print("streamed: 1.0 / 2.0 MiB (50%)", file=sys.stderr)
        return 0

    monkeypatch.setattr(
        setup_cli, "step_namespace",
        lambda _module: argparse.Namespace(func=_step))
    code, captured = setup_cli._run_step("gpuwm.table_assets", {})

    assert code == 0
    assert "captured:" in captured
    streamed = capsys.readouterr()
    assert "streamed: 1.0 / 2.0 MiB (50%)" in streamed.err
    assert "streamed:" not in streamed.out


@pytest.mark.slow
def test_fetch_tables_streams_progress_before_it_refuses(local_host, tmp_path):
    """The real door, over a real socket, with the pinned bars intact.

    The served bytes are not the pinned ones, so the command refuses --
    which is the point: the byte counter has to reach the reader from
    inside the transfer, not from the summary afterwards.
    """

    base, _body = local_host
    root = tmp_path / "tables"
    root.mkdir()
    packaged = Path(table_assets.staging_root())
    externalized = table_assets.EXTERNALIZED_TABLE_FILENAMES
    for asset in CLASSIC_TABLE_ASSETS:
        if asset.filename in externalized:
            continue
        source = packaged / asset.filename
        if not source.is_file():
            pytest.skip(f"packaged table {asset.filename} is not in this tree")
        shutil.copyfile(source, root / asset.filename)

    env = os.environ.copy()
    env["GPUWM_THOMPSON_TABLE_ROOT"] = str(root)
    env[table_assets.ASSET_URL_BASE_ENV] = base
    done = _door(["fetch-tables"], env=env)

    assert done.returncode == 2, done.stdout + done.stderr
    assert "refused and deleted" in done.stdout, done.stdout
    assert "MiB" in done.stderr and "%" in done.stderr, done.stderr


# ---------------------------------------------------------------------------
# N10: the large route says what it is doing, and logs stream
# ---------------------------------------------------------------------------

def _fake_downloader(record):
    def download(url, dest, *, magic, opener=None, progress=None):
        record.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = (b"GRIB" + url.encode() + b"7777" if magic == "GRIB"
                else b"BZh" + url.encode())
        dest.write_bytes(body)
        if progress is not None:
            progress(len(body))
        return {"name": dest.name, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(), "url": url}
    return download


def test_the_table_route_names_each_object_it_lands(tmp_path):
    """The wrfvet's 792 MB fetch printed nothing between the opening
    line and the manifest.  Every object gets a line now, in submission
    order, so a reader can tell a slow transfer from a hung one."""

    from datetime import datetime

    plan = fetch_routes.resolve_request(
        "hrrr-prs", cycle=datetime(2026, 8, 16, 0), hours=1)
    said: list[str] = []
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=said.append)

    assert said, "the route said nothing at all"
    opening = said[0]
    assert plan.source_id in opening
    for obj in plan.objects:
        assert any(obj.relpath in line for line in said[1:]), (
            f"{obj.relpath} was never named", said)


def test_a_route_object_reports_its_bytes_while_they_move(local_host,
                                                          tmp_path):
    """The counter behind the per-object lines: a route with two big
    objects must not be silent for minutes between them."""

    base, body = local_host
    moved: list[int] = []
    entry = fetch_routes._download_object(
        f"{base}/object.grib2", tmp_path / "object.grib2",
        magic="GRIB", progress=moved.append)

    assert entry["bytes"] == len(body)
    assert len(moved) > 1, moved
    assert sum(moved) == len(body)


def test_front_door_stdout_is_line_buffered_through_a_pipe():
    """The GFS route's per-file lines existed and never reached the log
    until the command exited (9.1 s of 9.8 silent, measured).  Block
    buffering is the whole of that: a front door line-buffers its own
    stdout so a redirected run streams."""

    program = (
        "import sys, gpuwm.cli\n"
        "gpuwm.cli.main(['version', '--offline'])\n"
        "sys.stderr.write(f'LINEBUFFERED={sys.stdout.line_buffering}\\n')\n")
    done = subprocess.run([sys.executable, "-c", program],
                          capture_output=True, text=True, cwd=_REPO,
                          timeout=300)
    assert done.returncode == 0, done.stderr
    assert "LINEBUFFERED=True" in done.stderr, done.stderr


# ---------------------------------------------------------------------------
# N13: re-authoring identical content is idempotent; different content
#      refuses and names the remedy
# ---------------------------------------------------------------------------

_GFS_MAPPING = _REPO / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json"
_GFS_COMPOSITION = _REPO / "configs" / "rw-wps-gfs-terrain.composition.json"


def _fake_bridge_identity(path, _role):
    payload = Path(path).read_bytes()
    return {"bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _authoring_case(tmp_path: Path) -> dict:
    primary_a = tmp_path / "primary-a.grib2"
    primary_b = tmp_path / "primary-b.grib2"
    provenance = tmp_path / "terrain.md"
    inventory = tmp_path / "grib2-inventory"
    dump = tmp_path / "grib2-dump"
    for path, value in ((primary_a, b"a"), (primary_b, b"bb"),
                        (inventory, b"inventory"), (dump, b"dump")):
        path.write_bytes(value)
    provenance.write_text("terrain provenance\n", encoding="utf-8")
    return {
        "mapping_path": _GFS_MAPPING,
        "composition_path": _GFS_COMPOSITION,
        "primary_files": (primary_a, primary_b),
        "supplement_files": {"gfs_valid_time_terrain": (primary_a,
                                                        primary_b)},
        "provenance_files": {
            "gfs_valid_time_terrain_provenance": provenance},
        "grib2_inventory": inventory,
        "grib2_dump": dump,
    }


def test_re_authoring_the_same_manifest_is_idempotent(tmp_path, monkeypatch):
    """DATA.md calls prep-command.txt "runnable as written"; the second
    paste exited 78.  Identical content is the ordinary case -- the
    command was pasted twice -- and it now succeeds without rewriting a
    byte."""

    from gpuwm import mapped_authoring

    monkeypatch.setattr(mapped_authoring, "bridge_identity",
                        _fake_bridge_identity)
    case = _authoring_case(tmp_path)
    manifest = tmp_path / "inputs.json"

    first = mapped_authoring.author_input_manifest(manifest, **case)
    sealed = manifest.read_bytes()
    second = mapped_authoring.author_input_manifest(manifest, **case)

    assert manifest.read_bytes() == sealed
    assert second["manifest"]["sha256"] == first["manifest"]["sha256"]
    assert first["reauthored"] is True
    assert second["reauthored"] is False


def test_a_different_manifest_still_refuses_and_names_the_remedy(
        tmp_path, monkeypatch):
    """The property the refusal exists for: a manifest already on disk
    binds a run to bytes THIS authoring did not seal.  It keeps
    refusing, and now says what to type."""

    from gpuwm import mapped_authoring

    monkeypatch.setattr(mapped_authoring, "bridge_identity",
                        _fake_bridge_identity)
    case = _authoring_case(tmp_path)
    manifest = tmp_path / "inputs.json"
    manifest.write_text("someone else's manifest\n", encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(FileExistsError) as error:
        mapped_authoring.author_input_manifest(manifest, **case)

    message = str(error.value)
    assert str(manifest) in message
    assert "remedy:" in message, message
    assert manifest.name in message.split("remedy:", 1)[1], message
    assert manifest.read_bytes() == before


def test_a_refusal_that_is_a_sentence_is_not_labelled_twice():
    """The prep door labels a bare-path OSError with its class, because
    a path alone says nothing about what failed.  A refusal written as
    a sentence -- this one -- must keep its own words: `FileExistsError:
    refusing to overwrite ...` labels the same thing twice."""

    from gpuwm.source_cli import reads_as_a_sentence

    assert reads_as_a_sentence(
        "refusing to overwrite C:\\case\\inputs.json: it already holds a "
        "different input manifest")
    # Bare paths, including the one shape a space test would misread.
    assert not reads_as_a_sentence("C:\\Program Files\\gpuwm\\a.grib2")
    assert not reads_as_a_sentence("/opt/gpuwm/data/a.grib2")
    assert not reads_as_a_sentence("inputs.json")
    assert not reads_as_a_sentence("")


# ---------------------------------------------------------------------------
# N15: the upgrade is visible from the doors an upgrader checks
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_doctor_names_what_changed_since_the_version_it_last_saw(tmp_path):
    """The upgrader ran doctor on 2.5.0 and read a report shaped
    exactly like 2.4.1's.  A recorded previous version turns the next
    report into an upgrade note, once."""

    state = tmp_path / "doctor-state.json"
    state.write_text(json.dumps({"version": "2.4.1"}), encoding="utf-8")
    env = os.environ.copy()
    env["GPUWM_DOCTOR_STATE"] = str(state)

    first = _door(["doctor"], env=env)
    assert "2.4.1" in first.stdout, first.stdout
    for fragment in ("gpuwm sim", "gpuwm prep", "run-", "<product>"):
        assert fragment in first.stdout, (fragment, first.stdout)

    # Said once: the state now records this install, so an unchanged
    # install does not re-announce an upgrade at every doctor run.
    second = _door(["doctor"], env=env)
    assert "gpuwm sim" not in second.stdout, second.stdout
    assert json.loads(state.read_text(encoding="utf-8"))["version"]


def test_doctor_says_nothing_about_change_on_a_first_run(tmp_path):
    """No recorded previous version is not a changed version."""

    from gpuwm import doctor

    state = tmp_path / "doctor-state.json"
    assert doctor.upgrade_note("2.5.0", state) is None
    assert json.loads(state.read_text(encoding="utf-8"))["version"] == "2.5.0"


def test_run_help_points_at_the_unbundled_stages_and_the_layouts():
    """`gpuwm run --help` was byte-identical to 2.4.1's, so the doors
    that changed named 2.4.1 and the doors that did not said nothing."""

    done = _door(["run", "--help"])
    assert done.returncode == 0, done.stderr
    text = " ".join(done.stdout.split())
    for fragment in ("gpuwm prep", "gpuwm sim", "gpuwm render",
                     "run-", "<domain>", "<product>", "<valid-day>"):
        assert fragment in text, (fragment, text)


# ---------------------------------------------------------------------------
# N16 residue: the ahead-of-PyPI sentence says the index version once
# ---------------------------------------------------------------------------

def _wheel_shape(tmp_path: Path, version: str) -> dict:
    return {
        "package_root": tmp_path / "site" / "gpuwm",
        "source_root": tmp_path / "site",
        "distribution": "gpuwm",
        "version": version,
        "editable": False,
        "site_dir": tmp_path / "site",
        "git": {},
    }


def test_the_ahead_sentence_names_each_version_once(monkeypatch, capsys,
                                                    tmp_path):
    from gpuwm import version_cli

    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _wheel_shape(tmp_path, "2.5.0"))
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "2.4.1")
    assert cli_main(["version"]) == 0
    line = next(row for row in capsys.readouterr().out.splitlines()
                if "ahead of" in row)

    assert line.count("2.4.1") == 1, line
    assert line.count("2.5.0") == 1, line
    assert "source or pre-release install" in line


# ---------------------------------------------------------------------------
# N17: --source choices are the registry's, not three of sixteen
# ---------------------------------------------------------------------------

def test_doctor_source_choices_are_the_registry_the_prep_door_serves():
    from gpuwm import doctor
    from gpuwm.source_adapters import source_adapters

    registry = {adapter.source_id for adapter in source_adapters()}
    assert registry <= set(doctor.DOCTOR_SOURCES), (
        sorted(registry - set(doctor.DOCTOR_SOURCES)))


@pytest.mark.slow
def test_doctor_reports_the_route_the_walk_actually_used():
    """`gpuwm doctor --source hrrr-prs` was an argparse usage error on a
    route `gpuwm fetch` and `gpuwm prep` both serve."""

    done = _door(["doctor", "--source", "hrrr-prs", "--json"])
    assert "invalid choice" not in done.stderr, done.stderr
    assert done.returncode in (0, 1), done.stderr
    names = [check["name"] for check in json.loads(done.stdout)]
    assert any(name.startswith("hrrr-prs route") for name in names), names


# ---------------------------------------------------------------------------
# N24: the polish bundle
# ---------------------------------------------------------------------------

def test_dash_dash_version_is_the_version_door():
    """Typing what every other tool answers to was a usage error."""

    aliased = _door(["--version"])
    assert aliased.returncode == 0, aliased.stderr
    spelled = _door(["version"])
    assert aliased.stdout.splitlines()[0] == spelled.stdout.splitlines()[0]


def test_the_layout_line_uses_one_separator():
    """The printed line mixed the reader's own `\\` with the template's
    `/`, which reads as two paths glued together."""

    from gpuwm import render_layout

    described = render_layout.describe(os.path.join("case", "png"))
    head = described.split(" ", 1)[0]
    assert os.sep in head
    if os.sep != "/":
        assert "/" not in head, head


def test_the_alias_lines_collapse_to_one(monkeypatch):
    """Three consecutive `info pip extra aliases (3)` lines read as
    three findings and carry no information between them."""

    from gpuwm import doctor

    checks = [
        doctor.Check("gpuwm[gpu]", "info", "alias", group="pip extra aliases"),
        doctor.Check("pip extra [render]", "verified", "wrf-rust",
                     group="pip extras"),
        doctor.Check("gpuwm[all]", "info", "alias", group="pip extra aliases"),
        doctor.Check("pip extra [obs]", "verified", "pyshp",
                     group="pip extras"),
        doctor.Check("gpuwm[gpu-cu13]", "info", "alias",
                     group="pip extra aliases"),
    ]
    brief = doctor.format_brief(checks)

    assert brief.count("pip extra aliases") == 1, brief
    assert "pip extra aliases (3)" in brief, brief
