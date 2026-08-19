"""Output-integrity findings from the 2.5.0 UX persona walks (N5/N8/N12/N16).

Four defects, each measured through the real front door on the RC and
each reproduced here the same way before it was fixed:

* **N5** -- a FAILED ``gpuwm render`` (nonzero exit, zero PNGs) still
  created the stamped run folder and pointed ``latest-run.txt`` at it,
  so a script following the documented pointer could not tell a failed
  render from a successful empty one.  The folder and the pointer now
  publish only after at least one PNG lands.
* **N8** -- every refusal tail printed ``(run gpuwm <command> --explain
  for the reason)``, and typing exactly that is an argparse usage error
  because ``--explain`` is a modifier that needs the original
  arguments.  The tail now prints the reader's own full invocation with
  ``--explain`` appended, on every door.
* **N12** -- FIRST-LIGHT documents ``--bridge`` as optional and
  self-resolving at the prep door; the measured door exited 64 with
  ``invalid or missing run arguments: --bridge``.  The door now
  resolves it through the same staged-bridge ladder every other bridge
  uses, and refuses by name -- listing what it searched -- only when
  nothing resolves.
* **N16** -- ``gpuwm version`` on a 2.5.0 install said ``PyPI latest is
  2.4.1 -- this install is current.`` and kept advising ``pip install
  --upgrade``.  The ahead-of-PyPI case now has its own sentence and the
  upgrade advice is withheld when it would point backwards.

The doors are driven as subprocesses (``python -m gpuwm.cli``) wherever
the finding is CLI behaviour; ``version`` is driven through the real
dispatch in-process because its index answer must be controlled.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm import bridges, run_stamp, rustwx
from gpuwm.cli import main as cli_main
from gpuwm.physics_compat import THOMPSON_PROFILE_ID

_REPO = Path(__file__).resolve().parents[1]
_CLI = (sys.executable, "-m", "gpuwm.cli")


def _door(argv, *, env=None):
    """One real front-door invocation, captured."""

    return subprocess.run(
        [*_CLI, *argv], capture_output=True, text=True, cwd=_REPO,
        env=env, timeout=300)


def _clean_env(home: Path) -> dict:
    """An environment whose bridge estate is exactly ``home``.

    ``Path.home()`` follows USERPROFILE on Windows and HOME elsewhere,
    so both are redirected; every per-bridge override is dropped so the
    resolution ladder sees only what a test staged.
    """

    home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    for variable in ("RUSTWX_WRFBATCH", "RUSTWX_BASEMAP_DIR",
                     "GPUWM_NATIVE_DISTRIBUTION_MANIFEST",
                     rustwx.RENDERER_ENV,
                     *bridges.BRIDGE_ENV.values()):
        env.pop(variable, None)
    return env


# ---------------------------------------------------------------------------
# N8: the refusal tail is the reader's own line, re-runnable as printed
# ---------------------------------------------------------------------------

def _pointer(argv) -> str:
    """The tail the fixed doors print: the invocation plus one flag."""

    return "(run gpuwm " + " ".join(argv) + " --explain for the reason)"


def _nocturnal_config(tmp_path: Path) -> Path:
    """A config every front door refuses at load, offline, layered.

    The nocturnal-radiation guard's own shape (shortwave on, longwave
    off, a 48 h window with local night, no acknowledgement), emitted by
    the wizard's own renderer so it is a config a real user could hold.
    """

    from gpuwm.domain_wizard import render_config

    text = render_config(
        name="uxtail", start_time=datetime.datetime(2011, 4, 26, 12),
        hours=48,
        projection={"map_proj": "lambert", "ref_lat": 33.8,
                    "ref_lon": -87.29, "truelat1": 23.8, "truelat2": 43.8,
                    "stand_lon": -87.29},
        dims=[(120, 100)], ratios=(), fetch_hints={"source": "gfs"},
        case_data=None, profile=THOMPSON_PROFILE_ID, acknowledgements=())
    path = tmp_path / "night.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_domain_refusal_tail_is_the_users_own_line(tmp_path):
    """The noob's dead end, measured: the printed tail was itself a
    usage error.  Whatever layered refusal fires first on this box (the
    nocturnal acknowledgement or a capability gap), the tail must be the
    reader's own domain line."""

    argv = ["domain", "--point=33.8,-87.29", "--source", "era5",
            "--cycle", "2011-04-26T12", "--hours", "48",
            "--physics-profile", THOMPSON_PROFILE_ID,
            "--out", str(tmp_path / "cfg.toml")]
    done = _door(argv)
    assert done.returncode == 2, done.stderr
    assert _pointer(argv) in done.stderr, done.stderr


def test_the_check_refusal_tail_is_the_users_own_line(tmp_path):
    config = _nocturnal_config(tmp_path)
    argv = ["check", str(config)]
    done = _door(argv)
    assert done.returncode == 2, done.stderr
    assert _pointer(argv) in done.stderr, done.stderr


def test_the_go_refusal_tail_is_the_users_own_line(tmp_path):
    config = _nocturnal_config(tmp_path)
    argv = ["go", str(config), "--outdir", str(tmp_path / "case")]
    done = _door(argv)
    assert done.returncode == 2, done.stderr
    assert _pointer(argv) in done.stderr, done.stderr


def test_the_render_refusal_tail_is_the_users_own_line(
        tmp_path, monkeypatch, capsys):
    """On an estate with no renderer, ``--engine auto`` refuses under
    the render law -- a layered refusal, so it carries the tail.

    The estate is arranged at the resolver seam (the ladder suites'
    established pattern) because a source checkout carrying its own
    built ``tools/rustwx`` renderer cannot present an empty estate
    through the environment alone; ``cli_main`` is still the real door
    and records the invocation from these tokens."""

    monkeypatch.setattr(rustwx, "find_renderer", lambda: None)
    argv = ["render", str(tmp_path / "wrfout_d01.nc"),
            "--out", str(tmp_path / "png")]
    rc = cli_main(argv)
    err = capsys.readouterr().err
    assert rc == 2, err
    assert _pointer(argv) in err, err


def test_the_report_refusal_tail_is_the_users_own_line(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    argv = ["report", str(empty), "--dry-run"]
    done = _door(argv)
    assert done.returncode == 2, done.stderr
    assert _pointer(argv) in done.stderr, done.stderr


def test_the_helper_falls_back_to_the_command_name_outside_the_cli():
    """A door reached without the ``gpuwm`` dispatcher (``python -m``
    doors, embedders, this test suite calling ``explain.render``
    directly) has no recorded invocation, and the pointer then names
    the command as before rather than guessing."""

    from gpuwm import explain

    message = explain.layered("refused", "the mechanism")
    terse = explain.render(message, explain=False, command="gpuwm fetch")
    assert "gpuwm fetch --explain" in terse


# ---------------------------------------------------------------------------
# N5: a failed render publishes neither the run folder nor the pointer
# ---------------------------------------------------------------------------

@pytest.fixture()
def science_core():
    """The matplotlib fallback engine computes through ``wrf``; these
    tests drive that engine explicitly so the failing render gets PAST
    every front-door refusal and fails in the rendering itself."""

    return pytest.importorskip(
        "wrf", reason="gpuwm render requires the wrf package (wrf-rust)")


def _run_folders(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(child for child in root.iterdir()
                  if child.is_dir() and run_stamp.is_run_folder(child))


def _corrupt_wrfout(tmp_path: Path) -> Path:
    path = tmp_path / "wrfout_d01_2026-05-17_18-00-00.nc"
    path.write_bytes(b"this is not a netcdf file")
    return path


def _good_wrfout(directory: Path) -> Path:
    """A small wrfout the matplotlib arm can actually draw ``t2`` from."""

    from gpuwm.io.wrfout import WrfoutWriter
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = 2, 8, 10
    path = directory / "wrfout_d01_2026-05-17_18-00-00.nc"
    lat = np.tile(np.linspace(38.0, 39.0, ny)[:, None], (1, nx))
    lon = np.tile(np.linspace(-98.0, -97.0, nx)[None, :], (ny, 1))
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                      global_attrs={
                          "GRID_ID": 1,
                          "SIMULATION_START_DATE": "2026-05-17_18:00:00"},
                      ) as writer:
        writer.write_frame("2026-05-17_18:00:00", {
            "T": np.zeros((nz, ny, nx), np.float32),
            "MU": np.zeros((ny, nx), np.float32),
            "T2": np.full((ny, nx), 290.0, np.float32),
            "XLAT": lat.astype(np.float32),
            "XLONG": lon.astype(np.float32),
            "HGT": np.zeros((ny, nx), np.float32),
            "SINALPHA": np.zeros((ny, nx), np.float32),
            "COSALPHA": np.ones((ny, nx), np.float32),
        })
    return path


def test_a_failed_render_publishes_no_folder_and_no_pointer(
        science_core, tmp_path, capsys):
    """The upgrader's finding verbatim: exit nonzero, zero PNGs -- and
    the case root afterwards must hold no new run folder and no
    ``latest-run.txt`` naming one."""

    out = tmp_path / "png"
    rc = cli_main(["render", str(_corrupt_wrfout(tmp_path)),
                   "--out", str(out), "--engine", "matplotlib",
                   "--products", "t2"])
    captured = capsys.readouterr()
    assert rc == 1, captured.err
    # The run got past every refusal and failed rendering itself.
    assert "unreadable wrfout" in captured.err
    assert _run_folders(out) == [], (
        "a render that drew nothing left a stamped run folder behind")
    assert not (out / run_stamp.LATEST_POINTER).exists(), (
        "a render that drew nothing published latest-run.txt")


def test_a_failed_render_does_not_move_an_existing_pointer(
        science_core, tmp_path, capsys):
    """A case root with one good run: the failed render may not touch
    the pointer a script is following."""

    good = _good_wrfout(tmp_path)
    out = tmp_path / "png"
    assert cli_main(["render", str(good), "--out", str(out),
                     "--engine", "matplotlib", "--products", "t2"]) == 0
    capsys.readouterr()
    folders = _run_folders(out)
    assert len(folders) == 1
    pointer = out / run_stamp.LATEST_POINTER
    assert pointer.read_text(encoding="utf-8").strip() == folders[0].name, (
        "a successful render must still publish the pointer")

    rc = cli_main(["render", str(_corrupt_wrfout(tmp_path)),
                   "--out", str(out), "--engine", "matplotlib",
                   "--products", "t2"])
    capsys.readouterr()
    assert rc == 1
    assert _run_folders(out) == folders, (
        "the failed render left a second run folder behind")
    assert pointer.read_text(encoding="utf-8").strip() == folders[0].name, (
        "the failed render moved latest-run.txt off the last good run")


def test_a_failed_render_with_run_stamp_off_stays_flat(
        science_core, tmp_path, capsys):
    """The flat/off arm measured clean in the walk must stay clean."""

    out = tmp_path / "png"
    rc = cli_main(["render", str(_corrupt_wrfout(tmp_path)),
                   "--out", str(out), "--engine", "matplotlib",
                   "--products", "t2", "--run-stamp", "off"])
    capsys.readouterr()
    assert rc == 1
    assert _run_folders(out) == []
    assert not (out / run_stamp.LATEST_POINTER).exists()


# ---------------------------------------------------------------------------
# N21: the documented pointer location is where the writer writes
# ---------------------------------------------------------------------------

def test_render_publishes_the_pointer_under_out_not_the_case_root(
        science_core, tmp_path, capsys):
    """The upgrader's finding: ``docs/run-output-folders.md`` told a
    script to read ``latest-run.txt`` in the CASE directory, and a
    render-only workflow writes it under ``--out``.  The snippet found
    nothing, which reads as "no run happened" rather than "you looked one
    level too high"."""

    case = tmp_path / "myarea"
    out = case / "png"
    assert cli_main(["render", str(_good_wrfout(case)), "--out", str(out),
                     "--engine", "matplotlib", "--products", "t2"]) == 0
    capsys.readouterr()

    folders = _run_folders(out)
    assert len(folders) == 1
    assert (out / run_stamp.LATEST_POINTER).is_file(), (
        "the pointer is not under --out, where the doc now sends readers")
    assert not (case / run_stamp.LATEST_POINTER).exists(), (
        "the pointer is not at the case root, where the doc used to send "
        "readers")
    assert run_stamp.latest(out) == folders[0]


def test_the_run_folder_doc_sends_readers_where_render_writes():
    """The doc and the writer cannot drift apart again silently."""

    page = (_REPO / "docs" / "run-output-folders.md").read_text(
        encoding="utf-8")
    assert "out/myarea/png/latest-run.txt" in page, (
        "the render-only pointer path is not shown anywhere in the doc")
    # And the bare case-root read is no longer offered as the way to find
    # a render's output.
    snippet = "run=$(cat out/myarea/latest-run.txt)"
    assert snippet in page and "after gpuwm go" in page, (
        "the case-root read must stay, labelled with the door it belongs to")


# ---------------------------------------------------------------------------
# N12: the prep door resolves --bridge through the staged-bridge ladder
# ---------------------------------------------------------------------------

def _staged_bridge(home: Path, name: str) -> Path:
    """A bridge staged the way ``gpuwm fetch-bridges`` stages one, with
    the REAL contract marker so the resolver's static ABI gate passes on
    genuine evidence."""

    staged = home / ".gpuwm" / "bridges"
    staged.mkdir(parents=True, exist_ok=True)
    path = staged / bridges.executable_name(name)
    path.write_bytes(b"MZ\0\0" + bridges.BRIDGE_ABI_MARKERS[name])
    return path


def _gfs_prep_argv(*extra: str) -> list[str]:
    return [
        "prep", "--source", "gfs",
        "--gfs-series", "series.tsv",
        "--cycle", "2026-07-29_06:00:00",
        "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--source-manifest", "manifest.json",
        "--source-manifest-sha256", "a" * 64,
        "--output-root", "prep-out",
        "--geog-root", "geog",
        *extra,
        "--dry-run",
    ]


def _ladder_answer(name: str, home: Path) -> Path:
    """The door subprocess's ladder answer: its first existing rung.

    The candidate list is asked of :mod:`gpuwm.bridges` itself so this
    cannot drift from the real ladder; only the HOME-anchored last rung
    is re-based onto the home ``_clean_env`` redirects the door to.  On
    a bare install the staged copy is the answer; a source checkout
    carrying its own built bridge deliberately outranks it ("a
    developer's rebuild must win"), and the door must name whichever
    the ladder names."""

    env_var = bridges.BRIDGE_ENV[name]
    filename = bridges.executable_name(name)
    assert not os.environ.get(env_var), (
        f"{env_var} is set in the ambient environment; this test needs "
        "the override rung empty to predict the ladder")
    candidates = list(bridges.artifact_candidates(env_var, filename))
    candidates[-1] = home / ".gpuwm" / "bridges" / filename
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise AssertionError("unreachable: the staged rung was just written")


def test_prep_gfs_resolves_the_staged_bridge_without_the_flag(tmp_path):
    """FIRST-LIGHT's sentence, made true: omitted, --bridge resolves
    through the ladder to the bridge this install has -- the staged
    copy on a bare box, the checkout's own build where one exists."""

    home = tmp_path / "home"
    _staged_bridge(home, "gfs_grib2_bridge")
    expected = _ladder_answer("gfs_grib2_bridge", home)
    done = _door(_gfs_prep_argv(), env=_clean_env(home))
    assert "invalid or missing run arguments" not in done.stderr, done.stderr
    assert done.returncode == 0, done.stderr
    # The composed command normalizes separators; compare in kind.
    assert str(expected).replace("\\", "/") in done.stdout.replace("\\", "/")


def test_prep_era5_resolves_the_staged_bridge_without_the_flag(tmp_path):
    home = tmp_path / "home"
    _staged_bridge(home, "grib1_bridge")
    expected = _ladder_answer("grib1_bridge", home)
    argv = [
        "prep", "--source", "era5",
        "--grib", "era5.grb",
        "--vtable", "Vtable.ERA5",
        "--wps-namelist", "namelist.wps",
        "--static-input", "static.npz",
        "--static-receipt", "static-receipt.json",
        "--source-orography", "met_em.nc",
        "--experiment-config", "experiment.toml",
        "--source-manifest", "manifest.json",
        "--source-manifest-sha256", "a" * 64,
        "--output-root", "prep-out",
        "--dry-run",
    ]
    done = _door(argv, env=_clean_env(home))
    assert "invalid or missing run arguments" not in done.stderr, done.stderr
    assert done.returncode == 0, done.stderr
    assert str(expected).replace("\\", "/") in done.stdout.replace("\\", "/")


def test_prep_gfs_names_the_ladder_when_nothing_resolves(monkeypatch, capsys):
    """Genuinely unresolvable: the refusal names what it looked for,
    like the other ladder doors -- never a bare demand for a flag.

    Arranged at the resolver's find seam (the ladder suites'
    established pattern), because a source checkout carrying its own
    built bridges cannot present an empty estate through the
    environment alone; the refusal text still comes from the real
    resolver over the real candidate list."""

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    rc = cli_main(_gfs_prep_argv())
    err = capsys.readouterr().err
    assert rc != 0
    assert "invalid or missing run arguments" not in err, err
    assert "gfs_grib2_bridge" in err, err
    assert "Searched, in order" in err, err


def test_prep_gfs_an_explicit_bridge_still_overrides_the_ladder(tmp_path):
    home = tmp_path / "home"
    _staged_bridge(home, "gfs_grib2_bridge")
    chosen = tmp_path / "my-own-bridge.exe"
    chosen.write_bytes(
        b"MZ\0\0" + bridges.BRIDGE_ABI_MARKERS["gfs_grib2_bridge"])
    done = _door(_gfs_prep_argv("--bridge", str(chosen)),
                 env=_clean_env(home))
    assert done.returncode == 0, done.stderr
    assert str(chosen).replace("\\", "/") in done.stdout.replace("\\", "/")


# ---------------------------------------------------------------------------
# N16: gpuwm version names the ahead-of-PyPI case and drops the advice
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


def test_version_names_the_ahead_of_pypi_case(monkeypatch, capsys, tmp_path):
    """Measured on the RC: a 2.5.0 install was called "current" against
    a 2.4.1 index, with pip upgrade advice attached."""

    from gpuwm import version_cli

    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _wheel_shape(tmp_path, "2.5.0"))
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "2.4.1")
    assert cli_main(["version"]) == 0
    out = capsys.readouterr().out
    # Each version number once: the first spelling of this sentence
    # named the index's version twice (the N16 residue the polish lane
    # closed), which reads as an unsubstituted template.
    assert "ahead of it" in out, out
    assert "source or pre-release install" in out, out
    assert "this install is current" not in out, out
    assert "pip install --upgrade" not in out, (
        "the upgrade advice still prints on an install that is ahead "
        "of the index it would upgrade from")


def test_version_keeps_the_advice_when_the_install_is_behind(
        monkeypatch, capsys, tmp_path):
    from gpuwm import version_cli

    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _wheel_shape(tmp_path, "2.4.1"))
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "2.5.0")
    assert cli_main(["version"]) == 0
    out = capsys.readouterr().out
    assert "behind" in out, out
    assert "pip install --upgrade" in out, out


def test_version_keeps_the_advice_when_the_index_is_silent(
        monkeypatch, capsys, tmp_path):
    """Offline is not ahead: with no index answer there is no basis to
    withhold the one command that upgrades a wheel."""

    from gpuwm import version_cli

    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _wheel_shape(tmp_path, "2.5.0"))
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: None)
    assert cli_main(["version"]) == 0
    out = capsys.readouterr().out
    assert "pip install --upgrade" in out, out

    monkeypatch.setattr(
        version_cli, "pypi_latest",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError(
            "--offline still reached the index")))
    assert cli_main(["version", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "pip install --upgrade" in out, out


def test_version_still_calls_a_current_wheel_current(
        monkeypatch, capsys, tmp_path):
    from gpuwm import version_cli

    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _wheel_shape(tmp_path, "2.5.0"))
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "2.5.0")
    assert cli_main(["version"]) == 0
    out = capsys.readouterr().out
    assert "this install is current" in out, out
