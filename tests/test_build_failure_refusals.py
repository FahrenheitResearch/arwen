"""A build error must never wear a data error's face.

Measured 2026-08-20 in a fresh worktree whose Rust bridge had never been
built.  ``gpuwm check --alloc CONFIG`` reaches the ERA5 GRIB1 route,
which builds ``tools/grib1_bridge`` with cargo on first use.  With the
crate's ``gpuwm_preprocess_cpu.dll`` held open by another process --
another lane's run, an editor, a previous gpuwm still exiting -- cargo
cannot relink and fails::

    error: failed to remove file `...\\target\\release\\gpuwm_preprocess_cpu.dll`
    Caused by:
      The process cannot access the file because it is being used by
      another process. (os error 32)

Three things were wrong with what the reader then saw.

1. The sentence was ``could not decode/merge forcing inputs:`` followed
   by ten lines of unrelated Rust compiler WARNINGS, with the one line
   that names the cause last.  Nothing said "this is a build, not your
   data".
2. Five more failures followed, all of them false -- ``forcing
   inventory is missing [22 fields]``, ``forcing has no pressure
   levels`` -- measured against an EMPTY catalog because the decoder
   never ran.  A reader hunting a missing variable is hunting nothing.
3. In ``--json`` mode the whole report goes to stderr and stdout carries
   the memory document, which is written only if the input preflight
   returned zero.  So a caller running ``gpuwm check --alloc --json``
   got an EMPTY stdout and a ``json.loads`` exception: a build error
   reaching a program as a corrupt reply.

Each of the three is guarded below.
"""

from __future__ import annotations

import json
import types

import pytest

#: Cargo's REAL output for the locked-DLL failure, captured verbatim
#: from the reproduction (Windows, RTX 3080 box, 2026-08-20).  Kept
#: whole -- warning wall included -- because the point of the classifier
#: is that it finds the one line that matters inside exactly this.
LOCKED_DLL_OUTPUT = r"""warning: value assigned to `raw_data` is never read
   --> vendor\grib-core\src\grib2\parser.rs:571:33
    |
571 |     let mut raw_data: Vec<u8> = Vec::new();
    |                                 ^^^^^^^^^^
    |
    = help: maybe it is overwritten before being read?
    = note: `#[warn(unused_assignments)]` (part of `#[warn(unused)]`) on by default

warning: `grib-core` (lib) generated 1 warning
   Compiling grib1_bridge v0.1.0 (C:\gpuwm\tools\grib1_bridge)
error: failed to remove file `C:\gpuwm\tools\grib1_bridge\target\release\gpuwm_preprocess_cpu.dll`

Caused by:
  The process cannot access the file because it is being used by another process. (os error 32)
"""


def test_the_locked_artifact_failure_names_itself_and_not_the_warnings():
    from gpuwm import bridges

    message = bridges.cargo_build_refusal(
        "grib1_bridge", "tools/grib1_bridge",
        returncode=101, output=LOCKED_DLL_OUTPUT)
    lowered = message.lower()
    # The CLASS, in words a reader can act on.
    assert "another process" in lowered
    assert "build" in lowered
    # The artifact that could not be replaced.
    assert "gpuwm_preprocess_cpu.dll" in message
    # A remedy, and one that exists.
    assert "remedy:" in lowered
    # And NOT the wall: the unrelated compiler warning must not be what
    # the reader has to read past to reach the cause.
    assert "raw_data" not in message
    assert "unused_assignments" not in message


def test_a_held_build_lock_is_a_different_class_than_a_held_artifact():
    from gpuwm import bridges

    message = bridges.cargo_build_refusal(
        "grib1_bridge", "tools/grib1_bridge", returncode=101,
        output="Blocking waiting for file lock on build directory\n"
               "error: build failed")
    assert "lock" in message.lower()
    assert "remedy:" in message.lower()


def test_an_unclassified_cargo_failure_still_says_it_was_a_build():
    """No needle matched is not permission to relay a raw wall."""

    from gpuwm import bridges

    message = bridges.cargo_build_refusal(
        "grib1_bridge", "tools/grib1_bridge", returncode=101,
        output="warning: something\nerror: could not compile `grib-core`")
    lowered = message.lower()
    assert "build" in lowered and "cargo" in lowered
    assert "could not compile `grib-core`" in message
    assert "remedy:" in lowered


def test_the_grib1_bridge_route_raises_the_named_refusal(monkeypatch):
    """The real call site, with cargo's real failure under it."""

    from gpuwm import bridges
    from gpuwm.ingest import grib

    def fake_run(command, **kwargs):
        assert command[0] == "cargo"
        return types.SimpleNamespace(
            returncode=101, stdout="", stderr=LOCKED_DLL_OUTPUT)

    monkeypatch.delenv(bridges.BRIDGE_ENV["grib1_bridge"], raising=False)
    monkeypatch.setattr(grib.subprocess, "run", fake_run)
    with pytest.raises(bridges.BridgeBuildError) as excinfo:
        grib.build_rust_bridge()
    message = str(excinfo.value)
    assert "another process" in message.lower()
    assert "raw_data" not in message


def test_no_cargo_on_path_is_a_refusal_and_not_an_oserror(monkeypatch):
    """``cargo`` absent used to escape as a bare WinError 2 traceback."""

    from gpuwm import bridges
    from gpuwm.ingest import grib

    def fake_run(command, **kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.delenv(bridges.BRIDGE_ENV["grib1_bridge"], raising=False)
    monkeypatch.setattr(grib.subprocess, "run", fake_run)
    with pytest.raises(bridges.BridgeBuildError) as excinfo:
        grib.build_rust_bridge()
    lowered = str(excinfo.value).lower()
    assert "cargo" in lowered
    assert "remedy:" in lowered


def test_the_grib2_tool_route_raises_the_same_named_refusal(monkeypatch):
    """The second cargo call site shares the classifier, not a copy."""

    from gpuwm import bridges, mapped_source

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.setattr(
        mapped_source.subprocess, "run",
        lambda command, **kwargs: types.SimpleNamespace(
            returncode=101, stdout="", stderr=LOCKED_DLL_OUTPUT))
    if not (mapped_source._grib2_tools_crate() /  # noqa: SLF001
            "Cargo.toml").is_file():
        pytest.skip("no checkout crate here, so cargo is never reached")
    with pytest.raises(bridges.BridgeBuildError) as excinfo:
        mapped_source._build_grib2_tools()  # noqa: SLF001
    assert "another process" in str(excinfo.value).lower()


# --------------------------------------------------------------------------
# The cascade, and the malformed JSON
# --------------------------------------------------------------------------

def test_a_decoder_that_could_not_be_built_is_not_reported_as_missing_data():
    """The report says which failure is the cause and which are echoes."""

    from gpuwm.ingest.preflight import (PreflightIssue, PreflightReport,
                                        _empty_catalog)

    report = PreflightReport(
        _empty_catalog("ERA5"),
        (PreflightIssue("decoder-build",
                        "the GRIB1 decoder could not be built here: "
                        "another process is holding it"),
         PreflightIssue("inventory", "forcing inventory is missing ['T']"),
         PreflightIssue("levels", "forcing has no pressure levels")),
        ("resolved input SHA-256 catalog",))
    text = report.format()
    assert "decoder-build" in text
    # The naming, so nobody hunts a variable that was never looked for.
    assert "measured no data" in text or "never ran" in text


#: A real experiment TOML with a [case_data] table, so the preflight
#: reaches the decode step instead of refusing the config first.
_REAL_CONFIG = "configs/may1999_d01_smoke.toml"


def _config_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / _REAL_CONFIG


def test_check_json_emits_a_document_when_the_decoder_cannot_be_built(
        capsys, monkeypatch):
    """stdout must ALWAYS parse: a build error is not a corrupt reply.

    THE #241 defect.  ``gpuwm check --alloc --json`` printed the report
    to stderr, returned 1 before the memory estimator wrote anything,
    and left stdout EMPTY -- so a caller doing ``json.loads(stdout)``
    saw a JSON parse error where a build failure had happened.
    """

    from gpuwm.ingest import preflight

    def explode(*_args, **_kwargs):
        raise RuntimeError(
            "the Rust bridge `grib1_bridge` is not built here and building "
            "it FAILED: another process has "
            "target/release/gpuwm_preprocess_cpu.dll open")

    monkeypatch.setattr(preflight, "preflight_report", explode)
    args = types.SimpleNamespace(config=_config_path(), json=True)
    code = preflight._check_command(args)  # noqa: SLF001
    captured = capsys.readouterr()
    assert code != 0
    document = json.loads(captured.out)
    assert document["ok"] is False
    assert "gpuwm_preprocess_cpu.dll" in json.dumps(document)


def test_check_json_emits_a_document_for_an_ordinary_preflight_failure(
        capsys, monkeypatch):
    """The guarantee is about the CHANNEL, so it cannot be class-bound."""

    from gpuwm.ingest.preflight import (PreflightIssue, PreflightReport,
                                        _empty_catalog)
    from gpuwm.ingest import preflight

    monkeypatch.setattr(
        preflight, "preflight_report",
        lambda *_a, **_k: PreflightReport(
            _empty_catalog("ERA5"),
            (PreflightIssue("levels", "forcing has no pressure levels"),),
            ("resolved input SHA-256 catalog",)))
    args = types.SimpleNamespace(config=_config_path(), json=True)
    code = preflight._check_command(args)  # noqa: SLF001
    document = json.loads(capsys.readouterr().out)
    assert code != 0
    assert document["ok"] is False
    assert any(issue["code"] == "levels"
               for issue in document["failures"])
