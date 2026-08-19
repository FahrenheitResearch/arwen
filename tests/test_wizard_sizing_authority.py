"""`gpuwm domain` must size without CuPy, and say where its budget came from.

The 2.5.0 persona walks measured the breakage this file pins: the wizard
is command 1 of the README's two-command forecast, and it refused EVERY
box without CuPy -- including a caller who had already declared the card
with ``--card`` -- because the sizing estimator's import chain pulled
``gpuwm.core.physics``, whose module body imports cupy.  The refusal it
printed was a package-presence check naming an install line, for a
command that integrates nothing on a card.

The fixed contract, which these tests hold:

* ``--card <tier>`` / ``--vram-gib N``: size against the declaration and
  never touch cupy or the device.
* neither flag: MEASURE the local card through the short-lived probe
  subprocess (the measured-thresholds rule -- a real number beats an
  assumed tier), and say so on stdout.
* neither flag and nothing to measure: refuse by naming the real choice
  -- declare the card or make it measurable -- never with a package
  check for a dependency this command does not use.

The two subprocess tests run the REAL CLI in a real interpreter whose
``sitecustomize`` makes cupy unresolvable to both ``find_spec`` and
``import``, which is what a CPU-only wheel install looks like.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm import domain_wizard
from gpuwm.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]

GIB = 1024 ** 3

#: One wizard invocation, minus the sizing flags under test.  Same
#: point/cycle family as the go-chain fixtures.
_WIZARD = ("--point", "35.3,-97.5", "--source", "gfs",
           "--ladder", "12", "--cycle", "2026-07-29T18", "--hours", "6",
           "--physics-profile",
           "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")

_NO_CUPY_SITECUSTOMIZE = '''\
"""Make cupy unresolvable, as on a CPU-only install (test scaffolding)."""
import importlib.abc
import sys


class _Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in ("cupy", "cupy_backends", "cupyx"):
            raise ModuleNotFoundError(
                "No module named %r" % top, name=top)
        return None


sys.meta_path.insert(0, _Absent())
'''


def _run_real_cli_without_cupy(tmp_path: Path, *argv: str):
    """The real ``gpuwm`` CLI, in an interpreter where cupy is absent."""

    shadow = tmp_path / "no-cupy-site"
    shadow.mkdir(exist_ok=True)
    (shadow / "sitecustomize.py").write_text(
        _NO_CUPY_SITECUSTOMIZE, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow) + os.pathsep + str(REPO)
    # sitecustomize is imported by `site`, which PYTHONSAFEPATH does not
    # disable -- but the repo-root entry above IS what -m needs.
    env.pop("PYTHONSAFEPATH", None)
    # This file controls the variable explicitly per test; the suite's
    # own setting must not leak into the door under test.
    env.pop("GPUWM_NO_LOCAL_GPU", None)
    return subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", *argv],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
        timeout=600)


def test_card_declared_wizard_emits_with_no_cupy_on_the_box(tmp_path):
    """`gpuwm domain --card 16gb` on a CPU-only install writes the TOML.

    The declared tier answers the only question cupy ever answered for
    this command, so there is nothing left for a runtime gate to guard.
    """

    out = tmp_path / "declared.toml"
    done = _run_real_cli_without_cupy(
        tmp_path, "domain", *_WIZARD, "--card", "16gb", "--out", str(out))
    assert done.returncode == 0, done.stderr
    assert out.is_file()
    assert "needs cupy" not in done.stderr


def test_vram_declared_wizard_emits_with_no_cupy_on_the_box(tmp_path):
    """`--vram-gib N` is the same declaration in different units."""

    out = tmp_path / "declared-vram.toml"
    done = _run_real_cli_without_cupy(
        tmp_path, "domain", *_WIZARD, "--vram-gib", "16", "--out", str(out))
    assert done.returncode == 0, done.stderr
    assert out.is_file()


def test_bare_wizard_that_cannot_measure_names_the_real_choice(tmp_path):
    """No declaration, no cupy: the refusal names both ways forward.

    Not a package check -- the sentence has to say what the wizard
    needed (a VRAM budget) and the two remedies the caller can type.
    """

    out = tmp_path / "bare.toml"
    done = _run_real_cli_without_cupy(
        tmp_path, "domain", *_WIZARD, "--out", str(out))
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not out.exists()
    assert "--card" in done.stderr
    assert "install cupy" in done.stderr
    assert "measure the local card" in done.stderr
    # The old shape: a capability refusal for a dependency this command
    # does not use.  Its sentence must be gone from this door.
    assert "this command needs cupy" not in done.stderr


def test_bare_wizard_measures_the_local_card_when_it_can(
        tmp_path, monkeypatch, capsys):
    """The measured-thresholds rule: a readable card is read, and said.

    The probe seam answers with a 16 GiB card; the emission must match
    ``--vram-gib 16`` byte for byte, because the measurement is a budget
    source and never a different sizing path.
    """

    monkeypatch.setattr(
        domain_wizard, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": 12 * GIB,
                           "total_bytes": 16 * GIB,
                           "profile": None})
    measured = tmp_path / "measured.toml"
    assert cli_main(["domain", *_WIZARD, "--out", str(measured)]) == 0
    stdout = capsys.readouterr().out
    assert "measured" in stdout
    assert "16" in stdout

    declared = tmp_path / "declared.toml"
    assert cli_main(
        ["domain", *_WIZARD, "--vram-gib", "16", "--out", str(declared)]) == 0
    assert measured.read_bytes() == declared.read_bytes()


def test_bare_wizard_under_no_local_gpu_refuses_naming_the_variable(
        tmp_path, monkeypatch, capsys):
    """GPUWM_NO_LOCAL_GPU forbids the read; the refusal must say so.

    With the variable set the wizard may not open the device even to
    measure it, so a bare invocation has no budget source left and the
    refusal names the variable alongside the declaration remedy --
    rather than measuring anyway (the go-gate defect, N20) or refusing
    with a package check.
    """

    from gpuwm.core import preflight

    # The suite pins the wizard's probe to a deterministic card; this
    # test needs the real one, whose env gate is the thing under test.
    monkeypatch.setattr(
        domain_wizard, "device_memory_probe_subprocess",
        preflight.device_memory_probe_subprocess)
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    out = tmp_path / "gated.toml"
    code = cli_main(["domain", *_WIZARD, "--out", str(out)])
    err = capsys.readouterr().err
    assert code == 2
    assert not out.exists()
    assert "GPUWM_NO_LOCAL_GPU" in err
    assert "--card" in err


def test_declared_tier_never_asks_the_probe(tmp_path, monkeypatch):
    """With --card the wizard takes no measurement at all.

    The declaration says "size for a machine that need not be this
    one"; a probe would at best be ignored and at worst stand up device
    contact the caller opted out of.
    """

    def _boom(**_kwargs):
        raise AssertionError("--card must not probe the local device")

    monkeypatch.setattr(
        domain_wizard, "device_memory_probe_subprocess", _boom)
    out = tmp_path / "declared.toml"
    assert cli_main(
        ["domain", *_WIZARD, "--card", "24gb", "--out", str(out)]) == 0
    assert out.is_file()
