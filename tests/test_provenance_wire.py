"""The provenance policy at the shared config load, proven through it.

``tests/test_provenance_gate.py`` proves the gate's verdicts against
real install shapes.  This file proves the WIRING: that the verdicts
fire from inside ``gpuwm.experiment.build_experiment`` -- the one load
every front door shares, the same seam the nocturnal-radiation guard
uses -- so a door nobody remembered to wire still inherits the policy.
Every test here goes through ``build_experiment`` itself; a test that
called the gate directly would stay green with the seam call deleted,
which is exactly the mutation this file exists to catch.

The policy, in full:

* REFUSE only a VERSION DISAGREEMENT -- distribution metadata naming
  one release while the code's own ``pyproject.toml`` declares another,
  in either direction.  That is the "plots say 1.6.2 while 1.8.7
  executes" shape.
* WARN, one line, non-fatal, on a BORROWED version -- metadata resolved
  by name against a distribution that does not provide the executing
  code, digits agreeing.  That is every worktree beside an editable
  install (260 on the reference box); a refusal here bricks
  development, so the warning test asserts the load SUCCEEDS as hard as
  it asserts the line appears.  The line names which install the number
  came from.
* A wheel, and an idealized load (no ``[projection]``): silence.

Install shapes are the real-file builders from the gate's own suite --
a real ``.dist-info`` read by a real ``PathDistribution``, a real
``git init`` -- injected by monkeypatching ``provenance_gate.resolve``,
which is the seam's only source of provenance.
"""

from __future__ import annotations

import tomllib

import pytest
from test_provenance_gate import _checkout, _wheel

from gpuwm import provenance_gate
from gpuwm.experiment import build_experiment

#: A minimal REAL case: it has a [projection], so it has a place and a
#: clock, which is what makes it the seam's business (the nocturnal
#: guard draws the same line).  12Z start, 60 s run: broad daylight in
#: Kansas, so the neighbouring guard stays quiet and any refusal seen
#: here is provenance's.
REAL_TOML = """
[experiment]
name = "provwire-case"
start_time = 2021-06-01T12:00:00
run_seconds = 60.0
restart_interval_s = 0.0
# This probe is a REAL case running Noah with no radiation selector, which
# the two 1.8.8 radiation guards refuse at load unless it declares.  What
# is under test here is the PROVENANCE policy at the same shared loader,
# so the case declares rather than mutes: a fixture that dodged the guard
# would stop exercising the door this file is about.  Both tokens, because
# radiation entirely off under a land surface is the one class both guards
# are about and they make different claims.
acknowledgements = ["radiation-off-land-surface-v1",
                    "constant-downward-longwave-v1"]

[projection]
map_proj = "lambert"
ref_lat = 38.0
ref_lon = -97.0
truelat1 = 30.0
truelat2 = 45.0
stand_lon = -97.0

[shared]
nz = 8
ztop = 20000.0
map_proj = 1
moist = true
mp_physics = 8
diff_6th_opt = 2
diff_6th_factor = 0.12
sf_sfclay_physics = 1
sf_surface_physics = 2

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
history_interval_s = 60.0
e_we = 41
e_sn = 41
dx = 3000.0
dy = 3000.0
time_step = 15
specified = true
"""

SOURCE = "provwire-case.toml"


def _load_real():
    return build_experiment(tomllib.loads(REAL_TOML), source=SOURCE)


def _load_idealized():
    raw = tomllib.loads(REAL_TOML)
    del raw["projection"]
    raw["shared"]["map_proj"] = 0     # idealized: no projection table
    raw["experiment"]["name"] = "provwire-idealized"
    return build_experiment(raw, source="provwire-idealized.toml")


@pytest.fixture(autouse=True)
def _fresh_latches(monkeypatch):
    """Every test gets an unannounced process and an enabled banner.

    Teardown marks both once-per-process latches CONSUMED rather than
    fresh: a suite that runs after this file in the same process should
    keep seeing the ambient state every earlier config load produces --
    the line already printed -- not a surprise re-warning inside its own
    capture.
    """

    monkeypatch.delenv(provenance_gate.BANNER_ENV, raising=False)
    provenance_gate.reset_announcement()
    yield
    provenance_gate._ANNOUNCED = True
    provenance_gate._BORROWED_WARNED = True


# ---------------------------------------------------------------------------
# REFUSE: the two directions of a version disagreement
# ---------------------------------------------------------------------------

def test_the_config_load_refuses_stale_metadata_under_newer_code(
        tmp_path, monkeypatch):
    """Metadata 1.8.7, code declares 1.9.0: the tree moved on, pip's
    record did not.  The refusal must name both numbers, both
    locations, the remedy, and the config that was loading."""

    prov = _checkout(tmp_path, declared="1.9.0", installed="1.8.7",
                     reported="1.8.7")
    monkeypatch.setattr(provenance_gate, "resolve", lambda **_: prov)
    with pytest.raises(ValueError) as caught:
        _load_real()
    message = str(caught.value)
    assert f"experiment config {SOURCE}" in message
    assert "REFUSING" in message
    assert "1.9.0" in message and "1.8.7" in message
    # Both locations: the distribution's name and the tree's own
    # declaration file.
    assert "gpuwm" in message and "pyproject.toml" in message
    # And the remedy, naming THIS tree.
    assert "pip install -e" in message
    assert str(prov.source_root) in message


def test_the_config_load_refuses_newer_metadata_under_older_code(
        tmp_path, monkeypatch):
    """The inverse arm: metadata 1.9.0 over code that declares 1.8.7 --
    an upgrade pip recorded but an editable ``.pth`` never delivered."""

    prov = _checkout(tmp_path, declared="1.8.7", installed="1.9.0",
                     reported="1.9.0")
    monkeypatch.setattr(provenance_gate, "resolve", lambda **_: prov)
    with pytest.raises(ValueError) as caught:
        _load_real()
    message = str(caught.value)
    assert "REFUSING" in message
    assert "1.9.0" in message and "1.8.7" in message
    assert "pip install -e" in message


# ---------------------------------------------------------------------------
# WARN: borrowed but agreeing -- fires, names the lender, never refuses
# ---------------------------------------------------------------------------

def test_a_borrowed_agreeing_version_warns_once_and_the_run_continues(
        tmp_path, monkeypatch, capsys):
    """The dev-box shape: a worktree nobody installed, wearing the
    editable install's number.  One line, naming the install the number
    came from -- and the load MUST succeed, twice, with the line printed
    once, because a refusal here bricks every worktree session."""

    prov = _checkout(tmp_path, declared="1.8.7", installed=None,
                     reported="1.8.7")
    assert prov.metadata_is_borrowed is True
    monkeypatch.setattr(provenance_gate, "resolve", lambda **_: prov)
    monkeypatch.setattr(
        provenance_gate, "borrowed_version_origin",
        lambda: "the editable install of gpuwm '1.8.7' declared at "
                "file:///C:/elsewhere/gpuwm")
    first = _load_real()
    second = _load_real()
    assert first.domains and second.domains
    err = capsys.readouterr().err
    assert err.count("provenance: WARNING") == 1
    assert "borrowed" in err
    # WHICH install the number came from, not merely that it was loaned.
    assert "file:///C:/elsewhere/gpuwm" in err
    assert str(prov.source_root) in err


def test_the_borrowed_warning_is_not_a_refusal_in_disguise(
        tmp_path, monkeypatch):
    """``warn_borrowed_version`` itself must never raise, and must stand
    down entirely when the disagreement refusal owns the shape."""

    agreeing = _checkout(tmp_path / "a", declared="1.8.7", installed=None,
                         reported="1.8.7")
    assert provenance_gate.warn_borrowed_version(agreeing) is not None
    disagreeing = _checkout(tmp_path / "b", declared="1.9.0", installed=None,
                            reported="1.6.2")
    assert provenance_gate.warn_borrowed_version(disagreeing) is None


# ---------------------------------------------------------------------------
# SILENCE: the wheel, and the idealized load
# ---------------------------------------------------------------------------

def test_a_plain_wheel_loads_in_silence(tmp_path, monkeypatch, capsys):
    """``pip install gpuwm``, no git anywhere: the normal user case.
    No banner clause, no warning, no refusal -- not a word."""

    prov = _wheel(tmp_path, version="1.8.7")
    monkeypatch.setattr(provenance_gate, "resolve", lambda **_: prov)
    experiment = _load_real()
    assert experiment.domains
    err = capsys.readouterr().err
    assert "provenance" not in err
    assert "WARNING" not in err and "REFUSING" not in err


def test_an_idealized_load_is_not_the_gates_business(tmp_path, monkeypatch):
    """No [projection]: no place, no clock, and -- like the nocturnal
    guard beside it -- no provenance verdict.  Even a flagrantly
    disagreeing install loads an idealized case, because idealized and
    test paths are NOT a disagreement."""

    prov = _checkout(tmp_path, declared="1.9.0", installed="1.6.2",
                     reported="1.6.2")
    assert provenance_gate.version_identity_refusal(prov) is not None
    monkeypatch.setattr(provenance_gate, "resolve", lambda **_: prov)
    experiment = _load_idealized()
    assert experiment.projection is None
