# tests/test_les_mixing_autoswitch.py
"""The anisotropic-K remedy is DEFAULT-ON, as an auto-switch (Drew, 2026-08-16).

The criterion is unchanged and still lives in one place --
``gpuwm.config.anisotropic_w_mixing_ratio``,
``mix_upper_bound*(dz_max/dx)^2`` against 0.25 -- but what happens when a
bare config violates it changes: a domain that LEAVES ``mix_isotropic``
UNSET (or writes the sentinel ``"auto"``) now runs the isotropic
``(dx*dy*dz)^(1/3)`` length instead of being advised about the
anisotropic one it never asked for.  A domain that writes 0 or 1 keeps
exactly what it wrote -- 0 in the danger zone with a blunt warning that
names the instability, the measured ratio, and the override state.

The four contracts pinned here:

* AUTO-SWITCH: unset + criterion violated -> ``mix_isotropic = 1``,
  recorded on ``ExperimentConfig.auto_mix_isotropic``, one loud line
  naming the ratio, the limit, and the selection.
* SAFE GRIDS UNTOUCHED: unset + criterion satisfied -> 0, silent, and
  the resolved experiment is IDENTICAL (restart identity included) to
  the same file with ``mix_isotropic = 0`` written out.
* EXPLICIT OVERRIDE RESPECTED: written values survive, both of them, and
  the label never binds identity -- an auto-selected 1 and a written 1
  carry the same fingerprint.
* RESTART HONESTY: both restart doors say plainly why a checkpoint
  written under the old anisotropic default will not bit-continue.

CPU only, no card, no data.
"""
from __future__ import annotations

import dataclasses
import math
import textwrap
import tomllib
from pathlib import Path

import pytest

from gpuwm.config import (EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT,
                          MIX_ISOTROPIC_RESTART_BREAK_NOTICE, RunConfig)
from gpuwm.core.grid import base_layer_depths
from gpuwm.experiment import load_experiment

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

MIX_UPPER_BOUND = 0.1

#: The synthetic single-domain tree of the advisory suite, with the
#: ``mix_isotropic`` line made OPTIONAL so unset / "auto" / 0 / 1 are all
#: writable from one template.  Radiation off for the same reason as
#: there: the subject is mixing, not the radiation level-count preflight.
_TREE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 600.0
restart_interval_s = 0.0

[shared]
nz = {nz}
ztop = 20000.0
p_top = {p_top}
eta_levels = {eta}
km_opt = {km_opt}
ra_lw_physics = 0
ra_sw_physics = 0
bl_pbl_physics = 0
mix_upper_bound = {mub}
{shared_mix}
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 40
ny = 40
time_step = 2
dx = {dx}
history_interval_s = 600.0
{domain_mix}"""

_NZ = 8
_P_TOP = 50000.0
_ETA = [1.0 - i / _NZ for i in range(_NZ)] + [0.0]


def _dz_max() -> float:
    return float(base_layer_depths(_ETA, 0, 0.2, _P_TOP).max())


def _dx_for_ratio(ratio: float) -> float:
    return _dz_max() * math.sqrt(MIX_UPPER_BOUND / ratio)


def _write(tmp_path, *, dx: float, shared_mix: str = "", domain_mix: str = "",
           km_opt: int = 3) -> Path:
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(_TREE).format(
        nz=_NZ, p_top=_P_TOP, eta=repr(_ETA), km_opt=km_opt,
        mub=repr(MIX_UPPER_BOUND), dx=repr(dx),
        shared_mix=shared_mix + "\n" if shared_mix else "",
        domain_mix=domain_mix + "\n" if domain_mix else ""))
    return path


OVER = 2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
UNDER = 0.5 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT


# ---------------------------------------------------------------------------
# The switch itself.
# ---------------------------------------------------------------------------

def test_an_unset_mixing_length_switches_to_isotropic_in_the_danger_zone(
        tmp_path, capsys):
    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER)))
    assert exp.domains[0].run.mix_isotropic == 1
    assert exp.auto_mix_isotropic == (1,)
    err = capsys.readouterr().err
    assert err.count("warning:") == 1
    # The loud line: what happened, and why -- the criterion value, the
    # limit, and that isotropic mixing was selected.
    assert "SELECTS ISOTROPIC MIXING" in err
    assert "mix_isotropic = 1" in err
    assert "mix_upper_bound*(dz_max/dx)^2" in err
    assert f"{OVER:.3g}" in err
    assert str(EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT) in err
    assert "UNSET" in err
    # It replaces the old advisory paragraph; it does not stack on it.
    assert "ADVISORY, not a refusal" not in err


def test_the_auto_string_is_the_same_as_leaving_it_unset(tmp_path, capsys):
    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                 shared_mix='mix_isotropic = "auto"'))
    assert exp.domains[0].run.mix_isotropic == 1
    assert exp.auto_mix_isotropic == (1,)
    assert "SELECTS ISOTROPIC MIXING" in capsys.readouterr().err


def test_a_domain_auto_overrides_an_explicit_shared_zero(tmp_path, capsys):
    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                 shared_mix="mix_isotropic = 0",
                                 domain_mix='mix_isotropic = "auto"'))
    assert exp.domains[0].run.mix_isotropic == 1
    assert exp.auto_mix_isotropic == (1,)
    assert "SELECTS ISOTROPIC MIXING" in capsys.readouterr().err


def test_a_misspelled_sentinel_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="mix_isotropic"):
        load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                               shared_mix='mix_isotropic = "Auto"'))


# ---------------------------------------------------------------------------
# Safe grids untouched, and identically so.
# ---------------------------------------------------------------------------

def test_a_safe_grid_is_untouched_and_identical_to_explicit_zero(
        tmp_path, capsys):
    from gpuwm.core.model import restart_identity_payload

    auto = load_experiment(_write(tmp_path, dx=_dx_for_ratio(UNDER)))
    assert auto.domains[0].run.mix_isotropic == 0
    assert auto.auto_mix_isotropic == (1,), (
        "the label records WHO chose, even when the choice is 0")
    assert capsys.readouterr().err == ""

    explicit = load_experiment(_write(tmp_path, dx=_dx_for_ratio(UNDER),
                                      shared_mix="mix_isotropic = 0"))
    # Everything a trajectory reads is equal: the whole config minus the
    # provenance label, and the restart identity with nothing removed.
    assert dataclasses.replace(auto, auto_mix_isotropic=()) == explicit
    assert restart_identity_payload(auto) == restart_identity_payload(explicit)


def test_the_switched_experiment_is_identical_to_explicit_one(tmp_path):
    from gpuwm.core.model import restart_identity_payload

    auto = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER)))
    explicit = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                      shared_mix="mix_isotropic = 1"))
    assert dataclasses.replace(auto, auto_mix_isotropic=()) == explicit
    assert restart_identity_payload(auto) == restart_identity_payload(explicit)


def test_the_label_never_reaches_the_restart_identity(tmp_path):
    from gpuwm.core.model import restart_identity_payload

    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER)))
    assert "auto_mix_isotropic" not in restart_identity_payload(exp)


# ---------------------------------------------------------------------------
# Explicit settings survive, both of them.
# ---------------------------------------------------------------------------

def test_an_explicit_zero_is_kept_with_the_blunt_override_warning(
        tmp_path, capsys):
    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                 shared_mix="mix_isotropic = 0"))
    assert exp.domains[0].run.mix_isotropic == 0
    assert exp.auto_mix_isotropic == ()
    err = capsys.readouterr().err
    assert "warning:" in err
    # The breakage, by name and by number...
    assert "mix_upper_bound*(dz_max/dx)^2" in err
    assert f"{OVER:.3g}" in err
    # ...and the override state: kept because the config said so.
    assert "OVERRIDE STATE" in err
    assert "EXPLICIT setting" in err
    assert "SELECTS ISOTROPIC MIXING" not in err


def test_an_explicit_one_is_untouched_and_unmarked(tmp_path, capsys):
    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                 shared_mix="mix_isotropic = 1"))
    assert exp.domains[0].run.mix_isotropic == 1
    assert exp.auto_mix_isotropic == ()
    assert capsys.readouterr().err == ""


def test_an_unresolvable_depth_does_not_switch(tmp_path, capsys):
    """No number, no switch: the criterion cannot say the grid violates
    it, so the WRF-default anisotropic form stays and the existing
    no-number advisory fires (without the forced-override clause -- the
    user forced nothing)."""

    from gpuwm.config import UNRESOLVED_ANISOTROPIC_DEPTH_MARK

    path = _write(tmp_path, dx=_dx_for_ratio(OVER))
    text = "\n".join(line for line in path.read_text().splitlines()
                     if not line.startswith(("p_top =", "eta_levels =")))
    text = text.replace("ztop = 20000.0", "ztop = 26000.0")
    path.write_text(text)
    exp = load_experiment(path)
    assert exp.domains[0].run.mix_isotropic == 0
    assert exp.auto_mix_isotropic == (1,)
    err = capsys.readouterr().err
    assert UNRESOLVED_ANISOTROPIC_DEPTH_MARK in err
    assert "SELECTS ISOTROPIC MIXING" not in err
    assert "OVERRIDE STATE" not in err


# ---------------------------------------------------------------------------
# gpuwm check tells the truth about what the run WILL do.
# ---------------------------------------------------------------------------

def test_gpuwm_check_reports_the_selection_in_the_danger_zone(tmp_path):
    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER)))
    lines = [line for line in check_advisories(exp)
             if "mix_isotropic" in line]
    assert len(lines) == 1
    assert lines[0].startswith("d01 ")
    assert "SELECTS ISOTROPIC MIXING" in lines[0]
    assert f"{OVER:.3g}" in lines[0]
    assert str(EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT) in lines[0]


def test_gpuwm_check_reports_the_override_state_when_anisotropic_is_forced(
        tmp_path, capsys):
    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                 shared_mix="mix_isotropic = 0"))
    capsys.readouterr()
    lines = [line for line in check_advisories(exp)
             if "mix_upper_bound*(dz_max/dx)^2" in line]
    assert len(lines) == 1
    assert "OVERRIDE STATE" in lines[0]
    assert "EXPLICIT setting" in lines[0]


def test_gpuwm_check_stays_silent_for_explicit_isotropic(tmp_path):
    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(OVER),
                                 shared_mix="mix_isotropic = 1"))
    assert [line for line in check_advisories(exp)
            if "mix" in line.lower()] == []


def test_gpuwm_check_stays_silent_on_a_safe_auto_grid(tmp_path):
    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write(tmp_path, dx=_dx_for_ratio(UNDER)))
    assert [line for line in check_advisories(exp)
            if "mix" in line.lower()] == []


# ---------------------------------------------------------------------------
# Restart honesty, both doors.
# ---------------------------------------------------------------------------

def _run_config(**overrides) -> RunConfig:
    base = dict(nx=40, ny=40, nz=8, dx=100.0, dy=100.0, ztop=20000.0,
                dt=1.0, run_seconds=600.0)
    base.update(overrides)
    return RunConfig(**base)


def test_the_single_domain_restart_door_names_the_autoswitch():
    from gpuwm.io.restart import RestartMismatchError, _require_config_match

    stored = dataclasses.asdict(_run_config(mix_isotropic=0))
    live = _run_config(mix_isotropic=1)
    with pytest.raises(RestartMismatchError) as caught:
        _require_config_match(stored, live, "restart_0001.npz")
    message = str(caught.value)
    assert "mix_isotropic: restart=0 run=1" in message
    assert MIX_ISOTROPIC_RESTART_BREAK_NOTICE in message


def test_the_notice_stays_out_of_unrelated_mismatches():
    from gpuwm.io.restart import RestartMismatchError, _require_config_match

    stored = dataclasses.asdict(_run_config(km_opt=4))
    live = _run_config(km_opt=1)
    with pytest.raises(RestartMismatchError) as caught:
        _require_config_match(stored, live, "restart_0001.npz")
    assert MIX_ISOTROPIC_RESTART_BREAK_NOTICE not in str(caught.value)
    # And the other direction of the flip is not the auto-switch story.
    stored = dataclasses.asdict(_run_config(mix_isotropic=1))
    with pytest.raises(RestartMismatchError) as caught:
        _require_config_match(stored, _run_config(mix_isotropic=0),
                              "restart_0001.npz")
    assert MIX_ISOTROPIC_RESTART_BREAK_NOTICE not in str(caught.value)


def test_the_tree_restart_reason_names_the_autoswitch():
    from gpuwm.io.restart import tree_fingerprint_mismatch_reason

    def components(mix: int) -> dict:
        return {"experiment_identity": {
            "domains": [{"grid_id": 1, "run": {"mix_isotropic": mix}}]}}

    class Model:
        _experiment_fingerprint_components = components(1)

    reason = tree_fingerprint_mismatch_reason(
        1, {"experiment_fingerprint_components": components(0)}, Model())
    assert "experiment_identity" in reason
    assert MIX_ISOTROPIC_RESTART_BREAK_NOTICE in reason

    class Unflipped:
        _experiment_fingerprint_components = components(0)

    reason = tree_fingerprint_mismatch_reason(
        1, {"experiment_fingerprint_components": components(0)},
        Unflipped())
    assert MIX_ISOTROPIC_RESTART_BREAK_NOTICE not in reason


# ---------------------------------------------------------------------------
# Every shipped config keeps its written meaning.
# ---------------------------------------------------------------------------

def _resolved_shipped_experiments():
    """(path, raw, experiment) for every shipped TOML that resolves.

    Unresolvable files are the shipped-configs screen's business
    (tests/test_shipped_configs_mixing_stability.py); this sweep is about
    the meaning of the ones the loader accepts.
    """

    out = []
    for path in sorted(CONFIGS.rglob("*.toml")):
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "experiment" not in raw or "domain" not in raw:
            continue
        try:
            if "case_data" in raw:
                from gpuwm.case_data import load_experiment_case_bytes
                exp = load_experiment_case_bytes(
                    path.read_bytes(), source=str(path),
                    base_dir=path.parent, require_inputs=False)[0]
            else:
                exp = load_experiment(path)
        except Exception:
            continue
        out.append((path, raw, exp))
    return out


def test_every_shipped_config_keeps_its_written_mixing_semantics(capsys):
    resolved = _resolved_shipped_experiments()
    names = {path.name for path, _, _ in resolved}
    # The sweep must actually cover the LES family, or it proves nothing.
    for required in ("les_nest_250m_km3.toml",
                     "les_tornado_100m_mayfield_20211210.toml"):
        assert required in names, f"{required} no longer resolves"
    for path, raw, exp in resolved:
        shared_value = raw.get("shared", {}).get("mix_isotropic")
        for table in raw["domain"]:
            written = table.get("mix_isotropic", shared_value)
            domain = exp.domain(int(table["grid_id"]))
            if isinstance(written, int) and not isinstance(written, bool):
                assert domain.run.mix_isotropic == written, path.name
                assert domain.grid_id not in exp.auto_mix_isotropic, path.name
            else:
                assert domain.grid_id in exp.auto_mix_isotropic, path.name
                # No shipped config changes behavior under the new
                # default: nothing in configs/ rides the switch.  A
                # config that starts riding it must write the value out
                # instead, so its file says what it runs.
                assert domain.run.mix_isotropic == 0, (
                    f"{path.name} d{domain.grid_id:02d} silently changed "
                    "meaning under the auto-switch; write its "
                    "mix_isotropic out explicitly")
    capsys.readouterr()
