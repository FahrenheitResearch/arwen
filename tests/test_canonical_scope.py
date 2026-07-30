"""F23: trajectory-scope canonical digest excludes child-duty scratch."""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.cases import real74_d02
from gpuwm.state_digest import canonical_state_digest


class _Clock:
    dtbc_fp32 = 12.0


class _State:
    elapsed_seconds = 900.0
    physics = None


def _patched_manifests(monkeypatch, parent_field_value):
    from gpuwm.io import restart as restart_io
    base = {
        "state/T": np.full((2, 3), 1.5, dtype=np.float32),
        "scratch/nest_parent_field": np.full(
            (2, 3), parent_field_value, dtype=np.float32),
        "scratch/nest_child_field": np.full(
            (2, 3), -parent_field_value, dtype=np.float32),
    }
    monkeypatch.setattr(restart_io, "state_manifest",
                        lambda state: dict(base))
    monkeypatch.setattr(restart_io, "_scratch_manifest", lambda state: {})
    monkeypatch.setattr(real74_d02, "_canonical_extra_manifest",
                        lambda state: {})
    monkeypatch.setattr(restart_io, "_host",
                        lambda value: np.asarray(value))


def test_trajectory_scope_ignores_child_duty_scratch(monkeypatch):
    _patched_manifests(monkeypatch, parent_field_value=0.0)
    zero = real74_d02.canonical_state_digest(_State(), _Clock())
    _patched_manifests(monkeypatch, parent_field_value=7.25)
    busy = real74_d02.canonical_state_digest(_State(), _Clock())
    assert zero["sha256"] == busy["sha256"]
    names = zero["field_order"]
    assert "scratch/nest_parent_field" not in names
    assert "scratch/nest_child_field" not in names


def test_full_scope_still_covers_child_duty_scratch(monkeypatch):
    _patched_manifests(monkeypatch, parent_field_value=0.0)
    zero = real74_d02.canonical_state_digest(_State(), _Clock(), scope="full")
    _patched_manifests(monkeypatch, parent_field_value=7.25)
    busy = real74_d02.canonical_state_digest(_State(), _Clock(), scope="full")
    assert zero["sha256"] != busy["sha256"]
    names = zero["field_order"]
    assert "scratch/nest_parent_field" in names
    assert "scratch/nest_child_field" in names


def test_live_nest_force_backings_are_registered_lazy_members():
    state = _State()
    state._scratch = {
        "nest_parent_field": np.zeros((3,), dtype=np.float32),
        "nest_child_field": np.zeros((5,), dtype=np.float32),
    }

    assert set(real74_d02._canonical_extra_manifest(state)) == {
        "scratch/nest_parent_field", "scratch/nest_child_field"}


def test_nssl_rolling_nest_fields_are_registered_lazy_members():
    state = _State()
    state._scratch = {
        f"nest_{kind}_{suffix}": np.zeros((2,), dtype=np.float32)
        for kind in (
            "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
            "qvolg", "qvolh",
        )
        for suffix in ("bxs", "btxe", "bys", "btye")
    }

    assert set(real74_d02._canonical_extra_manifest(state)) == {
        f"scratch/{name}" for name in state._scratch}


def test_scopes_can_never_collide(monkeypatch):
    _patched_manifests(monkeypatch, parent_field_value=0.0)
    trajectory = real74_d02.canonical_state_digest(_State(), _Clock())
    _patched_manifests(monkeypatch, parent_field_value=0.0)
    full = real74_d02.canonical_state_digest(_State(), _Clock(), scope="full")
    assert trajectory["sha256"] != full["sha256"]


def test_unknown_scope_rejected():
    with pytest.raises(ValueError, match="scope"):
        real74_d02.canonical_state_digest(_State(), _Clock(), scope="bogus")


@pytest.mark.parametrize("scope", ["trajectory", "full"])
def test_production_digest_matches_ratified_verification_digest(
        monkeypatch, scope):
    _patched_manifests(monkeypatch, parent_field_value=7.25)
    expected = real74_d02.canonical_state_digest(
        _State(), _Clock(), scope=scope)
    actual = canonical_state_digest(_State(), _Clock(), scope=scope)
    assert actual == expected
