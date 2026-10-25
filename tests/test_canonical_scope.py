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


def test_gsd_history_reflectivity_is_output_not_trajectory():
    """gsd_diagnostics' 1 km / 4 km AGL reflectivity is recomputed at every
    history time; a whole HRRR-recipe forecast failed its final digest on it
    (box D 2026-10-21: "'refl_10cm_1km' is inside an audited lazy prefix but
    is not a concrete registered member")."""
    from types import SimpleNamespace
    from gpuwm.state_digest import _canonical_extra_manifest
    state = SimpleNamespace(_scratch={"refl_10cm_1km": object(),
                                      "refl_10cm_4km": object()})
    assert _canonical_extra_manifest(state) == {}


def test_moving_nest_final_digest_includes_registered_placement_archive(monkeypatch):
    """Real six-hour member: (63,74)->(63,72)->(64,70), 160x160, ratio 4.

    All eight first-wave forecasts finished integration then failed on
    nest_accumulation/mp_graupelnc while computing their final digest.
    """
    from gpuwm.io import restart as restart_io
    from gpuwm import state_digest
    from gpuwm.core.nest_accumulation import META, PREFIX

    state = _State()
    state.elapsed_seconds = 21600.0
    state._scratch = {
        PREFIX + slot: np.zeros((176, 164), dtype=np.float32)
        for slot in ('mp_graupelnc', 'mp_rainnc', 'mp_snownc')
    }
    state._scratch[META] = np.array([4, -16, 0, -16], dtype=np.int64)
    # Keep the real restart scratch authority: only unrelated prognostics
    # are replaced, since this is a CPU-only inventory regression.
    monkeypatch.setattr(restart_io, 'state_manifest', lambda state: {})
    initial = canonical_state_digest(state, _Clock())
    assert set(initial['field_order']) == {f'scratch/{s}' for s in state._scratch}
    assert initial == real74_d02.canonical_state_digest(state, _Clock())
    for module in (state_digest, real74_d02):
        assert module._canonical_extra_manifest(state) == {}
        for slot in state._scratch:
            assert module.lazy_inventory_member_class(f'scratch/{slot}') == (
                'nest precipitation placement archive')
    state._scratch[PREFIX + 'mp_graupelnc'][0, 0] = 1
    changed_rain = canonical_state_digest(state, _Clock())
    assert changed_rain['sha256'] != initial['sha256']
    state._scratch[META][0] += 1
    assert canonical_state_digest(state, _Clock())['sha256'] != changed_rain['sha256']
    state._scratch[PREFIX + 'unregistered'] = np.zeros((1,), dtype=np.float32)
    for module in (state_digest, real74_d02):
        with pytest.raises(RuntimeError, match='not a concrete registered member'):
            module.canonical_state_digest(state, _Clock())
