"""Prepared built-area fractions reach the real canopy driver unchanged."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from conftest import requires_gpu


class _UnreadStatic(dict):
    def get(self, name, default=None):
        raise AssertionError(f"the static carrier was read for {name}")


def _driver(monkeypatch, option, *, static=None, explicit=None, initialize=None):
    from gpuwm.core import physics
    from test_urban_default_off_identity import _small_driver

    native = initialize or physics.initialize_physics

    def with_prepared_fraction(*args, **kwargs):
        kwargs["terrain_drag_static"] = static
        if explicit is not None:
            kwargs["frc_urb2d"] = explicit
        return native(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(physics, "initialize_physics", with_prepared_fraction)
        return _small_driver(sf_urban_physics=option)


def _source_fraction():
    from test_physics_driver import _base_config

    cfg = _base_config()
    # Values differ from the table defaults and retain meaningful mantissa bits.
    values = np.full((cfg.ny, cfg.nx), np.float32(0.27182818))
    values[:, 1] = np.float32(0.61803395)
    return values


def _assert_fraction(driver, expected):
    import cupy as cp

    city = cp.asnumpy(driver.urban.urban_mask).astype(bool)
    actual = cp.asnumpy(driver.urban.frc_urb2d)
    assert actual[city].tobytes() == np.asarray(expected, np.float32)[city].tobytes()
    assert not np.any(actual[~city])


def _field_words(driver):
    import cupy as cp

    return {name: cp.asnumpy(array).tobytes() for name, array in driver.fields.items()
            if hasattr(array, "__cuda_array_interface__")}


@requires_gpu
def test_prepared_fraction_is_the_same_driver_path_as_explicit_input(monkeypatch):
    fraction = _source_fraction()
    before = fraction.tobytes()
    state, cfg, prepared = _driver(
        monkeypatch, 1, static={"FRC_URB2D": fraction})
    _assert_fraction(prepared, fraction)
    other_state, other_cfg, explicit = _driver(monkeypatch, 1, explicit=fraction)
    _assert_fraction(explicit, fraction)
    prepared.compute(state, cfg)
    explicit.compute(other_state, other_cfg)
    assert _field_words(prepared) == _field_words(explicit)
    assert fraction.tobytes() == before


@requires_gpu
def test_explicit_fraction_wins_without_reading_prepared_fraction(monkeypatch):
    fraction = _source_fraction()
    _, _, driver = _driver(monkeypatch, 1, static=_UnreadStatic(), explicit=fraction)
    _assert_fraction(driver, fraction)


@requires_gpu
def test_disabled_canopy_never_reads_static_fraction_and_keeps_all_words(monkeypatch):
    state, cfg, baseline = _driver(monkeypatch, 0)
    other_state, other_cfg, prepared = _driver(monkeypatch, 0, static=_UnreadStatic())
    assert prepared.urban is None
    assert not any(name.endswith(("_urb2d", "_urb3d")) for name in prepared.fields)
    for _ in range(2):
        baseline.compute(state, cfg)
        prepared.compute(other_state, other_cfg)
    assert _field_words(baseline) == _field_words(prepared)


@requires_gpu
def test_fraction_gate_detects_removal_of_the_static_read(monkeypatch):
    from gpuwm.core import physics

    source = inspect.getsource(physics.initialize_physics)
    statement = 'frc_urb2d = terrain_drag_static.get("FRC_URB2D")'
    assert source.count(statement) == 1
    namespace = dict(physics.__dict__)
    exec(compile(source.replace(statement, "frc_urb2d = None"),
                 "<removed-prepared-fraction-control>", "exec"), namespace)
    fraction = _source_fraction()
    _, _, driver = _driver(monkeypatch, 1, static={"FRC_URB2D": fraction},
                           initialize=namespace["initialize_physics"])
    with pytest.raises(AssertionError):
        _assert_fraction(driver, fraction)
    print("MUTATION DETECTED: removing the prepared fraction read changes canopy fraction words")
