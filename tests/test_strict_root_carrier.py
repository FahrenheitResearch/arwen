"""Strict mode's theta carrier on the in-process prepared root (2.9.2 strict default go).

The breakage this pins: under GPUWM_WRF_EXACT=1 every strict kernel reads
``thp`` as WRF's ``t = theta - 300`` (``thb == 300``).  2.9.1 re-expressed a
prepared state that way only where a prepared cache is restored; the route
that builds its forcing in the forecast's own process (``gpuwm go`` on the
keyless ERA5 default) ran the preparation's 290-464 K base in ``thb``, so the
default ``gpuwm go`` config stopped at its first strict step with
"microphysics RAINNC contains a non-finite value".  CPU only: the one door
every in-process root passes (gpuwm.runtime.prepare_root_experiment_case).
"""
from types import SimpleNamespace

import numpy as np

import gpuwm.wrf_exact as wrf_exact
from gpuwm import runtime


def _base_and_perturbation():
    thb = np.linspace(290.0, 464.0, 12, dtype=np.float32).reshape(3, 2, 2)
    thp = np.linspace(-12.0, 26.0, 12, dtype=np.float32).reshape(3, 2, 2)
    return thb, thp


def _in_process_root(monkeypatch, prepared):
    monkeypatch.setattr(runtime, "_runtime_preparation_source", lambda: None)
    monkeypatch.setattr(runtime, "_prepare_root_experiment_case",
                        lambda *args, **kwargs: prepared)
    return runtime.prepare_root_experiment_case(SimpleNamespace(), SimpleNamespace())


def test_strict_in_process_root_holds_wrf_theta(monkeypatch):
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    thb, thp = _base_and_perturbation()
    state = SimpleNamespace(thb=thb.copy(), thp=thp.copy())
    prepared = SimpleNamespace(initial_result=SimpleNamespace(state=state))
    assert _in_process_root(monkeypatch, prepared) is prepared
    assert (state.thb == np.float32(300.0)).all()
    history_t = ((thb + thp) - np.float32(300.0)).astype(np.float32)
    assert state.thp.tobytes() == history_t.tobytes()
    assert state._setup_thb.tobytes() == thb.tobytes()


def test_default_in_process_root_is_untouched(monkeypatch):
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    thb, thp = _base_and_perturbation()
    state = SimpleNamespace(thb=thb.copy(), thp=thp.copy())
    prepared = SimpleNamespace(initial_result=SimpleNamespace(state=state))
    assert _in_process_root(monkeypatch, prepared) is prepared
    assert state.thb.tobytes() == thb.tobytes()
    assert state.thp.tobytes() == thp.tobytes()
    assert not hasattr(state, "_setup_thb")


def test_host_store_root_carries_no_state_to_re_express(monkeypatch):
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    prepared = SimpleNamespace(initial_result=SimpleNamespace(initial_perturbation=None))
    assert _in_process_root(monkeypatch, prepared) is prepared
