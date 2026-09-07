"""Relocation reads canonical store bytes even when its template is stale."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.physics_continuation import capture_carriers, capture_continuation


def test_streamed_continuation_uses_store_not_last_slab():
    stale = np.zeros((3, 2, 5), dtype=np.float32)
    state = SimpleNamespace(existing_scratch=lambda name: stale if name == "cu_nca" else None)
    driver = SimpleNamespace(gf_rthblten=stale, rthratenlw=stale,
                             cumulus_callable=SimpleNamespace(w0avg=stale))
    keys = ("scratch/cu_nca", "held/gf_rthblten", "driver/rthratenlw", "cumulus/w0avg")
    store = {k: np.full((3, 12, 5), i+1, dtype=np.float32) for i, k in enumerate(keys)}
    actual = capture_continuation(state, driver, store=store)
    for name, key in zip(("cu_nca", "held/gf_rthblten", "driver/rthratenlw", "cumulus/w0avg"), keys):
        np.testing.assert_array_equal(actual[name], store[key])
        assert not np.shares_memory(actual[name], store[key])
    del store["scratch/cu_nca"]
    with pytest.raises(ValueError, match="canonical carrier scratch/cu_nca"):
        capture_continuation(state, driver, store=store)


def test_streamed_continuation_consumes_the_real_carrier_inventory(monkeypatch):
    from gpuwm.io.restart import DRIVER_HELD_FORCING_ATTRS
    from test_restart import _cfg, _grell_freitas_shim_state
    from tilestream.physics_inventory import streaming_inventory

    cfg = _cfg(moist=True, mp_physics=10, cu_physics=3, cudt_minutes=0.)
    state, driver = _grell_freitas_shim_state(cfg, monkeypatch)
    for index, name in enumerate(sorted(DRIVER_HELD_FORCING_ATTRS), 1):
        setattr(driver, name, np.full((cfg.nz, cfg.ny, cfg.nx), index, dtype=np.float32))
    # Obtain keys from the actual producer used to populate streamed stores.
    # An independently invented mapping previously agreed with the bad reader.
    store = {key: np.array(value, copy=True) for key, value in streaming_inventory(state).items()}
    expected = {key: np.array(value, copy=True) for key, value in capture_continuation(state, driver).items()}
    for name in DRIVER_HELD_FORCING_ATTRS:
        assert f"held/{name}" in store
        assert f"driver/{name}" not in store
        getattr(driver, name)[:] = -99.  # The prepared/template state is stale.
    captured = capture_continuation(state, driver, store=store)
    assert captured.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(captured[name], value, err_msg=name)
    missing = dict(store)
    del missing["held/gf_rthblten"]
    with pytest.raises(ValueError, match="canonical carrier held/gf_rthblten"):
        capture_continuation(state, driver, store=missing)


def test_streamed_radiation_uses_canonical_values_and_ledger():
    driver = SimpleNamespace(fields={"glw": np.zeros((2, 5), np.float32)},
                             carriers=SimpleNamespace(state=lambda: {"glw": {"source": "unwritten"}}))
    store = {"fields/glw": np.full((12, 5), 247., np.float32)}
    ledger = {"glw": {"source": "radiation_scheme", "last_update_model_time": 30.}}
    actual = capture_carriers(driver, store=store, scalars={"carriers": ledger})
    np.testing.assert_array_equal(actual["fields"]["glw"], store["fields/glw"])
    assert not np.shares_memory(actual["fields"]["glw"], store["fields/glw"])
    assert actual["contract"] == ledger
    actual["contract"]["glw"]["last_update_model_time"] = 99.
    assert ledger["glw"]["last_update_model_time"] == 30.
    with pytest.raises(ValueError, match="scalar carriers"):
        capture_carriers(driver, store=store)
    with pytest.raises(ValueError, match="canonical carrier fields/glw"):
        capture_carriers(driver, store={}, scalars={"carriers": ledger})
