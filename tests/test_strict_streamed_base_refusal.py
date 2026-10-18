"""Strict WRF mode refuses a tile-streamed prepared slab whose base theta is not WRF's 300 K.

The breakage (2.9.1, strict-nonfinite): the resident prepared door now holds
theta as WRF's T = theta - 300, but the tile-streamed prepared road keeps the
preparation's base profile, so strict mode there would read theta up to 164 K
low and stop non-finite.  It is refused by name instead.
"""
import numpy as np
import pytest

from gpuwm.ingest import prepared_store


def _strict(monkeypatch, on):
    import gpuwm.wrf_exact as wrf_exact
    monkeypatch.setattr(wrf_exact, "ENABLED", on)


def test_strict_mode_refuses_a_prepared_base_that_is_not_300k(monkeypatch):
    _strict(monkeypatch, True)
    base = np.linspace(290.0, 464.0, 49, dtype=np.float32)
    with pytest.raises(prepared_store.PreparedStoreError,
                       match="does not run a tile-streamed prepared domain"):
        prepared_store.refuse_strict_streamed_base(base)


def test_strict_mode_passes_a_300k_base_and_default_mode_passes_any(monkeypatch):
    _strict(monkeypatch, True)
    prepared_store.refuse_strict_streamed_base(np.full(49, 300.0, dtype=np.float32))
    _strict(monkeypatch, False)
    prepared_store.refuse_strict_streamed_base(np.linspace(290.0, 464.0, 49, dtype=np.float32))
