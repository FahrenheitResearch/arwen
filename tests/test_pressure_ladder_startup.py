"""Prepared source checks must not import the preparation execution stack."""
from datetime import datetime
from types import SimpleNamespace
import builtins

import pytest
from gpuwm.prepared_single_domain_forecast import _gfs_manifest_source_receipt
from gpuwm.ingest.pressure_ladder import CERTIFIED_PRESSURE_LEVELS_HPA


def test_prepared_source_receipt_without_preparation_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"gpuwm.gfs_direct", "gpuwm.era5_direct"}:
            raise AssertionError("prepared receipt imported preparation execution")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    exp = SimpleNamespace(start_time=datetime(2026, 2, 28),
                          vertical=SimpleNamespace(p_top=5000.0))
    identity = dict(model="GFS", product="pgrb2.0p25",
                    cycle="2026-02-28T00:00:00Z",
                    pressure_levels_hpa=[40., 50., 70.] +
                    CERTIFIED_PRESSURE_LEVELS_HPA["GFS"].tolist(),
                    top_pressure_pa=4000.)
    receipt = _gfs_manifest_source_receipt({"source": identity}, exp, {})
    assert receipt["source_top_pressure_pa"] == 4000.
    assert receipt["pressure_levels_hpa"] == identity["pressure_levels_hpa"]
    identity["top_pressure_pa"] = 5000.
    with pytest.raises(ValueError, match=(
            "GFS input manifest declares a source top of 5000 Pa but its "
            "pressure ladder tops out at 4000 Pa")):
        _gfs_manifest_source_receipt({"source": identity}, exp, {})
