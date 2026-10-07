"""Chemistry outputs must reach the shared writer and its disk budget."""
from dataclasses import replace
import builtins

import pytest

from gpuwm.core.chem_history import history_schema
from gpuwm.io.history_layout import history_frame_bytes, produced_history_shapes

from test_chem_restart_roundtrip import _cfg, _state


def test_live_tracer_history_matches_declared_disk_inventory(monkeypatch):
    from gpuwm.io.wrfout import _CHEM_VAR_META, _live_state_history_fields

    cfg = _cfg(chem_sets="tracer_test")
    state = _state(cfg, monkeypatch)
    live = _live_state_history_fields(state)
    shapes = produced_history_shapes(cfg, include_reflectivity=False)
    for name in ("PASSIVE_1", "PASSIVE_2"):
        assert live[name].shape == shapes[name] == (cfg.nz, cfg.ny, cfg.nx)
        assert _CHEM_VAR_META[name][1] == "ug kg-1"
    plain = replace(cfg, chem_sets="")
    assert not {"PASSIVE_1", "PASSIVE_2"} & produced_history_shapes(plain).keys()
    added_bytes = 2 * (4 * cfg.nz * cfg.ny * cfg.nx + 256)
    assert (history_frame_bytes(cfg, include_reflectivity=False)
            - history_frame_bytes(plain, include_reflectivity=False)
            == added_bytes)


@pytest.mark.parametrize("sets", ["smoke", "dust", "gocart_primary", "cams_aq"])
def test_history_schema_reads_process_rows_without_cupy(monkeypatch, sets):
    original = builtins.__import__

    def cpu_only(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("history disk pricing cannot require a device")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only)
    schema = history_schema(_cfg(chem_sets=sets))
    assert schema
    assert all(len(shape) in (2, 3) for shape, _description, _units in schema.values())
    if sets == "smoke":
        assert schema["smoke"][0] == (5, 4, 6)
        assert schema["SMOKE_SFC"][0] == schema["SMOKE_COLUMN"][0] == (4, 6)
