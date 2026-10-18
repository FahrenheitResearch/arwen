"""The native door supplies one validated terrain-statistics carrier."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from gpuwm.ingest import wrfinput as wi
from gpuwm.static.orographic import required_static_fields
from test_wrfinput_physics_fields import (
    _cfg, _dimensions, _input, _surface_values,
)


@pytest.mark.parametrize("topo,gwd", [(0, 0), (1, 0), (2, 0), (0, 1), (0, 3), (2, 3)])
def test_native_door_passes_drag_statistics_once(tmp_path, monkeypatch, topo, gwd):
    cfg = _cfg(topo_wind=topo, gwd_opt=gwd)
    shape = cfg.ny, cfg.nx
    values = _surface_values(cfg)
    values["HGT"] = np.full(shape, np.float32(253.27182))
    for index, name in enumerate(required_static_fields(topo, gwd)):
        values[name] = np.arange(cfg.ny * cfg.nx, dtype=np.float32).reshape(shape) + np.float32(index + .125)
    restored = wi.read_wrfinput(
        _input(tmp_path / "input", cfg, values), cfg=cfg,
        expected_dimensions=_dimensions(cfg), require_complete=False)
    seen = {}
    driver = SimpleNamespace(
        fields={"albbck": np.zeros(shape, np.float32), "lai": np.zeros(shape, np.float32)},
        rainc=None, ruc_params=None)

    def initialize(state, actual_cfg, **kwargs):
        # Python rejects duplicate keyword expansions before entering here.
        assert actual_cfg is cfg
        seen.update(kwargs)
        return driver

    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setitem(sys.modules, "gpuwm.core.physics", SimpleNamespace(
        initialize_physics=initialize, physics_driver_required=lambda cfg: True))
    monkeypatch.setitem(sys.modules, "gpuwm.core.ruc_runtime", SimpleNamespace(
        ruc_cold_start=lambda *args,**kwargs: None))
    assert wi.initialize_wrfinput_physics(object(), restored, cfg) is driver
    names = required_static_fields(topo, gwd)
    if not names:
        assert "terrain_drag_static" not in seen
    else:
        static = seen["terrain_drag_static"]
        assert set(static) == set(names) | {"HGT_M"}
        for name in names:
            np.testing.assert_array_equal(np.asarray(static[name], np.float32).view("u4"), values[name].view("u4"))
        np.testing.assert_array_equal(np.asarray(static["HGT_M"], np.float32).view("u4"), values["HGT"].view("u4"))
