"""A model parent is sized against the card before its first leg.

`gpuwm cycle --parent-kind mpas-cuda` spawns the bridge worker, which
builds the mesh's host state and only then the port's device stack, and
nothing between the two asked whether the card could hold that mesh: a
mesh too big for the card died inside the worker after the host
preparation.  The spine now prices the leg with the port's own capacity
model, read by path from the port tree, and refuses by name, with both
numbers, before the first leg starts.

CPU-only: the port's admission module, the card probe and the mesh
header are stand-ins, and no worker is launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm.cycle import mpas_bridge
from gpuwm.cycle.mpas_bridge import BridgeRefusal

MIB = 1024 ** 2

#: A stand-in for the port's stdlib-only ``device_admission`` module with
#: its three-function surface: a core that scales with the card's
#: multiprocessors plus a per-cell slope.  It records what it was asked,
#: so the tests can hold that the spine priced THIS card's row.  Written
#: with postponed annotations on frozen dataclasses, as the port's own
#: module is, because a module loaded by path that is not registered
#: while it runs fails at its first such class.
_STAND_IN = '''
from __future__ import annotations

from dataclasses import dataclass

ASKED = []


@dataclass(frozen=True)
class CardProfile:
    name: str
    multiprocessors: int
    max_threads_per_sm: int = 1536


@dataclass(frozen=True)
class Model:
    core_bytes: float
    bytes_per_cell: float
    configuration: str
    measured: bool = True


def card_profile_from_attributes(name, attributes):
    return CardProfile(name, int(attributes["MultiProcessorCount"]),
                       int(attributes["MaxThreadsPerMultiProcessor"]))


def model_for_card(card=None, configuration="global", levels=55):
    ASKED.append((card.multiprocessors, configuration))
    return Model(core_bytes=1000 * 1024 ** 2 + card.multiprocessors * 1024 ** 2,
                 bytes_per_cell=100_000.0, configuration=configuration)


def required_free_bytes(cells, model=None, margin_bytes=None):
    return int(model.core_bytes + model.bytes_per_cell * int(cells))
'''


def _port(tmp_path, layout="src/hexcore/device_admission.py"):
    root = tmp_path / "port"
    module = root / layout
    module.parent.mkdir(parents=True)
    module.write_text(_STAND_IN, encoding="utf-8")
    return root


def admit_parent_device(**kwargs):
    return mpas_bridge.admit_parent_device(**kwargs)


def _card(free_mib, *, sms=70):
    return {"name": "stand-in card", "MultiProcessorCount": sms,
            "MaxThreadsPerMultiProcessor": 1536,
            "free_bytes": int(free_mib * MIB), "total_bytes": 16303 * MIB}


def _required_mib(cells, sms=70):
    return (1000 * MIB + sms * MIB + 100_000.0 * cells) / MIB


def test_a_mesh_the_card_cannot_hold_is_refused_with_both_numbers(tmp_path):
    cells = 163_842
    need = _required_mib(cells)
    with pytest.raises(BridgeRefusal) as refusal:
        admit_parent_device(port_root=_port(tmp_path), cells=cells,
                            mesh="x4.163842", card=_card(need - 1))
    observed = refusal.value.observed
    assert observed["mesh"] == "x4.163842" and observed["cells"] == cells
    assert observed["required_free_bytes"] == int(need * MIB)
    assert observed["free_bytes"] == int((need - 1) * MIB)
    text = str(refusal.value)
    assert f"{need:,.0f} MiB" in text and f"{need - 1:,.0f} MiB is free" in text
    assert "free the card" in observed["remedy"]


def test_a_mesh_that_fits_is_admitted_on_the_port_s_row_for_this_card(tmp_path):
    root = _port(tmp_path)
    cells = 40_962
    receipt = admit_parent_device(port_root=root, cells=cells,
                                  mesh="x1.40962", card=_card(16_000, sms=70))
    assert receipt["sized"] is True
    assert receipt["required_free_bytes"] == int(_required_mib(cells) * MIB)
    assert receipt["multiprocessors"] == 70
    assert receipt["configuration"] == "global"
    assert receipt["model_module"] == str(root / "src/hexcore/device_admission.py")
    # The price moves with the card's own row, not a fixed one.
    bigger = admit_parent_device(port_root=root, cells=cells, mesh="x1.40962",
                                 card=_card(32_000, sms=170))
    assert bigger["required_free_bytes"] - receipt["required_free_bytes"] == 100 * MIB


def test_the_older_port_layout_is_read_the_same_way(tmp_path):
    root = _port(tmp_path, "src/mpas_port/device_admission.py")
    receipt = admit_parent_device(port_root=root, cells=10, mesh="m",
                                  card=_card(16_000))
    assert receipt["sized"] is True
    assert receipt["model_module"].endswith("mpas_port/device_admission.py")


def test_a_port_with_no_admission_model_is_run_unsized_and_says_so(tmp_path):
    root = tmp_path / "port"
    root.mkdir()
    receipt = admit_parent_device(port_root=root, cells=10, mesh="m",
                                  card=_card(1))
    assert receipt["sized"] is False
    assert "device_admission.py" in receipt["reason"]


def test_the_mesh_is_counted_off_the_port_config_s_grid_header(
        tmp_path, monkeypatch):
    import gpuwm.netcdf_bridge as netcdf_bridge

    grid = tmp_path / "x1.40962.grid.nc"
    grid.write_bytes(b"stand-in")
    config = tmp_path / "port.json"
    config.write_text(json.dumps({"mesh": "x1.40962", "grid": str(grid)}),
                      encoding="utf-8")
    opened = []

    def open_dataset(path):
        opened.append(Path(path))
        return type("Dataset", (), {"dimensions": {
            "nCells": netcdf_bridge.Dimension("nCells", 40_962)}})()

    monkeypatch.setattr(netcdf_bridge, "open_dataset", open_dataset)
    assert mpas_bridge.mesh_cells(config) == (40_962, "x1.40962")
    assert opened == [grid]


def test_the_first_leg_is_refused_before_the_worker_is_launched(
        tmp_path, monkeypatch):
    from gpuwm.cycle.engine import build_model_parent_engine

    root = _port(tmp_path)
    launched = []
    monkeypatch.setattr(mpas_bridge, "launch",
                        lambda **kwargs: launched.append(kwargs))
    monkeypatch.setattr(mpas_bridge, "mesh_cells",
                        lambda port_config, **k: (163_842, "x4.163842"),
                        raising=False)
    monkeypatch.setattr(mpas_bridge, "mesh_levels",
                        lambda port_config, **k: 55)
    monkeypatch.setattr(mpas_bridge, "probe_card",
                        lambda python=None: _card(8_000), raising=False)
    advance = build_model_parent_engine(
        root=tmp_path / "cycle", clock=object(), parent_kind="mpas-cuda",
        mesh_id="x4.163842", port_root=root,
        port_config=tmp_path / "port.json", port_steps=4)
    with pytest.raises(BridgeRefusal) as refusal:
        advance(1, None)
    assert launched == []
    assert refusal.value.observed["mesh"] == "x4.163842"


#: The port's device_admission as it stood from 2026-08-25 to 08-27: the
#: retired affine model, with required_free_bytes and no per-card surface.
_AFFINE_VINTAGE = '''
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FootprintModel:
    fixed_bytes: int = 5018 * 1024 ** 2
    bytes_per_cell: int = 140_916


FOOTPRINT_MODEL = FootprintModel()


def required_free_bytes(cells, model=None, headroom_bytes=None):
    model = FOOTPRINT_MODEL if model is None else model
    return int(model.fixed_bytes + model.bytes_per_cell * int(cells))
'''


def test_a_port_admission_module_without_the_per_card_surface_runs_unsized(
        tmp_path):
    """That vintage used to raise AttributeError before the first leg, on
    a port 2.8.0 ran unsized."""

    root = tmp_path / "port"
    module = root / "src/mpas_port/device_admission.py"
    module.parent.mkdir(parents=True)
    module.write_text(_AFFINE_VINTAGE, encoding="utf-8")
    receipt = admit_parent_device(port_root=root, cells=163_842,
                                  mesh="x4.163842", card=_card(1))
    assert receipt["sized"] is False
    assert "card_profile_from_attributes" in receipt["reason"]
    assert "model_for_card" in receipt["reason"]
    assert "required_free_bytes" not in receipt["reason"]
    assert receipt["model_module"] == str(module)


def _recording_port(tmp_path):
    root = tmp_path / "port"
    module = root / "src/hexcore/device_admission.py"
    module.parent.mkdir(parents=True)
    module.write_text(_STAND_IN.replace(
        "ASKED.append((card.multiprocessors, configuration))",
        "ASKED.append((card.multiprocessors, configuration, levels))"),
        encoding="utf-8")
    return root


def test_the_leg_is_priced_on_the_case_s_level_count(tmp_path):
    import sys

    root = _recording_port(tmp_path)
    receipt = admit_parent_device(port_root=root, cells=10, mesh="m",
                                  card=_card(16_000), levels=75)
    asked = sys.modules["_mpas_port_device_admission"].ASKED
    assert asked == [(70, "global", 75)]
    assert receipt["levels"] == 75
    assert receipt["levels_basis"] == "the case's level count"

    unknown = admit_parent_device(port_root=root, cells=10, mesh="m",
                                  card=_card(16_000))
    # Each admission loads the port's module afresh, so read its record
    # again; 55 is the stand-in port's own default.
    assert sys.modules["_mpas_port_device_admission"].ASKED == [
        (70, "global", 55)]
    assert unknown["levels"] is None
    assert unknown["levels_basis"] == "the port's default level count"


def _headers(monkeypatch, dimensions_by_name):
    import gpuwm.netcdf_bridge as netcdf_bridge

    opened = []

    def open_dataset(path):
        opened.append(Path(path))
        return type("Dataset", (), {"dimensions": {
            name: netcdf_bridge.Dimension(name, size)
            for name, size in dimensions_by_name[Path(path).name].items()}})()

    monkeypatch.setattr(netcdf_bridge, "open_dataset", open_dataset)
    return opened


def test_a_relative_mesh_path_is_found_where_the_worker_finds_it(
        tmp_path, monkeypatch):
    """The worker runs with the bridge root as its working directory, so
    a relative path in the port config is relative to that root, not to
    wherever the spine was started."""

    config = tmp_path / "port.json"
    config.write_text(json.dumps({
        "mesh": "x1.40962", "grid": "cases/x1.40962.grid.nc",
        "init": "cases/x1.40962.init.nc"}), encoding="utf-8")
    opened = _headers(monkeypatch, {
        "x1.40962.grid.nc": {"nCells": 40_962},
        "x1.40962.init.nc": {"nCells": 40_962, "nVertLevels": 55}})
    monkeypatch.chdir(tmp_path)

    assert mpas_bridge.mesh_cells(config) == (40_962, "x1.40962")
    assert mpas_bridge.mesh_levels(config) == 55
    root = mpas_bridge.bridge_root()
    assert opened == [root / "cases/x1.40962.grid.nc",
                      root / "cases/x1.40962.init.nc"]


def test_an_init_header_with_no_level_dimension_prices_on_the_default(
        tmp_path, monkeypatch):
    init = tmp_path / "case.init.nc"
    config = tmp_path / "port.json"
    config.write_text(json.dumps({"mesh": "m", "init": str(init)}),
                      encoding="utf-8")
    _headers(monkeypatch, {"case.init.nc": {"nCells": 10}})
    assert mpas_bridge.mesh_levels(config) is None
