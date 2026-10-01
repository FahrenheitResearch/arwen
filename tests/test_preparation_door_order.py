"""Each preparation door DECIDES before its first device allocation, driven.

A CUDA preparation of 1792x1024x55 on a 24 GB card decoded its inputs,
built its statics and stopped about two minutes in with a raw CuPy
out-of-memory inside ``DomainState.__init__``: the engine had a device
price for the phase and no door asked it.  Every door now prices its
preparation and decides (``admit_preparation`` /
``resolve_preprocess_backend(..., price=...)``) before it allocates on the
card.

tests/test_preparation_price.py holds that order with an AST check over
each door's source (``DOORS``).  These tests DRIVE the doors instead: each
runs on a stand-in card, shaped like that file's ``card`` fixture, that
records the decision's card reading and the first device allocation, and
the decision has to come first.  An explicit ``cuda`` request is used on
purpose, because only the decision reads the card on that road (``auto``
also reads it when it resolves), so the recorded reading IS the decision.
``gpuwm run``'s experiment door asks ``auto`` itself, and its one reading
is taken inside the decision call.

The AST check misses an order the source still spells right: met_em
priced before its loop and decided inside it, after ``initialize_real``,
passes the AST check and fails here.  Driven here: the mapped, met_em,
experiment and GFS doors, each through the fixtures its own tests drive
it with.  The ERA5, native HRRR and downscale-child doors have no fixture
that reaches past their decode and are held by the AST check alone.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import device_probe
from gpuwm.ingest import preprocess_backend as backend

GIB = 2 ** 30
ROOMY_CARD = {"free_bytes": 900 * GIB, "total_bytes": 960 * GIB,
              "utilization_gpu_percent": 0}


class _Allocated(Exception):
    """The door reached its first device allocation; the test stops it."""


@pytest.fixture
def card(monkeypatch):
    """A certified stand-in CUDA backend on a stand-in card that records.

    ``events`` gets ``"decision"`` when the door's decision reads the card
    and ``("allocation", name)`` when the door makes its first device
    allocation, which stops the door there.
    """
    events = []

    def _allocate(*_args, **_kwargs):
        events.append(("allocation", "array_module"))
        raise _Allocated("the stand-in card was allocated on")

    runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                              runtimeGetVersion=lambda: 13020)
    module = SimpleNamespace(__version__="14.2.0", zeros=_allocate,
                             empty=_allocate, asarray=_allocate,
                             cuda=SimpleNamespace(runtime=runtime))

    def cuda_backend():
        return SimpleNamespace(
            name="cuda", array_module=module,
            receipt=lambda: {"backend": "cuda", "array_module": "stand-in"})

    def cpu_backend(**kwargs):
        return SimpleNamespace(name="cpu", workers=kwargs.get("workers"),
                               receipt=lambda: {"backend": "cpu"})

    def read(**_kwargs):
        events.append("decision")
        return dict(ROOMY_CARD)

    monkeypatch.setattr(backend, "CudaPreprocessBackend", cuda_backend)
    monkeypatch.setattr(backend, "ParallelCpuPreprocessBackend", cpu_backend)
    monkeypatch.setattr(backend, "_gpu_runtime_installed", lambda: True)
    monkeypatch.setattr(backend, "_ANNOUNCED_AUTO_REASONS", set())
    monkeypatch.delenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", raising=False)
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", read)

    def allocates(owner, *names):
        """``owner.<name>`` is a device allocation: record it and stop."""
        for name in names:
            def allocation(*_args, _name=name, **_kwargs):
                events.append(("allocation", _name))
                raise _Allocated(_name)
            monkeypatch.setattr(owner, name, allocation)

    return SimpleNamespace(events=events, allocates=allocates,
                           module=module)


def _decided_first(events, door, *, decisions=1):
    """Every decision the door takes reads the card before it allocates.

    ``decisions`` is how many the door takes before its first allocation
    (GFS and ERA5 take a floor before the decode and the binding price
    after it), so a door that moved its binding decision below the first
    allocation fails here even though its floor still came first.
    """
    assert events, f"{door} neither decided nor allocated"
    assert "decision" in events, f"{door} allocated with no decision: {events}"
    allocations = [index for index, event in enumerate(events)
                   if event != "decision"]
    assert allocations, f"{door} was stopped before its first allocation"
    assert events[:allocations[0]] == ["decision"] * decisions, (door, events)


# ------------------------------------------------------------------ mapped

def test_the_mapped_door_decides_before_it_allocates(card, monkeypatch,
                                                     tmp_path):
    import gpuwm.mapped_direct as mapped_direct
    from test_mapped_direct import _install_prepare_fakes

    args, _calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cuda")
    # The route's own backend resolution and decision, on the stand-in.
    monkeypatch.setattr(mapped_direct, "resolve_preprocess_backend",
                        backend.resolve_preprocess_backend)
    card.allocates(mapped_direct, "interpolate_era5_to_lambert",
                   "initialize_real")
    with pytest.raises(_Allocated):
        mapped_direct.prepare_mapped_wrf(**args)
    # The floor as the backend resolves, then the decoded composition's.
    _decided_first(card.events, "prepare_mapped_wrf", decisions=2)


# ------------------------------------------------------------------ met_em

def test_the_met_em_door_decides_before_it_allocates(card, monkeypatch,
                                                     tmp_path):
    import gpuwm.ingest.metem as metem
    import gpuwm.ingest.real as real
    from gpuwm import metem_door, metem_forecast
    from test_metem_memory_advisory import _case

    run = _case(tmp_path, monkeypatch, device_low=False, host_low=False)
    exp = run.experiment
    monkeypatch.setattr(metem_forecast, "read_met_em_terrain",
                        lambda path: np.zeros((exp.root.run.ny,
                                               exp.root.run.nx)))
    monkeypatch.setattr(metem_forecast, "adapt_experiment_vertical",
                        lambda experiment, *_a, **_k: (experiment, None))
    from gpuwm.static.projection import grids_from_projection_config
    grids = {int(domain.grid_id): grid for domain, grid in zip(
        exp.domains, grids_from_projection_config(exp))}

    def read_met_em(path):
        grid_id = next(gid for gid, files in run.paths.items()
                       if path in files)
        lat, lon = grids[grid_id].latlon_mass()
        return SimpleNamespace(
            path=path, snapshot=None, terrain=None, source_orography=None,
            statics={"XLAT_M": lat, "XLONG_M": lon})

    monkeypatch.setattr(metem, "read_met_em", read_met_em)
    monkeypatch.setattr(metem, "met_em_series_identity", lambda case: {})
    monkeypatch.setattr(metem_door, "metgrid_initialization_controls",
                        lambda case, run, cfg=None: {})
    card.allocates(real, "initialize_real")
    with pytest.raises(_Allocated):
        metem_forecast.prepare_metem_run(run, tmp_path / "prepared",
                                         preprocess_backend="cuda")
    _decided_first(card.events, "prepare_metem_run")


# ------------------------------------------------------- gpuwm run (ERA5)

def test_the_experiment_door_decides_before_it_allocates(card, monkeypatch,
                                                         tmp_path):
    from datetime import timedelta

    from gpuwm import runtime
    from test_runtime import _fixture_pair

    exp, data = _fixture_pair(tmp_path)
    cfg = runtime.single_domain(exp).run
    ny, nx = cfg.ny, cfg.nx

    class Grid:
        e_we = cfg.nx + 1
        e_sn = cfg.ny + 1
        dx = cfg.dx
        dy = cfg.dy

    ones = np.ones((ny, nx), dtype=np.float32)
    monkeypatch.setattr(runtime, "case_static_fields",
                        lambda *_a, **_k: {"LANDMASK": ones,
                                           "LU_INDEX": ones})
    monkeypatch.setattr(runtime, "WaterTemperatureStatics", SimpleNamespace(
        for_route=lambda **_k: None))
    selection = SimpleNamespace(landuse_global_attrs=lambda: {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
        "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13})
    snapshot = SimpleNamespace(fields={
        "TT": np.zeros((4, 3, 5), np.float32),
        "RH": np.zeros((4, 3, 5), np.float32),
        "PSFC": np.zeros((3, 5), np.float32)})
    card.allocates(runtime, "interpolate_era5_to_lambert", "initialize_real")
    with pytest.raises(_Allocated):
        runtime.prepare_real_case(
            cfg, grid=Grid(), geog_root=data.geog_root,
            vertical=exp.vertical, sfcp_to_sfcp=True,
            snapshot_for=lambda _time: snapshot,
            forcing_times=(exp.start_time,
                           exp.start_time + timedelta(hours=6)),
            start_time=exp.start_time, geog_selection=selection)
    _decided_first(card.events, "prepare_real_case")


# --------------------------------------------------------------------- GFS

def test_the_gfs_door_decides_before_it_allocates(card, monkeypatch,
                                                  tmp_path):
    from gpuwm import gfs_direct
    from gpuwm.experiment import load_experiment
    from test_gfs_initial_perturbation import (
        _config, _cpu_preparation, _inputs)

    config = _config(tmp_path, domains=1)
    _cpu_preparation(monkeypatch, load_experiment(config))
    monkeypatch.setattr(gfs_direct, "resolve_preprocess_backend",
                        backend.resolve_preprocess_backend)
    card.allocates(gfs_direct, "interpolate_era5_to_lambert",
                   "initialize_real")
    arguments = dict(_inputs(tmp_path, config, "inputs"),
                     preprocess_backend="cuda")
    with pytest.raises(_Allocated):
        gfs_direct.prepare_gfs_wrf(**arguments)
    # The floor before the decode, then the decoded price.
    _decided_first(card.events, "prepare_gfs_wrf", decisions=2)
