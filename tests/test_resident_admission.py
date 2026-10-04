"""Every road onto the card is admitted, or refused by name, before it allocates.

The A65 thorough review found four allocations no door priced, each of which
turned a case too big for the card into a CUDA out-of-memory part way through
a constructor instead of a refusal naming the terms:

* a RESIDENT domain (``[tiles]`` off, or no ``[tiles]`` block at all) reached
  its constructor with no admission: the prepared-cache restore, the native
  wrfinput restore and ``init_at_rest``, and the prepared reload doors above
  them;
* a PINNED tiling consulted no card, so buffers that could not fit were built
  anyway;
* an explicit loader ``rows_per_slab`` reached the CUDA slab constructor with
  only the HOST store budget checked;
* the PHYSICS attach followed the state onto the card at the ``gpuwm run`` and
  downscale doors with nothing pricing the two together.

Each door is driven here on a stand-in card whose free memory is too small,
and each test fails on the head because the constructor (or the fetch that
precedes it) is reached, and passes at the fix because the door refuses by
name first.  The stand-in is the card itself -- ``Machine.detect`` for a door,
the CUDA runtime's ``memGetInfo`` for a loader -- never the admission code.
"""
from __future__ import annotations

from dataclasses import replace
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import preflight as pf
from gpuwm.core import streaming as st
from tilestream import autoplan as ap

GIB = 1024 ** 3
POINT = (16.38581807563628, -123.76623740203063)
CYCLE = "2026091112"
REFUSAL = "refused before anything was allocated"


class _ConstructorReached(Exception):
    """The allocation every admission here has to come before."""


def _reached(*_args, **_kwargs):
    raise _ConstructorReached("the constructor was reached with no admission")


def _stand_in_card(monkeypatch, free_bytes):
    """A card with ``free_bytes`` free, as ``Machine.detect`` reads one."""
    card = ap.Machine(vram_bytes=int(free_bytes), host_bytes=64 * GIB,
                      name="stand-in card", host_source="explicit")
    monkeypatch.setattr(ap.Machine, "detect",
                        classmethod(lambda cls, **kwargs: card))
    return card


def _stand_in_runtime(monkeypatch, free_bytes):
    """A CUDA runtime whose ``memGetInfo`` reports ``free_bytes`` free.

    What a loader reads when it asks the card from inside the process that
    is about to allocate.  Installed as the ``cupy`` a function-local import
    resolves; modules already bound to a real one keep it.
    """
    pool = SimpleNamespace(free_bytes=lambda: 0, malloc=lambda size: None)
    fake = types.ModuleType("cupy")
    fake.__dict__.update({name: value for name, value in vars(np).items()
                          if not name.startswith("__")})
    fake.ndarray = np.ndarray
    fake.asarray = np.asarray
    fake.cuda = SimpleNamespace(
        runtime=SimpleNamespace(
            memGetInfo=lambda: (int(free_bytes), 16 * GIB)),
        get_allocator=lambda: pool.malloc)
    fake.get_default_memory_pool = lambda: pool
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return fake


def _reference(**overrides):
    """The A65 reference domain: 1792x1024x55 at 1 km."""
    values = dict(nx=1792, ny=1024, nz=55, dx=1000.0, dy=1000.0,
                  ztop=20000.0, dt=3.0, run_seconds=3600.0, terrain_opt=1,
                  moist=True, mp_physics=8, specified=True)
    values.update(overrides)
    return RunConfig(**values)


def _cyclone(tiles="off", *, single=True):
    from gpuwm import cyclone_setup as tc
    from gpuwm.experiment import RelocationConfig

    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=6,
                                       tiles=tiles)
    if single:
        exp = replace(exp, domains=exp.domains[:1],
                      relocation=RelocationConfig())
    return exp


def _envelope(exp, *, forcing_intervals=None, boundary_species=()):
    if forcing_intervals is None:
        return pf.admission_estimate(
            exp, source=tuple(boundary_species)).peak_envelope_bytes
    return pf.estimate_experiment(
        exp, column_chunk=exp.column_chunk,
        forcing_intervals=forcing_intervals,
        boundary_species=tuple(boundary_species)).peak_envelope_bytes


# ---------------------------------------------------------------------------
# resident loaders: the constructor floor
# ---------------------------------------------------------------------------


def test_the_reference_resident_floors_exceed_a_24_gib_card():
    """The arithmetic the loader floor refuses on, at the A65 reference.

    mp=18 (NSSL two-moment) holds more than 24 GiB of state alone; mp=8
    retaining 74 eager intervals (75 forcing times) holds 25,782,246,220
    bytes of state and boundary tables before any physics.
    """
    from gpuwm.core.device_inventory import lbc_interval_values
    from gpuwm.core.resident_admission import state_bytes

    assert state_bytes(_reference(mp_physics=18)) > 24 * GIB
    mp8 = _reference(spec_bdy_width=5)
    floor = state_bytes(mp8) + 4 * lbc_interval_values(mp8) * 74
    assert floor > 24 * GIB
    assert state_bytes(mp8) < 24 * GIB, (
        "the mp=8 state alone fits; the retained tables are what tip it")


def _written_cache(tmp_path):
    """A 2x2x2 prepared cache, and the RunConfig it restores against."""
    from test_prepared_cache import _fixture
    from gpuwm.ingest.prepared_cache import write_prepared_cache

    initial, met, boundaries = _fixture()
    identity = {"source": "abc", "config": {"nx": 2}}
    path = tmp_path / "prepared"
    write_prepared_cache(path, identity=identity, initial_result=initial,
                         met=met, boundaries=boundaries,
                         metadata={"forcing_hours": [0, 1]})
    cfg = RunConfig(nx=2, ny=2, nz=2, dx=1000.0, dy=1000.0, ztop=20000.0,
                    dt=3.0, run_seconds=3600.0, specified=True,
                    terrain_opt=1)
    static = {name: np.ones((2, 2), dtype=np.float32)
              for name in ("MAPFAC_M", "F", "E", "SINALPHA", "COSALPHA")}
    static["MAPFAC_U"] = np.ones((2, 3), dtype=np.float32)
    static["MAPFAC_V"] = np.ones((3, 2), dtype=np.float32)
    return path, identity, cfg, static


def test_a_prepared_cache_too_big_for_the_card_is_refused_before_its_state(
        tmp_path, monkeypatch):
    """The prepared-cache restore admits its state and tables first.

    Red on the head: the restore goes straight into ``DomainState``.
    """
    import gpuwm.core.state as state_module
    from gpuwm.ingest.prepared_cache import restore_prepared_cache

    path, identity, cfg, static = _written_cache(tmp_path)
    _stand_in_runtime(monkeypatch, free_bytes=1024)
    monkeypatch.setattr(state_module, "DomainState", _reached)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        restore_prepared_cache(path, expected_identity=identity, cfg=cfg,
                               static=static)
    assert "restoring this prepared cache" in str(refused.value)
    assert "model state" in str(refused.value)
    assert "lateral boundary tables" in str(refused.value)
    assert "CUDA out-of-memory" in str(refused.value)
    # The tables are the cache's own retained inventory: one interval of
    # four u sides, value and tendency, 1x2x2 each.
    assert refused.value.terms["lateral boundary tables"] == 4 * 4 * 2 * 4


def test_a_prepared_cache_that_fits_reaches_its_state_unchanged(
        tmp_path, monkeypatch):
    """The admission refuses only what does not fit."""
    import gpuwm.core.state as state_module
    from gpuwm.ingest.prepared_cache import restore_prepared_cache

    path, identity, cfg, static = _written_cache(tmp_path)
    _stand_in_runtime(monkeypatch, free_bytes=8 * GIB)
    monkeypatch.setattr(state_module, "DomainState", _reached)
    with pytest.raises(_ConstructorReached):
        restore_prepared_cache(path, expected_identity=identity, cfg=cfg,
                               static=static)


def test_a_host_setup_restore_asks_no_card(tmp_path, monkeypatch):
    """A NumPy setup state never touches the device, so it is not admitted."""
    import gpuwm.core.state as state_module
    from gpuwm.ingest.prepared_cache import restore_prepared_cache

    path, identity, cfg, static = _written_cache(tmp_path)
    _stand_in_runtime(monkeypatch, free_bytes=0)
    monkeypatch.setattr(state_module, "DomainState", _reached)
    with pytest.raises(_ConstructorReached):
        restore_prepared_cache(path, expected_identity=identity, cfg=cfg,
                               static=static, array_module=np)


def test_a_wrfinput_too_big_for_the_card_is_refused_before_its_state(
        monkeypatch):
    """The native wrfinput restore admits its state first."""
    import gpuwm.core.state as state_module
    from gpuwm.ingest.wrfinput import restore_domain_state

    cfg = RunConfig(nx=4, ny=3, nz=2, dx=1000.0, dy=1000.0, ztop=20000.0,
                    dt=3.0, run_seconds=3600.0)
    nz, ny, nx = 2, 3, 4
    shapes = {"U": (nz, ny, nx + 1), "V": (nz, ny + 1, nx),
              "W": (nz + 1, ny, nx), "T": (nz, ny, nx),
              "PH": (nz + 1, ny, nx), "MU": (ny, nx),
              "PHB": (nz + 1, ny, nx), "MUB": (ny, nx),
              "T_INIT": (nz, ny, nx), "P": (nz, ny, nx),
              "PB": (nz, ny, nx), "AL": (nz, ny, nx), "ALB": (nz, ny, nx)}
    restored = SimpleNamespace(raw={name: np.zeros(shape, dtype=np.float32)
                                    for name, shape in shapes.items()})
    _stand_in_runtime(monkeypatch, free_bytes=64)
    monkeypatch.setattr(state_module, "DomainState", _reached)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        restore_domain_state(restored, cfg)
    assert "restoring this wrfinput" in str(refused.value)


def test_an_idealized_state_too_big_for_the_card_is_refused_first(
        monkeypatch):
    """``init_at_rest`` admits its state first."""
    import gpuwm.core.state as state_module
    from gpuwm.core.grid import make_vertical_coord

    cfg = RunConfig(nx=8, ny=8, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=3.0, run_seconds=3600.0)
    _stand_in_runtime(monkeypatch, free_bytes=64)
    monkeypatch.setattr(state_module, "DomainState", _reached)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        state_module.init_at_rest(cfg, make_vertical_coord(4, hybrid_opt=0),
                                  SimpleNamespace(terrain_z=None))
    assert "idealized state" in str(refused.value)


# ---------------------------------------------------------------------------
# resident doors: the forecast, state and physics together
# ---------------------------------------------------------------------------


#: What the reload doors import inside their bodies before the restore,
#: bound here first so a stand-in ``cupy`` only ever answers the door's own
#: ``import cupy`` line (it touches no device before the restore).
_DOOR_MODULES = (
    "gpuwm.core.clock", "gpuwm.core.dycore", "gpuwm.core.gpu_mem_watch",
    "gpuwm.core.health", "gpuwm.core.model", "gpuwm.core.nest",
    "gpuwm.core.refl", "gpuwm.core.uh_diag", "gpuwm.core.state",
    "gpuwm.ingest.hrrr_physics", "gpuwm.ingest.lateral_bc",
    "gpuwm.ingest.prepared_cache", "gpuwm.io.restart", "gpuwm.io.wrfout",
    "gpuwm.runtime", "gpuwm.state_digest", "gpuwm.supervisor",
    "gpuwm.prepared_single_domain_forecast",
    "gpuwm.prepared_domain_tree_forecast")


def _door_runtime(monkeypatch):
    import importlib

    from test_downscale_pricing import _stub_cupy

    for name in _DOOR_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as missing:        # an install without the door
            pytest.skip(f"{name} is not importable here: {missing}")
    _stub_cupy(monkeypatch)


def _stub_prepared_single_door(monkeypatch, tmp_path, exp, *, intervals,
                               source="gfs", rows=None):
    """Drive the prepared single-domain door as far as its restore.

    ``rows`` are the cache's retained interval rows; the default names no
    fields, so the door prices ``source``'s own row.
    """
    from gpuwm import case_data
    from gpuwm import prepared_single_domain_forecast as psdf
    import gpuwm.ingest.prepared_cache as prepared_cache

    _door_runtime(monkeypatch)
    monkeypatch.setattr(psdf, "_verify_thompson_runtime_environment",
                        lambda receipt: None)
    monkeypatch.setattr(psdf, "_runtime_source_identity", lambda: {})
    monkeypatch.setattr(case_data, "trace_gas_overrides_from_config",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(prepared_cache, "restore_prepared_cache", _reached)
    reader = SimpleNamespace(header={"metadata": {"lbc": {
        "intervals": [{}] * intervals if rows is None else rows}}},
        metadata={})
    inputs = SimpleNamespace(
        experiment=exp, experiment_config=tmp_path / "experiment.toml",
        file_sha256={"experiment_config": "0" * 64}, physics_receipt=None,
        source=source, cache_reader=reader, stream_head=None,
        prepared_cache_path=tmp_path / "prepared", cache_identity={},
        static={}, boundary_interval_seconds=3600)
    return psdf, inputs


def test_the_prepared_single_domain_door_refuses_a_resident_reload_first(
        tmp_path, monkeypatch):
    """No ``[tiles]`` block: the reload is admitted before the restore.

    Priced at the retained interval count the cache carries, state and
    physics together.  Red on the head: the restore is reached.
    """
    exp = _cyclone("off")
    need = _envelope(exp, forcing_intervals=24)
    _stand_in_card(monkeypatch, need - 1)
    psdf, inputs = _stub_prepared_single_door(monkeypatch, tmp_path, exp,
                                              intervals=24)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        psdf.run_prepared_forecast(inputs, output_directory=tmp_path,
                                   kernel_cache_census=object())
    text = str(refused.value)
    assert "this prepared forecast, held resident on the card" in text
    assert "physics" in text and "[tiles] mode = 'auto'" in text
    assert refused.value.need_bytes == need


def test_the_prepared_single_domain_door_restores_a_reload_that_fits(
        tmp_path, monkeypatch):
    exp = _cyclone("off")
    _stand_in_card(monkeypatch, _envelope(exp, forcing_intervals=24))
    psdf, inputs = _stub_prepared_single_door(monkeypatch, tmp_path, exp,
                                              intervals=24)
    with pytest.raises(_ConstructorReached):
        psdf.run_prepared_forecast(inputs, output_directory=tmp_path,
                                   kernel_cache_census=object())


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_the_prepared_tree_door_refuses_a_resident_tree_first(
        tmp_path, monkeypatch, source):
    """A tree with nothing streamed is admitted before its first allocation.

    Red on the head: the shared workspaces are built with no admission.
    With a root cache that names no boundary fields the door prices its
    source's own row: ``hrrr`` publishes the analysed hydrometeors, so
    its root's tables carry them and the need is above water vapour's.
    """
    from gpuwm.boundary_fields import source_boundary_species

    exp = _cyclone("off", single=False)
    species = source_boundary_species(source)
    need = _envelope(exp, forcing_intervals=6, boundary_species=species)
    assert (need > _envelope(exp, forcing_intervals=6)) is bool(species)
    _stand_in_card(monkeypatch, need - 1)
    tree, inputs = _tree_door(monkeypatch, tmp_path, exp, source=source)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        tree.run_prepared_tree(inputs, output_directory=tmp_path,
                               io_mode="none")
    assert "this prepared domain tree" in str(refused.value)
    assert refused.value.need_bytes == need


def test_the_run_door_admits_state_and_physics_before_the_fetch(
        tmp_path, monkeypatch):
    """``gpuwm run`` with no ``[tiles]``: the forecast is priced first.

    The preparation price admits the transforms and the state; the physics
    driver follows them onto the card.  Red on the head: the fetch begins
    (and the physics attach later fails in CUDA) with nothing weighing the
    two together.
    """
    from gpuwm import runtime
    import gpuwm.ingest.preflight as ingest_preflight

    exp = _cyclone("off")
    need = _envelope(exp)
    card = _stand_in_card(monkeypatch, need - 1)
    monkeypatch.setattr(ingest_preflight, "build_input_catalog", _reached)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "out")
    assert refused.value.terms["physics"] > 0
    assert refused.value.free_bytes == int(card.vram_bytes)
    assert refused.value.need_bytes == need

    _stand_in_card(monkeypatch, need)
    with pytest.raises(_ConstructorReached):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "fits")


def test_the_downscale_door_admits_the_resident_child_before_it_interpolates(
        tmp_path, monkeypatch):
    """The offline child with no ``[tiles]``: state and physics priced first.

    Red on the head: the child interpolates its parent (and then builds its
    state and physics) with nothing admitted.
    """
    import gpuwm.offline_child_run as child_run
    from test_downscale_pricing import (
        _Sentinel, _drive_the_runner_to_its_decision, _stub_cupy)

    _stub_cupy(monkeypatch)
    _stand_in_card(monkeypatch, 1024 ** 2)
    monkeypatch.setattr(child_run, "interpolate_parent_initial_state",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            _Sentinel()))
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        _drive_the_runner_to_its_decision(tmp_path / "off", tiles_mode=None)
    assert "this downscaled child, held resident on the card" in str(
        refused.value)
    assert refused.value.terms["physics"] > 0


def test_an_admitted_decision_is_not_admitted_twice():
    """``auto``'s resident answer and a streamed road carry their admission."""
    exp = _cyclone("auto")
    machine = ap.Machine(64 * GIB, 256 * GIB)
    decision = st.cold_single_domain_decision(exp, machine=machine)
    assert not decision.stream and decision.budget_bytes is not None
    assert st.admit_resident_road(exp, decision, machine=_small()) is None
    assert st.admit_resident_road(
        exp, st.StreamingDecision(True, "streamed", 64, 64, 2, 16),
        machine=_small()) is None
    with pytest.raises(MemoryError, match=REFUSAL):
        st.admit_resident_road(exp, st.decide(exp.root.run, st.OFF),
                               machine=_small())
    # An unread card never refuses.
    assert st.admit_resident_road(exp, None, machine=None) is None


def _small():
    return ap.Machine(1024 ** 2, 64 * GIB)


# ---------------------------------------------------------------------------
# pinned tilings: priced on the card they will run on
# ---------------------------------------------------------------------------


def _pinned_reference(**overrides):
    """The A65 reference with ``tile 896x992, nbuffers 2`` and a 24 GiB budget."""
    cfg = _reference(time_step_sound=4)
    values = dict(mode="on", tile_nx=896, tile_ny=992, nbuffers=2, halo=16,
                  vram_budget_bytes=24 * GIB)
    values.update(overrides)
    pinned = st.StreamingOptions(**values)
    return cfg, pinned, st.decide(cfg, pinned)


def test_a_pinned_tiling_that_cannot_fit_is_refused_naming_one_that_does():
    """Two 928x1024 compute windows against a declared 24 GiB budget.

    Red on the head: nothing prices a pinned tiling, so the tiling is
    accepted and its buffers are built.
    """
    cfg, pinned, decision = _pinned_reference()
    assert decision.stream and decision.budget_bytes is None
    machine = ap.Machine(24 * GIB, 256 * GIB)
    with pytest.raises(ap.CannotPlan) as refused:
        st.admit_pinned_road(cfg, pinned, decision, machine=machine)
    text = str(refused.value)
    assert "[tiles] pins a 896x992 tiling" in text
    assert "928x1024x55 compute windows" in text
    assert "declared [tiles] vram_budget_bytes of 24.00 GiB" in text
    assert "CUDA out-of-memory" in text
    assert "The largest tile that fits with 2 buffer(s) is" in text
    assert refused.value.resource == "vram"
    largest = refused.value.detail["largest_tile"]
    assert largest is not None
    # The tile it names is one the same card admits.
    fits = replace(pinned, tile_nx=largest[0], tile_ny=largest[1])
    admitted = st.admit_pinned_road(cfg, fits, st.decide(cfg, fits),
                                    machine=machine)
    assert admitted["budget_bytes"] == 24 * GIB
    assert admitted["need_bytes"] <= admitted["budget_bytes"]


def test_a_pinned_tiling_with_no_card_is_still_the_configurations_decision():
    """The no-card road the bit-exactness proofs run on is unchanged."""
    cfg, pinned, decision = _pinned_reference(vram_budget_bytes=None)
    assert decision.stream and (decision.tile_nx, decision.tile_ny) == (896, 992)
    assert decision.budget_bytes is None
    assert st.admit_pinned_road(cfg, pinned, decision, machine=None) is None


def test_a_pinned_tiling_prices_only_the_buffers_the_driver_builds():
    """Never more buffers than tiles: an unused buffer is not charged."""
    cfg = _reference(nx=256, ny=256, nz=20, time_step_sound=4)
    one_tile = st.StreamingOptions(mode="on", tile_nx=256, tile_ny=256,
                                   nbuffers=3, halo=16)
    admitted = st.admit_pinned_road(
        cfg, one_tile, st.decide(cfg, one_tile),
        machine=ap.Machine(64 * GIB, 256 * GIB))
    assert admitted["nbuffers"] == 1


def _pinned_cyclone():
    exp = _cyclone("off")
    return replace(exp, tiles=st.StreamingOptions(
        mode="on", tile_nx=64, tile_ny=64, nbuffers=2))


PINNED_REFUSAL = r"\[tiles\] pins a 64x64"


def test_the_prepared_door_prices_a_pinned_tiling_on_its_card(
        tmp_path, monkeypatch):
    """A pinned domain at a reload door is refused before its restore.

    Red on the head: the door reads no card for a pinned road, so the
    tiling is never priced and the restore road is taken.
    """
    exp = _pinned_cyclone()
    _stand_in_card(monkeypatch, 1024 ** 2)
    psdf, inputs = _stub_prepared_single_door(monkeypatch, tmp_path, exp,
                                              intervals=2)
    with pytest.raises(ap.CannotPlan, match=PINNED_REFUSAL):
        psdf.run_prepared_forecast(inputs, output_directory=tmp_path,
                                   kernel_cache_census=object())


def test_the_run_door_prices_a_pinned_tiling_before_the_fetch(
        tmp_path, monkeypatch):
    """``gpuwm run`` with a pinned tiling: refused on the card before the fetch.

    Red on the head: the fetch begins with the tiling unpriced.
    """
    from gpuwm import runtime
    import gpuwm.ingest.preflight as ingest_preflight

    exp = _pinned_cyclone()
    _stand_in_card(monkeypatch, 1024 ** 2)
    monkeypatch.setattr(ingest_preflight, "build_input_catalog", _reached)
    with pytest.raises(ap.CannotPlan, match=PINNED_REFUSAL):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "out")


def test_the_downscale_door_prices_a_pinned_tiling_before_it_interpolates(
        tmp_path, monkeypatch):
    """The offline child with a pinned tiling: refused before its first read.

    Red on the head: the child interpolates its parent with the tiling
    unpriced.
    """
    import gpuwm.offline_child_run as child_run
    from gpuwm.offline_child import OfflineChildContractError
    from test_downscale_pricing import _stub_cupy

    _stub_cupy(monkeypatch)
    _stand_in_card(monkeypatch, 1024 ** 2)
    monkeypatch.setattr(child_run, "interpolate_parent_initial_state",
                        _reached)
    with pytest.raises(OfflineChildContractError, match=r"\[tiles\] pins a"):
        _drive_pinned_child(tmp_path / "on")


def _drive_pinned_child(run_dir):
    """The fixture child runner, its child pinned to 6x5 tiles."""
    from argparse import Namespace
    from datetime import datetime, timedelta

    import gpuwm.offline_child_run as child_run
    from test_downscale_pricing import _runnable_child
    from test_offline_child import _history

    run_dir.mkdir()
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(run_dir / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = run_dir / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child = _runnable_child(run_dir, tiles_mode="on")
    with child.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("tile_nx = 6\ntile_ny = 5\nnbuffers = 2\n")
    args = Namespace(
        parent_history=sorted(run_dir.glob("wrfout_d03_*")),
        parent_restart=None, parent_namelist=namelist, parent_domain_id=3,
        child_config=child, parent_grid_ratio=1, i_parent_start=4,
        j_parent_start=4, max_boundary_interval_seconds=3600.0,
        accepted_parent_cadence=True, child_surface_from=None,
        # This fixture owns no geography tree. The own-terrain default
        # otherwise refuses before the pinned-tiling admission it exercises.
        child_terrain="parent",
        preprocess_backend="cpu", health_interval_seconds=60.0,
        outdir=run_dir / "child-run")
    child_run._run(args, child_run._ChildProgress())


# ---------------------------------------------------------------------------
# the store loader's slab
# ---------------------------------------------------------------------------


def _preimport_store_loader():
    """The store loader's own imports, bound before the stand-in runtime."""
    for name in ("tilestream.driver", "tilestream.hoststore",
                 "tilestream.physics_inventory", "tilestream.realdata",
                 "gpuwm.ingest.hrrr_physics", "gpuwm.ingest.prepared_store"):
        pytest.importorskip(name)


def test_the_reference_slab_prices_like_the_planners_loader():
    """``rows_per_slab=1024`` at mp=18 is a full-size state on the card."""
    from gpuwm.core.resident_admission import slab_device_peak_bytes
    from gpuwm.ingest.prepared_store import default_slab_rows

    cfg = _reference(mp_physics=18)
    assert slab_device_peak_bytes(cfg, 1024, p_top=5000.0) > 24 * GIB
    rows = default_slab_rows(cfg.nx, cfg.ny)
    assert slab_device_peak_bytes(cfg, rows, p_top=5000.0) < GIB


def _drive_store_loader(tmp_path, monkeypatch, *, free_bytes, rows):
    _preimport_store_loader()
    import gpuwm.ingest.prepared_store as prepared_store

    path, identity, cfg, static = _written_cache(tmp_path)
    planned = []

    def plan(ny, rows_per_slab):
        planned.append(int(rows_per_slab))
        raise _ConstructorReached("the slab constructor is next")

    monkeypatch.setattr(prepared_store, "_plan_slabs", plan)
    _stand_in_runtime(monkeypatch, free_bytes=free_bytes)
    lines = []
    with pytest.raises(_ConstructorReached):
        prepared_store.store_from_prepared_cache(
            path, expected_identity=identity, cfg=cfg, static=static,
            landuse_attrs={}, grid=SimpleNamespace(cen_lat=40.0),
            valid_time=None, rows_per_slab=rows, log=lines.append)
    return cfg, planned, lines


def test_an_explicit_slab_too_tall_for_the_card_is_loaded_in_one_that_fits(
        tmp_path, monkeypatch):
    """The slab constructor is given a slab the card holds.

    Red on the head: the requested two-row slab reaches the constructor on
    a card that holds one row.
    """
    from gpuwm.core.resident_admission import slab_device_peak_bytes

    path, identity, cfg, static = _written_cache(tmp_path / "probe")
    one = slab_device_peak_bytes(cfg, 1, p_top=10_000.0)
    two = slab_device_peak_bytes(cfg, 2, p_top=10_000.0)
    assert one < two
    _cfg, planned, lines = _drive_store_loader(
        tmp_path, monkeypatch, free_bytes=one, rows=2)
    assert planned == [1]
    assert any("loads in 1-row slabs" in line for line in lines)


def test_a_slab_no_card_can_hold_is_refused_before_its_constructor(
        tmp_path, monkeypatch):
    with pytest.raises(MemoryError, match=REFUSAL):
        _drive_store_loader(tmp_path, monkeypatch, free_bytes=64, rows=2)


def test_a_slab_that_fits_is_loaded_as_asked(tmp_path, monkeypatch):
    _cfg, planned, lines = _drive_store_loader(
        tmp_path, monkeypatch, free_bytes=8 * GIB, rows=2)
    assert planned == [2] and not any("slabs;" in line for line in lines)


# ---------------------------------------------------------------------------
# --no-memory-gate: the envelope is skippable, the floor is not
# ---------------------------------------------------------------------------
#
# The forecast admission prices the peak ENVELOPE, the measured upper bound
# every review prices, which sits above the true peak by design (a 552x552x49
# child on a 5070 Ti card: envelope 14,922,267,728 B, pool peak
# 12,428,445,696 B).  `gpuwm go --no-memory-gate` exists for that margin, and
# the remote worker passes it on every job.  Without these tests the runner go
# starts refused the same envelope again after the fetch and the preparation,
# so the override did nothing for a resident forecast.


def _override(monkeypatch):
    from gpuwm.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV

    monkeypatch.setenv(MEMORY_GATE_OVERRIDE_ENV, "1")


def _tree_door(monkeypatch, tmp_path, exp, *, source="gfs", rows=None):
    """Drive the prepared tree door as far as its shared workspaces.

    ``rows`` are the root cache's retained interval rows; the default
    names no fields, so the door prices ``source``'s own row.
    """
    from gpuwm import case_data
    from gpuwm import prepared_domain_tree_forecast as tree
    import gpuwm.core.state as state_module

    _door_runtime(monkeypatch)
    monkeypatch.setattr(tree, "_verify_thompson_assets", lambda exp: None)
    monkeypatch.setattr(tree, "_with_terrain_acoustics", lambda inputs: inputs)
    monkeypatch.setattr(tree, "_runtime_source_identity", lambda: {})
    monkeypatch.setattr(tree, "scan_kernel_cache", lambda: None)
    monkeypatch.setattr(tree, "_prepared_planning_nodes",
                        lambda inputs: st._config_tree_nodes(exp.domains))
    monkeypatch.setattr(case_data, "trace_gas_overrides_from_config",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(state_module, "build_shared_scratch_arena", _reached)
    monkeypatch.setattr(state_module, "build_shared_dycore_state_workspace",
                        _reached)
    reader = SimpleNamespace(header={"metadata": {"lbc": {
        "intervals": [{}] * 6 if rows is None else rows}}})
    inputs = SimpleNamespace(
        experiment=exp, experiment_config=tmp_path / "experiment.toml",
        authority_sha256={"experiment_config": "0" * 64},
        execution_plan={}, boundary_interval_seconds=3600, source=source,
        domains=tuple(SimpleNamespace(parent_id=dc.parent_id,
                                      grid_id=dc.grid_id,
                                      cache_reader=reader)
                      for dc in exp.domains))
    return tree, inputs


def test_the_override_lets_every_resident_door_reach_its_constructor(
        tmp_path, monkeypatch, capsys):
    """Over the envelope, under ``--no-memory-gate``: each door proceeds.

    The prepared single-domain door, the prepared tree door, ``gpuwm run``'s
    single-domain arm and the downscale child, each on a card one byte short
    of its envelope (the child on a 1 MiB card), each reaching the
    allocation its admission stands in front of, with one warning line
    naming what was skipped.  Red at the returned tip: every one refused.
    """
    from gpuwm import runtime
    import gpuwm.ingest.preflight as ingest_preflight
    import gpuwm.offline_child_run as child_run
    from test_downscale_pricing import (
        _Sentinel, _drive_the_runner_to_its_decision)

    _override(monkeypatch)

    exp = _cyclone("off")
    _stand_in_card(monkeypatch, _envelope(exp, forcing_intervals=24) - 1)
    psdf, inputs = _stub_prepared_single_door(monkeypatch, tmp_path, exp,
                                              intervals=24)
    (tmp_path / "s").mkdir()
    with pytest.raises(_ConstructorReached):
        psdf.run_prepared_forecast(inputs, output_directory=tmp_path / "s",
                                   kernel_cache_census=object())
    warned = capsys.readouterr().err
    assert "this prepared forecast, held resident on the card" in warned
    assert "--no-memory-gate skips that check" in warned

    tree_exp = _cyclone("off", single=False)
    _stand_in_card(monkeypatch,
                   _envelope(tree_exp, forcing_intervals=6) - 1)
    tree, tree_inputs = _tree_door(monkeypatch, tmp_path, tree_exp)
    (tmp_path / "t").mkdir()
    with pytest.raises(_ConstructorReached):
        tree.run_prepared_tree(tree_inputs, output_directory=tmp_path / "t",
                               io_mode="none")

    _stand_in_card(monkeypatch, _envelope(exp) - 1)
    monkeypatch.setattr(ingest_preflight, "build_input_catalog", _reached)
    with pytest.raises(_ConstructorReached):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "run")

    _stand_in_card(monkeypatch, 1024 ** 2)
    monkeypatch.setattr(child_run, "interpolate_parent_initial_state",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            _Sentinel()))
    capsys.readouterr()
    _drive_the_runner_to_its_decision(tmp_path / "child", tiles_mode=None)
    assert ("this downscaled child, held resident on the card"
            in capsys.readouterr().err)


def test_the_override_never_skips_a_constructor_floor(tmp_path, monkeypatch):
    """The floor is exact bytes the loader allocates, so it still refuses."""
    import gpuwm.core.state as state_module
    from gpuwm.ingest.prepared_cache import restore_prepared_cache

    _override(monkeypatch)
    path, identity, cfg, static = _written_cache(tmp_path)
    _stand_in_runtime(monkeypatch, free_bytes=1024)
    monkeypatch.setattr(state_module, "DomainState", _reached)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        restore_prepared_cache(path, expected_identity=identity, cfg=cfg,
                               static=static)
    assert "would stop with a CUDA out-of-memory" in str(refused.value)
    assert "--no-memory-gate" not in str(refused.value)


def test_an_envelope_refusal_names_the_override_and_says_expected(
        monkeypatch):
    """The envelope is an upper bound, and the refusal says so.

    It used to say the forecast WOULD stop in a CUDA out-of-memory, which
    is false for a forecast inside the envelope's margin.
    """
    from gpuwm.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    exp = _cyclone("off")
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        st.admit_resident_road(exp, None, machine=_small())
    text = str(refused.value)
    assert "peak envelope" in text
    assert "expected to stop with a CUDA out-of-memory" in text
    assert "--no-memory-gate" in text and MEMORY_GATE_OVERRIDE_ENV in text


def test_the_override_skips_a_pinned_tiling_price_too(monkeypatch):
    """The pinned budget keeps the planner's headroom back: skippable."""
    cfg, pinned, decision = _pinned_reference()
    machine = ap.Machine(24 * GIB, 256 * GIB)
    _override(monkeypatch)
    admitted = st.admit_pinned_road(cfg, pinned, decision, machine=machine)
    assert admitted["overridden"] is True
    assert admitted["need_bytes"] > admitted["budget_bytes"]


def test_the_tree_door_prices_the_tables_the_run_holds(tmp_path, monkeypatch):
    """An external boundary set is priced by the resident admission.

    The door's own ledger estimate carries the external boundary set and
    the forcing cadence; the admission built a second estimate without
    them, so a tree whose external tables tipped it over the card was
    admitted and stopped in CUDA.  Red at the returned tip: the tree
    reaches its shared workspaces.

    An external boundary set is how the ``gpuwm run --wrfinput`` and
    ``--met-em`` doors reach this runner (``run_wrf_forecast`` and
    ``run_metem_forecast`` call ``run_prepared_tree`` with an
    initialization carrying ``lateral_boundaries``), so this admission is
    theirs too, priced on the tables their own inputs carry.
    """
    exp = _cyclone("off", single=False)
    values = 64 * 1024 ** 2
    side = SimpleNamespace(
        array_items=lambda: [("value", SimpleNamespace(size=values)),
                             ("tendency", SimpleNamespace(size=values))],
        time_law=None)
    interval = SimpleNamespace(fields={"u": SimpleNamespace(
        west=side, east=side, south=side, north=side)})
    boundaries = SimpleNamespace(intervals=[interval] * 6)
    initialization = SimpleNamespace(lateral_boundaries=boundaries,
                                     verify_inputs=lambda inputs: None)
    held = pf.estimate_experiment(
        exp, forcing_interval_seconds=3600, forcing_intervals=6,
        lateral_boundaries=boundaries).peak_envelope_bytes
    assert held - _envelope(exp, forcing_intervals=6) > 8 * GIB
    _stand_in_card(monkeypatch, held - 1)
    tree, inputs = _tree_door(monkeypatch, tmp_path, exp)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        tree.run_prepared_tree(inputs, output_directory=tmp_path,
                               io_mode="none", initialization=initialization)
    assert refused.value.need_bytes == held


def test_the_run_tree_door_admits_a_resident_tree_before_it_builds(
        tmp_path, monkeypatch, capsys):
    """``gpuwm run`` on a tree with no ``[tiles]``: priced before the build.

    The tree walk consults nothing when nothing streams, and the route went
    straight to ``build_experiment``, so a tree too big for the card stopped
    in a CUDA out-of-memory part way through its build.  Red on the head:
    the build is reached on a card one byte short of the envelope.  Under
    ``--no-memory-gate`` the same card reaches the build.
    """
    from gpuwm import runtime
    import gpuwm.core.model as model_module
    from gpuwm.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    exp = _cyclone("off", single=False)
    assert len(exp.domains) > 1
    need = _envelope(exp)
    card = _stand_in_card(monkeypatch, need - 1)
    monkeypatch.setattr(model_module, "build_experiment", _reached)
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "out")
    assert "this domain tree, held resident on the card" in str(
        refused.value)
    assert refused.value.terms["physics"] > 0
    assert refused.value.free_bytes == int(card.vram_bytes)
    assert refused.value.need_bytes == need

    _stand_in_card(monkeypatch, need)
    with pytest.raises(_ConstructorReached):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "fits")

    _override(monkeypatch)
    _stand_in_card(monkeypatch, need - 1)
    capsys.readouterr()
    with pytest.raises(_ConstructorReached):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "over")
    warned = capsys.readouterr().err
    assert "this domain tree, held resident on the card" in warned
    assert "--no-memory-gate skips that check" in warned
def _handed_in_boundaries(fields, *, intervals=6, values=64 * 1024 ** 2):
    """A root lateral boundary set as the wrfinput and met_em doors hand it in."""
    side = SimpleNamespace(
        array_items=lambda: [("value", SimpleNamespace(size=values)),
                             ("tendency", SimpleNamespace(size=values))],
        time_law=None)
    boundary = SimpleNamespace(west=side, east=side, south=side, north=side)
    interval = SimpleNamespace(fields={name: boundary for name in fields})
    return SimpleNamespace(intervals=[interval] * intervals)


def _wrf_door(monkeypatch, tmp_path, exp, door, boundaries):
    """The tree runner's inputs and initialization as a WRF-input door builds them.

    The door's own types: :class:`gpuwm.wrfinput_forecast.WrfTreeInputs`
    holding :class:`~gpuwm.wrfinput_forecast.WrfDomainBundle` (no
    ``parent_id``, no ``cache_reader``) for ``gpuwm run --wrfinput``, or
    :class:`gpuwm.metem_forecast.MetemDomainBundle` (a ``cache_reader`` but
    no ``parent_id``) for ``gpuwm run --met-em``, with the door's own
    initialization carrying the root's boundary set.  The met_em cache
    readers carry no boundary inventory, so a door that read one instead of
    the set it was handed stops on it.
    """
    import hashlib

    from gpuwm import metem_forecast as metem
    from gpuwm import wrfinput_forecast as wrf

    tree, _ = _tree_door(monkeypatch, tmp_path, exp)
    config = tmp_path / f"{door}-experiment.toml"
    config.write_bytes(b"# stand-in\n")
    sha = {"experiment_config": hashlib.sha256(config.read_bytes()).hexdigest()}
    identity = wrf.WrfLanduseIdentity({})
    if door == "wrfinput":
        bundles = tuple(
            wrf.WrfDomainBundle(grid_id=dc.grid_id, restored=None,
                                static_fields={}, authority_sha256={},
                                landuse=None, geog_selection=identity)
            for dc in exp.domains)
    else:
        reader = SimpleNamespace(
            header={}, verify_all=lambda: {"content_sha256": "c" * 64})
        bundles = tuple(
            metem.MetemDomainBundle(
                grid_id=dc.grid_id, cache=tmp_path / f"d{dc.grid_id:02d}",
                cache_identity={}, cache_reader=reader, static_fields={},
                authority_sha256={"cache_content": "c" * 64},
                geog_selection=identity, fractional_seaice=False,
                isoilwater=1)
            for dc in exp.domains)
    inputs = wrf.WrfTreeInputs(
        prepared_root=tmp_path, experiment_config=config, experiment=exp,
        grids=(), domains=bundles, forcing_hours=(0.0,),
        boundary_interval_seconds=3600, source_identity={},
        execution_plan={}, authority_sha256=sha,
        artifact_paths={"experiment_config": config}, boundaries=boundaries,
        **({} if door == "wrfinput" else {"source": "met_em"}))
    initialization = (wrf.WrfInitialization(inputs) if door == "wrfinput"
                      else metem.MetemInitialization(inputs))
    return tree, inputs, initialization


@pytest.mark.parametrize("door", ["wrfinput", "met_em"])
def test_the_wrf_input_doors_reach_their_admission_on_their_own_bundles(
        tmp_path, monkeypatch, door):
    """``gpuwm run --wrfinput`` and ``--met-em`` reach the tree admission.

    Driven on the doors' own bundle, input and initialization types, which
    carry no ``parent_id`` (and, for wrfinput, no ``cache_reader``).  Red at
    the returned tip: the runner looked the root's cache up by
    ``bundle.parent_id`` before asking whether a boundary set was handed in,
    and both doors stopped with AttributeError before step 0.  They are
    priced on the set they hand in, refused one byte short of it and
    admitted to their first allocation at it.
    """
    exp = _cyclone("off", single=False)
    boundaries = _handed_in_boundaries(("u", "qv"))
    held = pf.estimate_experiment(
        exp, forcing_interval_seconds=3600, forcing_intervals=6,
        lateral_boundaries=boundaries).peak_envelope_bytes
    for index, free in enumerate((held - 1, held)):
        _stand_in_card(monkeypatch, free)
        (tmp_path / f"{index}").mkdir()
        tree, inputs, initialization = _wrf_door(
            monkeypatch, tmp_path / f"{index}", exp, door, boundaries)
        outdir = tmp_path / f"{index}" / "out"
        outdir.mkdir()
        if free < held:
            with pytest.raises(MemoryError, match=REFUSAL) as refused:
                tree.run_prepared_tree(inputs, output_directory=outdir,
                                       io_mode="none",
                                       initialization=initialization)
            assert refused.value.need_bytes == held
            continue
        with pytest.raises(_ConstructorReached):
            tree.run_prepared_tree(inputs, output_directory=outdir,
                                   io_mode="none",
                                   initialization=initialization)


@pytest.mark.parametrize("door", ["wrfinput", "met_em"])
def test_a_handed_in_boundary_set_is_priced_on_the_masses_it_holds(
        tmp_path, monkeypatch, door):
    """The tiles decision of a WRF-input door prices its set's own masses.

    A set carrying cloud water and rain is what the run holds, so the
    shared admission estimate is handed those two masses; a set of
    dynamics and water vapour alone prices none, as the door's source
    name (``wrfinput``, ``met_em``) does.
    """
    seen = []

    def admission_estimate(exp, *, machine=None, source=None):
        seen.append(source)
        raise _Priced()

    exp = _cyclone("auto", single=False)
    _stand_in_card(monkeypatch, 64 * GIB)
    for index, fields in enumerate((("u", "qv", "qc", "qr"), ("u", "qv"))):
        (tmp_path / f"{index}").mkdir()
        tree, inputs, initialization = _wrf_door(
            monkeypatch, tmp_path / f"{index}", exp, door,
            _handed_in_boundaries(fields))
        monkeypatch.setattr(pf, "admission_estimate", admission_estimate)
        outdir = tmp_path / f"{index}" / "out"
        outdir.mkdir()
        with pytest.raises(_Priced):
            tree.run_prepared_tree(inputs, output_directory=outdir,
                                   io_mode="none",
                                   initialization=initialization)
    assert seen == [("qc", "qr"), ()]


# ---------------------------------------------------------------------------
# the boundary tables a prepared cache carries
# ---------------------------------------------------------------------------
#
# The root's specified boundary carries the analysed hydrometeors its
# preparation installed (gpuwm.boundary_fields).  A prepared door prices
# them from the cache it is about to restore, not from its source's name:
# a user's own mapping runs as ``mapped``, whose row publishes none while
# its cache carries every mass the mapping declares, and a cache sealed
# before the boundary carried them holds water vapour alone whatever its
# source publishes today.


def _sealed_rows(exp, count, masses):
    """``count`` retained interval rows whose tables carry ``masses``.

    The inventory a preparation writes (sorted, dynamics plus the scalar
    fields :func:`gpuwm.boundary_fields.external_scalar_fields` gives).
    """
    from gpuwm.boundary_fields import external_scalar_fields

    scalars = external_scalar_fields(exp.root.run, aerosol_from_input=False,
                                     boundary_species=masses)
    fields = sorted({"mu", "phi", "theta", "u", "v", *scalars})
    return [{"start_seconds": 3600.0 * index,
             "end_seconds": 3600.0 * (index + 1), "fields": list(fields)}
            for index in range(count)]


MASSES = ("qc", "qr", "qi", "qs", "qg")


def test_the_prepared_doors_price_the_masses_a_mapped_cache_carries(
        tmp_path, monkeypatch):
    """``--source mapped``: priced with the five masses its cache holds.

    Red at the returned tip: both doors priced the name ``mapped``, which
    publishes none, so each asked for the water-vapour envelope and a card
    one byte short of the real one admitted the forecast.
    """
    from gpuwm.boundary_fields import source_boundary_species

    assert source_boundary_species("mapped") == ()
    exp = _cyclone("off")
    need = _envelope(exp, forcing_intervals=24, boundary_species=MASSES)
    assert need > _envelope(exp, forcing_intervals=24)
    _stand_in_card(monkeypatch, need - 1)
    psdf, inputs = _stub_prepared_single_door(
        monkeypatch, tmp_path, exp, intervals=24, source="mapped",
        rows=_sealed_rows(exp, 24, MASSES))
    (tmp_path / "s").mkdir()
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        psdf.run_prepared_forecast(inputs, output_directory=tmp_path / "s",
                                   kernel_cache_census=object())
    assert refused.value.need_bytes == need

    tree_exp = _cyclone("off", single=False)
    tree_need = _envelope(tree_exp, forcing_intervals=6,
                          boundary_species=MASSES)
    assert tree_need > _envelope(tree_exp, forcing_intervals=6)
    _stand_in_card(monkeypatch, tree_need - 1)
    tree, tree_inputs = _tree_door(
        monkeypatch, tmp_path, tree_exp, source="mapped",
        rows=_sealed_rows(tree_exp, 6, MASSES))
    (tmp_path / "t").mkdir()
    with pytest.raises(MemoryError, match=REFUSAL) as refused:
        tree.run_prepared_tree(tree_inputs, output_directory=tmp_path / "t",
                               io_mode="none")
    assert refused.value.need_bytes == tree_need


def test_a_cache_sealed_with_water_vapour_edges_is_priced_on_them(
        tmp_path, monkeypatch):
    """A HRRR cache sealed before its boundary carried hydrometeors.

    It still runs, on its water-vapour-only boundary, so it holds no
    hydrometeor tables and is priced without them: it fits a card exactly
    its water-vapour envelope in size and is refused one byte below.  Red
    at the returned tip: priced by the name ``hrrr``, which publishes them,
    such a cache near the card's limit was refused although it fits.
    """
    from gpuwm.boundary_fields import source_boundary_species

    assert source_boundary_species("hrrr")
    exp = _cyclone("off")
    vapour = _envelope(exp, forcing_intervals=24)
    for index, free in enumerate((vapour, vapour - 1)):
        _stand_in_card(monkeypatch, free)
        psdf, inputs = _stub_prepared_single_door(
            monkeypatch, tmp_path, exp, intervals=24, source="hrrr",
            rows=_sealed_rows(exp, 24, ()))
        outdir = tmp_path / f"s{index}"
        outdir.mkdir()
        if free == vapour:
            with pytest.raises(_ConstructorReached):
                psdf.run_prepared_forecast(inputs, output_directory=outdir,
                                           kernel_cache_census=object())
            continue
        with pytest.raises(MemoryError, match=REFUSAL) as refused:
            psdf.run_prepared_forecast(inputs, output_directory=outdir,
                                       kernel_cache_census=object())
        assert refused.value.need_bytes == vapour

    tree_exp = _cyclone("off", single=False)
    tree_vapour = _envelope(tree_exp, forcing_intervals=6)
    _stand_in_card(monkeypatch, tree_vapour)
    tree, tree_inputs = _tree_door(monkeypatch, tmp_path, tree_exp,
                                   source="hrrr",
                                   rows=_sealed_rows(tree_exp, 6, ()))
    (tmp_path / "t").mkdir()
    with pytest.raises(_ConstructorReached):
        tree.run_prepared_tree(tree_inputs, output_directory=tmp_path / "t",
                               io_mode="none")


class _Priced(Exception):
    """The door handed its tiles admission the boundary it prices."""


def test_each_prepared_door_takes_its_tiles_decision_on_the_cache_it_restores(
        tmp_path, monkeypatch):
    """The ``[tiles]`` admission of both prepared doors, spied.

    Each door asks the shared admission estimate for its tiles decision
    before the restore; it has to hand that estimate the masses the cache
    carries, or a tiled or auto-resident mapped forecast is judged on
    water vapour alone.  Red at the returned tip: both handed ``mapped``.
    """
    seen = []

    def admission_estimate(exp, *, machine=None, source=None):
        seen.append(source)
        raise _Priced()

    exp = _cyclone("auto")
    _stand_in_card(monkeypatch, 64 * GIB)
    psdf, inputs = _stub_prepared_single_door(
        monkeypatch, tmp_path, exp, intervals=24, source="mapped",
        rows=_sealed_rows(exp, 24, MASSES))
    monkeypatch.setattr(pf, "admission_estimate", admission_estimate)
    (tmp_path / "s").mkdir()
    with pytest.raises(_Priced):
        psdf.run_prepared_forecast(inputs, output_directory=tmp_path / "s",
                                   kernel_cache_census=object())

    tree_exp = _cyclone("auto", single=False)
    tree, tree_inputs = _tree_door(
        monkeypatch, tmp_path, tree_exp, source="mapped",
        rows=_sealed_rows(tree_exp, 6, MASSES))
    (tmp_path / "t").mkdir()
    with pytest.raises(_Priced):
        tree.run_prepared_tree(tree_inputs, output_directory=tmp_path / "t",
                               io_mode="none")
    assert seen == [MASSES, MASSES]


def test_go_holds_the_override_for_the_runner_it_starts(tmp_path, monkeypatch):
    """``gpuwm go --no-memory-gate`` reaches the forecast runner.

    The remote worker passes the flag on every entry-door job; the runner go
    hosts (or starts as a subprocess, which inherits this environment) reads
    it where it prices the envelope.  Held for the chain and put back after.
    """
    import os

    from gpuwm import runplan
    from gpuwm.cli import main
    from gpuwm.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV
    from test_go_native_launch import allow_launch_resources, emit

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)
    seen = []

    def execute(*args, **kwargs):
        seen.append(os.environ.get(MEMORY_GATE_OVERRIDE_ENV))
        return 0

    monkeypatch.setattr(runplan, "execute_plan", execute)
    assert main(["go", str(config), "--outdir", str(tmp_path / "gated"),
                 "--products", "none"]) == 0
    assert main(["go", str(config), "--outdir", str(tmp_path / "skipped"),
                 "--products", "none", "--no-memory-gate"]) == 0
    assert seen == [None, "1"]
    assert MEMORY_GATE_OVERRIDE_ENV not in os.environ


def test_every_forecast_door_takes_no_memory_gate(tmp_path, monkeypatch):
    """The refusal names ``--no-memory-gate``; each door that prices it takes it.

    ``gpuwm run`` holds it for the worker it supervises (which inherits the
    environment), ``gpuwm sim`` passes it to the runner it names, and the
    two prepared runners and the child runner take it themselves.
    """
    import os

    from gpuwm import cli
    from gpuwm import offline_child_run
    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm import prepared_single_domain_forecast as single
    from gpuwm import stage_cli
    from gpuwm.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV

    parser = cli.build_parser()
    for argv in (["run", "case.toml"], ["resume", "case.toml"],
                 ["downscale", "parent-run", "--out", "child"],
                 ["sim", "prepared", "--experiment-config", "e.toml",
                  "--outdir", "out"]):
        assert parser.parse_args(argv).no_memory_gate is False
        assert parser.parse_args(argv + ["--no-memory-gate"]).no_memory_gate
    assert "--no-memory-gate" in single.build_parser().format_help()
    assert "--no-memory-gate" in tree.build_parser().format_help()
    assert "--no-memory-gate" in offline_child_run._parser().format_help()

    bundle = {"layout": "tree", "document": tmp_path / "proof.json"}
    monkeypatch.setattr(stage_cli, "tree_digests",
                        lambda bundle, config: {
                            "preparation_receipt": "0" * 64,
                            "experiment_config": "1" * 64})
    plain = stage_cli.sim_command(bundle, experiment_config=tmp_path / "e",
                                  wps_namelist=None, outdir=tmp_path / "o")
    assert "--no-memory-gate" not in plain
    assert "--no-memory-gate" in stage_cli.sim_command(
        bundle, experiment_config=tmp_path / "e", wps_namelist=None,
        outdir=tmp_path / "o", memory_gate=False)

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    seen = []
    monkeypatch.setattr(cli, "_dispatch", lambda args: seen.append(
        os.environ.get(MEMORY_GATE_OVERRIDE_ENV)) or 0)
    monkeypatch.setattr(cli.capabilities, "require_for_command",
                        lambda *args, **kwargs: None)
    assert cli.main(["run", "case.toml", "--no-memory-gate"]) == 0
    assert cli.main(["run", "case.toml"]) == 0
    assert seen == ["1", None]
    assert MEMORY_GATE_OVERRIDE_ENV not in os.environ
