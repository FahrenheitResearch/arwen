"""Ranked planning and refusals without a CUDA device."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.core import streaming
from gpuwm.core.devices import DeviceOptions
from gpuwm.core.cfl_inventory import fold_cfl_words, WRF_CFL_WORDS
from tilestream import driver, harness, multigpu, physics_inventory
from tilestream.ranks import RankedRun, RankedRunError, choose_transports


def test_ranked_plan_acoustic_ceiling_and_forcing():
    cfg = replace(multigpu.forced_config(1797, 1057, 50), dx=3000., dy=3000.,
                  relax_zone=9, use_adaptive_time_step=True, max_time_step=60)
    options = DeviceOptions(count=2, ids=(0, 0))
    halo = streaming.ranked_halo(cfg)
    from gpuwm.core.adaptive_clock import acoustic_step_ceiling
    assert halo == harness.halo_radius(replace(cfg,
        time_step_sound=acoustic_step_ceiling(cfg))) + 9
    specs = streaming.ranked_specs(cfg, options, halo=halo)
    assert [(s.i0, s.i1, s.j0, s.j1) for s in specs] == [
        (0, 898, 0, 1057), (898, 1797, 0, 1057)]
    assert specs[0].halo_left == specs[1].halo_right == 0
    assert specs[0].halo_right == specs[1].halo_left == halo
    decision = streaming.ranked_decision(cfg, options)
    assert (decision.road, decision.stream, decision.store, decision.nbuffers,
            decision.ntiles, decision.tile_nx, decision.tile_ny) == (
                "ranks", True, "host", 2, 2, 899, 1057)
    assert decision.redundancy == (1797+2*halo)/1797
    assert decision.detail["grid"] == [1, 2]
    assert decision.explain() == f"[devices] ON (2 slabs on cards [0, 0], grid 1x2, halo {halo})"


def test_small_ranked_plan_and_tiles_text_unchanged():
    cfg = harness.make_config(80, 64, 8)
    options = DeviceOptions(count=4, grid=(2, 2), ids=(0, 0, 0, 0))
    specs = streaming.ranked_specs(cfg, options, halo=4)
    assert [(s.interior_nx, s.interior_ny) for s in specs] == [(40, 32)]*4
    decision = streaming.StreamingDecision(True, "explicit", tile_nx=10,
        tile_ny=12, nbuffers=2, halo=16, ntiles=4, redundancy=2.)
    assert decision.road == "tiles"
    assert decision.explain() == ("[tiles] ON (host store): explicit; tile 10x12, "
        "nbuffers=2, halo=16, 4 tiles at 2.00x redundancy")


@pytest.mark.parametrize("matrix, expected", [
    ({(0, 1): True, (1, 0): True}, "peer"),
    ({(0, 1): True, (1, 0): False}, "staged"), ({}, "staged")])
def test_runtime_transport_choice(matrix, expected):
    paths = choose_transports([0, 1, 0], "auto", matrix)
    assert paths[(0, 0)] == "local"
    assert paths[(0, 1)] == paths[(1, 0)] == expected
    if expected == "staged":
        with pytest.raises(RankedRunError, match="both directions"):
            choose_transports([0, 1], "peer", matrix)
    assert choose_transports([0, 1], "host", matrix)[(0, 1)] == "host"


def test_live_config_refuses_structure_and_short_halo():
    cfg = harness.make_config(80, 64, 8)
    run = RankedRun.__new__(RankedRun)
    run.cfg = cfg
    run.halo = harness.halo_radius(cfg)
    run.sub_cfgs = [harness.tile_config(cfg, 56, 64)]*2
    with pytest.raises(RankedRunError, match="stale setup"):
        run._set_live_config(replace(cfg, dx=cfg.dx*2))
    with pytest.raises(RankedRunError, match="live time_step_sound"):
        run._set_live_config(replace(cfg, time_step_sound=cfg.time_step_sound+100))
    run._set_live_config(replace(cfg, dt=cfg.dt/2))
    assert all(rank.dt == cfg.dt/2 for rank in run.sub_cfgs)


@pytest.mark.parametrize("carrier", ["elapsed_seconds", "call_counts", "microphysics_updates"])
def test_clock_disagreement_refuses_different_physics(monkeypatch, carrier):
    monkeypatch.setattr(physics_inventory, "carrier_scalars", lambda tile: tile)
    base = {"elapsed_seconds": 1., "call_counts": {}, "microphysics_updates": 1}
    other = dict(base)
    other[carrier] = {"radiation": 2} if carrier == "call_counts" else 2
    tiles = [base, other]
    with pytest.raises(driver.TiledRunError, match="integrated different physics"):
        driver._advance_clock({}, tiles, 0, physics_inventory)


def test_cfl_device_fold_matches_atomic_word_arithmetic():
    rows = np.zeros((3, 2, WRF_CFL_WORDS), np.uint32)
    rows[:, :, 0] = np.array([1., 3., 2.], np.float32).view(np.uint32)[:, None]
    rows[:, :, 3] = np.array([4., 2., 8.], np.float32).view(np.uint32)[:, None]
    rows[:, :, 1:3] = np.arange(6, dtype=np.uint32).reshape(3, 2, 1)
    rows[:, :, 4:] = 7
    got = fold_cfl_words(rows)
    assert np.array_equal(got[:, 0].view(np.float32), [3., 3.])
    assert np.array_equal(got[:, 3].view(np.float32), [8., 8.])
    assert np.array_equal(got[:, 1:3], [[6, 6], [9, 9]])
    assert np.all(got[:, 4:] == 21)
    overflow = np.zeros((2, WRF_CFL_WORDS), np.uint32)
    overflow[:, 2] = [0xffffffff, 2]
    assert fold_cfl_words(overflow)[2] == 1


def test_builder_refuses_unbound_forcing_and_nests():
    from gpuwm.ingest.lateral_bc import LateralBoundaries
    bundle = SimpleNamespace(boundaries=LateralBoundaries(
        intervals=[], spec_bdy_width=5, spec_zone=1, relax_zone=4))
    cfg = multigpu.forced_config(80, 64, 8)
    build = streaming.ranked_domain_builder(bundle, options=DeviceOptions(count=2, ids=(0, 0)))
    with pytest.raises(streaming.StreamingRefused, match="ONE TIMESTEP LATE"):
        build(None, cfg, None)
    with pytest.raises(streaming.StreamingRefused, match="needs node="):
        build(None, replace(cfg, nested=True), None)


def test_nested_ranked_halo_pays_the_boundary_frame():
    """A nest's slabs carry the same seam frame a specified domain's do.

    The nest's boundary application writes max(spec_zone, relax_zone) cells
    from its parent's rolling tables, which are zeros on a seam side, so the
    halo must keep that frame off owned cells, as for a specified domain.
    """
    cfg = replace(multigpu.forced_config(160, 120, 8), specified=False,
                  nested=True, relax_zone=4, spec_zone=1)
    from gpuwm.core.adaptive_clock import acoustic_step_ceiling
    bare = harness.halo_radius(replace(cfg, time_step_sound=acoustic_step_ceiling(cfg)))
    assert streaming.ranked_halo(cfg) == bare + 4
    specs = streaming.ranked_specs(cfg, DeviceOptions(count=2, ids=(0, 0)),
                                   halo=streaming.ranked_halo(cfg))
    with pytest.raises(multigpu.MultiGPUError, match="nested forcing is not wired"):
        multigpu.validate_forced_plan(cfg, specs, streaming.ranked_halo(cfg), None)
    multigpu.validate_forced_plan(cfg, specs, streaming.ranked_halo(cfg), None,
                                  nest_forcing=True)
    with pytest.raises(multigpu.MultiGPUError, match="nested decomposition radius"):
        multigpu.validate_forced_plan(cfg, specs, 2, None, nest_forcing=True)


def test_store_frame_downloads_exactly_what_fields_reads():
    """The ranked output road copies ``StoreFrame.store_keys()`` to the host.

    A store member ``fields()`` reads that ``store_keys()`` leaves out would
    be published from a stale host copy, so the derived-field inputs are
    read off ``fields()``'s own source and held equal to the table.
    """
    import inspect
    import re
    from tilestream.output import StoreFrame
    source = inspect.getsource(StoreFrame.fields)
    read = set(re.findall(r'store\["([^"]+)"\]', source))
    declared = {key for keys in StoreFrame._DERIVED_INPUTS.values() for key in keys}
    assert read == declared, (read, declared)
    assert set(StoreFrame._DERIVED_INPUTS) == {"T", "P", "PSFC"}


def test_deferred_frame_waits_then_assembles_once():
    from tilestream.output import DeferredStoreFrame, SOURCE_CARRIER, SOURCE_DRIVER_DIAGNOSTIC
    order = []
    plan = SimpleNamespace(order=("A", "OLR", "B"),
                           source={"A": SOURCE_CARRIER, "OLR": SOURCE_DRIVER_DIAGNOSTIC,
                                   "B": SOURCE_CARRIER})
    frame = SimpleNamespace(plan=plan, nbytes=lambda: 24,
                            fields=lambda: order.append("fields") or {"A": 1, "B": 2})
    deferred = DeferredStoreFrame(frame, lambda: order.append("wait"))
    assert deferred.deferred and deferred.names == ("A", "B") and deferred.nbytes == 24
    assert order == []
    assert deferred.materialize() == {"A": 1, "B": 2}
    assert order == ["wait", "fields"]
    with pytest.raises(RuntimeError, match="assembled once"):
        deferred.materialize()


def test_host_frame_composition_order_is_one_rule():
    """Frame rows, then REFL_10CM, then extras that replace nothing: one order
    for the admitted frame and the deferred one, because the order is the file."""
    from gpuwm.io.wrfout import _compose_host_frame
    fields = _compose_host_frame([("T", 1), ("HGT", 2)], 3, {"HGT": 9, "XLAT": 4})
    assert list(fields) == ["T", "HGT", "REFL_10CM", "XLAT"]
    assert fields["HGT"] == 2
    assert list(_compose_host_frame([("REFL_10CM", 0), ("T", 1)], 3, {})) == ["REFL_10CM", "T"]
    assert list(_compose_host_frame([("T", 1)], None, {})) == ["T"]


class _FakeCupy:
    """Just enough CuPy for the first-use upload caches, with a slow upload.

    The slabs of a split domain step on their own threads and streams, so
    their first physics call reaches each card-wide table cache at once.
    Every upload here sleeps, which holds the window between "is it cached?"
    and "publish it" open long enough for every thread to fall into it.
    """
    def __init__(self, delay=0.02):
        import threading
        self.delay = delay
        self.uploads = 0
        self.synchronized = []
        self._local = threading.local()
        self._count = threading.Lock()
        fake = self

        class Stream:
            def __init__(self, ptr):
                self.ptr = ptr
            def wait_event(self, event):
                pass
            def synchronize(self):
                fake.synchronized.append(self.ptr)

        class Event:
            done = True
            def __init__(self, *args, **kwargs):
                pass
            def record(self, stream=None):
                pass

        class Device:
            id = 0
            def __init__(self, *args):
                pass
            def synchronize(self):
                fake.synchronized.append("device")

        self.Stream = Stream
        self.cuda = SimpleNamespace(
            Event=Event, Device=Device,
            runtime=SimpleNamespace(getDevice=lambda: 0),
            get_current_stream=lambda: Stream(getattr(fake._local, "ptr", 0)),
            Stream=SimpleNamespace(null=Stream("null")))
        self.float32, self.int32, self.bool_, self.float64 = (
            np.float32, np.int32, np.bool_, np.float64)

    def on_stream(self, ptr):
        self._local.ptr = ptr

    def asarray(self, value, dtype=None, order="C"):
        import time
        with self._count:
            self.uploads += 1
        time.sleep(self.delay)
        return np.array(value, dtype=dtype, order=order)

    def ascontiguousarray(self, value):
        return np.ascontiguousarray(value)


def _race(fake, call, threads=8):
    """``call()`` from ``threads`` slab threads at once, each on its own stream."""
    import threading
    barrier = threading.Barrier(threads)
    results = [None] * threads
    def slab(rank):
        fake.on_stream(rank + 1)
        barrier.wait()
        results[rank] = call()
    workers = [threading.Thread(target=slab, args=(rank,)) for rank in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return results


def test_rrtmgp_tables_upload_once_per_card_under_concurrent_slabs(monkeypatch):
    """RRTMGP's k-distribution and cloud tables: one upload per card, ever.

    MEASURED: one RTX 4090 split into 2x2 slabs, the default suite, 4 of 5
    runs differed from the unsplit run from the first radiation call's
    shortwave heating, and 3 of 3 were identical with radiation serialized.
    Unlocked, every slab uploaded its own copy and the last replaced the
    others, so a slab could read a copy its stream's pool had handed back.
    """
    import sys
    from gpuwm.core import rrtmgp
    fake = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    tables = rrtmgp.CloudTables(
        kind="sw", nband=2, nsize_liq=2, nsize_ice=2, nrghice=1,
        radliq_lwr=2.5, radliq_upr=21.5, diamice_lwr=10.0, diamice_upr=180.0,
        extliq=np.ones((2, 2)), ssaliq=np.ones((2, 2)), asyliq=np.ones((2, 2)),
        extice=np.ones((2, 2, 1)), ssaice=np.ones((2, 2, 1)),
        asyice=np.ones((2, 2, 1)))
    results = _race(fake, tables.to_device)
    assert fake.uploads == 6, "each table uploaded once, not once per slab"
    assert all(result is results[0] for result in results)
    assert tables._device == {0: results[0]}


def test_thompson_tables_load_once_and_wait_on_the_uploading_stream(monkeypatch):
    """Thompson's device tables: one load per card, waited on its own stream.

    The upload used to be followed by a synchronize of the legacy NULL
    stream, which waits for nothing a slab's non-blocking stream queued, and
    the cache was filled with no lock, so concurrent slabs each loaded and
    replaced it.
    """
    import threading
    from gpuwm.core import thompson_aerosol_runtime as aerosol
    from gpuwm.core import thompson_runtime as classic
    fake = _FakeCupy(delay=0.0)
    fake.on_stream(7)
    classic._synchronize(fake)
    aerosol._synchronize(fake)
    assert fake.synchronized == [7, 7], "the current stream, not the NULL stream"
    loads = []
    guard = threading.Lock()
    def slow_upload(table_set, backend, **kwargs):
        import time
        with guard:
            loads.append(1)
        time.sleep(0.02)
        return SimpleNamespace(roundtrip_verified=True)
    monkeypatch.setattr(classic, "load_validated_classic_tables",
                        lambda root, version="wrf_461": None)
    monkeypatch.setattr(classic, "_upload_table_set", slow_upload)
    monkeypatch.setattr(classic, "_DEVICE_CACHE", {})
    results = _race(fake, lambda: classic.load_classic_device_tables(
        "tables", backend=fake))
    assert len(loads) == 1
    assert all(result is results[0] for result in results)


def test_legacy_lw_and_ruc_constants_upload_once_per_card(monkeypatch):
    """The legacy RRTMG LW constants and RUC's constant arrays, the same way."""
    from collections import defaultdict
    from gpuwm.core import rrtmg_lw, ruc_gpu
    coefficients = defaultdict(lambda: np.zeros(4, dtype=np.float32))
    # The per-band pointer tables import CuPy themselves; stand them in.
    monkeypatch.setattr(rrtmg_lw, "gpu_band_tabs", lambda band, C: (
        [], SimpleNamespace(data=SimpleNamespace(ptr=band))))
    one = _FakeCupy(delay=0.0)
    monkeypatch.setattr(rrtmg_lw, "_LW_CONST_CACHE", {})
    rrtmg_lw._lw_dev_consts(one, coefficients)
    fake = _FakeCupy(delay=0.001)
    monkeypatch.setattr(rrtmg_lw, "_LW_CONST_CACHE", {})
    results = _race(fake, lambda: rrtmg_lw._lw_dev_consts(fake, coefficients))
    assert fake.uploads == one.uploads > 0, "one set of constants, not one per slab"
    assert all(result is results[0] for result in results)
    assert len(rrtmg_lw._LW_CONST_CACHE) == 1
    fake = _FakeCupy()
    fake.ndarray = type(None)
    monkeypatch.setattr(ruc_gpu, "cp", fake)
    monkeypatch.setattr(ruc_gpu, "_RUC_CONSTANT_CACHE", {})
    results = _race(fake, lambda: ruc_gpu._constant_array([1.0, 2.0], dtype=np.float32))
    assert fake.uploads == 1
    assert all(result is results[0] for result in results)
