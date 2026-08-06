"""CPU contracts for Phase-5 child initialization ordering and N1 tooling."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import threading

import numpy as np
import pytest

import gpuwm.ingest.nest_init as ni
from gpuwm.case_data import SourceOrography
from gpuwm.core.grid import BaseState
from gpuwm.experiment import VerticalConfig
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.hrrr import HrrrNativeSnapshot
from gpuwm.static.lambert import LambertGrid
from gpuwm.verify.cases.wk82 import wk82_sounding, wk82_theta
from gpuwm.verify.npref import np_sint


def test_pending_child_inputs_overlap_and_collect_in_parent_order(monkeypatch):
    domains = tuple(SimpleNamespace(grid_id=grid_id) for grid_id in (2, 3, 4))
    grids = {domain.grid_id: object() for domain in domains}
    barrier = threading.Barrier(len(domains))
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def prepare(domain, grid, catalog, source_orography,
                preprocess_backend, preprocess_workers, cpu_bridge):
        nonlocal active, peak_active
        assert grid is grids[domain.grid_id]
        assert catalog == "catalog"
        assert source_orography == "orography"
        assert preprocess_backend == "cpu"
        assert preprocess_workers == 1
        assert cpu_bridge is None
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        barrier.wait(timeout=5.0)
        with lock:
            active -= 1
        # Reverse completion order without relying on process scheduling.
        threading.Event().wait((4 - domain.grid_id) * 0.001)
        return SimpleNamespace(domain=domain)

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", prepare)
    pending = ni.PendingChildInputs(
        domains, grids, "catalog", "orography", workers=3)
    result = pending.result()

    assert peak_active == 3
    assert [item.domain.grid_id for item in result] == [2, 3, 4]
    assert pending._closed
    assert pending.worker_budget == 3
    assert pending.allocated_workers == 3


def test_nested_input_catalog_binds_ordered_source_to_static_catalog():
    snapshots = tuple(HrrrNativeSnapshot(
        valid_time=datetime(2026, 7, 20, hour), forecast_hour=hour,
        i_start=0, j_start=0, ny=1, nx=1, fields={})
        for hour in (0, 1))
    static_catalog = object()
    catalog = ni.NestedInputCatalog(
        snapshots=snapshots, static_catalog=static_catalog,
        inventory=("LANDSEA",))

    assert catalog.valid_times == tuple(
        datetime(2026, 7, 20, hour) for hour in (0, 1))
    assert catalog.static_catalog is static_catalog
    with pytest.raises(ValueError, match="unique increasing"):
        ni.NestedInputCatalog(
            snapshots=(snapshots[0], snapshots[0]),
            static_catalog=static_catalog)


def test_nested_input_catalog_preserves_era5_soilgeo_units_through_mapping(
        monkeypatch):
    valid_time = datetime(1974, 4, 3, 12)
    source = Era5Snapshot(
        valid_time=valid_time, levels_hpa=np.asarray([1000.0, 500.0]),
        latitude=np.asarray([30.0, 31.0]),
        longitude=np.asarray([-100.0, -99.0]),
        fields={"SOILGEO": np.full((2, 2), 981.0)})
    raw = SimpleNamespace(
        snapshots=(source,), inventory=("SOILGEO",), files=(),
        units={"SOILGEO": "m2 s-2"},
        provenance={"product": "ERA5"})
    static_catalog = object()
    catalog = ni.NestedInputCatalog.from_source_catalog(raw, static_catalog)
    child = SimpleNamespace(
        grid_id=2, start_time=valid_time,
        run=SimpleNamespace(moist=True, terrain_opt=1))
    grid = object()
    static = {
        "LANDMASK": np.ones((2, 2)),
        "LU_INDEX": np.ones((2, 2)),
    }
    backend = SimpleNamespace(receipt=lambda: {"backend": "cpu"})
    monkeypatch.setattr(
        "gpuwm.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(
        ni, "build_static_for_domain",
        lambda actual_grid, actual_catalog, grid_id: static
        if (actual_grid is grid and actual_catalog is static_catalog
            and grid_id == 2)
        else pytest.fail("wrong static binding"))
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda *_args: SimpleNamespace(
            landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}))
    monkeypatch.setattr(
        ni, "interpolate_lake_skin_temperature",
        lambda *_args: np.full((2, 2), np.nan))
    horizontal = SimpleNamespace(fields={})

    def map_era5(snapshot, actual_grid, **options):
        assert snapshot is source
        assert actual_grid is grid
        bound = options["source_orography_catalog"]
        assert bound is catalog
        assert bound.units["SOILGEO"] == "m2 s-2"
        assert bound.provenance["product"] == "ERA5"
        return horizontal

    monkeypatch.setattr(ni, "interpolate_era5_to_lambert", map_era5)
    prepared = ni._prepare_child_input_on_grid(
        child, grid, catalog, preprocess_backend="cpu")

    assert prepared.horizontal is horizontal
    assert prepared.preprocess_receipt["source_adapter"] == \
        "regular-grid-native-state-v1"


def test_pending_child_inputs_names_failed_domain_and_closes(monkeypatch):
    domain = SimpleNamespace(grid_id=3)

    def fail(*_args):
        raise ValueError("bad source record")

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", fail)
    pending = ni.PendingChildInputs(
        (domain,), {3: object()}, object(), None, workers=1)
    with pytest.raises(RuntimeError, match=r"d03") as caught:
        pending.result()
    assert isinstance(caught.value.__cause__, ValueError)
    assert pending._closed


def test_pending_child_inputs_bounds_nested_cpu_threads_to_budget(monkeypatch):
    domains = tuple(SimpleNamespace(grid_id=grid_id) for grid_id in (2, 3, 4))
    observed = []

    def prepare(domain, _grid, _catalog, _orography, backend, workers, _bridge):
        observed.append((domain.grid_id, backend, workers))
        return SimpleNamespace(domain=domain)

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", prepare)
    pending = ni.PendingChildInputs(
        domains, {domain.grid_id: object() for domain in domains},
        object(), None, workers=8)
    pending.result()

    assert pending.workers == 3
    assert dict(pending.preprocess_workers_by_domain) == {2: 3, 3: 3, 4: 2}
    assert pending.allocated_workers == 8
    assert sorted(observed) == [(2, "cpu", 3), (3, "cpu", 3), (4, "cpu", 2)]


def test_pending_child_inputs_limits_submitted_residency_window(monkeypatch):
    domains = tuple(SimpleNamespace(grid_id=grid_id) for grid_id in range(2, 7))
    started = []
    two_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    def prepare(domain, *_args):
        with lock:
            started.append(domain.grid_id)
            if len(started) == 2:
                two_started.set()
        assert release.wait(timeout=5.0)
        return SimpleNamespace(domain=domain)

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", prepare)
    pending = ni.PendingChildInputs(
        domains, {domain.grid_id: object() for domain in domains},
        object(), None, workers=2)
    assert two_started.wait(timeout=5.0)
    with lock:
        assert sorted(started) == [2, 3]
    release.set()
    result = pending.result()

    assert [item.domain.grid_id for item in result] == list(range(2, 7))
    assert pending._next_submit_index == len(domains)


def test_six_domain_branching_hierarchy_is_deterministic_and_bounded(
        monkeypatch):
    parents = {2: 1, 3: 2, 4: 3, 5: 1, 6: 2}
    domains = tuple(
        SimpleNamespace(grid_id=grid_id, parent_id=parents[grid_id])
        for grid_id in range(2, 7))
    grids = {domain.grid_id: object() for domain in domains}
    calls = []
    lock = threading.Lock()

    def prepare(domain, grid, *_args):
        assert grid is grids[domain.grid_id]
        with lock:
            calls.append(domain.grid_id)
        return _prepared_for_barrier(domain.grid_id, domain.parent_id)

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", prepare)
    observed_orders = []
    for worker_budget in (1, 3, 8):
        calls.clear()
        pending = ni.PendingChildInputs(
            domains, grids, object(), None, workers=worker_budget,
            preprocess_backend="cpu")
        assert len(pending._futures) <= min(worker_budget, len(domains))
        assert pending.allocated_workers <= worker_budget
        result = pending.result()
        observed_orders.append(tuple(item.domain.grid_id for item in result))
        assert sorted(calls) == list(range(2, 7))
        assert len(calls) == len(set(calls))
        assert not pending._futures
        assert pending._closed

    assert observed_orders == [(2, 3, 4, 5, 6)] * 3

    parent_calls = []

    def finalize(prepared, parent, _vertical, **_options):
        parent_calls.append((prepared.domain.grid_id, parent.cfg.grid_id))
        return SimpleNamespace(grid=prepared.grid, state=object())

    monkeypatch.setattr(ni, "finalize_prepared_child", finalize)
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    ni.finalize_prepared_child_chain(result, root, object())
    assert parent_calls == [
        (2, 1), (3, 2), (4, 3), (5, 1), (6, 2)]


def test_pending_child_inputs_empty_hierarchy_allocates_no_workers():
    pending = ni.PendingChildInputs((), {}, object(), None, workers=8)
    assert pending.result() == ()
    assert pending.workers == 0
    assert dict(pending.preprocess_workers_by_domain) == {}
    assert pending.allocated_workers == 0
    assert pending._closed


def test_pending_child_inputs_validates_domain_grid_inventory_before_launch():
    duplicate = SimpleNamespace(grid_id=2)
    with pytest.raises(ValueError, match="duplicate"):
        ni.PendingChildInputs(
            (duplicate, duplicate), {2: object()}, object(), None, workers=2)
    with pytest.raises(ValueError, match=r"d03"):
        ni.PendingChildInputs(
            (SimpleNamespace(grid_id=3),), {}, object(), None, workers=1)


def _prepared_for_barrier(grid_id, parent_id):
    return ni.PreparedChildInput(
        domain=SimpleNamespace(grid_id=grid_id, parent_id=parent_id),
        grid=object(), static_fields={}, horizontal=object(),
        declared_orography=None, lake_mask=np.zeros((1, 1), dtype=bool),
        lake_skin_temperature=np.full((1, 1), np.nan),
        preprocess_backend=object(), preprocess_receipt={"backend": "test"},
        preparation_seconds=0.0)


def test_finalize_prepared_child_chain_is_parent_before_child(monkeypatch):
    calls = []

    def finalize(prepared, parent, vertical, **options):
        calls.append((prepared.domain.grid_id, parent.cfg.grid_id, options))
        return SimpleNamespace(grid=prepared.grid, state=object())

    monkeypatch.setattr(ni, "finalize_prepared_child", finalize)
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    prepared = tuple(
        _prepared_for_barrier(grid_id, grid_id - 1)
        for grid_id in (2, 3, 4))
    results = ni.finalize_prepared_child_chain(
        prepared, root, object(), scratch_arena="arena",
        dycore_state_workspace="workspace", sfcp_to_sfcp=False,
        soil_layer_contract="soil-contract")

    assert len(results) == 3
    assert [(child, parent) for child, parent, _ in calls] == [
        (2, 1), (3, 2), (4, 3)]
    assert all(options == {
        "scratch_arena": "arena",
        "dycore_state_workspace": "workspace",
        "state_backend": "cuda",
        "sfcp_to_sfcp": False,
        "soil_layer_contract": "soil-contract",
    } for _child, _parent, options in calls)


def test_finalize_prepared_child_chain_rejects_missing_or_duplicate_parent(
        monkeypatch):
    monkeypatch.setattr(
        ni, "finalize_prepared_child",
        lambda prepared, *_args, **_kwargs: SimpleNamespace(
            grid=prepared.grid, state=object()))
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    with pytest.raises(ValueError, match=r"d03.*before parent d02"):
        ni.finalize_prepared_child_chain(
            (_prepared_for_barrier(3, 2),), root, object())
    with pytest.raises(ValueError, match=r"repeats d01"):
        ni.finalize_prepared_child_chain(
            (_prepared_for_barrier(1, 1),), root, object())


def test_initialize_child_chain_parallel_composes_prepare_and_barrier(
        monkeypatch):
    prepared = tuple(
        _prepared_for_barrier(grid_id, grid_id - 1)
        for grid_id in (2, 3, 4))
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1), *(
            item.domain for item in prepared)),
        vertical=object())
    events = []

    class Pending:
        def __enter__(self):
            events.append("enter")
            return self

        def iter_parent_order(self):
            events.append("iter")
            yield from prepared

        def __exit__(self, *_args):
            events.append("exit")

    def start(actual_exp, catalog, source_orography, **options):
        assert actual_exp is exp
        assert catalog == "catalog"
        assert source_orography == "orography"
        assert options == {
            "workers": 8, "preprocess_backend": "cpu",
            "cpu_bridge": "bridge"}
        events.append("start")
        return Pending()

    expected = (object(), object(), object())

    def finalize(actual, actual_root, vertical, **options):
        events.append("finalize")
        assert tuple(actual) == prepared
        assert actual_root is root
        assert vertical is exp.vertical
        assert options == {
            "scratch_arena": "arena",
            "dycore_state_workspace": "workspace",
            "state_backend": "cuda",
            "sfcp_to_sfcp": False,
            "soil_layer_contract": "soil-contract"}
        return expected

    monkeypatch.setattr(ni, "start_child_input_preparations", start)
    monkeypatch.setattr(ni, "finalize_prepared_child_chain", finalize)
    result = ni.initialize_child_chain_parallel(
        exp, root, "catalog", "orography", workers=8,
        preprocess_backend="cpu", cpu_bridge="bridge",
        scratch_arena="arena", dycore_state_workspace="workspace",
        sfcp_to_sfcp=False, soil_layer_contract="soil-contract")

    assert result is expected
    assert events == ["start", "enter", "finalize", "iter", "exit"]


def test_initialize_child_chain_parallel_rejects_wrong_root_before_launch(
        monkeypatch):
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1),), vertical=object())
    root = SimpleNamespace(cfg=SimpleNamespace(grid_id=2))
    monkeypatch.setattr(
        ni, "start_child_input_preparations",
        lambda *_args, **_kwargs: pytest.fail("preparation must not launch"))
    with pytest.raises(ValueError, match="root node"):
        ni.initialize_child_chain_parallel(exp, root, object())


def test_prepare_child_input_dispatches_hrrr_on_own_static_landmask(
        monkeypatch):
    valid_time = datetime(2026, 7, 20)
    source = HrrrNativeSnapshot(
        valid_time=valid_time, forecast_hour=0,
        i_start=0, j_start=0, ny=4, nx=5, fields={})
    static_catalog = object()
    catalog = SimpleNamespace(
        snapshots=(source,), valid_times=(valid_time,), inventory=(), files=(),
        static_catalog=static_catalog)
    child = SimpleNamespace(
        grid_id=2, start_time=valid_time,
        run=SimpleNamespace(moist=True, terrain_opt=1))
    grid = object()
    landmask = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    static = {
        "LANDMASK": landmask,
        "LU_INDEX": np.asarray([[21, 1], [16, 21]]),
    }
    horizontal = SimpleNamespace(fields={
        "SKINTEMP": np.asarray([[281.0, 282.0], [283.0, 284.0]])})

    class Backend:
        @staticmethod
        def receipt():
            return {"backend": "cpu", "workers": 2}

    backend = Backend()
    monkeypatch.setattr(
        "gpuwm.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: backend)
    def build_static(actual_grid, actual_catalog, grid_id):
        assert actual_grid is grid
        assert actual_catalog is static_catalog
        assert grid_id == 2
        return static

    monkeypatch.setattr(ni, "build_static_for_domain", build_static)
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda actual_catalog, grid_id: (
            SimpleNamespace(landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13})
            if actual_catalog is static_catalog and grid_id == 2
            else pytest.fail("wrong static catalog binding")))
    observed = {}

    def map_hrrr(snapshot, actual_grid, **options):
        assert snapshot is source
        assert actual_grid is grid
        np.testing.assert_array_equal(options["target_landmask"], landmask)
        assert options["backend"] is backend
        options["soil_mapping_report"]["status"] = "PASS"
        observed["mapped"] = True
        return horizontal

    monkeypatch.setattr(ni, "interpolate_hrrr_to_lambert", map_hrrr)
    monkeypatch.setattr(
        ni, "interpolate_era5_to_lambert",
        lambda *_args, **_kwargs: pytest.fail("ERA5 mapper must not run"))
    monkeypatch.setattr(
        ni, "interpolate_lake_skin_temperature",
        lambda *_args, **_kwargs: pytest.fail("regular lake mapper must not run"))

    prepared = ni._prepare_child_input_on_grid(
        child, grid, catalog, preprocess_backend="cpu",
        preprocess_workers=2)

    assert observed == {"mapped": True}
    np.testing.assert_array_equal(
        prepared.lake_skin_temperature,
        np.asarray([[281.0, np.nan], [np.nan, 284.0]]))
    assert prepared.declared_orography is None
    assert prepared.preprocess_receipt["source_adapter"] == \
        "hrrr-native-state-v1"
    assert prepared.preprocess_receipt["soil_mapping"] == {"status": "PASS"}


@pytest.mark.parametrize("backend", ("cuda", "auto", object()))
def test_pending_child_inputs_rejects_unsafe_parallel_backend(backend):
    domains = tuple(SimpleNamespace(grid_id=grid_id) for grid_id in (2, 3))
    with pytest.raises(ValueError, match="explicit CPU"):
        ni.PendingChildInputs(
            domains, {domain.grid_id: object() for domain in domains},
            object(), None, workers=2, preprocess_backend=backend)


@pytest.mark.parametrize("workers", (0, 33, True, 1.5))
def test_pending_child_inputs_rejects_unbounded_or_invalid_workers(workers):
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        ni.PendingChildInputs((), {}, object(), None, workers=workers)


@pytest.mark.parametrize("child_id", (2, 3, 4))
def test_initialize_child_binding_order_and_soil_never_readjusted(
        monkeypatch, child_id):
    """Review anchor: REAL -> SINT -> blend all 3 -> adjust -> rederive."""
    events = []

    class State:
        def __init__(self):
            self.ht = np.full((4, 5), 10.0, np.float32)
            self.mub2d = np.full((4, 5), 90000.0, np.float32)
            self.phb = np.full((3, 4, 5), 100.0, np.float32)

    state = State()
    real = SimpleNamespace(state=state)
    soil = object()
    static = {
        "HGT_M": np.full((4, 5), 10.0),
        "SCT_DOM": np.ones((4, 5)),
        "TMN": np.full((4, 5), 280.0),
        "LU_INDEX": np.vstack((np.array([[21.0, 1.0, 1.0, 1.0, 1.0]]),
                                np.ones((3, 5)))),
    }
    horizontal = SimpleNamespace(fields={})
    cfg = SimpleNamespace(
        nx=5, ny=4, nz=2, dx=1.0, dy=1.0, moist=True,
        terrain_opt=1, spec_bdy_width=5, hypsometric_opt=2,
        # The land-surface selector the soil seam routes on.  A cfg without
        # it no longer reaches an ingest at all, which is the point of the
        # seam: the geometry a run initialises on is a decision, not a
        # default.
        sf_surface_physics=2)
    # A parsed [[domain]] always carries a start_time -- it defaults to
    # [experiment].start_time rather than to None -- and the snapshot the
    # child initialises on is now chosen by it, so the stub carries one too.
    child_start = datetime(2026, 7, 20, 6)
    child_dc = SimpleNamespace(
        grid_id=child_id, parent_id=child_id - 1, run=cfg, i_parent_start=3,
        j_parent_start=3, parent_grid_ratio=3, start_time=child_start)
    parent_node = SimpleNamespace(cfg=SimpleNamespace(grid_id=child_id - 1))
    catalog = SimpleNamespace(inventory=("SOILGEO",), valid_times=(object(),),
                              snapshots=(object(),), files=())
    vertical = VerticalConfig(eta_levels=(1.0, 0.5, 0.0), p_top=10000.0,
                              hybrid_opt=2, etac=0.2)

    child_grid = object()
    monkeypatch.setattr(ni, "_child_grid", lambda *_: child_grid)
    monkeypatch.setattr(
        ni, "build_static_for_domain", lambda *_: static)
    def initial_snapshot(_catalog, valid_time):
        # The delayed-child work made this lookup start_time-driven; a stub
        # that swallowed the argument would let a regression to "first valid
        # time" pass unnoticed.
        assert valid_time == child_start
        events.append("snapshot")
        return object()

    monkeypatch.setattr(ni, "_initial_snapshot", initial_snapshot)
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda *_: SimpleNamespace(
            landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}))
    def lake_skin(_source, _grid, lake_mask):  # pragma: no cover
        raise AssertionError(
            "regular-grid lane must not run the lake skin override: "
            "metgrid's masked=both SKINTEMP chain already yields "
            "water-source skin at lakes")

    monkeypatch.setattr(ni, "interpolate_lake_skin_temperature", lake_skin)
    def interpolate(*_args, **kwargs):
        assert kwargs["source_orography_catalog"] is catalog
        events.append("era5-own-grid")
        return horizontal

    monkeypatch.setattr(ni, "interpolate_era5_to_lambert", interpolate)
    shared_coord = object()
    monkeypatch.setattr(
        ni, "_shared_vertical_coord", lambda *_: shared_coord)
    def initialize_real(*_args, **kwargs):
        assert kwargs["source_orography"] is None
        events.append("real-unblended")
        return real

    monkeypatch.setattr(ni, "initialize_real", initialize_real)
    monkeypatch.setattr(
        ni, "_set_map_fields", lambda *_: events.append("map"))
    monkeypatch.setattr(
        ni, "update_diagnostics", lambda *_: events.append("preblend-eos"))
    def preprocess(*_args, **kwargs):
        assert kwargs["lake_mask"] is None
        assert kwargs["lake_skin_temperature"] is None
        assert kwargs["soil_layer_contract"] == "soil-contract"
        events.append("soil-fine")
        return soil

    monkeypatch.setattr(ni, "preprocess_land_surface_soil", preprocess)
    captures = (np.zeros_like(state.ht), np.zeros_like(state.mub2d),
                np.zeros_like(state.phb))
    monkeypatch.setattr(
        ni, "_capture_parent_blend_fields",
        lambda *_: events.append("sint-ht-mub-phb") or captures)

    targets = {id(state.ht): "ht", id(state.mub2d): "mub",
               id(state.phb): "phb"}

    def blend(_parent, target, **_kwargs):
        events.append("blend-" + targets[id(target)])

    monkeypatch.setattr(ni, "blend_terrain", blend)
    base = object()
    def adjust_then_rederive(_state, _cfg, _coord, _save_mub, ht_fine):
        assert ht_fine is static["HGT_M"]
        events.append("adjust-then-rederive-eos-press-adj")
        return base

    monkeypatch.setattr(ni, "_adjust_and_rederive", adjust_then_rederive)
    updated = object()
    monkeypatch.setattr(ni, "_updated_real_result", lambda *_: updated)

    result = ni.initialize_child(
        child_dc, parent_node, catalog, vertical,
        soil_layer_contract="soil-contract")

    assert events == [
        "snapshot", "era5-own-grid", "real-unblended", "map",
        "preblend-eos", "soil-fine", "sint-ht-mub-phb", "blend-ht",
        "blend-mub", "blend-phb", "adjust-then-rederive-eos-press-adj",
    ]
    assert result.soil is soil
    assert result.static_fields is static
    assert result.real is updated
    assert result.coord is shared_coord


def test_as_like_moves_cuda_style_parent_operand_to_cpu_child():
    class DeviceOperand:
        def __init__(self, value):
            self.value = value
            self.readbacks = 0

        def get(self):
            self.readbacks += 1
            return self.value

    source = DeviceOperand(
        np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1])
    template = np.empty((3, 4), dtype=np.float32)
    result = ni._as_like(source, template)
    assert source.readbacks == 1
    assert isinstance(result, np.ndarray)
    assert result.dtype == template.dtype
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, source.value)


def _write_source_orography(path: Path, domain_id: int) -> np.ndarray:
    import netCDF4

    expected = np.full((domain_id + 1, domain_id + 2), domain_id, np.float32)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", expected.shape[0])
        dataset.createDimension("west_east", expected.shape[1])
        dataset.createVariable(
            "SOILHGT", "f4", ("Time", "south_north", "west_east"))[0] = expected
    return expected.astype(np.float64)


def test_domain_tagged_catalog_resolves_each_child_orography_artifact(tmp_path):
    expected = {}
    files = []
    for domain_id in (2, 3, 4):
        path = tmp_path / f"met_em.d{domain_id:02d}.nc"
        expected[domain_id] = _write_source_orography(path, domain_id)
        files.append(SimpleNamespace(
            role="source_orography", path=path,
            provenance=f"variable=SOILHGT;domain=d{domain_id:02d}"))
    catalog = SimpleNamespace(files=tuple(files))

    for domain_id in (2, 3, 4):
        np.testing.assert_array_equal(
            ni._declared_source_orography(catalog, domain_id),
            expected[domain_id])


def test_d01_only_declaration_fails_at_d02_initialization_with_domain_named(
        tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        nx=5, ny=4, nz=2, moist=True, terrain_opt=1,
        hypsometric_opt=2)
    child_dc = SimpleNamespace(grid_id=2, parent_id=1, run=cfg,
                               start_time=datetime(2026, 7, 20, 6))
    parent_node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1))
    catalog = SimpleNamespace(
        inventory=(), files=(), valid_times=(object(),), snapshots=(object(),))
    vertical = VerticalConfig(eta_levels=(1.0, 0.5, 0.0), p_top=10000.0,
                              hybrid_opt=2, etac=0.2)
    declaration = SourceOrography(tmp_path / "d01-only.nc", "SOILHGT")
    monkeypatch.setattr(ni, "_child_grid", lambda *_: object())
    monkeypatch.setattr(ni, "build_static_for_domain", lambda *_: {})
    monkeypatch.setattr(ni, "_initial_snapshot", lambda *_: object())

    with pytest.raises(ValueError, match=r"d02"):
        ni.initialize_child(
            child_dc, parent_node, catalog, vertical, declaration)


def test_press_adj_mu_matches_wrf_expression_after_eos(monkeypatch):
    """start_em.F:878-892: diagnose once, correct MU_2, seed MU_1."""
    events = []

    class State:
        def __init__(self):
            self.mub2d = np.full((1, 2), 90000.0, np.float32)
            self.c3h = np.array([0.75, 0.25], np.float32)
            self.c4h = np.zeros(2, np.float32)
            self.p_top = np.float32(10000.0)
            self.pb = np.ones((2, 1, 2), np.float32)
            self.p = self.pb.copy()
            self.qv = np.zeros_like(self.pb)
            self.thb = np.zeros_like(self.pb)
            self.thp = np.zeros_like(self.pb)
            self.ht = np.array([[120.0, 80.0]], np.float32)
            self.mup = np.array([[10.0, -4.0]], np.float32)
            self.mup0 = np.full((1, 2), -999.0, np.float32)
            self.al = np.zeros_like(self.pb)
            self.alt = np.zeros_like(self.pb)
            self.alb = np.zeros_like(self.pb)

        def total_theta(self):
            return np.full_like(self.thp, 300.0)

        def load_base(self, _coord, base):
            events.append("load-base")
            self.thb[...] = base.thb

    state = State()
    base = SimpleNamespace(thb=np.zeros_like(state.thb))
    monkeypatch.setattr(
        ni, "adjust_tempqv",
        lambda *_args, **_kwargs: events.append("adjust-tempqv"))
    monkeypatch.setattr(
        ni, "_base_from_blended",
        lambda *_args, **_kwargs: events.append("rederive-base") or base)

    def diagnose(_state, _hypsometric_opt):
        events.append("eos")
        _state.al[0, 0] = np.array([0.02, 0.03], np.float32)
        _state.alt[0, 0] = np.array([0.80, 0.81], np.float32)
        _state.alb[0, 0] = np.array([0.78, 0.78], np.float32)

    monkeypatch.setattr(ni, "update_diagnostics", diagnose)
    ht_fine = np.array([[100.0, 120.0]], np.float32)

    ni._adjust_and_rederive(
        state, SimpleNamespace(hypsometric_opt=2), object(),
        state.mub2d.copy(), ht_fine)

    expected = np.array([
        10.0 + 0.02 / (0.80 * 0.78) * 9.81 * (120.0 - 100.0),
        -4.0 + 0.03 / (0.81 * 0.78) * 9.81 * (80.0 - 120.0),
    ], np.float32)
    np.testing.assert_allclose(state.mup[0], expected, rtol=2e-6, atol=0.0)
    np.testing.assert_array_equal(state.mup0, state.mup)
    assert events == ["adjust-tempqv", "rederive-base", "load-base", "eos"]


def _flat_parent_state(nz=3, ny=12, nx=12):
    znw = np.linspace(1.0, 0.0, nz + 1)
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = np.diff(znw)
    dn = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    rdn = np.zeros(nz)
    rdn[1:] = 1.0 / dn[1:]
    fnp = np.zeros(nz)
    fnm = np.zeros(nz)
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    p_top = 10000.0
    mub = 90000.0
    pb = znu * mub + p_top
    height = np.array([500.0, 3500.0, 8500.0])
    thb = wk82_theta(height)
    alb = 287.0 * thb * (pb / 100000.0) ** (287.0 / 1004.0) / pb
    phb = np.array([0.0, 30000.0, 75000.0, 140000.0])
    qv = wk82_sounding(height)[1][:, None, None] * np.ones((nz, ny, nx))
    x = np.arange(nx, dtype=np.float32)[None, None, :]
    y = np.arange(ny, dtype=np.float32)[None, :, None]
    x_u = np.arange(nx + 1, dtype=np.float32)[None, None, :]
    y_v = np.arange(ny + 1, dtype=np.float32)[None, :, None]
    thp = np.broadcast_to(0.01 * x + 0.02 * y, (nz, ny, nx)).copy()
    return SimpleNamespace(
        znw=znw, znu=znu, dnw=dnw, rdnw=1.0 / dnw,
        dn=dn, rdn=rdn, fnp=fnp, fnm=fnm,
        c1f=np.ones(nz + 1), c2f=np.zeros(nz + 1), c3f=znw,
        c4f=np.zeros(nz + 1), c1h=np.ones(nz), c2h=np.zeros(nz),
        c3h=znu, c4h=np.zeros(nz),
        mub=mub, mub2d=np.full((ny, nx), mub, np.float32),
        p_top=p_top, pb=pb, alb=alb, thb=thb, phb=phb,
        ht=np.zeros((ny, nx), np.float32),
        u=np.broadcast_to(x_u, (nz, ny, nx + 1)).copy(),
        v=np.broadcast_to(y_v, (nz, ny + 1, nx)).copy(),
        w=np.zeros((nz + 1, ny, nx), np.float32),
        thp=thp.astype(np.float32),
        php=np.zeros((nz + 1, ny, nx), np.float32),
        mup=np.zeros((ny, nx), np.float32),
        qv=qv.astype(np.float32), qc=np.zeros_like(qv, np.float32),
        qr=np.zeros_like(qv, np.float32), h_diabatic=np.zeros_like(qv,
                                                                   np.float32))


class _CpuState:
    """Minimal NumPy DomainState twin for the WK82 parent-only fixture."""

    def __init__(self, cfg):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        self.u = np.zeros((nz, ny, nx + 1), np.float32)
        self.v = np.zeros((nz, ny + 1, nx), np.float32)
        self.w = np.zeros((nz + 1, ny, nx), np.float32)
        self.thp = np.zeros((nz, ny, nx), np.float32)
        self.php = np.zeros((nz + 1, ny, nx), np.float32)
        self.mup = np.zeros((ny, nx), np.float32)
        self.qv = np.zeros((nz, ny, nx), np.float32)
        self.qc = np.zeros_like(self.qv)
        self.qr = np.zeros_like(self.qv)
        self.h_diabatic = np.zeros_like(self.qv)
        for name, source in (("u0", self.u), ("v0", self.v),
                             ("w0", self.w), ("thp0", self.thp),
                             ("php0", self.php), ("mup0", self.mup),
                             ("qv0", self.qv), ("qc0", self.qc),
                             ("qr0", self.qr)):
            setattr(self, name, np.zeros_like(source))

    def load_base(self, coord, base: BaseState):
        for name in ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp",
                     "fnm", "c1f", "c2f", "c3f", "c4f", "c1h", "c2h",
                     "c3h", "c4h"):
            setattr(self, name, np.array(getattr(coord, name), copy=True))
        for name in ("pb", "alb", "thb", "phb"):
            setattr(self, name, np.array(getattr(base, name), copy=True))
        self.mub = float(base.mub)
        self.mub2d = np.full(self.mup.shape, self.mub, np.float32)
        self.ht = np.zeros(self.mup.shape, np.float32)
        self.p_top = float(base.p_top)

    def set_map_coriolis(self, *_args, **_kwargs):
        return None


def test_parent_only_wk82_child_fixture_is_cpu_testable(monkeypatch):
    """input_from_file=F keeps the full-parent SINT fill for WK82 nests."""
    parent_state = _flat_parent_state()
    parent_run = SimpleNamespace(nx=12, ny=12, nz=3)
    parent_grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    parent_node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=parent_run),
        state=parent_state, grid=parent_grid)
    child_run = SimpleNamespace(
        nx=4, ny=4, nz=3, dx=1000.0, dy=1000.0,
        terrain_opt=0, hypsometric_opt=1)
    child_dc = SimpleNamespace(
        grid_id=2, parent_id=1, i_parent_start=4, j_parent_start=4,
        parent_grid_ratio=1, run=child_run)

    monkeypatch.setattr(ni, "DomainState", _CpuState)
    monkeypatch.setattr(
        ni, "sint",
        lambda field, reg: np_sint(field, reg, dtype=np.float64).astype(
            np.float32))
    diagnosed = []
    monkeypatch.setattr(
        ni, "update_diagnostics", lambda *_: diagnosed.append(True))

    result = ni.parent_only_init(child_dc, parent_node)
    reg = ni._mass_registration(child_dc, parent_node)
    expected_qv = np_sint(parent_state.qv, reg, dtype=np.float64).astype(
        np.float32)
    np.testing.assert_array_equal(result.state.qv, expected_qv)
    np.testing.assert_array_equal(result.state.qv0, expected_qv)
    assert np.isfinite(result.state.thp).all()
    assert float(result.state.qv.max()) > 0.0  # the analytic WK82 moisture
    assert diagnosed == [True]
    assert result.real is result.soil is result.static_fields is None


def test_mixed_parent_only_init_reuses_flat_force_slot_and_resets_heating(
        monkeypatch):
    """MP8->MP18 cold-start translation cannot poison later FORCE shape."""
    parent_state = _flat_parent_state()
    parent_state.h_diabatic.fill(np.float32(77.0))
    for index, name in enumerate(("qi", "qs", "qg", "nr", "ni"), 1):
        setattr(parent_state, name, np.full(
            parent_state.qv.shape, np.float32(index), dtype=np.float32))
    parent_run = SimpleNamespace(
        nx=12, ny=12, nz=3, mp_physics=8, moist=True, moist_cq=True)
    parent_grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    parent_node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=parent_run),
        state=parent_state, grid=parent_grid)
    child_run = SimpleNamespace(
        nx=4, ny=4, nz=3, dx=1000.0, dy=1000.0,
        terrain_opt=0, hypsometric_opt=1, mp_physics=18,
        moist=True, moist_cq=True, spec_bdy_width=5,
        spec_zone=1, relax_zone=4,
        nest_microphysics_transition="mp8-to-mp18-mass-diagnosed-v1")
    child_dc = SimpleNamespace(
        grid_id=2, parent_id=1, i_parent_start=4, j_parent_start=4,
        parent_grid_ratio=1, run=child_run)

    class MixedCpuState(_CpuState):
        def __init__(self, cfg):
            super().__init__(cfg)
            for name in ("qi", "qs", "qg", "qh", "qndrop", "qnr",
                         "qni", "qns", "qng", "qnh", "qnn", "qvolg",
                         "qvolh"):
                value = np.zeros_like(self.qv)
                setattr(self, name, value)
                setattr(self, name + "0", np.zeros_like(value))
            self._scratch = {}

        def scratch(self, shape, slot, dtype=None):
            shape = tuple(shape)
            if slot not in self._scratch:
                self._scratch[slot] = np.zeros(shape, np.float32)
            value = self._scratch[slot]
            assert value.shape == shape
            return value

    monkeypatch.setattr(ni, "DomainState", MixedCpuState)

    def cpu_sint(field, reg, *, out=None):
        result = np_sint(field, reg, dtype=np.float64).astype(np.float32)
        if out is not None:
            out[...] = result
            return out
        return result

    field_value = {
        name: np.float32(index + 1)
        for index, name in enumerate((
            "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
            "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg",
            "qvolh"))
    }

    def translate(_contract, _state, name, *, out, coupled):
        assert coupled is False
        out.fill(field_value[name])
        return out

    monkeypatch.setattr(ni, "sint", cpu_sint)
    monkeypatch.setattr(
        ni, "launch_microphysics_edge_parent_field", translate)
    monkeypatch.setattr(ni, "update_diagnostics", lambda *_: None)

    result = ni.parent_only_init(child_dc, parent_node)
    state = result.state
    assert state._microphysics_transition_init_count == 1
    expected_capacity = max(
        3 * 12 * 12, 3 * 12 * 13, 3 * 13 * 12, 4 * 12 * 12)
    assert state._scratch["nest_parent_field"].shape == (expected_capacity,)
    np.testing.assert_array_equal(state.h_diabatic, 0.0)
    for name in field_value:
        target = getattr(state, name)
        seed = getattr(state, name + "0", None)
        assert np.all(target == field_value[name])
        if seed is not None:
            np.testing.assert_array_equal(seed, target)


def test_blend_zone_mask_matches_wrf_ten_row_frame():
    mask = ni.blend_zone_mask((24, 26), spec_bdy_width=5, blend_width=5)
    assert mask[0].all() and mask[:, 0].all()
    assert mask[9, 12] and mask[12, 9]
    assert not mask[10, 12] and not mask[12, 10]


def _write_reference(path: Path, hgt, mub=None):
    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = hgt.shape
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("west_east", nx)
        hgt_var = dataset.createVariable(
            "HGT" if mub is not None else "HGT_M", "f4",
            ("Time", "south_north", "west_east"))
        hgt_var[0] = hgt
        if mub is not None:
            dataset.createVariable(
                "MUB", "f4",
                ("Time", "south_north", "west_east"))[0] = mub


def test_n1_static_comparison_entry_point(tmp_path):
    bundle = tmp_path / "bundle"
    produced = {}
    hgt = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
    mub = np.full((24, 24), 90000.0, np.float32)
    for domain_id in (2, 3, 4):
        label = f"d{domain_id:02d}"
        _write_reference(
            bundle / "wrfout_reference" /
            f"wrfout_{label}_1974-04-03_13_15_00", hgt, mub)
        _write_reference(bundle / "geo_em" / f"geo_em.{label}.nc", hgt)
        path = tmp_path / f"{label}.npz"
        np.savez(path, HGT=hgt, MUB=mub, HGT_UNBLENDED=hgt)
        produced[domain_id] = path

    result = ni.compare_n1_static(produced, bundle)
    assert result.passed
    assert set(result.metrics) == {
        f"d{domain_id:02d}_{field}_blend_max_abs_diff_{unit}"
        for domain_id in (2, 3, 4)
        for field, unit in (("hgt", "m"), ("mub", "pa"))
    }
    assert all(value == 0.0 for value in result.metrics.values())
    assert all(value == 0.0
               for value in result.fine_hgt_max_abs_diff_m.values())


def _n1_bundle(root: Path, frames: dict[int, str]) -> tuple[Path, dict]:
    """Bundle + produced NPZs whose reference frames are named by caller."""
    bundle = root / "bundle"
    produced = {}
    hgt = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
    mub = np.full((24, 24), 90000.0, np.float32)
    for domain_id, frame in frames.items():
        label = f"d{domain_id:02d}"
        _write_reference(bundle / "wrfout_reference" / frame, hgt, mub)
        _write_reference(bundle / "geo_em" / f"geo_em.{label}.nc", hgt)
        path = root / f"{label}.npz"
        np.savez(path, HGT=hgt, MUB=mub, HGT_UNBLENDED=hgt)
        produced[domain_id] = path
    return bundle, produced


def test_n1_reference_frame_is_case_data_not_a_literal(tmp_path):
    """The comparator carries no campaign valid time.

    A bundle whose frames are stamped with a DIFFERENT valid time than the
    one this module used to hard-code compares fine, both by declaration
    and by discovery.  That is the negative control: with the literal in
    place, every assertion below raised FileNotFoundError."""
    frames = {domain_id: f"wrfout_d{domain_id:02d}_2031-07-19_04_30_00"
              for domain_id in (2, 3, 4)}
    bundle, produced = _n1_bundle(tmp_path / "declared", frames)
    declared = ni.compare_n1_static(
        produced, bundle,
        reference_frames={f"d{k:02d}": v for k, v in frames.items()})
    assert declared.passed

    # Same frames, keyed by domain id rather than dNN label.
    by_id = ni.compare_n1_static(produced, bundle, reference_frames=frames)
    assert by_id.metrics == declared.metrics

    # No declaration: the single frame per domain is discovered.
    discovered = ni.compare_n1_static(produced, bundle)
    assert discovered.metrics == declared.metrics


def test_n1_reference_frame_discovery_refuses_to_guess(tmp_path):
    """Ambiguity and absence are hard errors naming the candidates, never
    a silent pick of whichever frame sorts first."""
    frames = {domain_id: f"wrfout_d{domain_id:02d}_2031-07-19_04_30_00"
              for domain_id in (2, 3, 4)}
    bundle, produced = _n1_bundle(tmp_path / "ambiguous", frames)
    hgt = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
    _write_reference(bundle / "wrfout_reference" /
                     "wrfout_d02_2031-07-19_05_30_00", hgt,
                     np.full((24, 24), 90000.0, np.float32))
    with pytest.raises(ValueError, match="exactly one reference frame"):
        ni.compare_n1_static(produced, bundle)
    # Declaring the frame resolves the ambiguity without widening anything.
    assert ni.compare_n1_static(
        produced, bundle, reference_frames={"d02": frames[2]}).passed

    empty = tmp_path / "empty" / "bundle"
    (empty / "wrfout_reference").mkdir(parents=True)
    for domain_id in (2, 3, 4):
        _write_reference(empty / "geo_em" / f"geo_em.d{domain_id:02d}.nc",
                         hgt)
    with pytest.raises(ValueError, match="found 0"):
        ni.compare_n1_static(produced, empty)


def test_n1_cli_accepts_a_declared_reference_frame(tmp_path, capsys):
    import json as _json

    frames = {domain_id: f"wrfout_d{domain_id:02d}_2031-07-19_04_30_00"
              for domain_id in (2, 3, 4)}
    bundle, produced = _n1_bundle(tmp_path / "cli", frames)
    argv = ["--bundle", str(bundle)]
    for domain_id in (2, 3, 4):
        argv += [f"--d{domain_id:02d}", str(produced[domain_id])]
        argv += ["--reference-frame", f"d{domain_id:02d}={frames[domain_id]}"]
    assert ni.main(argv) == 0
    assert _json.loads(capsys.readouterr().out)["passed"] is True

    with pytest.raises(SystemExit):
        ni.main([*argv, "--reference-frame", "malformed"])
