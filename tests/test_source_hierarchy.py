"""Contracts for the source-neutral GFS/ERA5 hierarchy handoff."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.case_data import PerDomainSourceOrography, SourceOrography
import gpuwm.source_hierarchy as source_hierarchy
from gpuwm.preprocess_policy import HOST_TILED_CPU_REASON


@pytest.mark.parametrize("requested,backend,reason", [
    ("auto", "cpu", "no CUDA device is visible here"),
    ("cpu", "cpu", HOST_TILED_CPU_REASON),
    ("auto", "cuda", "the CUDA preparation backend is certified"),
])
def test_each_child_preserves_root_backend_selection(
        tmp_path, monkeypatch, requested, backend, reason):
    from gpuwm.ingest import nest_init, preprocess_backend

    root_selection = {
        "requested": requested, "backend": backend, "reason": reason}
    _, observed = _call(
        tmp_path, monkeypatch, preprocess_backend=backend, workers=1,
        input_provenance={"preprocessing": {
            "backend": backend, "selection": root_selection}})
    catalog = observed["initialize"]["catalog"]

    class Backend:
        name = backend

        def __init__(self, **_kwargs):
            self.selection = None

        def receipt(self):
            return {"backend": self.name, "selection": dict(self.selection)}

    monkeypatch.setattr(preprocess_backend, "ParallelCpuPreprocessBackend", Backend)
    monkeypatch.setattr(preprocess_backend, "CudaPreprocessBackend", Backend)
    monkeypatch.setattr(preprocess_backend, "_gpu_runtime_installed", lambda: True)
    monkeypatch.setattr(nest_init, "build_static_for_domain", lambda *_a: {
        "LU_INDEX": np.ones((2, 2))})
    monkeypatch.setattr(nest_init, "geog_selection_from_catalog", lambda *_a:
                        SimpleNamespace(landuse_global_attrs=lambda: {"ISLAKE": 21}))
    monkeypatch.setattr(nest_init, "interpolate_era5_to_lambert", lambda *_a, **_k:
                        SimpleNamespace(fields={}))
    monkeypatch.setattr(nest_init, "_child_soil_mesh", lambda *_a: None)
    for grid_id in (2, 3):
        child = SimpleNamespace(
            grid_id=grid_id, start_time=catalog.valid_times[0],
            run=SimpleNamespace(moist=True, terrain_opt=1))
        prepared = nest_init._prepare_child_input_on_grid(
            child, object(), catalog, preprocess_backend=backend)
        assert prepared.preprocess_receipt["selection"] == root_selection


@pytest.mark.parametrize("root_backend,child_backend", [
    ("cuda", "cpu"),
    ("cpu", "cuda"),
])
def test_a_child_on_another_backend_records_its_own_selection(
        tmp_path, monkeypatch, root_backend, child_backend):
    """The root's reason explains the root's backend.  A child that
    prepares on the other one must not carry it: its receipt would say
    the card was certified (or busy) about a preparation that ran on the
    CPU (or the card)."""

    from gpuwm.ingest import nest_init, preprocess_backend

    root_selection = {
        "requested": "auto", "backend": root_backend,
        "reason": "the root's own measured reason",
        "device_load": {"utilization_gpu_percent": 99}}
    _, observed = _call(
        tmp_path, monkeypatch, preprocess_backend=root_backend, workers=1,
        input_provenance={"preprocessing": {
            "backend": root_backend, "selection": root_selection}})
    catalog = observed["initialize"]["catalog"]

    def backend_named(name):
        class Backend:
            def __init__(self, **_kwargs):
                self.name = name
                self.selection = None

            def receipt(self):
                return {"backend": self.name,
                        "selection": dict(self.selection)}
        return Backend

    monkeypatch.setattr(preprocess_backend, "ParallelCpuPreprocessBackend",
                        backend_named("cpu"))
    monkeypatch.setattr(preprocess_backend, "CudaPreprocessBackend",
                        backend_named("cuda"))
    monkeypatch.setattr(preprocess_backend, "_gpu_runtime_installed", lambda: True)
    monkeypatch.setattr(nest_init, "build_static_for_domain", lambda *_a: {
        "LU_INDEX": np.ones((2, 2))})
    monkeypatch.setattr(nest_init, "geog_selection_from_catalog", lambda *_a:
                        SimpleNamespace(landuse_global_attrs=lambda: {"ISLAKE": 21}))
    monkeypatch.setattr(nest_init, "interpolate_era5_to_lambert", lambda *_a, **_k:
                        SimpleNamespace(fields={}))
    monkeypatch.setattr(nest_init, "_child_soil_mesh", lambda *_a: None)
    child = SimpleNamespace(
        grid_id=2, start_time=catalog.valid_times[0],
        run=SimpleNamespace(moist=True, terrain_opt=1))
    prepared = nest_init._prepare_child_input_on_grid(
        child, object(), catalog, preprocess_backend=child_backend)
    assert prepared.preprocess_receipt["selection"] == {
        "requested": child_backend, "backend": child_backend,
        "reason": preprocess_backend.NAMED_BY_CALLER}


class _Snapshot:
    def __init__(self, valid_time, fields):
        self.valid_time = valid_time
        self.fields = fields
        self.latitude = np.arange(40, dtype=np.float64)
        self.longitude = np.arange(40, dtype=np.float64)


class _Grid:
    def __init__(self, name, center):
        self.name = name
        self._latitude = np.array(
            [[center - 0.25, center + 0.25]], dtype=np.float64)
        self._longitude = np.array(
            [[center - 0.25, center + 0.25]], dtype=np.float64)

    def latlon_mass(self):
        return self._latitude, self._longitude

    def latlon_u(self):
        return self._latitude, self._longitude

    def latlon_v(self):
        return self._latitude, self._longitude


def _inputs(tmp_path, domain_count=3, parents=None, feedback=0):
    start = datetime(2026, 7, 20)
    if parents is None:
        parents = (0, *range(1, domain_count))
    dx_by_id = {}
    domains = []
    for grid_id, parent_id in zip(range(1, domain_count + 1), parents):
        root = grid_id == 1
        dx = 12000.0 if root else dx_by_id.get(parent_id, 12000.0) / 3.0
        dx_by_id[grid_id] = dx
        domains.append(SimpleNamespace(
            grid_id=grid_id,
            parent_id=parent_id,
            i_parent_start=1 if root else 11,
            j_parent_start=1 if root else 11,
            parent_grid_ratio=1 if root else 3,
            parent_time_step_ratio=1 if root else 3,
            run=SimpleNamespace(
                specified=root,
                nested=not root,
                nx=60,
                ny=60,
                dx=dx,
                dy=dx,
                dt=6.0 if root else 2.0,
                nz=49,
            ),
        ))
    domains = tuple(domains)
    exp = SimpleNamespace(
        start_time=start, run_seconds=7200, domains=domains,
        root=domains[0], feedback=feedback)
    boundaries = object()
    state = SimpleNamespace(lateral_boundaries=boundaries)
    initial = SimpleNamespace(state=state)
    snapshots = tuple(
        _Snapshot(start + timedelta(hours=hour), {
            "SOURCE_OROGRAPHY": object(),
        })
        for hour in (0, 1, 2)
    )
    return exp, boundaries, initial, snapshots


def _call(
        tmp_path, monkeypatch, *, domain_count=3, parents=None, feedback=0,
        **overrides):
    exp, boundaries, initial, snapshots = _inputs(
        tmp_path, domain_count, parents=parents, feedback=feedback)
    static_catalog = SimpleNamespace(files=("wps", "geog"))
    static_receipt = {"schema": "verified-static", "status": "PASS"}
    observed = {}

    def verified(wps, geog, domain_ids):
        assert tuple(domain_ids) == tuple(range(1, domain_count + 1))
        observed["verified"] = (wps, geog)
        return static_catalog, static_receipt

    def initialize(**kwargs):
        observed["initialize"] = kwargs
        return "native-result"

    monkeypatch.setattr(
        source_hierarchy, "verified_static_catalog", verified)
    monkeypatch.setattr(
        source_hierarchy, "initialize_and_export_native_hierarchy",
        initialize)
    arguments = dict(
        exp=exp,
        grids=tuple(
            _Grid(f"d{grid_id:02d}-grid", 17 + grid_id)
            for grid_id in range(1, domain_count + 1)
        ),
        snapshots=snapshots,
        forcing_hours=(0, 1, 2),
        wps_namelist=tmp_path / "namelist.wps",
        geog_root=tmp_path / "geog",
        source_name="GFS",
        artifact_output=tmp_path / "artifacts",
        wrf_output=tmp_path / "wrf",
        root_initial_result=initial,
        root_met="met",
        root_soil="soil",
        root_static_fields={"HGT_M": object()},
        root_boundaries=boundaries,
        bridge_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        namelist_sha256="c" * 64,
        source_identity={"adapter": "gfs"},
        workers=8,
        preprocess_backend="cpu",
        cpu_bridge="bridge",
        soil_layer_contract="soil-contract",
    )
    arguments.update(overrides)
    result = source_hierarchy.initialize_and_export_regular_source_hierarchy(
        **arguments)
    return result, observed


def test_regular_source_hierarchy_preserves_root_lbc_and_full_series(
        tmp_path, monkeypatch):
    result, observed = _call(tmp_path, monkeypatch)
    assert result.hierarchy == "native-result"
    assert result.boundary_interval_seconds == 3600
    call = observed["initialize"]
    assert call["root_node"].state is call["root_initial_result"].state
    assert call["root_node"].grid.name == "d01-grid"
    assert call["catalog"].snapshots == tuple(
        call["catalog"].snapshots)
    assert len(call["catalog"].snapshots) == 3
    assert call["catalog"].static_catalog.files == ("wps", "geog")
    assert call["workers"] == 8
    assert call["preprocess_backend"] == "cpu"
    assert call["boundary_interval_seconds"] == 3600
    assert call["source_orography"] is None
    assert call["soil_layer_contract"] == "soil-contract"
    assert set(result.source_coverage_receipt["domains"]) == {
        "d01", "d02", "d03"}
    assert result.topology_receipt["status"] == "PASS"
    assert result.topology_receipt["max_dom"] == 3


def test_regular_source_hierarchy_preserves_five_minute_offsets(
        tmp_path, monkeypatch):
    exp, _boundaries, _initial, _snapshots = _inputs(tmp_path)
    exp.run_seconds = 600
    snapshots = tuple(
        _Snapshot(exp.start_time + timedelta(seconds=offset), {
            "SOURCE_OROGRAPHY": object(),
        })
        for offset in (0, 300, 600))
    result, observed = _call(
        tmp_path, monkeypatch, exp=exp, snapshots=snapshots,
        forcing_hours=None, forcing_offsets_seconds=(0, 300, 600))

    call = observed["initialize"]
    assert result.boundary_interval_seconds == 300
    assert call["forcing_hours"] is None
    assert call["forcing_offsets_seconds"] == (0, 300, 600)
    assert call["boundary_interval_seconds"] == 300


@pytest.mark.parametrize("source_name", ("ERA5", "GFS"))
def test_regular_source_hierarchy_routes_every_d01_through_d06_domain(
        tmp_path, monkeypatch, source_name):
    result, observed = _call(
        tmp_path,
        monkeypatch,
        domain_count=6,
        source_name=source_name,
    )
    assert result.hierarchy == "native-result"
    assert set(result.source_coverage_receipt["domains"]) == {
        f"d{grid_id:02d}" for grid_id in range(1, 7)
    }
    assert tuple(
        domain.grid_id for domain in observed["initialize"]["exp"].domains
    ) == tuple(range(1, 7))
    assert result.topology_receipt["max_dom"] == 6


def test_regular_source_hierarchy_accepts_branched_d06_parent_order(
        tmp_path, monkeypatch):
    result, _observed = _call(
        tmp_path,
        monkeypatch,
        domain_count=6,
        parents=(0, 1, 2, 2, 3, 4),
    )
    assert [
        row["parent_id"] for row in result.topology_receipt["domains"]
    ] == [0, 1, 2, 2, 3, 4]


def test_regular_source_hierarchy_accepts_two_way_and_rejects_cycles(
        tmp_path, monkeypatch):
    """feedback=1 authors the same artifacts (the old refusal is lifted:
    coupling is runtime behaviour with no ingest footprint); a malformed
    value and a parent cycle still refuse before any static work."""
    result, observed = _call(tmp_path, monkeypatch, feedback=1)
    assert result.hierarchy == "native-result"
    with pytest.raises(ValueError, match="feedback must be 0 or 1"):
        _call(tmp_path, monkeypatch, feedback=2)
    with pytest.raises(ValueError, match="must precede"):
        _call(
            tmp_path,
            monkeypatch,
            parents=(0, 1, 3),
        )


def test_regular_source_hierarchy_rejects_time_drift_before_static_work(
        tmp_path, monkeypatch):
    exp, boundaries, initial, snapshots = _inputs(tmp_path)
    snapshots = list(snapshots)
    snapshots[1] = _Snapshot(
        exp.start_time + timedelta(minutes=59),
        {"SOURCE_OROGRAPHY": object()})
    monkeypatch.setattr(
        source_hierarchy, "verified_static_catalog",
        lambda *_args, **_kwargs: pytest.fail("static work must not start"))
    with pytest.raises(ValueError, match="differ from forcing hours"):
        source_hierarchy.initialize_and_export_regular_source_hierarchy(
            exp=exp, grids=(1, 2, 3), snapshots=snapshots,
            forcing_hours=(0, 1, 2),
            wps_namelist=tmp_path / "namelist.wps",
            geog_root=tmp_path / "geog", source_name="GFS",
            artifact_output=tmp_path / "artifacts",
            wrf_output=tmp_path / "wrf", root_initial_result=initial,
            root_met=object(), root_soil=object(), root_static_fields={},
            root_boundaries=boundaries,
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64, source_identity={})


def test_regular_source_hierarchy_rejects_child_source_halo_clipping(
        tmp_path, monkeypatch):
    clipped = (_Grid("d01-grid", 18), _Grid("d02-grid", 1),
               _Grid("d03-grid", 20))
    with pytest.raises(ValueError, match="d02 mass"):
        _call(tmp_path, monkeypatch, grids=clipped)


def test_regular_source_hierarchy_requires_serial_cuda_workers(
        tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="workers=1"):
        _call(
            tmp_path, monkeypatch,
            preprocess_backend="cuda", workers=2)


def test_regular_source_hierarchy_routes_explicit_serial_cuda(
        tmp_path, monkeypatch):
    result, observed = _call(
        tmp_path, monkeypatch, preprocess_backend="cuda", workers=1,
        cpu_bridge=None)
    assert result.hierarchy == "native-result"
    assert observed["initialize"]["preprocess_backend"] == "cuda"
    assert observed["initialize"]["workers"] == 1
    assert observed["initialize"]["cpu_bridge"] is None


def test_regular_source_hierarchy_refuses_d01_orography_reuse(
        tmp_path, monkeypatch):
    legacy = tmp_path / "d01-orography.nc"
    legacy.write_bytes(b"netcdf-placeholder")
    with pytest.raises(ValueError, match="separately for every domain"):
        _call(
            tmp_path, monkeypatch,
            source_orography=SourceOrography(legacy, "HGT"),
            snapshots=tuple(
                _Snapshot(datetime(2026, 7, 20) + timedelta(hours=hour), {})
                for hour in (0, 1, 2)))


def test_regular_source_hierarchy_accepts_complete_per_domain_orography(
        tmp_path, monkeypatch):
    artifacts = []
    for domain_id in (1, 2, 3):
        path = tmp_path / f"d{domain_id:02d}-orography.nc"
        path.write_bytes(b"netcdf-placeholder")
        artifacts.append((domain_id, SourceOrography(path, "HGT")))
    declaration = PerDomainSourceOrography(tuple(artifacts))
    snapshots = tuple(
        _Snapshot(datetime(2026, 7, 20) + timedelta(hours=hour), {})
        for hour in (0, 1, 2))
    result, observed = _call(
        tmp_path, monkeypatch, snapshots=snapshots,
        source_name="ERA5", source_orography=declaration)
    assert result.hierarchy == "native-result"
    assert observed["initialize"]["source_orography"] is declaration


@pytest.mark.parametrize("units", ("m2 s-2", "m^2 s^-2", "m**2 s**-2"))
def test_regular_source_hierarchy_uses_source_invariant_orography(
        tmp_path, monkeypatch, units):
    snapshots = tuple(
        _Snapshot(
            datetime(2026, 7, 20) + timedelta(hours=hour),
            {"SOILGEO": np.full((40, 40), 981.0)},
        )
        for hour in (0, 1, 2)
    )
    result, observed = _call(
        tmp_path,
        monkeypatch,
        snapshots=snapshots,
        source_name="ERA5",
        source_inventory=("SOILGEO",),
        source_units={"SOILGEO": units},
    )
    assert result.hierarchy == "native-result"
    assert observed["initialize"]["source_orography"] is None


def test_regular_source_hierarchy_rejects_unbound_invariant_units(
        tmp_path, monkeypatch):
    snapshots = tuple(
        _Snapshot(
            datetime(2026, 7, 20) + timedelta(hours=hour),
            {"SOILGEO": np.full((40, 40), 981.0)},
        )
        for hour in (0, 1, 2)
    )
    with pytest.raises(ValueError, match="geopotential units"):
        _call(
            tmp_path,
            monkeypatch,
            snapshots=snapshots,
            source_name="ERA5",
            source_inventory=("SOILGEO",),
            source_units={"SOILGEO": "m"},
        )


def test_regular_source_hierarchy_rejects_extra_per_domain_orography(
        tmp_path, monkeypatch):
    artifacts = []
    for domain_id in (1, 2, 3, 4):
        path = tmp_path / f"d{domain_id:02d}-orography.nc"
        path.write_bytes(b"netcdf-placeholder")
        artifacts.append((domain_id, SourceOrography(path, "HGT")))
    snapshots = tuple(
        _Snapshot(datetime(2026, 7, 20) + timedelta(hours=hour), {})
        for hour in (0, 1, 2)
    )
    with pytest.raises(ValueError, match="exactly cover"):
        _call(
            tmp_path,
            monkeypatch,
            snapshots=snapshots,
            source_name="ERA5",
            source_orography=PerDomainSourceOrography(tuple(artifacts)),
        )


def test_regular_source_hierarchy_requires_bound_external_lbc(
        tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="complete external LBC"):
        _call(tmp_path, monkeypatch, root_boundaries=object())


def test_regular_source_hierarchy_emits_corridors_only_on_opt_in(
        tmp_path, monkeypatch):
    """statics_corridor=None is byte-inert; 'all' builds one corridor per
    child through the real corridor module seam and returns the set
    receipt for the source door to bind into its preparation document."""

    from pathlib import Path

    import gpuwm.static.corridor as corridor_module

    calls = []

    def fake_build(*, child_dc, parent_run, reference_grid, static_catalog,
                   frame_kwargs, window, reach):
        assert static_catalog.files == ("wps", "geog")
        # A tree with no mover anchors every corridor to the child's own
        # parent, which is what {} means -- so the geometry, the receipt
        # and the sealed bytes are what they were before mid-tree moves
        # existed.  This is the regression that catches a frame chosen
        # two ways.
        assert frame_kwargs == {}
        # And nothing moves, so each child reaches exactly its own
        # footprint: the window is the child's size at its placement.
        assert tuple(window[2:]) == (int(child_dc.run.nx),
                                     int(child_dc.run.ny))
        assert reach["movers"] == [] and reach["whole_frame"] is False
        calls.append((int(child_dc.grid_id), int(parent_run.nx),
                      reference_grid.name))
        return SimpleNamespace(grid_id=int(child_dc.grid_id), fields={},
                               entry={})

    written = {}
    set_receipt = {"schema": "gpuwm-statics-corridor-set-v1",
                   "status": "READY", "domains": {}}

    def fake_write(directory, builds):
        written["directory"] = Path(directory)
        written["grid_ids"] = [build.grid_id for build in builds]
        return set_receipt

    monkeypatch.setattr(
        corridor_module, "build_child_statics_corridor", fake_build)
    monkeypatch.setattr(
        corridor_module, "write_statics_corridor_set", fake_write)

    result, _ = _call(tmp_path, monkeypatch, statics_corridor="all")
    assert dict(result.statics_corridor_receipt) == set_receipt
    assert written["grid_ids"] == [2, 3]
    assert written["directory"] == (
        tmp_path / "artifacts" / "statics-corridor")
    assert calls == [(2, 60, "d02-grid"), (3, 60, "d03-grid")]

    calls.clear()
    written.clear()
    inert, _ = _call(tmp_path, monkeypatch)
    assert inert.statics_corridor_receipt is None
    assert calls == [] and written == {}

    selected, _ = _call(tmp_path, monkeypatch, statics_corridor=(3,))
    assert written["grid_ids"] == [3]

    with pytest.raises(ValueError, match="not child domains"):
        _call(tmp_path, monkeypatch, statics_corridor=(1,))


def test_a_chained_head_builds_the_corridor_its_seal_copies(
        tmp_path, monkeypatch):
    """A136 L7d: a moving tree's corridor is built into its head.

    The corridor is child-resolution statics over the ground a nest can
    reach; it needs the geography and the tree but no boundary time, so a
    chained tree builds it at its head, where a head-bound forecast moves
    its nest over it.  The seal then copies the head's set into the
    one-shot tree instead of building it again, and the sealed set is byte
    for byte the one-shot tree's.  The breakage this prevents: every
    moving and cyclone tree's forecast waited for the seal; and a head set
    that changed before the seal is refused rather than sealed.
    """

    from pathlib import Path

    import gpuwm.static.corridor as corridor_module

    builds = []

    def fake_build(*, child_dc, parent_run, reference_grid, static_catalog,
                   frame_kwargs, window, reach):
        grid_id = int(child_dc.grid_id)
        builds.append(grid_id)
        return SimpleNamespace(
            grid_id=grid_id,
            fields={"HGT_M": np.full((4, 3), float(grid_id)),
                    "LANDMASK": np.eye(4, 3)},
            entry={"schema": corridor_module.STATICS_CORRIDOR_SCHEMA,
                   "status": "READY", "grid_id": grid_id})

    monkeypatch.setattr(
        corridor_module, "build_child_statics_corridor", fake_build)

    # The one-shot tree: the reference set.
    oneshot, _ = _call(tmp_path, monkeypatch, statics_corridor="all")
    oneshot_set = tmp_path / "artifacts" / "statics-corridor"
    assert builds == [2, 3]

    # The chained head, then its seal.
    builds.clear()
    exp, boundaries, initial, snapshots = _inputs(tmp_path, 3)
    monkeypatch.setattr(
        source_hierarchy, "initialize_native_hierarchy_children",
        lambda **_kwargs: ((), 0.0))
    monkeypatch.setattr(
        source_hierarchy, "export_native_hierarchy",
        lambda **_kwargs: "native-result")
    grids = tuple(_Grid(f"d{grid_id:02d}-grid", 17 + grid_id)
                  for grid_id in range(1, 4))
    head_artifacts = tmp_path / "chained" / "hierarchy-head"
    head = source_hierarchy.prepare_regular_source_hierarchy_head(
        exp=exp, grids=grids, snapshots=snapshots, forcing_hours=(0, 1, 2),
        wps_namelist=tmp_path / "namelist.wps", geog_root=tmp_path / "geog",
        source_name="GFS", root_initial_result=initial,
        source_manifest_sha256="b" * 64, statics_corridor="all",
        head_artifacts=head_artifacts)
    assert builds == [2, 3]
    assert dict(head.statics_corridor_receipt) == dict(
        oneshot.statics_corridor_receipt)
    head_set = head_artifacts / "statics-corridor"

    def files(folder: Path) -> dict:
        return {path.name: path.read_bytes()
                for path in sorted(folder.iterdir())}

    assert files(head_set) == files(oneshot_set)

    def seal(artifact_output):
        return source_hierarchy.seal_regular_source_hierarchy(
            head, artifact_output=artifact_output,
            wrf_output=tmp_path / "chained" / "wrf",
            root_initial_result=initial, root_met="met", root_soil="soil",
            root_static_fields={}, root_boundaries=boundaries,
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64, namelist_sha256="c" * 64,
            source_identity={"adapter": "gfs"},
            head_artifacts=head_artifacts)

    sealed = seal(tmp_path / "chained" / "hierarchy-artifacts")
    # Copied, not built again, and equal to the one-shot tree's bytes.
    assert builds == [2, 3]
    assert dict(sealed.statics_corridor_receipt) == dict(
        oneshot.statics_corridor_receipt)
    assert files(tmp_path / "chained" / "hierarchy-artifacts"
                 / "statics-corridor") == files(oneshot_set)

    # A head set whose bytes changed since its receipt is not sealed.  One
    # byte in the middle flips: an .npz ends in a zip trailer whose last
    # byte is already zero, so zeroing the last byte changed nothing.
    cache = head_set / "d03.npz"
    tampered = bytearray(cache.read_bytes())
    tampered[len(tampered) // 2] ^= 0xFF
    cache.write_bytes(bytes(tampered))
    with pytest.raises(corridor_module.CorridorRefusal, match="d03"):
        seal(tmp_path / "tampered" / "hierarchy-artifacts")
    # And a head that built its set cannot be sealed without it.
    with pytest.raises(ValueError, match="head's artifact folder"):
        source_hierarchy.seal_regular_source_hierarchy(
            head, artifact_output=tmp_path / "no-head" / "artifacts",
            wrf_output=tmp_path / "no-head" / "wrf",
            root_initial_result=initial, root_met="met", root_soil="soil",
            root_static_fields={}, root_boundaries=boundaries,
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64, namelist_sha256="c" * 64,
            source_identity={"adapter": "gfs"})
