"""Shared source inventories and original runtime delegation without CUDA."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble import runtime_preparation as preparation
from gpuwm.ensemble.runtime_preparation import RuntimePreparationSource


def test_unscoped_catalog_keeps_original_builder_and_has_no_cache(monkeypatch):
    from gpuwm.ingest import preflight
    calls = []
    monkeypatch.setattr(preflight, "build_input_catalog", lambda data: calls.append(data) or object())
    data = SimpleNamespace(forcing="inputs")
    assert preparation.current_runtime_preparation() is None
    assert preparation.runtime_input_catalog(data) is not preparation.runtime_input_catalog(data)
    assert calls == [data, data]


def test_catalog_and_decode_are_shared_only_inside_source_scope(tmp_path, monkeypatch):
    from gpuwm import runtime
    from gpuwm.ingest import preflight
    data = SimpleNamespace(forcing="source")
    catalog = SimpleNamespace(fingerprint="a" * 64)
    calls = []
    monkeypatch.setattr(preflight, "build_input_catalog", lambda data: calls.append("catalog") or catalog)
    snapshots = {datetime(2026, 10, 2): object()}
    monkeypatch.setattr(runtime, "_forcing_snapshots", lambda data, catalog: calls.append("decode") or snapshots)
    source = RuntimePreparationSource(tmp_path / "owned")
    with source.scope():
        assert preparation.runtime_input_catalog(data) is catalog
        assert preparation.runtime_input_catalog(data) is catalog
        assert runtime.forcing_snapshots(data) is snapshots
        assert runtime.forcing_snapshots(data, catalog) is snapshots
    assert preparation.current_runtime_preparation() is None
    assert calls == ["catalog", "decode"]
    assert source.receipt()["counts"]["catalog_builds"] == 1
    assert source.receipt()["counts"]["forcing_decodes"] == 1
    source.close()
    assert not source.root.exists()


def test_geometry_reuses_immutable_arrays_but_not_member_mapping(tmp_path):
    source = RuntimePreparationSource(tmp_path / "owned")
    words = np.arange(6, dtype=np.float64).reshape(2, 3)
    calls = []
    options = dict(selection=SimpleNamespace(resolution="native"), static_highres=None,
                   domain_id=1, case_date=datetime(2026, 10, 2).date(),
                   build=lambda: calls.append("static") or {"HGT_M": words})
    grid = SimpleNamespace(ref_lat=35., ref_lon=-99., dx=3000., dy=3000., e_we=4, e_sn=3)
    first = source.static_fields(grid, tmp_path, **options)
    second = source.static_fields(grid, tmp_path, **options)
    assert calls == ["static"]
    assert first is not second and first["HGT_M"] is second["HGT_M"]
    assert not first["HGT_M"].flags.writeable
    first["HGT_M"] = np.zeros_like(words)
    assert second["HGT_M"].tobytes() == words.tobytes()
    assert source.receipt()["host_retained_payload_bytes"] == words.nbytes
    source.close()


def test_cached_child_surface_upload_preserves_original_wind_arithmetic_precision():
    from gpuwm.ingest.real import surface_fields_to_device
    epsilon = float(np.finfo(np.float32).eps)
    wind = np.array([[1. + .49*epsilon, 1. + 1.49*epsilon]], np.float64)
    met = SimpleNamespace(fields={"T2": wind.copy(), "U10": wind.copy(), "V10": wind.copy()})
    for value in met.fields.values():
        value.flags.writeable = False
    restored = surface_fields_to_device(met, np, preserve_dtype=True)
    assert all(value.dtype == np.dtype("float64") for value in restored.values())
    assert all(restored[name].tobytes() == met.fields[name].tobytes() for name in restored)
    original = np.float32(.5 * (wind[0, 0] + wind[0, 1]))
    observed = np.float32(.5 * (restored["U10"][0, 0] + restored["U10"][0, 1]))
    assert original.tobytes() == observed.tobytes()
    # Rounding the two inputs before the staggered average changes the word.
    rounded = surface_fields_to_device(met, np)
    early = np.float32(.5 * (rounded["U10"][0, 0] + rounded["U10"][0, 1]))
    assert early.tobytes() != original.tobytes()


def test_retained_memory_counts_shared_views_once():
    array = np.arange(40, dtype=np.float64).reshape(5, 8)
    values = (array, {"alias": array, "slice": array[1:3]}, SimpleNamespace(words=array[:, ::-1]))
    assert preparation._retained_array_bytes(values) == array.nbytes


def _root_fixture(tmp_path):
    from test_prepared_cache import _fixture
    from gpuwm.config import RunConfig
    from gpuwm.experiment import VerticalConfig
    from gpuwm.runtime import PreparedRealCase
    initial, met, boundaries = _fixture()
    cfg = RunConfig(nx=2, ny=2, nz=2, dx=3000., dy=3000., ztop=10000., dt=12., run_seconds=24.)
    vertical = VerticalConfig(eta_levels=(1., 0.5, 0.), p_top=10000., hybrid_opt=0, etac=0.2)
    start = datetime(2026, 10, 2)
    grid = SimpleNamespace(ref_lat=35., ref_lon=-99., dx=3000., dy=3000., e_we=3, e_sn=3)
    static = {"HGT_M": np.zeros((2, 2), np.float64)}
    exp = SimpleNamespace(root=SimpleNamespace(grid_id=1, run=cfg), start_time=start, vertical=vertical)
    data = SimpleNamespace(geog_root=tmp_path, static_highres=None, forcing="source-a")
    catalog = SimpleNamespace(fingerprint="a" * 64)
    water = np.arange(4, dtype=np.float64).reshape(2, 2) + 270.123456789
    soil = SimpleNamespace(**{name: water.copy() for name in (
        "tsk", "soil_temperature", "soil_moisture", "liquid_moisture", "deep_soil_temperature",
        "xice", "xland", "landmask", "snow_water", "snow_depth")})
    prepared = PreparedRealCase(cfg, grid, static, initial, None, water.copy(),
                                (start, start + timedelta(hours=1)))
    inputs = dict(cfg=cfg, vertical=vertical, times=prepared.forcing_times,
        initial_result=initial, met=met, soil=soil, soil_fields={"SST": water},
        reconciled_soil_type=np.ones((2, 2), np.int32), boundaries=boundaries,
        landuse_attrs={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": 21, "ISICE": 15},
        trace_gas_overrides=None, radiation_column_chunk=7, constant_glw_wm2=None,
        cam_ozone=None, preprocess_backend="cpu")
    return exp, data, catalog, prepared, inputs


def test_first_root_is_original_then_typed_cache_is_reused_with_exact_fp64_soil(tmp_path, monkeypatch):
    from gpuwm import runtime
    from gpuwm.ingest.prepared_cache import PreparedCacheReader
    exp, data, catalog, prepared, inputs = _root_fixture(tmp_path)
    monkeypatch.setattr(preparation, "_device_id", lambda: 7)
    monkeypatch.setattr(runtime, "case_static_fields", lambda *a, **k: prepared.static_fields)
    source = RuntimePreparationSource(tmp_path / "owned")
    calls = []
    def build():
        calls.append("original")
        source.capture_root_inputs(**inputs)
        return prepared
    options = dict(grid=prepared.grid, selection=SimpleNamespace(resolution="native"), catalog=catalog,
                   scratch_arena=None, dycore_state_workspace=None, store_request=None, build=build)
    with source.scope():
        first = source.prepare_root(exp, data, **options)
        assert first is prepared and first.initial_result.state is prepared.initial_result.state
        restored = object()
        monkeypatch.setattr(source, "_restore_root", lambda *a, **k: calls.append("restore") or restored)
        assert source.prepare_root(exp, data, **options) is restored
    assert calls == ["original", "restore"]
    template = next(iter(source._roots.values()))
    assert not hasattr(template.initial_result, "state")
    reader = PreparedCacheReader(template.store_input.path, expected_identity=template.store_input.identity)
    actual = reader.read_array("surface/TSK")
    assert actual.dtype == np.dtype("float64")
    assert actual.tobytes() == inputs["soil"].tsk.tobytes()
    counts = source.receipt()["counts"]
    assert counts["root_preparations"] == counts["root_restores"] == 1
    created = source.receipt()["created_files"]
    source.close()
    assert source.receipt()["deleted_files"] == created


def test_root_cache_key_keeps_actual_geometry_source_and_configuration(tmp_path, monkeypatch):
    from gpuwm import runtime
    exp, data, catalog, prepared, inputs = _root_fixture(tmp_path)
    monkeypatch.setattr(preparation, "_device_id", lambda: 7)
    terrain = dict(prepared.static_fields)
    monkeypatch.setattr(runtime, "case_static_fields", lambda *a, **k: terrain)
    source = RuntimePreparationSource(tmp_path / "owned")
    builds = []
    def build():
        builds.append(True)
        source.capture_root_inputs(**inputs)
        return prepared
    options = dict(grid=prepared.grid, selection=None, catalog=catalog, scratch_arena=None,
                   dycore_state_workspace=None, store_request=None, build=build)
    with source.scope():
        source.prepare_root(exp, data, **options)
        terrain["HGT_M"] = terrain["HGT_M"] + 1.
        source.prepare_root(exp, data, **options)
        source.prepare_root(exp, SimpleNamespace(**{**vars(data), "forcing": "source-b"}), **options)
    assert len(builds) == 3
    assert len(source._roots) == 3
    source.close()


@dataclass(frozen=True)
class _Child:
    domain: object
    horizontal: object


def test_child_cache_stops_before_each_parent_dependent_finalize(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_device_id", lambda: 7)
    source = RuntimePreparationSource(tmp_path / "owned")
    domain = SimpleNamespace(grid_id=2, start_time=datetime(2026, 10, 2))
    grid = SimpleNamespace(dx=1000., dy=1000., e_we=3, e_sn=3)
    catalog = SimpleNamespace(fingerprint="a" * 64)
    words = np.asarray([0x80000000, 0x3f800001], np.uint32).view(np.float32).reshape(1, 2)
    child = _Child(domain, SimpleNamespace(fields={"TT": words}))
    calls = []
    options = dict(preprocess_backend="cpu", preprocess_workers=1, cpu_bridge=None,
                   build=lambda: calls.append("mapping") or child)
    first = source.prepare_child_input(domain, grid, catalog, None, **options)
    next_domain = SimpleNamespace(**vars(domain))
    second = source.prepare_child_input(next_domain, grid, catalog, None, **options)
    assert calls == ["mapping"] and first is child
    assert second is not first and second.domain is next_domain
    assert second.horizontal.fields["TT"].tobytes() == words.tobytes()
    assert source.receipt()["counts"]["child_input_reuses"] == 1
    assert source.receipt()["device_retained_payload_bytes"] == 0
    source.close()


def _labelled_member(member_id, seed, *, source="a", recipe="r", verification=None):
    def verify():
        if verification is not None:
            verification.append(member_id)
        return {"manifest_sha256": source}
    return SimpleNamespace(member_id=member_id, seed=seed,
        trajectory=SimpleNamespace(identity=source), recipe_sha256=recipe,
        geometry_sha256="geometry", boundary_valid_times=(datetime(2026, 10, 2),),
        source_manifests=(SimpleNamespace(verify=verify),), donor_manifests=(),
        inputs=SimpleNamespace(authority_sha256={"native_head": source}, source_identity={"source": source}),
        preparation_receipt={"cold_words_sha256": source})


def test_same_source_labels_reuse_decode_and_child_words_but_verify_every_member(tmp_path, monkeypatch):
    from gpuwm.ingest import preflight
    monkeypatch.setattr(preparation, "_device_id", lambda: 7)
    source = RuntimePreparationSource(tmp_path / "owned")
    calls, verified = [], []
    catalog = SimpleNamespace(fingerprint="a" * 64)
    monkeypatch.setattr(preflight, "build_input_catalog", lambda data: calls.append("catalog") or catalog)
    data = SimpleNamespace(forcing="native-source")
    domain = SimpleNamespace(grid_id=2)
    grid = SimpleNamespace(dx=1000., dy=1000., e_we=3, e_sn=3)
    child = _Child(domain, SimpleNamespace(fields={"TT": np.array([1.25], np.float32)}))
    def exercise(member):
        with source.scope(member):
            selected = source.catalog(data)
            words = source.forcing_snapshots(data, selected,
                build=lambda: calls.append("decode") or {"initial": np.array([271.125], np.float64)})
            mapped = source.prepare_child_input(domain, grid, selected, None,
                preprocess_backend="cpu", preprocess_workers=1, cpu_bridge=None,
                build=lambda: calls.append("child") or child)
        return words, mapped
    first = exercise(_labelled_member(19, 101, verification=verified))
    second = exercise(_labelled_member(44, 202, verification=verified))
    assert calls == ["catalog", "decode", "child"]
    assert verified == [19, 19, 19, 44, 44, 44]
    assert first[0] is second[0]
    assert first[1].horizontal.fields["TT"].tobytes() == second[1].horizontal.fields["TT"].tobytes()
    # Same file locations cannot erase a changed native or recipe authority.
    exercise(_labelled_member(3, 303, source="b", verification=verified))
    exercise(_labelled_member(5, 404, recipe="other", verification=verified))
    assert calls == ["catalog", "decode", "child"] * 3
    source.close()


def test_cold_root_reuses_same_source_across_labels_and_retains_each_attribution(tmp_path, monkeypatch):
    from gpuwm import runtime
    exp, data, catalog, prepared, inputs = _root_fixture(tmp_path)
    monkeypatch.setattr(preparation, "_device_id", lambda: 7)
    monkeypatch.setattr(runtime, "case_static_fields", lambda *a, **k: prepared.static_fields)
    source = RuntimePreparationSource(tmp_path / "owned")
    calls = []
    def build():
        calls.append("prepare")
        source.capture_root_inputs(**inputs)
        return prepared
    restored = object()
    monkeypatch.setattr(source, "_restore_root", lambda *a, **k: calls.append("restore") or restored)
    options = dict(grid=prepared.grid, selection=None, catalog=catalog,
                   scratch_arena=None, dycore_state_workspace=None, store_request=None, build=build)
    for member in (_labelled_member(19, 101), _labelled_member(44, 202)):
        with source.scope(member):
            result = source.prepare_root(exp, data, **options)
        assert result is (prepared if member.member_id == 19 else restored)
    assert calls == ["prepare", "restore"]
    receipt = source.receipt()
    assert len(receipt["root_inputs"]) == 1
    labels = receipt["root_inputs"][0]["binding"]["member_sources"]
    assert [(row["member_id"], row["seed"]) for row in labels] == [(19, 101), (44, 202)]
    source.close()


def test_runtime_ensemble_dispatch_precedes_every_original_operation(tmp_path):
    from gpuwm import runtime
    from gpuwm.ensemble.runtime_context import ensemble_scope
    calls = []
    class Session:
        def run_experiment(self, runner, exp, data, **options):
            calls.append((runner, exp, data, options))
            return "ensemble"
    exp, data = object(), object()
    with ensemble_scope(Session()):
        assert runtime.run_experiment(exp, data, tmp_path, health_debug=True) == "ensemble"
    assert calls[0][0] is runtime.run_experiment
    assert calls[0][1:3] == (exp, data)
    assert calls[0][3]["health_debug"] is True
