"""Shared mapping uses the ordinary static preparation and terrain policies."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.mapped_physical import MappedPhysicalPreparation
from gpuwm.ensemble.physical_store import NativePhysicalStore, physical_static_identity


def _context(monkeypatch, tmp_path, *, prepared=False):
    from gpuwm import experiment, native_wrf_contract
    from gpuwm.ingest import preprocess_backend, water_temperature
    from gpuwm.static import highres_production as highres
    from gpuwm.static.lambert import LambertGrid
    from test_nest_spawn_init import _experiment

    original = _experiment()
    root = replace(original.root, run=replace(original.root.run, nx=16, ny=14))
    exp = replace(original, domains=(root,), smooth_cg_topo=True)
    grid = LambertGrid(ref_lat=35, ref_lon=-97, truelat1=30, truelat2=60,
                       stand_lon=-97, dx=root.run.dx, dy=root.run.dy,
                       e_we=17, e_sn=15)
    fields = {"HGT_M": np.full((14, 16), 100.), "LANDMASK": np.ones((14, 16)),
              "LU_INDEX": np.ones((14, 16))}
    overlay_fields = {**fields, "HGT_M": np.full((14, 16), 500.),
                      "LANDMASK": np.zeros((14, 16)), "LU_INDEX": np.full((14, 16), 17.)}
    config = highres.HighresStaticConfig(enabled=True, cache_root=tmp_path / "highres")
    overlay_receipt = {"status": "APPLIED", "config": config.echo(),
                       "case_date": exp.start_time.date().isoformat(),
                       "grid": highres._grid_identity(grid, 1)}
    receipt = {"highres": overlay_receipt} if prepared else {}
    calls = []

    def apply_overlay(baseline, mapped_grid, **kwargs):
        assert mapped_grid is grid
        assert kwargs["landuse_attrs"] == native_wrf_contract.NATIVE_LANDUSE_IDENTITY
        calls.append(kwargs)
        return dict(overlay_fields), overlay_receipt

    paths = [tmp_path / name for name in ("experiment.toml", "namelist.wps", "static.npz", "receipt.json")]
    for path in paths:
        path.write_text('name = "static-policy-fixture"\n', encoding="utf-8")
    monkeypatch.setattr(experiment, "load_experiment", lambda _: exp)
    monkeypatch.setattr(native_wrf_contract, "validate_native_lambert_contract", lambda *_a, **_k: grid)
    monkeypatch.setattr(native_wrf_contract, "verify_native_static_receipt", lambda *_a: receipt)
    monkeypatch.setattr(native_wrf_contract, "load_native_static_cache",
                        lambda *_a: dict(overlay_fields if prepared else fields))
    monkeypatch.setattr(highres, "load_static_highres", lambda _: config)
    monkeypatch.setattr(highres, "overlay_active", lambda *_: True)
    monkeypatch.setattr(highres, "apply_highres_statics", apply_overlay)
    water_inputs = {}

    def water_statics(**kwargs):
        water_inputs.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(water_temperature.WaterTemperatureStatics, "for_route", water_statics)
    backend_receipt = {"backend": "cpu", "workers": 1, "native_bridge_sha256": "a" * 64}
    monkeypatch.setattr(preprocess_backend, "ParallelCpuPreprocessBackend",
                        lambda **_: SimpleNamespace(receipt=lambda: dict(backend_receipt)))
    # String paths are part of the public constructor contract.
    context = MappedPhysicalPreparation(
        experiment_config=str(paths[0]), wps_namelist=str(paths[1]),
        static_input=str(paths[2]), static_receipt=str(paths[3]))
    return context, calls, water_inputs, backend_receipt


@pytest.mark.parametrize("prepared", [False, True])
def test_target_highres_is_applied_or_revalidated_before_mapping(monkeypatch, tmp_path, prepared):
    context, calls, water, _ = _context(monkeypatch, tmp_path, prepared=prepared)
    try:
        assert len(calls) == (0 if prepared else 1)
        assert np.all(context.static["HGT_M"] == 500)
        assert np.all(water["landmask"] == 0)
        assert np.all(water["lu_index"] == 17)
        assert context.static_identity == physical_static_identity(
            context.static, context.static_identity["attributes"])
    finally:
        context.close()


def test_each_source_blends_once_without_mutating_the_shared_statics(monkeypatch, tmp_path):
    from gpuwm.io.nc_writer_bridge import unavailable_reason
    from gpuwm.netcdf_bridge import find_netcdf_bin
    reason = unavailable_reason()
    if reason or find_netcdf_bin() is None:
        pytest.skip("native physical-store writer/reader required: " + str(reason))
    from gpuwm import mapped_composition, mapped_source, prep_handoff, source_adapters, source_cli
    from gpuwm.ingest import horiz
    from gpuwm.ingest.cg_topo import RootTerrainBlend
    from gpuwm.ensemble import mapped_physical_contract
    from physical_field_fixtures import analytic_field_contract

    context, _, _, backend_receipt = _context(monkeypatch, tmp_path)
    field_contract = analytic_field_contract(context.geometry)
    field_contract["vertical"]["pressure_field"] = None
    monkeypatch.setattr(mapped_physical_contract, "mapped_physical_field_contract",
                        lambda *_args, **_kwargs: field_contract)
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"provenance": {}}), encoding="utf-8")
    args = SimpleNamespace(source="fixture", input_list="unused", supplement=None,
        provenance=None, contributing_mapping=None, source_sha256s=str(manifest),
        author_input_manifest=None, source_sha256s_sha256=None,
        mapping="mapping.json", composition="composition.json", source_format="grib2")
    monkeypatch.setattr(prep_handoff, "preparation_arguments", lambda _: [])
    monkeypatch.setattr(source_cli, "_parser", lambda: SimpleNamespace(parse_args=lambda _: args))
    monkeypatch.setattr(source_adapters, "get_source_adapter", lambda _: SimpleNamespace(
        runner="mapped_composition_v1", packaged_profile=None))
    monkeypatch.setattr(mapped_source, "read_input_list", lambda _: ())
    monkeypatch.setattr(mapped_composition, "mapped_composition_receipt", lambda _: {})
    monkeypatch.setattr(mapped_composition, "composition_receipt_identity_sha256", lambda _: "d" * 64)
    mapping_masks = []

    def map_snapshot(snapshot, _grid, **kwargs):
        mapping_masks.append(kwargs["target_landmask"].copy())
        return snapshot

    monkeypatch.setattr(horiz, "interpolate_era5_to_lambert", map_snapshot)
    shared_before = {name: array.tobytes() for name, array in context.static.items()}
    closed = []
    try:
        for index, height in enumerate((10., 30.)):
            snapshots = tuple(horiz.HorizontalSnapshot(
                valid_time=context.experiment.start_time + timedelta(seconds=elapsed),
                levels_hpa=np.array([500.], dtype=np.float64),
                fields={"SOURCE_OROGRAPHY": np.full((14, 16), value, dtype=np.float32)})
                for elapsed, value in ((0, height), (context.experiment.run_seconds, 999.)))
            bundle = SimpleNamespace(mapping_sha256="b" * 64, composition_sha256="c" * 64,
                decoder_sha256={}, decoder_paths={},
                regular_snapshots=lambda: snapshots, close=lambda: closed.append(True))
            monkeypatch.setattr(mapped_composition, "decode_composed_source", lambda *_a, **_k: bundle)
            fetched = tmp_path / f"fetched-{index}"
            fetched.mkdir()
            (fetched / "prep-arguments.json").write_text("{}", encoding="utf-8")
            output = tmp_path / f"physical-{index}"
            result = context.capture(fetched, output)
            store = NativePhysicalStore(output)
            expected = dict(context.static)
            blend = RootTerrainBlend(context.experiment, expected, route="mapped")
            for snapshot in snapshots:
                blend.before_initialize(snapshot.fields["SOURCE_OROGRAPHY"])
            assert store.document["source"]["static_identity"] == physical_static_identity(
                expected, context.static_identity["attributes"])
            assert store.document["source"]["preprocessing"] == backend_receipt
            assert result["preprocessing"] == backend_receipt
            assert len(store.times) == 2
            assert store.read(1).fields["SOURCE_OROGRAPHY"].tobytes() == snapshots[1].fields["SOURCE_OROGRAPHY"].tobytes()
            assert all(array.tobytes() == shared_before[name] for name, array in context.static.items())
        assert len(closed) == 2
        assert len(mapping_masks) == 4 and all(not mask.any() for mask in mapping_masks)
    finally:
        context.close()
