from __future__ import annotations

from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.mapped_direct as mapped_direct


_START = datetime(2026, 7, 20)
_DIGEST = "a" * 64


def test_provenance_evidence_name_is_short_stable_and_role_bound():
    role = "twentycrv3_in_band_surface_provenance"
    name = mapped_direct._provenance_evidence_name(role, ".json")

    assert name == "provenance-b3611ca2660a0367.json"
    assert len(name) == 32
    assert mapped_direct._provenance_evidence_name(role, ".json") == name
    assert mapped_direct._provenance_evidence_name(role + "-other", ".json") \
        != name


def _target_mapping(**updates):
    target = {
        "max_dom": 4,
        "target_vertical_levels": 49,
        "require_lateral_boundaries": True,
        "boundary_interval_seconds": 3600,
    }
    target.update(updates)
    return {"format": "netcdf", "target": target}


def _experiment(domain_count: int, *, nz: int = 49, run_seconds: int = 3600):
    domains = []
    for index in range(domain_count):
        run = SimpleNamespace(
            nz=nz,
            hybrid_opt=2,
            etac=0.2,
            spec_bdy_width=5,
            spec_zone=1,
            relax_zone=4,
            # The land-surface selector the soil seam routes on.  Noah here,
            # which is the geometry this adapter's declarative soil contract
            # has a target for; the seam refuses the others by name rather
            # than defaulting.
            sf_surface_physics=2,
        )
        domains.append(SimpleNamespace(
            grid_id=index + 1, parent_id=0 if index == 0 else index,
            start_time=None, run=run))
    exp = SimpleNamespace(
        domains=tuple(domains),
        root=domains[0],
        start_time=_START,
        run_seconds=run_seconds,
        vertical=SimpleNamespace(
            eta_levels=tuple(np.linspace(1.0, 0.0, nz + 1)),
            p_top=10_000.0,
        ),
    )
    exp.dt_exact = lambda grid_id: Fraction(60, 3 ** (grid_id - 1))
    exp.domain_start_offset_exact = lambda _grid_id: Fraction(0)
    exp.domain_start_time = lambda _grid_id: exp.start_time
    return exp


@pytest.mark.parametrize(
    ("mapping", "exp", "interval", "hierarchy", "message"),
    [
        (
            _target_mapping(max_dom=1),
            _experiment(2),
            3600,
            True,
            "max_dom",
        ),
        (
            _target_mapping(target_vertical_levels=48),
            _experiment(1),
            3600,
            False,
            "vertical|levels|nz",
        ),
        (
            _target_mapping(require_lateral_boundaries=False),
            _experiment(1),
            3600,
            False,
            "lateral boundar",
        ),
        (
            _target_mapping(boundary_interval_seconds=10_800),
            _experiment(1),
            3600,
            False,
            "boundary.*interval|cadence",
        ),
    ],
)
def test_mapped_target_contract_fails_closed(
        mapping, exp, interval, hierarchy, message):
    with pytest.raises(ValueError, match=message):
        mapped_direct._validate_target_contract(
            mapping, exp, interval, hierarchy=hierarchy,
        )


def test_mapped_target_contract_returns_bound_receipt():
    receipt = mapped_direct._validate_target_contract(
        _target_mapping(), _experiment(2), 3600, hierarchy=True,
    )
    assert receipt["domain_count"] == 2
    assert receipt["mapping_max_dom"] == 4
    assert receipt["target_vertical_levels"] == 49
    assert receipt["boundary_interval_seconds"] == 3600
    assert receipt["require_lateral_boundaries"] is True


def test_mapped_target_contract_accepts_five_minute_hierarchy_and_names_real_refusal():
    receipt = mapped_direct._validate_target_contract(
        _target_mapping(boundary_interval_seconds=300),
        _experiment(2), 300, hierarchy=True)
    assert receipt["boundary_interval_seconds"] == 300

    with pytest.raises(
            ValueError,
            match=r"310 s.*whole number of root-domain steps.*31/6"):
        mapped_direct._validate_target_contract(
            _target_mapping(boundary_interval_seconds=310),
            _experiment(2), 310, hierarchy=True)


class _Grid:
    def __init__(self, domain_id: int):
        self.domain_id = domain_id

    def mapfac_m(self):
        return np.ones((2, 2), dtype=np.float64)

    def mapfac_u(self):
        return np.ones((2, 3), dtype=np.float64)

    def mapfac_v(self):
        return np.ones((3, 2), dtype=np.float64)

    def coriolis_m(self):
        plane = np.ones((2, 2), dtype=np.float64)
        return plane, plane

    def rotation_m(self):
        plane = np.ones((2, 2), dtype=np.float64)
        return np.zeros_like(plane), plane


class _State:
    def __init__(self):
        self.lateral_boundaries = None

    def set_map_coriolis(self, *_args, **_kwargs):
        pass


class _Selection:
    resolution_tokens = ("default",)

    def __init__(self, geog_root: Path):
        self._geog_root = geog_root

    def path(self, field):
        return self._geog_root / f"{field}.bin"


def _write(path: Path, contents: bytes = b"test") -> Path:
    path.write_bytes(contents)
    return path


def _install_prepare_fakes(
        monkeypatch, tmp_path, *, domain_count: int, backend: str,
        cadence: int = 3600, run_seconds: int | None = None,
        mapping_updates=None, nz: int = 49):
    run_seconds = cadence if run_seconds is None else run_seconds
    exp = _experiment(domain_count, nz=nz, run_seconds=run_seconds)
    mapping = _target_mapping(**(mapping_updates or {}))
    grids = tuple(_Grid(index + 1) for index in range(domain_count))
    snapshots = tuple(
        SimpleNamespace(
            valid_time=_START + timedelta(seconds=index * cadence),
            levels_hpa=np.array([1000.0, 500.0, 100.0]),
            fields={
                "SOURCE_OROGRAPHY": np.zeros((2, 2), dtype=np.float64),
                "PRES": np.ones((3, 2, 2), dtype=np.float64),
            },
        )
        for index in range(2)
    )
    bundle = SimpleNamespace(
        regular_snapshots=lambda: snapshots,
        mapping_sha256="1" * 64,
        composition_sha256="2" * 64,
        input_manifest_sha256="3" * 64,
        decoder_paths={},
        decoder_sha256={},
        soil_layer_contract={"fixture": "declarative-soil-contract"},
    )
    composition_receipt = {"receipt_content_sha256": "4" * 64}
    states = tuple(_State() for _ in snapshots)
    results = tuple(SimpleNamespace(state=state) for state in states)
    mets = tuple(
        SimpleNamespace(fields={
            "frame": index,
            "SOURCE_OROGRAPHY": np.full((2, 2), 123.0 + index),
        })
        for index in range(len(snapshots))
    )
    boundaries = object()
    soil = object()
    calls = {
        "build_static": 0,
        "interpolate": 0,
        "initialize": 0,
        "hierarchy": [],
        "single_export": [],
    }

    geog_root = tmp_path / "geog"
    geog_root.mkdir()
    files = {
        name: _write(tmp_path / name)
        for name in (
            "composition.json", "mapping.json", "primary.bin",
            "supplement.bin", "provenance.md", "manifest.json",
            "namelist.wps", "experiment.toml",
        )
    }
    files["mapping.json"].write_text(json.dumps(mapping), encoding="utf-8")
    bundle.mapping_sha256 = hashlib.sha256(
        files["mapping.json"].read_bytes()
    ).hexdigest()
    bundle.composition_sha256 = hashlib.sha256(
        files["composition.json"].read_bytes()
    ).hexdigest()
    bundle.input_manifest_sha256 = hashlib.sha256(
        files["manifest.json"].read_bytes()
    ).hexdigest()
    bundle.mapping_path = files["mapping.json"].resolve()
    bundle.composition_path = files["composition.json"].resolve()
    bundle.input_manifest_path = files["manifest.json"].resolve()
    bundle.terrain_data_paths = (files["supplement.bin"].resolve(),)
    bundle.terrain_provenance_path = files["provenance.md"].resolve()
    bundle.terrain_provenance_sha256 = hashlib.sha256(
        files["provenance.md"].read_bytes()
    ).hexdigest()

    monkeypatch.setattr(
        mapped_direct,
        "load_mapping",
        lambda _path, **_kwargs: mapping,
    )
    monkeypatch.setattr(mapped_direct, "load_experiment", lambda _path: exp)
    monkeypatch.setattr(
        mapped_direct, "validate_native_lambert_contracts",
        lambda *_args, **_kwargs: grids,
    )
    if hasattr(mapped_direct, "validate_native_lambert_contract"):
        monkeypatch.setattr(
            mapped_direct, "validate_native_lambert_contract",
            lambda *_args, **_kwargs: grids[0],
        )
    monkeypatch.setattr(
        mapped_direct, "decode_composed_source",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        mapped_direct, "mapped_composition_receipt",
        lambda _bundle: composition_receipt,
    )
    monkeypatch.setattr(
        mapped_direct, "validate_explicit_eta_grid",
        lambda *_args, **_kwargs: None,
    )

    class FakeGeogSelection:
        @staticmethod
        def from_case_data(_case, _domain_id):
            return _Selection(geog_root)

    monkeypatch.setattr(mapped_direct, "GeogSelection", FakeGeogSelection)

    static = {
        "HGT_M": np.zeros((2, 2), dtype=np.float64),
        "LU_INDEX": np.ones((2, 2), dtype=np.int32),
        "SCT_DOM": np.ones((2, 2), dtype=np.int32),
        "TMN": np.full((2, 2), 280.0, dtype=np.float64),
        "LANDMASK": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
    }

    def build_static(*_args, **_kwargs):
        calls["build_static"] += 1
        return static

    monkeypatch.setattr(mapped_direct, "build_static", build_static)
    preprocess = SimpleNamespace(receipt=lambda: {"backend": backend})
    monkeypatch.setattr(
        mapped_direct, "resolve_preprocess_backend",
        lambda *_args, **_kwargs: preprocess,
    )

    def interpolate(source, _grid, **kwargs):
        calls["interpolate"] += 1
        # Metgrid classifies masked-field target cells by the model
        # landmask; the mapped lane must declare it like every other lane.
        np.testing.assert_array_equal(
            kwargs["target_landmask"],
            np.asarray(static["LANDMASK"]) >= 0.5,
            err_msg="mapped lane must pass the static landmask as the "
                    "masked-field target classification")
        return mets[snapshots.index(source)]

    def initialize(met, *_args, **_kwargs):
        calls["initialize"] += 1
        return results[mets.index(met)]

    monkeypatch.setattr(
        mapped_direct, "interpolate_era5_to_lambert", interpolate,
    )
    monkeypatch.setattr(mapped_direct, "initialize_real", initialize)
    # The mapped lane no longer hands every state to one builder: it
    # streams, adding each state's perimeter frames as that state is
    # built so the state itself can be released.  This double records
    # the sequence so the test still proves the builder saw the states,
    # AND now proves they arrived one at a time in forcing order.
    frame_calls = {"added": [], "built": []}

    class RecordingFrames:
        def __init__(self, **kwargs):
            frame_calls["kwargs"] = kwargs

        def add_state(self, state):
            frame_calls["added"].append(state)

        def build(self, actual_times):
            frame_calls["built"].append(tuple(actual_times))
            return boundaries

    monkeypatch.setattr(mapped_direct, "StateBoundaryFrames", RecordingFrames)
    calls["frames"] = frame_calls

    def attach(state, value):
        state.lateral_boundaries = value

    monkeypatch.setattr(mapped_direct, "attach_lateral_boundaries", attach)
    def lake_skin(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError(
            "mapped lane must not run the retired lake skin override")

    if hasattr(mapped_direct, "interpolate_lake_skin_temperature"):
        monkeypatch.setattr(
            mapped_direct, "interpolate_lake_skin_temperature", lake_skin,
        )

    def preprocess_soil(*_args, **kwargs):
        assert kwargs.get("lake_mask") is None
        assert kwargs.get("lake_skin_temperature") is None
        np.testing.assert_array_equal(
            kwargs["landmask"], static["LANDMASK"],
            err_msg="mapped soil must classify by the static landmask")
        assert kwargs["terrain"] is static["HGT_M"]
        np.testing.assert_array_equal(
            kwargs["source_orography"],
            mets[0].fields["SOURCE_OROGRAPHY"],
            err_msg="mapped soil must receive the composition source "
                    "orography for the elevation lapse")
        return soil

    monkeypatch.setattr(
        mapped_direct, "preprocess_land_surface_soil", preprocess_soil,
    )
    monkeypatch.setattr(
        mapped_direct, "native_static_export_fields",
        lambda actual, _grid: actual,
    )
    monkeypatch.setattr(
        mapped_direct, "canonical_noah_surface", lambda value: {"soil": value},
    )

    def write_static(path, *_args, **_kwargs):
        path.write_bytes(b"static")
        return {"sha256": hashlib.sha256(b"static").hexdigest()}

    def write_geometry(path, *_args, **_kwargs):
        path.write_text("{}", encoding="utf-8")
        return {"status": "PASS"}

    monkeypatch.setattr(mapped_direct, "write_native_static_cache", write_static)
    monkeypatch.setattr(
        mapped_direct, "write_native_geometry_receipt", write_geometry,
    )
    monkeypatch.setattr(
        mapped_direct, "prepared_cache_identity",
        lambda **kwargs: {"identity": kwargs},
    )

    def write_cache(path, **_kwargs):
        path.mkdir()
        return {"status": "PASS"}

    monkeypatch.setattr(mapped_direct, "write_prepared_cache", write_cache)

    def single_export(*args, **kwargs):
        calls["single_export"].append((args, kwargs))
        return {"schema": "gpuwm-native-direct-wrf-export-v2"}

    monkeypatch.setattr(mapped_direct, "export_prepared_wrf", single_export)

    hierarchy_result = SimpleNamespace(
        static_catalog_receipt={"status": "PASS"},
        source_coverage_receipt={"status": "PASS"},
        forcing_times=tuple(snapshot.valid_time for snapshot in snapshots),
        boundary_interval_seconds=cadence,
        hierarchy=SimpleNamespace(
            artifacts=SimpleNamespace(receipt={"status": "PASS"}),
            wrf_manifest={"schema": "gpuwm-native-direct-wrf-hierarchy-export-v1"},
            timings_seconds={"initialize_children": 0.1},
        ),
    )

    def hierarchy(**kwargs):
        calls["hierarchy"].append(kwargs)
        return hierarchy_result

    monkeypatch.setattr(
        mapped_direct, "initialize_and_export_regular_source_hierarchy",
        hierarchy,
    )
    args = {
        "composition": files["composition.json"],
        "mapping": files["mapping.json"],
        "primary_files": [files["primary.bin"]],
        "supplement_files": {"terrain": files["supplement.bin"]},
        "provenance_files": {"terrain": files["provenance.md"]},
        "input_manifest": files["manifest.json"],
        "input_manifest_sha256": _DIGEST,
        "wps_namelist": files["namelist.wps"],
        "geog_root": geog_root,
        "experiment_config": files["experiment.toml"],
        "output_root": tmp_path / "output",
        "preprocess_backend": backend,
    }
    expected = SimpleNamespace(
        exp=exp,
        mapping=mapping,
        grids=grids,
        snapshots=snapshots,
        results=results,
        mets=mets,
        boundaries=boundaries,
        soil=soil,
        bundle=bundle,
        static=static,
    )
    return args, calls, expected


def test_single_domain_preserves_legacy_direct_export(monkeypatch, tmp_path):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert proof["schema"] == mapped_direct.PROOF_SCHEMA
    assert len(calls["single_export"]) == 1
    assert not calls["hierarchy"]
    assert calls["build_static"] == 1
    assert calls["interpolate"] == len(expected.snapshots)
    assert args["output_root"].is_dir()
    # Streaming contract: every forcing time contributed its perimeter
    # frames, one state at a time and in forcing order, and the intervals
    # were assembled once from those frames.  The old builder took all the
    # states at once, which is what made preprocessing hold them all.
    frames = calls["frames"]
    assert len(frames["added"]) == len(expected.snapshots)
    assert frames["added"] == [r.state for r in expected.results]
    assert len(frames["built"]) == 1
    assert frames["kwargs"]["spec_bdy_width"] == (
        expected.exp.root.run.spec_bdy_width)


def test_long_provenance_role_publishes_short_hash_named_evidence(
    monkeypatch,
    tmp_path,
):
    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    role = "twentycrv3_in_band_surface_provenance"
    args["provenance_files"] = {
        role: expected.bundle.terrain_provenance_path,
    }

    mapped_direct.prepare_mapped_wrf(**args)

    evidence = args["output_root"] / "source-evidence" / (
        mapped_direct._provenance_evidence_name(role, ".md")
    )
    assert evidence.read_bytes() == b"test"
    assert role not in evidence.name


def test_predecoded_bundle_uses_bound_authorities_and_named_adapter(
    monkeypatch,
    tmp_path,
):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cuda",
    )
    args["input_manifest_sha256"] = expected.bundle.input_manifest_sha256
    args["_predecoded_bundle"] = expected.bundle
    args["_predecoded_seconds"] = 1.25
    args["_source_adapter"] = "rw-wps-fixture-member-v1"
    monkeypatch.setattr(
        mapped_direct,
        "decode_composed_source",
        lambda *_args, **_kwargs: pytest.fail(
            "predecoded bundle unexpectedly entered generic decode"
        ),
    )

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert len(calls["hierarchy"]) == 1
    routed = calls["hierarchy"][0]
    assert routed["source_identity"]["adapter"] \
        == "rw-wps-fixture-member-v1"
    assert proof["timing_seconds"]["decode_and_compose"] == 1.25


def test_prepare_rejects_mapping_change_between_target_and_decode(
    monkeypatch,
    tmp_path,
):
    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )

    def decode(*_args, **_kwargs):
        args["mapping"].write_text(
            json.dumps(_target_mapping(max_dom=3)),
            encoding="utf-8",
        )
        expected.bundle.mapping_sha256 = hashlib.sha256(
            args["mapping"].read_bytes()
        ).hexdigest()
        return expected.bundle

    monkeypatch.setattr(mapped_direct, "decode_composed_source", decode)
    with pytest.raises(ValueError, match="target validation and decode"):
        mapped_direct.prepare_mapped_wrf(**args)


def test_prepare_rejects_changed_evidence_bytes_before_publication(
    monkeypatch,
    tmp_path,
):
    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )

    def receipt(_bundle):
        args["composition"].write_bytes(b"changed composition")
        return {"receipt_content_sha256": "4" * 64}

    monkeypatch.setattr(mapped_direct, "mapped_composition_receipt", receipt)
    with pytest.raises(ValueError, match="evidence changed"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


@pytest.mark.parametrize(("domain_count", "backend", "expected_workers"), [
    (2, "cpu", 8),
    (2, "cuda", 1),
    (4, "cpu", 8),
])
def test_mapped_hierarchy_routes_complete_root_inputs(
        monkeypatch, tmp_path, domain_count, backend, expected_workers):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=domain_count, backend=backend,
    )
    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert not calls["single_export"]
    assert len(calls["hierarchy"]) == 1
    routed = calls["hierarchy"][0]
    assert routed["exp"] is expected.exp
    assert routed["grids"] == expected.grids
    assert routed["snapshots"] == expected.snapshots
    assert routed["root_initial_result"] is expected.results[0]
    assert routed["root_met"] is expected.mets[0]
    assert routed["root_soil"] is expected.soil
    assert routed["root_static_fields"] is expected.static
    assert routed["root_boundaries"] is expected.boundaries
    assert routed["bridge_manifest_sha256"] == expected.bundle.input_manifest_sha256
    assert routed["source_manifest_sha256"] == expected.bundle.input_manifest_sha256
    assert routed["source_identity"]["mapping_sha256"] == (
        expected.bundle.mapping_sha256
    )
    assert routed["source_identity"]["composition_sha256"] == (
        expected.bundle.composition_sha256
    )
    assert routed["source_identity"]["adapter"] == \
        "rw-wps-mapped-composition-v2"
    assert set(routed["source_inventory"]) == set(expected.snapshots[0].fields)
    assert routed["workers"] == expected_workers
    assert routed["preprocess_backend"] == backend
    assert routed["soil_layer_contract"] is expected.bundle.soil_layer_contract
    assert routed["artifact_manifest_reference"] == (
        "../hierarchy-artifacts/domain-artifacts.json"
    )
    assert proof["domain_count"] == domain_count
    assert proof["hierarchy_workers"] == expected_workers


def test_mapped_hierarchy_preserves_five_minute_offsets_end_to_end(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu",
        cadence=300, mapping_updates={"boundary_interval_seconds": 300})

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert proof["forcing_offsets_seconds"] == [0, 300]
    hierarchy_call = calls["hierarchy"][0]
    assert hierarchy_call["forcing_offsets_seconds"] == (0, 300)
    assert "forcing_hours" not in hierarchy_call
    assert args["output_root"].is_dir()


@pytest.mark.parametrize(
    ("domain_count", "cadence", "nz", "mapping_updates", "message"),
    [
        (2, 3600, 49, {"max_dom": 1}, "max_dom"),
        (1, 3600, 48, {}, "vertical|levels|nz"),
        (
            1,
            3600,
            49,
            {"require_lateral_boundaries": False},
            "lateral boundar",
        ),
        (
            1,
            3600,
            49,
            {"boundary_interval_seconds": 10_800},
            "boundary.*interval|cadence",
        ),
        (
            2,
            310,
            49,
            {"boundary_interval_seconds": 310},
            "whole number of root-domain steps",
        ),
    ],
)
def test_invalid_target_contract_stops_before_static_or_preprocessing(
        monkeypatch, tmp_path, domain_count, cadence, nz, mapping_updates,
        message):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch,
        tmp_path,
        domain_count=domain_count,
        backend="cpu",
        cadence=cadence,
        nz=nz,
        mapping_updates=mapping_updates,
    )
    with pytest.raises(ValueError, match=message):
        mapped_direct.prepare_mapped_wrf(**args)

    assert calls["build_static"] == 0
    assert calls["interpolate"] == 0
    assert calls["initialize"] == 0
    assert not calls["hierarchy"]
    assert not calls["single_export"]
    assert not args["output_root"].exists()


def _mapped_cli_args(source_format: str, *decoder_args: str) -> list[str]:
    return [
        "--source-format", source_format,
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/forcing-000",
        "--supplement", "terrain=/source/terrain-000",
        "--provenance", "terrain_provenance=/case/terrain.md",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        *decoder_args,
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
    ]


def test_mapped_cli_rejects_source_format_mapping_mismatch(
        monkeypatch, capsys):
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda _path: {"format": "netcdf"},
    )

    with pytest.raises(SystemExit) as error:
        mapped_direct.main(_mapped_cli_args(
            "grib1", "--grib1-bridge", "/bin/grib1_bridge",
        ))

    assert error.value.code == 2
    assert "differs from mapping format netcdf" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source_format", "decoder_args", "expected"),
    [
        ("grib1", (), "grib1_bridge"),
        (
            "grib2",
            ("--grib2-inventory", "/bin/grib2_inventory"),
            "grib2_dump",
        ),
        (
            "netcdf",
            ("--grib1-bridge", "/bin/grib1_bridge"),
            "grib1_bridge",
        ),
    ],
)
def test_mapped_cli_decoder_inventory_fails_closed(
        monkeypatch, capsys, source_format, decoder_args, expected):
    monkeypatch.setattr(
        mapped_direct, "load_mapping",
        lambda _path: {"format": source_format},
    )

    with pytest.raises(SystemExit) as error:
        mapped_direct.main(_mapped_cli_args(source_format, *decoder_args))

    assert error.value.code == 2
    assert expected in capsys.readouterr().err


def test_role_bindings_preserve_aliases_paths_and_supplement_order():
    bindings = mapped_direct._role_bindings([
        "terrain-height.v1=/source/terrain=analysis-a.grib2",
        "terrain-height.v1=/source/terrain-b.grib2",
        "land_mask=/source/land.nc",
    ], multiple=True)

    assert bindings == {
        "terrain-height.v1": (
            Path("/source/terrain=analysis-a.grib2"),
            Path("/source/terrain-b.grib2"),
        ),
        "land_mask": (Path("/source/land.nc"),),
    }


def test_role_bindings_reject_duplicate_singleton_and_malformed_roles():
    with pytest.raises(ValueError, match="duplicate binding.*terrain"):
        mapped_direct._role_bindings([
            "terrain=/source/a", "terrain=/source/b",
        ], multiple=False)
    for binding in ("terrain", "=/source/a", "bad role=/source/a", "terrain="):
        with pytest.raises(ValueError, match="ROLE=PATH"):
            mapped_direct._role_bindings([binding], multiple=False)


def test_mapped_cli_forwards_exact_composed_hierarchy_arguments(
        monkeypatch, capsys):
    observed = {}

    def load_mapping(path):
        observed["loaded_mapping"] = path
        return {"format": "grib2"}

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        return {"schema": "proof", "status": "PASS"}

    monkeypatch.setattr(mapped_direct, "load_mapping", load_mapping)
    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    argv = [
        "--source-format", "grib2",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/f000.grib2",
        "--input", "/source/f003.grib2",
        "--supplement", "terrain=/source/terrain-f000.grib2",
        "--supplement", "terrain=/source/terrain-f003.grib2",
        "--provenance", "terrain_provenance=/case/terrain.md",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--grib2-inventory", "/bin/grib2_inventory",
        "--grib2-dump", "/bin/grib2_dump",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "7",
        "--cpu-preprocess-bridge", "/bin/libgpuwm_preprocess_cpu.so",
        "--hierarchy-workers", "6",
    ]

    assert mapped_direct.main(argv) == 0

    assert observed["loaded_mapping"] == Path("/case/mapping.json")
    assert observed["prepare"] == {
        "composition": Path("/case/composition.json"),
        "mapping": Path("/case/mapping.json"),
        "primary_files": [
            Path("/source/f000.grib2"),
            Path("/source/f003.grib2"),
        ],
        "supplement_files": {
            "terrain": (
                Path("/source/terrain-f000.grib2"),
                Path("/source/terrain-f003.grib2"),
            ),
        },
        "provenance_files": {
            "terrain_provenance": Path("/case/terrain.md"),
        },
        "input_manifest": Path("/case/input-manifest.json"),
        "input_manifest_sha256": _DIGEST,
        "grib1_bridge": None,
        "grib2_inventory": Path("/bin/grib2_inventory"),
        "grib2_dump": Path("/bin/grib2_dump"),
        "wps_namelist": Path("/case/namelist.wps"),
        "geog_root": Path("/static/WPS_GEOG"),
        "experiment_config": Path("/case/experiment.toml"),
        "output_root": Path("/output/mapped"),
        "preprocess_backend": "cpu",
        "preprocess_workers": 7,
        "cpu_preprocess_bridge": Path("/bin/libgpuwm_preprocess_cpu.so"),
        "hierarchy_workers": 6,
    }
    assert '"status": "PASS"' in capsys.readouterr().out
