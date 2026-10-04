"""GFS native physical replacements reach initial and boundary preparation."""
from dataclasses import replace
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import gfs_direct
from gpuwm.ensemble.physical_store import NativePhysicalStore, digest_file


@pytest.fixture
def physical_route(monkeypatch, tmp_path):
    from gpuwm.io.nc_writer_bridge import unavailable_reason
    from gpuwm.netcdf_bridge import find_netcdf_bin
    reason = unavailable_reason()
    if reason or find_netcdf_bin() is None:
        pytest.skip("native physical-store writer/reader required: " + str(reason))
    from gpuwm import native_wrf_contract
    from gpuwm.experiment import load_experiment
    from gpuwm.ingest import prepared_cache
    from gpuwm.ingest.horiz import HorizontalSnapshot
    from test_gfs_chained_tree import _RecordingCacheStream
    from test_gfs_initial_perturbation import _config, _cpu_preparation, _inputs

    config = _config(tmp_path, domains=1)
    exp = load_experiment(config)
    native_subprocess_run = gfs_direct.subprocess.run
    _cpu_preparation(monkeypatch, exp)
    source_decode = gfs_direct.subprocess.run
    monkeypatch.setattr(gfs_direct.subprocess, "run", lambda command, **kwargs:
        source_decode(command, **kwargs) if str(command[0]).endswith("bridge")
        else native_subprocess_run(command, **kwargs))
    monkeypatch.setattr(prepared_cache, "PreparedCacheStream", _RecordingCacheStream)
    monkeypatch.setattr(gfs_direct, "_canonical_surface", lambda _: {})
    geometry = {"mass_shape": [3, 3], "nz": 2, "dx_m": exp.root.run.dx, "dy_m": exp.root.run.dy}
    monkeypatch.setattr(gfs_direct, "_geometry_contract", lambda *_: dict(geometry))
    monkeypatch.setattr(native_wrf_contract, "native_geometry_contract", lambda *_: dict(geometry))
    monkeypatch.setattr(gfs_direct, "native_static_export_fields", lambda fields, _: fields)
    snapshots = tuple(HorizontalSnapshot(
        valid_time=exp.start_time + timedelta(hours=hour),
        levels_hpa=np.array([900., 500.], dtype=np.float64),
        fields={"TT": np.full((2, 3, 3), 280. + hour, np.float32),
                "RH": np.full((2, 3, 3), 50. + hour, np.float32),
                "UU": np.ones((2, 3, 4), np.float32),
                "VV": np.ones((2, 4, 3), np.float32),
                "GHT": np.full((2, 3, 3), 1000., np.float32),
                "PSFC": np.full((3, 3), 100000., np.float32),
                "T2": np.full((3, 3), 280., np.float32),
                "RH2": np.full((3, 3), 50., np.float32),
                "U10": np.ones((3, 4), np.float32),
                "V10": np.ones((4, 3), np.float32),
                "SKINTEMP": np.full((3, 3), 280., np.float32),
                "SOURCE_OROGRAPHY": np.full((3, 3), 10., np.float32)})
        for hour in (0, 3))
    monkeypatch.setattr(gfs_direct, "_load_bridge_snapshots", lambda *_a, **_k: snapshots)
    initialized, mapped = [], []

    def initialize(met, *_args, **kwargs):
        initialized.append(met)
        return SimpleNamespace(state=SimpleNamespace(
            thp=met.fields["TT"].copy(), set_map_coriolis=lambda *_a, **_k: None),
            initial_perturbation={})

    def interpolate(source, *_args, **kwargs):
        mapped.append(source.valid_time)
        return source

    monkeypatch.setattr(gfs_direct, "initialize_real", initialize)
    monkeypatch.setattr(gfs_direct, "interpolate_era5_to_lambert", interpolate)
    arguments = _inputs(tmp_path, config, "inputs")

    def prepare(name, **kwargs):
        initialized.clear()
        mapped.clear()
        result = gfs_direct.prepare_gfs_wrf(
            **{**arguments, "output_root": tmp_path / name}, stock_wrf_export=False, **kwargs)
        identity = json.loads((tmp_path / name / "prepared-cache" / "header.json").read_text())["identity"]
        return result, identity, tuple(initialized), tuple(mapped)

    return prepare, snapshots, arguments, initialized


def test_gfs_capture_preserves_rh_branch_and_default_native_inputs(physical_route, tmp_path):
    prepare, snapshots, _, _ = physical_route
    ordinary, old_identity, old_inputs, old_maps = prepare("ordinary")
    captured, capture_identity, capture_inputs, capture_maps = prepare(
        "capture", physical_output_store=tmp_path / "physical")
    assert old_identity == capture_identity
    assert "ensemble_physical_output" not in ordinary
    assert "ensemble_physical_output" not in captured
    assert old_maps == capture_maps == tuple(frame.valid_time for frame in snapshots)
    store = NativePhysicalStore(tmp_path / "physical")
    assert len(store.times) == 2
    assert store.document["source"]["adapter"] == "gfs-pgrb2-0p25-direct-v1"
    for index, (before, after) in enumerate(zip(old_inputs, capture_inputs)):
        saved = store.read(index)
        assert saved.specific_humidity_authority is False
        assert not {"PRES", "SPFH", "Q2"} & saved.fields.keys()
        assert saved.fields.keys() == before.fields.keys() == after.fields.keys()
        for field in before.fields:
            assert before.fields[field].tobytes() == after.fields[field].tobytes() == saved.fields[field].tobytes()


def test_gfs_replacement_enters_before_every_native_initializer(physical_route, tmp_path):
    prepare, _, _, _ = physical_route
    _, ordinary_identity, _, _ = prepare("capture", physical_output_store=tmp_path / "physical")
    base = NativePhysicalStore(tmp_path / "physical")
    output = NativePhysicalStore(tmp_path / "changed", grid_identity=base.document["grid"],
                                source_identity=base.document["source"], field_contract=base.field_contract)
    expected = []
    for index in range(2):
        snapshot = base.read(index)
        fields = {**snapshot.fields, "TT": snapshot.fields["TT"] + np.float32(index + 1),
                  "RH": snapshot.fields["RH"] + np.float32(index + 2)}
        changed = replace(snapshot, fields=fields)
        expected.append(changed)
        output.write(changed)
    output.seal()
    _, identity, initialized, mapped = prepare("changed-prepare", physical_input_store=output.root)
    assert mapped == ()
    assert len(initialized) == 2
    for wanted, actual in zip(expected, initialized):
        assert actual.specific_humidity_authority is False
        for field in wanted.fields:
            assert actual.fields[field].tobytes() == wanted.fields[field].tobytes()
    source = dict(identity["source_identity"])
    binding = source.pop("ensemble_physical_input")
    assert source == ordinary_identity["source_identity"]
    assert binding["manifest_sha256"] == digest_file(output.manifest_path)


@pytest.mark.parametrize("drift", ["grid", "times", "source", "static"])
def test_gfs_replacement_refuses_drift_before_native_initialization(physical_route, tmp_path, drift):
    prepare, _, _, initialized = physical_route
    prepare("capture", physical_output_store=tmp_path / "physical")
    manifest = tmp_path / "physical" / "physical-store.json"
    document = json.loads(manifest.read_text())
    if drift == "grid":
        document["grid"]["dx_m"] += 1
    elif drift == "times":
        document["frames"][1]["valid_time"] = "2026-09-05T06:00:00"
    elif drift == "source":
        document["source"]["relative_humidity_convention"] = "ice"
    else:
        document["source"]["static_identity"]["fields"]["HGT_M"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match={"grid": "grid|geometry", "times": "valid times",
                                        "source": "source authority", "static": "static"}[drift]):
        prepare("refused", physical_input_store=manifest.parent)
    assert initialized == []


def test_gfs_field_contract_uses_native_rh_height_and_staggering():
    from gpuwm.ensemble.gfs_physical_contract import native_gfs_field_contract
    contract = native_gfs_field_contract({"mass_shape": [3, 4]}, evidence={"decoder": "a" * 64})
    arrays = contract["arrays"]
    renames = {"T": "TT", "U": "UU", "V": "VV"}
    expected_fields = {"field__" + renames.get(name, name)
                       for name in (*gfs_direct._THREE_D, *gfs_direct._TWO_D)}
    assert expected_fields == {name for name in arrays if name.startswith("field__")}
    assert arrays["field__RH"]["units"] == arrays["field__RH2"]["units"] == "%"
    assert not {"field__PRES", "field__SPFH", "field__Q2"} & arrays.keys()
    for name in ("field__GHT", "field__SOURCE_OROGRAPHY"):
        assert arrays[name]["source_units"] == "gpm"
        assert arrays[name]["units"] == "m"
        assert "no division" in arrays[name]["operation"]
    for name in ("field__UU", "field__U10"):
        assert arrays[name]["basis"] == "grid_x"
        assert arrays[name]["dimensions"][-2:] == ["y", "x_stag"]
    for name in ("field__VV", "field__V10"):
        assert arrays[name]["basis"] == "grid_y"
        assert arrays[name]["dimensions"][-2:] == ["y_stag", "x"]


def test_gfs_replacement_refuses_explicit_but_incompatible_humidity_units(physical_route, tmp_path):
    prepare, _, _, initialized = physical_route
    prepare("capture", physical_output_store=tmp_path / "physical")
    base = NativePhysicalStore(tmp_path / "physical")
    contract = deepcopy(base.field_contract)
    contract["arrays"]["field__RH"]["units"] = "K"
    output = NativePhysicalStore(tmp_path / "wrong-units", grid_identity=base.document["grid"],
        source_identity=base.document["source"], field_contract=contract)
    for index in range(2):
        output.write(base.read(index))
    output.seal()
    with pytest.raises(ValueError, match="field__RH units"):
        prepare("refused-units", physical_input_store=output.root)
    assert initialized == []
