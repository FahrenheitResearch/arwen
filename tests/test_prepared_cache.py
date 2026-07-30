"""CPU-only integrity tests for the prepared real-data cache container."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.ingest.prepared_cache as prepared_cache_module
from gpuwm.core.grid import BaseState, make_vertical_coord
from gpuwm.ingest.lateral_bc import (
    BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
)
from gpuwm.ingest.prepared_cache import (
    _prepared_cache_staging_path,
    _restore_lbc_mode,
    PreparedCacheCorruptError, PreparedCacheMismatchError,
    PreparedCacheReader, prepared_cache_identity,
    select_prepared_met_fields, write_prepared_cache,
)
from gpuwm.io.restart import STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS


def _fixture():
    coord = make_vertical_coord(2, hybrid_opt=0)
    base = BaseState(
        mub=np.full((2, 2), 90_000.0), p_top=10_000.0,
        pb=np.full((2, 2, 2), 50_000.0),
        alb=np.full((2, 2, 2), 0.8),
        thb=np.full((2, 2, 2), 290.0),
        phb=np.zeros((3, 2, 2)), terrain_z=np.zeros((2, 2)))
    side = SideBoundary(
        np.arange(4, dtype=np.float64).reshape(1, 2, 2),
        np.full((1, 2, 2), 0.25, dtype=np.float64))
    boundaries = LateralBoundaries((BoundaryInterval(
        0.0, 3600.0, {"u": FieldBoundary(side, side, side, side)}),),
        5, 1, 4)
    state = SimpleNamespace(u=np.arange(12, dtype=np.float32).reshape(3, 2, 2))
    for index, name in enumerate(STATE_SETUP_ARRAYS):
        setattr(state, name, np.array([index], dtype=np.float32))
    scalar_values = {
        "mub": None, "p_top": 10_000.0,
        "cf1": 1.0, "cf2": 2.0, "cf3": 3.0,
        "cfn": 4.0, "cfn1": 5.0,
        "has_msf": True, "rotational": True,
    }
    assert set(scalar_values) == set(STATE_SETUP_SCALARS)
    for name, value in scalar_values.items():
        setattr(state, name, value)
    state.lateral_boundaries = boundaries
    initial = SimpleNamespace(
        state=state, coord=coord, base=base,
        surface_pressure=np.full((2, 2), 99_000.0),
        surface_qv=np.full((2, 2), 0.01))
    surface = np.ones((2, 2), dtype=np.float32)
    met = SimpleNamespace(fields={
        "LANDSEA": surface, "SKINTEMP": 280.0 * surface,
        "SOILT": np.ones((9, 2, 2), dtype=np.float32),
        "SOILW": np.full((9, 2, 2), 0.2, dtype=np.float32),
        "T2": 279.0 * surface,
        "U10": np.ones((2, 3), dtype=np.float32),
        "V10": np.ones((3, 2), dtype=np.float32),
    })
    return initial, met, boundaries


def test_prepared_identity_serializes_per_domain_start_time():
    @dataclass(frozen=True)
    class Domain:
        start_time: datetime

    identity = prepared_cache_identity(
        bridge_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        static_cache_sha256="c" * 64,
        namelist_sha256="d" * 64,
        domain_config=Domain(datetime(2026, 7, 20, 0, 5)),
        forcing_offsets_seconds=(0, 300),
        source_identity={"adapter": "fixture"})

    assert identity["domain_config"]["start_time"] == \
        "2026-07-20T00:05:00"
    assert identity["forcing_offsets_seconds"] == [0, 300]


def test_nested_lbc_restore_opt_in_is_identity_bound_and_root_safe():
    child_identity = {
        "domain_config": {
            "parent_id": 1,
            "run": {"nested": True, "specified": False},
        },
    }
    root_identity = {
        "domain_config": {
            "parent_id": 0,
            "run": {"nested": False, "specified": True},
        },
    }

    assert _restore_lbc_mode(
        lbc_metadata=None, identity=child_identity,
        allow_nested_without_lbc=True) == "nested-parent-forced"
    assert _restore_lbc_mode(
        lbc_metadata={}, identity=root_identity,
        allow_nested_without_lbc=False) == "external"
    with pytest.raises(PreparedCacheMismatchError, match="standalone"):
        _restore_lbc_mode(
            lbc_metadata=None, identity=child_identity,
            allow_nested_without_lbc=False)
    with pytest.raises(PreparedCacheMismatchError, match="standalone"):
        _restore_lbc_mode(
            lbc_metadata=None, identity=root_identity,
            allow_nested_without_lbc=True)
    with pytest.raises(TypeError, match="must be bool"):
        _restore_lbc_mode(
            lbc_metadata={}, identity=root_identity,
            allow_nested_without_lbc=1)


def test_prepared_cache_staging_path_is_compact_and_target_independent(
        tmp_path):
    short = _prepared_cache_staging_path(
        tmp_path / "prepared", nonce="012345abcd")
    long = _prepared_cache_staging_path(
        tmp_path / ("prepared-" + "x" * 120), nonce="012345abcd")

    assert short == tmp_path / ".p-012345abcd"
    assert long == short
    assert len(long.name) == 13


@pytest.mark.parametrize(
    "nonce", ("short", "012345ABCD", "012345abcg", 1234))
def test_prepared_cache_staging_path_rejects_unsafe_explicit_nonce(
        tmp_path, nonce):
    with pytest.raises(ValueError, match="10 lowercase hex"):
        _prepared_cache_staging_path(tmp_path / "prepared", nonce=nonce)


def test_prepared_cache_paths_fit_failed_windows_parent_budget():
    parent = PureWindowsPath("C:\\" + "x" * 225)
    target = parent / "prepared-cache"
    staging = _prepared_cache_staging_path(
        target, nonce="012345abcd")

    assert len(str(parent)) == 228
    assert len(str(target)) == 243
    assert len(str(staging)) == 242
    assert len(str(staging / "a00000.npy")) == 253
    assert len(str(staging / "header.json")) == 254
    assert len(str(target / "a00000.npy")) == 254
    assert len(str(target / "header.json")) == 255


def test_prepared_cache_staging_collision_preserves_foreign_tree(
        tmp_path, monkeypatch):
    initial, met, boundaries = _fixture()
    target = tmp_path / "prepared"
    staging = tmp_path / ".p-012345abcd"
    staging.mkdir()
    sentinel = staging / "foreign.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")
    monkeypatch.setattr(
        prepared_cache_module, "_prepared_cache_staging_path",
        lambda _path: staging)

    with pytest.raises(FileExistsError):
        write_prepared_cache(
            target, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)

    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"
    assert not target.exists()


def test_prepared_cache_mid_write_failure_removes_only_owned_staging(
        tmp_path, monkeypatch):
    initial, met, boundaries = _fixture()
    target = tmp_path / "prepared"
    staging = tmp_path / ".p-012345abcd"
    monkeypatch.setattr(
        prepared_cache_module, "_prepared_cache_staging_path",
        lambda _path: staging)

    def injected_write_failure(_writer, _key, _value):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        prepared_cache_module._BundleWriter, "add",
        injected_write_failure)

    with pytest.raises(RuntimeError, match="injected write failure"):
        write_prepared_cache(
            target, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)

    assert not staging.exists()
    assert not target.exists()


def test_prepared_cache_publication_race_preserves_competing_target(
        tmp_path, monkeypatch):
    initial, met, boundaries = _fixture()
    target = tmp_path / "prepared"
    staging = tmp_path / ".p-012345abcd"
    monkeypatch.setattr(
        prepared_cache_module, "_prepared_cache_staging_path",
        lambda _path: staging)

    def competing_publish(_source, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / "foreign.txt").write_text(
            "competing publisher", encoding="utf-8")
        raise FileExistsError("injected publication race")

    monkeypatch.setattr(
        prepared_cache_module.os, "replace", competing_publish)

    with pytest.raises(FileExistsError, match="injected publication race"):
        write_prepared_cache(
            target, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)

    assert not staging.exists()
    assert (target / "foreign.txt").read_text(encoding="utf-8") == (
        "competing publisher")


def test_prepared_cache_round_trip_verifies_every_array(tmp_path):
    initial, met, boundaries = _fixture()
    identity = {"source": "abc", "config": {"nx": 2}}
    path = tmp_path / "prepared"
    receipt = write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries, metadata={"forcing_hours": [0, 1]})

    reader = PreparedCacheReader(path, expected_identity=identity)
    verified = reader.verify_all()
    assert receipt["status"] == "BUILT"
    assert verified["status"] == "PASS"
    assert verified["content_sha256"] == receipt["content_sha256"]
    assert verified["array_count"] == receipt["array_count"]
    assert verified["payload_bytes"] == receipt["payload_bytes"]
    assert all(
        spec["file"].startswith("a")
        and spec["file"].endswith(".npy")
        and len(spec["file"]) == 10
        for spec in reader.arrays.values())


def test_prepared_cache_refuses_identity_drift_and_payload_corruption(tmp_path):
    initial, met, boundaries = _fixture()
    identity = {"source": "abc"}
    path = tmp_path / "prepared"
    write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries)

    with pytest.raises(PreparedCacheMismatchError, match="identity differs"):
        PreparedCacheReader(path, expected_identity={"source": "changed"})

    reader = PreparedCacheReader(path, expected_identity=identity)
    payload = path / reader.arrays["state/u"]["file"]
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0x01
    payload.write_bytes(raw)
    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        PreparedCacheReader(
            path, expected_identity=identity).read_array("state/u")


def test_prepared_cache_never_overwrites_valid_bundle(tmp_path):
    initial, met, boundaries = _fixture()
    path = tmp_path / "prepared"
    write_prepared_cache(
        path, identity={"source": "abc"}, initial_result=initial, met=met,
        boundaries=boundaries)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_prepared_cache(
            path, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)


def test_nested_child_cache_may_omit_external_lbc_for_wrfinput_export(tmp_path):
    initial, met, _boundaries = _fixture()
    identity = {
        "domain_config": {
            "grid_id": 2,
            "parent_id": 1,
            "run": {"nested": True, "specified": False},
        },
    }
    path = tmp_path / "prepared-child"

    receipt = write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=None)
    reader = PreparedCacheReader(path, expected_identity=identity)

    assert receipt["status"] == "BUILT"
    assert reader.header["metadata"]["lbc"] is None
    assert not any(name.startswith("lbc/") for name in reader.arrays)
    assert reader.verify_all()["status"] == "PASS"


@pytest.mark.parametrize("identity", (
    {"source": "root"},
    {"domain_config": {"parent_id": 0,
                       "run": {"nested": False, "specified": True}}},
    {"domain_config": {"parent_id": 1,
                       "run": {"nested": True, "specified": True}}},
))
def test_prepared_cache_refuses_lbc_omission_without_nested_identity(
        tmp_path, identity):
    initial, met, _boundaries = _fixture()
    path = tmp_path / "invalid-child"
    with pytest.raises(ValueError, match="identity-bound nested"):
        write_prepared_cache(
            path, identity=identity, initial_result=initial, met=met,
            boundaries=None)
    assert not path.exists()


def test_prepared_cache_accepts_source_neutral_canonical_surface(tmp_path):
    initial, met, boundaries = _fixture()
    fields = dict(met.fields)
    fields.pop("SOILT")
    fields.pop("SOILW")
    met = SimpleNamespace(fields=fields)
    plane = np.ones((2, 2), dtype=np.float32)
    surface = {
        "TSK": 280.0 * plane,
        "TSLB": np.full((4, 2, 2), 279.0, dtype=np.float32),
        "SMOIS": np.full((4, 2, 2), 0.2, dtype=np.float32),
        "SH2O": np.full((4, 2, 2), 0.2, dtype=np.float32),
        "TMN": 278.0 * plane,
        "SEAICE": np.zeros((2, 2), dtype=np.float32),
        "XLAND": plane,
        "LANDMASK": plane,
        "SNOW": np.zeros((2, 2), dtype=np.float32),
        "SNOWH": np.zeros((2, 2), dtype=np.float32),
    }
    path = tmp_path / "prepared"
    identity = {"source": "era5"}
    write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries, surface=surface)
    reader = PreparedCacheReader(path, expected_identity=identity)
    assert set(reader.header["metadata"]["surface_fields"]) == set(surface)
    np.testing.assert_array_equal(
        reader.read_array("surface/TSLB"), surface["TSLB"])


def test_select_prepared_met_fields_detaches_exact_physics_contract():
    _, met, _ = _fixture()
    fields = dict(met.fields)
    fields.update({
        "SST": np.full((2, 2), 281.0, dtype=np.float32),
        "PRES": np.full((2, 2, 2), 80_000.0, dtype=np.float32),
    })
    met = SimpleNamespace(fields=fields)

    selected = select_prepared_met_fields(met)

    assert set(selected.fields) == {
        "LANDSEA", "SKINTEMP", "SOILT", "SOILW", "SST", "T2", "U10",
        "V10",
    }
    assert "PRES" not in selected.fields
    for name, value in selected.fields.items():
        assert value.flags.c_contiguous
        assert not np.shares_memory(value, fields[name])
        np.testing.assert_array_equal(value, fields[name])
    with pytest.raises(TypeError):
        selected.fields["PRES"] = fields["PRES"]

    fields["T2"][...] = -1.0
    np.testing.assert_array_equal(
        selected.fields["T2"], np.full((2, 2), 279.0, dtype=np.float32))


def test_select_prepared_met_fields_keeps_legacy_soil_only_when_required():
    _, met, _ = _fixture()
    canonical_surface = {"placeholder": object()}

    selected = select_prepared_met_fields(
        met, surface=canonical_surface)

    assert "SOILT" not in selected.fields
    assert "SOILW" not in selected.fields
