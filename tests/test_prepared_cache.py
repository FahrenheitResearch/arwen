"""CPU-only integrity tests for the prepared real-data cache container."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from types import MappingProxyType, SimpleNamespace

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
    PreparedCacheReader, extend_prepared_cache,
    prepared_cache_identity,
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


def _extension_identity(*, source_hours, model_start, domain_start,
                        bridge, source_manifest):
    model_hours = list(range(len(source_hours)))
    namelist_sha256 = "d" * 64 if max(source_hours) == 1 else "e" * 64
    return {
        "bridge_manifest_sha256": bridge,
        "source_manifest_sha256": source_manifest,
        "static_cache_sha256": "c" * 64,
        "namelist_sha256": namelist_sha256,
        "namelist_extension_invariant": {
            "schema": "gpuwm-namelist-extension-invariant-v1",
            "sha256": "1" * 64,
        },
        "domain_config": {
            "start_time": domain_start.isoformat(),
            "run": {
                "run_seconds": float(len(model_hours) - 1) * 3600.0,
                "specified": True,
                "nested": False,
            },
        },
        "forcing_hours": model_hours,
        "source_identity": {
            "adapter": "fixture",
            "source_cycle": datetime(2026, 7, 20).isoformat(),
            "model_start_time": model_start.isoformat(),
            "source_forecast_hours": list(source_hours),
            "model_forcing_hours": model_hours,
        },
    }


def _suffix_fixture(*, seam_delta=0.0):
    initial, met, old = _fixture()
    old_side = old.intervals[0].fields["u"].west
    endpoint = old_side.value + 3600.0 * old_side.tendency + seam_delta
    suffix_side = SideBoundary(
        endpoint, np.full_like(endpoint, 0.5, dtype=np.float64))
    suffix = LateralBoundaries((BoundaryInterval(
        0.0, 3600.0,
        {"u": FieldBoundary(
            suffix_side, suffix_side, suffix_side, suffix_side)}),),
        5, 1, 4)
    initial.state.lateral_boundaries = suffix
    return initial, met, suffix


def _manifest_extension(prior_identity, new_identity):
    return {
        "schema": "gpuwm-source-manifest-prefix-extension-v1",
        "predecessor_sha256": prior_identity["source_manifest_sha256"],
        "extended_sha256": new_identity["source_manifest_sha256"],
        "old_source_forecast_hours": [0, 1],
        "new_source_forecast_hours": [0, 1, 2],
        "suffix_source_forecast_hours": [1, 2],
        "retained_entries": 4,
        "added_entries": ["atmosphere-f02", "surface-f02"],
    }


def _bridge_extension(prior_identity, suffix_identity, new_identity):
    return {
        "schema": "gpuwm-bridge-manifest-prefix-extension-v1",
        "predecessor_sha256": prior_identity["bridge_manifest_sha256"],
        "suffix_sha256": suffix_identity["bridge_manifest_sha256"],
        "extended_sha256": new_identity["bridge_manifest_sha256"],
        "old_source_forecast_hours": [0, 1],
        "new_source_forecast_hours": [0, 1, 2],
        "suffix_source_forecast_hours": [1, 2],
        "retained_entries": 48,
        "added_entries": ["atmosphere-f02", "soil-f02"],
    }


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


def test_default_prepared_cache_bytes_ignore_disabled_seal_option(tmp_path):
    initial, met, boundaries = _fixture()
    first = tmp_path / "implicit-default"
    second = tmp_path / "explicit-default"

    one = write_prepared_cache(
        first, identity={"source": "unchanged"}, initial_result=initial,
        met=met, boundaries=boundaries)
    two = write_prepared_cache(
        second, identity={"source": "unchanged"}, initial_result=initial,
        met=met, boundaries=boundaries, sealed_forcing_extension=False)

    assert one["content_sha256"] == two["content_sha256"]
    first_header = PreparedCacheReader(
        first, expected_identity={"source": "unchanged"}).header
    second_header = PreparedCacheReader(
        second, expected_identity={"source": "unchanged"}).header
    for key in ("schema", "status", "identity", "metadata", "arrays",
                "content_sha256", "payload_bytes"):
        assert first_header[key] == second_header[key]
    assert "forcing_extension_mode" not in first_header["metadata"]
    for key, spec in first_header["arrays"].items():
        other = second_header["arrays"][key]
        assert (first / spec["file"]).read_bytes() == \
            (second / other["file"]).read_bytes()


def test_prepared_cache_extension_reuses_prefix_and_appends_nonzero_hour(
        tmp_path):
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    composite = "9" * 64
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge=composite, source_manifest="f" * 64)
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture()
    prior = tmp_path / "prior"
    suffix = tmp_path / "suffix"
    output = tmp_path / "extended"
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)

    receipt = extend_prepared_cache(
        output, predecessor=prior, suffix=suffix,
        identity=extended_identity, metadata={"forcing_hours": [0, 1, 2]},
        source_manifest_extension=_manifest_extension(
            prior_identity, extended_identity),
        bridge_manifest_extension=_bridge_extension(
            prior_identity, suffix_identity, extended_identity))

    reader = PreparedCacheReader(output, expected_identity=extended_identity)
    assert reader.verify_all()["status"] == "PASS"
    assert len(reader.header["metadata"]["lbc"]["intervals"]) == 2
    assert receipt["appended_interval"] == [3600.0, 7200.0]
    assert receipt["bridge_manifest_sha256"] == composite
    prior_reader = PreparedCacheReader(prior, expected_identity=prior_identity)
    for key, old_spec in prior_reader.arrays.items():
        new_spec = reader.arrays[key]
        assert Path(prior / old_spec["file"]).samefile(
            output / new_spec["file"])
    np.testing.assert_array_equal(
        reader.read_array("lbc/0/u/west/value"),
        prior_reader.read_array("lbc/0/u/west/value"))
    np.testing.assert_array_equal(
        reader.read_array("lbc/1/u/west/value"),
        PreparedCacheReader(
            suffix, expected_identity=suffix_identity).read_array(
                "lbc/0/u/west/value"))


def test_prepared_cache_extension_refuses_changed_seam_without_output(
        tmp_path):
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge="9" * 64,
        source_manifest="f" * 64)
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture(
        seam_delta=1.0)
    prior, suffix, output = (
        tmp_path / "prior", tmp_path / "suffix", tmp_path / "extended")
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)

    with pytest.raises(PreparedCacheMismatchError, match="shared endpoint"):
        extend_prepared_cache(
            output, predecessor=prior, suffix=suffix,
            identity=extended_identity, metadata={},
            source_manifest_extension=_manifest_extension(
                prior_identity, extended_identity),
            bridge_manifest_extension=_bridge_extension(
                prior_identity, suffix_identity, extended_identity))
    assert not output.exists()


def test_prepared_cache_extension_refuses_changed_namelist_invariant(
        tmp_path):
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge="9" * 64, source_manifest="f" * 64)
    extended_identity["namelist_extension_invariant"]["sha256"] = "2" * 64
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture()
    prior, suffix, output = (
        tmp_path / "prior", tmp_path / "suffix", tmp_path / "extended")
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)

    with pytest.raises(
            PreparedCacheMismatchError, match="immutable namelist fields"):
        extend_prepared_cache(
            output, predecessor=prior, suffix=suffix,
            identity=extended_identity, metadata={},
            source_manifest_extension=_manifest_extension(
                prior_identity, extended_identity),
            bridge_manifest_extension=_bridge_extension(
                prior_identity, suffix_identity, extended_identity))
    assert not output.exists()


def test_prepared_cache_extension_refuses_predecessor_mutation(tmp_path):
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge="9" * 64,
        source_manifest="f" * 64)
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture()
    prior, suffix, output = (
        tmp_path / "prior", tmp_path / "suffix", tmp_path / "extended")
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)
    prior_reader = PreparedCacheReader(prior, expected_identity=prior_identity)
    payload = prior / prior_reader.arrays["state/u"]["file"]
    array = np.load(payload, allow_pickle=False)
    array.flat[0] += 1.0
    with payload.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)

    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        extend_prepared_cache(
            output, predecessor=prior, suffix=suffix,
            identity=extended_identity, metadata={},
            source_manifest_extension=_manifest_extension(
                prior_identity, extended_identity),
            bridge_manifest_extension=_bridge_extension(
                prior_identity, suffix_identity, extended_identity))
    assert not output.exists()


def test_prepared_cache_extension_rechecks_hardlinked_stage_for_toctou(
        tmp_path, monkeypatch):
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge="9" * 64, source_manifest="f" * 64)
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture()
    prior, suffix, output = (
        tmp_path / "prior", tmp_path / "suffix", tmp_path / "extended")
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)
    original = prepared_cache_module._BundleWriter.link_verified
    injected = False

    def mutate_after_link(writer, key, reader):
        nonlocal injected
        original(writer, key, reader)
        if key == "state/u" and not injected:
            injected = True
            payload = reader.path / reader.arrays[key]["file"]
            array = np.load(payload, allow_pickle=False)
            array.flat[-1] += 1.0
            with payload.open("wb") as stream_handle:
                np.save(stream_handle, array, allow_pickle=False)

    monkeypatch.setattr(
        prepared_cache_module._BundleWriter, "link_verified",
        mutate_after_link)

    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        extend_prepared_cache(
            output, predecessor=prior, suffix=suffix,
            identity=extended_identity, metadata={},
            source_manifest_extension=_manifest_extension(
                prior_identity, extended_identity),
            bridge_manifest_extension=_bridge_extension(
                prior_identity, suffix_identity, extended_identity))
    assert injected
    assert not output.exists()


def test_prepared_cache_extension_refuses_a_gap(tmp_path):
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge="9" * 64, source_manifest="f" * 64)
    extended_identity["forcing_hours"] = [0, 1, 3]
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture()
    prior, suffix, output = (
        tmp_path / "prior", tmp_path / "suffix", tmp_path / "extended")
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)

    with pytest.raises(PreparedCacheMismatchError, match="exactly one"):
        extend_prepared_cache(
            output, predecessor=prior, suffix=suffix,
            identity=extended_identity, metadata={},
            source_manifest_extension=_manifest_extension(
                prior_identity, extended_identity),
            bridge_manifest_extension=_bridge_extension(
                prior_identity, suffix_identity, extended_identity))
    assert not output.exists()


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

    # The refusal names the release that wrote the cache and the exact
    # fields, rather than asserting that something differs: the old
    # wording sent a node-7 pilot to their experiment TOML after a
    # package upgrade had changed the identity document.
    with pytest.raises(PreparedCacheMismatchError,
                       match="these identity fields differ: source"):
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


def test_child_cache_writes_a_restored_mappingproxy_hydrometeor_record(
        tmp_path):
    """The exact v1.3.1 crash, reproduced through the type that caused it.

    ``restore_prepared_cache`` hands back
    ``CachedInitialResult.hydrometeor_initialization`` as a
    ``MappingProxyType`` on purpose -- a restored document must not be
    mutable.  ``write_prepared_cache`` then copies that record into the
    child cache it derives, through ``_json_copy -> _canonical ->
    json.dumps``, which serializes ``dict`` and nothing that merely
    behaves like one: "TypeError: Object of type mappingproxy is not JSON
    serializable", from inside a nested hierarchy publication, naming no
    field.  Nesting one proxy inside another is deliberate: a fix that
    only unwrapped the top level would still crash on the real payload.
    """

    initial, met, boundaries = _fixture()
    record = MappingProxyType({
        "schema": "gpuwm-hrrr-microphysics-initialization-v3",
        "state_source_absent_fields": MappingProxyType({
            "qnr": MappingProxyType({"expected_float32": 0.0}),
        }),
        "source_mass_fields": ("QC", "QR"),
    })
    initial = SimpleNamespace(
        **vars(initial), hydrometeor_initialization=record)
    identity = {
        "domain_config": {
            "grid_id": 2,
            "parent_id": 1,
            "run": {"nested": True, "specified": False},
        },
    }
    path = tmp_path / "prepared-child-mappingproxy"

    receipt = write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=None)
    reader = PreparedCacheReader(path, expected_identity=identity)

    assert receipt["status"] == "BUILT"
    stored = reader.header["metadata"]["hydrometeor_initialization"]
    assert stored == {
        "schema": "gpuwm-hrrr-microphysics-initialization-v3",
        "state_source_absent_fields": {"qnr": {"expected_float32": 0.0}},
        # Tuples normalize to lists exactly as they always did.
        "source_mass_fields": ["QC", "QR"],
    }
    # A proxy and its underlying mapping must hash the same, or the same
    # prepared state would carry two content digests depending on which
    # object the caller happened to hold.
    assert prepared_cache_module._canonical(record) \
        == prepared_cache_module._canonical(dict(stored))
    # An unordered container still has no canonical serialization.
    with pytest.raises(TypeError, match="cannot contain a set"):
        prepared_cache_module._canonical({"species": {"QC", "QR"}})


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


# ---------------------------------------------------------------------------
# V-12: a package upgrade must not make every existing prepared tree
# unrunnable, and must never blame the user's experiment file for it
# ---------------------------------------------------------------------------
# v1.1.0 gave every domain an optional per-domain `start_time` for
# staggered nest starts.  The prepared-cache identity is compared by
# strict equality, and a v1.0.1 header was serialized before the field
# existed, so after upgrading the wheel EVERY prepared tree in the field
# refused with "d01 cache domain config differs from experiment" -- a
# sentence naming the experiment TOML, which was innocent.  A node-7
# validation run diffed the two documents on a real preserved 1.0.1
# tree: eleven cached top-level keys against twelve live ones, exactly
# one added key, and zero value differences among the eleven shared keys
# or the ~110 `run` fields.  These tests mirror that shape.


def _live_experiment():
    from gpuwm.experiment import load_experiment

    return load_experiment(
        Path(__file__).parents[1] / "configs"
        / "gfs_wrf_hierarchy_proof.toml")


def _live_domain_identity():
    """The live 12-key identity, from the shipped two-domain config."""

    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity

    return prepared_domain_config_identity(_live_experiment().root)


def _undelayed():
    """What "no delayed start" looks like for this experiment."""

    from gpuwm.ingest.prepared_cache import undelayed_identity_defaults

    return undelayed_identity_defaults(_live_experiment())


def test_a_v101_shape_header_still_binds_after_the_upgrade():
    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    assert len(live) == 12 and "start_time" in live
    cached = {key: value for key, value in live.items() if key != "start_time"}
    assert len(cached) == 11

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert differing == []
    assert tolerated == ["start_time"]


def test_a_field_absent_from_the_header_but_IN_USE_still_refuses():
    """The narrowness is what makes tolerating the absence honest."""

    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    live["start_time"] = "2026-07-20T03:00:00"
    cached = {key: value for key, value in live.items() if key != "start_time"}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == []
    assert differing == ["start_time"]


def test_a_real_configuration_change_is_still_refused():
    """Tolerance is about absent fields, never about differing values."""

    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    cached = {key: value for key, value in live.items() if key != "start_time"}
    cached["run"] = {**cached["run"], "nx": int(cached["run"]["nx"]) + 1}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == ["start_time"]
    assert differing == ["run.nx"]


def test_a_header_from_a_NEWER_gpuwm_is_refused_not_tolerated():
    """Absence in the live document is skew in the other direction."""

    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    cached = {**live, "a_field_this_build_does_not_have": 1}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == []
    assert differing == ["a_field_this_build_does_not_have"]


def test_only_the_domain_config_is_default_tolerant():
    """Every other identity member is a hash of bytes and stays strict."""

    from gpuwm.ingest.prepared_cache import compare_prepared_identity

    live = _live_domain_identity()
    expected = {
        "domain_config": live,
        "namelist_sha256": "a" * 64,
        "static_cache_sha256": "b" * 64,
    }
    cached = {
        "domain_config": {k: v for k, v in live.items() if k != "start_time"},
        "namelist_sha256": "a" * 64,
        "static_cache_sha256": "b" * 64,
    }
    assert compare_prepared_identity(
        cached, expected, not_in_use=_undelayed()) == (["start_time"], [])

    # A missing digest is not schema growth; it is a different cache.
    del cached["static_cache_sha256"]
    tolerated, differing = compare_prepared_identity(
        cached, expected, not_in_use=_undelayed())
    assert differing == ["static_cache_sha256"]


def test_the_refusal_names_the_versions_and_the_fields():
    """Never again a message that blames the experiment file."""

    from gpuwm import __version__
    from gpuwm.ingest.prepared_cache import (
        CACHE_WRITER_KEY, UNSTAMPED_WRITER, prepared_identity_refusal,
    )

    unstamped = prepared_identity_refusal(
        subject="d01 prepared cache", header={},
        differing=["run.nx", "start_time"])
    assert "d01 prepared cache" in unstamped
    assert UNSTAMPED_WRITER in unstamped
    assert f"gpuwm {__version__}" in unstamped
    assert "run.nx, start_time" in unstamped
    # It never points at the experiment file for a package difference.
    assert "differs from experiment" not in unstamped

    stamped = prepared_identity_refusal(
        subject="d01 prepared cache",
        header={CACHE_WRITER_KEY: {"gpuwm_version": "1.0.1"}},
        differing=["run.nx"], re_prepare="rw-wps --source gfs ...")
    assert "prepared by 1.0.1" in stamped
    assert "Re-prepare it with: rw-wps --source gfs ..." in stamped


def test_a_cache_written_now_stamps_the_release_that_wrote_it():
    """So the next schema change can say which release wrote the bundle."""

    from gpuwm import __version__
    from gpuwm.ingest.prepared_cache import (
        CACHE_WRITER_KEY, cache_writer_version,
    )

    assert cache_writer_version({}) != __version__
    assert cache_writer_version(
        {CACHE_WRITER_KEY: {"gpuwm_version": __version__}}) == __version__
