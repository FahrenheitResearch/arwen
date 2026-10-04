"""The native physical-store seam binds source bytes and precedes real setup."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import hrrr_single_domain_benchmark as benchmark
from tools.prepare_hrrr_wrf import _reuse_physical_base_bridge


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _base(tmp_path):
    base = tmp_path / "base"
    bridge = base / "native/native-bridge"
    field_sha = _write(bridge / "frame.bin", b"native field words")
    bridge_sha = _write(bridge / "SHA256SUMS", f"{field_sha}  frame.bin\n".encode())
    static = tmp_path / "static.npz"
    static_sha = _write(static, b"static source bytes")
    identity = {"source_manifest_sha256": "a" * 64,
                "static_cache_sha256": static_sha,
                "bridge_manifest_sha256": bridge_sha}
    from gpuwm.ingest.prepared_cache import PREPARED_CACHE_SCHEMA, _canonical
    cache = base / "native/prepared-cache"
    cache.mkdir(parents=True)
    array = np.asarray([1.0], np.float32)
    np.save(cache / "a00000.npy", array)
    arrays = {"state/test": {"file": "a00000.npy", "shape": [1], "dtype": "<f4",
                              "nbytes": 4, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}}
    basis = {"schema": PREPARED_CACHE_SCHEMA, "identity": identity,
             "metadata": {}, "arrays": arrays, "payload_bytes": 4}
    header = {**basis, "status": "READY",
              "content_sha256": hashlib.sha256(_canonical(basis).encode()).hexdigest()}
    _write(cache / "header.json", json.dumps(header).encode())
    return base, static, bridge_sha


def test_reused_bridge_preserves_exact_bytes_and_refuses_changed_base(tmp_path):
    base, static, bridge_sha = _base(tmp_path)
    output = tmp_path / "reused"
    assert _reuse_physical_base_bridge(base, output, source_manifest_sha256="a" * 64,
                                      static_cache=static) == bridge_sha
    assert (output / "frame.bin").read_bytes() == b"native field words"
    with pytest.raises(ValueError, match="source manifest"):
        _reuse_physical_base_bridge(base, tmp_path / "wrong", source_manifest_sha256="b" * 64,
                                   static_cache=static)
    (base / "native/native-bridge/frame.bin").write_bytes(b"changed native field words")
    with pytest.raises(ValueError):
        _reuse_physical_base_bridge(base, tmp_path / "corrupt", source_manifest_sha256="a" * 64,
                                   static_cache=static)
    assert not (tmp_path / "corrupt").exists()


def test_physical_snapshot_is_consumed_and_captured_before_native_real(monkeypatch, tmp_path):
    from gpuwm.ingest import real
    from gpuwm.core import grid as grid_module
    events = []
    time = datetime(2024, 1, 1)
    met = SimpleNamespace(valid_time=time, levels_hpa=np.arange(1., 51.),
                          fields={"TT": np.ones((50, 3, 4), np.float32)})
    manifest = tmp_path / "physical-store.json"
    manifest.write_text("verified manifest")
    source = SimpleNamespace(read=lambda index: met, manifest_path=manifest)
    target = SimpleNamespace(write=lambda value: events.append(("capture", value)))
    preprocess = SimpleNamespace(receipt=lambda: {"backend": "cpu", "workers": 2})
    cfg = SimpleNamespace(nz=2, hybrid_opt=2, etac=.2)
    dc = SimpleNamespace(run=cfg)
    static = {name: np.ones((3, 4)) for name in
              ("HGT_M", "LANDMASK", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA")}
    result = SimpleNamespace(state=SimpleNamespace(set_map_coriolis=lambda *args, **kwargs: None))
    monkeypatch.setattr(benchmark, "_map_snapshot", lambda *args, **kwargs: pytest.fail("physical input remapped"))
    monkeypatch.setattr(grid_module, "make_vertical_coord", lambda *args, **kwargs: object())
    def initialize(snapshot, *args, **kwargs):
        events.append(("initialize", snapshot))
        assert kwargs["preprocess_backend"] is preprocess
        return result
    monkeypatch.setattr(real, "initialize_real", initialize)
    output = benchmark._initialize_state(
        SimpleNamespace(valid_time=time), dc, object(), static, np.asarray([1, .5, 0]), {},
        p_top=5000, preprocess_backend=preprocess, state_backend="preprocess",
        physical_input=source, physical_output=target)
    assert output[0] is result and output[1] is met
    assert events == [("capture", met), ("initialize", met)]
    source.read = lambda index: SimpleNamespace(valid_time=time + timedelta(hours=1))
    with pytest.raises(ValueError, match="valid time"):
        benchmark._initialize_state(
            SimpleNamespace(valid_time=time), dc, object(), static, np.asarray([1, .5, 0]), {},
            p_top=5000, preprocess_backend=preprocess, physical_input=source)


def test_public_parsers_preserve_native_physical_store_paths():
    from gpuwm.source_cli import _parser, _hrrr_command
    from tools.prepare_hrrr_wrf import _parser as native_parser
    args = _parser().parse_args([
        "--source", "hrrr", "--physical-input-store", "member-store",
        "--physical-output-store", "captured", "--physical-base-prepared", "base"])
    command = _hrrr_command(args)
    for flag, expected in (("--physical-input-store", "member-store"),
                           ("--physical-output-store", "captured"),
                           ("--physical-base-prepared", "base")):
        assert command[command.index(flag) + 1] == expected
    parsed = native_parser().parse_args(command[2:])
    assert parsed.physical_input_store == Path("member-store")
    assert parsed.physical_base_prepared == Path("base")


def test_mapped_cli_reuses_the_supplied_static_pair_and_rejects_half_pair():
    from gpuwm.source_cli import _parser, _mapped_command, _required_mapped_args
    args = _parser().parse_args([
        "--source", "mapped", "--source-format", "grib2",
        "--composition", "composition.json", "--mapping", "mapping.json",
        "--input", "input.grib2", "--supplement", "static=donor.grib2",
        "--provenance", "source=provenance.json", "--source-manifest", "inputs.json",
        "--source-manifest-sha256", "a" * 64, "--wps-namelist", "namelist.wps",
        "--geog-root", "geog", "--experiment-config", "experiment.toml",
        "--output-root", "prepared", "--static-input", "common-static.npz",
        "--static-receipt", "common-static.json", "--physical-output-store", "captured"])
    assert _required_mapped_args(args) == []
    command = _mapped_command(args)
    assert command[command.index("--static-input") + 1] == "common-static.npz"
    assert command[command.index("--static-receipt") + 1] == "common-static.json"
    assert command[command.index("--physical-output-store") + 1] == "captured"
    args.static_receipt = None
    assert "--static-input and --static-receipt must be supplied together" in _required_mapped_args(args)


def test_physical_replay_keeps_applied_overlay_and_rechecks_its_bytes(tmp_path):
    from gpuwm.ingest.water_overlay import WaterTemperatureOverlay, overlay_file_identity, WaterOverlayError
    path = tmp_path / "overlay.nc"
    path.write_bytes(b"the already-loaded overlay payload")
    binding = overlay_file_identity(path)
    overlay = WaterTemperatureOverlay(
        path=path, source_format="netcdf", variable="sst", declared_units="K",
        latitude=np.asarray([0., 1.]), longitude=np.asarray([0., 1.]),
        temperature_k=np.full((2, 2), 280.), valid=np.ones((2, 2), bool))
    # The reused native bridge supplies only time metadata. Mapped weather
    # arrays are supplied separately by the verified physical input store.
    placeholder = SimpleNamespace(valid_time=datetime(2024, 1, 1))
    raw = lambda hour: placeholder
    physical = object()
    acquire, sequence = benchmark._overlay_acquirer(
        raw, (0, 1), overlay, binding, workers=1, physical_input=physical)
    assert acquire(0) is placeholder and sequence is None
    benchmark._verify_preparation_overlay(sequence, physical_input=physical, binding=binding)
    # This negative control reaches the original reported AttributeError:
    # replay's time placeholder must never enter the source-grid overlay.
    broken, _ = benchmark._overlay_acquirer(raw, (0, 1), overlay, binding, workers=1)
    with pytest.raises(AttributeError, match="fields"):
        broken(0)
    path.write_bytes(b"changed overlay payload")
    with pytest.raises(WaterOverlayError, match="changed during physical replay"):
        benchmark._verify_preparation_overlay(sequence, physical_input=physical, binding=binding)
