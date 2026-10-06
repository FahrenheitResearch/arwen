"""Parameter identity binds prepared and every physics checkpoint writer."""

import json
from types import SimpleNamespace

import pytest

from gpuwm import physics_params as pp
from gpuwm import prepared_single_domain_forecast as single
from gpuwm.io import restart
from test_restart import _cfg, _fill_setup, _shim_driver_state


@pytest.fixture(autouse=True)
def _binding(monkeypatch):
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    pp.reset_for_tests()
    yield
    pp.reset_for_tests()


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _prepared_inputs(mode, pset=None):
    inputs = SimpleNamespace(
        source="mapped", cache_reader=SimpleNamespace(content_sha256="cache"),
        file_sha256={name: name + "-digest" for name in (
            "experiment_config", "proof", "cache_header", "source_manifest", "static")},
        experiment=SimpleNamespace(physics_params=pset))
    if mode == "head":
        inputs.stream_head = {"head_sha256": "head", "basis": {}}
    elif mode == "as-posted-head":
        inputs.stream_head = {"head_sha256": "head", "basis": {"as_posted": {}}}
    return inputs


@pytest.mark.parametrize("mode", ("sealed", "head", "as-posted-head"))
def test_prepared_single_set_binds_without_changing_preparation_authorities(mode):
    runtime = {"commit": "runtime"}
    inputs = _prepared_inputs(mode)
    default = single._single_checkpoint_identity(inputs, runtime)
    assert "physics_params" not in default
    expected = {
        "schema": "gpuwm.prepared-single-checkpoint.v1",
        "source": inputs.source,
        "runtime_source_identity": runtime,
    }
    if mode == "sealed":
        expected.update(prepared_content_sha256="cache",
                        authority_sha256=dict(inputs.file_sha256))
    else:
        ignored = {"proof", "cache_header", "prepared_head"}
        if mode == "as-posted-head":
            ignored.add("source_manifest")
        expected.update(prepared_head_sha256="head", authority_sha256={
            name: value for name, value in inputs.file_sha256.items()
            if name not in ignored})
    assert _bytes(default) == _bytes(expected)
    pset = pp.make_set("member", {"mynn.prandtl": 0.8})
    member = single._single_checkpoint_identity(_prepared_inputs(mode, pset), runtime)
    assert member.pop("physics_params") == pp.document(pset)
    assert _bytes(member) == _bytes(default)
    other = pp.make_set("member", {"mynn.prandtl": 0.9})
    assert _bytes(single._single_checkpoint_identity(_prepared_inputs(mode, pset), runtime)) != (
        _bytes(single._single_checkpoint_identity(_prepared_inputs(mode, other), runtime)))


def _writer_headers(cfg, monkeypatch, directory):
    """Write the same host-backed state through all three actual writers."""
    from tilestream import checkpoint, physics_inventory, restart_stream

    directory.mkdir()
    state, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    paths = {"resident": restart.write_restart(directory / "resident.npz", state, cfg)}
    store = {key: value.copy() for key, value in
             physics_inventory.carrier_manifest(state).items()}
    scalars = physics_inventory.carrier_scalars(state)
    paths["streamed"] = restart_stream.write_streamed_restart(
        directory / "streamed.npz", store, cfg, scalars=scalars,
        setup=restart_stream.capture_domain_setup(state),
        template_state=state, check_pinned=False).path
    paths["store"] = checkpoint.write_store_restart(
        directory / "store.npz", store, scalars,
        checkpoint.DomainSetup.capture(state, cfg), cfg)
    return {road: restart.read_restart_header(path) for road, path in paths.items()}, paths


def test_literal_only_set_is_in_every_writer_and_default_setup_bytes_are_inert(
        monkeypatch, tmp_path):
    cfg = _cfg(bl_pbl_physics=5)
    defaults, _ = _writer_headers(cfg, monkeypatch, tmp_path / "default")
    pset = pp.make_set("member", {"mynn.prandtl": 0.8})
    pp.declare(pset, source="checkpoint")
    members, _ = _writer_headers(cfg, monkeypatch, tmp_path / "member")
    for road in defaults:
        default = defaults[road]["physics_setup"]
        member = dict(members[road]["physics_setup"])
        assert "physics_params" not in default
        assert member.pop("physics_params") == pp.document(pset)
        assert _bytes(member) == _bytes(default), road
        assert members[road]["physics_setup_fingerprint"] != (
            defaults[road]["physics_setup_fingerprint"]), road
    assert len({_bytes(header["physics_setup"]) for header in members.values()}) == 1
    pp.reset_for_tests()
    again, _ = _writer_headers(cfg, monkeypatch, tmp_path / "default-again")
    for road in defaults:
        assert _bytes(again[road]["physics_setup"]) == _bytes(defaults[road]["physics_setup"])
        assert again[road]["physics_setup_fingerprint"] == defaults[road]["physics_setup_fingerprint"]


@pytest.mark.parametrize("transport", ("resident", "streamed", "store"))
def test_literal_only_set_change_refuses_restore_before_array_mutation(
        monkeypatch, tmp_path, transport):
    from tilestream import checkpoint, physics_inventory, restart_stream
    cfg = _cfg(bl_pbl_physics=5)
    pp.declare(pp.make_set("member", {"mynn.prandtl": 0.8}), source="checkpoint")
    _, paths = _writer_headers(cfg, monkeypatch, tmp_path / "member")
    pp.reset_for_tests()
    pp.declare(pp.make_set("member", {"mynn.prandtl": 0.9}), source="resume")
    target, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(target)
    before = target.u.tobytes()
    store = {key: value.copy() for key, value in
             physics_inventory.carrier_manifest(target).items()}
    store_before = {key: value.tobytes() for key, value in store.items()}
    with pytest.raises((restart.RestartMismatchError, restart_stream.RestartRefused),
                       match="physics"):
        if transport == "resident":
            restart.restore_restart(paths[transport], target, cfg)
        elif transport == "streamed":
            restart_stream.read_streamed_restart(
                paths[transport], store, cfg,
                setup=restart_stream.capture_domain_setup(target),
                template_state=target, scalars={})
        else:
            checkpoint.read_store_restart(
                paths[transport], store, checkpoint.DomainSetup.capture(target, cfg), cfg)
    assert target.u.tobytes() == before
    assert {key: value.tobytes() for key, value in store.items()} == store_before
