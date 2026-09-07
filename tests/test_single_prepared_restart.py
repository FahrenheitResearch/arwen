"""One-root prepared restart uses the canonical strict checkpoint transport."""
from datetime import datetime, timedelta
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import prepared_single_domain_forecast as runner, stage_cli
from gpuwm.cli import main as cli_main
from gpuwm.io import restart
from test_restart import _sealed_tree_fixture
from test_stage_seams import _authority, _single_domain_bundle


def _inputs():
    return SimpleNamespace(source="mapped", cache_reader=SimpleNamespace(content_sha256="payload"),
        file_sha256={key: key + "-digest" for key in (
            "experiment_config", "wps_namelist", "proof", "source_manifest", "cache_header",
            "static", "geometry_receipt", "wrf_direct_contract", "mapping")})


def _one_root(monkeypatch, *, seed=31, inputs=None):
    model, start = _sealed_tree_fixture(monkeypatch, forcing_count=2,
                                       run_seconds=7200., payload_seed=seed)
    # A current specified-boundary state has zero diabatic heating in its fixed ring.
    state = model.root.state
    width = model.root.cfg.run.spec_zone
    state.h_diabatic[:, :width, :] = 0
    state.h_diabatic[:, -width:, :] = 0
    state.h_diabatic[:, :, :width] = 0
    state.h_diabatic[:, :, -width:] = 0
    model.root.children.clear()
    model.root.grid = object()
    model.nodes_by_grid_id.pop(2)
    model.walk_parent_first = lambda: iter((model.root,))
    identity = runner._single_checkpoint_identity(inputs or _inputs(), {"git_commit": "source"})
    model._experiment_fingerprint_components = identity
    model.experiment_fingerprint = hashlib.sha256(runner._canonical(identity).encode()).hexdigest()
    return model, start


@pytest.mark.parametrize("store_direct", [False, True])
def test_prepared_root_and_tile_share_the_actual_boundary_clock(monkeypatch, store_direct):
    from gpuwm.core import streaming
    from gpuwm.ingest import lateral_bc

    model, _ = _one_root(monkeypatch)
    original = model.root
    from test_davies_clock_bind import _root_clock
    original.clock = _root_clock(dt_s=20, lbc_interval_s=3600, run_s=7200)
    state = original.state
    state._lateral_boundary_device.clock = None
    if store_direct:
        # A store template has no full-domain external boundary mirror.
        del state._lateral_boundary_device
    node = runner._single_prepared_root(
        original.cfg, original.grid, state, original.clock,
        store_direct=store_direct)
    assert node.state is state and node.clock is original.clock
    assert node.cfg is original.cfg and node.grid is original.grid
    if not store_direct:
        assert state._lateral_boundary_device.clock is node.clock
        assert restart.root_external_lbc_clock_identity(state, node.cfg.run) == "wrf-postincrement-v1"

    # Only upload/allocation is replaced; the real tile hook and binder run.
    def attach(tile, table):
        tile.lateral_boundaries = table
        tile._lateral_boundary_device = SimpleNamespace(
            clock=None, rolling=False, streaming_external=True,
            active_host_interval_id=0)

    monkeypatch.setattr(lateral_bc, "attach_streaming_lateral_boundaries", attach)
    kwargs = ({"external_clock": node.clock} if store_direct
              else {"domain_state": node.state})
    hook = streaming.make_tile_hook(["first", "second"], **kwargs)
    tile = SimpleNamespace()
    hook(tile, None, 0, None)
    assert tile._lateral_boundary_device.clock is node.clock
    node.clock.mark_force()
    node.clock.prepare_step()
    hook(tile, None, 1, None)
    assert tile._lateral_boundary_device.clock is node.clock
    assert tile._lateral_boundary_device.clock.dtbc_launch_fp32 == node.clock.dt_fp32
    assert tile.lateral_boundaries == "second"


def test_prepared_resident_root_requires_its_real_boundary_attachment(monkeypatch):
    model, _ = _one_root(monkeypatch)
    node = model.root
    del node.state._lateral_boundary_device
    with pytest.raises(RuntimeError, match="external lateral boundaries must be attached first"):
        runner._single_prepared_root(node.cfg, node.grid, node.state, node.clock,
                                     store_direct=False)


def test_single_prepared_legacy_phase_checkpoint_is_not_reinterpreted(monkeypatch, tmp_path):
    source, start = _one_root(monkeypatch)
    source.root.state._lateral_boundary_device.clock = None
    path = restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    assert restart.read_restart_header(path)["root_external_lbc_clock"] == "legacy-elapsed-v0"
    target, _ = _one_root(monkeypatch, seed=91)
    original = target.root
    original.state._lateral_boundary_device.clock = None
    target.root = runner._single_prepared_root(
        original.cfg, original.grid, original.state, original.clock, store_direct=False)
    target.nodes_by_grid_id[1] = target.root
    before = target.root.state.u.tobytes()
    with pytest.raises(restart.RestartMismatchError,
                       match="root_external_lbc_clock semantic 'legacy-elapsed-v0'.*wrf-postincrement-v1"):
        runner._restore_single_checkpoint(target, path)
    assert target.root.state.u.tobytes() == before


def test_single_prepared_adapter_restores_actual_canonical_payload(monkeypatch, tmp_path):
    source, start = _one_root(monkeypatch)
    path = restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    target, _ = _one_root(monkeypatch, seed=91)
    target.root.clock.ticks = 0
    before = target.root.state.u.copy()
    assert not np.array_equal(before, source.root.state.u)
    info = runner._restore_single_checkpoint(target, path)
    assert info.elapsed_ticks == 3600 and target.root.clock.ticks == 3600
    assert target._resumed
    assert restart.read_restart_header(path)["domain_ids"] == [1]
    for key in restart.STATE_SERIALIZED_ATTRS:
        actual, expected = getattr(target.root.state, key, None), getattr(source.root.state, key, None)
        if expected is None:
            assert actual is None
            continue
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), key


@pytest.mark.parametrize("role", list(_inputs().file_sha256))
def test_each_sealed_authority_change_refuses_before_mutation(monkeypatch, tmp_path, role):
    source, start = _one_root(monkeypatch)
    path = restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    inputs = _inputs()
    inputs.file_sha256[role] = "changed-content"
    target, _ = _one_root(monkeypatch, seed=91, inputs=inputs)
    before = target.root.state.u.copy()
    with pytest.raises(restart.RestartMismatchError, match="different run"):
        runner._restore_single_checkpoint(target, path)
    assert target.root.state.u.tobytes() == before.tobytes()


def test_payload_corruption_does_not_reach_state(monkeypatch, tmp_path):
    source, start = _one_root(monkeypatch)
    path = restart.write_tree_restart(tmp_path, source, start + timedelta(seconds=3600))
    (tmp_path / "corrupt").mkdir()
    bad = tmp_path / "corrupt" / path.name
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    # A missing member must fail the unchanged canonical manifest gate.
    payload.pop("state/u")
    with bad.open("wb") as stream:
        np.savez(stream, **payload)
    target, _ = _one_root(monkeypatch, seed=91)
    before = target.root.state.u.copy()
    from gpuwm.supervisor import CheckpointValidationError
    with pytest.raises(CheckpointValidationError, match="member set disagrees") :
        runner._restore_single_checkpoint(target, bad)
    assert target.root.state.u.tobytes() == before.tobytes()


@pytest.mark.parametrize("after, expected", [(None, [0, 20, 40, 60, 80, 100, 120]),
    (60, [80, 100, 120]), (70, [80, 100, 120]), (120, [])])
def test_only_uncommitted_history_belongs_to_resumed_segment(after, expected):
    schedule = runner._history_output_schedule(start_time=datetime(2026, 1, 1),
        run_seconds=120, cadence_seconds=20, after_seconds=after)
    assert [row[0] for row in schedule] == expected


@pytest.mark.parametrize("diagnostic", [False, True])
def test_public_sim_forwards_single_restart_and_diagnostics(tmp_path, monkeypatch, diagnostic):
    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    checkpoint = tmp_path / "previous" / "checkpoint.npz"
    seen = []
    def execute(argv):
        parsed = runner.build_parser().parse_args(argv)
        assert parsed.restart == checkpoint
        assert parsed.experiment_config == config and parsed.wps_namelist == wps
        assert parsed.health_debug == diagnostic
        seen.append(parsed)
        return 19
    monkeypatch.setattr(runner, "main", execute)
    assert cli_main(["sim", str(root), "--experiment-config", str(config),
        "--wps-namelist", str(wps), "--restart", str(checkpoint),
        "--outdir", str(tmp_path / "new"),
        *(["--health-debug"] if diagnostic else [])]) == 19
    assert len(seen) == 1


def test_fresh_single_command_is_unchanged_by_default_operands(tmp_path):
    root = _single_domain_bundle(tmp_path / "prepared")
    config, wps = _authority(tmp_path / "authority")
    command = stage_cli.sim_command(stage_cli.resolve_bundle(root),
        experiment_config=config, wps_namelist=wps, outdir=tmp_path / "new")
    assert "--restart" not in command and "--health-debug" not in command
    parsed = runner.build_parser().parse_args(command[3:])
    assert parsed.restart is None and not parsed.health_debug


@pytest.mark.parametrize("failure", [restart.RestartMismatchError("wrong checkpoint"),
                                     FloatingPointError("restored state is invalid")])
def test_restore_refusal_joins_real_writer_and_preserves_error(failure):
    import threading
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from test_wrfout import _manual_async_writer

    abort = threading.Event()
    writer = _manual_async_writer(abort)
    owner = object.__new__(PerDomainWrfoutWriters)
    owner._writers = {1: writer}
    owner._abort_event = abort
    calls = []
    log = SimpleNamespace(close=lambda **kwargs: calls.append(kwargs))
    assert writer._thread.is_alive()
    with pytest.raises(type(failure)) as caught:
        with runner._checkpoint_restore_resources(owner, log):
            raise failure
    assert caught.value is failure
    assert abort.is_set() and writer._closed and not writer._thread.is_alive()
    assert calls == [{"status": "FAIL", "error": f"{type(failure).__name__}: {failure}"}]


def test_cleanup_failure_cannot_replace_checkpoint_refusal():
    failure = restart.RestartMismatchError("original identity mismatch")
    calls = []
    def broken_close(*args, **kwargs):
        calls.append(True)
        raise OSError("cleanup failed")
    writer = SimpleNamespace(__exit__=broken_close)
    log = SimpleNamespace(close=broken_close)
    with pytest.raises(restart.RestartMismatchError) as caught:
        with runner._checkpoint_restore_resources(writer, log):
            raise failure
    assert caught.value is failure and len(calls) == 2
    assert len(failure.__notes__) == 2
