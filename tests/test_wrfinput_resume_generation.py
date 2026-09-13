"""Checkpoint replay cannot replace a prior generation's committed frames."""

from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm import stage_reuse


def test_replayed_valid_time_preserves_the_original_frame_and_receipt(tmp_path, monkeypatch):
    from gpuwm.io import wrfout

    root = tmp_path / "out"
    history = root / "wrfout"
    history.mkdir(parents=True)
    (root / "evidence").mkdir()
    checkpoint = root / "restart_d01_2026-09-12_00-30-00.gpuwmrst"
    checkpoint.write_bytes(b"checkpoint content belongs to the restart reader")
    when = datetime(2026, 9, 12, 1)
    old_path = history / wrfout.wrfout_filename(when, 1)
    with wrfout.WrfoutWriter(old_path, nx=1, ny=1, nz=1, dx=1., dy=1.) as writer:
        writer.write_frame("2026-09-12_01:00:00", {"T": np.ones((1, 1, 1), np.float32)})
    original = old_path.read_bytes()
    receipt = root / "evidence" / "run-receipt.json"
    receipt.write_text(hashlib.sha256(original).hexdigest())
    claim = stage_reuse.claim_run_output(root, resume=checkpoint)
    output = claim.path
    assert output != root
    new_history = output / "wrfout"
    new_history.mkdir(exist_ok=True)

    class CpuSink:
        global_attrs = {}

        def submit(self, path, stamp, state, *, frame, **kwargs):
            with wrfout.WrfoutWriter(path, nx=1, ny=1, nz=1, dx=1., dy=1.) as writer:
                writer.write_frame(stamp.strftime("%Y-%m-%d_%H:%M:%S"), frame)

        def drain(self):
            pass

    owner = object.__new__(wrfout.PerDomainWrfoutWriters)
    owner.start_time = datetime(2026, 9, 12)
    owner.output_dir = new_history
    owner._episode_by_grid_id = {}
    owner._published_paths = set()
    owner._metadata_by_grid_id = {1: {}}
    owner._writers = {1: CpuSink()}
    stream = SimpleNamespace(history_fields=lambda: {"T": np.full((1, 1, 1), 2., np.float32)})
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1), clock=SimpleNamespace(tick_den=1),
                           state=SimpleNamespace(_streamed_domain=stream))
    monkeypatch.setattr(wrfout, "streamed_carrier_provenance_attrs", lambda _: {})
    owner.submit(node, 3600)
    assert old_path.read_bytes() == original
    assert receipt.read_text() == hashlib.sha256(old_path.read_bytes()).hexdigest()
    with netCDF4.Dataset(new_history / old_path.name) as dataset:
        np.testing.assert_array_equal(dataset["T"][:], 2.)
    assert checkpoint.read_bytes() == b"checkpoint content belongs to the restart reader"
    claim.close()


def test_supervised_resume_keeps_the_checkpoint_and_one_output_generation(tmp_path, monkeypatch):
    from gpuwm.wrfinput_forecast import run_wrf_forecast
    from test_wrfinput_metem_products import _supervised
    from test_wrfinput_metem_resume import _resumable

    calls = {}
    _supervised(monkeypatch, calls)
    import subprocess
    original_run = subprocess.run

    def joined_worker(command, **kwargs):
        result = original_run(command, **kwargs)
        claimed = Path(command[command.index('--outdir') + 1])
        token = command[command.index('--_output-owner') + 1]
        with stage_reuse.claim_run_output(claimed, owner_token=token) as worker:
            assert worker.path == claimed
        return result

    monkeypatch.setattr(subprocess, 'run', joined_worker)
    root = tmp_path / "out"
    checkpoint = _resumable(root)
    assert run_wrf_forecast(tmp_path / "wrf", root, restart=checkpoint) == 0
    arguments = calls["argv"]
    child_output = Path(arguments[arguments.index("--outdir") + 1])
    child_checkpoint = Path(arguments[arguments.index("--restart") + 1])
    assert child_checkpoint == checkpoint
    assert child_output == root / "segment-001"
    assert not (child_output / "segment-001").exists()
    assert (root / "evidence" / "run-receipt.json").exists()


def test_a_broken_prior_generation_link_is_preserved_and_skipped(tmp_path):
    link = tmp_path / "segment-001"
    try:
        link.symlink_to(tmp_path / "absent-target", target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows account cannot create symbolic links; covered by the POSIX control")
        raise
    assert stage_reuse._next_segment(tmp_path) == tmp_path / "segment-002"
    assert link.is_symlink()
