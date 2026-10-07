"""CPU metadata and source-ownership tests for the public member-leg worker."""
from __future__ import annotations

import ast
from dataclasses import fields, replace
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.da_member_leg import MemberLegContext, MemberLegResult, _WorkerStage


def context(tmp_path):
    return MemberLegContext(
        args=SimpleNamespace(), inputs=SimpleNamespace(),
        identity=SimpleNamespace(members=4), cfg_perturb=None, hot_cfg=None,
        leg=0, absolute_leg_number=0, t_start=0.0, t_end=60.0,
        stage_root=tmp_path / "stage", out=tmp_path / "out")


def test_import_and_metadata_have_no_cuda_dependency():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from tools.da_member_leg import MemberLegContext; assert 'cupy' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_context_accepts_control_and_declared_members(tmp_path):
    ctx = context(tmp_path)
    for name in ("control", 0, 1, 2, 3):
        ctx.validate(name)


@pytest.mark.parametrize("name", [-1, 4, True, "0"])
def test_context_refuses_changed_member_identity(tmp_path, name):
    with pytest.raises(ValueError, match="member identity"):
        context(tmp_path).validate(name)


@pytest.mark.parametrize("changes,message", [
    ({"t_end": 0.0}, "clocks"),
    ({"t_end": float("nan")}, "clocks"),
    ({"nested": True}, "child domain"),
    ({"nest_birth": 61.0}, "child birth"),
    ({"resumed": True}, "complete tree restart"),
    ({"leg": 1}, "complete tree restart"),
    ({"leg": -1}, "indices"),
    ({"absolute_leg_number": 0.5}, "indices"),
    ({"restart": Path("source.npz")}, "silently ignore"),
    ({"pending": {"qv": [0.0]}}, "pending analysis"),
    ({"t_start": 1.0}, "elapsed zero"),
])
def test_context_refuses_unowned_clock_or_lineage(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(context(tmp_path), **changes).validate(0)


def test_worker_stage_retains_source_and_records_consumption_intent(tmp_path):
    stage = _WorkerStage(tmp_path / "stage")
    directory = stage.directory(7, 2)
    directory.mkdir(parents=True)
    source = directory / "gpuwmrst_d01.npz"
    source.write_bytes(b"retained checkpoint source")
    before = source.read_bytes()
    assert stage.consume(source) is False
    assert stage.consume_intent == source
    assert source.read_bytes() == before
    assert stage.directory(8, 2) != source.parent


def test_durable_generation_is_never_marked_for_consumption(tmp_path):
    stage = _WorkerStage(tmp_path / "stage")
    source = tmp_path / "generation" / "gpuwmrst_d01.npz"
    source.parent.mkdir()
    source.write_bytes(b"durable generation")
    assert stage.consume(source) is False
    assert stage.consume_intent is None
    assert source.exists()


def test_result_explicitly_separates_arrays_restart_and_source_consumption():
    names = {f.name for f in fields(MemberLegResult)}
    assert {"restart", "snapshot", "H_Z", "H_surface", "hot_pending", "setup_arrays",
            "thb_host", "nest_birth", "record", "consume_restart", "pending_consumed"} <= names


def test_worker_has_no_input_deletion_call():
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("tools/da_member_leg.py").read_text())
    dangerous = {"unlink", "rmdir", "rmtree", "remove", "clear"}
    assert not [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in dangerous]
