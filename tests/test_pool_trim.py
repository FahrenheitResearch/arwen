"""CPU pins for the per-period pool trim (execute_experiment)."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from unittest import mock

from gpuwm.core import model as model_mod


def test_pool_trim_defaults_on():
    signature = inspect.signature(model_mod.execute_experiment)
    parameter = signature.parameters["pool_trim_per_period"]
    assert parameter.default is True
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_trim_default_pool_releases_unused_blocks(monkeypatch):
    fake_cupy = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    model_mod._trim_default_pool()
    fake_cupy.get_default_memory_pool.assert_called_once_with()
    pool = fake_cupy.get_default_memory_pool.return_value
    pool.free_all_blocks.assert_called_once_with()


def test_period_commit_guards_trim_with_flag():
    """The trim call sits inside on_period_commit behind the flag."""
    source = Path(model_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    execute = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_experiment")
    commit = next(
        node for node in ast.walk(execute)
        if isinstance(node, ast.FunctionDef)
        and node.name == "on_period_commit")
    guarded_calls = [
        node for node in ast.walk(commit)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "pool_trim_per_period"
        and any(isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_trim_default_pool"
                for statement in node.body
                for call in ast.walk(statement))]
    assert len(guarded_calls) == 1


def test_on_step_guards_trim_with_flag():
    """The step-cadence trim sits inside on_step behind the same flag."""
    source = Path(model_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    execute = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_experiment")
    step = next(
        node for node in ast.walk(execute)
        if isinstance(node, ast.FunctionDef)
        and node.name == "on_step")
    guarded_calls = [
        node for node in ast.walk(step)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "pool_trim_per_period"
        and any(isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_trim_default_pool"
                for statement in node.body
                for call in ast.walk(statement))]
    assert len(guarded_calls) == 1
