"""CPU pins for the per-period pool trim (execute_experiment)."""

from __future__ import annotations

import inspect
import sys
from unittest import mock

from gpuwm.core import model as model_mod


def test_pool_trim_defaults_to_platform_policy():
    signature = inspect.signature(model_mod.execute_experiment)
    parameter = signature.parameters["pool_trim_per_period"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_trim_default_pool_releases_unused_blocks(monkeypatch):
    fake_cupy = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    model_mod._trim_default_pool()
    fake_cupy.get_default_memory_pool.assert_called_once_with()
    pool = fake_cupy.get_default_memory_pool.return_value
    pool.free_all_blocks.assert_called_once_with()
