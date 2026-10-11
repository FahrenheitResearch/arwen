"""Grade the real float32 DomainState TEMPO adapter, every output and step.

Breakage prevented: adapter rounding, seed lifetime, emission or radiation
conversion moving even one word away from the unmodified TEMPO driver.
TEMPO_ADAPTER_ORACLE names build_adapter_reference.sh's output directory.
There is no tolerance and no seed exception. Only the documented upstream
undefined frozen_fraction on a dry early return can be excused.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]
ORACLE = os.environ.get("TEMPO_ADAPTER_ORACLE")


def _tool():
    path = ROOT / "tools/tempo_ref/oracle/adapter/run_adapter_oracle.py"
    spec = importlib.util.spec_from_file_location("tempo_adapter_oracle", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


@pytest.mark.skipif(not ORACLE, reason="TEMPO_ADAPTER_ORACLE not set")
def test_apply_domain_state_all_200_columns_every_word(tmp_path):
    tool = _tool()
    result = tool.run(Path(ORACLE), tmp_path / "adapter.json")
    assert result["columns"] == 200
    assert len(result["fields"]) == 26
    assert result["mismatched"] == 0, result["first_miss"]
    assert result["values"] == result["expected_values"]


@pytest.mark.skipif(not ORACLE, reason="TEMPO_ADAPTER_ORACLE not set")
def test_native_optional_cloud_radii_and_pre_physics_emission_every_word(tmp_path):
    tool = _tool()
    result = tool.run(Path(ORACLE), tmp_path / "native-cloud.json", native_cloud=True)
    assert result["native_cloud"]
    assert result["columns"] == 200
    assert result["mismatched"] == 0, result["first_miss"]
    assert result["values"] == 8365440


def test_comparator_catches_a_planted_float32_bit():
    import numpy as np
    tool = _tool()
    want = np.array([300.0], dtype=np.float32)
    got = want.copy()
    got.view(np.uint32)[0] ^= np.uint32(1)
    assert tool.word_misses(got, want).tolist() == [True]
    assert tool.word_misses(want.copy(), want).tolist() == [False]


def test_frozen_fraction_exception_requires_positive_zero_and_zero_precipitation():
    import numpy as np
    tool = _tool()
    ref = {n: np.zeros((1,1)) for n in tool.SURFACE}
    ref["frozen_fraction"][:] = np.nextafter(0.0,1.0)
    assert tool.undefined_frozen_fraction(ref,np.zeros((1,1))).item()
    assert not tool.undefined_frozen_fraction(ref,np.full((1,1),-0.0)).item()
    assert not tool.undefined_frozen_fraction(ref,np.ones((1,1))).item()
    ref["rain_precip"][:] = 1.0
    assert not tool.undefined_frozen_fraction(ref,np.zeros((1,1))).item()
