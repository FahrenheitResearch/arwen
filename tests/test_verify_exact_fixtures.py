"""Per-scheme column fixtures for `gpuwm verify-exact`, on CPU.

What breaks without these: a scheme lane drops its oracle into the
fixtures directory and the sweep reports PASS whatever the runner returns,
or a malformed fixture is skipped in silence and reads as covered.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from gpuwm.verify_exact.fixtures import FIXTURE_SCHEMA, discover_fixtures, run_fixture

RUNNER = '''
import numpy as np
def double(inputs, scale=2.0):
    return {"QR": (inputs["QR"] * np.float32(scale)).astype(np.float32),
            "N": inputs["N"] + 1}
def broken(inputs):
    raise RuntimeError("kernel launch failed")
'''


def _fixture(root, ident, runner, expected, *, compare=None, options=None, schema=FIXTURE_SCHEMA):
    d = root / "toy" / ident
    d.mkdir(parents=True)
    np.savez(d / "inputs.npz", QR=np.array([1.0, 2.5, 3.0], np.float32), N=np.array([1, 2], np.int32))
    np.savez(d / "expected.npz", **expected)
    document = {"schema": schema, "scheme": "toy", "runner": runner,
                "reference": {"build": "test"}}
    if compare is not None:
        document["compare"] = compare
    if options is not None:
        document["options"] = options
    (d / "fixture.json").write_text(json.dumps(document))
    return d


@pytest.fixture
def runner_module(tmp_path, monkeypatch):
    (tmp_path / "mods").mkdir()
    (tmp_path / "mods" / "toy_runner_vx.py").write_text(RUNNER)
    monkeypatch.syspath_prepend(str(tmp_path / "mods"))
    yield "toy_runner_vx"
    sys.modules.pop("toy_runner_vx", None)


def test_a_bitwise_fixture_passes(tmp_path, runner_module):
    _fixture(tmp_path / "fx", "ok", f"{runner_module}:double",
             {"QR": np.array([2.0, 5.0, 6.0], np.float32), "N": np.array([2, 3], np.int32)})
    (fixture,) = discover_fixtures(tmp_path / "fx")
    assert fixture.key == "toy/ok"
    result = run_fixture(fixture)
    assert result.outcome == "PASS", result.line()


def test_a_one_ulp_miss_fails_with_its_numbers(tmp_path, runner_module):
    want = np.array([2.0, 5.0, 6.0], np.float32)
    want[1] = np.nextafter(want[1], np.float32(9))
    _fixture(tmp_path / "fx", "miss", f"{runner_module}:double",
             {"QR": want, "N": np.array([2, 3], np.int32)})
    result = run_fixture(discover_fixtures(tmp_path / "fx")[0])
    assert result.outcome == "FAIL"
    assert result.differing == ["QR"]
    assert (result.details[0].max_ulp, result.details[0].differing_words) == (1, 1)
    assert "QR max 1 ulp" in result.line()


def test_options_compare_lists_missing_outputs_and_runner_errors(tmp_path, runner_module):
    root = tmp_path / "fx"
    _fixture(root, "a-opts", f"{runner_module}:double",
             {"QR": np.array([3.0, 7.5, 9.0], np.float32), "QX": np.zeros(1, np.float32)},
             compare=["QR"], options={"scale": 3.0})
    _fixture(root, "b-missing", f"{runner_module}:double", {"QX": np.zeros(1, np.float32)})
    _fixture(root, "c-error", f"{runner_module}:broken", {"QR": np.zeros(3, np.float32)})
    a, b, c = (run_fixture(f) for f in discover_fixtures(root))
    assert a.outcome == "PASS"
    assert b.outcome == "FAIL" and b.missing == ["QX"]
    assert c.outcome == "ERROR" and "kernel launch failed" in c.error


def test_malformed_fixtures_are_refused_by_name(tmp_path, runner_module):
    _fixture(tmp_path / "a", "x", f"{runner_module}:double", {"QR": np.zeros(3, np.float32)},
             schema="something-else")
    with pytest.raises(ValueError, match="schema"):
        discover_fixtures(tmp_path / "a")
    _fixture(tmp_path / "b", "x", "no_colon", {"QR": np.zeros(3, np.float32)})
    with pytest.raises(ValueError, match="runner"):
        discover_fixtures(tmp_path / "b")
