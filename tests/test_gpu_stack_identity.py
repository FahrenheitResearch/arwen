# tests/test_gpu_stack_identity.py
"""One probe, two consumers.

The device/driver/runtime probe is asserted to be the SAME function object on
both sides by import identity, not by comparing two copies of a literal: a
copied literal drifts silently, an identity check cannot.  The published
``gpuwm-native-wrf-runtime-v1`` block shape is checked against the sample
transcribed from the source that predates the extraction.
"""

import json
from pathlib import Path

import pytest

import gpuwm.gpu_stack_identity as gpu_stack_identity
import gpuwm.native_wrf_distribution as native_wrf_distribution
from gpuwm.gpu_stack_identity import gpu_cuda_stack_identity

_SAMPLE = json.loads(
    (Path(__file__).parent / "data"
     / "native_wrf_runtime_gpu_block_pre_change.json").read_text())


def test_bridge_receipt_and_capsule_share_one_probe_object():
    """Import identity: a change to the probe reaches both receipts."""

    assert (native_wrf_distribution.gpu_cuda_stack_identity
            is gpu_stack_identity.gpu_cuda_stack_identity)
    assert (native_wrf_distribution.gpu_cuda_stack_identity
            is gpu_cuda_stack_identity)


def test_runtime_receipt_schema_is_unchanged_by_the_extraction():
    assert native_wrf_distribution.RUNTIME_SCHEMA == _SAMPLE["schema"]


def test_unprobed_block_matches_the_pre_change_sample():
    """The declared-no-device block still says only what it measured."""

    block = gpu_cuda_stack_identity(require_gpu=False)
    assert sorted(block) == sorted(
        _SAMPLE["gpu_block_keys"]["require_gpu_false"])
    assert block == _SAMPLE["gpu_block_values_require_gpu_false"]


def test_probe_body_carries_every_pre_change_key():
    """Every key of the probed block survives the move, none is invented.

    Read off the returned dict literal rather than a device, so the shape
    contract is checked on a runner with no CUDA card as well as on one with.
    """

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(gpu_cuda_stack_identity))
    returned_dicts = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
    keys = set()
    for literal in returned_dicts:
        for key in literal.keys:
            assert isinstance(key, ast.Constant), (
                "the probe's block keys must be literals, not computed names")
            keys.add(key.value)
    expected = set(_SAMPLE["gpu_block_keys"]["require_gpu_true"])
    expected |= set(_SAMPLE["gpu_block_keys"]["require_gpu_false"])
    assert keys == expected


def test_cuda_runtime_range_is_the_pre_change_range():
    assert (list(gpu_stack_identity.CUDA_RUNTIME_RANGE)
            == _SAMPLE["cuda_runtime_accepted_range"])


def test_probe_is_not_reimplemented_inside_the_bridge_module():
    """FAILURE CONTROL: a second copy of the probe would pass every check above.

    The bridge module must not contain its own device interrogation, or the
    identity assertion would be satisfied by an unused import while the
    receipt was built from a copy.
    """

    source = Path(native_wrf_distribution.__file__).read_text()
    for reimplementation in (
            "cp.cuda.runtime.getDeviceCount",
            "cp.cuda.runtime.driverGetVersion",
            "cp.cuda.runtime.runtimeGetVersion",
            "cp.cuda.runtime.getDeviceProperties"):
        assert reimplementation not in source, (
            f"{reimplementation} is probed inside the bridge module; the "
            "shared probe in gpuwm/gpu_stack_identity.py owns it")


@pytest.mark.parametrize("require_gpu", [False])
def test_declared_no_device_never_reports_a_device(require_gpu):
    block = gpu_cuda_stack_identity(require_gpu=require_gpu)
    assert block["status"] == gpu_stack_identity.GPU_STATUS_NOT_CHECKED
    assert "device_0_name" not in block
    assert "driver_version" not in block
    assert "runtime_version" not in block
