"""The FTZ receipt's device block names the card, not the moment.

``receipt.json`` records ``device.uuid`` so a reader can tell which physical
GPU produced the bit table.  The value the probe wrote was 19 bytes -- 38 hex
characters -- where a CUDA UUID is 16: ``cudaDeviceProp`` stores ``uuid`` as
``char bytes[16]`` with no terminator and places ``luid`` immediately after
it, and the conversion that produces the buffer stops at the first zero byte
rather than at the array bound, so it ran on into the neighbour.

That made the field regeneration-unstable.  ``luid`` is re-issued when the
adapter is re-enumerated, so re-running the probe on the *same* card could
rewrite the field -- a diff that reads exactly like the receipt having moved
to different silicon, which is the single question the field answers.

No device is opened here.  CuPy is stubbed at ``sys.modules`` (the pattern
``tests/test_health_integer_policy.py`` uses, and for the same reason: the
probe imports CuPy *inside* ``device_identity``, which no attribute patch can
reach), so the two "consecutive probe runs" below differ in exactly one
thing -- the bytes trailing the UUID -- and the receipt block they produce is
compared byte for byte.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ftz_receipt import probe  # noqa: E402

#: A plausible 16-byte device UUID.  Nothing here asserts a real card's
#: value; the point is only that these 16 bytes survive and nothing else does.
UUID16 = bytes.fromhex("e302d74f28a933cb8853ae03ac17c249")

#: What the shipped receipt actually carried: the UUID plus three bytes that
#: belong to the next struct member.  Recorded here as the observed length,
#: not as a rule -- the rule is 16, and it lives in ``probe``.
OBSERVED_OVERREAD = bytes.fromhex("e302d74f28a933cb8853ae03ac17c2499e5d01")


def _cupy_stub(uuid_bytes: bytes) -> tuple[types.ModuleType, ...]:
    """A CuPy that answers the calls ``device_identity`` makes, on the host.

    Every value except ``uuid`` is a constant, so any difference between two
    runs of ``device_identity`` against two of these is the UUID handling and
    nothing else.
    """
    runtime = types.ModuleType("cupy.cuda.runtime")
    runtime.getDeviceProperties = lambda index: {
        "name": b"STUB DEVICE",
        "uuid": uuid_bytes,
        "major": 12,
        "minor": 0,
    }
    runtime.deviceGetPCIBusId = lambda index: "0000:01:00.0"
    runtime.driverGetVersion = lambda: 12080
    runtime.runtimeGetVersion = lambda: 12080

    nvrtc = types.ModuleType("cupy.cuda.nvrtc")
    nvrtc.getVersion = lambda: (12, 8)

    class _Device:
        id = 0

    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = runtime
    cuda.nvrtc = nvrtc
    cuda.Device = _Device

    stub = types.ModuleType("cupy")
    stub.cuda = cuda
    stub.__version__ = "13.0.0"
    return stub, cuda, runtime, nvrtc


def _identity(monkeypatch, uuid_bytes: bytes) -> dict:
    stub, cuda, runtime, nvrtc = _cupy_stub(uuid_bytes)
    monkeypatch.setitem(sys.modules, "cupy", stub)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", runtime)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvrtc", nvrtc)
    return probe.device_identity()


# ---- the slice itself, with no CuPy anywhere near it ---------------------

def test_the_uuid_is_the_first_sixteen_bytes_of_whatever_arrives():
    assert probe.device_uuid_hex(OBSERVED_OVERREAD) == UUID16.hex()
    assert len(probe.device_uuid_hex(OBSERVED_OVERREAD)) == 32


def test_a_longer_read_and_an_exact_read_agree():
    """The over-read and a correctly bounded read name the same card."""

    assert probe.device_uuid_hex(UUID16) == probe.device_uuid_hex(
        OBSERVED_OVERREAD)


@pytest.mark.parametrize("tail", [b"", b"\x01", b"\x9e\x5d\x01",
                                  b"\xff" * 8, b"\x00" * 4])
def test_no_trailing_byte_can_reach_the_recorded_value(tail):
    assert probe.device_uuid_hex(UUID16 + tail) == UUID16.hex()


@pytest.mark.parametrize("shape", [bytearray, memoryview])
def test_every_buffer_shape_is_bounded_the_same_way(shape):
    assert probe.device_uuid_hex(shape(OBSERVED_OVERREAD)) == UUID16.hex()


def test_a_non_buffer_value_is_passed_through_untouched():
    """``None`` means the property was absent; the receipt says so."""

    assert probe.device_uuid_hex(None) is None
    assert probe.device_uuid_hex("GPU-abc") == "GPU-abc"


# ---- two consecutive probe runs, end to end ------------------------------

def test_two_probe_runs_on_one_card_produce_the_same_device_block(monkeypatch):
    """The regeneration-stability claim, stated as the receipt states it.

    Run two differs from run one only in the bytes past the UUID, which is
    what re-enumerating the adapter changes.  ``json.dumps`` because that is
    how the block reaches ``receipt.json``: equal here is no diff there.
    """
    first = _identity(monkeypatch, UUID16 + b"\x9e\x5d\x01")
    second = _identity(monkeypatch, UUID16 + b"\x4b\xc2")

    assert first["uuid"] == second["uuid"] == UUID16.hex()
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)


def test_the_device_block_still_carries_every_field_it_promised(monkeypatch):
    """The slice must not have cost the receipt a key it documents."""

    identity = _identity(monkeypatch, OBSERVED_OVERREAD)
    for key in ("name", "uuid", "pci_bus_id", "compute_capability",
                "cuda_driver_version", "cuda_runtime_version",
                "nvrtc_version", "cupy_version", "numpy_version",
                "python", "platform"):
        assert identity.get(key) not in (None, ""), key
    assert identity["name"] == "STUB DEVICE"
    assert identity["compute_capability"] == "12.0"


def test_the_shipped_receipt_is_the_one_this_fix_is_about():
    """Guards the diagnosis, not the card.

    If the recorded value is ever 32 characters, the receipt has been
    regenerated through the bounded read and this file's premise -- that a
    19-byte value is what is on disk -- has expired with it.
    """
    receipt = json.loads(
        (Path(__file__).resolve().parents[1] / "tools" / "ftz_receipt"
         / "receipt" / "receipt.json").read_text(encoding="utf-8"))
    recorded = receipt["device"]["uuid"]
    assert len(recorded) in (32, 38)
    if len(recorded) == 38:
        assert probe.device_uuid_hex(bytes.fromhex(recorded)) == recorded[:32]
