"""Keep setup/restart fingerprint bytes stable while avoiding a host byte copy."""
import hashlib

import numpy as np
import pytest

from gpuwm.state_serialization_contract import _digest_array


def _cases():
    words = np.array([0, 0x80000000, 0x7fc00001, 0x7fc00123,
                      0x00000001, 0xff800000], dtype=np.uint32).view(np.float32)
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    return [
        words, base, np.asfortranarray(base), base[:, ::2], base[::-1, ::-1],
        np.array(3.5), np.empty((0, 3)), np.zeros((3, 0)),
        base.astype('>f8'), np.array([1+2j, -3-4j]),
        np.array([True, False]), np.array([b'alpha', b'beta']),
        np.array(['alpha', 'beta']),
        np.zeros(3, dtype=[('x', '<f4'), ('tag', '<i2')]),
        np.array(['2026-10-18'], dtype='datetime64[D]'),
        np.array([None, 'legacy-object'], dtype=object),
    ]


@pytest.mark.parametrize('array', _cases())
def test_setup_digest_retains_legacy_byte_stream(array):
    legacy = hashlib.sha256(b'existing-prefix')
    for part in ('sample'.encode(), str(array.shape).encode(),
                 str(array.dtype).encode(), array.tobytes()):
        legacy.update(part)
    current = hashlib.sha256(b'existing-prefix')
    _digest_array(current, 'sample', array)
    assert current.digest() == legacy.digest()


def test_setup_digest_reads_device_host_copy_once():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)

    class DeviceValue:
        calls = 0

        def get(self):
            self.calls += 1
            return array

    value = DeviceValue()
    expected = hashlib.sha256()
    _digest_array(expected, 'field', array)
    observed = hashlib.sha256()
    _digest_array(observed, 'field', value)
    assert value.calls == 1
    assert observed.digest() == expected.digest()
