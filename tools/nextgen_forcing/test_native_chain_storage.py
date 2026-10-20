"""Exercise fixture admission against the actual Rust packet reader, without CUDA.

Set NEXTGEN_FORCING_MANIFEST, NEXTGEN_PET_CAPTURE and
GPUWM_NEXTGEN_FORCING_BRIDGE to the retained native example artifacts.
All temporary negative controls go beneath TMPDIR in owned scratch.
"""
import ctypes
import json
import os
from pathlib import Path
import tempfile
import unittest

from native_chain_factory import _fixture_storage


@unittest.skipUnless(all(os.environ.get(k) for k in (
    "NEXTGEN_FORCING_MANIFEST", "NEXTGEN_PET_CAPTURE",
    "GPUWM_NEXTGEN_FORCING_BRIDGE")), "retained native artifacts required")
class NativeStorage(unittest.TestCase):
    def setUp(self):
        self.manifest = Path(os.environ["NEXTGEN_FORCING_MANIFEST"])
        self.capture = Path(os.environ["NEXTGEN_PET_CAPTURE"])
        self.library = os.environ["GPUWM_NEXTGEN_FORCING_BRIDGE"]
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_original_capture_is_loaded_unchanged(self):
        storage = _fixture_storage(self.manifest, self.capture, self.library)
        self.assertEqual(bytes(storage), self.capture.read_bytes())

    def test_capture_corruption_and_truncation_are_refused(self):
        original = self.capture.read_bytes()
        changed = bytearray(original)
        changed[0] ^= 1
        for label, data in (("changed", changed), ("truncated", original[:-1])):
            with self.subTest(label=label):
                path = self.root / label
                path.write_bytes(data)
                with self.assertRaisesRegex(RuntimeError, "bytes/hash differ"):
                    _fixture_storage(self.manifest, path, self.library)

    def test_clock_order_and_forcing_seal_are_refused(self):
        for key, value in (("time_origin", "2015-12-01T01:00:00Z"),
                           ("catchment_ids", ["another-catchment"]),
                           ("interval_seconds", 1800.),
                           ("data_sha256", {"forcing.f64": "0" * 64})):
            with self.subTest(key=key):
                metadata = json.loads(self.manifest.read_text())
                metadata[key] = value
                path = self.root / "manifest.json"
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    _fixture_storage(path, self.capture, self.library)

    def test_old_unsealed_reader_accepts_same_size_corruption(self):
        # Reproduce the concrete hole closed by switching to read_packet.
        changed = bytearray(self.capture.read_bytes())
        changed[0] ^= 1
        path = self.root / "changed"
        path.write_bytes(changed)
        lib = ctypes.CDLL(self.library)
        read = lib.arwen_forcing_read_f64
        read.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        read.restype = ctypes.c_int
        storage = (ctypes.c_double * (720 * 39))()
        self.assertEqual(read(bytes(path), storage, len(storage)), 0)
        with self.assertRaises(RuntimeError):
            _fixture_storage(self.manifest, path, self.library)


if __name__ == "__main__":
    unittest.main()
