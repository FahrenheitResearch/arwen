"""Consume the real Rust lake fragment on CUDA; synthetic ABI fixture only."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.hydro import output
from gpuwm.hydro.reservoirs import prepare


class PreparedLakeTests(unittest.TestCase):
    def component(self, *, changed_datum=False):
        root = Path(os.environ["NEXTGEN_LAKE_FRAGMENT"])
        manifest = json.loads((root / "manifest.json").read_text())
        arrays = {}
        for name, spec in manifest["files"].items():
            raw = (root / spec["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), spec["sha256"])
            array = cp.empty(spec["shape"], dtype=spec["dtype"])
            self.assertEqual(len(raw), array.nbytes)
            if raw:
                host = cp.cuda.alloc_pinned_memory(len(raw))
                ctypes.memmove(host.ptr, raw, len(raw))  # Opaque I/O; Rust packed numbers.
                cp.cuda.runtime.memcpy(array.data.ptr, host.ptr, len(raw),
                                       cp.cuda.runtime.memcpyHostToDevice)
            arrays[name] = array
        if changed_datum:
            manifest["reservoirs"]["initialization"]["vertical_datum"] = "different-producer-datum"
        identity = {"source_sha256": manifest["source_manifest"]["sha256"],
                    "payload_sha256": {k: v["sha256"] for k, v in manifest["files"].items()}}
        source = SimpleNamespace(arrays=arrays, manifest=manifest, restart_identity=identity)
        component = prepare(settings={}, options={"network_component": "fabric"},
                            state=None, physics=None, device_id=0, mode=None,
                            components={"fabric": source})
        return component, manifest

    def driver(self, component):
        return ResidentHydroDriver(device_id=0, interval_seconds=300,
            identity={"scope": "Rust native-free synthetic lake ABI qualification"},
            after=[Stage("reservoir", component, component.launch)])

    def advance(self, driver, stream):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
        stream.synchronize()

    def test_factory_consumes_exact_rust_state_and_advances_healthy(self):
        component, manifest = self.component()
        root = Path(os.environ["NEXTGEN_LAKE_FRAGMENT"])
        for key in ("parameters", "height", "initial_height", "release"):
            spec = manifest["files"][manifest["reservoirs"]["arrays"][key]]
            self.assertEqual(cp.asnumpy(component.arrays[key]).tobytes(),
                             (root / spec["path"]).read_bytes())
        self.assertEqual(component.restart_identity["initialization"],
                         manifest["reservoirs"]["initialization"])
        driver = self.driver(component)
        pointers = {k: v.data.ptr for k, v in driver.arrays.items()}
        stream = cp.cuda.Stream(non_blocking=True)
        self.advance(driver, stream)
        self.assertEqual(driver.check_health(stream=stream), {"reservoir/fault": 0})
        self.assertTrue(bool(cp.all(cp.isfinite(component.arrays["height"]))))
        self.assertTrue(bool(cp.all(cp.isfinite(component.arrays["release"]))))
        self.assertTrue(bool(cp.all(component.arrays["clock"] == 300)))
        self.assertEqual(pointers, {k: v.data.ptr for k, v in driver.arrays.items()})
        pool = cp.get_default_memory_pool()
        used = pool.used_bytes()
        self.advance(driver, stream)
        self.assertEqual(used, pool.used_bytes())
        self.assertEqual(pointers, {k: v.data.ptr for k, v in driver.arrays.items()})
        print(json.dumps({"resource_scope": "two synthetic Rust-packed lakes; no canonical speed claim",
            "resident_arrays": len(driver.arrays),
            "resident_logical_bytes": sum(a.nbytes for a in driver.arrays.values()),
            "cupy_used_bytes": used, "cupy_pool_limit_bytes": pool.get_limit(),
            "stable_after_warm_step": True, "graphs": False,
            "reservoir_launches_per_step": 2, "compilation": component.compile_receipt,
            "levelpool_attributes": component._levelpool.attributes,
            "assimilation_attributes": component._assimilate.attributes,
            "nvrtc_version": cp.cuda.nvrtc.getVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion()}, sort_keys=True))

    def test_serialized_checkpoint_one_plus_one_matches_two(self):
        stream = cp.cuda.Stream(non_blocking=True)
        whole, _ = self.component()
        continued, _ = self.component()
        a, b = self.driver(whole), self.driver(continued)
        self.advance(a, stream)
        self.advance(a, stream)
        self.advance(b, stream)
        fresh, _ = self.component()
        c = self.driver(fresh)
        pointers = {key: value.data.ptr for key, value in c.arrays.items()}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lakes.nc"
            arrays, _, payload = output._snapshot(b.arrays, stream)
            started = time.perf_counter()
            output._call({"operation": "checkpoint-write", "path": str(path),
                "metadata": b.checkpoint_metadata(), "arrays": arrays}, payload)
            write_seconds = time.perf_counter() - started
            started = time.perf_counter()
            output.restore_checkpoint(path, c)
            restore_seconds = time.perf_counter() - started
            # A real serialized packet with incompatible initialization cannot
            # mutate any carrier or metadata, even when its initial bytes match.
            wrong, _ = self.component(changed_datum=True)
            d = self.driver(wrong)
            before = {key: value.get().tobytes() for key, value in d.arrays.items()}
            metadata = d.checkpoint_metadata()
            with self.assertRaisesRegex(ValueError, "composition, hydrofabric or parameters"):
                output.restore_checkpoint(path, d)
            self.assertEqual(metadata, d.checkpoint_metadata())
            for key, value in d.arrays.items():
                self.assertEqual(before[key], value.get().tobytes(), key)
            print(json.dumps({"scope": "actual Rust serialized lake checkpoint; synthetic ABI",
                "checkpoint_bytes": path.stat().st_size,
                "container_prefix_hex": path.read_bytes()[:4].hex(),
                "payload_arrays": len(arrays), "payload_bytes": len(payload),
                "write_seconds": write_seconds, "restore_seconds": restore_seconds,
                "datum_mismatch_refuses_before_mutation": True}, sort_keys=True))
        self.advance(c, stream)
        self.assertEqual(pointers, {key: value.data.ptr for key, value in c.arrays.items()})
        self.assertEqual(a.checkpoint_metadata(), c.checkpoint_metadata())
        for key in a.arrays:
            self.assertEqual(cp.asnumpy(a.arrays[key]).tobytes(),
                             cp.asnumpy(c.arrays[key]).tobytes(), key)

    def test_equal_payloads_cannot_hide_changed_initialization_datum(self):
        original, _ = self.component()
        other, _ = self.component(changed_datum=True)
        a, b = self.driver(original), self.driver(other)
        # Physical bytes are identical; the initialization contract differs.
        for key in a.arrays:
            self.assertEqual(cp.asnumpy(a.arrays[key]).tobytes(),
                             cp.asnumpy(b.arrays[key]).tobytes())
        with self.assertRaisesRegex(ValueError, "composition, hydrofabric or parameters"):
            b.validate_checkpoint(a.checkpoint_metadata(), a.arrays)


if __name__ == "__main__":
    unittest.main(verbosity=2)
