"""Actual MC preparation and continuation; small synthetic transport network."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.hydro import output, troute_mc


class PreparedInitialTests(unittest.TestCase):
    def packet(self, *, declared=True):
        arrays = {
            "node_ids": cp.asarray([17, 29], cp.int64),
            "routing_parameters": cp.asarray([[1000, 10, 20, 30, .03, .06, 1, .001]]*2, cp.float32),
            "upstream_offsets": cp.asarray([0, 0, 1], cp.int32),
            "upstream_indices": cp.asarray([0], cp.int32),
            "ordered_nodes": cp.asarray([0, 1], cp.int32),
            "routing_state": cp.asarray([[3, .25, .5], [5, .5, .75]], cp.float32),
            "routing_lateral": cp.asarray([.2, .3], cp.float32),
        }
        # Descriptor-only fixture. No Rust parsing or physical basin claim.
        identity = dict(source_sha256="synthetic-transport", payload_sha256={
            key: "synthetic-"+key for key in arrays})
        network = SimpleNamespace(arrays=arrays, restart_identity=identity,
            level_offsets=[0, 1, 2], manifest={"routing": {
                "initial_state_array": "routing_state"} if declared else {}})
        return network

    def prepare(self, network, **options):
        return troute_mc.prepare(settings=dict(interval_seconds=300,
            time_origin="2023-01-01T00:00:00Z"),
            options=dict(network_component="network", lateral_component="network", **options),
            state=None, physics=None, device_id=cp.cuda.runtime.getDevice(),
            mode="standalone", components={"network": network})

    def driver(self):
        router = self.prepare(self.packet())
        driver = ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
            interval_seconds=300, identity={"scope": "synthetic prepared initial state"},
            after=[Stage("routing", router, router.launch)],
            reach_output=router.reach_output())
        return driver

    def advance(self, driver):
        driver.before_lsm(); driver.advance()

    def test_declared_initial_state_is_bound_and_published(self):
        network = self.packet()
        expected = network.arrays["routing_state"].get().tobytes()
        router = self.prepare(network)
        self.assertEqual(router.arrays["state"].data.ptr, network.arrays["routing_state"].data.ptr)
        self.assertEqual(router.arrays["previous"].get().tobytes(), expected)
        self.assertEqual(router.reach_output()["discharge_m3_s"].get().tobytes(),
                         cp.asarray([3, 5], cp.float32).get().tobytes())
        self.assertNotIn("cold_start", router.restart_identity["initial_state"])

    def test_default_first_step_matches_explicit_prepared_state(self):
        default = self.prepare(self.packet())
        explicit = self.prepare(self.packet(), initial_state_component="network")
        default.launch(300); explicit.launch(300)
        for key in ("state", "previous", "diagnostics", "published"):
            self.assertEqual(default.arrays[key].get().tobytes(), explicit.arrays[key].get().tobytes(), key)

    def test_explicit_override_and_undeclared_cold_start(self):
        network = self.packet()
        network.arrays["override"] = cp.full((2, 3), .125, cp.float32)
        router = self.prepare(network, initial_state_component="network", initial_state_array="override")
        self.assertEqual(router.arrays["state"].data.ptr, network.arrays["override"].data.ptr)
        legacy = self.prepare(self.packet(declared=False))
        self.assertIn("cold_start", legacy.restart_identity["initial_state"])
        self.assertEqual(legacy.arrays["state"].get().tobytes(), bytes(24))

    def test_malformed_declared_initial_state_cannot_silently_cold_start(self):
        for problem in ("missing array", "missing hash", "empty name", "non-string name"):
            with self.subTest(problem=problem):
                network = self.packet()
                if problem == "missing array":
                    del network.arrays["routing_state"]
                elif problem == "missing hash":
                    del network.restart_identity["payload_sha256"]["routing_state"]
                else:
                    network.manifest["routing"]["initial_state_array"] = "" if problem == "empty name" else []
                with self.assertRaisesRegex(ValueError, "initial_state_array"):
                    self.prepare(network)

    def test_uncheckpointed_reach_publication_is_refused(self):
        for field in ("reach_id", "discharge_m3_s"):
            with self.subTest(field=field):
                router = self.prepare(self.packet())
                sample = dict(router.reach_output())
                sample[field] = sample[field].copy()
                with self.assertRaisesRegex(ValueError, "reach output.*checkpointed"):
                    ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
                        interval_seconds=300, identity={},
                        after=[Stage("routing", router, router.launch)], reach_output=sample)

    def test_registered_publication_views_are_borrowed_without_copy(self):
        driver = self.driver()
        router = driver.components["routing"]
        for field in ("reach_id", "discharge_m3_s"):
            self.assertEqual(driver.reach_output()[field].data.ptr,
                             router.reach_output()[field].data.ptr)
        # Discharge is a contiguous row view into the registered packed buffer.
        self.assertEqual(driver.reach_output()["discharge_m3_s"].data.ptr,
                         router.arrays["published"].data.ptr)

    def test_serialized_continuation_keeps_all_carriers_and_pointers(self):
        full = self.driver(); self.advance(full); self.advance(full)
        partial = self.driver(); self.advance(partial)
        resumed = self.driver()
        pointers = {key: a.data.ptr for key, a in resumed.arrays.items()}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"restart.nc"
            stream = cp.cuda.get_current_stream()
            arrays, _, payload = output._snapshot(partial.arrays, stream)
            output._call({"operation": "checkpoint-write", "path": str(path),
                "metadata": partial.checkpoint_metadata(), "arrays": arrays}, payload)
            output.restore_checkpoint(path, resumed)
        for field in ("reach_id", "discharge_m3_s"):
            self.assertEqual(partial.reach_output()[field].get().tobytes(),
                             resumed.reach_output()[field].get().tobytes(), field)
        self.advance(resumed)
        self.assertEqual(resumed.steps, 2)
        for key, a in full.arrays.items():
            self.assertEqual(a.get().tobytes(), resumed.arrays[key].get().tobytes(), key)
        self.assertEqual(pointers, {key: a.data.ptr for key, a in resumed.arrays.items()})
        for field in ("reach_id", "discharge_m3_s"):
            self.assertEqual(full.reach_output()[field].get().tobytes(),
                             resumed.reach_output()[field].get().tobytes(), field)
        print(json.dumps({"scope": "synthetic MC Rust checkpoint transport",
            "restored_steps": resumed.steps,
            "carriers": {key: {"bytes": a.nbytes,
                "sha256": hashlib.sha256(a.get().tobytes()).hexdigest()}
                for key, a in resumed.arrays.items()}}, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
