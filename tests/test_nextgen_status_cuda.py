"""Boundary error handling on real CUDA; instrumentation, not hydro replay."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import cupy as cp
import numpy as np

from gpuwm.hydro import ResidentHydroDriver, Stage


class ResidentDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = cp.RawKernel(r'''
        extern "C" __global__ void status(int* current, int* sticky, int value) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 257) {
                current[i] = i == 256 ? value : 0;
                sticky[i] |= current[i];
            }
        }''', "status")
        cls.kernel.compile()

    def component(self):
        current, sticky = cp.zeros(257, cp.int32), cp.zeros(257, cp.int32)
        component = SimpleNamespace(
            arrays={"current": current, "sticky": sticky},
            status_arrays={"lsm": sticky},
            restart_identity={"instrumentation": "sticky-status-boundary/v1"},
            next_status=np.int32(0))

        def advance(dt, *, stream=None):
            self.kernel((3,), (128,), (current, sticky, component.next_status), stream=stream)

        component.advance = advance
        return component

    def driver(self, component, **kwargs):
        return ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
            interval_seconds=60, identity={"scope": "CUDA protocol instrumentation"},
            after=[Stage("noah", component, component.advance)], **kwargs)

    def advance(self, driver, component, mask, stream):
        component.next_status = np.int32(mask)
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)

    def test_fault_survives_healthy_step_and_blocks_checkpoint_on_producer_stream(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            component = self.component()
            driver = self.driver(component)
            self.advance(driver, component, 2, stream)
            self.advance(driver, component, 0, stream)
            component.arrays["sticky"][0] = 4
            component.arrays["sticky"][1] = np.int32(-2147483648)
            # No boundary read is hidden in either interval; both complete.
            self.assertEqual(driver.steps, 2)
            with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
                driver.checkpoint_metadata()
            self.assertEqual(error.exception.diagnostics, {"noah/lsm": 0x80000006})
            self.assertEqual(int(component.arrays["current"].sum()), 0)
            self.assertEqual(int(component.arrays["sticky"].sum()), -2147483642)

    def test_advisory_report_and_in_place_restore_preserve_bits_and_policy(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            component = self.component()
            from gpuwm.hydro.assembly import prepare
            factory = SimpleNamespace(prepare=lambda **unused: component)
            with patch.dict(sys.modules, {"gpuwm._status_protocol": factory}):
                driver = prepare(settings={"interval_seconds": 60, "stages": [{
                    "name": "noah", "factory": "gpuwm._status_protocol:prepare",
                    "status_fatal_masks": {"lsm": 6}}]}, state=None, physics=None,
                    device_id=cp.cuda.runtime.getDevice(), mode="standalone")
            self.advance(driver, component, 1, stream)
            metadata = driver.checkpoint_metadata()
            self.assertEqual(metadata["diagnostics"], {"noah/lsm": 1})
            saved = {key: value.copy() for key, value in driver.arrays.items()}
            pointers = {key: value.data.ptr for key, value in driver.arrays.items()}
            self.advance(driver, component, 2, stream)
            with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
                driver.check_health(stream=stream)
            self.assertEqual(error.exception.diagnostics, {"noah/lsm": 3})
            driver.validate_checkpoint(metadata, saved)
            for key, value in driver.arrays.items():
                cp.copyto(value, saved[key])
            driver.restore_metadata(metadata)
            self.assertEqual(driver.check_health(stream=stream), {"noah/lsm": 1})
            self.assertEqual(driver.steps, 1)
            self.assertEqual(pointers, {key: value.data.ptr for key, value in driver.arrays.items()})
            other_component = self.component()
            other = ResidentHydroDriver(device_id=driver.device_id, interval_seconds=60,
                identity=driver.restart_identity["composition"],
                after=[Stage("noah", other_component, other_component.advance)],
                status_fatal_masks={"noah/lsm": 0})
            with self.assertRaisesRegex(ValueError, "differ"):
                other.validate_checkpoint(metadata, saved)

    def test_unregistered_status_and_unknown_policy_refused_at_preparation(self):
        component = self.component()
        component.status_arrays = {"lsm": cp.zeros(257, cp.int32)}
        with self.assertRaisesRegex(ValueError, "checkpointed"):
            self.driver(component)
        with self.assertRaisesRegex(ValueError, "bound diagnostics"):
            self.driver(self.component(), status_fatal_masks={"absent": 1})
        for policy in ({"absent": 1}, {"lsm": True}, {"lsm": -1}, []):
            component = self.component()
            component.status_fatal_masks = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "bound diagnostics"):
                self.driver(component)

    def test_legacy_component_and_mid_interval_boundary(self):
        component = self.component()
        del component.status_arrays
        driver = self.driver(component)
        driver.before_lsm()
        with self.assertRaisesRegex(RuntimeError, "between"):
            driver.check_health()
        driver.advance()
        self.assertEqual(driver.check_health(), {})
        self.assertNotIn("diagnostics", driver.checkpoint_metadata())
        self.assertNotIn("status_fatal_masks", driver.restart_identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
