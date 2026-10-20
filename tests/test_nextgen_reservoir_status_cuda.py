"""Real RFC bounds failures must reach the resident publication boundary."""
import unittest

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.hydro.reservoirs import Reservoirs, PARAMETERS, CONTROLS, SOURCE_PIN


class ReservoirStatusTests(unittest.TestCase):
    def component(self, *, enabled):
        # Deliberately empty RFC forecast with an enabled horizon. This is a
        # bounds/error protocol test, not a physical hydrology fixture.
        arrays = {key: cp.zeros(1, cp.float32) for key in (
            "previous_inflow", "inflow", "lateral", "height", "initial_height",
            "levelpool_release", "levelpool_height")}
        arrays["parameters"] = cp.asarray(
            [[1], [1], [10], [0.6], [1], [1], [0], [0.6], [1]], dtype=cp.float32)
        arrays.update({key: cp.zeros(1, cp.int32) for key in (
            "indices", "dynamic_type", "active_source", "fault")})
        arrays["types"] = cp.full(1, 4, cp.int32)
        arrays["offsets"] = cp.zeros(2, cp.int64)
        arrays.update({key: cp.zeros(1, cp.float64) for key in (
            "clock", "release", "assimilated")})
        arrays["carriers"] = cp.zeros((3, 1), cp.float64)
        arrays["controls"] = cp.zeros((8, 1), cp.float64)
        arrays["controls"][3, 0] = enabled
        arrays["controls"][4, 0] = 300
        arrays["controls"][5, 0] = 3600
        arrays["series"] = cp.empty(0, cp.float64)
        arrays["times"] = cp.empty(0, cp.float64)
        metadata = dict(parameter_columns=PARAMETERS, control_columns=CONTROLS,
            source_pin=SOURCE_PIN, validated_csr=True,
            network_sha256="instrumentation", parameters_sha256="instrumentation",
            series_sha256="instrumentation", time_origin="instrumentation",
            reservoir_order_sha256="instrumentation")
        return Reservoirs(arrays, metadata)

    def driver(self, component):
        return ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
            interval_seconds=300, identity={"scope": "RFC bounds instrumentation"},
            after=[Stage("reservoir", component, component.launch)])

    def advance(self, driver, stream):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)

    def test_real_rfc_fault_stays_sticky_and_prevents_publication(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            component = self.component(enabled=True)
            driver = self.driver(component)
            self.assertIs(driver.status_arrays["reservoir/fault"], component.arrays["fault"])
            self.advance(driver, stream)
            # A subsequent non-assimilating step must not erase the fault.
            component.arrays["controls"][3, 0] = 0
            self.advance(driver, stream)
            self.assertEqual(driver.steps, 2)
            with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
                driver.check_health(stream=stream)
            self.assertEqual(error.exception.diagnostics, {"reservoir/fault": 1})
            with self.assertRaisesRegex(RuntimeError, "prevent publication"):
                driver.checkpoint_metadata()

    def test_disabled_empty_forecast_remains_healthy(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            component = self.component(enabled=False)
            driver = self.driver(component)
            self.advance(driver, stream)
            self.assertEqual(driver.check_health(stream=stream), {"reservoir/fault": 0})
            self.assertEqual(driver.checkpoint_metadata()["diagnostics"], {"reservoir/fault": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
