"""Real terrain CUDA error flags must prevent resident publication.

Small imposed-field protocol checks; no physical event or numerical replay claim.
"""
import unittest

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.nextgen.terrain import Terrain2D, TerrainLateral
from gpuwm.nextgen.inundation import InundationDepth


class TerrainStatusTests(unittest.TestCase):
    def terrain(self):
        fields = {name: cp.full((2, 2), value, cp.float32) for name, value in {
            "surface_depth": .1, "groundwater_storage": .05, "elevation": 1,
            "soil_depth": 2, "specific_yield": .25, "manning": .03,
            "conductivity": 0, "surface_source": 0, "recharge": 0,
            "outlet_stage": 0, "outlet_width": 0}.items()}
        fields["active"] = cp.ones((2, 2), cp.int32)
        return Terrain2D(dx=1000, dt=1, period_seconds=1,
                         identity={"scope": "terrain error protocol"}, **fields)

    def lateral(self, terrain):
        return TerrainLateral(terrain=terrain, offsets=cp.asarray([0, 1], cp.int64),
            indices=cp.asarray([0], cp.int32), identity={"scope": "one cell export"})

    def inundation(self, terrain, stream):
        return InundationDepth(surface_depth=terrain.depth_m,
            active=terrain.arrays["active"], source_health=terrain.health,
            source_identity=terrain.restart_identity, period_seconds=1, stream=stream)

    def driver(self, stages, *, resources=(), **kwargs):
        return ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
            interval_seconds=1, identity={"scope": "terrain error protocol"},
            after=[Stage(name, component, component.advance) for name, component in stages],
            resources=resources, **kwargs)

    def advance(self, driver, stream):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)

    def blocked(self, driver, stream, expected):
        with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
            driver.check_health(stream=stream)
        self.assertEqual(error.exception.diagnostics, expected)
        with self.assertRaisesRegex(RuntimeError, "prevent publication"):
            driver.checkpoint_metadata()

    def test_healthy_complete_sequence_keeps_original_u32_carriers(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            terrain = self.terrain()
            lateral = self.lateral(terrain)
            inundation = self.inundation(terrain, stream)
            components = [("terrain", terrain), ("lateral", lateral), ("inundation", inundation)]
            driver = self.driver(components)
            for name, component in components:
                health = component.arrays["health"]
                self.assertIs(driver.status_arrays[name + "/health"], health)
                self.assertEqual(health.dtype, cp.dtype("uint32"))
            self.advance(driver, stream)
            expected = {name + "/health": 0 for name, _ in components}
            self.assertEqual(driver.check_health(stream=stream), expected)
            self.assertEqual(driver.checkpoint_metadata()["diagnostics"], expected)

    def test_invalid_terrain_parameter_stays_sticky_and_propagates(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            terrain = self.terrain()
            lateral = self.lateral(terrain)
            inundation = self.inundation(terrain, stream)
            driver = self.driver([("terrain", terrain), ("lateral", lateral), ("inundation", inundation)])
            terrain.arrays["manning"].fill(0)
            self.advance(driver, stream)
            terrain.arrays["manning"].fill(.03)
            self.advance(driver, stream)
            self.assertEqual(driver.steps, 2)
            self.blocked(driver, stream, {"terrain/health": 1, "lateral/health": 0,
                                          "inundation/health": 1})

    def test_invalid_csr_bounds_stay_sticky_after_repair(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            terrain = self.terrain()
            lateral = self.lateral(terrain)
            driver = self.driver([("lateral", lateral)], resources=[("terrain", terrain)])
            lateral.arrays["routing_cell_offsets"][1] = 2
            self.advance(driver, stream)
            lateral.arrays["routing_cell_offsets"][1] = 1
            self.advance(driver, stream)
            self.blocked(driver, stream, {"lateral/health": 8, "terrain/health": 0})

    def test_invalid_depth_stays_sticky_after_repair(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            terrain = self.terrain()
            inundation = self.inundation(terrain, stream)
            driver = self.driver([("inundation", inundation)], resources=[("terrain", terrain)])
            terrain.depth_m.fill(-1)
            self.advance(driver, stream)
            terrain.depth_m.fill(.1)
            self.advance(driver, stream)
            self.blocked(driver, stream, {"inundation/health": 2, "terrain/health": 0})

    def test_unsigned_high_bit_advisory_and_in_place_restore(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            terrain = self.terrain()
            driver = self.driver([("terrain", terrain)], status_fatal_masks={"terrain/health": 1})
            terrain.health.fill(0x80000000)
            self.advance(driver, stream)
            metadata = driver.checkpoint_metadata()
            self.assertEqual(metadata["diagnostics"], {"terrain/health": 0x80000000})
            saved = {name: array.copy() for name, array in driver.arrays.items()}
            pointers = {name: array.data.ptr for name, array in driver.arrays.items()}
            terrain.arrays["manning"].fill(0)
            self.advance(driver, stream)
            self.blocked(driver, stream, {"terrain/health": 0x80000001})
            driver.validate_checkpoint(metadata, saved)
            for name, array in driver.arrays.items():
                cp.copyto(array, saved[name])
            driver.restore_metadata(metadata)
            self.assertEqual(driver.check_health(stream=stream), {"terrain/health": 0x80000000})
            self.assertEqual(pointers, {name: array.data.ptr for name, array in driver.arrays.items()})

    def test_status_requires_original_checkpointed_integer_buffer(self):
        terrain = self.terrain()
        terrain.status_arrays = {"health": cp.zeros(1, cp.uint32)}
        with self.assertRaisesRegex(ValueError, "checkpointed"):
            self.driver([("terrain", terrain)])
        terrain.status_arrays = {"health": terrain.health.view(cp.uint16)}
        with self.assertRaisesRegex(ValueError, "32"):
            self.driver([("terrain", terrain)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
