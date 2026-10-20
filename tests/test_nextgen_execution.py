"""Real CUDA regression for stream ordering, physical steps and timing isolation.

The integer kernels exercise launch transport only, never hydro acceptance.
Run directly with unittest to avoid unrelated repository collection hooks.
"""

from functools import partial
import unittest
from types import SimpleNamespace

from gpuwm.nextgen_execution import ResidentSequence, Stage, bind_driver, measure


class ResidentExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        cls.cp = cp
        cls.module = cp.RawModule(code=r'''
        extern "C" __global__ void advance(int *state, int *clock) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 257) state[i] += 1;
            if (i == 0) clock[0] += 60;
        }
        extern "C" __global__ void route(const int *state, int *out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 257) out[i] += state[i];
        }
        ''')
        cls.advance = cls.module.get_function("advance")
        cls.route = cls.module.get_function("route")

    def factory(self):
        cp = self.cp
        state, out = cp.zeros(257, dtype=cp.int32), cp.zeros(257, dtype=cp.int32)
        clock = cp.zeros(1, dtype=cp.int32)
        stages = [
            Stage("advance", partial(self.advance, (3,), (128,), (state, clock)),
                  "GPU: 3 blocks x 128 threads"),
            Stage("route", partial(self.route, (3,), (128,), (state, out)),
                  "GPU: 3 blocks x 128 threads"),
        ]
        host_steps = [0]

        def whole_interval():
            for stage in stages:
                stage.enqueue()
            host_steps[0] += 1

        sequence = ResidentSequence(stages, step_seconds=60,
                                   enqueue_interval=whole_interval,
                                   keepalive=(state, out, clock, self.module, host_steps), metadata={
            "scope": "launch-transport regression; no hydrologic physics",
            "numerics": "fast", "native_replay": False,
        })
        self.built.append(sequence)
        return sequence

    def test_parent_stream_steps_and_independent_profile(self):
        cp = self.cp
        self.built = []
        with cp.cuda.Stream(non_blocking=True) as parent:
            untouched = self.factory()
            parent.synchronize()
            self.assertEqual(int(untouched.keepalive[2].get()[0]), 0)
            untouched.enqueue()
            untouched.enqueue()
            parent.synchronize()
            self.assertTrue(bool(cp.all(untouched.keepalive[0] == 2).item()))
            self.assertTrue(bool(cp.all(untouched.keepalive[1] == 3).item()))
            self.assertEqual(int(untouched.keepalive[2].get()[0]), 120)
            receipt = measure(self.factory, steps=64, warmup=1)
            timed, profiled = self.built[-2:]
            self.assertTrue(bool(cp.all(timed.keepalive[0] == 65).item()))
            self.assertTrue(bool(cp.all(timed.keepalive[1] == 2145).item()))
            self.assertEqual(int(timed.keepalive[2].get()[0]), 3900)
            self.assertEqual(timed.keepalive[4][0], 65)
            self.assertTrue(bool(cp.all(profiled.keepalive[0] == 2).item()))
            self.assertTrue(bool(cp.all(profiled.keepalive[1] == 3).item()))
            self.assertEqual(int(profiled.keepalive[2].get()[0]), 120)
            self.assertEqual(profiled.keepalive[4][0], 1)
            self.assertEqual(cp.cuda.get_current_stream().ptr, parent.ptr)
        self.assertEqual(receipt["batch"]["events"], 0)
        self.assertEqual(receipt["batch"]["stream_waits"], 2)
        self.assertFalse(receipt["batch"]["cuda_graphs"])
        self.assertEqual(receipt["batch"]["simulated_seconds"], 3840)
        self.assertEqual([s["name"] for s in receipt["profile"]["stages"]],
                         ["advance", "route"])
        self.assertGreater(receipt["batch"]["wall_ns"], 0)

    def test_bound_real_driver(self):
        try:
            from gpuwm.hydro.driver import ResidentHydroDriver, Stage as DriverStage
        except ModuleNotFoundError as exc:
            if exc.name != "gpuwm.hydro":
                raise
            self.skipTest("coupled-driver unit has not been integrated")
        self.built = []
        base = self.factory()
        state, out, clock = base.keepalive[:3]
        component = SimpleNamespace(arrays={"state": state, "out": out, "clock": clock},
                                    restart_identity={"scope": "transport regression"})

        def advance(dt_seconds, *, stream=None):
            self.advance((3,), (128,), (state, clock), stream=stream)

        def route(dt_seconds, *, stream=None):
            self.route((3,), (128,), (state, out), stream=stream)

        driver = ResidentHydroDriver(device_id=self.cp.cuda.runtime.getDevice(),
                                     interval_seconds=60, identity={"scope": "transport"},
                                     before=[DriverStage("advance", component, advance)],
                                     after=[DriverStage("route", component, route)])
        plan = bind_driver(driver, parallelism={"advance": "GPU 3x128", "route": "GPU 3x128"},
                           metadata={"numerics": "fast", "mode": "coupled"},
                           parent_lsm=Stage("parent-lsm", partial(route, 60), "GPU 3x128"))
        self.cp.cuda.get_current_stream().synchronize()
        with self.cp.cuda.Stream(non_blocking=True) as parent:
            plan.enqueue()
            plan.enqueue()
            parent.synchronize()
            self.assertTrue(bool(self.cp.all(out == 6).item()))
            self.assertEqual(int(clock.get()[0]), 120)
        self.assertEqual(driver.steps, 2)
        self.assertEqual(driver.elapsed_seconds, 120)
        self.assertEqual([s.name for s in plan.stages], ["advance", "parent-lsm", "route"])


if __name__ == "__main__":
    unittest.main()
