"""Real CUDA scheduling instrumentation; no runoff or routing physics replay."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np

from gpuwm.hydro.assembly import prepare
from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.nextgen_execution import bind_driver


class CadenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = cp.RawKernel(r'''
        extern "C" __global__ void record(unsigned long long* trace,
            unsigned long long* cursor, unsigned long long* count,
            unsigned long long* elapsed, unsigned long long tag,
            unsigned long long dt) {
            trace[(*cursor)++] = tag;
            ++*count;
            *elapsed += dt;
        }''', "record")
        cls.kernel.compile()

    def build(self, *, routing_dt=300, repeat=12, schedule=True):
        trace, cursor = cp.zeros(128, cp.uint64), cp.zeros(1, cp.uint64)
        components, rows = {}, []
        for tag, name in enumerate(("producer", "transfer", "route", "reservoir", "history"), 1):
            period = routing_dt if name in ("route", "reservoir") else 3600
            count, elapsed = cp.zeros(1, cp.uint64), cp.zeros(1, cp.uint64)
            args = (trace, cursor, count, elapsed, np.uint64(tag), np.uint64(period))

            def advance(dt, *, stream=None, expected=period, arguments=args):
                if dt != expected:
                    raise ValueError("instrumentation component period differs")
                self.kernel((1,), (1,), arguments, stream=stream)

            components[name] = SimpleNamespace(
                arrays=dict(trace=trace, cursor=cursor, count=count, elapsed=elapsed),
                restart_identity=dict(instrumentation=name, period_seconds=period),
                advance=advance)
            rows.append(dict(name=name, factory="gpuwm._cadence_protocol:prepare",
                             options=dict(name=name), parallelism="CUDA protocol recorder: 1 thread"))
        settings = dict(interval_seconds=3600, stages=rows)
        if schedule:
            settings["after_schedule"] = [
                dict(stages=["producer", "transfer"], interval_seconds=3600, repeat=1),
                dict(stages=["route", "reservoir"], interval_seconds=routing_dt, repeat=repeat),
                dict(stages=["history"], interval_seconds=3600, repeat=1)]
        factory = SimpleNamespace(prepare=lambda *, options, **unused: components[options["name"]])
        with patch.dict(sys.modules, {"gpuwm._cadence_protocol": factory}):
            driver = prepare(settings=settings, state=None, physics=None,
                             device_id=cp.cuda.runtime.getDevice(), mode="standalone")
        return driver, components

    def run_intervals(self, driver, count, stream):
        for _ in range(count):
            driver.before_lsm(stream=stream)
            driver.advance(stream=stream)

    def opaque_snapshot(self, driver):
        return {key: array.get().tobytes() for key, array in driver.arrays.items()}

    def test_macrostep_matches_explicit_cuda_order_and_periods(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            driver, _ = self.build()
            reference, parts = self.build()
            self.assertEqual(int(driver.arrays["producer/cursor"].get()[0]), 0)
            self.run_intervals(driver, 3, stream)
            # Independent explicit launch order, not the driver's compiled list.
            for _ in range(3):
                parts["producer"].advance(3600, stream=stream)
                parts["transfer"].advance(3600, stream=stream)
                for _ in range(12):
                    parts["route"].advance(300, stream=stream)
                    parts["reservoir"].advance(300, stream=stream)
                parts["history"].advance(3600, stream=stream)
            self.assertEqual(self.opaque_snapshot(driver), self.opaque_snapshot(reference))
            self.assertEqual(driver.steps, 3)
            self.assertEqual(driver.elapsed_seconds, 10800)
            self.assertEqual(len(driver.after_stages), 27)
            self.assertEqual(set(driver.components), {"producer", "transfer", "route", "reservoir", "history"})

    def test_fresh_restore_preserves_macro_and_component_clocks(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            whole, _ = self.build()
            partial, _ = self.build()
            self.run_intervals(whole, 3, stream)
            self.run_intervals(partial, 1, stream)
            metadata = partial.checkpoint_metadata()
            saved = {key: array.copy() for key, array in partial.arrays.items()}
            resumed, _ = self.build()
            pointers = {key: array.data.ptr for key, array in resumed.arrays.items()}
            resumed.validate_checkpoint(metadata, saved)
            for key, target in resumed.arrays.items():
                cp.copyto(target, saved[key])
            resumed.restore_metadata(metadata)
            self.run_intervals(resumed, 2, stream)
            self.assertEqual(self.opaque_snapshot(resumed), self.opaque_snapshot(whole))
            self.assertEqual(resumed.checkpoint_metadata(), whole.checkpoint_metadata())
            self.assertEqual(pointers, {key: array.data.ptr for key, array in resumed.arrays.items()})
            other, _ = self.build(routing_dt=600, repeat=6)
            with self.assertRaisesRegex(ValueError, "differ"):
                other.validate_checkpoint(metadata, saved)
            partial.before_lsm(stream=stream)
            with self.assertRaisesRegex(RuntimeError, "between"):
                partial.checkpoint_metadata()

    def test_profile_enqueues_exact_whole_macrostep(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            driver, _ = self.build()
            reference, _ = self.build()
            sequence = bind_driver(driver, parallelism={s.name: s.parallelism for s in driver.prepared_stages},
                                   metadata={"scope": "CUDA schedule instrumentation"})
            profile = bind_driver(reference, parallelism={s.name: s.parallelism for s in reference.prepared_stages},
                                  metadata={"scope": "CUDA schedule instrumentation"})
            sequence.enqueue()
            for stage in profile.stages:
                stage.enqueue()
            self.assertEqual(sequence.step_seconds, 3600)
            self.assertEqual(len(sequence.stages), 27)
            self.assertEqual(self.opaque_snapshot(driver), self.opaque_snapshot(reference))

    def test_invalid_groups_and_legacy_identity(self):
        driver, parts = self.build()
        base = dict(device_id=driver.device_id, interval_seconds=3600, identity={"test": "cadence"},
                    after=[Stage(name, component, component.advance) for name, component in parts.items()])
        groups = driver.restart_identity["after_schedule"]
        for schedule in ([], groups[:-1], [dict(groups[0], repeat=2), *groups[1:]],
                         [dict(groups[0], stages=["producer", "producer"]), *groups[1:]],
                         [dict(groups[0], repeat=True), *groups[1:]],
                         [dict(groups[0], stages=["missing"]), *groups[1:]]):
            with self.assertRaises(ValueError):
                ResidentHydroDriver(**base, after_schedule=schedule)
        legacy = ResidentHydroDriver(**base)
        self.assertNotIn("after_schedule", legacy.restart_identity)
        self.assertEqual([s.name for s in legacy.after_stages], list(parts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
