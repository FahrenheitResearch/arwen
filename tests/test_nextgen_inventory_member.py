"""Original member identity at the existing resident inventory boundary."""
from types import SimpleNamespace
import unittest
import pytest

from gpuwm.ensemble.nextgen_inventory import ResidentInventory


class Identity(unittest.TestCase):
    def test_foreign_identity_rejected_before_carrier_access(self):
        class ForeignDriver:
            member_id = 7

            @property
            def arrays(self):
                raise AssertionError("foreign member carriers must not be inspected")

        inventory = ResidentInventory()
        with self.assertRaisesRegex(ValueError, "member identity"):
            inventory.inspect(ForeignDriver(), member_id=22, device_id=0)
        self.assertEqual(inventory.receipt()["members"], [])


@pytest.mark.gpu
class Device(unittest.TestCase):
    def test_scheduled_member_publication_restores_complete_macrostep(self):
        """Cadence, sticky health and both collectors share a completed boundary.

        The CUDA recorder is synthetic instrumentation, not runoff/routing physics.
        """
        try:
            import cupy as cp
            from gpuwm.hydro import ResidentHydroDriver, Stage
        except ImportError as error:
            self.skipTest(str(error))
        from datetime import datetime
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import numpy as np
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope

        recorder = cp.RawKernel(r'''
        extern "C" __global__ void record(unsigned long long* counts,
            double* flow, int* status, const int* inject, int stage) {
            ++counts[stage];
            if (stage == 1) {
                flow[0] += 1.0;
                flow[1] += 2.0;
                if (*inject && counts[1] == 5) *status |= 2;
            }
        }''', "record")
        recorder.compile()

        for mass in (False, True):
            with self.subTest(mass=mass), TemporaryDirectory() as temporary:
                atmosphere, depths = [], []
                original = SimpleNamespace(root=Path(temporary), submit=lambda **row: atmosphere.append(row))
                if mass:
                    from gpuwm.ensemble.nextgen_mass import MemberReachCapture
                    collector = MemberReachCapture(original.submit, member_id=7,
                        directory=temporary, array_module=cp, options={"thresholds_m3_s": [10.]})
                    publish, points = collector, collector.points
                else:
                    collector = NextGenCollector(original, (7,), array_module=cp,
                                                 options={"thresholds_m3_s": [10.]})
                    publish, points = collector.submit, collector.nextgen_points
                producer = cp.cuda.Stream(non_blocking=True)
                with member_output_scope(MemberOutputCapture(publish, 7)), producer:
                    ids = cp.asarray([2**60 + 3, 42], cp.int64)
                    flow, counts = cp.zeros(2, cp.float64), cp.zeros(3, cp.uint64)
                    status, inject = cp.zeros(1, cp.int32), cp.zeros(1, cp.int32)
                    arrays = dict(ids=ids, flow=flow, counts=counts, status=status, inject=inject)
                    stages = []
                    for index, name in enumerate(("producer", "route", "history")):
                        period = 300 if name == "route" else 3600
                        arguments = (counts, flow, status, inject, np.int32(index))

                        def launch(dt, *, stream=None, expected=period, args=arguments):
                            if dt != expected:
                                raise ValueError("recorder period differs from its prepared period")
                            recorder((1,), (1,), args, stream=stream)

                        component = SimpleNamespace(arrays=arrays,
                            status_arrays={"fault": status} if name == "route" else {},
                            restart_identity={"kind": "cadence-publication-recorder", "period": period})
                        stages.append(Stage(name, component, launch))
                    driver = ResidentHydroDriver(device_id=ids.device.id, interval_seconds=3600,
                        identity={"kind": "cadence-publication-instrumentation"}, after=stages,
                        after_schedule=[
                            dict(stages=["producer"], interval_seconds=3600, repeat=1),
                            dict(stages=["route"], interval_seconds=300, repeat=12),
                            dict(stages=["history"], interval_seconds=3600, repeat=1)],
                        reach_output={"reach_id": ids, "discharge_m3_s": flow, "stream": producer})
                    state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
                    initial = driver.checkpoint_metadata()
                    saved = {name: array.copy() for name, array in driver.arrays.items()}
                    pointers = {name: array.data.ptr for name, array in driver.arrays.items()}

                    def submit():
                        publish(state=state, valid_time=datetime(2024, 1, 1, 1), grid_id=1, member_id=7)

                    with patch("gpuwm.ensemble.nextgen_depth.submit_channel_depth",
                               side_effect=lambda *args, **kwargs: depths.append(kwargs)):
                        inject.fill(1)
                        driver.before_lsm(stream=producer)
                        with self.assertRaisesRegex(RuntimeError, "between"):
                            submit()
                        driver.advance(stream=producer)
                        self.assertEqual(driver.steps, 1)
                        self.assertEqual(driver.elapsed_seconds, 3600)
                        self.assertEqual(counts.get().tolist(), [1, 12, 1])
                        self.assertEqual(status.get().tolist(), [2])
                        with self.assertRaisesRegex(RuntimeError, "prevent publication"):
                            submit()
                        with self.assertRaisesRegex(RuntimeError, "prevent publication"):
                            driver.checkpoint_metadata()
                        self.assertEqual(points.frames, {})
                        self.assertEqual(atmosphere, [])
                        self.assertEqual(depths, [])
                        # Restore every carrier and driver clock, including the
                        # pre-fault injection control, before retrying the interval.
                        driver.validate_checkpoint(initial, saved)
                        for name, target in driver.arrays.items():
                            cp.copyto(target, saved[name])
                        driver.restore_metadata(initial)
                        self.assertEqual(driver.steps, 0)
                        driver.before_lsm(stream=producer)
                        driver.advance(stream=producer)
                        submit()
                    self.assertEqual(counts.get().tolist(), [1, 12, 1])
                    self.assertEqual(flow.get().tolist(), [12., 24.])
                    self.assertEqual(status.get().tolist(), [0])
                    self.assertEqual(driver.elapsed_seconds, 3600)
                    self.assertEqual(pointers, {name: array.data.ptr for name, array in driver.arrays.items()})
                    self.assertEqual(len(depths), 1)
                    self.assertEqual(len(atmosphere), 1)
                    self.assertEqual(len(points.frames), 1)
                    points.require_complete()
                    frame = next(iter(points.frames.values()))
                    shard = frame["cards"][driver.device_id]
                    shard["done"].synchronize()
                    self.assertEqual(shard["count"].get().tolist(), [[1, 1]])
                    self.assertEqual(shard["finite"].get().tolist(), [1, 1])

    def test_fatal_status_prevents_root_publication_on_producer_stream(self):
        """A sticky device error cannot turn into a valid ensemble product."""
        try:
            import cupy as cp
            from gpuwm.hydro import ResidentHydroDriver, Stage
        except ImportError as error:
            self.skipTest(str(error))
        if not hasattr(ResidentHydroDriver, "check_health"):
            self.skipTest("requires driver status boundary contract")
        from datetime import datetime
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope

        for mass in (False, True):
            atmosphere, depth_calls, health_streams = [], [], []
            with TemporaryDirectory() as temporary:
                original = SimpleNamespace(root=Path(temporary), submit=lambda **row: atmosphere.append(row))
                if mass:
                    from gpuwm.ensemble.nextgen_mass import MemberReachCapture
                    collector = MemberReachCapture(original.submit, member_id=7,
                        directory=temporary, array_module=cp, options={"thresholds_m3_s": [.1]})
                    publish, points = collector, collector.points
                else:
                    collector = NextGenCollector(original, (7,), array_module=cp,
                                                 options={"thresholds_m3_s": [.1]})
                    publish, points = collector.submit, collector.nextgen_points
                capture = MemberOutputCapture(publish, 7)
                producer = cp.cuda.Stream(non_blocking=True)
                with member_output_scope(capture), producer:
                    ids = cp.asarray([2**60 + 3, 42], dtype=cp.int64)
                    q = cp.asarray([.2, 0.], dtype=cp.float64)
                    status = cp.zeros(2, dtype=cp.int32)
                    component = SimpleNamespace(arrays={"ids": ids, "q": q, "status": status},
                        status_arrays={"health": status}, restart_identity={"kind": "status-transport"})
                    driver = ResidentHydroDriver(device_id=ids.device.id, interval_seconds=300,
                        identity=component.restart_identity,
                        after=(Stage("routing", component, lambda dt, *, stream=None: None),),
                        reach_output={"reach_id": ids, "discharge_m3_s": q, "stream": producer})
                    state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
                    actual_check = driver.check_health

                    def check(*, stream=None):
                        health_streams.append(stream)
                        return actual_check(stream=stream)

                    def submit(grid_id):
                        publish(state=state, valid_time=datetime(2024, 1, 1),
                                         grid_id=grid_id, member_id=7)

                    with patch.object(driver, "check_health", side_effect=check), patch(
                            "gpuwm.ensemble.nextgen_depth.submit_channel_depth",
                            side_effect=lambda *args, **kwargs: depth_calls.append(kwargs)):
                        submit(2)
                        self.assertEqual(health_streams, [])
                        self.assertEqual(len(atmosphere), 1)
                        # Enqueue the signed high bit on the producer stream.
                        status[1:] = -(1 << 31)
                        for _ in range(2):
                            with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
                                submit(1)
                            self.assertEqual(error.exception.diagnostics, {"routing/health": 1 << 31})
                            self.assertEqual(points.frames, {})
                            self.assertEqual(depth_calls, [])
                            self.assertEqual(len(atmosphere), 1)
                        self.assertTrue(bool(cp.array_equal(status, cp.asarray([0, -(1 << 31)], cp.int32))))
                        # Explicit test reset; publication must never clear status.
                        status.fill(0)
                        submit(1)
                    self.assertEqual(health_streams, [producer, producer, producer])
                    self.assertEqual(len(depth_calls), 1)
                    self.assertEqual(len(atmosphere), 2)
                    points.require_complete()
                    frame = next(iter(points.frames.values()))
                    shard = frame["cards"][driver.device_id]
                    shard["done"].synchronize()
                    self.assertTrue(bool(cp.array_equal(shard["count"], cp.asarray([[1, 0]], cp.uint32))))
                    self.assertTrue(bool(cp.array_equal(shard["finite"], cp.asarray([1, 1], cp.uint32))))

    def test_bound_driver_uses_existing_root_history_consumer_once(self):
        """Member binding and the real history deadline share one capture scope.

        CUDA buffers here instrument transport, not a hydrologic algorithm.
        """
        try:
            import cupy as cp
            from gpuwm.hydro import ResidentHydroDriver, Stage
        except ImportError as error:
            self.skipTest(str(error))
        from datetime import datetime
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import ModuleType
        from unittest.mock import patch
        import sys

        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.optional_routing import member_settings_scope
        from gpuwm.ensemble.runtime_context import (
            MemberOutputCapture, current_capture, member_output_scope,
        )
        from gpuwm.hydro import runtime
        from gpuwm.io.wrfout import PerDomainWrfoutWriters

        borrowed, atmosphere, drivers = [], [], []
        module = ModuleType("gpuwm.hydro._ensemble_binding_test")

        def factory(**kwargs):
            ids = cp.asarray([2**60 + 3, 7, 42, 99], dtype=cp.int64)
            member = current_capture().member_id
            q = cp.asarray([0., .1, 1., 2. if member == 7 else float("nan")],
                           dtype=cp.float64 if member == 7 else cp.float32)
            component = SimpleNamespace(arrays={"ids": ids, "q": q},
                restart_identity={"kind": "member-history-transport-instrumentation"})
            return ResidentHydroDriver(device_id=kwargs["device_id"], interval_seconds=300,
                identity=component.restart_identity,
                after=(Stage("publication", component, lambda dt, *, stream=None: None),),
                reach_output={"reach_id": ids, "discharge_m3_s": q,
                              "stream": cp.cuda.get_current_stream()})

        module.prepare = factory

        class Capture(MemberOutputCapture):
            def submit_hydro(self, **row):
                borrowed.append((self.member_id, row))
                return super().submit_hydro(**row)

        from gpuwm import numerics
        settings = {"enabled": True, "driver_factory": module.__name__ + ":prepare",
                    "numerics": numerics.mode()}
        with TemporaryDirectory() as temporary, patch.dict(sys.modules, {module.__name__: module}):
            original = SimpleNamespace(root=Path(temporary), submit=lambda **row: atmosphere.append(row))
            collector = NextGenCollector(original, (7, 22), array_module=cp,
                                         options={"thresholds_m3_s": [.1]})
            for member in (7, 22):
                capture = Capture(collector.submit, member)
                stream = cp.cuda.Stream(non_blocking=True)
                with member_output_scope(capture), member_settings_scope(
                        SimpleNamespace(experiment_config=None), inherited_settings=settings), stream:
                    driver = runtime.prepare(runtime.current_settings(), device_id=cp.cuda.runtime.getDevice())
                    self.assertEqual(driver.member_id, member)
                    self.assertEqual(driver.restart_identity["member_id"], member)
                    state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
                    collector.bind_member(member_id=member, state=state)
                    with self.assertRaisesRegex(ValueError, "member identity"):
                        collector.bind_member(member_id=99, state=state)
                    if drivers:
                        module.prepare = lambda **unused: drivers[0]
                        with self.assertRaisesRegex(ValueError, "another member"):
                            runtime.prepare(runtime.current_settings(), device_id=driver.device_id)
                        module.prepare = factory
                    writer = object.__new__(PerDomainWrfoutWriters)
                    writer.start_time, writer.output_dir = datetime(2024, 1, 1), Path(temporary)
                    writer._episode_by_grid_id = {1: 0, 2: 0}
                    writer._published_paths = set()
                    writer._captured_paths, writer._archived_paths = [], []
                    writer._metadata_by_grid_id, writer._writers = {1: {}, 2: {}}, {}
                    child = SimpleNamespace(state=SimpleNamespace(), cfg=SimpleNamespace(grid_id=2),
                                            clock=SimpleNamespace(tick_den=10))
                    root = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=1),
                                           clock=SimpleNamespace(tick_den=10))
                    writer.submit(child, 300, refl_field=None)
                    self.assertEqual(len(borrowed), len(drivers))
                    writer.submit(root, 300, refl_field=None)
                    with self.assertRaisesRegex(RuntimeError, "duplicate valid time"):
                        writer.submit(root, 300, refl_field=None)
                    sample = driver.reach_output()
                    self.assertIs(sample, driver.reach_output())
                    self.assertIs(borrowed[-1][1]["reach_ids"], sample["reach_id"])
                    self.assertIs(borrowed[-1][1]["discharge"], sample["discharge_m3_s"])
                    self.assertIs(borrowed[-1][1]["stream"], stream)
                    sample["discharge_m3_s"].fill(999.)  # same-stream reuse after capture
                drivers.append(driver)
            self.assertIsNone(current_capture())
            collector.nextgen_points.require_complete()
            self.assertEqual([row[0] for row in borrowed], [7, 22])
            self.assertEqual(len(atmosphere), 4)  # parent and child, each member
            self.assertEqual([str(row[1]["discharge"].dtype) for row in borrowed], ["float64", "float32"])
            frame = next(iter(collector.nextgen_points.frames.values()))
            shard = frame["cards"][drivers[0].device_id]
            shard["done"].synchronize()  # verification at output boundary only
            self.assertTrue(bool(cp.array_equal(shard["count"], cp.asarray([[0, 1, 2, 1]], cp.uint32))))
            self.assertTrue(bool(cp.array_equal(shard["finite"], cp.asarray([2, 2, 2, 1], cp.uint32))))
            self.assertTrue({x.data.mem.ptr for x in drivers[0].arrays.values()}.isdisjoint(
                x.data.mem.ptr for x in drivers[1].arrays.values()))

    def test_scoped_driver_identity_and_unscoped_compatibility(self):
        try:
            import cupy as cp
            from gpuwm.hydro import ResidentHydroDriver, Stage
        except ImportError as error:
            self.skipTest(str(error))
        from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope

        def prepare():
            field = cp.empty((4,), dtype=cp.float64)
            component = SimpleNamespace(arrays={"native_state": field},
                                        restart_identity={"kind": "identity-instrumentation"})
            return ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
                interval_seconds=300, identity={"kind": "identity-instrumentation"},
                after=(Stage("routing", component, lambda dt, *, stream=None: None),))

        drivers = []
        inventory = ResidentInventory()
        for member_id in (7, 22):
            with member_output_scope(MemberOutputCapture(lambda **row: None, member_id)):
                driver = prepare()
            self.assertEqual(driver.member_id, member_id)
            self.assertEqual(driver.restart_identity["member_id"], member_id)
            with self.assertRaisesRegex(ValueError, "member identity"):
                inventory.inspect(driver, member_id=99, device_id=driver.device_id)
            inventory.inspect(driver, member_id=member_id, device_id=driver.device_id)
            drivers.append(driver)
        self.assertEqual([row["member_id"] for row in inventory.receipt()["members"]], [7, 22])
        self.assertNotEqual(drivers[0].arrays["routing/native_state"].data.ptr,
                            drivers[1].arrays["routing/native_state"].data.ptr)
        # Standalone qualification drivers have no scoped member identity.
        unscoped = prepare()
        self.assertIsNone(unscoped.member_id)
        inventory.inspect(unscoped, member_id=99, device_id=unscoped.device_id)


if __name__ == "__main__":
    unittest.main()
