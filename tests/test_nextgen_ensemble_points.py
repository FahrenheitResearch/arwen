"""Targeted synthetic CUDA/output contract checks; not native replay evidence."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np

from gpuwm.ensemble.nextgen_points import ReachAccumulator, ReachSample, product_options


class Options(unittest.TestCase):
    def test_policy(self):
        for bad in ([float("nan")], [-1], [1, 1], [True]):
            with self.assertRaises(ValueError):
                product_options({"thresholds_m3_s": bad})
        from gpuwm.ensemble.request import EnsembleRequest
        request = EnsembleRequest.from_mapping({"members": 2, "hydro": {"thresholds_m3_s": [2.]}})
        self.assertEqual(EnsembleRequest.from_mapping(request.receipt()).receipt(), request.receipt())


class Device(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
        except ImportError:
            raise unittest.SkipTest("CUDA contract checks require CuPy")
        cls.cp = cupy
        cls.cp.cuda.Device().use()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def sample(self, values, dtype="float64", ids=None, stream=None):
        cp = self.cp
        return ReachSample(cp.asarray(ids if ids is not None else [2**60+3, 7, 42, 99], dtype=cp.int64),
                           cp.asarray(values, dtype=dtype), stream=stream)

    def read(self, path):
        from gpuwm.netcdf_bridge import Dataset
        with Dataset(path) as source:
            source.set_auto_maskandscale(False)
            return {k: np.asarray(v[:]) for k, v in source.variables.items()}

    def driver(self, sample):
        arrays = {"reach_id": sample.reach_id, "discharge": sample.discharge_m3_s}
        if sample.channel_depth_m is not None:
            arrays["depth"] = sample.channel_depth_m
        return SimpleNamespace(device_id=int(sample.reach_id.device.id), arrays=MappingProxyType(arrays),
                               reach_output=lambda: sample)

    def test_native_precision_streams_and_nonfinite(self):
        cp = self.cp
        points = ReachAccumulator((4, 7), options={"thresholds_m3_s": [1., 2.]}, array_module=cp)
        above = np.nextafter(1., 2.)
        below = np.nextafter(1., 0.)
        for member, values in ((4, [below, above, np.nan, 2.]), (7, [1., 0., 3., 2.])):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                sample = self.sample(values, stream=stream)
                points.accept(member_id=member, valid_time="T", sample=sample)
                sample.discharge_m3_s.fill(999.)  # borrowing ends after stream-ordered capture
        data = self.read(points.finish(self.root)[0])
        np.testing.assert_array_equal(data["exceedance_count"], [[1, 1, 1, 2], [0, 0, 1, 2]])
        np.testing.assert_array_equal(data["finite_count"], [2, 2, 1, 2])
        np.testing.assert_array_equal(data["probability"], [[.5, .5, np.nan, 1.], [0., 0., np.nan, 1.]])
        document = json.loads((self.root / "reach-ensemble.json").read_text())
        frame = document["frames"][0]
        self.assertEqual(frame["requested_member_ids"], [4, 7])
        self.assertEqual(frame["comparison"], "ge")
        self.assertEqual(frame["missing_policy"], "complete_roster")
        self.assertEqual(frame["probability"]["shape"], [2, 4])
        np.testing.assert_array_equal(data["feature_id_words"].astype(np.uint32).reshape(-1).view(np.int64),
                                      [2**60+3, 7, 42, 99])
        from gpuwm.netcdf_bridge import Dataset
        with Dataset(points.products[0]) as source:
            self.assertEqual(json.loads(source.getncattr("requested_member_ids")), [4, 7])
            self.assertEqual(source.getncattr("missing_policy"), "complete_roster")
        self.assertEqual(points.finish(self.root), points.products)

    def test_float32_threshold_contract_preserves_native_float64_flow(self):
        cp = self.cp
        points = ReachAccumulator((0, 1), options={"thresholds_m3_s": [.1]}, array_module=cp)
        threshold = cp.asarray([.1], dtype=cp.float64).astype(cp.float32).astype(cp.float64)
        q = cp.concatenate((cp.asarray([.1], dtype=cp.float64),
                            cp.nextafter(threshold, cp.zeros_like(threshold)), threshold,
                            cp.asarray([float("nan")], dtype=cp.float64)))
        points.accept(member_id=0, valid_time="T", sample=ReachSample(
            cp.asarray([2**60+3, 7, 42, 99], dtype=cp.int64), q))
        snapshot = points.save(self.root)
        resumed = ReachAccumulator((0, 1), options=points.options, array_module=cp)
        resumed.restore(self.root, snapshot)
        resumed.accept(member_id=1, valid_time="T", sample=self.sample([.1]*4, "float32"))
        data = self.read(resumed.finish(self.root)[0])
        np.testing.assert_array_equal(data["exceedance_count"], [[1, 1, 2, 1]])
        np.testing.assert_array_equal(data["probability"], [[.5, .5, 1., np.nan]])
        np.testing.assert_array_equal(data["discharge_threshold"], cp.asnumpy(threshold))
        frame = resumed.published_frames[0]
        self.assertEqual(frame["threshold_m3_s"]["dtype"], "<f4")
        self.assertEqual(resumed.frames["T"]["cards"][0]["threshold"].dtype, cp.float32)
        from gpuwm.netcdf_bridge import Dataset
        with Dataset(resumed.products[0]) as source:
            self.assertEqual(source.variables["discharge_threshold"].dtype, np.dtype("float32"))
        old = dict(snapshot, schema="arwen.nextgen.reach-ensemble/v1")
        fresh = ReachAccumulator((0, 1), options=points.options, array_module=cp)
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            fresh.restore(self.root, old)
        old_member = dict(old, members=[0])
        with self.assertRaisesRegex(ValueError, "shard differs"):
            fresh.merge_member(self.root, old_member, 0)
        for member, sample in ((0, ReachSample(cp.asarray([2**60+3, 7, 42, 99], dtype=cp.int64), q)),
                               (1, self.sample([.1]*4, "float32"))):
            worker = ReachAccumulator((member,), options=points.options, array_module=cp)
            worker.accept(member_id=member, valid_time="T", sample=sample)
            fresh.merge_member(self.root, worker.save(self.root), member)
        mass = self.read(fresh.finish(self.root / "mass")[0])
        np.testing.assert_array_equal(mass["exceedance_count"], data["exceedance_count"])
        np.testing.assert_array_equal(mass["probability"], data["probability"])

    def test_threshold_rounding_refuses_overflow_and_collapsed_axis(self):
        for thresholds in ([1e100], [1., 1.000000001]):
            points = ReachAccumulator((0,), options={"thresholds_m3_s": thresholds}, array_module=self.cp)
            with self.assertRaisesRegex(ValueError, "float32 rounding"):
                points.accept(member_id=0, valid_time="T", sample=self.sample([1, 2, 3, 4]))

    def test_partial_checkpoint_exact_ids_and_resume(self):
        cp = self.cp
        points = ReachAccumulator((0, 1, 2), array_module=cp)
        sample = self.sample([1, 10, 100, 0], "float32")
        points.accept(member_id=0, valid_time="T", sample=sample)
        with self.assertRaises(RuntimeError):
            points.finish(self.root)
        snapshot = points.save(self.root)
        continued = ReachAccumulator((0, 1, 2), array_module=cp)
        continued.restore(self.root, snapshot)
        ids = continued.frames["T"]["cards"][0]["ids"]
        np.testing.assert_array_equal(cp.asnumpy(ids), [2**60+3, 7, 42, 99])
        continued.accept(member_id=0, valid_time="T", sample=sample)  # checkpoint frame reemit
        with self.assertRaises(ValueError):
            continued.accept(member_id=0, valid_time="T", sample=sample)
        for member in (1, 2):
            continued.accept(member_id=member, valid_time="T", sample=sample)
        expected = self.read(continued.finish(self.root)[0])
        np.testing.assert_array_equal(expected["exceedance_count"], [[3, 3, 3, 0], [0, 3, 3, 0], [0, 0, 3, 0]])
        self.assertTrue(np.all(expected["finite_count"] == 3))

    def test_axis_changes_refused(self):
        points = ReachAccumulator((0, 1), array_module=self.cp)
        points.accept(member_id=0, valid_time="T", sample=self.sample([1, 2, 3, 4]))
        points.accept(member_id=1, valid_time="T", sample=self.sample([1, 2, 3, 4], ids=[3, 7, 42, 99]))
        with self.assertRaisesRegex(ValueError, "feature IDs"):
            points.finish(self.root)
        self.assertFalse(list(self.root.glob("*.nc")))

    def test_depth_is_borrowed_from_producer(self):
        cp = self.cp
        seen = []
        points = ReachAccumulator((0,), array_module=cp, depth_sink=lambda **row: seen.append(row))
        original = self.sample([1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "channel_depth_m"):
            points.accept(member_id=0, valid_time="T", sample=original)
        self.assertFalse(points.frames)
        depth = cp.asarray([.1, .2, .3, .4], dtype=cp.float64)
        points.accept(member_id=0, valid_time="T", sample=ReachSample(original.reach_id,
                      original.discharge_m3_s, channel_depth_m=depth))
        self.assertIs(seen[0]["channel_depth_m"], depth)

    def test_parent_output_hook(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        calls = []
        original = SimpleNamespace(submit=lambda **kw: calls.append(kw), root=self.root)
        collector = NextGenCollector(original, (0,), array_module=self.cp)
        sample = self.sample([1, 2, 3, 4])
        driver = self.driver(sample)
        driver.reach_output = lambda: MappingProxyType({"reach_id": sample.reach_id,
            "discharge_m3_s": sample.discharge_m3_s, "velocity_m_s": sample.discharge_m3_s,
            "depth_m": sample.discharge_m3_s})  # reservoir elevations are ignored
        state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
        collector.submit(state=state, valid_time="T", grid_id=2, member_id=0)
        self.assertFalse(collector.nextgen_points.frames)
        collector.submit(state=state, valid_time="T", grid_id=1, member_id=0)
        self.assertEqual(len(calls), 2)
        collector.nextgen_points.require_complete()

    def test_explicit_current_capture_hydro_api(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.runtime_context import MemberOutputCapture, current_capture, member_output_scope
        cp = self.cp
        borrowed = []
        class Collector(NextGenCollector):
            def submit_hydro(self, **row):
                borrowed.append(row)
                return super().submit_hydro(**row)
        collector = Collector(SimpleNamespace(submit=lambda **kw: None), (7,), array_module=cp)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            sample = self.sample([np.nextafter(1., 0.), 1., np.nan, 2.], stream=stream)
            with member_output_scope(MemberOutputCapture(collector.submit, 7)):
                current_capture().submit_hydro(valid_time="2024-01-01T00:00:00Z",
                    reach_ids=sample.reach_id, discharge=sample.discharge_m3_s, stream=stream)
        self.assertIs(borrowed[0]["reach_ids"], sample.reach_id)
        self.assertIs(borrowed[0]["discharge"], sample.discharge_m3_s)
        self.assertIs(borrowed[0]["stream"], stream)
        data = self.read(collector.nextgen_points.finish(self.root)[0])
        np.testing.assert_array_equal(data["exceedance_count"][0], [0, 1, 0, 1])
        np.testing.assert_array_equal(data["finite_count"], [1, 1, 0, 1])
        np.testing.assert_array_equal(data["feature_id_words"].astype(np.uint32).reshape(-1).view(np.int64),
                                      [2**60+3, 7, 42, 99])

    def test_history_deadline_calls_hydro_api_once_for_both_collectors(self):
        from datetime import datetime
        from gpuwm.io.wrfout import PerDomainWrfoutWriters
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.nextgen_mass import MemberReachCapture
        from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
        cp = self.cp
        for mass in (False, True):
            atmospheric, borrowed = [], []
            callback = lambda **row: atmospheric.append(row)
            owner = (MemberReachCapture(callback, member_id=3, directory=self.root,
                         array_module=cp) if mass else NextGenCollector(
                         SimpleNamespace(submit=callback), (3,), array_module=cp))
            class Capture(MemberOutputCapture):
                def submit_hydro(self, **row):
                    borrowed.append(row)
                    return super().submit_hydro(**row)
            writer = object.__new__(PerDomainWrfoutWriters)
            writer.start_time = datetime(2024, 1, 1)
            writer.output_dir = self.root
            writer._episode_by_grid_id = {1: 0, 2: 0}
            writer._published_paths = set()
            writer._captured_paths, writer._archived_paths = [], []
            writer._metadata_by_grid_id = {1: {}, 2: {}}
            writer._writers = {}
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                sample = self.sample([1, 2, 3, 4], stream=stream)
            driver = self.driver(sample)
            # The actual resident provider is an immutable full-network map.
            # Its reservoir depth_m must not become above-bed channel depth.
            driver.reach_output = lambda: MappingProxyType(dict(reach_id=sample.reach_id,
                discharge_m3_s=sample.discharge_m3_s, stream=stream, depth_m=sample.discharge_m3_s))
            state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
            node = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=1),
                                   clock=SimpleNamespace(tick_den=10))
            child = SimpleNamespace(state=SimpleNamespace(), cfg=SimpleNamespace(grid_id=2),
                                    clock=SimpleNamespace(tick_den=10))
            capture = Capture(owner if mass else owner.submit, 3)
            points = owner.points if mass else owner.nextgen_points
            self.assertFalse(points.frames)
            with member_output_scope(capture):
                writer.submit(child, 300, refl_field=None)
                self.assertFalse(borrowed)
                writer.submit(node, 300, refl_field=None)
                with self.assertRaisesRegex(RuntimeError, "duplicate valid time"):
                    writer.submit(node, 300, refl_field=None)
            points.require_complete()
            self.assertEqual(len(borrowed), 1)
            self.assertEqual(len(atmospheric), 2)
            self.assertIs(borrowed[0]["reach_ids"], sample.reach_id)
            self.assertIs(borrowed[0]["discharge"], sample.discharge_m3_s)
            self.assertIs(borrowed[0]["stream"], stream)
            self.assertIsNone(borrowed[0]["channel_depth_m"])
            # A second explicit call at this UTC remains a duplicate error;
            # the new forwarding API must not hide repeated history output.
            with member_output_scope(capture):
                with self.assertRaisesRegex(ValueError, "duplicate member/time"):
                    capture.submit_hydro(**borrowed[0])

    def test_channel_only_depth_uses_its_own_axis(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        cp = self.cp
        seen = []
        collector = NextGenCollector(SimpleNamespace(root=self.root, submit=lambda **kw: None), (0,),
            array_module=cp, channel_depth_sink=lambda **row: seen.append(row))
        sample = self.sample([1, 2, 3, 4])  # full network axis includes reservoirs
        driver = self.driver(sample)
        ids = cp.asarray([7, 99], dtype=cp.int64)
        depth = cp.asarray([.2, .3], dtype=cp.float64)
        stream = cp.cuda.Stream(non_blocking=True)
        driver.channel_output = lambda: MappingProxyType({"reach_id": ids,
            "channel_depth_m": depth, "stream": stream})
        state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
        collector.submit(state=state, valid_time="T", grid_id=1, member_id=0)
        self.assertIs(seen[0]["feature_id"], ids)
        self.assertIs(seen[0]["channel_depth_m"], depth)
        self.assertIs(seen[0]["stream"], stream)
        self.assertEqual(collector.nextgen_points.frames["T"]["n"], 4)
        self.assertEqual(seen[0]["feature_id"].size, 2)

    def test_mass_integer_shards(self):
        from gpuwm.ensemble.nextgen_mass import MemberReachCapture, reduce_members
        rows = []
        for member, values in ((5, [1, 2, 100, 0]), (9, [0, 20, 101, 1])):
            directory = self.root / str(member)
            capture = MemberReachCapture(lambda **kw: None, member_id=member,
                directory=directory, array_module=self.cp)
            sample = self.sample(values)
            state = SimpleNamespace(physics=SimpleNamespace(hydro=self.driver(sample)))
            capture(state=state, valid_time="T", grid_id=1, member_id=member)
            rows.append(dict(member_id=member, status="completed", nextgen_directory=str(member),
                             nextgen_points=capture.seal()))
        with self.assertRaisesRegex(ValueError, "full requested roster"):
            reduce_members(self.root, rows[:1], expected_member_ids=(5, 9), array_module=self.cp)
        result = reduce_members(self.root, rows, expected_member_ids=(5, 9), array_module=self.cp)
        data = self.read(result["products"][0])
        np.testing.assert_array_equal(data["probability"], [[.5, 1, 1, .5], [0, .5, 1, 0], [0, 0, 1, 0]])

    def test_combined_checkpoint_transaction(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.restart_roster import atomic_json
        restored = []
        def save():
            document = {"files": []}
            atomic_json(self.root / ".ensemble-resume" / "collector.json", document)
            return document
        original = SimpleNamespace(root=self.root, save_resume=save,
            restore_resume=lambda: restored.append(True))
        first = NextGenCollector(original, (0, 1), array_module=self.cp)
        sample = self.sample([1, 10, 100, 0])
        first.submit_hydro(member_id=0, valid_time="T", sample=sample)
        document = first.save_resume()
        self.assertIn(document["nextgen_points"]["arrays"], document["files"])
        second = NextGenCollector(original, (0, 1), array_module=self.cp)
        second.restore_resume()
        self.assertEqual(restored, [True])
        second.submit_hydro(member_id=1, valid_time="T", sample=sample)
        second.nextgen_points.require_complete()

    def test_production_session_keeps_member_states(self):
        from datetime import datetime
        from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
        from gpuwm.ensemble.packing import CardBudget
        from gpuwm.ensemble.production import PreparedEnsembleSession
        from gpuwm.ensemble.runtime_context import current_capture
        from gpuwm.hydro import runtime as hydro_runtime
        cp = self.cp
        config = self.root / "experiment.toml"
        config.write_text("[hydro]\nenabled=true\nmember_tag=-1\n")
        cfg = SimpleNamespace(dt=3, use_adaptive_time_step=True, mp_physics=16)
        exp = SimpleNamespace(run_seconds=60, start_time=datetime(2024, 1, 1),
                              root=SimpleNamespace(run=cfg))
        inputs = SimpleNamespace(experiment=exp, boundary_interval_seconds=3600,
                                 experiment_config=config)
        retained = []
        member_inputs = {}
        for member in range(3):
            path = self.root / f"member-{member}.toml"
            path.write_text(f"[hydro]\nenabled=true\nmember_tag={member}\n")
            member_inputs[member] = SimpleNamespace(experiment=exp,
                boundary_interval_seconds=3600, experiment_config=path)
        original = SimpleNamespace(root=self.root, submit=lambda **kw: None,
            finish_run=lambda: [], require_complete=lambda: None)
        def runner(inputs, **kwargs):
            capture = current_capture()
            self.assertEqual(hydro_runtime.current_settings()["member_tag"], capture.member_id)
            sample = self.sample([capture.member_id] * 4)
            state = SimpleNamespace(physics=SimpleNamespace(hydro=self.driver(sample)))
            capture.initialize_callback(state=state, cfg=SimpleNamespace(grid_id=1))
            retained.append(state)
            capture.submit(state=state, streamed=None, metadata={}, refl_field=None,
                valid_time=exp.start_time, grid_id=1)
            return {"status": "PASS"}
        run = PreparedEnsembleSession({"members": 3}, output_directory=self.root,
            collector=original, cards=(CardBudget(0, 1 << 30),), array_module=cp,
            device_scope=cp.cuda.Device,
            memory_model=EnsembleMemoryModel((MemoryComponent("synthetic", "state", fixed_bytes=100),)),
            input_provider=lambda **row: member_inputs[row["member_id"]])
        ambient = {"enabled": True, "member_tag": -2}
        with hydro_runtime.settings_scope(ambient):
            receipt = run.run_prepared(hydro_runtime.forecast_scope(runner), inputs)
            self.assertIs(hydro_runtime.current_settings(), ambient)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len({state.physics.hydro.reach_output().discharge_m3_s.data.ptr for state in retained}), 3)
        self.assertEqual(receipt["nextgen"]["members"], [0, 1, 2])
        self.assertEqual(len(receipt["nextgen"]["resident_inventory"]["members"]), 3)
        self.assertFalse(receipt["nextgen"]["resident_inventory"]["pre_factory_admission"])
        data = self.read(receipt["products"][0])
        np.testing.assert_array_equal(data["exceedance_count"][0], [2, 2, 2, 2])

    def test_resident_allocation_inventory_and_live_member_ownership(self):
        from gpuwm.ensemble.nextgen_inventory import ResidentInventory
        cp = self.cp
        owner = ResidentInventory()
        base = cp.empty(1000, dtype=cp.float64)
        driver = SimpleNamespace(device_id=0, arrays=MappingProxyType({
            "even": base[::2], "reverse": base[::-1], "matrix": base.reshape(20, 50).T}))
        row = owner.inspect(driver, member_id=4, device_id=0)
        self.assertEqual(row["logical_bytes"], 20000)
        self.assertEqual(row["allocation_bytes"], base.data.mem.size)
        self.assertEqual(len(row["allocations"]), 1)
        with self.assertRaisesRegex(ValueError, "share live resident storage"):
            owner.inspect(SimpleNamespace(device_id=0, arrays={"disjoint": base[1::2]}),
                          member_id=7, device_id=0)
        with self.assertRaisesRegex(ValueError, "physical card"):
            owner.inspect(driver, member_id=4, device_id=1)
        owner.inspect(driver, member_id=4, device_id=0)  # same member rechecks are legal
        # Holding allocation metadata must not keep the completed arrays alive.
        pointer = base.data
        del driver, base
        import gc
        gc.collect()
        replacement = cp.ndarray((1000,), dtype=cp.float64, memptr=pointer)
        owner.inspect(SimpleNamespace(device_id=0, arrays={"replacement": replacement}),
                      member_id=7, device_id=0)

    def test_shared_state_refused_before_probability_capture(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        collector = NextGenCollector(SimpleNamespace(submit=lambda **kw: None), (0, 1), array_module=self.cp)
        # Reconstructed children pass RunConfig (no grid_id); their clock
        # identifies the child and they must not acquire a parent hydro owner.
        collector.bind_member(member_id=0, state=SimpleNamespace(), cfg=SimpleNamespace(),
                              clock=SimpleNamespace(spec=SimpleNamespace(grid_id=2)))
        sample = self.sample([1, 2, 3, 4])
        state = SimpleNamespace(physics=SimpleNamespace(hydro=self.driver(sample)))
        collector.bind_member(member_id=0, state=state)
        with self.assertRaisesRegex(ValueError, "share live resident storage"):
            collector.bind_member(member_id=1, state=state)
        self.assertFalse(collector.nextgen_points.frames)

    def test_measured_metadata_inventory(self):
        from gpuwm.ensemble.nextgen_inventory import ResidentInventory
        cp = self.cp
        owner = ResidentInventory()
        base = cp.empty((128, 128), dtype=cp.float64)
        driver = SimpleNamespace(device_id=0, arrays=MappingProxyType({
            f"component{n}/workspace": base[n::32, ::-1] for n in range(32)}))
        owner.inspect(driver, member_id=0, device_id=0)
        started = time.perf_counter()
        for _ in range(100):
            row = owner.inspect(driver, member_id=0, device_id=0)
        seconds = (time.perf_counter()-started)/100
        receipt = dict(kind="initialized host metadata inventory; not routing or pre-factory admission",
            carriers=32, allocation_bytes=row["allocation_bytes"], logical_bytes=row["logical_bytes"],
            mean_wall_seconds=seconds, repetitions=100, data_transfers=0, cuda_launches=0,
            card=cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
        print("NEXTGEN_INVENTORY " + json.dumps(receipt, sort_keys=True), flush=True)
        if os.environ.get("NEXTGEN_INVENTORY_RECEIPT"):
            Path(os.environ["NEXTGEN_INVENTORY_RECEIPT"]).write_text(json.dumps(receipt, indent=2)+"\n")

    def test_measured_output_deadline(self):
        cp = self.cp
        # Synthetic throughput only. No hydrological step or reference claim.
        n, members = 100000, 36
        points = ReachAccumulator(range(members), array_module=cp)
        sample = ReachSample(cp.arange(n, dtype=cp.int64), cp.ones(n, dtype=cp.float32))
        points.accept(member_id=0, valid_time="warm", sample=sample)
        cp.cuda.get_current_stream().synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        started = time.perf_counter()
        begin.record()
        for member in range(1, members):
            points.accept(member_id=member, valid_time="warm", sample=sample)
        end.record()
        end.synchronize()
        capture_wall = time.perf_counter()-started
        capture_ms = cp.cuda.get_elapsed_time(begin, end)
        started = time.perf_counter()
        paths = points.finish(self.root)
        write_seconds = time.perf_counter()-started
        data = self.read(paths[0])
        np.testing.assert_array_equal(data["probability"][0], np.ones(n, dtype=np.float32))
        np.testing.assert_array_equal(data["exceedance_count"][0], np.full(n, members, dtype=np.uint32))
        receipt = dict(kind="synthetic output reduction; not hydro step timing", reaches=n,
            members=members, timed_members=members-1, thresholds=3, capture_cuda_ms=capture_ms,
            capture_wall_seconds=capture_wall, merge_and_rust_write_seconds=write_seconds,
            card=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            parallelism="one CUDA thread/reach, 256 threads/block; event-ordered member streams",
            cuda_graphs=False, mode="fast", shared_pool=True)
        print(json.dumps(receipt), flush=True)
        if os.environ.get("NEXTGEN_TEST_RECEIPT"):
            Path(os.environ["NEXTGEN_TEST_RECEIPT"]).write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
