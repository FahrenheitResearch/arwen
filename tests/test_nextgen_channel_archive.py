"""Output-cadence depth transport, independent IDs and durable roster checks."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest


class ChannelArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        cls.cp = cp
        cp.cuda.Device().use()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def driver(self, dtype="float64"):
        cp = self.cp
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            ids = cp.asarray([7, 900, 2**60+3], dtype=cp.int64)
            q = cp.asarray([1, 10, 100], dtype=cp.float32)
            channel_ids = cp.asarray([2**60+3, 7], dtype=cp.int64)
            depth = cp.asarray([.25, .75], dtype=dtype)
            if dtype == "float64":
                depth[0] = cp.nextafter(cp.float64(.25), cp.float64(.5))
        return SimpleNamespace(device_id=0,
            arrays=MappingProxyType(dict(ids=ids, q=q, channel_ids=channel_ids, depth=depth)),
            reach_output=lambda: MappingProxyType(dict(reach_id=ids, discharge_m3_s=q,
                depth_m=q, stream=stream)),
            channel_output=lambda: MappingProxyType(dict(reach_id=channel_ids,
                channel_depth_m=depth, stream=stream))), depth, stream

    def read(self, source):
        from gpuwm.netcdf_bridge import Dataset
        with Dataset(self.root / source["path"]) as file:
            file.set_auto_maskandscale(False)
            # Rust reader transports doubles: reconstruct exact IDs from words on CUDA.
            ids = self.cp.asarray(file.variables["reach_id_words"][:], dtype=self.cp.uint32).reshape(-1).view(self.cp.int64)
            depth = self.cp.asarray(file.variables["channel_depth_m"][:], dtype=source["dtype"])
        return ids, depth

    def test_default_history_captures_depth_before_producer_reuse(self):
        from gpuwm.io.wrfout import PerDomainWrfoutWriters
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        from gpuwm.ensemble.nextgen_mass import MemberReachCapture, reduce_members
        from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
        cp = self.cp
        for mass in (False, True):
            with self.subTest(mass=mass):
                directory = self.root / str(mass)
                directory.mkdir()
                atmospheric = []
                callback = lambda **row: atmospheric.append(row)
                owner = (MemberReachCapture(callback, member_id=3, directory=directory,
                    array_module=cp) if mass else NextGenCollector(SimpleNamespace(root=directory,
                        submit=callback, require_complete=lambda: None, finish_run=lambda: []),
                        (3,), array_module=cp))
                writer = object.__new__(PerDomainWrfoutWriters)
                writer.start_time, writer.output_dir = datetime(2024, 1, 1), directory
                writer._episode_by_grid_id, writer._published_paths = {1: 0, 2: 0}, set()
                writer._captured_paths, writer._archived_paths = [], []
                writer._metadata_by_grid_id, writer._writers = {1: {}, 2: {}}, {}
                driver, depth, stream = self.driver()
                node = SimpleNamespace(state=SimpleNamespace(physics=SimpleNamespace(hydro=driver)),
                    cfg=SimpleNamespace(grid_id=1), clock=SimpleNamespace(tick_den=10))
                child = SimpleNamespace(state=SimpleNamespace(), cfg=SimpleNamespace(grid_id=2),
                                        clock=SimpleNamespace(tick_den=10))
                with member_output_scope(MemberOutputCapture(owner if mass else owner.submit, 3)):
                    writer.submit(child, 300, refl_field=None)
                    writer.submit(node, 300, refl_field=None)
                    with stream:
                        depth.fill(1.25)
                    writer.submit(node, 600, refl_field=None)
                    with stream:
                        depth.fill(999.)
                if mass:
                    document = owner.seal()
                    products = reduce_members(self.root, [dict(member_id=3, status="completed",
                        nextgen_directory=str(mass), nextgen_points=document)],
                        expected_member_ids=(3,), array_module=cp)["products"]
                    manifest_path = self.root / "nextgen/channel-depth-ensemble.json"
                else:
                    products = owner.finish_run()
                    manifest_path = directory / "nextgen/channel-depth-ensemble.json"
                self.assertTrue(manifest_path.is_file(), "default history dropped above-bed depth")
                self.assertIn(str(manifest_path), products)
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["requested_member_ids"], [3])
                self.assertEqual(manifest["comparison"], "gt")
                self.assertEqual(manifest["missing_policy"], "complete_roster")
                self.assertEqual(len(manifest["frames"]), 2)
                for index, frame in enumerate(manifest["frames"]):
                    self.assertEqual(frame["missing_member_ids"], [])
                    self.assertEqual(frame["members"][0]["member_id"], 3)
                    source = frame["members"][0]["source"]
                    self.assertEqual(source["dtype"], "<f8")
                    # Sources are relative to the archive's root, explicitly declared.
                    source = dict(source, path=str(manifest_path.parent / manifest["root"] / source["path"]))
                    ids, values = self.read(source)
                    self.assertTrue(bool(cp.array_equal(ids, cp.asarray([2**60+3, 7], dtype=cp.int64))))
                    expected = cp.asarray([.25, .75] if index == 0 else [1.25, 1.25], dtype=cp.float64)
                    if index == 0:
                        expected[0] = cp.nextafter(cp.float64(.25), cp.float64(.5))
                    self.assertTrue(bool(cp.array_equal(values.view(cp.uint64), expected.view(cp.uint64))))
                self.assertEqual(len(atmospheric), 3)

    def test_missing_depth_keeps_requested_roster(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        owner = NextGenCollector(SimpleNamespace(root=self.root, submit=lambda **kw: None,
            require_complete=lambda: None, finish_run=lambda: []), (0, 1), array_module=self.cp)
        for member in (0, 1):
            driver, depth, stream = self.driver("float32")
            if member == 1:
                driver.channel_output = lambda: None
            else:
                with stream:
                    depth[0], depth[1] = float("nan"), -.25
            owner.submit(state=SimpleNamespace(physics=SimpleNamespace(hydro=driver)),
                member_id=member, valid_time="T", grid_id=1)
        owner.finish_run()
        doc = json.loads((self.root / "nextgen/channel-depth-ensemble.json").read_text())
        self.assertEqual(doc["requested_member_ids"], [0, 1])
        frame = doc["frames"][0]
        self.assertEqual(frame["missing_member_ids"], [1])
        self.assertEqual([row["member_id"] for row in frame["members"]], [0, 1])
        self.assertIsNone(frame["members"][1]["source"])
        self.assertEqual(frame["missing_member_action"], "mask_all_cells")
        source = frame["members"][0]["source"]
        self.assertEqual(source["dtype"], "<f4")
        _, values = self.read(source)
        self.assertTrue(bool(self.cp.isnan(values[0])))
        self.assertEqual(float(values[1].item()), -.25)

    def test_depth_checkpoint_reemit_and_hash_binding(self):
        from gpuwm.ensemble.nextgen_capture import NextGenCollector
        original = SimpleNamespace(root=self.root, submit=lambda **kw: None,
            save_resume=lambda: {"files": []}, restore_resume=lambda: {},
            require_complete=lambda: None, finish_run=lambda: [])
        first = NextGenCollector(original, (0, 1), array_module=self.cp)
        driver, depth, stream = self.driver()
        state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
        first.submit(state=state, member_id=0, valid_time="T", grid_id=1)
        doc = first.save_resume()
        archive = doc["nextgen_channel_depth"]
        source = archive["samples"][0]["source"]
        self.assertIn(source["path"], doc["files"])
        second = NextGenCollector(original, (0, 1), array_module=self.cp)
        second.restore_resume()
        second.submit(state=state, member_id=0, valid_time="T", grid_id=1)  # restored deadline
        with self.assertRaisesRegex(ValueError, "duplicate member/time"):
            second.submit(state=state, member_id=0, valid_time="T", grid_id=1)
        driver2, _, _ = self.driver()
        second.submit(state=SimpleNamespace(physics=SimpleNamespace(hydro=driver2)),
            member_id=1, valid_time="T", grid_id=1)
        second.finish_run()
        path = self.root / source["path"]
        path.write_bytes(path.read_bytes()+b"corrupt")
        third = NextGenCollector(original, (0, 1), array_module=self.cp)
        with self.assertRaisesRegex(ValueError, "depth.*hash"):
            third.restore_resume()


if __name__ == "__main__":
    unittest.main(verbosity=2)
