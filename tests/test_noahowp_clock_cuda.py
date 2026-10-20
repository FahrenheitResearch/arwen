"""Factory clock admission using an actual Rust-initialized Noah-OWP packet.

Set GPUWM_TEST_NOAHOWP_INITIAL to its sealed manifest and resolve the ordinary
initializer cdylib. No scientific fixture or initializer is substituted.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import unittest


@unittest.skipUnless(os.environ.get("GPUWM_TEST_NOAHOWP_INITIAL"),
                     "requires an actual Rust-prepared Noah-OWP initial packet")
class NoahClock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        from gpuwm.hydro.noahowp.initial import load_initial_cuda
        cls.cp = cp
        cls.manifest = Path(os.environ["GPUWM_TEST_NOAHOWP_INITIAL"])
        cls.slabs, cls.metadata = load_initial_cuda(cls.manifest, device_id=0)
        cls.origin = datetime.fromtimestamp(cls.metadata["columns"][0]["start_datetime"],
                                            timezone.utc)
        cls.iso = cls.origin.isoformat()

    def prepare(self, kind="manifest", *, origin=None, mixed=False):
        from gpuwm.hydro.noahowp.factory import prepare_slab
        from gpuwm.hydro.noahowp.runtime import schema
        options = dict(fabric_sha256="clock-test-boundary", time_origin=(
            self.iso if origin is None else origin),
            parameter_sha256=self.metadata["parameter_sha256"])
        components = {}
        if kind == "manifest":
            options["initial_manifest"] = str(self.manifest)
        else:
            slabs = {key: value.copy() for key, value in self.slabs.items()}
            if mixed:
                # Deliberately corrupt one resident clock, on the GPU.
                field = next(f for f in schema()["fields"]
                             if f["name"] == "domain.start_datetime")
                slabs[field["dtype"]][field["offset"], -1] += self.cp.float64(3600)
            arrays = dict(slabs)
            components["input"] = SimpleNamespace(arrays=arrays)
            if kind == "slabs":
                options["slabs"] = {key: "input:" + key for key in slabs}
            else:
                for field in schema()["fields"]:
                    view = slabs[field["dtype"]][field["offset"]:
                        field["offset"] + field["count"]].T
                    arrays[field["name"]] = view if field["bounds"] else view[:, 0]
                options["fields"] = {f["name"]: "input:" + f["name"]
                                     for f in schema()["fields"]}
        return prepare_slab(settings={"interval_seconds": self.metadata["columns"][0][
            "dt_seconds"], "numerics": os.environ.get("GPUWM_NUMERICS", "fast")},
            options=options, state=None, physics=None, device_id=0,
            mode="standalone", components=components)

    def test_wrong_epoch_refused_for_every_input_path(self):
        for kind in ("manifest", "slabs", "fields"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "start.*time_origin"):
                self.prepare(kind, origin=(self.origin + timedelta(hours=1)).isoformat())

    def test_every_resident_column_is_checked(self):
        self.assertGreater(len(self.metadata["columns"]), 1)
        for kind in ("slabs", "fields"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "start.*time_origin"):
                self.prepare(kind, mixed=True)

    def test_invalid_epoch_names_the_binding(self):
        for value in ("not-an-epoch", 123, "2015-12-01T00:00:00+bogus"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "time_origin"):
                self.prepare(origin=value)

    def test_valid_paths_preserve_all_initial_payload_bytes(self):
        for kind in ("manifest", "slabs", "fields"):
            with self.subTest(kind=kind):
                component = self.prepare(kind)
                for dtype, spec in self.metadata["files"].items():
                    payload = component.arrays["slab/" + dtype].get().tobytes()
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), spec["sha256"])
                    self.assertEqual(len(payload), spec["bytes"])

    def test_equivalent_utc_offset_is_accepted(self):
        origin = self.origin.astimezone(timezone(timedelta(hours=1))).isoformat()
        self.prepare(origin=origin)

    def test_naive_timestamp_retains_utc_convention(self):
        self.prepare(origin=self.origin.replace(tzinfo=None).isoformat())


if __name__ == "__main__":
    unittest.main()
