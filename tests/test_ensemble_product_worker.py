"""CPU product process boundaries preserve parent authority and typed inputs."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from gpuwm.ensemble import product_worker as worker

WORKSPACE = Path(__file__).resolve().parents[1]


class ProductWorkerTest(unittest.TestCase):
    def fixture(self, root):
        coordinate = root / "coordinates.nc"
        coordinate.write_bytes(b"committed coordinate words")
        renderer = root / "native-renderer"
        renderer.write_bytes(b"native executable fixture")
        packs = []
        for member in (11, 7):
            path = root / f"member-{member}.nc"
            path.write_bytes(b"committed spill " + bytes([member]))
            identity, _ = worker._identity(path)
            packs.append(dict(identity, path=str(path), member_ids=[member]))
        identity, _ = worker._identity(coordinate)
        return {"schema": worker.REQUEST_SCHEMA, "root": str(root), "domain": "d01",
                "valid_time": "2026-10-04T03:00:00Z", "shape": [2, 2],
                "member_order": [11, 7], "member_metadata": [{"member_id": 11, "seed": (1 << 64) - 1}],
                "requests": [{"field": "wind10", "units": "m s-1", "thresholds": [10.0],
                              "comparison": "ge", "paintball": True, "spaghetti": False,
                              "postage_stamp": True}], "events": {}, "tile_rows": 17,
                "renderer": str(renderer),
                "coordinate_file": dict(identity, path=str(coordinate)), "available_bytes": 10_000,
                "frame": {"status": "pending", "valid_time": "2026-10-04T03:00:00Z",
                          "members_received": [7, 11], "available_fields": ["wind10"],
                          "unavailable_fields": [], "packs": packs}}

    def test_hash_mismatch_and_duplicate_member_fail_before_replay(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            request = self.fixture(root)
            damaged = deepcopy(request)
            damaged["frame"]["packs"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "committed hash"):
                worker.execute_request(damaged)
            duplicated = deepcopy(request)
            duplicated["frame"]["packs"][1]["member_ids"] = [11]
            with self.assertRaisesRegex(ValueError, "spill roster"):
                worker.execute_request(duplicated)

    def test_source_escape_is_rejected_before_open(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "escapes"):
                worker._owned_path(root, str(root.parent / "foreign.nc"))
            with self.assertRaisesRegex(ValueError, "owned output root"):
                worker._owned_path(root, "../foreign.nc")

    def test_child_preserves_coordinate_bits_spills_and_parent_manifest(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            request = self.fixture(root)
            request["member_metadata"] = [{"member_id": 11, "seed": (1 << 64)-1, "weight": .75}, {"member_id": 7, "weight": .25}]
            authority = root / "d01" / "ensemble-manifest.json"
            authority.parent.mkdir()
            authority.write_bytes(b"newer parent authority must survive")
            words = np.array([0x80000000, 0x3F800000, 0x7FC00001, 1], np.uint32).reshape(2, 2)
            values = words.view(np.float32)
            dataset = SimpleNamespace(variables={"XLAT": values, "XLONG": values})

            def finish(spool, valid, **options):
                self.assertFalse(options["retire_diagnostics"])
                self.assertFalse(spool.gpu_replay)
                self.assertEqual(spool.member_order, (11, 7))
                self.assertEqual(spool.member_weights, (.75, .25))
                np.testing.assert_array_equal(spool.latitude.view(np.uint32), words)
                self.assertEqual(spool.member_metadata[0]["seed"], (1 << 64) - 1)
                self.assertEqual(spool.requests[0].thresholds, (10.0,))
                self.assertEqual(spool.tile_rows, 17)
                spool._manifest()
                product = root / "d01" / "ensemble" / "products" / "frame.nc"
                product.parent.mkdir(parents=True)
                product.write_bytes(b"exact existing CDF5 bytes")
                image = root / "maps" / "native.png"
                image.parent.mkdir()
                image.write_bytes(b"exact existing Rust PNG bytes")
                return {"status": "complete", "valid_time": valid,
                        "products": [product.relative_to(root).as_posix()],
                        "maps": [image.relative_to(root).as_posix()]}

            with patch("gpuwm.netcdf_bridge.open_dataset", return_value=nullcontext(dataset)), \
                    patch("gpuwm.ensemble.batch_product_output.NativeDiagnosticSpool._finish", finish):
                result = worker.execute_request(request)
            self.assertEqual(authority.read_bytes(), b"newer parent authority must survive")
            self.assertEqual(result["schema"], worker.RESULT_SCHEMA)
            self.assertEqual(len(result["artifactRecords"]), 2)
            for record in result["artifactRecords"]:
                self.assertEqual(hashlib.sha256((root / record["path"]).read_bytes()).hexdigest(), record["sha256"])
            self.assertTrue(all(Path(pack["path"]).is_file() for pack in request["frame"]["packs"]))
            self.assertEqual(request["frame"]["status"], "pending")

    def test_compound_describe_metadata_roundtrips_without_new_weather_math(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            request = self.fixture(root)
            request["events"] = {"fire_weather": [
                {"field": "wind10", "units": "m s-1", "threshold": 10.0, "comparison": "ge"},
                {"field": "humidity2", "units": "%", "threshold": 20.0, "comparison": "le"}]}
            request["requests"].append({"field": "fire_weather", "units": "1", "thresholds": [0.5],
                "comparison": "ge", "paintball": True, "spaghetti": True, "postage_stamp": True})
            values = np.zeros((2, 2), np.float32)
            dataset = SimpleNamespace(variables={"XLAT": values, "XLONG": values})
            def finish(spool, valid, **unused):
                self.assertEqual(spool.events, request["events"])
                self.assertEqual([row.describe() for row in spool.requests], request["requests"])
                return {"status": "complete", "valid_time": valid, "products": [], "maps": []}
            with patch("gpuwm.netcdf_bridge.open_dataset", return_value=nullcontext(dataset)), \
                    patch("gpuwm.ensemble.batch_product_output.NativeDiagnosticSpool._finish", finish):
                self.assertEqual(worker.execute_request(request)["artifactRecords"], [])

    def test_native_wrf_clock_and_moving_domain_keep_their_exact_identity(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            request = self.fixture(root)
            request["valid_time"] = request["frame"]["valid_time"] = "2026-10-04_03:00:00"
            request["domain"] = "d03-episode-001-grid-012345abcdef"
            values = np.zeros((2, 2), np.float32)
            dataset = SimpleNamespace(variables={"XLAT": values, "XLONG": values})
            def finish(spool, valid, **unused):
                self.assertEqual(valid, "2026-10-04_03:00:00")
                self.assertEqual(spool.domain, request["domain"])
                return {"status": "complete", "valid_time": valid, "products": [], "maps": []}
            with patch("gpuwm.netcdf_bridge.open_dataset", return_value=nullcontext(dataset)), \
                    patch("gpuwm.ensemble.batch_product_output.NativeDiagnosticSpool._finish", finish):
                result = worker.execute_request(request)
            self.assertEqual(result["domain"], request["domain"])
            self.assertEqual(result["valid_time"], request["valid_time"])

    def test_coordinate_replacement_during_rust_read_fails_before_finish(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            request = self.fixture(root)
            coordinate = Path(request["coordinate_file"]["path"])
            class ChangedCoordinates:
                def __enter__(self):
                    coordinate.write_bytes(b"changed coordinate payload")
                    return SimpleNamespace(variables={"XLAT": np.ones((2, 2), np.float32),
                                                       "XLONG": np.ones((2, 2), np.float32)})
                def __exit__(self, *unused):
                    return False
            with patch("gpuwm.netcdf_bridge.open_dataset", return_value=ChangedCoordinates()), \
                    patch("gpuwm.ensemble.batch_product_output.NativeDiagnosticSpool._finish") as finish:
                with self.assertRaisesRegex(ValueError, "committed identity"):
                    worker.execute_request(request)
                finish.assert_not_called()

    def test_atomic_result_has_no_pending_file_and_preserves_integer_identity(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            result = root / "result.json"
            expected = {"schema": worker.RESULT_SCHEMA, "member": (1 << 64) - 1}
            worker._atomic_json(result, expected)
            self.assertEqual(json.loads(result.read_text()), expected)
            self.assertFalse(result.with_suffix(".json.pending").exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux owned process-group guard")
    def test_termination_stops_the_owned_rust_child_group(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary).resolve()
            ready = root / "ready.json"
            script = "\n".join((
                "import json, os, pathlib, subprocess, sys, time",
                "from gpuwm.ensemble.product_worker import _parent_guard",
                "_parent_guard(int(sys.argv[1]))",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])",
                "pathlib.Path(sys.argv[2]).write_text(json.dumps({'child': child.pid}))",
                "time.sleep(30)",
            ))
            process = subprocess.Popen([sys.executable, "-c", script, str(os.getpid()), str(ready)], start_new_session=True)
            try:
                import time
                deadline = time.monotonic() + 5
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.is_file())
                child_pid = json.loads(ready.read_text())["child"]
                process.terminate()
                self.assertEqual(process.wait(timeout=5), -signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status = Path(f"/proc/{child_pid}/stat")
                    if not status.exists() or status.read_text().split()[2] == "Z":
                        break
                    time.sleep(.02)
                else:
                    self.fail("owned Rust subprocess survived worker termination")
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)



# fix289/ens-symlink: an ensemble output folder reached through a symlink.
# The parent spells request, result and spill paths through its own output
# folder while the worker holds the resolved root, so every product hour of
# such a run was refused as escaping the output root.


def _symlinked_output(tmp_path):
    """Return (resolved real output folder, symlink that reaches it)."""
    real = tmp_path / "real-output"
    real.mkdir()
    alias = tmp_path / "linked-output"
    try:
        os.symlink(real, alias, target_is_directory=True)
    except OSError as error:
        # Windows without Developer Mode or the symlink privilege refuses
        # os.symlink (WinError 1314); every other platform must create it.
        if os.name != "nt":
            raise
        pytest.skip(f"this Windows account may not create symlinks: {error}")
    return real.resolve(), alias


def _patched_worker():
    return patch.multiple(worker, _parent_guard=lambda pid: None, _cpu_policy=lambda: None,
        execute_request=lambda value: {"schema": worker.RESULT_SCHEMA, "root": value["root"]})


def test_worker_accepts_request_spelled_through_symlinked_output_folder(tmp_path):
    root, alias = _symlinked_output(tmp_path)
    request = ProductWorkerTest().fixture(root)
    work = root / ".ensemble-diagnostics" / "d01" / "product-process-x"
    work.mkdir(parents=True)
    (work / "request.json").write_text(json.dumps(request), encoding="utf-8")
    spelled = alias / ".ensemble-diagnostics" / "d01" / "product-process-x"
    with _patched_worker():
        assert worker.main([str(spelled / "request.json"), "--result", str(spelled / "result.json")]) == 0
    assert json.loads((work / "result.json").read_text())["root"] == str(root)
    owned = worker._owned_path(root, str(alias / "coordinates.nc"))
    assert owned == root / "coordinates.nc"
    assert worker._owned_path(alias, str(root / "coordinates.nc")) == alias / "coordinates.nc"


def test_symlinked_output_folder_keeps_every_outside_the_folder_refusal(tmp_path):
    root, alias = _symlinked_output(tmp_path)
    foreign = tmp_path / "foreign.nc"
    foreign.write_bytes(b"not owned")
    with pytest.raises(ValueError, match="escapes"):
        worker._owned_path(root, str(alias.parent / "foreign.nc"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "spill.nc").write_bytes(b"not owned")
    os.symlink(outside, root / "inner", target_is_directory=True)
    with pytest.raises(ValueError, match="cannot traverse symlinks"):
        worker._owned_path(root, str(alias / "inner" / "spill.nc"))
    # A link inside the folder back onto the folder itself is still a
    # traversed symlink, not a second spelling of the root.
    (root / "spill.nc").write_bytes(b"owned")
    os.symlink(root, root / "loop", target_is_directory=True)
    with pytest.raises(ValueError, match="cannot traverse symlinks"):
        worker._owned_path(root, str(alias / "loop" / "spill.nc"))
    request = ProductWorkerTest().fixture(root)
    (outside / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with _patched_worker(), pytest.raises(ValueError, match="escapes"):
        worker.main([str(outside / "request.json"), "--result", str(alias / "result.json")])
    assert not (root / "result.json").exists()
    with pytest.raises(ValueError, match="output root cannot be a symlink"):
        worker.execute_request(dict(request, root=str(alias)))


if __name__ == "__main__":
    unittest.main()
