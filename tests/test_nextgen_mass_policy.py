"""Mass worker/product wiring on synthetic CUDA samples, not weather proof."""
from contextlib import nullcontext
from datetime import datetime
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class MassPolicy(unittest.TestCase):
    def test_launch_policy_reaches_workers_and_final_reduction(self):
        self._exercise(inherited=False)

    def test_parent_runtime_snapshot_reaches_workers_without_context_inheritance(self):
        self._exercise(inherited=True)

    def _exercise(self, *, inherited):
        try:
            import cupy as cp
            launcher = importlib.import_module("gpuwm.ensemble.mass_launcher")
        except ImportError as error:
            self.skipTest(f"needs CUDA and the mass owner's staged launcher: {error}")
        from gpuwm import run_stamp, supervisor
        from gpuwm.ensemble.runtime_context import current_capture
        from gpuwm.hydro import runtime
        from gpuwm.netcdf_bridge import Dataset

        # Pipes and the real completion queue run normally. Threads replace
        # spawn only so this targeted test can inject the weather boundary;
        # CUDA work stays on the canonical pool's one assigned physical card.
        context = multiprocessing.get_context("spawn")
        connections = []
        class Connection:
            def __init__(self, value):
                self.value = value
                connections.append(value)
            def __getattr__(self, key):
                return getattr(self.value, key)
            def close(self):
                pass  # A real process has its own handle; threads share one.
        class Process:
            def __init__(self, *, target, args):
                self.thread = threading.Thread(target=target, args=args)
                self.exitcode = None
            def start(self):
                self.thread.start()
            def is_alive(self):
                return self.thread.is_alive()
            def join(self, timeout=None):
                self.thread.join(timeout)
                if not self.thread.is_alive():
                    self.exitcode = 0
        class Context:
            def Pipe(self):
                return tuple(Connection(value) for value in context.Pipe())
        Context.Process = Process
        class AtmosphereCapture:
            def __init__(self, *args):
                self.frames = []
            def __call__(self, **row):
                self.frames.append({"grid_id": row["grid_id"]})
            def finish(self):
                return self.frames

        with tempfile.TemporaryDirectory(dir=os.environ.get("NEXTGEN_TEST_ROOT", os.environ.get("TMPDIR"))) as directory:
            root = Path(directory)
            config = root / "launch.toml"
            config.write_text("[hydro]\nenabled=false\n[ensemble]\nmembers=2\n"
                              "[ensemble.hydro]\nthresholds_m3_s=[2.5,25.0]\n"
                              "device_budget_bytes=1048576\n")
            parents = []
            for member in range(2):
                runtime_config = root / f"runtime-{member}.toml"
                runtime_config.write_text("[run]\n" if inherited else
                    f"[hydro]\nenabled=true\nmember_tag={member}\n")
                checkpoint = root / f"parent-{member}.nc"
                checkpoint.touch()  # Header/atmosphere boundary is injected.
                parents.append(dict(parent_id=member, checkpoint=str(checkpoint),
                    prepared_root=str(root), experiment_config=str(runtime_config),
                    provenance=dict(source="test", cycle="2024-01-01T00:00:00Z")))
            parent_file = root / "parents.json"
            parent_file.write_text(json.dumps(parents))
            out = root / "out"
            out.mkdir()
            retained = []
            def load(parent, directory, hours):
                return (SimpleNamespace(experiment_config=parent["experiment_config"]),
                        None, SimpleNamespace(sha256="test-boundary", path=Path(parent["checkpoint"])))
            def forecast(inputs, **kwargs):
                from cupy.cuda import Device
                with Device(0):
                    capture = current_capture()
                    member = capture.member_id
                    self.assertEqual(runtime.current_settings()["member_tag"], "parent" if inherited else member)
                    if inherited:
                        self.assertEqual(runtime.current_settings()["member_tags"], [])
                        runtime.current_settings()["member_tags"].append(member)
                    self.assertEqual(capture.provenance["parent"]["source"], "test")
                    ids = cp.asarray([2**60+3, 7, 42, 99], dtype=cp.int64)
                    flow = cp.asarray(([2.5, 25., np.nan, 0.] if member == 0 else
                                       [0., 25., 30., 3.]), dtype=cp.float64)
                    hydro = SimpleNamespace(device_id=0, arrays={"ids": ids, "flow": flow},
                        reach_output=lambda: {"reach_id": ids, "discharge_m3_s": flow})
                    state = SimpleNamespace(physics=SimpleNamespace(hydro=hydro))
                    retained.append(state)
                    capture.submit(state=state, valid_time=datetime(2024, 1, 1), grid_id=1,
                                   streamed=None, metadata={}, refl_field=None)
                return {"scope": "synthetic output boundary, no atmospheric forecast"}
            uuid = "0"  # Visible ordinal inside the enclosing canonical pool.
            args = SimpleNamespace(branch_from=parent_file, members=2, config=config,
                wind_amplitude=0., ensemble_seed=0, branch_surface=False, branch_hours=.25,
                dry_run=False, cards="all", outdir=str(out), keep_member_files=False)
            products = SimpleNamespace(reduce_set=lambda *args: [])
            common_settings = {"enabled": True, "member_tag": "parent", "member_tags": []}
            try:
                with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), \
                     patch.dict("sys.modules", {"gpuwm.ensemble.mass_products": products}), \
                     patch.object(launcher, "_load_inputs", side_effect=load), \
                     patch.object(launcher, "ThinCapture", AtmosphereCapture), \
                     patch.object(launcher, "select_cards", return_value=(SimpleNamespace(uuid=uuid),)), \
                     patch.object(supervisor, "query_gpus", return_value=[]), \
                     patch.object(supervisor, "GPUFileLock", side_effect=lambda *a, **kw: nullcontext()), \
                     patch.object(supervisor, "preflight_exclusive_gpu"), \
                     patch.object(run_stamp, "resolve", return_value=out), \
                     patch.object(run_stamp, "run_stamp_enabled", return_value=False), \
                     patch("gpuwm.prepared_domain_tree_forecast.run_prepared_tree", side_effect=forecast), \
                     patch.object(multiprocessing, "get_context", return_value=Context()), \
                     runtime.settings_scope(common_settings if inherited else None):
                    try:
                        self.assertEqual(launcher.run(args), 0)
                    except Exception:
                        for failed in out.glob("members/*/receipt.json"):
                            row = json.loads(failed.read_text())
                            if row.get("status") == "failed":
                                print(row.get("traceback", row.get("error")))
                        raise
            finally:
                for connection in connections:
                    connection.close()
            self.assertEqual(common_settings["member_tags"], [])
            receipt = json.loads((out / "ensemble-set.json").read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(len(receipt["completed"]), 2)
            self.assertEqual(len({state.physics.hydro.arrays["flow"].data.ptr
                                  for state in retained}), 2)
            policy = {"thresholds_m3_s": [2.5, 25.], "device_budget_bytes": 1048576}
            # Assert the actual published values before receipt metadata: the
            # old handoff silently used the default [1,10,100] thresholds.
            with Dataset(receipt["nextgen"]["products"][0]) as source:
                source.set_auto_maskandscale(False)
                np.testing.assert_array_equal(source.variables["discharge_threshold"][:], [2.5, 25.])
                np.testing.assert_array_equal(source.variables["exceedance_count"][:],
                                              [[1, 2, 1, 1], [0, 2, 1, 0]])
                np.testing.assert_array_equal(source.variables["finite_count"][:], [2, 2, 1, 2])
                np.testing.assert_array_equal(source.variables["probability"][:],
                                              [[.5, 1., np.nan, .5], [0., 1., np.nan, 0.]])
            self.assertEqual(receipt["nextgen_options"], policy)
            for member in receipt["completed"]:
                self.assertEqual(member["nextgen_points"]["options"], policy)
            print("mass_policy_metrics=" + json.dumps({
                "scope": "synthetic CUDA samples; injected atmosphere; thread worker; no weather/spawn proof",
                "forecast_wall_seconds": receipt["forecast_wall_seconds"],
                "product_seconds": receipt["product_seconds"],
                "wall_seconds": receipt["wall_seconds"],
                "member_wall_seconds": [row["wall_seconds"] for row in receipt["completed"]],
                "options": policy, "status": "PASS"}, sort_keys=True))
