"""Actual CUDA capture and Rust publication clocks; synthetic transport only."""
import json
from pathlib import Path
import tempfile
import unittest

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.hydro import output, module_history


class Source:
    def __init__(self):
        self.arrays = {"reach_id": cp.asarray([17, 29], cp.int64),
                       "flow": cp.zeros(2, cp.float64)}
        self.restart_identity = {"instrumentation": "history clock; no hydrologic physics"}
        self.kernel = cp.RawKernel('''extern "C" __global__ void update(double* q) {
            int i=threadIdx.x; q[i] += (double)(i+1);
        }''', "update")

    def advance(self, dt, *, stream=None):
        self.kernel((1,), (2,), (self.arrays["flow"],), stream=stream)

    def reach_output(self):
        return {"reach_id": self.arrays["reach_id"], "discharge_m3_s": self.arrays["flow"]}


class HistoryClockTests(unittest.TestCase):
    def build(self, *, module=False, capacity=24, cadence=1, period=300, repeat=12,
              scheduled=True):
        source = Source()
        settings = {"interval_seconds": period}
        options = dict(capacity_steps=capacity, cadence_steps=cadence,
                       origin="2023-04-02 00:00:00", types=["wb", "wb"],
                       formats=["analysis_csv"])
        if module:
            options.update(catchment_ids=["transport-a", "transport-b"], fields=[
                dict(name="flow", component="source", field="flow", units="m3 s-1")])
            history = module_history.ModuleHistory(settings=settings, options=options,
                components={"source": source}, device_id=cp.cuda.runtime.getDevice())
        else:
            history = output.History(settings=settings, options=options, source=source,
                                     device_id=cp.cuda.runtime.getDevice())
        kwargs = {"after_schedule": [dict(stages=["source", "history"],
                    interval_seconds=period, repeat=repeat)]} if scheduled else {}
        driver = ResidentHydroDriver(device_id=cp.cuda.runtime.getDevice(),
            interval_seconds=period*repeat, identity={"scope": "history clock transport"},
            after=[Stage("source", source, source.advance), Stage("history", history, history.advance)],
            **kwargs)
        return driver, history

    def advance(self, driver, count):
        for _ in range(count):
            driver.before_lsm()
            driver.advance()

    def publish(self, driver, path, module=False):
        writer = (module_history.Writer if module else output.Writer)(
            settings={}, driver=driver, outdir=path)
        writer.finish(driver, {})
        return writer.outdir

    def csv(self, path):
        return (path / "history/streamflow.csv").read_text().splitlines()

    def test_routing_substeps_keep_every_frame_and_capture_period(self):
        with tempfile.TemporaryDirectory() as root:
            driver, history = self.build()
            self.advance(driver, 1)
            self.assertEqual(driver.steps, 1)
            self.assertEqual(int(history.arrays["step"].item()), 12)
            path = self.publish(driver, Path(root)/"routing")
            lines = self.csv(path)
            self.assertEqual(len(lines), 25)
            self.assertEqual(lines[1], "17,2023-04-02T00:05:00Z,1.0")
            self.assertEqual(lines[-1], "29,2023-04-02T01:00:00Z,24.0")

    def test_module_substeps_keep_native_payload_extent(self):
        with tempfile.TemporaryDirectory() as root:
            driver, history = self.build(module=True)
            self.advance(driver, 1)
            path = self.publish(driver, Path(root)/"module", module=True)
            metadata, payload = output._call({"operation": "dataset-read", "path": str(path/"history.nc")})
            dims = {d["name"]: d["len"] for d in metadata["schema"]["dimensions"]}
            self.assertEqual(dims, {"time": 12, "catchment": 2})
            row = next(v for v in metadata["schema"]["variables"] if v["name"] == "flow")
            expected = history.arrays["flow"][:12].get().tobytes()
            self.assertEqual(payload[row["offset"]:row["offset"]+len(expected)], expected)

    def test_decimation_uses_capture_steps_and_keeps_partial_tail(self):
        with tempfile.TemporaryDirectory() as root:
            driver, _ = self.build(cadence=5)
            self.advance(driver, 2)
            path = self.publish(driver, Path(root)/"decimated")
            self.assertTrue((path/"history/streamflow.csv").exists())
            lines = self.csv(path)
            self.assertEqual(len(lines), 9)
            self.assertEqual(lines[1], "17,2023-04-02T00:25:00Z,5.0")
            self.assertEqual(lines[-1], "29,2023-04-02T01:40:00Z,40.0")

    def test_serialized_restore_keeps_clock_frames_and_pointers(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            full, _ = self.build(); self.advance(full, 2)
            full_path = self.publish(full, root/"full")
            partial, _ = self.build(); self.advance(partial, 1)
            first = self.publish(partial, root/"first")
            resumed, history = self.build()
            pointers = {k: a.data.ptr for k, a in resumed.arrays.items()}
            output.restore_checkpoint(first/"restart.nc", resumed)
            self.advance(resumed, 1)
            self.assertEqual(resumed.steps, 2)
            self.assertEqual(int(history.arrays["step"].item()), 24)
            resumed_path = self.publish(resumed, root/"resumed")
            self.assertEqual((full_path/"history/streamflow.csv").read_bytes(),
                             (resumed_path/"history/streamflow.csv").read_bytes())
            self.assertEqual(len(self.csv(resumed_path)), 49)
            self.assertEqual(pointers, {k: a.data.ptr for k, a in resumed.arrays.items()})

    def test_overflow_refuses_before_any_checkpoint_or_product(self):
        with tempfile.TemporaryDirectory() as root:
            for module in (False, True):
                with self.subTest(module=module):
                    driver, history = self.build(module=module, capacity=12)
                    self.advance(driver, 2)
                    self.assertEqual(int(history.arrays["overflow"].item()), 1)
                    path = Path(root)/str(module)
                    with self.assertRaisesRegex(ValueError, "history.*exhausted"):
                        self.publish(driver, path, module=module)
                    self.assertFalse(path.exists())

    def test_legacy_one_capture_per_parent_step_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            driver, _ = self.build(period=60, repeat=1, scheduled=False)
            self.advance(driver, 2)
            lines = self.csv(self.publish(driver, Path(root)/"legacy"))
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[-1], "29,2023-04-02T00:02:00Z,4.0")

    def test_history_period_is_part_of_restart_identity(self):
        first, first_history = self.build(period=300, repeat=12)
        other, other_history = self.build(period=600, repeat=6)
        self.assertNotEqual(first_history.restart_identity, other_history.restart_identity)
        # Same parent interval but a different capture period must not restore.
        metadata = first.checkpoint_metadata()
        saved = {k: a.copy() for k, a in first.arrays.items()}
        with self.assertRaisesRegex(ValueError, "differ"):
            other.validate_checkpoint(metadata, saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
