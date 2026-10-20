"""Analytical CUDA transfer checks; no physical basin/native phase claim."""
import unittest
from types import SimpleNamespace

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.hydro.runoff import PreparedRunoffRouting, prepare


class RunoffTests(unittest.TestCase):
    def build(self, *, units="m2", dtype=cp.float64):
        return PreparedRunoffRouting(
            depth=cp.asarray([.5, 1., 2., .25], dtype=dtype),
            area=cp.asarray([8., 4., 2., 16.], dtype=cp.float64),
            catchment_to_node=cp.asarray([2, 0, 2, 0], dtype=cp.int32),
            node_count=4, period_seconds=4, area_units=units,
            identity={"analytical": "four columns; two receivers plus two empty"})

    def same(self, actual, expected):
        self.assertTrue(bool(cp.array_equal(actual, cp.asarray(expected, dtype=actual.dtype)).item()))

    def test_order_units_empty_receivers_and_precision(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            for dtype in (cp.float32, cp.float64):
                for units, q, v in (("m2", [2., 0., 2., 0.], [8., 0., 8., 0.]),
                                    ("km2", [2e6, 0., 2e6, 0.], [8e6, 0., 8e6, 0.])):
                    transfer = self.build(units=units, dtype=dtype)
                    pointers = {k: a.data.ptr for k, a in transfer.arrays.items()}
                    transfer.advance(4, stream=stream)
                    self.same(transfer.arrays["indices"], [1, 3, 0, 2])
                    self.same(transfer.arrays["offsets"], [0, 2, 2, 4, 4])
                    self.same(transfer.arrays["routing_lateral"], q)
                    self.same(transfer.arrays["routing_discharge_m3_s"], q)
                    self.same(transfer.arrays["routing_volume_m3"], v)
                    self.same(transfer.arrays["status"], [0, 0, 0, 0])
                    self.assertEqual(pointers, {k: a.data.ptr for k, a in transfer.arrays.items()})
                    self.assertEqual(transfer.kernel.attributes["local_size_bytes"], 0)

    def test_fault_sticks_until_full_checkpoint_restore(self):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            transfer = self.build()
            driver = ResidentHydroDriver(device_id=0, interval_seconds=4, identity={"analytical": True},
                after=[Stage("transfer", transfer, transfer.advance)])
            driver.before_lsm(stream=stream); driver.advance(stream=stream)
            metadata = driver.checkpoint_metadata()
            saved = {k: a.copy() for k, a in driver.arrays.items()}
            pointers = {k: a.data.ptr for k, a in driver.arrays.items()}
            transfer.arrays["depth"][0] = cp.nan
            transfer.advance(4, stream=stream)
            self.assertTrue(bool(cp.isnan(transfer.arrays["routing_lateral"][2]).item()))
            transfer.arrays["depth"][0] = .5
            transfer.advance(4, stream=stream)
            self.same(transfer.arrays["routing_lateral"], [2., 0., 2., 0.])
            self.same(transfer.arrays["status"], [0, 0, 1, 0])
            with self.assertRaises(RuntimeError): driver.check_health(stream=stream)
            with self.assertRaises(RuntimeError): driver.checkpoint_metadata()
            driver.validate_checkpoint(metadata, saved)
            for k, a in driver.arrays.items(): cp.copyto(a, saved[k])
            driver.restore_metadata(metadata)
            self.assertEqual(driver.check_health(stream=stream), {"transfer/runoff": 0})
            self.assertEqual(pointers, {k: a.data.ptr for k, a in driver.arrays.items()})
            self.assertTrue(all(bool(cp.array_equal(a, saved[k]).item()) for k, a in driver.arrays.items()))
            self.assertEqual(driver.checkpoint_metadata(), metadata)
            other = self.build(units="km2")
            changed = ResidentHydroDriver(device_id=0, interval_seconds=4, identity={"analytical": True},
                after=[Stage("transfer", other, other.advance)])
            with self.assertRaises(ValueError): changed.validate_checkpoint(metadata, saved)

    def test_divide_each_native_publication_before_aggregation(self):
        t = PreparedRunoffRouting(depth=cp.asarray([1., 2.], dtype=cp.float64),
            area=cp.ones(2, cp.float64), catchment_to_node=cp.zeros(2, cp.int32),
            node_count=1, period_seconds=10, area_units="m2", identity={"analytical": "rounding order"})
        t.advance(10)
        self.same(t.arrays["routing_discharge_m3_s"], [0.30000000000000004])
        self.same(t.arrays["routing_volume_m3"], [3.])

    def test_corrupt_csr_negative_depth_and_output_overflow_refuse(self):
        for array, index, value, bit in (("depth", 1, -1., 1), ("area", 1, -1., 1),
                ("indices", 0, 99, 4), ("offsets", 0, -1, 4),
                ("depth", 1, 1e100, 2)):
            transfer = self.build(); transfer.arrays[array][index] = value
            transfer.advance(4)
            self.assertEqual(int(transfer.arrays["status"][0].item()), bit)
            self.assertTrue(bool(cp.isnan(transfer.arrays["routing_lateral"][0]).item()))

    def test_duration_and_setup_refusals(self):
        t = self.build()
        for dt in (300, 0, float("nan")):
            with self.assertRaisesRegex(ValueError, "producer period"): t.advance(dt)
        args = dict(depth=t.arrays["depth"], area=t.arrays["area"],
                    catchment_to_node=t.arrays["catchment_to_node"], node_count=4,
                    period_seconds=4, area_units="m2", identity={})
        for changes in (dict(area_units="acres"), dict(period_seconds=True), dict(node_count=0),
                        dict(area=cp.asarray([0., 1., 1., 1.], dtype=cp.float64)),
                        dict(catchment_to_node=cp.asarray([4, 0, 2, 0], dtype=cp.int32))):
            with self.assertRaises(ValueError): PreparedRunoffRouting(**(args | changes))

    def test_realization_factory_axis_and_area_binding(self):
        t = self.build(); ids = ("a", "b", "c", "d")
        source = SimpleNamespace(fields={"depth": t.arrays["depth"]}, arrays={}, period_seconds=4,
                                 restart_identity={"preparation": {"catchments": ids}})
        fabric = SimpleNamespace(arrays={"catchment_to_node": t.arrays["catchment_to_node"]},
                                 node_ids=("n0", "n1", "n2", "n3"), catchment_ids=ids,
                                 restart_identity={"network": "analytical"})
        areas = SimpleNamespace(arrays={"areas": t.arrays["area"]}, catchment_ids=ids,
                                restart_identity={"area": "explicit analytical m2"})
        options = dict(source_component="producer", network_component="fabric", area_component="areas",
                       depth_field="depth", area_array="areas", area_units="m2",
                       source_catchments_path=["preparation", "catchments"])
        kwargs = dict(settings={}, options=options, state=None, physics=None, device_id=0,
                      mode="standalone", components=dict(producer=source, fabric=fabric, areas=areas))
        transfer = prepare(**kwargs); transfer.advance(4)
        self.same(transfer.arrays["routing_lateral"], [2., 0., 2., 0.])
        source.fields["depth"] = cp.zeros(3, cp.float64)
        with self.assertRaisesRegex(ValueError, "extent differs"): prepare(**kwargs)
        source.fields["depth"] = t.arrays["depth"]
        areas.catchment_ids = tuple(reversed(ids))
        with self.assertRaisesRegex(ValueError, "ordering differ"): prepare(**kwargs)


if __name__ == "__main__": unittest.main(verbosity=2)
