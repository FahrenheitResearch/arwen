"""Real PET CUDA faults at resident publication boundaries; synthetic inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cupy as cp

from gpuwm.hydro import ResidentHydroDriver, Stage, output
from gpuwm.hydro.pet import PARAMETER_NAMES, prepare


class PETHealth(unittest.TestCase):
    def build(self, dtype):
        fields = {key: cp.full(257, value, dtype) for key, value in
                  dict(T2D=298., Q2D=.008, U2D=3., V2D=1., PSFC=100000.,
                       LWDOWN=310., SWDOWN=450.).items()}
        source = SimpleNamespace(fields=fields, restart_identity={
            "identity": {"catchment_ids": list(range(257))},
            "scope": "synthetic publication regression"})
        parameters = dict(zip(PARAMETER_NAMES,
                              (10., 2., .5, .1, .03, .003, .98, .23,
                               70., 20., 0., 25.)))
        pet = prepare(settings={}, options={"forcing_component": "forcing",
                      "parameters": parameters, "method": 5},
                      state=None, physics=None, device_id=0, mode="standalone",
                      components={"forcing": source})
        driver = ResidentHydroDriver(device_id=0, interval_seconds=3600,
            identity={"scope": "synthetic PET publication regression"},
            after=[Stage("pet", pet, pet.advance)])
        return fields, pet, driver

    def advance(self, driver, stream):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)

    def invalid(self, dtype):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            fields, pet, driver = self.build(dtype)
            fields["Q2D"][-1] = -1.
            self.advance(driver, stream)
            self.assertEqual(int(pet.arrays["status"][-1].item()), 1)
            with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
                driver.check_health(stream=stream)
            self.assertEqual(error.exception.diagnostics, {"pet/status": 1})
            with TemporaryDirectory() as directory:
                path = Path(directory) / "must-not-exist.nc"
                # Exercise the actual metadata -> Rust checkpoint publication
                # sequence. Metadata must refuse before a container is created.
                with self.assertRaisesRegex(RuntimeError, "prevent publication"):
                    metadata = driver.checkpoint_metadata()
                    arrays, _, payload = output._snapshot(driver.arrays, stream)
                    output._call({"operation": "checkpoint-write", "path": str(path),
                                  "metadata": metadata, "arrays": arrays}, payload)
                self.assertFalse(path.exists())

    def sticky(self, dtype):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            fields, pet, driver = self.build(dtype)
            fields["Q2D"][-1] = -1.
            self.advance(driver, stream)
            fields["Q2D"][-1] = .008
            self.advance(driver, stream)
            self.assertTrue(bool(cp.isfinite(pet.fields["potential_et_m_s"]).all()))
            self.assertEqual(int(pet.arrays["status"][-1].item()), 1)
            # An independent post-preparation parameter fault exercises PET's
            # numerical-result bit. No parameter-validation guard is waived.
            pet.arrays["parameters"][-1, 11] = float("nan")
            self.advance(driver, stream)
            self.assertEqual(int(pet.arrays["status"][-1].item()), 3)
            pet.arrays["parameters"][-1, 11] = 25.
            self.advance(driver, stream)
            self.assertTrue(bool(cp.isfinite(pet.fields["potential_et_m_s"]).all()))
            with self.assertRaisesRegex(RuntimeError, "prevent publication") as error:
                driver.checkpoint_metadata()
            self.assertEqual(error.exception.diagnostics, {"pet/status": 3})

    def continuation(self, dtype):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            fields, pet, driver = self.build(dtype)
            fresh_fields, fresh_pet, fresh = self.build(dtype)
            self.advance(driver, stream)
            self.assertEqual(driver.check_health(stream=stream), {"pet/status": 0})
            pointers = {key: value.data.ptr for key, value in fresh.arrays.items()}
            with TemporaryDirectory() as directory:
                path = Path(directory) / "healthy.nc"
                arrays, _, payload = output._snapshot(driver.arrays, stream)
                output._call({"operation": "checkpoint-write", "path": str(path),
                              "metadata": driver.checkpoint_metadata(),
                              "arrays": arrays}, payload)
                self.assertEqual(path.read_bytes()[:4], b"CDF\x05")
                output.restore_checkpoint(path, fresh)
            self.assertEqual(driver.checkpoint_metadata(), fresh.checkpoint_metadata())
            fields["SWDOWN"].fill(300.)
            fresh_fields["SWDOWN"].fill(300.)
            self.advance(driver, stream)
            self.advance(fresh, stream)
            stream.synchronize()
            for key, value in driver.arrays.items():
                self.assertEqual(value.get().tobytes(), fresh.arrays[key].get().tobytes(), key)
            self.assertEqual(pointers, {key: value.data.ptr for key, value in fresh.arrays.items()})
            fresh_fields["Q2D"][-1] = -1.
            self.advance(fresh, stream)
            with self.assertRaisesRegex(RuntimeError, "prevent publication"):
                fresh.checkpoint_metadata()
            self.assertEqual(driver.check_health(stream=stream), {"pet/status": 0})


for _case in ("invalid", "sticky", "continuation"):
    for _dtype in ("float32", "float64"):
        def _test(self, case=_case, dtype=_dtype):
            getattr(self, case)(dtype)
        setattr(PETHealth, "test_" + _case + "_" + _dtype, _test)


if __name__ == "__main__":
    unittest.main(verbosity=2)
