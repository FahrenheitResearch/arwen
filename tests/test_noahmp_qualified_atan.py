"""Qualified native vegetation arithmetic and default table identity."""
import hashlib
import unittest
from unittest.mock import patch
from gpuwm.core.noahmp_kernel_sources import KERNEL_DIR, runtime_unit, compile_runtime_unit

class AtanProfileContract(unittest.TestCase):
    def test_legacy_remains_the_default_translation_unit(self):
        unit = runtime_unit("noahmp_vegeflux")
        self.assertEqual(unit, runtime_unit("noahmp_vegeflux", atan_profile="glibc_2_39"))
        self.assertEqual(unit.source, (KERNEL_DIR / "noahmp_vegeflux.cu").read_text())
        self.assertEqual(unit.options, ("-std=c++14", "--ftz=false"))

    def test_new_profile_seals_header_and_preserves_exports(self):
        unit = runtime_unit("noahmp_vegeflux", atan_profile="glibc_2_43")
        digest = hashlib.sha256((KERNEL_DIR / "glibc_trig_flt32.cuh").read_bytes()).hexdigest()
        self.assertIn(("glibc_trig_flt32.cuh", digest), unit.parts)
        self.assertIn({"file": "glibc_trig_flt32.cuh", "sha256": digest},
                      unit.identity()["parts"])
        self.assertIn("#define GPUWM_NOAHMP_ATAN_GLIBC243 1", unit.source)
        self.assertEqual(unit.exports, runtime_unit("noahmp_vegeflux").exports)
        self.assertIn("-std=c++17", unit.options)

    def test_profile_reaches_the_authoritative_compiler(self):
        unit = runtime_unit("noahmp_vegeflux", atan_profile="glibc_2_43")
        with patch("gpuwm.core.noahmp_kernel_sources._compile_loaded_module") as compile_:
            compile_runtime_unit("noahmp_vegeflux", module_key="profile-test",
                                 atan_profile="glibc_2_43")
        compile_.assert_called_once_with(unit.source, unit.options, "profile-test", None)

    def test_unknown_or_unimplemented_profile_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            runtime_unit("noahmp_vegeflux", atan_profile="host")
        with self.assertRaisesRegex(ValueError, "only for VEGE_FLUX"):
            runtime_unit("noahmp_bareflux", atan_profile="glibc_2_43")


class ParameterAssetIdentityContract(unittest.TestCase):
    def test_default_restart_identity_contains_actual_parameter_table_seals(self):
        from gpuwm.core.noahmp_runtime import NoahmpRuntimeParameters
        params = NoahmpRuntimeParameters()
        tables = params.restart_identity()["tables"]
        self.assertEqual(set(tables), {"MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"})
        for name, entry in tables.items():
            path = KERNEL_DIR.parents[1] / "data/noahmp" / name
            self.assertEqual(entry["bytes"], path.stat().st_size)
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


    def test_different_table_seal_cannot_reuse_restart_identity(self):
        from dataclasses import replace
        from gpuwm.core.noahmp_runtime import NoahmpRuntimeParameters
        params = NoahmpRuntimeParameters()
        receipt = dict(params.bundle.receipt)
        receipt["assets"] = [dict(x) for x in receipt["assets"]]
        receipt["assets"][0]["sha256"] = "0" * 64
        other = NoahmpRuntimeParameters(bundle=replace(params.bundle, receipt=receipt))
        self.assertNotEqual(params.restart_identity(), other.restart_identity())
        with self.assertRaisesRegex(ValueError, "requires parameter table seals"):
            NoahmpRuntimeParameters(bundle=replace(params.bundle, receipt={})).restart_identity()


if __name__ == "__main__":
    unittest.main()
