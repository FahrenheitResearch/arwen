"""Member configuration must beat inherited launcher settings at preparation."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpuwm.ensemble.nextgen_capture import enabled_for, member_settings, member_settings_scope


class Metadata(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def inputs(self, name, text):
        path = self.root / (name + ".toml")
        path.write_text(text)
        return SimpleNamespace(experiment_config=path)

    def test_member_table_overrides_enabled_launcher(self):
        common = self.inputs("common", '[hydro]\nenabled=true\nrealization="common"\n')
        os.environ["GPUWM_NEXTGEN_SETTINGS"] = str(common.experiment_config)
        os.environ["GPUWM_NEXTGEN_ENABLED"] = "1"
        disabled = self.inputs("disabled", "[hydro]\nenabled=false\n")
        self.assertFalse(enabled_for(disabled))
        member = self.inputs("member", '[hydro]\nenabled=true\nrealization="member"\n')
        self.assertEqual(member_settings(member)["realization"], "member")
        self.assertTrue(enabled_for(member))

    def test_missing_table_inherits_common_metadata(self):
        common = self.inputs("common", '[hydro]\nenabled=false\n[[hydro.stages]]\nname="routing"\n')
        os.environ["GPUWM_NEXTGEN_SETTINGS"] = str(common.experiment_config)
        os.environ["GPUWM_NEXTGEN_ENABLED"] = "1"
        member = self.inputs("member", "[ensemble.hydro]\nthresholds_m3_s=[1,10]\n")
        first = member_settings(member)
        self.assertTrue(enabled_for(member))
        first["stages"][0]["name"] = "changed"
        self.assertEqual(member_settings(member)["stages"][0]["name"], "routing")
        self.assertEqual(os.environ["GPUWM_NEXTGEN_SETTINGS"], str(common.experiment_config))


@unittest.skipUnless(importlib.util.find_spec("gpuwm.hydro") is not None,
                     "requires the fresh NextGen driver dependency")
class Runtime(Metadata):
    def test_disabled_member_masks_active_launcher_without_changing_it(self):
        from gpuwm.hydro import runtime
        member = self.inputs("disabled", "[hydro]\nenabled=false\n")
        ambient = {"enabled": True, "member_tag": -1}
        with runtime.settings_scope(ambient):
            with member_settings_scope(member):
                self.assertFalse(runtime.current_settings()["enabled"])
            self.assertIs(runtime.current_settings(), ambient)

    def test_concurrent_member_preparation_and_exception_reset(self):
        from gpuwm.hydro import runtime
        barrier = threading.Barrier(2)
        inputs = [self.inputs(str(member),
            f'[hydro]\nenabled=true\nmember_tag={member}\n[[hydro.stages]]\nname="member-{member}"\n')
            for member in (5, 9)]
        ambient = {"enabled": True, "member_tag": -1, "stages": [{"name": "ambient"}]}
        @runtime.forecast_scope
        def prepare(inputs):
            barrier.wait(timeout=10)
            settings = runtime.current_settings()
            tag = settings["member_tag"]
            self.assertEqual(settings["stages"][0]["name"], f"member-{tag}")
            settings["stages"][0]["name"] = "mutated"
            return tag
        def worker(inputs):
            with runtime.settings_scope(ambient):
                with member_settings_scope(inputs):
                    tag = prepare(inputs)
                self.assertIs(runtime.current_settings(), ambient)
                with self.assertRaisesRegex(RuntimeError, "forecast failed"):
                    with member_settings_scope(inputs):
                        raise RuntimeError("forecast failed")
                self.assertIs(runtime.current_settings(), ambient)
                return tag
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(worker, inputs)), [5, 9])
        self.assertEqual(ambient["stages"][0]["name"], "ambient")
        self.assertEqual(member_settings(inputs[0])["stages"][0]["name"], "member-5")

    def test_in_process_common_settings_are_not_shared(self):
        from gpuwm.hydro import runtime
        member = self.inputs("member", "[ensemble.hydro]\nthresholds_m3_s=[1]\n")
        ambient = {"enabled": True, "stages": [{"name": "common"}]}
        with runtime.settings_scope(ambient):
            self.assertTrue(enabled_for(member))
            with member_settings_scope(member):
                runtime.current_settings()["stages"][0]["name"] = "member-only"
            self.assertIs(runtime.current_settings(), ambient)
            self.assertEqual(ambient["stages"][0]["name"], "common")

    def test_common_snapshot_survives_worker_without_inherited_context(self):
        from gpuwm.hydro import runtime
        member = self.inputs("member", "[ensemble.hydro]\nthresholds_m3_s=[1]\n")
        ambient = {"enabled": True, "stages": [{"name": "common"}]}
        with runtime.settings_scope(ambient):
            snapshot = member_settings(member)
        def worker():
            self.assertIsNone(runtime.current_settings())
            with member_settings_scope(member, inherited_settings=snapshot):
                self.assertEqual(runtime.current_settings()["stages"][0]["name"], "common")
                runtime.current_settings()["stages"][0]["name"] = "worker-only"
            self.assertIsNone(runtime.current_settings())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(Context().run, worker).result()
        self.assertEqual(snapshot["stages"][0]["name"], "common")
        self.assertEqual(ambient["stages"][0]["name"], "common")


if __name__ == "__main__":
    unittest.main(verbosity=2)
