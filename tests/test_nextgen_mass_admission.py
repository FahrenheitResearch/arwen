"""Mass routing roster admission, before card acquisition or forecasting."""
from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import builtins

from gpuwm.ensemble.optional_routing import mass_member_settings


class Metadata(unittest.TestCase):
    def test_inactive_snapshot_needs_no_optional_provider(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(("gpuwm.hydro", "gpuwm.ensemble.nextgen_")):
                raise AssertionError("optional numerical import on inactive path")
            return original(name, *args, **kwargs)
        with patch.dict("sys.modules", {"gpuwm.hydro.runtime": None}), \
             patch.dict("os.environ", {"GPUWM_NEXTGEN_SETTINGS": "", "GPUWM_NEXTGEN_ENABLED": "0"}), \
             patch("builtins.__import__", side_effect=guarded):
            self.assertEqual(mass_member_settings([{}], [{"member_id": 7, "parent_index": 0}]),
                             {7: {"enabled": False, "activation_reason": "cli_off"}})

    def test_selected_parents_get_independent_settings_snapshots(self):
        runtime = SimpleNamespace(current_settings=lambda: {"enabled": True, "nested": {"ids": []}})
        with patch.dict("sys.modules", {"gpuwm.hydro.runtime": runtime}):
            result = mass_member_settings([{}], [{"member_id": 3, "parent_index": 0},
                                                {"member_id": 9, "parent_index": 0}])
        result[3]["nested"]["ids"].append(3)
        self.assertEqual(result[9]["nested"]["ids"], [])

    def test_only_selected_parents_define_the_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"enabled.toml"
            path.write_text("[hydro]\nenabled=true\n")
            result = mass_member_settings([{"experiment_config": path}, {}],
                [{"member_id": 0, "parent_index": 0}], products_requested=True)
            self.assertTrue(result[0]["enabled"])


class MassDoor(unittest.TestCase):
    def test_product_request_and_mixed_roster_fail_before_cards(self):
        try:
            from gpuwm.ensemble import mass_launcher
        except ImportError as error:
            self.skipTest(f"requires the mass owner's staged launcher: {error}")
        from gpuwm import supervisor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root/"policy.toml"
            policy.write_text("[ensemble.hydro]\nthresholds_m3_s=[1]\n")
            parents = []
            for index in range(2):
                config = root/f"member-{index}.toml"
                config.write_text("[hydro]\nenabled=false\n")
                checkpoint = root/f"parent-{index}.nc"
                checkpoint.touch()  # Metadata admission must not open fields.
                parents.append(dict(checkpoint=str(checkpoint), prepared_root=str(root),
                    experiment_config=str(config), provenance=dict(source="test", cycle="2024-01-01T00:00:00Z")))
            source = root/"parents.json"
            source.write_text(json.dumps(parents))
            args = SimpleNamespace(branch_from=source, members=2, config=policy,
                wind_amplitude=0., ensemble_seed=0, branch_surface=False, branch_hours=1.,
                dry_run=False, cards="all", outdir=str(root/"out"))
            with patch.object(supervisor, "query_gpus", side_effect=AssertionError("card query before roster admission")):
                with self.assertRaisesRegex(ValueError, "hydro enabled for every member"):
                    mass_launcher.run(args)
                self.assertFalse((root/"out").exists())
                Path(parents[0]["experiment_config"]).write_text("[hydro]\nenabled=true\n")
                args.config = None
                with self.assertRaisesRegex(ValueError, "hydro enabled for every member"):
                    mass_launcher.run(args)
            # The generic inactive path still supports a no-card dry run.
            Path(parents[0]["experiment_config"]).write_text("[hydro]\nenabled=false\n")
            args.dry_run = True
            with redirect_stdout(StringIO()):
                self.assertEqual(mass_launcher.run(args), 0)
