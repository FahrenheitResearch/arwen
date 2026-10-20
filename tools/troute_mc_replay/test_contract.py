"""Acceptance metadata must retain every diagnostic and observer equivalence."""
import copy
import unittest

from replay import INPUTS, OUTPUTS, PIN, validate


class Contract(unittest.TestCase):
    def test_no_diagnostic_exclusions_or_observer_waivers(self):
        fixture = dict(schema="troute-mc-calls-v1", source_pin=PIN, calls=2,
                       inputs=INPUTS, outputs=OUTPUTS, nonperturbation={"pass": True},
                       input_array=dict(dtype="<f4", shape=[2, 15], sha256="a" * 64),
                       expected_array=dict(dtype="<f4", shape=[2, 6], sha256="b" * 64))
        validate(fixture)
        for mutation in ("omit_X", "swap_courant", "observer_changes_values", "different_source"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(fixture)
                if mutation == "omit_X":
                    bad["outputs"] = OUTPUTS[:-1]
                elif mutation == "swap_courant":
                    bad["outputs"][3:5] = ["cn", "ck"]
                elif mutation == "observer_changes_values":
                    bad["nonperturbation"]["pass"] = False
                else:
                    bad["source_pin"] = "0" * 40
                with self.assertRaises(ValueError):
                    validate(bad)


if __name__ == "__main__":
    unittest.main()
