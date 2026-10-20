"""Reservoir overrides must agree with the fixed process compiler mode."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ReservoirNumericsTest(unittest.TestCase):
    def run_process(self, source, *, mode=None, exact=False):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("GPUWM_NUMERICS", None)
        env.pop("GPUWM_WRF_EXACT", None)
        if mode is not None:
            env["GPUWM_NUMERICS"] = mode
        if exact:
            env["GPUWM_WRF_EXACT"] = "1"
        result = subprocess.run([sys.executable, "-B", "-c", source], env=env,
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_disagreement_rejected_before_cuda_or_arrays(self):
        source = '''
from gpuwm import numerics
from gpuwm.hydro.reservoirs import Reservoirs, kernel_source, _module
opposite = not numerics.is_strict()
for action in (lambda: Reservoirs({}, {}, strict=opposite),
               lambda: kernel_source(opposite), lambda: _module(opposite, 0)):
    try:
        action()
    except ValueError as error:
        assert "disagrees with process numerics" in str(error), str(error)
    else:
        raise AssertionError("local override silently changed reservoir arithmetic")
'''
        for mode in (None, "fast", "strict"):
            with self.subTest(mode=mode):
                self.run_process(source, mode=mode)
        self.run_process(source, mode="fast", exact=True)

    def test_composed_source_inherits_mode_and_matching_explicit_choice(self):
        source = '''
from gpuwm import numerics
from gpuwm.hydro.reservoirs import kernel_source
strict = numerics.is_strict()
source = kernel_source()
assert source == kernel_source(strict)
assert ("#define GPUWM_NEXTGEN_RESERVOIR_STRICT 1" in source) == strict
'''
        for mode in (None, "fast", "strict"):
            with self.subTest(mode=mode):
                self.run_process(source, mode=mode)

    def test_environment_change_cannot_change_fixed_source_policy(self):
        self.run_process('''
import os
from gpuwm.hydro.reservoirs import kernel_source, Reservoirs
source = kernel_source()
os.environ["GPUWM_NUMERICS"] = "strict"
assert kernel_source() == source
try:
    Reservoirs({}, {}, strict=True)
except ValueError as error:
    assert "disagrees with process numerics='fast'" in str(error)
else:
    raise AssertionError("constructor reread environment after mode was fixed")
''', mode="fast")


if __name__ == "__main__":
    unittest.main()
