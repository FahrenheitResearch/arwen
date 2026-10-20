"""Speed CLI's reported mode must match the actual CUDA FMA word.

Run directly with TMPDIR on the lane's data disk. No hydrologic physics here.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class CompilerModeTests(unittest.TestCase):
    def test_default_fast_and_explicit_strict(self):
        script = textwrap.dedent('''
            from functools import partial
            import json, runpy, sys
            from types import ModuleType
            from gpuwm import numerics
            numerics.arm()
            from gpuwm.nextgen_execution import ResidentSequence, Stage
            outputs = []
            module = ModuleType("speed_cli_probe")
            def factory(*, strict=False):
                import cupy as cp
                x = cp.asarray([1.0000001192092896, .9999998807907104, -1.], cp.float32)
                y = cp.zeros(1, cp.float32)
                kernel = cp.RawKernel('extern "C" __global__ void probe(const float* x, float* y) {y[0]=x[0]*x[1]+x[2];}',
                                      "probe", options=("--fmad=false",))
                kernel.compile()
                outputs.append(y)
                return ResidentSequence([Stage("compiler-probe", partial(kernel, (1,), (1,), (x,y)),
                                               "GPU 1 block x 1 thread; compiler diagnostic")],
                                        step_seconds=1, keepalive=(x,y,kernel),
                                        metadata={"numerics": "strict" if strict else "fast",
                                                  "scope": "compiler regression only"})
            module.factory = factory
            sys.modules[module.__name__] = module
            tool, output, expected = sys.argv[1:]
            sys.argv = [tool, "--factory", "speed_cli_probe:factory", "--steps", "1",
                        "--warmup", "0", "--output", output] + (["--strict"] if expected == "strict" else [])
            runpy.run_path(tool, run_name="__main__")
            from pathlib import Path
            receipt = json.loads(Path(output).read_text())
            assert receipt["numerics"]["mode"] == expected
            assert receipt["numerics"]["compiler_hook_installed"]
            for y in outputs:
                word = int(y.get().view("uint32")[0])
                assert word == (0 if expected == "strict" else 0xa8800000), (expected, hex(word))
            print("SPEED_CLI_COMPILER", expected, hex(word), flush=True)
        ''')
        tool = Path(__file__).resolve().parents[1] / "tools/nextgen_speed.py"
        environment = dict(os.environ)
        environment.pop("GPUWM_NUMERICS", None)
        environment.pop("GPUWM_WRF_EXACT", None)
        with tempfile.TemporaryDirectory(prefix="nextgen-speed-cli-") as scratch:
            for mode in ("fast", "strict"):
                with self.subTest(mode=mode):
                    result = subprocess.run([sys.executable, "-c", script, str(tool),
                                             str(Path(scratch) / f"{mode}.json"), mode],
                                            env=environment, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"SPEED_CLI_COMPILER {mode}", result.stdout)
                    print(result.stdout, end="", flush=True)


if __name__ == "__main__":
    unittest.main()
