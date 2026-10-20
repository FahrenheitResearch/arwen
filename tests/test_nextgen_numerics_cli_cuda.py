"""Actual root CLI compilation of FMA instrumentation; no hydrology claim."""
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("requested,inherited,exact,expected", [
    (None, None, False, "fast"),
    (None, "strict", False, "strict"),
    (None, None, True, "strict"),
    ("strict", None, False, "strict"),
    ("fast", None, False, "fast"),
])
@pytest.mark.parametrize("boundary", ["cli", "direct"])
def test_root_hydro_cli_selects_compiler_mode_before_stage_prepare(
        tmp_path, requested, inherited, exact, expected, boundary):
    config = tmp_path / "hydro.toml"
    config.write_text('[hydro]\nenabled=true\ninterval_seconds=60\n'
                      + (f'numerics="{requested}"\n' if requested else '')
                      + '[[hydro.stages]]\nname="probe"\n'
                        'factory="gpuwm._nextgen_probe:prepare"\n'
                        'launch_method="launch"\nparallelism="1 CUDA thread"\n')
    script = textwrap.dedent('''
        import sys
        from types import ModuleType
        from gpuwm import numerics
        probe_module = ModuleType("gpuwm._nextgen_probe")
        class Probe:
            def __init__(self):
                import cupy as cp
                self.arrays = {"input": cp.asarray([1.0000001192092896, .9999998807907104, -1.], dtype=cp.float32),
                               "output": cp.zeros(1, cp.float32)}
                self.restart_identity = {"instrumentation": "FMA contraction", "mode": numerics.mode()}
                self.kernel = cp.RawKernel('extern "C" __global__ void probe(const float* x, float* y) {y[0]=x[0]*x[1]+x[2];}',
                                           "probe", options=("--fmad=false",))
                self.kernel.compile()
            def launch(self, dt, *, stream=None):
                self.kernel((1,), (1,), (self.arrays["input"], self.arrays["output"]), stream=stream)
        def prepare(**kwargs):
            assert kwargs["settings"]["numerics"] == numerics.mode() == sys.argv[3]
            return Probe()
        probe_module.prepare = prepare
        sys.modules[probe_module.__name__] = probe_module
        output_module = ModuleType("gpuwm.hydro.output")
        class Writer:
            def finish(self, driver, *, receipt):
                expected = sys.argv[3]
                assert receipt["numerics"]["mode"] == expected
                assert driver.restart_identity["numerics"] == expected
                assert driver.restart_identity["composition"]["numerics"] == expected
                value = driver.components["probe"].arrays["output"].get().view("uint32")[0]
                assert int(value) == (0 if expected == "strict" else 0xa8800000), (expected, hex(int(value)))
                assert driver.steps == 1
                print("ROOT_HYDRO_COMPILER", expected, hex(int(value)), flush=True)
        output_module.prepare = lambda **kwargs: Writer()
        sys.modules[output_module.__name__] = output_module
        if sys.argv[4] == "cli":
            from gpuwm.cli import main
            assert main(["hydro", "run", sys.argv[1], "--seconds", "60", "--outdir", sys.argv[2]]) == 0
        else:
            from gpuwm.hydro.runtime import _prepare_resolved, read_settings
            driver = _prepare_resolved(read_settings(sys.argv[1]), device_id=0)
            driver.before_lsm()
            driver.advance()
            Writer().finish(driver, receipt={"numerics": numerics.receipt()})
    ''')
    environment = dict(os.environ)
    environment.pop("GPUWM_NUMERICS", None)
    environment.pop("GPUWM_WRF_EXACT", None)
    if inherited is not None:
        environment["GPUWM_NUMERICS"] = inherited
    if exact:
        environment["GPUWM_WRF_EXACT"] = "1"
    result = subprocess.run([sys.executable, "-c", script, str(config), str(tmp_path / "out"),
                             expected, boundary], env=environment,
                            capture_output=True, text=True, timeout=60)
    (tmp_path / "compiler.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ROOT_HYDRO_COMPILER {expected}" in result.stdout
