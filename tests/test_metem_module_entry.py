import os
import subprocess
import sys


def test_module_entry_has_the_terrain_preparation_helper():
    # The coarse-grid public door reaches this helper during main(), before
    # any forecast step. Exercise Python's actual module entry, not import.
    code = '''
import runpy, sys
sys.argv = ["gpuwm.metem_forecast", "--help"]
def observe(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "main" and frame.f_code.co_filename.endswith("gpuwm/metem_forecast.py"):
        assert callable(frame.f_globals.get("_smooth_cg_case")), "module entry calls forecast preparation before the terrain helper is defined"
    return observe
sys.settrace(observe)
runpy.run_module("gpuwm.metem_forecast", run_name="__main__")
'''
    result = subprocess.run([sys.executable, '-c', code], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                 'GPUWM_NO_LOCAL_GPU': '1'})
    assert result.returncode == 0, result.stdout
    assert 'usage:' in result.stdout
