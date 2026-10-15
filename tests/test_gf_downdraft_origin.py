"""Compare the real GF origin against a qualified native leaf recording."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_native_downdraft_origin_after_the_descending_pass(tmp_path, strict):
    inputs = os.environ.get("GPUWM_GF_INPUT_CAPTURE")
    native_log = os.environ.get("GPUWM_GF_NATIVE_LOG")
    identity = os.environ.get("GPUWM_GF_NATIVE_IDENTITY")
    compiler = shutil.which("g++")
    if not (inputs and native_log and identity and compiler):
        pytest.skip("set the actual GF input capture and qualified native leaf log")
    assert json.loads(Path(identity).read_text())["ok"]
    rows = [line.split() for line in Path(native_log).read_text().splitlines()
            if line.startswith("GFDD")]
    assert rows, "the native downdraft moisture loop must have been reached"
    expected_origin = int(rows[0][1]) + 1

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools/gf_wrf461_oracle"))
    from gf_field_lists import LEV_FIELDS, SCA_FIELDS, ISCA_FIELDS
    from gpuwm.core.gf import _OUT_LEV, _OUT_SCA, _OUT_ISCA

    captured = np.load(inputs, allow_pickle=False)
    arrays = [np.ascontiguousarray(captured[key]) for key in ("lvin", "scin", "iin")]
    nz = arrays[0].shape[-1]
    assert arrays[0].shape[0] == 1
    assert np.count_nonzero(arrays[1][..., 6:]) == 0, "normalization is computed"

    # The existing write-only sinks expose the same body the forecast calls.
    source = (root / "gpuwm/core/kernels/gf.cu").read_text()
    anchor = ("            kpbl, nz, ichoice,\n"
              "            GfSink{(float *)0}, GfSink{(float *)0}, GfSinkI{(int *)0},\n"
              "            outt, outq, outqc, outu, outv, cupclw,")
    assert source.count(anchor) == 1
    replacement = anchor.replace(
        "GfSink{(float *)0}, GfSink{(float *)0}, GfSinkI{(int *)0}",
        "GfSink{probe_levels}, GfSink{probe_scalars}, GfSinkI{probe_integers}")
    (tmp_path / "probe.cu").write_text(source.replace(anchor, replacement))
    harness = (root / "tools/gf_wrf461_oracle/gf_host_harness.cpp").read_text()
    harness = harness.replace('#include "gf.cu"',
        'static float *probe_levels, *probe_scalars; static int *probe_integers;\n'
        '#include "probe.cu"')
    harness += ('\nextern "C" void set_probe(float*l,float*s,int*i) {'
                'probe_levels=l;probe_scalars=s;probe_integers=i;}\n')
    cpp = tmp_path / "probe.cpp"
    cpp.write_text(harness)
    library = tmp_path / "probe.so"
    command = [compiler, "-shared", "-fPIC", "-O2", "-fno-fast-math",
               "-ffp-contract=off", f"-DGF_KMAX={nz}", "-I",
               str(root / "gpuwm/core/kernels")]
    if strict:
        command.append("-DGPUWM_WRF_EXACT=1")
    subprocess.run([*command, str(cpp), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int)
    pointer = lambda a: a.ctypes.data_as(int_ptr if a.dtype == np.int32 else float_ptr)
    levels = np.zeros((len(LEV_FIELDS), nz), np.float32)
    scalars = np.zeros(len(SCA_FIELDS), np.float32)
    integers = np.zeros(len(ISCA_FIELDS), np.int32)
    lib.set_probe.argtypes = [float_ptr, float_ptr, int_ptr]
    lib.set_probe(pointer(levels), pointer(scalars), pointer(integers))
    output = [np.empty((1, len(_OUT_LEV), nz), np.float32),
              np.empty((1, len(_OUT_SCA)), np.float32),
              np.empty((1, len(_OUT_ISCA)), np.int32)]
    lib.host_gf_gfdrv_stage.argtypes = [float_ptr, float_ptr, int_ptr,
        float_ptr, float_ptr, int_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.host_gf_gfdrv_stage(*map(pointer, arrays), *map(pointer, output), 0, 1, nz)
    assert integers[ISCA_FIELDS.index("jmin")] == expected_origin
    assert integers[ISCA_FIELDS.index("ierr")] == 0
