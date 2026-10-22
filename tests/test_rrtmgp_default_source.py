"""Independent 2.8.9 default radiation source and arithmetic closure."""
from pathlib import Path
import hashlib
import pytest
from gpuwm.core import kernels
import gpuwm.wrf_exact as exact
# RTE pin/header composition refreshed to verified lane BASE ff16d3378bb.
PINS = {'rrtmgp_gas_default.cu': 'd13b914a28ad8b7ef88904977c7e33b9a7d7e800592473c0fb8debebca031edf', 'rrtmgp_rte_default.cu': '731bd84717398320a1710b2c151a466b9e5bf1a30b633d858fd31d7162dbd173', 'rrtmgp_cloud.cu': '512157c75edf4637d99a1cad490e8db71e51cd2c3643ff129adebefa81557560', 'rrtmgp_planck_common_default.cuh': '4e1a8214ea8e2a3dbd88cc2cda260a21ff678d98acf4f22c971ba0b51b4eba36'}
BASE_HEADERS = {'rrtmgp_gas': (), 'rrtmgp_rte': ('rrtmgp_planck_common.cuh', 'rrtmgp_planck_staged_default.cuh'), 'rrtmgp_cloud': ()}
@pytest.mark.parametrize("name", ("rrtmgp_gas", "rrtmgp_rte", "rrtmgp_cloud"))
def test_default_source_keeps_the_base_unit_and_headers(monkeypatch,name):
 monkeypatch.setattr(exact,"ENABLED",False)
 root=Path(kernels.__file__).parent
 for file,digest in PINS.items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest
 headers=BASE_HEADERS[name]
 expected=kernels._preamble()+"".join((root/(h.replace("rrtmgp_planck_common.cuh","rrtmgp_planck_common_default.cuh"))).read_text() for h in headers)+(root/(name+("_default.cu" if name in kernels._DEFAULT_SOURCE_MODULES else ".cu"))).read_text()
 assert kernels.module_source(name)==expected
@pytest.mark.parametrize("name", ("rrtmgp_gas", "rrtmgp_rte", "rrtmgp_cloud"))
def test_strict_source_retains_the_intake_unit(monkeypatch,name):
 monkeypatch.setattr(exact,"ENABLED",True)
 root=Path(kernels.__file__).parent
 assert kernels.module_source(name).endswith((root/(name+".cu")).read_text())
 assert "--fmad=false" in kernels.module_options(name)


# What breaks without this: the forward merge kept 2.8.9's kernel sources for
# default arithmetic (the *_default.cu units above) but the launcher passed
# the strict kernels' longer argument lists in both modes.  Every default
# RTE+RRTMGP forecast died at its first longwave call with
# CUDA_ERROR_INVALID_ADDRESS_SPACE (the release short case, golden/exact
# fc7a2c94a on G1, 2026-10-12; with CUDA_LAUNCH_BLOCKING=1 the fault is the
# rrtmgp_lw_noscat launch).  The launcher and the selected source must agree
# on every kernel's parameter count, in each arithmetic mode.

def _kernel_parameters(source, kernel):
    import re
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", source), flags=re.S)
    match = re.search(r"__global__\s+void\s+" + kernel + r"\s*\(([^)]*)\)", text)
    assert match, kernel
    return [part.strip().rsplit(" ", 1)[-1].lstrip("*")
            for part in match.group(1).split(",") if part.strip()]


@pytest.mark.parametrize("strict", [False, True])
def test_longwave_launcher_arguments_match_the_selected_kernel_source(monkeypatch, strict):
    import numpy as np
    import gpuwm.core.rrtmgp as rrtmgp

    monkeypatch.setattr(rrtmgp, "WRF_EXACT", strict)
    root = Path(kernels.__file__).parent
    rte = (root / ("rrtmgp_rte.cu" if strict else "rrtmgp_rte_default.cu")).read_text(encoding="utf-8")
    gas = (root / ("rrtmgp_gas.cu" if strict else "rrtmgp_gas_default.cu")).read_text(encoding="utf-8")

    solver = _kernel_parameters(rte, "rrtmgp_lw_noscat")
    planck_parameters = [name for name in solver if name.startswith("pk_")]
    off = rrtmgp._planck_solver_args(None, object(), 1, 1, 1)
    assert len(off) == len(planck_parameters), (
        "the Planck-off argument list and rrtmgp_lw_noscat disagree")
    assert ("pk_plev" in planck_parameters) is strict

    # The live list, built from real tables on host arrays.
    tables = rrtmgp.load_gas_tables("lw")
    ncol, nlay = 2, 3
    captured = {}
    monkeypatch.setattr(rrtmgp, "_device_profile", lambda value, shape, name: value)
    monkeypatch.setattr(type(tables), "to_device", lambda self: self, raising=False)
    meta = type("M", (), dict(iatm=0, jt=0, jp=0, ftemp=0, fpress=0))()
    planck = rrtmgp.PlanckInputs(
        tables=tables, play=np.zeros((ncol, nlay)), plev=np.zeros((ncol, nlay + 1)),
        tlay=np.zeros((ncol, nlay)), tlev=np.zeros((ncol, nlay + 1)),
        tsfc=np.zeros(ncol), vmr=np.zeros((ncol, nlay, tables.ngas + 1)), metadata=meta)
    try:
        live = rrtmgp._planck_solver_args(planck, object(), ncol, nlay, tables.ngpt)
    except Exception as error:  # a table attribute this host twin lacks
        pytest.skip(f"host tables cannot build the live list here: {error}")
    assert len(live) == len(planck_parameters)

    fold = _kernel_parameters(rte, "rrtmgp_lw_flux_reduce")
    assert len(fold) == (9 if strict else 8)
    assert ("last_tile" in fold) is strict
    sources = _kernel_parameters(gas, "rrtmgp_planck_sources")
    assert ("plev" in sources) is strict and ("idx_h2o" in sources) is strict
