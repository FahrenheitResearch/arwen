"""Independent 2.8.9 default radiation source and arithmetic closure."""
from pathlib import Path
import hashlib
import pytest
from gpuwm.core import kernels
import gpuwm.wrf_exact as exact
PINS = {'rrtmgp_gas_default.cu': 'd13b914a28ad8b7ef88904977c7e33b9a7d7e800592473c0fb8debebca031edf', 'rrtmgp_rte_default.cu': 'b9474494d0784a97f66742c3d8ab6788c6be09741a2b74a602b970149f0c579f', 'rrtmgp_cloud_default.cu': '512157c75edf4637d99a1cad490e8db71e51cd2c3643ff129adebefa81557560', 'rrtmgp_planck_common_default.cuh': '4e1a8214ea8e2a3dbd88cc2cda260a21ff678d98acf4f22c971ba0b51b4eba36'}
BASE_HEADERS = {'rrtmgp_gas': (), 'rrtmgp_rte': ('rrtmgp_planck_common.cuh',), 'rrtmgp_cloud': ()}
@pytest.mark.parametrize("name", ("rrtmgp_gas", "rrtmgp_rte", "rrtmgp_cloud"))
def test_default_source_keeps_the_base_unit_and_headers(monkeypatch,name):
 monkeypatch.setattr(exact,"ENABLED",False)
 root=Path(kernels.__file__).parent
 for file,digest in PINS.items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest
 headers=BASE_HEADERS[name]
 expected=kernels._preamble()+"".join((root/(h.replace("rrtmgp_planck_common.cuh","rrtmgp_planck_common_default.cuh"))).read_text() for h in headers)+(root/(name+"_default.cu")).read_text()
 assert kernels.module_source(name)==expected
@pytest.mark.parametrize("name", ("rrtmgp_gas", "rrtmgp_rte", "rrtmgp_cloud"))
def test_strict_source_retains_the_intake_unit(monkeypatch,name):
 monkeypatch.setattr(exact,"ENABLED",True)
 root=Path(kernels.__file__).parent
 assert kernels.module_source(name).endswith((root/(name+".cu")).read_text())
 assert "--fmad=false" in kernels.module_options(name)
