"""Compiled WRF dycore coverage required on the CPU and card battery legs."""

DYCORE_PARITY_FILES = {
    "tests/test_advect_wrf471_parity.py",
    "tests/test_smallstep_bookkeeping_wrf471_parity.py",
    "tests/test_smallstep_horizontal_wrf471_parity.py",
    "tests/test_smallstep_vertical_wrf471_parity.py",
    "tests/test_bigstep_coupling_wrf471_parity.py",
    "tests/test_bigstep_momentum_wrf471_parity.py",
    "tests/test_bigstep_prep_wrf471_parity.py",
    "tests/test_bigstep_rk_wrf471_parity.py",
    "tests/test_diff6_wrf471_parity.py",
    "tests/test_deformation_wrf471_parity.py",
    "tests/test_horizontal_diffusion_wrf471_parity.py",
    "tests/test_vertical_diffusion_wrf471_parity.py",
    "tests/test_constant_diffusion_wrf471_parity.py",
    "tests/test_diffusion_drivers_wrf471_parity.py",
}

DYCORE_CPU_FILES = DYCORE_PARITY_FILES | {"tests/test_smallstep_oracle_contract.py"}

# These tests reach CUDA through oracle launch adapters. Their own lazy
# imports preserve CPU fixture checks, so the shard gate verifies both the
# imported adapter and its CuPy use rather than requiring a device import
# in the test module itself.
DEVICE_ORACLE_IMPORTS = {
    "tests/test_advect_wrf471_parity.py": "gpuwm.verify.advect_oracle",
    "tests/test_smallstep_bookkeeping_wrf471_parity.py": "gpuwm.verify.smallstep_bookkeeping_oracle",
    "tests/test_smallstep_horizontal_wrf471_parity.py": "gpuwm.verify.smallstep_horizontal_oracle",
    "tests/test_smallstep_vertical_wrf471_parity.py": "gpuwm.verify.smallstep_vertical_oracle",
    "tests/test_bigstep_coupling_wrf471_parity.py": "gpuwm.verify.bigstep_coupling_oracle",
    "tests/test_bigstep_momentum_wrf471_parity.py": "gpuwm.verify.bigstep_momentum_oracle",
    "tests/test_bigstep_prep_wrf471_parity.py": "gpuwm.verify.bigstep_prep_oracle",
    "tests/test_bigstep_rk_wrf471_parity.py": "gpuwm.verify.bigstep_rk_oracle",
    "tests/test_diff6_wrf471_parity.py": "gpuwm.verify.diffusion_oracle",
    "tests/test_constant_diffusion_wrf471_parity.py": "gpuwm.verify.diffusion_oracle",
    "tests/test_deformation_wrf471_parity.py": "deformation_compare",
    "tests/test_horizontal_diffusion_wrf471_parity.py": "horizontal_compare",
}


def adapter_path(module: str) -> str:
    if module.startswith("gpuwm."):
        return module.replace(".", "/") + ".py"
    return "tools/wrf_diffusion_oracle/" + module + ".py"
