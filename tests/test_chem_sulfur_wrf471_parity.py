"""WRF sulfur driver and exact chemics_init extract against CUDA columns."""
import json

import numpy as np
import pytest

from conftest import requires_gpu
from chem_gocart_checks import FIXTURES, check_fixture_hashes, sulfur_case
from gpuwm.core.chem_sulfur import pack_parameters
from gpuwm.verify.chem_oracle import load


def test_fixture_pins():
    check_fixture_hashes("sulfur", "421fbacee3b18661222fcbe4f0e22243a2d06ad35637833b2dc63bfd26c20a91")


def test_roles_are_row_parameters():
    p = pack_parameters([{}, {"sulfur": {"role": "so4"}}, {"sulfur": {"role": "so2"}}])
    assert p.tolist() == [-1,2,1,-1]
    assert p.dtype == np.int32
    with pytest.raises(ValueError, match="duplicate"):
        pack_parameters([{"sulfur": {"role": "so2"}}]*2)


def test_case_coverage():
    cases = load(FIXTURES / "sulfur")
    assert len(cases) == 48
    assert {float(c["dt"]) for c in cases.values()} == {30,36,60,7.5}
    assert any(np.any(c["tcosz"] == 0) for c in cases.values())
    assert any(np.any(c["ttday"] >= 86400) for c in cases.values())
    assert any(np.any(c["cossza_1"] > 0) for c in cases.values())
    assert any(np.any(c["cossza_1"] == 0) for c in cases.values())
    assert set(np.concatenate([c["branch_input_1"].ravel() for c in cases.values()])) == {0,1,2,3}
    for name, case in cases.items():
        if case["zero_roles"]:
            assert np.all(case["input_dms"] == 0) and np.all(case["input_msa"] == 0)
        if name.endswith("cloud0"):
            other = cases[name[:-1]+"1"]
            assert np.any(case["step1_so2"] != other["step1_so2"])


@requires_gpu
@pytest.mark.gpu
def test_wrf471_parity():
    baseline = json.loads((FIXTURES / "sulfur/ULP_BASELINE.json").read_text())
    for name, case in load(FIXTURES / "sulfur").items():
        for mode, oracle_solar in (("native", False), ("oracle_solar", True)):
            assert sulfur_case(case, oracle_solar=oracle_solar) == baseline[name][mode], (name,mode)
            if case["zero_roles"]:
                assert sulfur_case(case, oracle_solar=oracle_solar, lite=True) == baseline[name][mode+"_lite"], (name,mode)


@requires_gpu
@pytest.mark.gpu
def test_absent_roles_match_zero_roles():
    for name, case in load(FIXTURES / "sulfur").items():
        if not case["zero_roles"]:
            continue
        full = sulfur_case(case, oracle_solar=True)
        lite = sulfur_case(case, oracle_solar=True, lite=True)
        for step in range(1,4):
            for role in ("so2", "so4"):
                key = f"step{step}_{role}"
                assert full[key] == lite[key] == {"max_ulp":0,"n_nonzero":0,"n":40}, (name,key)
