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


def _nonzero(tables):
    return {key: t for key, t in tables.items() if t["max_ulp"] or t["n_nonzero"]}


def test_recorded_baseline_is_exact():
    """ULP_BASELINE.json is the measured table; every entry must be 0 ULP.

    It used to carry the solar-geometry residual of CUDA's sinf/cosf (cos(SZA)
    up to 36 ULP, tcosz up to 25, SO2/SO4 1 ULP in native mode).  Since the
    kernel takes WOOF's own float32 trig that residual is gone, and a nonzero
    row here would be a regression recorded as if it were expected.
    """
    baseline = json.loads((FIXTURES / "sulfur/ULP_BASELINE.json").read_text())
    assert len(baseline) == 48
    bad = {(name, mode): _nonzero(tables)
           for name, modes in baseline.items() for mode, tables in modes.items()
           if _nonzero(tables)}
    assert not bad, bad


@requires_gpu
@pytest.mark.gpu
def test_wrf471_parity():
    """Every output word, both solar modes, all 48 cases: 0 ULP.

    native: the kernel's own szangle (WOOF float32 trig), as a forecast runs.
    oracle_solar: WRF's cos(SZA)/tcosz/ttday fed in, isolating the chemistry.
    """
    for name, case in load(FIXTURES / "sulfur").items():
        for mode, oracle_solar in (("native", False), ("oracle_solar", True)):
            tables = sulfur_case(case, oracle_solar=oracle_solar)
            assert not _nonzero(tables), (name, mode, _nonzero(tables))
            if case["zero_roles"]:
                lite = sulfur_case(case, oracle_solar=oracle_solar, lite=True)
                assert not _nonzero(lite), (name, mode + "_lite", _nonzero(lite))


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
