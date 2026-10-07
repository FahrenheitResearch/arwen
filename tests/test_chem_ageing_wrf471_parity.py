"""Measured CUDA output against the byte-unmodified WRF aerosols driver."""
import json

import numpy as np
import pytest

from conftest import requires_gpu
from chem_gocart_checks import FIXTURES, check_fixture_hashes, aging_case
from gpuwm.core.chem_ageing import pack_parameters
from gpuwm.verify.chem_oracle import load


def test_fixture_pins():
    check_fixture_hashes("ageing", "7e325dd37932be0cd543b1b299cc280b47d5e4550bf286856ff4cdbcb2f819dd")


def test_row_parameters():
    p = pack_parameters([{"aging": {"to": 1, "rate_s": 4.63e-6}}, {}])
    assert p["to"].tolist() == [1, -1]
    assert p["rate"].dtype == np.float64
    assert p["mw"].dtype == p["factor"].dtype == np.float32
    with pytest.raises(ValueError, match="chains"):
        pack_parameters([{"aging": {"to": 1, "rate_s": 4.63e-6}},
                         {"aging": {"to": 2, "rate_s": 4.63e-6}}, {}])


def test_case_coverage():
    cases = load(FIXTURES / "ageing")
    assert {float(c["dt"]) for c in cases.values()} == set(map(float, np.float32([30,36,60,7.5,1.9,3600.5,0.5])))
    c = cases["dt1"]
    assert np.min(c["input_bc1"]) == 0
    assert np.max(c["input_bc1"]) >= 1000
    assert np.any(c["step1_bc2"] != c["input_bc2"])
    assert np.any(c["step1_oc2"] != c["input_oc2"])


@requires_gpu
@pytest.mark.gpu
def test_wrf471_parity():
    baseline = json.loads((FIXTURES / "ageing/ULP_BASELINE.json").read_text())
    for name, case in load(FIXTURES / "ageing").items():
        assert aging_case(case) == baseline[name], name


@requires_gpu
@pytest.mark.gpu
def test_row_switches_move_outputs():
    from chem_gocart_checks import device3
    from gpuwm.core.chem_ageing import upload_parameters, launch_ageing
    case = load(FIXTURES / "ageing")["dt6"]
    labels = ("bc1", "bc2", "oc1", "oc2")

    def run(rate=4.63e-6, factor=8.0, swap=False):
        rows = [{"aging": {"to": 3 if swap else 1, "rate_s": rate,
                           "coproduct": {"to": 1 if swap else 3, "factor": factor}}}, {},
                {"aging": {"to": 1 if swap else 3, "rate_s": 4.63e-6}}, {}]
        arrays = [device3(case["input_"+label]) for label in labels]
        launch_ageing(arrays, upload_parameters(pack_parameters(rows)), case["dt"])
        return [a.get() for a in arrays]

    base = run()
    assert np.any(run(rate=9.26e-6)[0] != base[0])
    assert np.any(run(factor=0.0)[3] != base[3])
    assert np.any(run(swap=True)[1] != base[1])
