"""Shared comparison helpers. Physics is executed only by the CUDA launchers."""
import hashlib
from pathlib import Path

import numpy as np

from gpuwm.verify.chem_oracle import ulp_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/data/oracles/chem/gocart"


def check_fixture_hashes(process, pin):
    folder = FIXTURES / process
    sums = folder / "oracle-sha256sums.txt"
    assert hashlib.sha256(sums.read_bytes()).hexdigest() == pin
    for line in sums.read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((folder / rel).read_bytes()).hexdigest() == digest, rel


def device3(a):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(a.transpose(1, 2, 0)))


def device2(a):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(a.T))


def compare3(a, ref):
    return ulp_table(a.get().transpose(2, 0, 1), ref)


def aging_case(case):
    import cupy as cp
    from gpuwm.core.chem_ageing import pack_parameters, upload_parameters, launch_ageing
    # Deliberately permute WRF's order and add an unrelated row.
    labels = ("oc2", "bc1", "bc2", "oc1")
    arrays = [device3(case["input_"+label]) for label in labels]
    untouched = cp.full_like(arrays[0], 17.0)
    arrays.append(untouched)
    rows = [{}, {"aging": {"to": 2, "rate_s": 4.63e-6,
                          "coproduct": {"to": 0, "factor": 8.0}}}, {},
            {"aging": {"to": 0, "rate_s": 4.63e-6}}, {}]
    params = upload_parameters(pack_parameters(rows))
    workspace = None
    tables = {}
    for step in range(1, 4):
        workspace = launch_ageing(arrays, params, case["dt"], workspace=workspace)
        for label, a in zip(labels, arrays):
            key = f"step{step}_{label}"
            tables[key] = compare3(a, case[key])
            np.testing.assert_array_equal(a[-1].get().T, case["input_"+label][:, -1, :])
        np.testing.assert_array_equal(untouched.get(), 17.0)
    return tables


def sulfur_case(case, *, oracle_solar=False, lite=False):
    import cupy as cp
    from gpuwm.core.chem_sulfur import (pack_parameters, upload_parameters,
        launch_solar_init, launch_solar_step, launch_sulfur)
    labels = ("msa", "so4", "dms", "so2") if not lite else ("so4", "so2")
    arrays = [device3(case["input_"+label]) for label in labels]
    untouched = cp.full_like(arrays[0], 17.0)
    arrays.append(untouched)
    rows = [{"sulfur": {"role": label}} for label in labels] + [{}]
    roles = upload_parameters(pack_parameters(rows))
    lat, lon = device2(case["latitude"]), device2(case["longitude"])
    if oracle_solar:
        tcosz, ttday = device2(case["tcosz"]), device2(case["ttday"])
    else:
        tcosz, ttday = launch_solar_init(lat, lon, case["dt"], case["gmt"], case["julday"])
    fields = [device3(case[key]) for key in
              ("temp", "rho", "backg_oh", "backg_h2o2", "backg_no3")]
    qc, qi = device3(case["qc"]), device3(case["qi"])
    gd_cldf = device3(case["gd_cldf"]) if case["optional_cloud"] else None
    tables = {"tcosz": ulp_table(tcosz.get().T, case["tcosz"]),
              "ttday": ulp_table(ttday.get().T, case["ttday"])}
    for step in range(1, 4):
        key = f"cossza_{step}"
        if oracle_solar:
            cosz = device2(case[key])
        else:
            cosz = launch_solar_step(lat, lon, float(case[f"curr_secs_{step}"]),
                                    case["gmt"], case["julday"])
        tables[key] = ulp_table(cosz.get().T, case[key])
        launch_sulfur(arrays, roles, *fields, cosz, tcosz, ttday, case["dt"],
                      qc=qc, qi=qi, gd_cldf=gd_cldf)
        for label, a in zip(labels, arrays):
            key = f"step{step}_{label}"
            tables[key] = compare3(a, case[key])
            np.testing.assert_array_equal(a[-1].get().T, case["input_"+label][:, -1, :])
        np.testing.assert_array_equal(untouched.get(), 17.0)
    return tables
