"""Prevent silently substituting MYNN EDMF mixing for scalar_pblmix.

The fixture comes from WRF's unmodified diff4d/diff/invert, whose reverse
elimination and prescribed top value differ from MYNN's scalar solve.
Every active number rate is checked bitwise, including strong mixing and
an aerosol inversion near the top. Precipitating scalars retain a sentinel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.mynn_scalar_mix import scalar_pblmix_column

DATA = Path(__file__).parent / "data" / "scalar_pblmix_wrf461.npz"
RECEIPT = DATA.with_suffix(".json")


def test_wrf_fixture_provenance_and_excluded_scalars():
    receipt = json.loads(RECEIPT.read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == receipt["fixture_sha256"]
    assert receipt["source_sha256"] == (
        "90336e30296991fb397ffde87649a4bd20eaa2b7dc6e90639b043810c8420b56")
    assert receipt["extracted_routines"] == ["diff4d", "diff", "invert"]
    with np.load(DATA) as data:
        assert data["qn"].shape == (8, 11, 50)
        np.testing.assert_array_equal(data["tendency"][:, 4:], np.float32(-765.25))
        assert np.count_nonzero(data["tendency"][2:, :4]) > 0


@pytest.mark.parametrize("case", range(8))
def test_scalar_pblmix_matches_wrf_fortran_bits(case):
    with np.load(DATA) as data:
        for species in range(4):
            solved, rate = scalar_pblmix_column(
                data["qn"][case, species], data["dz"][case, species],
                data["rho"][case, species], data["exch_h"][case, species],
                data["dt"][case, species, 0])
            np.testing.assert_array_equal(
                rate.view(np.uint32), data["tendency"][case, species].view(np.uint32))
            assert solved[-1] == data["qn"][case, species, -1]
            assert rate[-1] == np.float32(0)


def test_bottom_exchange_is_zero_flux_and_input_arrays_stay_unchanged():
    with np.load(DATA) as data:
        qn, dz, rho, kh = (data[key][2, 2].copy()
                           for key in ("qn", "dz", "rho", "exch_h"))
    before = [a.copy() for a in (qn, dz, rho, kh)]
    solved, rate = scalar_pblmix_column(qn, dz, rho, kh, 15.0)
    kh_zero = kh.copy()
    kh_zero[0] = 0.0
    solved_zero, rate_zero = scalar_pblmix_column(qn, dz, rho, kh_zero, 15.0)
    np.testing.assert_array_equal(solved, solved_zero)
    np.testing.assert_array_equal(rate, rate_zero)
    for actual, original in zip((qn, dz, rho, kh), before):
        np.testing.assert_array_equal(actual, original)


def test_scalar_pblmix_rejects_missing_vertical_extent():
    with pytest.raises(ValueError, match="nz >= 2"):
        scalar_pblmix_column([1], [1], [1], [1], 1)
