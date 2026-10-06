"""Device prescribed-smoke optics against the compiled source fixture."""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

from gpuwm.core import rrtmg_aerosol_optics as ao

ORACLE = Path(__file__).parents[1] / "tests/data/smoke_aer3_oracle/smoke_aer3_oracle.npz"


def _words(got, expected, name):
    assert got.shape == expected.shape, name
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), name


def test_device_smoke_math_and_optics_match_the_source():
    import cupy as cp

    assert ORACLE.is_file(), "compile the source smoke oracle before running this gate"
    with np.load(ORACLE) as f:
        inputs = [cp.asarray(f[f"in/{key}"]) for key in ("p", "t", "qv", "dz8w", "nwfa", "nifa")]
        recovered = ao.smoke_dry_mixing_ratio_from_posted_density_device(
            cp.asarray(f["out/pm_posted_kgm3"]), inputs[0], inputs[1])
        _words(cp.asnumpy(recovered), f["out/smoke_recovered_ugkg"], "source posted inverse")
        aod = ao.smoke_aod_from_dry_mixing_ratio_device(
            cp.asarray(f["in/smoke_ugkg"]), cp.asarray(f["in/rho_dry"]), inputs[3])
        _words(cp.asnumpy(aod), f["out/smoke_aod"], "source smoke AOD")
        nz = aod.shape[1]
        tau, asy, ssa, combined = ao.aer3_sw_optics_device(
            *inputs, nz + 1, smoke_aod=aod, smoke_feedback=True)
        for value, key in ((tau, "tauaer"), (asy, "asyaer"), (ssa, "ssaaer")):
            expected = f[f"out/{key}"].transpose(0, 2, 1)
            _words(cp.asnumpy(value[:, :, :nz]), expected, key)
        _words(cp.asnumpy(combined), f["out/taod5503d"], "combined AOD")
        assert bool(cp.all(tau[:, :, nz] == 0))
        assert bool(cp.all(asy[:, :, nz] == 0))
        assert bool(cp.all(ssa[:, :, nz] == 1))


def test_explicit_zero_smoke_retains_default_optics_bytes():
    import cupy as cp

    assert ORACLE.is_file(), "compile the source smoke oracle before running this gate"
    with np.load(ORACLE) as f:
        inputs = [cp.asarray(f[f"in/{key}"]) for key in ("p", "t", "qv", "dz8w", "nwfa", "nifa")]
        nz = inputs[0].shape[1]
        old = ao.aer3_sw_optics_device(*inputs, nz + 1)
        zero = ao.aer3_sw_optics_device(
            *inputs, nz + 1, smoke_aod=cp.zeros_like(inputs[0]), smoke_feedback=True)
        for a, b in zip(old, zero):
            _words(cp.asnumpy(a), cp.asnumpy(b), "zero-smoke identity")


def test_device_missing_smoke_input_is_not_a_clean_sky_run():
    import cupy as cp

    values = [cp.full((1, 2), x, cp.float32) for x in (95000, 290, .005, 100, 1.e9, 1.e6)]
    with pytest.raises(ValueError, match="requires smoke_aod"):
        ao.aer3_sw_optics_device(*values, 3, smoke_feedback=True)
    with pytest.raises(ValueError, match="disabled"):
        ao.aer3_sw_optics_device(*values, 3, smoke_aod=cp.zeros((1, 2), cp.float32))
    with pytest.raises(ValueError, match="physically invalid"):
        ao.aer3_sw_optics_device(*values, 3,
            smoke_aod=cp.full((1, 2), np.nan, cp.float32), smoke_feedback=True)


def test_posted_density_device_inverse_uses_the_donor_coefficient():
    import cupy as cp

    p = np.array([[95000, 70000]], np.float32)
    t = np.array([[290, 250]], np.float32)
    q = np.array([[100, 25]], np.float32)
    emitted = ((np.float32(1) / np.float32(287.04)) * (p / t) * q) * np.float32(1.e-9)
    expected = ao.smoke_dry_mixing_ratio_from_posted_density(emitted, p, t)
    actual = ao.smoke_dry_mixing_ratio_from_posted_density_device(
        cp.asarray(emitted), cp.asarray(p), cp.asarray(t))
    _words(cp.asnumpy(actual), expected, "posted source inverse")
    with pytest.raises(ValueError, match="donor_pressure"):
        ao.smoke_dry_mixing_ratio_from_posted_density_device(
            cp.asarray(emitted), cp.zeros_like(actual), cp.asarray(t))


def test_model_dry_density_uses_explicit_alt_reciprocal():
    import cupy as cp

    alt = np.array([[1, 2, .25, 1.3]], np.float32)
    expected = np.float32(1) / alt
    _words(cp.asnumpy(ao.smoke_dry_air_density_device(cp.asarray(alt))), expected, "source 1/ALT")
    with pytest.raises(ValueError, match="ALT_dry_specific_volume.*invalid"):
        ao.smoke_dry_air_density_device(cp.zeros((1, 2), cp.float32))
