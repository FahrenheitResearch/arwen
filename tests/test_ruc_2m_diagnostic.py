"""RUC's 2 m diagnostic by name: public WRF's flux form, or the log profile.

``log_profile`` adds the block the operational RAP/HRRR branch carries in
``module_sf_sfcdiags_ruclsm.F:150-179`` and no public WRF 3.9 to 4.7.1 has.
The default stays ``flux`` (gpuwm.core.ruc_tier RUC_2M_DIAGNOSTIC_FORMS
records the measurement behind that).
"""

from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.ruc_runtime import _sfcdiags_log_profile
from gpuwm.core.ruc_tier import (RUC_2M_DIAGNOSTIC_DEFAULT,
                                 RUC_2M_DIAGNOSTIC_FORMS,
                                 ruc_2m_diagnostic_form)

f32 = np.float32


def test_the_default_is_the_flux_form():
    assert RUC_2M_DIAGNOSTIC_FORMS == ("flux", "log_profile")
    assert RUC_2M_DIAGNOSTIC_DEFAULT == "flux"
    assert RunConfig.__dataclass_fields__["ruc_2m_diagnostic"].default == "flux"
    assert ruc_2m_diagnostic_form("flux") == 0
    assert ruc_2m_diagnostic_form("log_profile") == 1


@pytest.mark.parametrize("bad", ["typo", "", 0, None, "LOG_PROFILE", "log"])
def test_an_unknown_form_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="ruc_2m_diagnostic"):
        ruc_2m_diagnostic_form(bad)


def _reference(tsk, t1, qlev1, qsfcmr, half, scale, t2, th2, q2):
    """The branch's lines in float64, one cell at a time."""
    d_t, d_q = t1 - tsk, qlev1 - qsfcmr
    if d_t > 0.0:
        fh = min(max(1.0 - d_t / 10.0, 0.01), 1.0)
        fac = math.log(2.05 / (0.05 + fh)) / math.log((half + 0.05) / (0.05 + fh))
        t2 = tsk + fac * d_t
        th2 = t2 * scale
    if d_q > 0.0:
        fh = min(max(1.0 - d_q / 0.003, 0.01), 1.0)
        fac = math.log(2.05 / (0.05 + fh)) / math.log((half + 0.05) / (0.05 + fh))
        q2 = qsfcmr + fac * d_q
    return t2, th2, q2


def test_the_log_profile_is_the_branch_expression():
    rng = np.random.default_rng(7)
    n = 400
    tsk = rng.uniform(270.0, 310.0, n).astype(f32)
    t1 = (tsk + rng.uniform(-6.0, 12.0, n)).astype(f32)
    qsfcmr = rng.uniform(0.002, 0.02, n).astype(f32)
    qlev1 = (qsfcmr + rng.uniform(-0.004, 0.006, n)).astype(f32)
    half = rng.uniform(3.0, 12.0, n).astype(f32)
    scale = rng.uniform(1.0, 1.05, n).astype(f32)
    t2 = ((tsk + t1) / f32(2.0)).astype(f32)
    th2 = (t2 * scale).astype(f32)
    q2 = ((qsfcmr + qlev1) / f32(2.0)).astype(f32)
    got = _sfcdiags_log_profile(tsk=tsk, t1=t1, qlev1=qlev1, qsfcmr=qsfcmr,
                                half_layer=half, scale=scale, t2=t2, th2=th2,
                                q2=q2)
    for i in range(n):
        want = _reference(float(tsk[i]), float(t1[i]), float(qlev1[i]),
                          float(qsfcmr[i]), float(half[i]), float(scale[i]),
                          float(t2[i]), float(th2[i]), float(q2[i]))
        for name, value, expected in zip(("t2", "th2", "q2"),
                                         (got[0][i], got[1][i], got[2][i]), want):
            assert abs(float(value) - expected) <= 2e-6 * max(1.0, abs(expected)), (
                name, i, float(value), expected)


def test_where_the_air_is_not_warmer_or_moister_the_flux_values_stand():
    tsk = np.array([300.0, 300.0], f32)
    t1 = np.array([299.0, 299.5], f32)
    qsfcmr = np.array([0.012, 0.010], f32)
    qlev1 = np.array([0.011, 0.010], f32)
    t2 = np.array([299.6, 299.8], f32)
    th2 = np.array([301.0, 301.2], f32)
    q2 = np.array([0.0115, 0.0101], f32)
    got = _sfcdiags_log_profile(tsk=tsk, t1=t1, qlev1=qlev1, qsfcmr=qsfcmr,
                                half_layer=np.full(2, 8.0, f32),
                                scale=np.full(2, 1.01, f32), t2=t2, th2=th2,
                                q2=q2)
    np.testing.assert_array_equal(got[0], t2)
    np.testing.assert_array_equal(got[1], th2)
    np.testing.assert_array_equal(got[2], q2)


def test_q2_has_no_saturation_cap():
    # A moist lowest level over a much drier surface: the profile value is
    # written as is (the branch's block runs after the flux form's cap).
    got = _sfcdiags_log_profile(
        tsk=np.array([280.0], f32), t1=np.array([279.0], f32),
        qlev1=np.array([0.030], f32), qsfcmr=np.array([0.001], f32),
        half_layer=np.array([8.0], f32), scale=np.array([1.0], f32),
        t2=np.array([279.5], f32), th2=np.array([279.5], f32),
        q2=np.array([0.0055], f32))
    assert float(got[2][0]) > 0.0055


def _cfg(**kwargs):
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_the_configuration_refuses_an_unknown_form():
    validate_run_config(_cfg(ruc_2m_diagnostic="log_profile"))
    with pytest.raises(ValueError, match="ruc_2m_diagnostic"):
        validate_run_config(_cfg(ruc_2m_diagnostic="profile"))


def test_restart_flip_refuses_and_the_old_header_reads_as_the_default():
    from gpuwm.io.restart import _require_config_match, configuration_echo
    cfg = _cfg()
    stored = asdict(cfg)
    stored.pop("ruc_2m_diagnostic")
    _require_config_match(stored, cfg, "checkpoint")
    assert "ruc_2m_diagnostic" not in configuration_echo(cfg)
    other = replace(cfg, ruc_2m_diagnostic="log_profile")
    assert configuration_echo(other)["ruc_2m_diagnostic"] == "log_profile"
    with pytest.raises(ValueError, match="ruc_2m_diagnostic"):
        _require_config_match(stored, other, "checkpoint")


def test_the_form_is_preparation_inert():
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    assert "run.ruc_2m_diagnostic" in PREPARATION_INERT_RUN_FIELDS
