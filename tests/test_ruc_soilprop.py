"""The two WRF lineages of LSMRUC's SOILPROP, and which one a run takes.

WRF v4.6.1 (``phys/module_sf_ruclsm.F:6261-6267, :6289``) normalises the
soil-water diffusivity and hydraulic conductivity by total moisture over
porosity; WRF v4.0-4.5 (v4.5.2 ``:6213-6216, :6245``, the form the
operational RAP/HRRR branch carries) by the moisture above the residual.
In dry soil the v4.6.1 diffusivity is 2.5 to 8 times larger, which wets a
dry top soil level from below within the first forecast hour.
``ruc_soilprop`` names the lineage; ``wrf_45`` is the default.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.ruc import ruc_soil_properties
from gpuwm.core.ruc_tier import (RUC_SOILPROP_DEFAULT, RUC_SOILPROP_FORMS,
                                 ruc_kernel_source, ruc_module_defines,
                                 ruc_soilprop_form)

f32 = np.float32


def _oracle_inputs():
    from test_ruc import _soilprop_oracle
    _, profiles, columns = _soilprop_oracle()
    inputs = {name: profiles[name].copy() for name in (
        "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
        "soilice", "soilmoism", "soiliqwm", "soilicem")}
    inputs.update({name: value.copy() for name, value in columns.items()})
    return inputs, profiles


def test_the_default_is_the_v45_lineage():
    assert RUC_SOILPROP_DEFAULT == "wrf_45"
    assert RUC_SOILPROP_FORMS == ("wrf_45", "wrf_461")
    assert RunConfig.__dataclass_fields__["ruc_soilprop"].default == "wrf_45"


@pytest.mark.parametrize("bad", ["typo", "", 0, None, "WRF_45"])
def test_an_unknown_lineage_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="ruc_soilprop"):
        ruc_soilprop_form(bad)
    with pytest.raises(ValueError, match="ruc_soilprop"):
        ruc_soil_properties(_oracle_inputs()[0], soilprop=bad)


def test_the_v461_name_still_reproduces_the_unmodified_wrf_oracle():
    inputs, profiles = _oracle_inputs()
    actual = ruc_soil_properties(inputs, soilprop="wrf_461")
    for name in ("thdif", "diffu", "hydro", "cap"):
        np.testing.assert_allclose(getattr(actual, name), profiles[name],
                                   rtol=2.0e-6, atol=2.0e-8, err_msg=name)


def test_the_default_diffusivity_is_the_v45_expression():
    """v4.5.2 :6213-6216 on ice-free levels: (theta - qmin) over dqm."""
    inputs, _ = _oracle_inputs()
    v45 = ruc_soil_properties(inputs, soilprop="wrf_45")
    v461 = ruc_soil_properties(inputs, soilprop="wrf_461")
    dqm, qmin = inputs["dqm"].astype(np.float64), inputs["qmin"].astype(np.float64)
    bclh, ksat = inputs["bclh"].astype(np.float64), inputs["ksat"].astype(np.float64)
    psis = inputs["psis"].astype(np.float64)
    checked = 0
    for column in range(dqm.size):
        if dqm[column] + qmin[column] < 0.12:
            continue
        for level in range(8):
            if inputs["soilicem"][level, column] != 0.0:
                continue
            theta = float(inputs["soilmoism"][level, column])
            h = max(0.0, theta / dqm[column])
            expected = (-bclh[column] * ksat[column] * psis[column] / dqm[column]
                        * h ** (bclh[column] + 2.0))
            np.testing.assert_allclose(v45.diffu[level, column], expected,
                                       rtol=2.0e-5, atol=1.0e-20)
            checked += 1
    assert checked >= 8
    # Only the water terms moved: the heat capacity is the same expression.
    np.testing.assert_array_equal(v45.cap, v461.cap)


def test_a_dry_top_level_gets_several_times_the_v45_diffusivity_under_v461():
    """STAS-RUC loam at the measured afternoon state (theta 0.161)."""
    loam = dict(bclh=5.39, dqm=0.451 - 0.050, qmin=0.050, psis=-0.478,
                ksat=6.95e-6, qwrtz=0.40, rhocs=1.21e6)
    nzs, ncol = 9, 1
    theta = np.linspace(0.161, 0.30, nzs).astype(f32)[:, None]
    inputs = {name: np.full((nzs, ncol), f32(0.0)) for name in (
        "fwsat", "keepfr", "soilice", "soilicem")}
    inputs.update(lwsat=np.ones((nzs, ncol), f32), tav=np.full((nzs, ncol), f32(295.0)),
                  soilmois=(theta - f32(0.050)).astype(f32), soiliqw=(theta - f32(0.050)).astype(f32),
                  soilmoism=(theta - f32(0.050)).astype(f32), soiliqwm=(theta - f32(0.050)).astype(f32))
    inputs.update({name: np.array([value], f32) for name, value in loam.items()})
    ratio = (ruc_soil_properties(inputs, soilprop="wrf_461").diffu[0, 0]
             / ruc_soil_properties(inputs, soilprop="wrf_45").diffu[0, 0])
    # (0.161/0.451)**7.39 * 0.401/0.451 over (0.111/0.401)**7.39: about 5.8.
    assert 4.0 < ratio < 8.0, ratio


def test_the_translation_unit_carries_the_lineage_only_off_its_default():
    assert ruc_module_defines(9) == ()
    assert ruc_module_defines(9, "wrf_461") == (("GPUWM_SOILPROP_WRF461", 1),)
    assert ruc_module_defines(6, "wrf_461") == (
        ("RUC_NZS", 6), ("GPUWM_SOILPROP_WRF461", 1))
    source = ruc_kernel_source(9, "wrf_461")
    assert "#define GPUWM_SOILPROP_WRF461 1" in source
    assert "#define GPUWM_SOILPROP_WRF461" not in ruc_kernel_source(9)
    from gpuwm.core.ruc_tier import ruc_fused_source
    assert "#define GPUWM_SOILPROP_WRF461 1" in ruc_fused_source(9, soilprop="wrf_461")
    assert "#define GPUWM_SOILPROP_WRF461" not in ruc_fused_source(9)


def _top_level_gain(steps, soilprop, case):
    """Snow-free ``soil`` on one oracle column with a dry top level, no rain.

    The column keeps its own wetter levels below; the top level starts at
    30 percent of level 2, and each call's state feeds the next.
    """
    from gpuwm.core.ruc import ruc_soil_step
    from test_ruc import _soil_inputs, _soil_oracle

    _, profiles, columns = _soil_oracle()
    values = {name: np.array(value[..., case:case + 1], copy=True)
              for name, value in _soil_inputs(profiles, columns).items()}
    values["prcpms"] = np.zeros_like(values["prcpms"])
    dry = f32(f32(0.3) * values["soilmois"][1, 0])
    values["soilmois"][0, 0] = dry
    gains = []
    for _ in range(steps):
        result = ruc_soil_step(
            values, columns["iland"][case:case + 1].astype(np.int32),
            nroot=columns["nroot"][case:case + 1].astype(np.int32),
            delt=float(columns["delt"][case]),
            conflx=float(columns["conflx"][case]), soilprop=soilprop)
        for name in ("soilmois", "tso", "smfrkeep", "keepfr", "cst", "soilt",
                     "qvg", "qsg", "qcg", "mavail"):
            values[name] = np.array(getattr(result, name), dtype=np.float32)
        gains.append(float(values["soilmois"][0, 0]) - float(dry))
    return gains


def test_a_dry_top_level_is_refilled_from_below_faster_under_v461():
    """The defect's signature: a dry top level over wetter soil, many steps.

    Oracle column 3 is a loam-class soil (DRYSMC 0.050).  MEASURED on it:
    one 60 s step lifts the dry top level 0.0039 m3/m3 under wrf_45 and
    0.0087 under wrf_461, with no rain in either.
    """
    v45 = _top_level_gain(5, "wrf_45", 2)
    v461 = _top_level_gain(5, "wrf_461", 2)
    assert v45[0] > 0.0 and v461[0] > 1.5 * v45[0], (v45, v461)
    assert all(b > a for a, b in zip(v45, v461)), (v45, v461)
    # Sand (DRYSMC 0.002): residual and total moisture nearly coincide, so
    # the two lineages nearly do too.
    sand45 = _top_level_gain(1, "wrf_45", 0)[0]
    sand461 = _top_level_gain(1, "wrf_461", 0)[0]
    assert abs(sand461 - sand45) < 0.02 * abs(sand45), (sand45, sand461)


def _cfg(**kwargs):
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_the_configuration_refuses_an_unknown_lineage():
    validate_run_config(_cfg(ruc_soilprop="wrf_461"))
    with pytest.raises(ValueError, match="ruc_soilprop"):
        validate_run_config(_cfg(ruc_soilprop="wrf_46"))


def test_restart_flip_refuses_and_the_old_header_reads_as_the_default():
    from gpuwm.io.restart import (_configuration_digest_values,
                                  _require_config_match, configuration_echo)
    cfg = _cfg()
    stored = asdict(cfg)
    stored.pop("ruc_soilprop")
    _require_config_match(stored, cfg, "checkpoint")
    assert _configuration_digest_values(stored) == _configuration_digest_values(
        asdict(cfg))
    assert "ruc_soilprop" not in configuration_echo(cfg)
    legacy = replace(cfg, ruc_soilprop="wrf_461")
    assert configuration_echo(legacy)["ruc_soilprop"] == "wrf_461"
    with pytest.raises(ValueError, match="ruc_soilprop"):
        _require_config_match(stored, legacy, "checkpoint")
    with pytest.raises(ValueError, match="ruc_soilprop"):
        _require_config_match(asdict(legacy), cfg, "checkpoint")


def test_the_lineage_is_preparation_inert():
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    assert "run.ruc_soilprop" in PREPARATION_INERT_RUN_FIELDS
