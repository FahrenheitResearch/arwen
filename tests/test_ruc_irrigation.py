"""The two WRF lineages of LSMRUC irrigation, and which one a run takes.

WRF v4.6.1 (``phys/module_sf_ruclsm.F:985-1009``) relaxes every root layer
toward ``1.1*WLTSMC - DRYSMC`` on EVERY step for any cell with any cropland
or crop/natural fraction, so a cell with a sliver of cropland reaches 1.1
times its wilting point within the first forecast hour.  WRF v4.0-4.5
(``:970-999``, the rule the operational RAP/HRRR branch carries) holds the
layers at a hard floor scaled by the crop FRACTION, which is idempotent.
``ruc_irrigation`` names the rule; ``wrf_461`` is the generic default.
The HRRR importer and recipe select ``wrf_45`` explicitly.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.ruc_mosaic import (IRRIGATION_DEFAULT, IRRIGATION_FORMS,
                                   irrigate, irrigation_form)

f32 = np.float32
#: STAS-RUC loam: WLTSMC 0.137, DRYSMC 0.050, so cropsm = 1.1*0.137 - 0.050.
WILT = f32(0.137)
QMIN = f32(0.050)
CROP, NATURAL = 12, 14


def _column(*, crop=0.05, natural=0.0, soil=0.03, nroot=4, lai=5.68,
            ivgtyp=10, nzs=9, ncat=21):
    landusef = np.zeros((ncat, 1), f32)
    landusef[CROP - 1, 0] = crop
    landusef[NATURAL - 1, 0] = natural
    soilm1d = np.full((nzs, 1), soil, f32)
    common = dict(landusef=landusef, vegfrac=np.array([80.0], f32),
                  shdmin=np.array([10.0], f32), shdmax=np.array([90.0], f32),
                  wilt=np.array([WILT]), qmin=np.array([QMIN]),
                  nroot=np.array([nroot], np.int32), crop=CROP, natural=NATURAL,
                  active=np.array([True]), lai=np.array([lai], f32),
                  ivgtyp=np.array([ivgtyp], np.int32))
    return soilm1d, common


def _cropsm(scale):
    return f32(f32(f32(scale) * WILT) - QMIN)


def test_the_generic_default_is_the_v461_relaxation():
    assert IRRIGATION_DEFAULT == "wrf_461"
    assert IRRIGATION_FORMS == ("wrf_45", "wrf_461")
    assert RunConfig.__dataclass_fields__["ruc_irrigation"].default == "wrf_461"
    assert irrigation_form("wrf_45") == 0 and irrigation_form("wrf_461") == 1


@pytest.mark.parametrize("bad", ["typo", "", 0, None, "WRF_45"])
def test_an_unknown_rule_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="ruc_irrigation"):
        irrigation_form(bad)


def test_the_v461_rule_relaxes_a_sliver_of_cropland_to_the_full_wilting_floor():
    # 5 percent cropland, greenness factor (80-10)/(90-10) = 0.875 > 0.75.
    soilm1d, common = _column(crop=0.05)
    first = None
    for _ in range(180):
        irrigate(soilm1d, form="wrf_461", **common)
        first = soilm1d[0, 0] if first is None else first
    # One step moves 5 percent of the way; 180 steps (an hour at 20 s)
    # arrive at 1.1 x wilting point whatever the crop fraction was.
    assert first < f32(0.04)
    assert abs(float(soilm1d[0, 0]) - float(_cropsm(1.1))) < 1e-4
    assert np.all(soilm1d[:4, 0] == soilm1d[0, 0])
    assert np.all(soilm1d[4:, 0] == f32(0.03))


def test_the_v45_rule_holds_a_sliver_of_cropland_at_its_own_share():
    soilm1d, common = _column(crop=0.05)
    for _ in range(180):
        irrigate(soilm1d, form="wrf_45", **common)
    # The floor is cropsm * 0.05 = 0.005, below the 0.03 the column holds.
    assert np.all(soilm1d == f32(0.03))


def test_the_v45_floor_is_the_fortran_expression_and_idempotent():
    soilm1d, common = _column(crop=0.6, soil=0.01)
    irrigate(soilm1d, form="wrf_45", **common)
    floor = f32(_cropsm(1.1) * f32(0.6))
    assert np.all(soilm1d[:4, 0] == floor)
    assert np.all(soilm1d[4:, 0] == f32(0.01))
    again = soilm1d.copy()
    for _ in range(50):
        irrigate(again, form="wrf_45", **common)
    np.testing.assert_array_equal(again, soilm1d)


def test_the_v45_floor_reads_the_leaf_area_the_scheme_runs_with():
    # The crop arm needs LAI above 1.1; a dormant 1.0 adds nothing.
    soilm1d, common = _column(crop=0.6, soil=0.01, lai=1.0)
    irrigate(soilm1d, form="wrf_45", **common)
    assert np.all(soilm1d == f32(0.01))
    soilm1d, common = _column(crop=0.6, soil=0.01, lai=1.1000001)
    irrigate(soilm1d, form="wrf_45", **common)
    assert soilm1d[0, 0] > f32(0.01)


def test_the_natural_arm_takes_a_dominant_mosaic_cell_at_forty_percent():
    soilm1d, common = _column(crop=0.0, natural=0.5, soil=0.01, lai=0.8,
                              ivgtyp=NATURAL)
    irrigate(soilm1d, form="wrf_45", **common)
    floor = f32(f32(_cropsm(1.2) * f32(0.5)) * f32(0.4))
    assert np.all(soilm1d[:4, 0] == floor)
    # Not dominant, or LAI at or below 0.7: nothing.
    for kwargs in (dict(ivgtyp=10), dict(lai=0.7)):
        soilm1d, common = _column(crop=0.0, natural=0.5, soil=0.01, lai=0.8,
                                  ivgtyp=NATURAL)
        common.update({k: np.array([v], common[k].dtype) for k, v in kwargs.items()})
        irrigate(soilm1d, form="wrf_45", **common)
        assert np.all(soilm1d == f32(0.01))
    # The crop arm wins the ELSEIF when it applies.
    soilm1d, common = _column(crop=0.2, natural=0.5, soil=0.01, lai=5.0,
                              ivgtyp=NATURAL)
    irrigate(soilm1d, form="wrf_45", **common)
    assert np.all(soilm1d[:4, 0] == f32(_cropsm(1.1) * f32(0.2)))


def test_without_landusef_the_dominant_category_is_the_whole_cell():
    """mosaic_lu = 0: the run carries no fractions, WRF's floor still runs.

    WRF fills lufrac from LANDUSEF whatever mosaic_lu says; with no
    fractions the dominant category stands for the whole cell.
    """
    def column(ivgtyp, lai=5.68, soil=0.01):
        soilm1d, common = _column(soil=soil, lai=lai, ivgtyp=ivgtyp)
        common["landusef"] = None
        return soilm1d, common

    # Dominant cropland: the full (1.1*WLTSMC - DRYSMC) floor, idempotent.
    soilm1d, common = column(CROP)
    for _ in range(180):
        irrigate(soilm1d, form="wrf_45", **common)
    assert np.all(soilm1d[:4, 0] == _cropsm(1.1))
    assert np.all(soilm1d[4:, 0] == f32(0.01))
    # Dominant crop/natural mosaic: forty percent of the 1.2 x floor.
    soilm1d, common = column(NATURAL, lai=0.8)
    irrigate(soilm1d, form="wrf_45", **common)
    assert np.all(soilm1d[:4, 0] == f32(_cropsm(1.2) * f32(0.4)))
    # Any other category, or a leaf area at or below the gate: nothing.
    for ivgtyp, lai in ((10, 5.68), (CROP, 1.1), (NATURAL, 0.7)):
        soilm1d, common = column(ivgtyp, lai=lai)
        irrigate(soilm1d, form="wrf_45", **common)
        assert np.all(soilm1d == f32(0.01)), (ivgtyp, lai)
    # A wet column is never lowered.
    soilm1d, common = column(CROP, soil=0.3)
    irrigate(soilm1d, form="wrf_45", **common)
    assert np.all(soilm1d == f32(0.3))
    # The v4.6.1 rule has no meaning without fractions.
    soilm1d, common = column(CROP)
    with pytest.raises(ValueError, match="LANDUSEF"):
        irrigate(soilm1d, form="wrf_461", **common)


def test_both_branches_hold_a_small_crop_share_over_many_steps():
    """The defect's signature: a column with a small crop share, many steps.

    Under wrf_461 a 5 percent crop cell climbs to 1.1 x wilting point; under
    wrf_45 it stays where it is with fractions (mosaic_lu = 1) and
    without them (mosaic_lu = 0, dominant category not cropland).
    """
    target = float(_cropsm(1.1))
    soilm1d, common = _column(crop=0.05, soil=0.03)
    for _ in range(360):
        irrigate(soilm1d, form="wrf_461", **common)
    assert abs(float(soilm1d[0, 0]) - target) < 1e-4
    for landusef in ("fractions", None):
        soilm1d, common = _column(crop=0.05, soil=0.03, ivgtyp=10)
        if landusef is None:
            common["landusef"] = None
        for _ in range(360):
            irrigate(soilm1d, form="wrf_45", **common)
        assert np.all(soilm1d == f32(0.03)), landusef


def test_the_v45_rule_needs_its_gates():
    soilm1d, common = _column(crop=0.6)
    common.pop("lai")
    with pytest.raises(ValueError, match="leaf area"):
        irrigate(soilm1d, form="wrf_45", **common)


def _single_column(values, keywords):
    column = {name: np.ascontiguousarray(value[..., :1])
              for name, value in values.items()}
    narrowed = dict(keywords)
    for name in ("ivgtyp", "isltyp"):
        narrowed[name] = np.ascontiguousarray(keywords[name][:1])
    for name in ("landusef", "soilctop"):
        narrowed[name] = np.ascontiguousarray(keywords[name][:, :1])
    return column, narrowed


def _top_layer_after(steps, irrigation):
    from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_STATE,
                                RUC_DRIVER_PROFILE_STATE,
                                ruc_land_surface_step)
    from ruc_mosaic_fixture import driver_calls

    _label, values, keywords, _expected = next(driver_calls())
    values, keywords = _single_column(values, keywords)
    keywords["irrigation"] = irrigation
    ktau = int(keywords["ktau"])
    history = []
    for step in range(steps):
        keywords["ktau"] = ktau + step
        result = ruc_land_surface_step(values, **keywords)
        for name in RUC_DRIVER_PROFILE_STATE + RUC_DRIVER_COLUMN_STATE:
            values[name] = np.ascontiguousarray(
                np.asarray(getattr(result, name), dtype=np.float32))
        history.append(float(values["soilmois"][0, 0]))
    return history


def test_the_v45_driver_does_not_keep_adding_soil_water_step_after_step():
    """The floor is idempotent; v4.6.1 relaxation keeps lifting the layer."""
    history = _top_layer_after(40, "wrf_45")
    # Whatever the first step's floor did, the next 39 add nothing more:
    # a hard floor is idempotent, and evaporation can only lower the layer.
    assert max(history[1:]) <= history[0] + 1e-5, history[:5] + history[-3:]
    legacy = _top_layer_after(40, "wrf_461")
    # The same column under the v4.6.1 name: the floor keeps climbing.
    assert legacy[-1] > legacy[0] + 1e-3, legacy[:5] + legacy[-3:]
    assert legacy[-1] > history[-1] + 1e-3


def _cfg(**kwargs):
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_the_configuration_refuses_an_unknown_rule():
    validate_run_config(_cfg(ruc_irrigation="wrf_461"))
    with pytest.raises(ValueError, match="ruc_irrigation"):
        validate_run_config(_cfg(ruc_irrigation="wrf_460"))


def test_restart_flip_refuses_and_the_old_header_reads_as_the_default():
    from gpuwm.io.restart import (_configuration_digest_values,
                                  _require_config_match, configuration_echo)
    cfg = _cfg()
    stored = asdict(cfg)
    stored.pop("ruc_irrigation")
    _require_config_match(stored, cfg, "checkpoint")
    assert _configuration_digest_values(stored) == _configuration_digest_values(
        asdict(cfg))
    assert "ruc_irrigation" not in configuration_echo(cfg)
    fork = replace(cfg, ruc_irrigation="wrf_45")
    assert configuration_echo(fork)["ruc_irrigation"] == "wrf_45"
    with pytest.raises(ValueError, match="ruc_irrigation"):
        _require_config_match(stored, fork, "checkpoint")
    with pytest.raises(ValueError, match="ruc_irrigation"):
        _require_config_match(asdict(fork), cfg, "checkpoint")


def test_the_rule_is_preparation_inert_and_fingerprint_bound():
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.experiment import build_experiment
    from gpuwm.ingest.prepared_cache import (
        PREPARATION_INERT_RUN_FIELDS, compare_prepared_domain_config,
        prepared_domain_config_identity)
    from datetime import datetime

    document = {
        "experiment": {"name": "irrigation-rule",
                       "start_time": datetime(2020, 1, 1),
                       "run_seconds": 60., "restart_interval_s": 0.},
        "shared": {"nz": 8, "ztop": 10000.},
        "domain": [{"grid_id": 1, "parent_id": 0, "nx": 24, "ny": 24,
                    "i_parent_start": 1, "j_parent_start": 1,
                    "parent_grid_ratio": 1, "parent_time_step_ratio": 1,
                    "dx": 3000., "time_step": 6, "history_interval_s": 60.}],
    }
    default = build_experiment(document, source="irrigation rule test")
    assert default.root.run.ruc_irrigation == "wrf_461"
    assert "ruc_irrigation" not in str(restart_identity_payload(default))
    document["shared"]["ruc_irrigation"] = "wrf_45"
    fork = build_experiment(document, source="irrigation rule test")
    assert fork.root.run.ruc_irrigation == "wrf_45"
    assert restart_identity_payload(default) != restart_identity_payload(fork)
    assert "run.ruc_irrigation" in PREPARATION_INERT_RUN_FIELDS
    old = prepared_domain_config_identity(default.root)
    new = prepared_domain_config_identity(fork.root)
    assert not compare_prepared_domain_config(old, new)[1]
    # Current public documents omit the generic default. Older documents
    # could carry it explicitly; both must reuse the same preparation.
    assert "ruc_irrigation" not in old["run"]
    old["run"]["ruc_irrigation"] = "wrf_461"
    assert not compare_prepared_domain_config(old, new)[1]
    old["run"].pop("ruc_irrigation", None)
    assert not compare_prepared_domain_config(old, new)[1]
