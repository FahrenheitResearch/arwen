"""LSMRUC's cold start of the ground vapour, by lineage.

Public WRF (v4.6.1 ``phys/module_sf_ruclsm.F:505-514``) starts an invalid
QVG from saturation at the skin times moisture availability and an invalid
QCG from the lowest-level condensate.  The operational RAP/HRRR branch
(``:479-483`` there) starts an invalid QVG from the lowest-level vapour and
sets QCG to zero, with no separate QCG check.  ``ruc_qvg_cold_start``
names the form; the default is ``wrf`` everywhere.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.ruc_tier import (RUC_QVG_COLD_START_DEFAULT,
                                 RUC_QVG_COLD_START_FORMS,
                                 ruc_qvg_cold_start_form)

f32 = np.float32


def test_the_default_is_the_public_form():
    assert RUC_QVG_COLD_START_FORMS == ("air", "wrf")
    assert RUC_QVG_COLD_START_DEFAULT == "wrf"
    assert RunConfig.__dataclass_fields__["ruc_qvg_cold_start"].default == "wrf"
    assert ruc_qvg_cold_start_form("air") == 1
    assert ruc_qvg_cold_start_form("wrf") == 0


@pytest.mark.parametrize("bad", ["typo", "", 0, None, "AIR", "wrf_461"])
def test_an_unknown_form_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        ruc_qvg_cold_start_form(bad)


def _first_call(form, *, qvg, qcg):
    """One cold LSMRUC call on one oracle column with QVG and QCG preset."""
    from gpuwm.core.ruc import ruc_land_surface_step
    from ruc_mosaic_fixture import driver_calls

    _label, values, keywords, _expected = next(driver_calls())
    values = {name: np.ascontiguousarray(value[..., :1])
              for name, value in values.items()}
    for name in ("ivgtyp", "isltyp"):
        keywords[name] = np.ascontiguousarray(keywords[name][:1])
    for name in ("landusef", "soilctop"):
        keywords[name] = np.ascontiguousarray(keywords[name][:, :1])
    keywords["ktau"] = 1
    keywords["qvg_cold_start"] = form
    values["qvg"][...] = f32(qvg)
    values["qcg"][...] = f32(qcg)
    return values, ruc_land_surface_step(dict(values), **keywords)


def test_the_two_forms_start_different_ground_vapour():
    values, air = _first_call("air", qvg=0.0, qcg=0.0)
    _, wrf = _first_call("wrf", qvg=0.0, qcg=0.0)
    # Both start, then SFCTMP advances QVG one step, so compare the start
    # through what it changes: the two forms give different surface states.
    assert not np.array_equal(np.asarray(air.qvg), np.asarray(wrf.qvg))
    assert float(values["qv3d"][0]) > 0.0


def test_a_valid_ground_vapour_is_kept_by_both_forms():
    _, air = _first_call("air", qvg=0.01, qcg=0.0)
    _, wrf = _first_call("wrf", qvg=0.01, qcg=0.0)
    for name in ("qvg", "qsfc", "soilt", "hfx", "qfx"):
        np.testing.assert_array_equal(np.asarray(getattr(air, name)),
                                      np.asarray(getattr(wrf, name)),
                                      err_msg=name)


def test_the_transcription_default_is_the_oracle_form():
    import inspect

    from gpuwm.core.ruc import ruc_land_surface_step
    default = inspect.signature(ruc_land_surface_step).parameters[
        "qvg_cold_start"].default
    assert default == "wrf"


def _cfg(**kwargs):
    base = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                dt=6.0, run_seconds=60.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_the_configuration_refuses_an_unknown_form():
    validate_run_config(_cfg(ruc_qvg_cold_start="air"))
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        validate_run_config(_cfg(ruc_qvg_cold_start="saturated"))


def test_restart_flip_refuses_and_the_old_header_reads_as_the_default():
    from gpuwm.io.restart import _require_config_match, configuration_echo
    cfg = _cfg()
    stored = asdict(cfg)
    stored.pop("ruc_qvg_cold_start")
    _require_config_match(stored, cfg, "checkpoint")
    assert "ruc_qvg_cold_start" not in configuration_echo(cfg)
    legacy = replace(cfg, ruc_qvg_cold_start="air")
    assert configuration_echo(legacy)["ruc_qvg_cold_start"] == "air"
    with pytest.raises(ValueError, match="ruc_qvg_cold_start"):
        _require_config_match(stored, legacy, "checkpoint")


def test_the_form_is_preparation_inert():
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    assert "run.ruc_qvg_cold_start" in PREPARATION_INERT_RUN_FIELDS
