"""Sulfur sources must fill three prescribed daily-mean oxidant fields.

The source catalog's acquisition grammar previously passed configuration
while no species row carried OH, H2O2 or NO3. The first sulfur process then
failed after meteorological preparation. Keep that failure at the door and
check that an enabled source cannot stand in for its absent state fields.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm.chem_table import SpeciesRow, load_sets
from gpuwm.core import chem_sulfur


def _cfg(sets="gocart_lite", sources="cams-oxidants"):
    return SimpleNamespace(chem_sets=sets, chem_sources=sources)


@pytest.mark.parametrize("sets", ["gocart_lite", "gocart_simple"])
def test_catalog_oxidant_source_cannot_replace_absent_background_rows(sets):
    reason = chem_sulfur.refusal(_cfg(sets))
    assert "['oh', 'h2o2', 'no3']" in reason
    assert "active chem table" in reason
    assert "source alone supplies no state field" in reason


@pytest.mark.parametrize("sets", ["gocart_lite", "gocart_simple"])
def test_the_normal_configuration_door_rejects_missing_prescribed_inputs(sets):
    from gpuwm.config import RunConfig, validate_chem_config

    cfg = RunConfig(nx=12, ny=10, nz=49, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=18.0, run_seconds=3600.0,
                    chem_sets=sets, chem_sources="cams-oxidants",
                    bl_pbl_physics=1, sf_sfclay_physics=1,
                    sf_surface_physics=2)
    with pytest.raises(ValueError, match="chem process chem.sulfur.*state field"):
        validate_chem_config(cfg)


@pytest.mark.parametrize("sets", ["", "cams_aq", "gocart_primary", "dust"])
def test_non_sulfur_sets_do_not_acquire_a_new_refusal(sets):
    assert chem_sulfur.refusal(_cfg(sets)) is None


def test_missing_oxidants_stop_initialization_before_importing_a_device():
    ctx = SimpleNamespace(table=load_sets(("gocart_lite",),
                                         ("cams-oxidants",)),
                          cfg=_cfg())
    with pytest.raises(ValueError, match="source alone supplies no state field"):
        chem_sulfur.init(ctx)


def _inputs():
    species = load_sets(("gocart_lite",)).row("so2")
    oxidants = [
        SpeciesRow(name=f"background_{role}", output_name=None,
                   long_name=f"prescribed {role}", units="mol mol-1",
                   phase="gas", family="oxidant", sets=("gocart_lite",),
                   default_inflow=0.0, processes=("chem.sulfur",),
                   provenance="test prescribed input", source_file="test",
                   transported=False, sulfur_role=role,
                   boundary=({"source": "background", "fields": [role],
                              "weights": [1.0], "conversion": "identity"},))
        for role in chem_sulfur.OXIDANT_ROLES]
    source = SimpleNamespace(kind="oxidant", acquisition={"kind": "cads"},
                             time={"interpolation": "daily_mean"})
    return species, oxidants, source


def _table(species, oxidants, source):
    return SimpleNamespace(rows_for=lambda _key: (species, *oxidants),
                           sources={"background": source})


def test_partial_prescribed_inputs_name_the_missing_role():
    species, oxidants, source = _inputs()
    reason = chem_sulfur._oxidant_refusal(
        _table(species, oxidants[:-1], source), ("background",))
    assert "['no3']" in reason


@pytest.mark.parametrize("change", [dict(transported=True), dict(units="ppmv")])
def test_oxidants_must_be_prescribed_molar_daily_means(change):
    species, oxidants, source = _inputs()
    oxidants[0] = replace(oxidants[0], **change)
    reason = chem_sulfur._oxidant_refusal(
        _table(species, oxidants, source), ("background",))
    assert "must be prescribed in mol mol-1" in reason


@pytest.mark.parametrize("change", [
    dict(acquisition=None), dict(kind="boundary"),
    dict(time={"interpolation": "linear"})])
def test_an_unserved_or_instantaneous_source_is_not_a_daily_mean(change):
    species, oxidants, source = _inputs()
    source = SimpleNamespace(**(vars(source) | change))
    reason = chem_sulfur._oxidant_refusal(
        _table(species, oxidants, source), ("background",))
    assert "no enabled data-store oxidant source with daily-mean" in reason


def test_a_disabled_source_leaves_the_prescribed_input_unserved():
    species, oxidants, source = _inputs()
    reason = chem_sulfur._oxidant_refusal(
        _table(species, oxidants, source), ())
    assert "allocated field would stay zero" in reason


def test_three_served_prescribed_roles_pass_the_input_contract():
    species, oxidants, source = _inputs()
    assert chem_sulfur._oxidant_refusal(
        _table(species, oxidants, source), ("background",)) is None
