"""The operational fork's MYNN generation rides the source's generation row.

``bl_mynn_version = "gsd_41"`` (GSD MYNN v4.1 of NOAA-EMC/HRRR v4.1.21) is
declared by the HRRR and RAP rows of
``gpuwm/data/physics_sources/request-defaults.v1.toml`` under
``[request.load_generations]`` and filled at load, beside the surface-layer
and Thompson generations, wherever a configuration omits it and every domain
admits it.  Origin: the WOOF-HRRR door's carried experiment.toml and the HRRR
demo configurations named MYNN and the legacy RRTMG pair with no generation,
and ran WRF v4.6.1's MYNN against HRRR's own analysis.
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from gpuwm.physics_source_defaults import (
    GENERATION_SELECTORS, gsd41_admitted, omitted_generation_selectors,
    recipe_load_generations, recipe_physics_defaults)

ROOT = Path(__file__).resolve().parents[1]
FORK_SOURCES = ("hrrr", "hrrr-native", "hrrr-prs", "rap", "rap-native")
HRRR_SUITE = {"mp_physics": 28, "bl_pbl_physics": 5, "sf_sfclay_physics": 5,
              "ra_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy"}


@pytest.mark.parametrize("source", FORK_SOURCES)
def test_the_fork_rows_declare_the_gsd41_generation_for_the_load_fill(source):
    assert recipe_load_generations(source) == {"bl_mynn_version": "gsd_41"}
    # The authoring and import doors read the row's defaults, which do not
    # carry it: they state the generation where they mean it.
    assert "bl_mynn_version" not in recipe_physics_defaults(source)
    assert "bl_mynn_version" in GENERATION_SELECTORS
    assert omitted_generation_selectors(dict(HRRR_SUITE), [{}], source)[
        "bl_mynn_version"] == "gsd_41"


@pytest.mark.parametrize("source", ("gfs", "era5"))
def test_other_sources_declare_no_mynn_generation(source):
    assert recipe_load_generations(source) == {}
    assert "bl_mynn_version" not in omitted_generation_selectors(
        dict(HRRR_SUITE), [{}], source)


def test_a_written_generation_is_kept():
    for written in ("wrf_461", "gsd_41"):
        shared = {**HRRR_SUITE, "bl_mynn_version": written}
        assert "bl_mynn_version" not in omitted_generation_selectors(
            shared, [{}], "hrrr")


def test_one_inadmissible_domain_keeps_the_whole_tree_on_its_written_generation():
    # The fill writes [shared]; a nest on RRTMGP would refuse gsd_41.
    domains = [{}, {"ra_rrtmg_variant": "rte-rrtmgp"}]
    assert "bl_mynn_version" not in omitted_generation_selectors(
        dict(HRRR_SUITE), domains, "hrrr")
    assert omitted_generation_selectors(
        dict(HRRR_SUITE), [{}, {}], "hrrr")["bl_mynn_version"] == "gsd_41"


def _validator_admits(settings) -> bool:
    """Whether gpuwm.config accepts the generation on these settings, given
    that it accepts the same settings on the v4.6.1 MYNN."""
    from gpuwm.config import RunConfig, validate_run_config
    grid = {"nx": 40, "ny": 40, "nz": 40, "dx": 3000.0, "dy": 3000.0,
            "ztop": 20000.0, "dt": 15.0, "run_seconds": 3600.0,
            "moist": True}
    base = RunConfig(**grid, **settings)
    validate_run_config(base)
    try:
        validate_run_config(replace(base, bl_mynn_version="gsd_41"))
    except ValueError as error:
        assert "gsd_41" in str(error)
        return False
    return True


_MATRIX = list(itertools.product(
    (1, 5),                                    # bl_pbl_physics
    ((4, "rrtmg_legacy"), (4, "rte-rrtmgp"), (0, "rrtmg_legacy")),
    (0, 1),                                    # spp_pbl
    (0, 1),                                    # bl_mynn_mixscalars
    (8, 28),                                   # mp_physics
))


@pytest.mark.parametrize("pbl,radiation,spp,mixscalars,mp", _MATRIX)
def test_admission_is_the_validators_own_answer(pbl, radiation, spp,
                                                 mixscalars, mp):
    """gsd41_admitted restates validate_run_config's gsd_41 refusals; this
    binds the two so a new refusal cannot make a door fill its own refusal."""
    ra, variant = radiation
    settings = {"bl_pbl_physics": pbl, "sf_sfclay_physics": 5 if pbl == 5 else 1,
                "ra_physics": ra, "ra_rrtmg_variant": variant,
                "spp_pbl": spp, "bl_mynn_mixscalars": mixscalars,
                "mp_physics": mp}
    try:
        expected = _validator_admits(settings)
    except (ValueError, NotImplementedError):
        pytest.skip("the base settings are themselves refused")
    if pbl != 5:
        # Not read outside MYNN: admitted by the validator, never filled.
        assert gsd41_admitted(settings) is False
    else:
        assert gsd41_admitted(settings) is expected


@pytest.mark.parametrize("name", ("hrrr_native_3km_demo", "hrrr_native_quick_demo",
                                  "hrrr_prs_3km_demo", "hrrr_prs_demo",
                                  "hrrr_v4_vertical_order5"))
def test_the_shipped_hrrr_demos_run_the_fork_mynn(name):
    from gpuwm.experiment import load_experiment
    path = ROOT / "configs" / f"{name}.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    # A demo with a [fetch] source takes the generation from its row; the
    # one without states it.
    assert ("bl_mynn_version" in raw.get("shared", {})) == ("fetch" not in raw)
    exp = load_experiment(path)
    for domain in exp.domains:
        assert domain.run.bl_pbl_physics == 5
        assert domain.run.bl_mynn_version == "gsd_41", name
