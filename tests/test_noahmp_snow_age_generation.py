"""Night aging must run before the solar branch, as in WRF v4.7.0."""
import csv
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.noahmp_glacier import _radiation_glacier, _snow_age_glacier
from gpuwm.core.noahmp_radiation import albedo, snow_age


def land_input():
    path = Path(__file__).parent / "data/oracles/noahmp/noahmp-radiation-albedo.csv"
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    def f(name):
        return np.array(int(row[name], 16), dtype=np.uint32).view(np.float32)[()]
    def vec(name):
        return (f(name + "1"), f(name + "2"))
    names = ("tau0 grain_growth extra_growth dirt_soot swemx xl betads betais "
             "dt cosz fage elai esai tg tv snowh fsno fwet sneqvo sneqv qsnow "
             "fveg albold tauss").split()
    kw = {name: f(name + "_in") if name in ("fage", "albold", "tauss")
          else f(name) for name in names}
    kw.update({name: vec(name) for name in
               "albsat albdry alblak rhol rhos taul taus omegas".split()})
    kw.update(vegtyp=int(row["vegtyp"]), ist=int(row["ist"]), ice=int(row["ice"]),
              nsoil=4, smc=[f("smc" + str(i)) for i in range(1, 5)])
    return kw


@pytest.mark.parametrize("cosz", [-0.3, -0.0, 0.0])
def test_default_land_ages_at_night(cosz):
    kw = land_input()
    kw.update(cosz=np.float32(cosz), sneqv=np.float32(50),
              sneqvo=np.float32(50), tauss=np.float32(0.8))
    want = snow_age(*(kw[k] for k in
        "tau0 grain_growth extra_growth dirt_soot swemx dt tg sneqvo sneqv tauss".split()))
    got = albedo(**kw)
    assert np.float32(got["tauss"]).view(np.uint32) == want[0].view(np.uint32)
    assert np.float32(got["fage"]).view(np.uint32) == want[1].view(np.uint32)
    assert all(value == 0 for key in ("albd", "albi", "fabd", "fabi", "ftdd", "ftid", "ftii")
               for value in got[key])


def test_default_glacier_ages_at_night():
    args = (np.float32(300), np.float32(265), np.float32(50),
            np.float32(50))
    want, _ = _snow_age_glacier(*args, np.float32(0.8))
    got = _radiation_glacier(*args, np.float32(-0.3), np.float32(0),
                             (0, 0), (0, 0), np.float32(0.7), np.float32(0.8))
    assert got[1].view(np.uint32) == want.view(np.uint32)
    assert got[2:] == (0, 0, 0)


@pytest.mark.parametrize("snow,previous", [(0, 0), (50, 50), (50, 49), (800, 800), (801, 801)])
def test_legacy_glacier_holds_night_state(snow, previous):
    got = _radiation_glacier(300, 265, previous, snow, 0, 0,
                             (0, 0), (0, 0), np.float32(0.7),
                             np.float32(0.8), "wrf_461")
    assert got[0].view(np.uint32) == np.float32(0.7).view(np.uint32)
    assert got[1].view(np.uint32) == np.float32(0.8).view(np.uint32)


def test_generation_is_a_registered_default_and_roundtrips_comment():
    from gpuwm.config import RunConfig
    from gpuwm.physics_registry import physics_registry
    from gpuwm.physics_source_defaults import (read_physics_selector_comment,
                                               with_physics_selector_comment)
    assert RunConfig.__dataclass_fields__["noahmp_snow_age"].default == "wrf_470"
    row = physics_registry()["parameters"]["noahmp_snow_age"]
    assert row["default"] == "wrf_470"
    assert row["enum"] == ["wrf_470", "wrf_461"]
    for value in row["enum"]:
        settings = {"noahmp_snow_age": value}
        text = with_physics_selector_comment("&physics\n/\n", settings)
        assert read_physics_selector_comment(text) == settings


def test_restart_refuses_generation_flip_and_old_header_under_new_default(tmp_path):
    from dataclasses import asdict
    from gpuwm.config import RunConfig
    from gpuwm.io.restart import _require_config_match, RestartMismatchError
    kw = dict(nx=6, ny=4, nz=8, dx=1000, dy=1000,
              ztop=10000, dt=10, run_seconds=60, sf_surface_physics=4)
    cfg = RunConfig(**kw)
    stored = asdict(cfg)
    _require_config_match(stored, cfg, tmp_path / "restart")
    legacy = RunConfig(**kw, noahmp_snow_age="wrf_461")
    with pytest.raises(RestartMismatchError, match="noahmp_snow_age"):
        _require_config_match(stored, legacy, tmp_path / "restart")
    old = asdict(legacy)
    old.pop("noahmp_snow_age")
    _require_config_match(old, legacy, tmp_path / "restart")
    with pytest.raises(RestartMismatchError, match="noahmp_snow_age"):
        _require_config_match(old, cfg, tmp_path / "restart")


def test_unknown_generation_refuses_concrete_numerical_substitution():
    from gpuwm.noahmp_generation import snow_age_generation, radiation_kernel
    with pytest.raises(ValueError, match="TAUSS"):
        snow_age_generation("latest")
    assert radiation_kernel("wrf_461") == "noahmp_radiation_batch"
    assert radiation_kernel("wrf_470") == "noahmp_radiation_batch_night_age"


def test_cold_start_carries_the_two_computed_temperature_diagnostics():
    from gpuwm.core.noahmp_runtime import COLD_START_WRITES
    assert {"t2mvxy", "t2mbxy"}.issubset(COLD_START_WRITES)


@pytest.mark.parametrize("generation", [None, "wrf_461", "wrf_470"])
def test_real_namelist_import_reaches_generation(tmp_path, generation):
    from test_namelist_import import _pair, INPUT_TEXT
    from gpuwm.namelist_import import import_namelists
    from gpuwm.experiment import load_experiment
    from gpuwm.physics_source_defaults import with_physics_selector_comment
    text = INPUT_TEXT.replace("sf_surface_physics = 2, 2", "sf_surface_physics = 4, 4")
    text = text.replace("sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 5, 5")
    text = text.replace("bl_pbl_physics = 11, 11", "bl_pbl_physics = 5, 5")
    text = text.replace("&physics\n", "&physics\n num_soil_layers = 4,\n")
    if generation is not None:
        text = with_physics_selector_comment(text, {"noahmp_snow_age": generation})
    wps, inp = _pair(tmp_path, inp=text)
    output, _report = import_namelists(wps, inp)
    path = tmp_path / "experiment.toml"
    path.write_text(output)
    experiment = load_experiment(path)
    assert all(domain.run.noahmp_snow_age == (generation or "wrf_470")
               for domain in experiment.domains)
