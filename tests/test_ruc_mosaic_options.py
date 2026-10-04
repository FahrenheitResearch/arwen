"""The RUC switches must survive both config doors without changing defaults."""
from dataclasses import replace
import re

import pytest

from gpuwm.config import RunConfig, validate_run_config
from test_namelist_import import _import_with, _load


def _config(**options):
    return RunConfig(nx=12, ny=12, nz=8, dx=3000., dy=3000.,
                     ztop=12000., dt=6., run_seconds=60., moist=True,
                     mp_physics=6,
                     sf_surface_physics=3, sf_sfclay_physics=91,
                     bl_pbl_physics=1, num_soil_layers=9, **options)


def _ruc_namelist(text):
    text = text.replace("&physics\n", "&physics\n num_soil_layers = 9,\n", 1)
    for key, value in (("sf_surface_physics", "3, 3"),
                       ("bl_pbl_physics", "1, 1")):
        text = re.sub(rf"{key}\s*=\s*[^\n]*", f"{key} = {value},", text)
    return text


@pytest.mark.parametrize("land,soil", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_ruc_switches_reach_every_domain(tmp_path, land, soil):
    text, _ = _import_with(
        tmp_path, inp_filter=_ruc_namelist,
        extra_physics=f"mosaic_lu = {land},\n mosaic_soil = {soil},\n")
    for domain in _load(tmp_path, text).domains:
        assert domain.run.mosaic_lu == land
        assert domain.run.mosaic_soil == soil
        validate_run_config(domain.run)


def test_explicit_off_preserves_existing_namelist_bytes(tmp_path):
    baseline, _ = _import_with(tmp_path, inp_filter=_ruc_namelist)
    off, _ = _import_with(tmp_path, inp_filter=_ruc_namelist,
                          extra_physics="mosaic_lu = 0,\n mosaic_soil = 0,\n")
    assert off == baseline
    assert _config().mosaic_lu == _config().mosaic_soil == 0


@pytest.mark.parametrize("key", ["mosaic_lu", "mosaic_soil"])
@pytest.mark.parametrize("value", [-1, 2, True, 1.0])
def test_noninteger_or_nonbinary_selector_is_refused(key, value):
    with pytest.raises(ValueError, match=key):
        validate_run_config(_config(**{key: value}))


@pytest.mark.parametrize("key", ["mosaic_lu", "mosaic_soil"])
def test_noah_cannot_silently_ignore_ruc_switch(key):
    config = replace(_config(**{key: 1}), sf_surface_physics=2,
                     num_soil_layers=4)
    with pytest.raises(ValueError, match="only RUC reads"):
        validate_run_config(config)
