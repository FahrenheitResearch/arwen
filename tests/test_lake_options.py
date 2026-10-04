"""WRF lake selectors retain defaults, domain scope and restart identity."""
from dataclasses import asdict, replace

import pytest

from gpuwm.config import validate_run_config
from gpuwm.io.restart import (RestartMismatchError, _require_config_match,
                               configuration_echo)
from test_namelist_import import _import_with, _load
from test_ruc_mosaic_options import _config, _ruc_namelist


def test_lake_default_off_preserves_older_checkpoint_config():
    cfg = _config()
    old = asdict(cfg)
    for name in ("sf_lake_physics", "use_lakedepth", "lakedepth_default", "lake_min_elev"):
        old.pop(name)
    _require_config_match(old, cfg, "older-checkpoint")
    assert configuration_echo(cfg) == configuration_echo(
        replace(cfg, use_lakedepth=0, lakedepth_default=20.0))
    with pytest.raises(RestartMismatchError, match="sf_lake_physics"):
        _require_config_match(old, replace(cfg, sf_lake_physics=1), "older-checkpoint")


def test_lake_active_depth_changes_are_bound_to_restart():
    cfg = _config(sf_lake_physics=1)
    for changes in ({"use_lakedepth": 0}, {"lakedepth_default": 20.0},
                    {"lake_min_elev": 10.0}):
        with pytest.raises(RestartMismatchError, match=next(iter(changes))):
            _require_config_match(configuration_echo(cfg), replace(cfg, **changes),
                                  "lake-checkpoint")


def test_lake_namelist_scope_and_missing_tail(tmp_path):
    text, _ = _import_with(
        tmp_path, inp_filter=_ruc_namelist,
        extra_physics="sf_lake_physics = 1,\n use_lakedepth = 0, 1,\n"
                      " lakedepth_default = 20., 80.,\n")
    root, child = _load(tmp_path, text).domains
    assert (root.run.sf_lake_physics, child.run.sf_lake_physics) == (1, 0)
    assert (root.run.use_lakedepth, child.run.use_lakedepth) == (0, 1)
    assert (root.run.lakedepth_default, child.run.lakedepth_default) == (20., 80.)


def test_explicit_lake_defaults_do_not_change_imported_bytes(tmp_path):
    baseline, _ = _import_with(tmp_path, inp_filter=_ruc_namelist)
    defaults, _ = _import_with(
        tmp_path, inp_filter=_ruc_namelist,
        extra_physics="sf_lake_physics = 0, 0,\n use_lakedepth = 1, 1,\n"
                      " lakedepth_default = 50., 50.,\n")
    assert defaults == baseline


@pytest.mark.parametrize("name,value", [
    ("sf_lake_physics", True), ("sf_lake_physics", 2),
    ("use_lakedepth", True), ("use_lakedepth", -1),
    ("lakedepth_default", True), ("lakedepth_default", float("inf")),
    ("lakedepth_default", float("nan")), ("lakedepth_default", "50"),
])
def test_invalid_lake_options_are_refused(name, value):
    with pytest.raises(ValueError, match=name):
        validate_run_config(_config(**{name: value}))


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_nonpositive_default_depth_selects_wrfs_reference_geometry(value):
    validate_run_config(_config(sf_lake_physics=1, use_lakedepth=0,
                                lakedepth_default=value))


@pytest.mark.parametrize("kind", ["relocation", "follow", "spawn"])
def test_rebuilt_lake_nest_cannot_discard_thermal_storage(kind):
    from types import SimpleNamespace
    from gpuwm.experiment import _refuse_rebuilt_nest_lake
    child = SimpleNamespace(grid_id=2, run=_config(sf_lake_physics=1),
                            follow=object() if kind == "follow" else None,
                            spawn=object() if kind == "spawn" else None)
    relocation = (SimpleNamespace(enabled=True, grid_id=2, moves=(object(),), follow=None)
                  if kind == "relocation" else None)
    with pytest.raises(ValueError, match="does not carry lake water.*heat storage"):
        _refuse_rebuilt_nest_lake([child], relocation, "lake.toml")
    child.run = _config()
    _refuse_rebuilt_nest_lake([child], relocation, "off.toml")
    child.run = _config(sf_lake_physics=1)
    child.follow = child.spawn = None
    _refuse_rebuilt_nest_lake([child], None, "fixed.toml")


def test_toml_door_refuses_lake_storage_reset_at_relocation(tmp_path):
    from test_noah_mosaic_doors import _RELOCATION_TABLES
    text, _ = _import_with(tmp_path, inp_filter=_ruc_namelist,
                          extra_physics="sf_lake_physics = 0, 1,\n")
    assert _load(tmp_path, text).domains[1].run.sf_lake_physics == 1
    with pytest.raises(ValueError, match="does not carry lake water"):
        _load(tmp_path, text + _RELOCATION_TABLES, name="moving-lake.toml")
