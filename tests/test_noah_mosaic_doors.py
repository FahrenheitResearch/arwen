"""Real namelist, wizard, output, registry and memory doors for Noah tiles."""
import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm.config import RunConfig
from gpuwm.core.noah_mosaic import mosaic_array_shapes
from gpuwm.core.noah_mosaic_door import attach_noah_mosaic_to_driver
from gpuwm.core.preflight import physics_array_shapes, domain_kernel_modules, UNMEASURED_KERNEL_MODULES
from gpuwm.domain_wizard import with_noah_mosaic_options, register_cli
from gpuwm.io.wrfout import wrf_physics_selector_attrs
from gpuwm.physics_registry import physics_registry
from test_namelist_import import _import_with, _load


def _cfg(**kw):
    return RunConfig(nx=12, ny=12, nz=8, dx=3000.0, dy=3000.0,
                     ztop=12000.0, dt=6.0, run_seconds=60.0, **kw)



def test_import_off_toml_is_byte_identical(tmp_path):
    baseline, _ = _import_with(tmp_path)
    off, _ = _import_with(tmp_path, extra_physics="sf_surface_mosaic = 0,\n mosaic_cat = 5,\n")
    assert off.encode() == baseline.encode()


def test_import_on_reads_run_wide_first_value(tmp_path):
    text, receipt = _import_with(tmp_path, extra_physics="sf_surface_mosaic = 1, 0,\n mosaic_cat = 5, 2,\n")
    assert "sf_surface_mosaic = 1" in text and "mosaic_cat = 5" in text
    for dc in _load(tmp_path, text).domains:
        assert (dc.run.sf_surface_mosaic, dc.run.mosaic_cat) == (1, 5)
    fixed = {row.key: row for row in receipt.fixed}
    assert "first value" in fixed["mosaic_cat"].reason


def test_import_mosaic_with_urban_canopy_loads_and_bep_is_refused_as_in_wrf(tmp_path):
    # The WRF namelist door: mosaic plus the single-layer UCM (option 1)
    # reaches every domain; options 2 and 3 stop with WRF's own reason
    # (module_check_a_mundo.F:505-518), under a PBL BEP itself accepts.
    import re
    text, _ = _import_with(
        tmp_path, extra_physics="sf_surface_mosaic = 1,\n sf_urban_physics = 1,\n")
    for dc in _load(tmp_path, text).domains:
        assert (dc.run.sf_surface_mosaic, dc.run.sf_urban_physics) == (1, 1)

    def ysu(inp):
        return re.sub(r"bl_pbl_physics\s*=\s*[^\n]*", "bl_pbl_physics = 1, 1,", inp)
    for urban in (2, 3):
        with pytest.raises(ValueError, match="mosaic option cannot work with urban options 2 and 3"):
            text, _ = _import_with(
                tmp_path, inp_filter=ysu,
                extra_physics=f"sf_surface_mosaic = 1,\n sf_urban_physics = {urban},\n")
            _load(tmp_path, text)


@pytest.mark.parametrize("line,message", [
    ("sf_surface_mosaic = 2,", "land surface would not be integrated"),
    ("mosaic_lu = 1,", "mosaic land/soil physics is not implemented"),
    ("mosaic_soil = 1,", "mosaic land/soil physics is not implemented"),
    ("sf_surface_mosaic = 1,\n mosaic_cat = 0,", "Noah has no tile")])
def test_import_refusals(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        _import_with(tmp_path, extra_physics=line)


def test_wizard_flags_and_emitted_config(tmp_path):
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["domain", "--point", "40,10", "--cycle", "1999-05-03T12", "--out", "area.toml", "--sf-surface-mosaic", "1", "--mosaic-cat", "5"])
    assert (args.sf_surface_mosaic, args.mosaic_cat) == (1, 5)
    baseline, _ = _import_with(tmp_path)
    assert with_noah_mosaic_options(baseline) == baseline
    text = with_noah_mosaic_options(baseline, args.sf_surface_mosaic, args.mosaic_cat)
    assert all(dc.run.mosaic_cat == 5 for dc in _load(tmp_path, text).domains)
    with pytest.raises(ValueError, match="silently ignores"):
        with_noah_mosaic_options(baseline.replace("sf_surface_physics = 2", "sf_surface_physics = 0"), 1, 3)


def test_output_and_registry():
    cfg = _cfg(sf_surface_mosaic=1, sf_surface_physics=2)
    assert wrf_physics_selector_attrs(cfg)["SF_SURFACE_MOSAIC"] == 1
    specs = physics_registry()["parameters"]
    assert specs["sf_surface_mosaic"].get("implemented", True) is True
    assert specs["sf_surface_mosaic"]["enum"] == [0, 1]
    assert specs["mosaic_cat"]["minimum"] == 1


def test_fit_gate_prices_every_tile_and_selects_kernel():
    cfg = _cfg(sf_surface_mosaic=1, sf_surface_physics=2, bl_pbl_physics=1, sf_sfclay_physics=1, mosaic_cat=5)
    shapes = physics_array_shapes(cfg)
    for name, (shape, dtype) in mosaic_array_shapes(5, cfg.ny, cfg.nx).items():
        assert shapes["fields/" + name] == shape
    off = physics_array_shapes(replace(cfg, sf_surface_mosaic=0))
    assert not any("mosaic" in name or "landusef2" in name for name in off)
    modules = domain_kernel_modules(SimpleNamespace(run=cfg, grid_id=1), prices_refl=False)
    assert "noah_mosaic_unit" in modules and "noah" not in modules
    assert "noah_mosaic" in UNMEASURED_KERNEL_MODULES
    # Priced from the measured unit, never at the assumed bound.
    from gpuwm.core.preflight import (CHAINED_TRANSLATION_UNIT_FRAMES,
                                      assumed_bound_modules)
    assert CHAINED_TRANSLATION_UNIT_FRAMES["noah_mosaic_unit"].covers == {"noah_mosaic"}
    assert not assumed_bound_modules(modules) & {"noah_mosaic", "noah_mosaic_unit"}


def test_door_off_never_touches_inputs():
    attach_noah_mosaic_to_driver(None, _cfg(), landusef=None, processed=False,
                                 landuse_attrs=None, fractional_seaice=False)


def test_ucm_tile_memory_and_composed_unit_are_priced():
    cfg = _cfg(sf_surface_mosaic=1, sf_surface_physics=2, sf_urban_physics=1)
    shapes = physics_array_shapes(cfg)
    urban = mosaic_array_shapes(3, cfg.ny, cfg.nx, urban=True)
    plain = mosaic_array_shapes(3, cfg.ny, cfg.nx)
    assert len(urban) - len(plain) == 15
    for name, (shape, dtype) in urban.items():
        assert shapes["fields/" + name] == shape
        assert dtype in ("float32", "int32")
    assert urban["trl_urb3d_mosaic"][0] == (12, cfg.ny, cfg.nx)
    modules = domain_kernel_modules(SimpleNamespace(run=cfg, grid_id=1), prices_refl=False)
    assert "noah_mosaic_ucm_unit" in modules
    assert "noah_mosaic_unit" not in modules and "noah" not in modules


def test_missing_fractions_refuses_dominant_column():
    with pytest.raises(ValueError, match="without fractions.*dominant category"):
        attach_noah_mosaic_to_driver(
            None, _cfg(sf_surface_mosaic=1, sf_surface_physics=2),
            landusef=None, processed=True, landuse_attrs={},
            fractional_seaice=False)


@pytest.mark.parametrize("processed", [False, True])
def test_helper_passes_final_fields_and_processes_raw_only(monkeypatch, processed):
    import numpy as np
    from gpuwm.core import noah_mosaic
    from gpuwm.core.noah_mosaic import MosaicCategories
    calls = []
    categories = MosaicCategories(13, 15, 17, 21, ())
    fractions = np.zeros((21, 2, 2), np.float32)
    edited = fractions + 1
    final_fields = {"xice": np.zeros((2, 2), np.float32), "tsk": np.ones((2, 2), np.float32)}
    driver = SimpleNamespace(fields=final_fields, noah_params=SimpleNamespace(lucats=21), noah_mosaic=None)
    def categories_read(dataset, **kw):
        calls.append(("categories", dataset, kw))
        return categories
    def real_edits(value, **kw):
        calls.append(("real", value, kw))
        return edited
    def attach(fields, **kw):
        calls.append(("attach", fields, kw))
    monkeypatch.setattr(noah_mosaic, "load_mosaic_categories", categories_read)
    monkeypatch.setattr(noah_mosaic, "real_exe_landusef", real_edits)
    monkeypatch.setattr(noah_mosaic, "attach_noah_mosaic", attach)
    attach_noah_mosaic_to_driver(
        driver, _cfg(sf_surface_mosaic=1, sf_surface_physics=2, mosaic_cat=5),
        landusef=fractions, processed=processed,
        landuse_attrs={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISURBAN": 13,
                       "ISICE": 15, "ISWATER": 17, "ISLAKE": 21},
        landmask=np.ones((2, 2)), fractional_seaice=False)
    assert [row[0] for row in calls] == (["categories", "attach"] if processed else ["categories", "real", "attach"])
    assert calls[-1][1] is final_fields
    assert calls[-1][2]["landusef"] is (fractions if processed else edited)
    assert calls[-1][2]["mosaic_cat"] == 5
    assert calls[-1][2]["fractional_seaice"] is False
    assert driver.noah_mosaic.categories is categories
    assert driver.noah_mosaic.mosaic_cat == 5
    # One sea-ice split for the tiles and the tile loop (WRF derives both
    # from fractional_seaice, module_sf_noahdrv.F:5028-5032).
    assert driver.noah_mosaic.xice_threshold == 0.5


def test_domain_cli_writes_mosaic_to_every_grid(tmp_path):
    from test_domain_wizard import _run_wizard
    from gpuwm.experiment import load_experiment
    rc, path = _run_wizard(tmp_path, "--sf-surface-mosaic", "1", "--mosaic-cat", "5", point="40,10", card="24gb")
    assert rc == 0
    assert all((dc.run.sf_surface_mosaic, dc.run.mosaic_cat) == (1, 5)
               for dc in load_experiment(path).domains)


@pytest.mark.parametrize("lsm,refused", [("noah", False), ("off", True), ("noah-mp", True)])
def test_registry_and_run_door_share_the_noah_pairing(lsm, refused):
    from test_authority_agreement import _permissive_registry, _base_template_id, _single_domain_plan, _PERMISSIVE_RUNNER
    from gpuwm.physics_registry import validate_physics_plan
    registry = _permissive_registry()
    plan = _single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source", _base_template_id(registry),
        components={"land_surface": lsm},
        parameters={"sf_surface_mosaic": 1, "mosaic_cat": 5})
    report = validate_physics_plan(plan, registry=registry)
    pairing = [issue for issue in report["errors"] if issue["code"] == "noah-mosaic-pairing"]
    assert bool(pairing) is refused
    if refused:
        assert "silently ignores" in pairing[0]["message"]


def test_import_off_list_reads_first_and_ignores_tile_count(tmp_path):
    baseline, _ = _import_with(tmp_path)
    text, receipt = _import_with(tmp_path, extra_physics="sf_surface_mosaic = 0, 1,\n mosaic_cat = 0, 5,\n")
    assert text == baseline
    fixed = {row.key: row for row in receipt.fixed}
    assert "first value" in fixed["sf_surface_mosaic"].reason
    assert "unread when mosaic is off" in fixed["mosaic_cat"].reason


# ---------------------------------------------------------------------------
# mosaic_urban_canopy through every door: WRF's rule unless a door names the
# town rule, and a namelist without the key imports byte for byte as before
# ---------------------------------------------------------------------------

_URBAN_MOSAIC = "sf_surface_mosaic = 1,\n sf_urban_physics = 1,\n"


def test_import_without_the_canopy_key_keeps_wrfs_rule(tmp_path):
    text, _ = _import_with(tmp_path, extra_physics=_URBAN_MOSAIC)
    assert "mosaic_urban_canopy" not in text
    assert all(dc.run.mosaic_urban_canopy == "dominant"
               for dc in _load(tmp_path, text).domains)
    named, receipt = _import_with(
        tmp_path, extra_physics=_URBAN_MOSAIC
        + " mosaic_urban_canopy = 'dominant', 'dominant',\n")
    assert named == text
    assert "WRF v4.7.1's rule" in {r.key: r for r in receipt.fixed}["mosaic_urban_canopy"].reason


def test_import_reads_the_canopy_rule_per_domain(tmp_path):
    text, _ = _import_with(
        tmp_path, extra_physics=_URBAN_MOSAIC
        + " mosaic_urban_canopy = 'dominant', 'every_tile',\n")
    rules = [dc.run.mosaic_urban_canopy for dc in _load(tmp_path, text).domains]
    assert rules == ["dominant", "every_tile"]
    text, _ = _import_with(
        tmp_path, extra_physics=_URBAN_MOSAIC + " mosaic_urban_canopy = 'every_tile',\n")
    # An omitted tail keeps WRF's rule, as a Registry default would.
    assert [dc.run.mosaic_urban_canopy for dc in _load(tmp_path, text).domains] == [
        "every_tile", "dominant"]


@pytest.mark.parametrize("physics,message", [
    (" mosaic_urban_canopy = 'town',\n", "must name a rule on every domain"),
    (" mosaic_urban_canopy = 1,\n", "must name a rule on every domain"),
    ("sf_surface_mosaic = 1,\n mosaic_urban_canopy = 'every_tile',\n",
     "needs sf_urban_physics = 1"),
    ("sf_urban_physics = 1,\n mosaic_urban_canopy = 'every_tile',\n",
     "needs sf_surface_mosaic = 1")])
def test_import_refuses_the_town_rule_where_it_cannot_run(tmp_path, physics, message):
    with pytest.raises(ValueError, match=message):
        _import_with(tmp_path, extra_physics=physics)


def test_experiment_toml_takes_the_rule_per_domain(tmp_path):
    text, _ = _import_with(tmp_path, extra_physics=_URBAN_MOSAIC)
    lines = text.splitlines(keepends=True)
    second = [i for i, line in enumerate(lines) if line.strip() == "[[domain]]"][1]
    lines.insert(second + 1, 'mosaic_urban_canopy = "every_tile"\n')
    domains = _load(tmp_path, "".join(lines)).domains
    assert [dc.run.mosaic_urban_canopy for dc in domains] == ["dominant", "every_tile"]
    shared = text.replace("[shared]\n", '[shared]\nmosaic_urban_canopy = "every_tile"\n', 1)
    assert all(dc.run.mosaic_urban_canopy == "every_tile"
               for dc in _load(tmp_path, shared).domains)
    with pytest.raises(ValueError, match="mosaic_urban_canopy must be one of"):
        _load(tmp_path, text.replace("[shared]\n", '[shared]\nmosaic_urban_canopy = "town"\n', 1))


def test_wizard_writes_the_canopy_rule(tmp_path):
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["domain", "--point", "40,10", "--cycle", "1999-05-03T12",
                              "--out", "area.toml", "--mosaic-urban-canopy", "every_tile"])
    assert args.mosaic_urban_canopy == "every_tile"
    with pytest.raises(SystemExit):
        parser.parse_args(["domain", "--point", "40,10", "--cycle", "1999-05-03T12",
                           "--out", "area.toml", "--mosaic-urban-canopy", "town"])
    baseline, _ = _import_with(tmp_path, extra_physics=_URBAN_MOSAIC)
    text = with_noah_mosaic_options(baseline, None, None, "every_tile")
    assert all(dc.run.mosaic_urban_canopy == "every_tile"
               for dc in _load(tmp_path, text).domains)
    plain, _ = _import_with(tmp_path)
    with pytest.raises(ValueError, match="needs sf_surface_mosaic = 1"):
        with_noah_mosaic_options(plain, None, None, "every_tile")


def test_registry_row_and_plan_pairing_for_the_canopy_rule():
    spec = physics_registry()["parameters"]["mosaic_urban_canopy"]
    assert spec["enum"] == ["dominant", "every_tile"]
    assert spec["default"] == "dominant" and spec["per_domain"] is True
    assert spec["consuming_read"] == "gpuwm/core/noah_mosaic_door.py"
    from test_authority_agreement import _permissive_registry, _base_template_id, _single_domain_plan, _PERMISSIVE_RUNNER
    from gpuwm.physics_registry import validate_physics_plan
    registry = _permissive_registry()
    plan = _single_domain_plan(
        registry, _PERMISSIVE_RUNNER, "any-source", _base_template_id(registry),
        components={"land_surface": "noah"},
        parameters={"sf_surface_mosaic": 1, "mosaic_urban_canopy": "every_tile"})
    report = validate_physics_plan(plan, registry=registry)
    pairing = [issue for issue in report["errors"] if issue["code"] == "noah-mosaic-pairing"]
    assert pairing and "needs sf_urban_physics = 1" in pairing[0]["message"]


# ---------------------------------------------------------------------------
# What Noah mosaic cannot run is refused by name before anything is prepared:
# a nest whose physics a move or a spawn rebuilds, and a streamed domain.
# Both rebuild a physics driver without a door, and only the doors build the
# land-use tiles (gpuwm/core/noah_mosaic_door.py).
# ---------------------------------------------------------------------------

_RELOCATION_TABLES = """
[relocation]
enabled = true
grid_id = 2
max_move_parent_cells = 4
min_overlap_fraction = 0.5
cadence_seconds = 3600.0

[relocation.follow]
field = "uh"
threshold = 25.0
fallback_threshold = 40.0
search_margin_cells = 15
min_shift_cells = 2
max_shift_cells = 4
cooldown_seconds = 3600.0
"""


def _tree(*, mosaic, follow=None, spawn=None, relocation=None):
    run = SimpleNamespace(sf_surface_mosaic=mosaic)
    return ([SimpleNamespace(grid_id=1, run=run, follow=None, spawn=None),
             SimpleNamespace(grid_id=2, run=run, follow=follow, spawn=spawn)],
            relocation)


@pytest.mark.parametrize("kind,words", [
    ("relocation", "grid_id = 2 of t.toml moves and runs Noah mosaic"),
    ("follow", "grid_id = 2 of t.toml moves and runs Noah mosaic"),
    ("spawn", "grid_id = 2 of t.toml is spawned mid-run and runs Noah mosaic")])
def test_a_nest_rebuilt_mid_run_refuses_mosaic_by_name(kind, words):
    from gpuwm.experiment import _refuse_rebuilt_nest_noah_mosaic
    relocation = SimpleNamespace(enabled=True, grid_id=2, moves=(), follow=object())
    domains, relocation = _tree(
        mosaic=1, follow=object() if kind == "follow" else None,
        spawn=object() if kind == "spawn" else None,
        relocation=relocation if kind == "relocation" else None)
    with pytest.raises(ValueError, match=words + r".*interp_mask_land_field.*sf_surface_mosaic = 0"):
        _refuse_rebuilt_nest_noah_mosaic(domains, relocation, "t.toml")
    # Mosaic off, or the same nest held still, is not this refusal's business.
    off, relocation_off = _tree(
        mosaic=0, follow=object() if kind == "follow" else None,
        spawn=object() if kind == "spawn" else None,
        relocation=relocation if kind == "relocation" else None)
    _refuse_rebuilt_nest_noah_mosaic(off, relocation_off, "t.toml")
    still, _ = _tree(mosaic=1)
    _refuse_rebuilt_nest_noah_mosaic(
        still, SimpleNamespace(enabled=False, grid_id=2, moves=(), follow=None),
        "t.toml")


def test_the_toml_door_refuses_a_moving_mosaic_nest(tmp_path):
    text, _ = _import_with(tmp_path, extra_physics="sf_surface_mosaic = 1,\n")
    assert all(dc.run.sf_surface_mosaic == 1 for dc in _load(tmp_path, text).domains)
    with pytest.raises(ValueError, match=r"grid_id = 2 of .* moves and runs Noah mosaic"):
        _load(tmp_path, text + _RELOCATION_TABLES, name="moving.toml")
    # The same moving tree without mosaic loads: the refusal is mosaic's.
    off, _ = _import_with(tmp_path)
    moving = _load(tmp_path, off + _RELOCATION_TABLES, name="moving-off.toml")
    assert moving.relocation.enabled and int(moving.relocation.grid_id) == 2


def test_a_streamed_domain_refuses_mosaic_by_name():
    from gpuwm.core.streaming import StreamingRefused, attach
    cfg = SimpleNamespace(grid_id=3, slope_rad=0, sf_surface_mosaic=1)
    with pytest.raises(StreamingRefused,
                       match=r"grid_id = 3 sets sf_surface_mosaic = 1.*cannot "
                             r"run streamed.*mode = 'off'.*sf_surface_mosaic = 0"):
        attach(None, cfg, SimpleNamespace(stream=True), tile_state_factory=None)
    # Mosaic off reaches attach's own checks, past this refusal.
    off = SimpleNamespace(grid_id=3, slope_rad=0, sf_surface_mosaic=0)
    with pytest.raises(StreamingRefused, match="does not stream"):
        attach(None, off, SimpleNamespace(stream=False), tile_state_factory=None)


# ---------------------------------------------------------------------------
# The offline child (gpuwm downscale) builds its child's land surface from a
# child-grid surface file or the parent's history, with no door that builds
# the tiles: a mosaic child used to be accepted, interpolate the parent, and
# stop on its first land-surface step.  Refused at the review and at the run.
# ---------------------------------------------------------------------------


def test_the_offline_child_names_why_it_cannot_run_mosaic():
    from gpuwm.offline_child import child_mosaic_refusal
    words = child_mosaic_refusal(SimpleNamespace(sf_surface_mosaic=1))
    assert "sf_surface_mosaic = 1 (Noah mosaic)" in words
    assert "--child-config with sf_surface_mosaic = 0" in words
    assert child_mosaic_refusal(SimpleNamespace(sf_surface_mosaic=0)) is None
    assert child_mosaic_refusal(SimpleNamespace()) is None


@pytest.mark.parametrize("door", ["review", "run"])
def test_a_mosaic_child_is_refused_before_the_parent_is_interpolated(
        tmp_path, capsys, monkeypatch, door):
    """The deck's fitting --point child, its derived config given mosaic:
    the review (and the runner door, which no review stands in front of)
    refuse it by name, and nothing is interpolated or started."""
    import gpuwm.downscale as downscale
    import gpuwm.offline_child as offline_child
    import gpuwm.offline_child_run as offline_child_run
    from gpuwm.cli import main as cli_main
    from test_downscale_checkpoint_disk import _fitting_point_args
    from test_downscale_cli import _a_box_that_can_draw  # noqa: F401

    original = offline_child.resolve_child_run_config

    def with_mosaic(*args, **kwargs):
        cfg = original(*args, **kwargs)
        assert cfg.sf_surface_physics == 2
        return replace(cfg, sf_surface_mosaic=1)

    if door == "review":
        monkeypatch.setattr(downscale, "resolve_child_run_config", with_mosaic)
    else:
        monkeypatch.setattr(offline_child_run, "resolve_child_run_config",
                            with_mosaic)
    monkeypatch.setattr(offline_child_run, "interpolate_parent_initial_state",
                        lambda *a, **k: pytest.fail("the parent was interpolated"))
    started = []
    if door == "review":
        monkeypatch.setattr(offline_child_run, "run",
                            lambda namespace: started.append(namespace)
                            or {"result": "PASS"})
    rc = cli_main(_fitting_point_args(tmp_path))
    err = capsys.readouterr().err
    assert rc == 2 and started == []
    assert "sf_surface_mosaic = 1 (Noah mosaic)" in err, err[-2000:]
