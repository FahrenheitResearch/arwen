"""A191: every checkpoint writer echoes a run's configuration the same way.

The breakage this prevents: the resident writer dropped nine default-off
RunConfig keys from a checkpoint's ``config`` echo and the streamed writer
wrote ``dataclasses.asdict(cfg)`` whole, so two checkpoints of one state,
one per memory road, compared unequal though every array matched
(tests/test_streamed_relocation_rebuild_gpu.py's
test_live_bounded_move_keeps_owner_and_continues_exactly failed on the
2.8.1 bench cards).  The out-of-core store writer had the same raw echo.

Each case writes the same NumPy-backed state through the resident writer
(``gpuwm.io.restart.write_restart``), the streamed writer
(``tilestream.restart_stream.write_streamed_restart``) and the store writer
(``tilestream.checkpoint.write_store_restart``) and requires one echo from
all three, at the defaults and with each default-off field moved.  CPU only.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from gpuwm.io import restart
from test_restart import (
    _cfg, _fill_setup, _identity_bound_physics_state, _shim_driver_state)


#: Every default-off field restart._drop_default_off_run_keys drops at its
#: default, moved, as (overrides, the keys the echo must then carry).
MOVED = {
    "adaptive_nest_lattice": (dict(adaptive_nest_lattice=True),
                              {"adaptive_nest_lattice"}),
    "zadvect_implicit": (dict(zadvect_implicit=1), {"zadvect_implicit"}),
    "zadvect_implicit_variant": (
        dict(zadvect_implicit=1, zadvect_implicit_variant="wrf_legacy"),
        {"zadvect_implicit", "zadvect_implicit_variant"}),
    "w_crit_cfl": (dict(zadvect_implicit=1, w_crit_cfl=2.0),
                   {"zadvect_implicit", "w_crit_cfl"}),
    "slope_rad": (dict(slope_rad=1), {"slope_rad"}),
    "topo_shading": (dict(slope_rad=1, topo_shading=1, shadlen=20000.0),
                     {"slope_rad", "topo_shading", "shadlen"}),
    "topo_wind": (dict(bl_pbl_physics=1, sf_sfclay_physics=1, topo_wind=1),
                  {"topo_wind"}),
    "gwd_opt": (dict(gwd_opt=1), {"gwd_opt"}),
    "sf_surface_mosaic": (dict(sf_surface_physics=2, sf_surface_mosaic=1,
                               mosaic_cat=2),
                          {"sf_surface_mosaic", "mosaic_cat"}),
    "mosaic_urban_canopy": (dict(sf_surface_physics=2, sf_urban_physics=1,
                                 sf_surface_mosaic=1,
                                 mosaic_urban_canopy="every_tile"),
                            {"sf_surface_mosaic", "mosaic_cat",
                             "mosaic_urban_canopy"}),
    "diff_opt": (dict(diff_opt=1), {"diff_opt", "mix_full_fields"}),
    "mix_full_fields": (dict(diff_opt=1, mix_full_fields=False),
                        {"diff_opt", "mix_full_fields"}),
}

#: The keys an echo at every default leaves out.
DEFAULT_OFF = frozenset({
    "adaptive_nest_lattice", "zadvect_implicit", "w_crit_cfl", "slope_rad",
    "topo_shading", "shadlen", "topo_wind", "gwd_opt", "sf_surface_mosaic",
    "mosaic_cat", "mosaic_urban_canopy", "diff_opt", "mix_full_fields",
    "zadvect_implicit_variant",
})


def _echoes(cfg, monkeypatch, tmp_path) -> dict[str, dict]:
    """The ``config`` each of the three writers puts in its header."""
    from tilestream import checkpoint, physics_inventory, restart_stream

    state, driver = _shim_driver_state(cfg, monkeypatch)
    if cfg.sf_surface_physics == 2:
        driver.noah_params = _identity_bound_physics_state(
            cfg, monkeypatch)[1].noah_params
    if cfg.sf_surface_mosaic:
        # The tile identity a mosaic header binds, without the tiles: this
        # file reads the configuration echo, not the tile carriers
        # (tests/test_noah_mosaic_driver.py checkpoints those on a card).
        from gpuwm.core.noah_mosaic import MosaicCategories
        driver.noah_mosaic = SimpleNamespace(
            mosaic_cat=cfg.mosaic_cat, xice_threshold=0.5,
            categories=MosaicCategories(isurban=13, isice=15, iswater=17,
                                        natural=7, lcz=()))
    _fill_setup(state)
    paths = {"resident": restart.write_restart(
        tmp_path / "resident.npz", state, cfg)}
    # After the resident write, which seeds diff_opt = 1's thermal
    # reference on the state it checkpoints.
    store = {key: value.copy() for key, value in
             physics_inventory.carrier_manifest(state).items()}
    scalars = physics_inventory.carrier_scalars(state)
    paths |= {
        "streamed": restart_stream.write_streamed_restart(
            tmp_path / "streamed.npz", store, cfg, scalars=scalars,
            setup=restart_stream.capture_domain_setup(state),
            template_state=state, check_pinned=False).path,
        "store": checkpoint.write_store_restart(
            tmp_path / "store.npz", store, scalars,
            checkpoint.DomainSetup.capture(state, cfg), cfg),
    }
    return {road: restart.read_restart_header(path)["config"]
            for road, path in paths.items()}


def test_every_writer_echoes_the_defaults_alike(monkeypatch, tmp_path):
    cfg = _cfg()
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert echo == expected, road
    # At the defaults every default-off key is absent, as in a header
    # written before the field existed, and every other field is echoed.
    assert not DEFAULT_OFF & set(expected)
    assert set(expected) == {f.name for f in dataclasses.fields(cfg)} - DEFAULT_OFF


@pytest.mark.parametrize("field", sorted(MOVED))
def test_every_writer_echoes_a_moved_field_alike(monkeypatch, tmp_path, field):
    overrides, kept = MOVED[field]
    cfg = _cfg(**overrides)
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert echo == expected, road
    assert DEFAULT_OFF & set(expected) == kept
    for key in kept:
        assert expected[key] == getattr(cfg, key), key


def test_the_digest_drops_what_the_echo_drops():
    """The configuration digest reads the echo's drops, so a default-off
    field absent from one is absent from the other."""
    for overrides, kept in [({}, set())] + list(MOVED.values()):
        cfg = _cfg(**overrides)
        digest = restart._configuration_digest_values(dataclasses.asdict(cfg))
        assert DEFAULT_OFF & set(digest) == kept, overrides
