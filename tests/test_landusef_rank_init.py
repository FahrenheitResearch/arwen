"""Two-kilometre geogrid fractions must initialize every resident rank."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.devices import DeviceOptions
from gpuwm.core.ruc_mosaic import mosaic_fractions
from gpuwm.core.streaming import ranked_halo, ranked_specs
from gpuwm.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.static.lambert import LambertGrid


ATTRS = dict(MMINLU="MODIFIED_IGBP_MODIS_NOAH", ISWATER=17,
             ISLAKE=21, ISICE=15, ISURBAN=13)


def _write_source(directory, data, **extra):
    """Write one ordinary WPS tile with explicit geographic registration."""
    directory.mkdir(parents=True)
    nz, ny, nx = data.shape
    index = dict(type="continuous", signed="yes", projection="regular_ll",
                 dx=.003, dy=.0179, known_x=1., known_y=1.,
                 known_lat=37., known_lon=-98.6, wordsize=2,
                 tile_x=nx, tile_y=ny, tile_z=nz)
    index.update(extra)
    (directory / "index").write_text(
        "".join(f"{key} = {value}\n" for key, value in index.items()),
        encoding="ascii")
    data.astype(">i2").tofile(directory / f"00001-{nx:05d}.00001-{ny:05d}")


def _geog(root):
    selection = GeogSelection.fallback(root)
    shape = (128, 1024)
    # Thin open-water strips through inland water give model cells with
    # seven or fourteen source pixels. WPS's reciprocal multiplication
    # then rounds legal water/lake planes to a combined area above one.
    categories = np.full(shape, 21, np.int16)
    categories[:, ::7] = 17
    _write_source(selection.path("landuse"), categories[None],
                  type="categorical", category_min=1, category_max=21,
                  **{key.lower(): value for key, value in ATTRS.items()})
    for role in ("soil_top", "soil_bottom"):
        _write_source(selection.path(role), np.full((1, *shape), 14, np.int16),
                      type="categorical", category_min=1, category_max=16)
    for role, months, value in (("terrain", 1, 100),
                                ("greenfrac", 12, 50),
                                ("lai", 12, 2),
                                ("albedo", 12, 8),
                                ("snow_albedo", 1, 60),
                                ("soil_temperature", 1, 280)):
        _write_source(selection.path(role),
                      np.full((months, *shape), value, np.int16))
    return selection


def test_two_km_geogrid_initializes_all_eight_rank_slabs(tmp_path, monkeypatch):
    from gpuwm.static import rust_bridge

    # A Python fallback would not exercise the production statics builder.
    monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV, raising=False)
    assert rust_bridge.unavailable_reason() is None
    cfg = RunConfig(nx=101, ny=51, nz=50, dx=2000., dy=2000.,
                    ztop=20000., dt=12., run_seconds=12.,
                    time_step_sound=6, specified=True,
                    open_x=True, open_y=True, sf_surface_physics=3,
                    mosaic_lu=1, mosaic_soil=1, num_soil_layers=9)
    grid = LambertGrid(38., -97., 38., 38., -97., cfg.dx, cfg.dy,
                       cfg.nx + 1, cfg.ny + 1)
    root = tmp_path / "geog"
    selection = _geog(root)
    static = build_static(grid, root, selection=selection)
    raw = np.asarray(static["LANDUSEF"], dtype=np.float32)
    assert np.isfinite(raw).all()
    assert np.all((raw >= 0) & (raw <= 1))
    # The sealed source planes are each valid. Only category coalescence
    # creates the invalid weight that used to stop physics initialization.
    naive_water = raw[16] + raw[20]
    assert np.any(naive_water > np.float32(1))
    before = raw.copy()
    halo = ranked_halo(cfg)
    specs = ranked_specs(cfg, DeviceOptions(count=8, grid=(2, 4)), halo=halo)
    assert len(specs) == 8
    assert halo > 0
    for rank, spec in enumerate(specs):
        slab = {}
        for name in ("LANDUSEF", "SOILCTOP", "LANDMASK"):
            source = np.asarray(static[name])
            slab[name] = np.empty((*source.shape[:-2], spec.cny, spec.cnx),
                                  dtype=source.dtype)
            spec.apply_gather(source, slab[name], "mass")
        fractions = ruc_mosaic_physics_inputs(
            cfg, slab, landuse_attrs=selection.landuse_global_attrs(), xice=0)
        for name, maximum in (("landusef", 21), ("soilctop", 16)):
            checked = mosaic_fractions(fractions[name],
                                       (spec.cny, spec.cnx), name, maximum)
            assert np.isfinite(checked).all(), rank
            assert np.all((checked >= 0) & (checked <= 1)), rank
        assert np.all(fractions["landusef"][20] == 0), rank
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.01, 1.01])
def test_landuse_fraction_guard_still_refuses_invalid_weights(bad):
    fractions = np.zeros((21, 1, 1), np.float32)
    fractions[16] = bad
    with pytest.raises(ValueError, match="finite and within 0..1"):
        mosaic_fractions(fractions, (1, 1), "landusef", 21)


@pytest.mark.parametrize("water,lake,other", [
    (.8, .8, 0),
    (np.nextafter(np.float32(1), np.float32(2)), 0, 0),
    (np.nan, .5, 0),
    (-.01, 0, 0),
    (np.float32(1) / np.float32(7),
     np.float32(6) * (np.float32(1) / np.float32(7)), 1.e-8),
    (.5, np.float32(.5) + np.float32(2) * np.spacing(np.float32(1)), 0),
])
def test_merge_does_not_hide_invalid_source_area(water, lake, other):
    fractions = np.zeros((21, 1, 1), np.float32)
    fractions[16], fractions[20], fractions[6] = water, lake, other
    result = ruc_mosaic_physics_inputs(
        SimpleNamespace(sf_surface_physics=3, mosaic_lu=1, mosaic_soil=0),
        dict(LANDUSEF=fractions, LANDMASK=np.zeros((1, 1), np.float32)),
        landuse_attrs=ATTRS, xice=0)
    with pytest.raises(ValueError, match="finite and within 0..1"):
        mosaic_fractions(result["landusef"], (1, 1), "landusef", 21)
