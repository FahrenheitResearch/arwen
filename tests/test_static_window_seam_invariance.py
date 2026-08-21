"""Statics may not depend on the SOURCE WINDOW a build happened to read.

THE BREAKAGE THIS PREVENTS, measured on node-1 2026-08-20: a prepared
moving-nest run refused every relocation with

    footprint-rebuilt statics differ from the outgoing child's on shared
    ground ... {'HGT_M': 10773, 'TMN': 65}

and produced ZERO wrfout frames on all four relocation arms.  The
outgoing child's statics were built on its own 198x198 footprint; the
rebuild cropped the parent-extent statics corridor.  Both cover the same
cells of the same geography, so both must give the same bytes -- but the
corridor's source window straddles the terrain dataset's x-wrap seam
(that WPS_GEOG terrain tree starts at longitude 0.0042, and the corridor
reaches west of Greenwich) while the footprint's window does not.

The mechanism: ``cell_coords`` used to shift a point into the window's
frame (``xi + nx_global`` when ``xi < win.x0 - 0.5``) and the tile
interpolator shifted it straight back (``xx - nx_global``).  In float64
that round trip is LOSSY -- 567.121676837268 + 43200 - 43200 comes back
567.1216768372697 -- so the same cell of the same geography sampled the
source at coordinates that differed in their last bits, and the
interpolated terrain differed by up to 2.8e-11 m in 11581 of 39204
shared cells.  Whether the shift fires at all depends on the window,
which depends on the extent, which is why two builds of the same ground
disagreed.

These tests build the same footprint twice -- once directly, once as a
crop of a wider grid on the same lattice whose window crosses the seam
-- and demand byte equality, on the Rust default route and on the
``GPUWM_STATIC_PYTHON=1`` numpy route alike.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm.static.build import GeogSelection, build_static
from gpuwm.static.lambert import LambertGrid


# ---------------------------------------------------------------------------
# A synthetic WPS_GEOG whose datasets wrap in x with the seam INSIDE the
# domain: global regular_ll geometry, tiles staged only where the domain
# reads (the shape every real 30s tree has around longitude zero).
# ---------------------------------------------------------------------------

def _write_index(dirpath: Path, kv: dict) -> dict:
    dirpath.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {value}" for key, value in kv.items()
             if value is not None]
    (dirpath / "index").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return kv


def _dtype(kv: dict) -> np.dtype:
    ws = int(kv["wordsize"])
    base = {1: "i1", 2: "i2", 4: "i4"}[ws]
    if str(kv.get("signed", "no")).lower() not in ("yes", "true", ".true."):
        base = "u" + base[1:]
    return np.dtype(">" + base) if ws > 1 else np.dtype(base)


def _stage(dirpath: Path, kv: dict, origins, values) -> None:
    """Write the staged tiles; ``values(x, y, z)`` takes GLOBAL indices."""
    tx, ty = int(kv["tile_x"]), int(kv["tile_y"])
    nz = int(kv.get("tile_z", 1))
    dt = _dtype(kv)
    for xs, ys in origins:
        z, y, x = np.meshgrid(np.arange(nz),
                              np.arange(ys, ys + ty),
                              np.arange(xs, xs + tx), indexing="ij")
        name = (f"{xs:05d}-{xs + tx - 1:05d}."
                f"{ys:05d}-{ys + ty - 1:05d}")
        values(x, y, z).astype(dt).tofile(dirpath / name)


#: 0.02 deg sources: 18000x9000 global, x = 1 at longitude 0.01, tiles
#: staged either side of the seam.  The domain sits just east of it.
_FINE = dict(projection="regular_ll", dx=0.02, dy=0.02,
             known_x=1.0, known_y=1.0, known_lat=-89.99, known_lon=0.01,
             tile_x=200, tile_y=200)
_FINE_ORIGINS = ((1, 5801), (1, 6001), (17801, 5801), (17801, 6001))

#: 0.1 deg sources: 3600x1800 global, same seam story, coarser.
_COARSE = dict(projection="regular_ll", dx=0.1, dy=0.1,
               known_x=1.0, known_y=1.0, known_lat=-89.95, known_lon=0.05,
               tile_x=60, tile_y=60)
_COARSE_ORIGINS = ((1, 1141), (1, 1201), (3541, 1141), (3541, 1201))


def _seam_wps_geog(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    # terrain: fine, continuous, signed metres, strongly varying so a
    # last-bit coordinate difference moves the interpolated value.
    kv = _write_index(root / "topo_gmted2010_30s",
                      dict(_FINE, type="continuous", signed="yes",
                           wordsize=2, tile_z=1, units='"meters MSL"'))
    _stage(root / "topo_gmted2010_30s", kv, _FINE_ORIGINS,
           lambda x, y, z: (37 * x + 91 * y) % 1500 + 3 * ((x * y) % 7))

    # landuse: fine, categorical, with water blocks (category 17).
    kv = _write_index(root / "modis_landuse_20class_30s_with_lakes",
                      dict(_FINE, type="categorical", category_min=1,
                           category_max=21, wordsize=1, tile_z=1,
                           mminlu='"MODIFIED_IGBP_MODIS_NOAH"',
                           iswater=17, islake=21, isice=15, isurban=13))
    _stage(root / "modis_landuse_20class_30s_with_lakes", kv, _FINE_ORIGINS,
           lambda x, y, z: np.where(((x // 37) + (y // 41)) % 5 == 0, 17,
                                    1 + (3 * x + 5 * y) % 21))

    for name in ("soiltype_top_30s", "soiltype_bot_30s"):
        kv = _write_index(root / name,
                          dict(_COARSE, type="categorical", category_min=1,
                               category_max=16, wordsize=1, tile_z=1))
        _stage(root / name, kv, _COARSE_ORIGINS,
               lambda x, y, z: 1 + (2 * x + y) % 16)

    for name, nz, scale, span in (("greenfrac_fpar_modis", 12, 0.01, 100),
                                  ("lai_modis_10m", 12, 0.1, 60),
                                  ("albedo_modis", 12, 1.0, 30),
                                  ("maxsnowalb_modis", 1, 1.0, 80),
                                  ("soiltemp_1deg", 1, 1.0, 300)):
        kv = _write_index(root / name,
                          dict(_COARSE, type="continuous", signed="yes",
                               wordsize=2, tile_z=nz, scale_factor=scale))
        _stage(root / name, kv, _COARSE_ORIGINS,
               lambda x, y, z, span=span: (x + 2 * y + 5 * z) % span + 1)
    return root


@pytest.fixture(scope="module")
def seam_geog(tmp_path_factory) -> Path:
    return _seam_wps_geog(tmp_path_factory.mktemp("wps-geog-seam"))


#: The footprint: 40x40 cells of 2 km centred at 0.75E, entirely east of
#: the terrain tree's x = 1 seam, so ITS window never wraps.
_NX = 40
_WEST_CELLS = 60


def _footprint_grid() -> LambertGrid:
    return LambertGrid(ref_lat=30.0, ref_lon=0.75, truelat1=30.0,
                       truelat2=30.0, stand_lon=0.75,
                       dx=2000.0, dy=2000.0, e_we=_NX + 1, e_sn=_NX + 1)


def _wide_grid() -> LambertGrid:
    """The same lattice, reaching 120 km further west -- across the seam."""
    return _footprint_grid().translated(
        -_WEST_CELLS, 0, e_we=_NX + _WEST_CELLS + 1, e_sn=_NX + 1)


def _crop(field: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(field)[..., :, _WEST_CELLS:_WEST_CELLS + _NX])


def _assert_windows_straddle(seam_geog: Path) -> None:
    """The fixture must actually pose the question (validate the instrument).

    A footprint window that also wrapped, or a wide window that did not,
    would make the equality below pass for the wrong reason.
    """
    from gpuwm.static.build import GeogDataset, _DomainSampler

    topo = GeogDataset(GeogSelection.fallback(seam_geog).path("terrain"))
    assert topo.wraps_x, "the terrain fixture must be a wrapping global source"
    narrow = _DomainSampler(_footprint_grid(), 3).window(topo)
    wide = _DomainSampler(_wide_grid(), 3).window(topo)
    assert narrow.x0 >= 1 and narrow.x1 < topo.nx_global, (
        f"the footprint window {narrow.x0}..{narrow.x1} was expected to sit "
        "east of the seam")
    assert wide.x1 > topo.nx_global > wide.x0, (
        f"the wide window {wide.x0}..{wide.x1} was expected to cross the "
        f"x = {topo.nx_global} seam")


@pytest.mark.parametrize("route", ("rust-default", "python-fallback"))
def test_same_ground_same_bytes_across_a_seam_crossing_window(
        seam_geog, monkeypatch, route):
    """Identical source + identical cells = identical bytes, whatever
    window the build read them through."""
    from gpuwm.static import rust_bridge

    _assert_windows_straddle(seam_geog)
    if route == "python-fallback":
        monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    else:
        monkeypatch.delenv(rust_bridge.STATIC_PYTHON_ENV, raising=False)
        reason = rust_bridge.unavailable_reason()
        if reason is not None:
            pytest.fail(
                "the Rust static-fields bridge is not loadable, so the "
                f"DEFAULT statics route cannot be gated here: {reason}")

    selection = GeogSelection.fallback(seam_geog)
    direct = build_static(_footprint_grid(), seam_geog, selection=selection)
    wide = build_static(_wide_grid(), seam_geog, selection=selection)

    assert sorted(direct) == sorted(wide)
    for name in sorted(direct):
        expected = np.asarray(direct[name])
        actual = _crop(wide[name])
        assert actual.shape == expected.shape, name
        unequal = int(np.count_nonzero(actual != expected))
        assert unequal == 0, (
            f"{name} differs in {unequal} of {expected.size} cells between "
            f"two builds of the same ground (max |delta| "
            f"{float(np.abs(actual - expected).max()):.3e}); the source "
            "window is not part of the answer")
