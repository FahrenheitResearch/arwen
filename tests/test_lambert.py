"""Phase 3 Task 4: Lambert conformal projection module vs the geo_em oracle.

Formula authority: WRF v4.6.1 ``share/module_llxy.F`` (``lc_cone`` /
``set_lc`` / ``llij_lc`` / ``ijll_lc`` -- the same code WPS geogrid links)
plus the ARW tech note sec 2 / Snyder map-factor form.  **Acceptance oracle
= the bundle geo_em files**: the exact projection parameters come from
``namelist.wps`` / the geo_em global attributes, and XLAT/XLONG (mass, U, V),
MAPFAC_M/U/V, F, E, SINALPHA, COSALPHA must be reproduced on all four
domains' grids -- nest positions from the namelist parent_start/ratio
arithmetic (part of this task).

Gates (plan Task 4): coordinates <= 1e-3 deg absolute; map factors <= 1e-4
relative; F/E <= 1e-4 relative and sin/cos alpha <= 1e-4 absolute (the
"reproduced" fields -- observed residuals are ~1e-6, float32-storage bound).
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from gpuwm.case_data import load_experiment_case
from gpuwm.static.lambert import (
    EARTH_RADIUS_M,
    OMEGA_E,
    LambertGrid,
    grids_from_projection_config,
    grids_from_wps_namelist,
)

REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
GEO_EM_DIR = BUNDLE / "geo_em"
NAMELIST_WPS = BUNDLE / "namelists" / "namelist.wps"

requires_bundle = pytest.mark.skipif(
    not GEO_EM_DIR.is_dir(),
    reason="WRF_1974_MP55 reference bundle not present",
)

# d01 parameters from namelist.wps / geo_em.d01.nc attributes.
D01_KW = dict(ref_lat=39.6848, ref_lon=-83.9297, truelat1=30.0, truelat2=60.0,
              stand_lon=-83.9297, dx=12000.0, dy=12000.0, e_we=251, e_sn=201)

#: Fields gated on every domain (geo_em name -> tolerance spec).
COORD_TOL = 1.0e-3       # degrees, absolute
MAPFAC_RELTOL = 1.0e-4   # relative
FE_RELTOL = 1.0e-4       # relative (F, E)
ALPHA_ABSTOL = 1.0e-4    # absolute (SINALPHA, COSALPHA)

_GEO_VARS = ("XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
             "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
             "SINALPHA", "COSALPHA")


@lru_cache(maxsize=4)
def geo_fields(dom: int) -> dict:
    """Load the gated geo_em fields for domain ``dom`` as float64 (time 0)."""
    import netCDF4
    with netCDF4.Dataset(GEO_EM_DIR / f"geo_em.d{dom:02d}.nc") as ds:
        return {name: np.asarray(ds.variables[name][0], dtype=np.float64)
                for name in _GEO_VARS}


@lru_cache(maxsize=1)
def wps_grids() -> tuple:
    """All four domain grids built from the bundle namelist.wps."""
    return tuple(grids_from_wps_namelist(NAMELIST_WPS))


# ---------------------------------------------------------------------------
# Pure-math unit tests (no bundle required)
# ---------------------------------------------------------------------------

def test_cone_secant_30_60():
    g = LambertGrid(**D01_KW)
    # ln(cos30/cos60)/ln(tan(30 deg)/tan(15 deg)), evaluated independently.
    assert abs(g.cone - 0.7155668471806276) < 1e-12


def test_cone_tangent_equals_sin_truelat():
    kw = dict(D01_KW, truelat1=40.0, truelat2=40.0)
    g = LambertGrid(**kw)
    assert abs(g.cone - np.sin(np.deg2rad(40.0))) < 1e-15


def test_map_factor_is_one_at_both_true_latitudes():
    g = LambertGrid(**D01_KW)
    m = g.map_factor(np.array([30.0, 60.0]))
    np.testing.assert_allclose(m, 1.0, rtol=0, atol=1e-12)


def test_ij_latlon_roundtrip():
    g = LambertGrid(**D01_KW)
    rng = np.random.default_rng(4)
    x = rng.uniform(1.0, g.e_we - 1.0, size=64)
    y = rng.uniform(1.0, g.e_sn - 1.0, size=64)
    lat, lon = g.ij_to_latlon(x, y)
    xi, yj = g.latlon_to_ij(lat, lon)
    np.testing.assert_allclose(xi, x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(yj, y, rtol=0, atol=1e-9)


def test_staggered_shapes():
    g = LambertGrid(**D01_KW)
    ny, nx = g.e_sn - 1, g.e_we - 1
    assert g.latlon_mass()[0].shape == (ny, nx)
    assert g.latlon_u()[0].shape == (ny, nx + 1)
    assert g.latlon_v()[0].shape == (ny + 1, nx)
    assert g.latlon_c()[0].shape == (ny + 1, nx + 1)
    assert g.mapfac_m().shape == (ny, nx)
    assert g.mapfac_u().shape == (ny, nx + 1)
    assert g.mapfac_v().shape == (ny + 1, nx)
    f, e = g.coriolis_m()
    sina, cosa = g.rotation_m()
    for a in (f, e, sina, cosa):
        assert a.shape == (ny, nx)
        assert a.dtype == np.float64


def test_all_outputs_float64():
    g = LambertGrid(**D01_KW)
    for arr in (*g.latlon_mass(), *g.latlon_u(), *g.latlon_v(),
                g.mapfac_m(), g.mapfac_u(), g.mapfac_v()):
        assert arr.dtype == np.float64


def test_ratio_one_nest_is_identity():
    g = LambertGrid(**D01_KW)
    n = g.nest(i_parent_start=1, j_parent_start=1, parent_grid_ratio=1,
               e_we=g.e_we, e_sn=g.e_sn)
    lat0, lon0 = g.latlon_mass()
    lat1, lon1 = n.latlon_mass()
    np.testing.assert_allclose(lat1, lat0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(lon1, lon0, rtol=0, atol=1e-9)


def test_southern_hemisphere_mirror():
    """Mirroring ref/true lats about the equator mirrors the domain.

    (1, 1) stays the SW corner in both hemispheres (module_llxy hemi
    flipping), so the SH grid equals the NH grid negated *and* row-reversed:
    lat_s[j, i] == -lat_n[ny-1-j, i], with longitudes row-reversed too.
    """
    gn = LambertGrid(**D01_KW)
    gs = LambertGrid(**dict(D01_KW, ref_lat=-D01_KW["ref_lat"],
                            truelat1=-30.0, truelat2=-60.0))
    lat_n, lon_n = gn.latlon_mass()
    lat_s, lon_s = gs.latlon_mass()
    np.testing.assert_allclose(lat_s, -lat_n[::-1, :], rtol=0, atol=1e-9)
    np.testing.assert_allclose(lon_s, lon_n[::-1, :], rtol=0, atol=1e-9)
    np.testing.assert_allclose(gs.map_factor(lat_s),
                               gn.map_factor(lat_n[::-1, :]),
                               rtol=1e-12, atol=0)


def test_dx_dy_mismatch_raises():
    with pytest.raises(ValueError):
        LambertGrid(**dict(D01_KW, dy=11000.0))


def test_constants_are_wps_values():
    assert EARTH_RADIUS_M == 6370000.0        # module_llxy.F:152
    assert OMEGA_E == 7.292e-5                # WPS constants_module


def test_parse_minimal_wps_namelist(tmp_path):
    nml = tmp_path / "namelist.wps"
    nml.write_text(
        "&share\n max_dom = 2,\n/\n"
        "&geogrid\n"
        " parent_id         = 1, 1,\n"
        " parent_grid_ratio = 1, 3,\n"
        " i_parent_start    = 1, 10,\n"
        " j_parent_start    = 1, 20,\n"
        " e_we              = 100, 61,\n"
        " e_sn              = 80, 61,\n"
        " dx = 12000,\n dy = 12000,\n"
        " map_proj = 'lambert',\n"
        " ref_lat   = 39.6848,\n ref_lon   = -83.9297,\n"
        " truelat1  = 30.0,\n truelat2  = 60.0,\n"
        " stand_lon = -83.9297,\n"
        "/\n")
    grids = grids_from_wps_namelist(nml)
    assert len(grids) == 2
    assert grids[0].dx == 12000.0 and grids[1].dx == 4000.0
    assert grids[1].e_we == 61 and grids[1].e_sn == 61
    # nest arithmetic: nest mass point (1,1) at parent coord ips-0.5+0.5/r
    lat, lon = grids[0].ij_to_latlon(10.0 - 0.5 + 0.5 / 3.0,
                                     20.0 - 0.5 + 0.5 / 3.0)
    lat1, lon1 = grids[1].latlon_mass()
    assert abs(float(lat) - lat1[0, 0]) < 1e-9
    assert abs(float(lon) - lon1[0, 0]) < 1e-9


def test_unimplemented_projection_namelist_rejected(tmp_path):
    """Worldwide contract: mercator/polar namelists now BUILD grids
    (gpuwm.static.projection); only genuinely unimplemented WPS
    projections refuse."""
    nml = tmp_path / "namelist.wps"
    nml.write_text("&geogrid\n map_proj = 'lat-lon',\n/\n")
    with pytest.raises(NotImplementedError, match="lat-lon"):
        grids_from_wps_namelist(nml)
    merc = tmp_path / "mercator.wps"
    merc.write_text(
        "&share\n max_dom = 1,\n/\n"
        "&geogrid\n map_proj = 'mercator',\n"
        " ref_lat = -17.8,\n ref_lon = 178.5,\n truelat1 = -17.8,\n"
        " e_we = 111,\n e_sn = 89,\n dx = 12000.0,\n dy = 12000.0,\n/\n")
    (grid,) = grids_from_wps_namelist(merc)
    assert type(grid).__name__ == "MercatorGrid"
    # WPS-optional keys default per module_llxy semantics.
    assert grid.truelat2 == grid.truelat1
    assert grid.stand_lon == grid.ref_lon


def test_projection_config_builds_every_registered_domain():
    """The TOML projection plus nest layout reproduces the full chain."""
    from gpuwm.experiment import load_experiment

    exp = load_experiment_case(REPO / "configs" / "real74_4dom.toml")[0]
    grids = grids_from_projection_config(exp)
    assert len(grids) == len(exp.domains) == 4
    assert [(grid.e_we, grid.e_sn) for grid in grids] == [
        (251, 201), (501, 401), (502, 502), (601, 601)]
    np.testing.assert_allclose(
        [grid.dx for grid in grids],
        [12000.0, 3000.0, 1000.0, 1000.0 / 3.0], rtol=0, atol=0)

    by_id = dict(zip((dc.grid_id for dc in exp.domains), grids))
    for dc, child in zip(exp.domains[1:], grids[1:]):
        parent = by_id[dc.parent_id]
        ratio = dc.parent_grid_ratio
        xp = dc.i_parent_start - 0.5 + 0.5 / ratio
        yp = dc.j_parent_start - 0.5 + 0.5 / ratio
        expected = parent.ij_to_latlon(xp, yp)
        actual = child.ij_to_latlon(1.0, 1.0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        assert child.moad_cen_lat == grids[0].ref_lat
        assert child.moad_cen_lon == grids[0].ref_lon


# ---------------------------------------------------------------------------
# Oracle gates: all four geo_em domains
# ---------------------------------------------------------------------------

@requires_bundle
def test_namelist_grid_count_and_dx():
    grids = wps_grids()
    assert len(grids) == 4
    np.testing.assert_allclose([g.dx for g in grids],
                               [12000.0, 3000.0, 1000.0, 1000.0 / 3.0],
                               rtol=1e-12)
    assert [(g.e_we, g.e_sn) for g in grids] == [
        (251, 201), (501, 401), (502, 502), (601, 601)]


@requires_bundle
@pytest.mark.parametrize("dom", [1, 2, 3, 4])
def test_geo_em_latlon(dom):
    g = wps_grids()[dom - 1]
    ref = geo_fields(dom)
    for (lat, lon), tag in ((g.latlon_mass(), "M"), (g.latlon_u(), "U"),
                            (g.latlon_v(), "V")):
        dlat = np.abs(lat - ref[f"XLAT_{tag}"]).max()
        dlon = np.abs(lon - ref[f"XLONG_{tag}"]).max()
        assert dlat <= COORD_TOL, (dom, tag, "lat", dlat)
        assert dlon <= COORD_TOL, (dom, tag, "lon", dlon)


@requires_bundle
@pytest.mark.parametrize("dom", [1, 2, 3, 4])
def test_geo_em_map_factors(dom):
    g = wps_grids()[dom - 1]
    ref = geo_fields(dom)
    for ours, name in ((g.mapfac_m(), "MAPFAC_M"), (g.mapfac_u(), "MAPFAC_U"),
                       (g.mapfac_v(), "MAPFAC_V")):
        rel = np.abs(ours / ref[name] - 1.0).max()
        assert rel <= MAPFAC_RELTOL, (dom, name, rel)


@requires_bundle
@pytest.mark.parametrize("dom", [1, 2, 3, 4])
def test_geo_em_coriolis(dom):
    g = wps_grids()[dom - 1]
    ref = geo_fields(dom)
    f, e = g.coriolis_m()
    rel_f = np.abs(f / ref["F"] - 1.0).max()
    rel_e = np.abs(e / ref["E"] - 1.0).max()
    assert rel_f <= FE_RELTOL, (dom, "F", rel_f)
    assert rel_e <= FE_RELTOL, (dom, "E", rel_e)


@requires_bundle
@pytest.mark.parametrize("dom", [1, 2, 3, 4])
def test_geo_em_wind_rotation(dom):
    g = wps_grids()[dom - 1]
    ref = geo_fields(dom)
    sina, cosa = g.rotation_m()
    dsin = np.abs(sina - ref["SINALPHA"]).max()
    dcos = np.abs(cosa - ref["COSALPHA"]).max()
    assert dsin <= ALPHA_ABSTOL, (dom, "SINALPHA", dsin)
    assert dcos <= ALPHA_ABSTOL, (dom, "COSALPHA", dcos)


@requires_bundle
def test_d01_center_is_ref_point():
    """ref_lat/ref_lon sits at (e_we/2, e_sn/2) -- WPS default ref_x/ref_y."""
    g = wps_grids()[0]
    lat, lon = g.ij_to_latlon(g.e_we / 2.0, g.e_sn / 2.0)
    assert abs(float(lat) - 39.6848) < 1e-10
    assert abs(float(lon) - (-83.9297)) < 1e-10


@requires_bundle
def test_config_driven_d01_reproduces_geo_em_projection_fields():
    """Phase-5 G6 regression at the established Phase-3 tolerances."""
    from gpuwm.experiment import load_experiment

    exp = load_experiment_case(REPO / "configs" / "real74_4dom.toml")[0]
    grid = grids_from_projection_config(exp)[0]
    ref = geo_fields(1)
    lat, lon = grid.latlon_mass()
    assert np.abs(lat - ref["XLAT_M"]).max() <= COORD_TOL
    assert np.abs(lon - ref["XLONG_M"]).max() <= COORD_TOL
    for actual, name in (
            (grid.mapfac_m(), "MAPFAC_M"),
            (grid.mapfac_u(), "MAPFAC_U"),
            (grid.mapfac_v(), "MAPFAC_V")):
        assert np.abs(actual / ref[name] - 1.0).max() <= MAPFAC_RELTOL
