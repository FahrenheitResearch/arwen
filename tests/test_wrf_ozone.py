"""Gates for gpuwm.ingest.wrf_ozone (WRF v4.6.1 o3input=2 ozone pipeline).

Two layers:

* Always-on structural tests: pinned SHA-256 of the packaged climatology
  files, shapes/dtypes/monotonicity, and spot values pinned as exact
  float32 bit patterns transcribed from the verified oracle dump.
* Dump-gated bitwise tests (max_ulp 0) against the fixture written by
  the compiled control built from the UNMODIFIED WRF Fortran
  (tools/rrtmg_wrf461_oracle/ozn_build.sh + ozn_extract.F90; inputs from
  ozn_make_fixtures.py).  These skip cleanly when the dump is absent.

Run from the repo root: python -m pytest tests/test_wrf_ozone.py -q
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "rrtmg_wrf461_oracle"))
import ozn_gate  # noqa: E402

from gpuwm.ingest import wrf_ozone as wo  # noqa: E402


needs_fixture = pytest.mark.skipif(
    not ozn_gate.have_fixture(),
    reason=(
        f"ozone oracle dump not present at {ozn_gate.fixture_path()}; "
        "build it with tools/rrtmg_wrf461_oracle/ozn_build.sh (WSL "
        "gfortran; see its header, inputs from ozn_make_fixtures.py) or "
        "point GPUWM_WRF_OZONE_FIXTURES at an existing dump dir"))


@pytest.fixture(scope="module")
def fx():
    return ozn_gate.load_fixture()


@pytest.fixture(scope="module")
def climo():
    return wo.load_ozone_climatology()


def _bits(arr) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr, np.float32)).view(np.uint32)


# ---------------------------------------------------------------------------
# Always-on structural tests
# ---------------------------------------------------------------------------

def test_packaged_files_match_pinned_sha256():
    for name, sha in (("ozone.formatted", wo.OZONE_SHA256),
                      ("ozone_lat.formatted", wo.OZONE_LAT_SHA256),
                      ("ozone_plev.formatted", wo.OZONE_PLEV_SHA256)):
        digest = hashlib.sha256((wo.DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == sha, f"{name} does not match the WRF authority"


def test_loader_structure_and_pinned_spot_values(climo):
    assert climo.plev.shape == (59,) and climo.plev.dtype == np.float32
    assert climo.lat.shape == (64,) and climo.lat.dtype == np.float32
    assert climo.ozmix.shape == (59, 64, 12)
    assert climo.ozmix.dtype == np.float32
    assert np.all(np.diff(climo.plev) > 0), "plev must increase top-down"
    assert np.all(np.diff(climo.lat) > 0), "lat must increase S->N"
    assert np.isfinite(climo.ozmix).all() and (climo.ozmix > 0).all()
    # Spot values pinned as exact float32 bit patterns, transcribed from
    # the verified oracle dump (ozn_fixture.bin static/plev, /lat_ozone,
    # /ozmixin; see the dump-gated tests for the full comparison).
    pins = {
        ("plev", (0,)): 0x41E35C2A,          # 28.420002 Pa (0.2842 hPa*100.)
        ("plev", (58,)): 0x47C40880,         # 100369.0 Pa
        ("lat", (0,)): 0xC2AFBA44,           # -87.8638
        ("lat", (28,)): 0xC11C4600,          # -9.76709 (see asymmetry note)
        ("ozmix", (0, 0, 0)): 0x35D500E8,    # 1.587e-06 (Jan, lat 1, lev 1)
        ("ozmix", (58, 63, 11)): 0x33102E9B,  # 3.357e-08 (Dec, last, last)
        ("ozmix", (29, 31, 5)): 0x371E2CF4,  # 9.428e-06 (June interior)
    }
    for (name, idx), bits in pins.items():
        got = getattr(climo, name)[idx]
        assert got.view(np.uint32) == np.uint32(bits), \
            f"{name}{idx}: {got!r} lost the verified bit pattern"
    # Climatology grid asymmetry oddity, pinned so it stays documented:
    # the file stores -9.76709 (five decimals) in the south but 9.7671
    # in the north, so the grid is NOT mirror-symmetric in float32.
    assert climo.lat[28] != -climo.lat[35]
    assert climo.ozmix.max().view(np.uint32) == np.uint32(0x373A3A23)


def test_time_interp_is_pure_month_at_exact_midpoint(climo):
    # julian=15.0 -> intjulian=16.0 == date_oz(1): fact1=(45-16)/29=1.0
    # exactly, fact2=0.0, so ozmixt must equal January bitwise.
    ozm = wo.interp_ozone_to_latitudes(
        np.array([-60.0, 0.0, 42.5], np.float32), climo)
    ozt = wo.ozn_time_int(16, np.float32(15.0), ozm)
    assert np.array_equal(_bits(ozt), _bits(ozm[..., 0]))


def test_time_interp_julday_argument_is_ignored(climo):
    # WRF declares julday INTENT(IN) and never reads it; the port keeps
    # that (documented oddity).
    ozm = wo.interp_ozone_to_latitudes(np.array([12.0], np.float32), climo)
    a = wo.ozn_time_int(1, np.float32(200.5), ozm)
    b = wo.ozn_time_int(999, np.float32(200.5), ozm)
    assert np.array_equal(_bits(a), _bits(b))


def test_latitude_extrapolation_beyond_grid_is_not_clamping(climo):
    # Poleward of +-87.8638 deg WRF's lin_interpol2 extends with the edge
    # interval's slope; equality with the edge column would mean clamping.
    ozm = wo.interp_ozone_to_latitudes(
        np.array([-90.0, float(climo.lat[0]), float(climo.lat[63]), 90.0],
                 np.float32), climo)
    assert not np.array_equal(ozm[0], ozm[1])
    assert not np.array_equal(ozm[3], ozm[2])
    assert np.isfinite(ozm).all()


def test_ozn_p_int_constant_below_data_bottom(climo):
    # Levels deeper than pin(59)=100369 Pa copy ozmixt(:,59) unchanged.
    ozm = wo.interp_ozone_to_latitudes(np.array([35.0], np.float32), climo)
    ozt = wo.ozn_time_int(100, np.float32(99.5), ozm)
    p = np.array([[101100.0, 100900.0, 100700.0]], np.float32)
    o3 = wo.ozn_p_int(p, climo.plev, ozt)
    assert np.array_equal(_bits(o3), _bits(np.repeat(ozt[:, 58:59], 3,
                                                     axis=1)))


def test_ozn_p_int_rejects_non_monotonic_pin(climo):
    bad = climo.plev.copy()
    bad[5], bad[6] = bad[6], bad[5]
    p = np.array([[90000.0, 50000.0]], np.float32)
    ozt = np.full((1, 59), np.float32(1e-6))
    with pytest.raises(ValueError, match="non-monotonicity"):
        wo.ozn_p_int(p, bad, ozt)


def test_o33d_profile_shape_dtype_and_magnitude():
    p = np.geomspace(101325.0, 10000.0, 49).astype(np.float32)
    p = np.stack([p, (p * np.float32(0.99)).astype(np.float32)])
    out = wo.o33d_profile(200, np.float32(199.5),
                          np.array([30.0, -30.0], np.float32), p)
    assert out.shape == (2, 49) and out.dtype == np.float32
    # O33D carries the climatology's absolute mixing ratio (consumed by
    # the RRTMG wrappers directly as o3vmr) -- order 1e-8..1.2e-5.
    assert np.isfinite(out).all()
    assert out.min() > 1e-9 and out.max() < 1.2e-5


# ---------------------------------------------------------------------------
# Dump-gated bitwise tests (max_ulp 0 against the compiled WRF control)
# ---------------------------------------------------------------------------

@needs_fixture
def test_fixture_provenance_matches_packaged_digests():
    # The control parsed run/ozone*.formatted from the WRF source
    # authority; ozn_build.sh recorded their SHA-256.  They must be the
    # same bytes as the packaged copies the loader reads.
    manifest = Path(ozn_gate.fixture_path()).parent / \
        "ozn_oracle_sha256sums.txt"
    if not manifest.is_file():
        pytest.skip("ozn_oracle_sha256sums.txt not present beside fixture")
    text = manifest.read_text()
    by_name = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            by_name[os.path.basename(parts[1]).lstrip("*")] = parts[0]
    assert by_name.get("ozone.formatted") == wo.OZONE_SHA256
    assert by_name.get("ozone_lat.formatted") == wo.OZONE_LAT_SHA256
    assert by_name.get("ozone_plev.formatted") == wo.OZONE_PLEV_SHA256


@needs_fixture
def test_climatology_parse_bitwise(fx, climo):
    ozn_gate.assert_ulp0(climo.plev, fx["static/plev"], "plev")
    ozn_gate.assert_ulp0(climo.plev, fx["static/pin"], "pin")
    ozn_gate.assert_ulp0(climo.lat, fx["static/lat_ozone"], "lat_ozone")
    ozn_gate.assert_ulp0(np.ascontiguousarray(climo.ozmix),
                         np.ascontiguousarray(fx["static/ozmixin"][0]),
                         "ozmixin")


@needs_fixture
def test_latitude_interpolation_bitwise(fx, climo):
    xlat = fx["static/xlat"]
    # Coverage guard: on-grid, off-grid and out-of-range points in both
    # hemispheres must stay in the deck.
    assert (xlat < climo.lat[0]).any() and (xlat > climo.lat[63]).any()
    assert np.isin(climo.lat[[0, 63]], xlat).all()
    got = wo.interp_ozone_to_latitudes(xlat, climo)
    ozn_gate.assert_ulp0(np.ascontiguousarray(got),
                         np.ascontiguousarray(fx["static/ozmixm"]),
                         "ozmixm")


@needs_fixture
def test_time_interpolation_bitwise(fx):
    njul = int(fx["meta/njul"])
    assert njul >= 12
    ozmixm = np.ascontiguousarray(fx["static/ozmixm"])
    julians = []
    for j in range(1, njul + 1):
        julian = np.float32(fx[f"jul_{j}/julian"])
        julians.append(float(julian))
        got = wo.ozn_time_int(int(fx[f"jul_{j}/julday"]), julian, ozmixm)
        ozn_gate.assert_ulp0(got, fx[f"jul_{j}/ozmixt"], f"jul_{j}/ozmixt")
    # Coverage guard: January side, December side and day-366 wrap.
    assert min(julians) < 15.0 and max(julians) >= 365.0
    assert any(350.0 <= v < 365.0 for v in julians)


@needs_fixture
def test_pressure_interpolation_bitwise_stage_isolated(fx, climo):
    npc = int(fx["meta/npc"])
    assert npc >= 5
    saw_above = saw_below = saw_exact_top = False
    for pc in range(1, npc + 1):
        p = np.ascontiguousarray(fx[f"pc_{pc}/p"])
        ijul = int(fx[f"pc_{pc}/ijul"])
        ozmixt = np.ascontiguousarray(
            fx[f"jul_{ijul}/ozmixt"][:p.shape[0]])
        got = wo.ozn_p_int(p, climo.plev, ozmixt)
        ozn_gate.assert_ulp0(got, fx[f"pc_{pc}/o3vmr"], f"pc_{pc}/o3vmr")
        saw_above |= bool((p < climo.plev[0]).any())
        saw_below |= bool((p > climo.plev[58]).any())
        saw_exact_top |= bool((p == climo.plev[0]).any())
    # Coverage guard: top scaling, bottom constant, and the exact
    # pmid == pin(1) stale-kupper path must all stay exercised.
    assert saw_above and saw_below and saw_exact_top


@needs_fixture
def test_o33d_profile_end_to_end_bitwise(fx):
    xlat = np.float32(fx["static/xlat"])
    npc = int(fx["meta/npc"])
    for pc in range(1, npc + 1):
        p = np.ascontiguousarray(fx[f"pc_{pc}/p"])
        ijul = int(fx[f"pc_{pc}/ijul"])
        got = wo.o33d_profile(int(fx[f"jul_{ijul}/julday"]),
                              np.float32(fx[f"jul_{ijul}/julian"]),
                              xlat[:p.shape[0]], p)
        ozn_gate.assert_ulp0(got, fx[f"pc_{pc}/o3vmr"],
                             f"pc_{pc}/o33d end-to-end")
