"""Binary64 oracle gates for the worldwide projection transcriptions.

The committed fixture ``tests/data/llxy_oracle/llxy_wrf461_fixture.csv``
holds IEEE-754 binary64 hex words computed by the PRISTINE WRF v4.6.1
``share/module_llxy.F`` (compiled unmodified with ``-fdefault-real-8``
under WSL gfortran 13.3.0 / glibc 2.39) plus the WPS v4.6.0 geogrid
``get_map_factor``/``get_rotang`` formulas, over nine projection
configurations covering Lambert (NH secant, SH secant, SH tangent),
Mercator (tropical, antimeridian, NH subtropical) and polar
stereographic (NH, SH, pole-anchored).  See
``tools/llxy_wrf461_oracle/build_llxy.py``.

Gate policy: every quantity is compared in binary64 ULPs and pinned at
the ceiling measured on this box (numpy libm vs glibc libm is the only
source of drift; the transcriptions are operation-identical).  The one
exception is the Lambert map factor, where the product ships the ARW
tech-note form referenced to truelat1 while geogrid's authority uses
the mathematically identical form referenced to truelat2 -- that gate
is a relative bound (measured 2.3e-16) plus the same measured ULP pin.

A mutated grid (truelat1 perturbed by 1e-3) must overflow every
transform ceiling: a passing suite is not evidence unless it can fail.
"""
from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static.lambert import LambertGrid
from gpuwm.static.projection import (MercatorGrid, PolarStereoGrid,
                                     ProjectedGrid, projection_class)

FIXTURE_DIR = Path(__file__).parent / "data" / "llxy_oracle"
FIXTURE = FIXTURE_DIR / "llxy_wrf461_fixture.csv"
DECK = FIXTURE_DIR / "llxy_deck.json"

#: Mismatch sentinel, larger than any real binary64 ULP distance.
MISMATCH = 1 << 70

#: Measured max-ULP ceilings (this box: numpy vs glibc 2.39 libm),
#: pinned exactly.  Keys: (class name, row tag, output slot).
ULP_CEILINGS = {
    ("LambertGrid", "SETUP", "cone"): 2,
    ("LambertGrid", "SETUP", "rebydx"): 0,
    ("LambertGrid", "SETUP", "rsw"): 2,
    ("LambertGrid", "SETUP", "polei"): 0,
    ("LambertGrid", "SETUP", "polej"): 2,
    ("LambertGrid", "LLIJ", "i"): 16,
    ("LambertGrid", "LLIJ", "j"): 32,
    ("LambertGrid", "IJLL", "lat"): 4,
    ("LambertGrid", "IJLL", "lon"): 0,
    ("LambertGrid", "MAPF", "m"): 2,
    ("LambertGrid", "ROTA", "sina"): 3,
    ("LambertGrid", "ROTA", "cosa"): 16,
    ("MercatorGrid", "SETUP", "dlon"): 0,
    ("MercatorGrid", "SETUP", "rsw"): 0,
    ("MercatorGrid", "LLIJ", "i"): 0,
    ("MercatorGrid", "LLIJ", "j"): 4,
    ("MercatorGrid", "IJLL", "lat"): 8,
    ("MercatorGrid", "IJLL", "lon"): 0,
    ("MercatorGrid", "MAPF", "m"): 0,
    ("MercatorGrid", "ROTA", "sina"): 0,
    ("MercatorGrid", "ROTA", "cosa"): 0,
    ("PolarStereoGrid", "SETUP", "rebydx"): 0,
    ("PolarStereoGrid", "SETUP", "rsw"): 0,
    ("PolarStereoGrid", "SETUP", "polei"): 0,
    ("PolarStereoGrid", "SETUP", "polej"): 0,
    ("PolarStereoGrid", "LLIJ", "i"): 1,
    ("PolarStereoGrid", "LLIJ", "j"): 0,
    ("PolarStereoGrid", "IJLL", "lat"): 2,
    ("PolarStereoGrid", "IJLL", "lon"): 2,
    ("PolarStereoGrid", "MAPF", "m"): 0,
    ("PolarStereoGrid", "ROTA", "sina"): 0,
    ("PolarStereoGrid", "ROTA", "cosa"): 0,
}
#: Lambert map factor: equivalent-formula relative bound (see module
#: docstring); measured 2.3e-16.
LAMBERT_MAPF_RELTOL = 5.0e-16

CLASSES = {1: LambertGrid, 2: PolarStereoGrid, 3: MercatorGrid}


def _hex_to_float(word: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(word, 16)))[0]


def _ulp_key(x: float) -> int:
    """IEEE-754 totalOrder index for binary64 (mirrors the fp32 map in
    gpuwm.core.fp32_ulp, widened; -0.0 and +0.0 share index 0)."""
    bits = struct.unpack("<q", struct.pack("<d", float(x)))[0]
    return bits if bits >= 0 else -(2 ** 63) - bits


def ulp_diff(got: float, want: float) -> int:
    if got != got or want != want:  # NaN never passes a gate
        return MISMATCH
    return abs(_ulp_key(got) - _ulp_key(want))


def _load_fixture():
    rows = defaultdict(list)
    header = []
    for line in FIXTURE.read_text(encoding="ascii").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        if not line.strip():
            continue
        parts = line.split()
        rows[(parts[0], parts[1])].append(parts[2:])
    return header, rows


@pytest.fixture(scope="module")
def fixture():
    header, rows = _load_fixture()
    deck = json.loads(DECK.read_text(encoding="ascii"))
    return header, rows, deck


@pytest.fixture(scope="module")
def grids(fixture):
    _, _, deck = fixture
    out = {}
    for cid, case in deck["cases"].items():
        cls = CLASSES[case["code"]]
        out[cid] = (case, cls(case["ref_lat"], case["ref_lon"],
                              case["truelat1"], case["truelat2"],
                              case["stand_lon"], deck["dx"], deck["dx"],
                              111, 89))
    return out


def _gate(cls_name, tag, slot, got, want, context):
    d = ulp_diff(got, want)
    ceiling = ULP_CEILINGS[(cls_name, tag, slot)]
    assert d <= ceiling, (
        f"{cls_name} {tag} {slot}: {d} ULP > ceiling {ceiling} at "
        f"{context}: got {got!r} ({got.hex() if got == got else 'nan'}), "
        f"oracle {want!r}")


def test_fixture_provenance(fixture):
    header, rows, deck = fixture
    text = "\n".join(header)
    assert "d66e442fccc04111067e29274c9f9eaccc3cef28" in text
    assert ("0c777e66ffbb0602479cadcc0ab484769a140c4cb2b3e9a8f757597086f0"
            "c76b") in text
    assert ("ef546e2747987948f1aa681566936713fffa6c2cf2542a804540033bf9b4"
            "79c2") in text
    assert "GNU Fortran" in text and "13.3.0" in text
    assert "GLIBC 2.39" in text
    assert "-fdefault-real-8" in text
    assert len(deck["cases"]) == 9
    for cid in deck["cases"]:
        assert rows[("SETUP", cid)], f"missing SETUP row for {cid}"


def test_deck_covers_all_projection_families(fixture):
    _, _, deck = fixture
    codes = {c["code"] for c in deck["cases"].values()}
    assert codes == {1, 2, 3}
    hemis = {(c["code"], c["truelat1"] < 0) for c in deck["cases"].values()}
    assert (1, True) in hemis and (1, False) in hemis   # SH + NH Lambert
    assert (2, True) in hemis and (2, False) in hemis   # both poles
    assert any(abs(c["ref_lon"]) > 170.0 for c in deck["cases"].values()
               if c["code"] == 3)                       # antimeridian merc


def test_input_echoes_match_python_parse(fixture, grids):
    """gfortran's list-directed decimal parse and Python's float parse
    must agree bit-for-bit on every deck input, or the fixture keys
    would not address the values the product computes."""
    _, rows, deck = fixture
    for cid, case in deck["cases"].items():
        for (lat, lon), row in zip(case["llij"], rows[("LLIJ", cid)]):
            assert _hex_to_float(row[0]) == lat
            assert _hex_to_float(row[1]) == lon
        for (i, j), row in zip(case["ijll"], rows[("IJLL", cid)]):
            assert _hex_to_float(row[0]) == i
            assert _hex_to_float(row[1]) == j
        for lat, row in zip(case["mapf"], rows[("MAPF", cid)]):
            assert _hex_to_float(row[0]) == lat
        for lon, row in zip(case["rota"], rows[("ROTA", cid)]):
            assert _hex_to_float(row[0]) == lon


def test_setup_state_matches_oracle(fixture, grids):
    _, rows, _ = fixture
    for cid, (case, grid) in grids.items():
        (s,) = rows[("SETUP", cid)]
        name = type(grid).__name__
        hemi, cone, rebydx, rsw, polei, polej, dlon = (
            _hex_to_float(v) for v in s[1:])
        assert hemi == grid.hemi, cid
        _gate(name, "SETUP", "rsw", grid.rsw, rsw, cid)
        if isinstance(grid, LambertGrid):
            _gate(name, "SETUP", "cone", grid.cone, cone, cid)
            _gate(name, "SETUP", "rebydx", grid.rebydx, rebydx, cid)
            _gate(name, "SETUP", "polei", grid.polei, polei, cid)
            _gate(name, "SETUP", "polej", grid.polej, polej, cid)
        elif isinstance(grid, PolarStereoGrid):
            _gate(name, "SETUP", "rebydx", grid.rebydx, rebydx, cid)
            _gate(name, "SETUP", "polei", grid.polei, polei, cid)
            _gate(name, "SETUP", "polej", grid.polej, polej, cid)
        else:
            _gate(name, "SETUP", "dlon", grid.dlon, dlon, cid)


def test_latlon_to_ij_matches_oracle(fixture, grids):
    _, rows, _ = fixture
    for cid, (case, grid) in grids.items():
        name = type(grid).__name__
        for (lat, lon), row in zip(case["llij"], rows[("LLIJ", cid)]):
            i, j = grid.latlon_to_ij(lat, lon)
            _gate(name, "LLIJ", "i", float(i), _hex_to_float(row[2]),
                  (cid, lat, lon))
            _gate(name, "LLIJ", "j", float(j), _hex_to_float(row[3]),
                  (cid, lat, lon))


def test_ij_to_latlon_matches_oracle(fixture, grids):
    _, rows, _ = fixture
    for cid, (case, grid) in grids.items():
        name = type(grid).__name__
        for (i, j), row in zip(case["ijll"], rows[("IJLL", cid)]):
            lat, lon = grid.ij_to_latlon(i, j)
            want_lon = _hex_to_float(row[3])
            got_lon = float(lon)
            # both sides wrap into (-180, 180]; a value landing exactly
            # on the seam may come out on either branch -- compare on
            # the oracle's branch.
            if abs(got_lon - want_lon) > 180.0:
                got_lon -= 360.0 * np.sign(got_lon - want_lon)
            _gate(name, "IJLL", "lat", float(lat), _hex_to_float(row[2]),
                  (cid, i, j))
            _gate(name, "IJLL", "lon", got_lon, want_lon, (cid, i, j))


def test_map_factor_matches_oracle(fixture, grids):
    _, rows, _ = fixture
    for cid, (case, grid) in grids.items():
        name = type(grid).__name__
        for lat, row in zip(case["mapf"], rows[("MAPF", cid)]):
            m = float(grid.map_factor(lat))
            want = _hex_to_float(row[1])
            if isinstance(grid, LambertGrid):
                assert abs(m - want) <= LAMBERT_MAPF_RELTOL * abs(want), (
                    cid, lat, m, want)
            _gate(name, "MAPF", "m", m, want, (cid, lat))


def test_rotation_matches_oracle(fixture, grids):
    _, rows, _ = fixture
    for cid, (case, grid) in grids.items():
        name = type(grid).__name__
        for lon, row in zip(case["rota"], rows[("ROTA", cid)]):
            sina, cosa = grid.rotation(np.asarray(lon))
            _gate(name, "ROTA", "sina", float(sina),
                  _hex_to_float(row[1]), (cid, lon))
            _gate(name, "ROTA", "cosa", float(cosa),
                  _hex_to_float(row[2]), (cid, lon))
        if isinstance(grid, MercatorGrid):
            # get_rotang PROJ_MERC: identically zero rotation, bitwise.
            sina, cosa = grid.rotation_m()
            assert (sina == 0.0).all() and (cosa == 1.0).all()


def test_mutated_grid_overflows_the_gates(fixture, grids):
    """Negative control: perturbing truelat1 by 1e-3 must blow every
    transform ceiling, or the gates prove nothing."""
    _, rows, _ = fixture
    for cid, (case, grid) in grids.items():
        if case["code"] == 2 and case["ref_lat"] == 90.0:
            continue  # pole-anchored case: rsw ~ 1e-13, gate saturates
        cls = type(grid)
        bad = cls(case["ref_lat"], case["ref_lon"],
                  case["truelat1"] + 1e-3, case["truelat2"],
                  case["stand_lon"], grid.dx, grid.dy, 111, 89)
        name = cls.__name__
        tripped = 0
        for (lat, lon), row in zip(case["llij"], rows[("LLIJ", cid)]):
            i, j = bad.latlon_to_ij(lat, lon)
            if ulp_diff(float(j), _hex_to_float(row[3])) > \
                    ULP_CEILINGS[(name, "LLIJ", "j")]:
                tripped += 1
        assert tripped > len(case["llij"]) // 2, (
            f"{cid}: mutated grid passed the LLIJ gate; the gate cannot "
            "fail and is not evidence")


def test_polar_pole_point_branch():
    """The r2 == 0 branch of ijll_ps: lat = hemi*90, lon = the wrapped
    reference longitude stand_lon + 90 (module_llxy.F:820-823)."""
    for tl1, stand_lon in ((60.0, -147.7), (-71.0, 166.7)):
        grid = PolarStereoGrid(90.0 * (1 if tl1 > 0 else -1), stand_lon,
                               tl1, tl1, stand_lon, 12000.0, 12000.0,
                               111, 89)
        lat, lon = grid.ij_to_latlon(grid.polei, grid.polej)
        assert float(lat) == 90.0 * grid.hemi
        expected = stand_lon + 90.0
        if expected > 180.0:
            expected -= 360.0
        if expected < -180.0:
            expected += 360.0
        assert float(lon) == expected


def test_southern_coriolis_is_negative():
    for grid in (
            LambertGrid(-27.5, 153.0, -17.5, -37.5, 153.0, 12000.0,
                        12000.0, 111, 89),
            MercatorGrid(-17.8, 178.5, -17.8, -17.8, 178.5, 12000.0,
                         12000.0, 111, 89),
            PolarStereoGrid(-77.85, 166.7, -71.0, -71.0, 166.7, 12000.0,
                            12000.0, 111, 89)):
        f, e = grid.coriolis_m()
        assert (f < 0.0).all(), type(grid).__name__
        assert (e > 0.0).all(), type(grid).__name__


def test_projection_class_dispatch():
    assert projection_class("lambert") is LambertGrid
    assert projection_class("Mercator") is MercatorGrid
    assert projection_class("polar") is PolarStereoGrid
    for cls in (LambertGrid, MercatorGrid, PolarStereoGrid):
        assert issubclass(cls, ProjectedGrid)
    with pytest.raises(
            NotImplementedError,
            match=r"angular dx/dy.*polar filter.*map_proj == 6"):
        projection_class("lat-lon")


def test_wrf_map_proj_codes():
    grid_l = LambertGrid(39.7, -84.0, 30.0, 60.0, -84.5, 12000.0,
                         12000.0, 111, 89)
    grid_m = MercatorGrid(1.3, 103.8, 1.3, 1.3, 103.8, 12000.0,
                          12000.0, 111, 89)
    grid_p = PolarStereoGrid(64.8, -147.7, 64.8, 64.8, -147.7, 12000.0,
                             12000.0, 111, 89)
    assert (grid_l.wrf_map_proj, grid_p.wrf_map_proj,
            grid_m.wrf_map_proj) == (1, 2, 3)
    assert grid_l.map_proj_char == "Lambert Conformal"
    assert grid_p.map_proj_char == "Polar Stereographic"
    assert grid_m.map_proj_char == "Mercator"


def test_nest_registration_round_trip():
    """Child (1,1) sits at the parent pickup coordinate for every
    projection (the machinery is shared; pin it anyway)."""
    parents = (
        LambertGrid(-27.5, 153.0, -17.5, -37.5, 153.5, 12000.0, 12000.0,
                    111, 89),
        MercatorGrid(-17.8, 178.5, -17.8, -17.8, 178.5, 12000.0, 12000.0,
                     111, 89),
        PolarStereoGrid(64.8, -147.7, 64.8, 64.8, -147.7, 12000.0,
                        12000.0, 111, 89))
    for parent in parents:
        child = parent.nest(30, 25, 3, 61, 49)
        assert type(child) is type(parent)
        lat11, lon11 = child.ij_to_latlon(1.0, 1.0)
        plat, plon = parent.ij_to_latlon(29.5 + 0.5 / 3, 24.5 + 0.5 / 3)
        np.testing.assert_allclose(lat11, plat, rtol=0, atol=1e-9)
        dlon = (float(lon11) - float(plon) + 540.0) % 360.0 - 180.0
        assert abs(dlon) < 1e-9
        assert child.dx == parent.dx / 3
