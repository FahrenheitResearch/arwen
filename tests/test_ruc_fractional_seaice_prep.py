"""fractional_seaice = 1 reaches the RUC initial state the way real.exe builds it.

Before this, every preparation road called the soil ingest at its default
(binary, 0.5), so a sea-ice fraction in [0.02, 0.5) was zeroed and one in
[0.5, 1) snapped to 1 before the run started: the 0.02 threshold the seam,
the LSM and the lake read had no cell to act on.  Measured on the go door
(GFS start over the Gulf of St Lawrence, 2026-02-10 18Z): GFS carried 62
points in the band, the prepared input none, in the fractional_seaice = 1
arm too.

The referee is the HRRR v4.1.21 fork's share/module_soil_pre.F
adjust_for_seaice_post, RUCLSMSCHEME arm (:337-393), transcribed below as a
scalar loop independent of gpuwm.ingest.ruc_soil.fractional_sea_ice_post.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from tests.test_soil_node_contract import _node_contract

ROOT = Path(__file__).resolve().parents[1]


def _fork_ruc_post(xice, tsk, sst, tmn, tslb, smois, sh2o, *, flag_sst):
    """module_soil_pre.F:337-393 (fork), one cell at a time, float32."""
    f = np.float32
    n = tslb.shape[0]
    tsk = tsk.astype(f).copy()
    tslb = tslb.astype(f).copy()
    smois = smois.astype(f).copy()
    sh2o = sh2o.astype(f).copy()
    for j in range(xice.shape[0]):
        for i in range(xice.shape[1]):
            x = f(xice[j, i])
            if not x > f(0):
                continue
            t = f(tsk[j, i])
            if flag_sst:
                t = f(x * f(min(f(271.4), t)) + f(f(1) - x) * f(sst[j, i]))
            else:
                t = f(x * f(min(f(271.4), t)) + f(f(1) - x) * t)
            tsk[j, i] = t
            total = f(3.0)
            tslb[0, j, i] = t
            tslb[n - 1, j, i] = f(tmn[j, i])
            for loop in range(2, n):
                mid = f(f(f(total / f(n)) / f(4.0))
                        + f(f(loop - 2) * f(total / f(n))))
                tslb[loop - 1, j, i] = f(f(f(f(total - mid) * t)
                                           + f(mid * f(tmn[j, i]))) / total)
            smois[:, j, i] = f(1.0)
            sh2o[:, j, i] = f(0.0)
    return tsk, tslb, smois, sh2o


def _fields(shape, xice, *, sst=None):
    from gpuwm.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE)
    fields = {"SKINTEMP": np.full(shape, 268.0), "LANDSEA": np.zeros(shape),
              "SEAICE": np.asarray(xice, dtype=np.float64),
              "SNOW": np.zeros(shape), "SNOWH": np.zeros(shape),
              MAPPED_SOIL_TEMPERATURE: np.full((9, *shape), 280.0),
              MAPPED_SOIL_MOISTURE: np.full((9, *shape), 0.3)}
    if sst is not None:
        fields["SST"] = np.asarray(sst, dtype=np.float64)
    return fields


def _kw(shape):
    return dict(sf_surface_physics=3, num_soil_layers=9,
                soil_type=np.full(shape, 14),
                deep_soil_temperature=np.full(shape, 280.0),
                soil_layer_contract=_node_contract(),
                landmask=np.zeros(shape), water_temperature_policy="wrf_compat")


def test_fractional_sea_ice_post_is_the_fork_arm():
    from gpuwm.ingest.ruc_soil import fractional_sea_ice_post
    rng = np.random.default_rng(7)
    shape = (3, 5)
    xice = np.array([[0, .02, .05, .3, .49], [.5, .7, .99, 1, 0],
                     [.1, .2, 0, .45, .03]], dtype=np.float32)
    tsk = rng.uniform(255, 276, shape).astype(np.float32)
    sst = rng.uniform(270, 279, shape).astype(np.float32)
    tmn = np.where(xice > 0, 271.4, 280.0).astype(np.float32)
    tslb = rng.uniform(260, 285, (9, *shape)).astype(np.float32)
    smois = rng.uniform(.1, .4, (9, *shape)).astype(np.float32)
    sh2o = smois.copy()
    for flag_sst in (True, False):
        got = fractional_sea_ice_post(
            xice=xice, tsk=tsk, sst=sst if flag_sst else None,
            deep_soil_temperature=tmn, soil_temperature=tslb,
            soil_moisture=smois, liquid_moisture=sh2o)
        want = _fork_ruc_post(xice, tsk, sst, tmn, tslb, smois, sh2o,
                              flag_sst=flag_sst)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(
                np.asarray(g, np.float32).view(np.uint32),
                np.asarray(w, np.float32).view(np.uint32))


def test_ruc_soil_keeps_the_fraction_and_builds_the_ice_column():
    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    shape = (1, 4)
    xice = [[.01, .2, .7, 1.0]]
    sst = [[272.0, 271.8, 271.5, 271.4]]
    fields = _fields(shape, xice, sst=sst)
    default = preprocess_land_surface_soil(fields, **_kw(shape))
    binary = preprocess_land_surface_soil(fields, fractional_seaice=False,
                                          **_kw(shape))
    frac = preprocess_land_surface_soil(fields, fractional_seaice=True,
                                        **_kw(shape))
    # the default is the binary arm, word for word
    for name in ("tsk", "soil_temperature", "soil_moisture",
                 "liquid_moisture", "landmask", "xland", "xice"):
        np.testing.assert_array_equal(getattr(default, name),
                                      getattr(binary, name))
    np.testing.assert_array_equal(binary.xice, [[0, 0, 1, 1]])
    np.testing.assert_array_equal(np.asarray(frac.xice, np.float32),
                                  np.float32([[0, .2, .7, 1]]))
    np.testing.assert_array_equal(frac.landmask, [[0, 1, 1, 1]])
    # the ice cells take the fork's blended TSK and ice column
    want = _fork_ruc_post(
        np.asarray(frac.xice, np.float32), np.float32([[268, 268, 268, 268]]),
        np.float32(sst), np.asarray(frac.deep_soil_temperature, np.float32),
        np.asarray(binary.soil_temperature, np.float32),
        np.asarray(binary.soil_moisture, np.float32),
        np.asarray(binary.liquid_moisture, np.float32), flag_sst=True)
    ice = np.asarray(frac.xice) > 0
    np.testing.assert_array_equal(np.asarray(frac.tsk, np.float32)[ice],
                                  want[0][ice])
    np.testing.assert_array_equal(
        np.asarray(frac.soil_temperature, np.float32)[:, ice], want[1][:, ice])
    np.testing.assert_array_equal(np.asarray(frac.soil_moisture)[:, ice], 1.0)
    np.testing.assert_array_equal(np.asarray(frac.liquid_moisture)[:, ice], 0.0)
    # the open-water cell below 0.02 is the binary arm's
    np.testing.assert_array_equal(np.asarray(frac.tsk)[0, 0],
                                  np.asarray(binary.tsk)[0, 0])


def test_every_preparation_road_hands_the_soil_ingest_the_switch():
    """A road that calls the soil ingest without fractional_seaice runs the
    binary arm under fractional_seaice = 1 and erases every fraction."""
    calls = []
    for path in sorted((ROOT / "gpuwm").rglob("*.py")) + sorted(
            (ROOT / "tilestream").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "preprocess_land_surface_soil"):
                keys = {k.arg for k in node.keywords}
                calls.append((path.relative_to(ROOT).as_posix(), node.lineno,
                              "fractional_seaice" in keys))
    assert len(calls) >= 10, calls
    missing = [c for c in calls if not c[2]]
    assert not missing, missing


def test_preparation_keeps_the_real_002_boundary():
    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    edge = np.float32(0.02)
    below = np.nextafter(edge, np.float32(0.0))
    above = np.nextafter(edge, np.float32(1.0))
    # The static/mapped carrier is float64, even for source REAL values.
    xice = np.asarray([[below, edge, above, 0.0199999993]], np.float64)
    assert xice[0, 1] < 0.02
    assert np.float32(xice[0, 3]) == edge
    fields = _fields((1, 4), xice, sst=np.full((1, 4), 272.0))
    actual = preprocess_land_surface_soil(
        fields, fractional_seaice=True, **_kw((1, 4)))
    want = xice.astype(np.float32)
    want[want < edge] = 0.0
    np.testing.assert_array_equal(
        np.asarray(actual.xice, np.float32).view(np.uint32),
        want.view(np.uint32))
    np.testing.assert_array_equal(actual.landmask, [[0, 1, 1, 1]])
