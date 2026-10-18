"""New Tiedtke against WRF v4.6.1 on real columns, through the adapter.

The fixture (tests/data/oracles/ntiedtke_columns) is 25 columns of the WRF
CONUS 12 km benchmark state, cut by tools/ntiedtke_column_oracle, with the
answers of WRF's own cu_ntiedtke_driver at three configurations.  They are
the columns on which the full-domain oracle found the defects this file
pins:

* cuflxn's sub-cloud evaporation and the deep closure's CAPE integral
  summed in a different order than the reference (PORT-RECORD names the
  evaporation as a coverage gap: the analytic fixture never moved it);
* the adapter never told the pipeline it was on the first step, so WRF's
  itimestep == 1 forcing reset never happened;
* mfub and the closure bound the model step where cumastrn uses its own
  step ztmst = dt*stepcu.

The WRF answers come from the pinned scheme with ONE declaration
corrected: cutypen declares its cloud-base pressure ``zcbase`` INTEGER
(cu_ntiedtke.F90:1300), truncating it to whole Pa before the nearest-level
choice.  That is a WRF bug, documented and not copied (see
tools/ntiedtke_column_oracle/README.md).  The pinned build's answer is in
the fixture too, and the last test checks that WOOF parts from it only in
the columns the bug decides.

Every comparison is of IEEE-754 bit patterns.
"""
from __future__ import annotations

import csv
import pathlib
import struct
from types import SimpleNamespace

import numpy as np
import pytest

_DIR = (pathlib.Path(__file__).resolve().parent / "data" / "oracles"
        / "ntiedtke_columns")
_LEV = ("t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d", "pcps", "dz8w",
        "rho3d", "exner", "rthdynten", "rthratenlw", "rthratensw",
        "rthblten", "rqvdynten", "rqvblten")
_OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten",
            "rvcuten")
_LANES = {"rthratenlw": "rthratenlw", "rthratensw": "rthratensw",
          "gf_rthdynten": "rthdynten", "gf_rthblten": "rthblten",
          "gf_rqvdynten": "rqvdynten", "gf_rqvblten": "rqvblten"}


def _w(h):
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _rows(name):
    with open(_DIR / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _meta():
    out = {}
    for line in (_DIR / "nt-columns-meta.txt").read_text().splitlines():
        k, _, v = line.partition(" ")
        out[k] = v
    return out


def _levels(name, fields, nk, ncol):
    a = {f: np.zeros((nk, ncol), dtype=np.float32) for f in fields}
    for r in _rows(name):
        c, k = int(r["col"]) - 1, int(r["k"]) - 1
        for f in fields:
            a[f][k, c] = _w(r[f])
    return a


def _surface(name, fields, ncol):
    a = {f: np.zeros(ncol, dtype=np.float32) for f in fields}
    for r in _rows(name):
        for f in fields:
            a[f][int(r["col"]) - 1] = _w(r[f])
    return a


@pytest.fixture(scope="module")
def case():
    m = _meta()
    nz, ncol = int(m["nz"]), int(m["ncol"])
    lev = _levels("nt-columns-input-levels.csv", _LEV, nz, ncol)
    iface = _levels("nt-columns-input-iface.csv", ("p8w", "w"), nz + 1, ncol)
    sfc = _surface("nt-columns-input-surface.csv", ("xland", "hfx", "qfx"),
                   ncol)
    return m, lev, iface, sfc


def _wrf(tag, nz, ncol):
    out = _levels(f"nt-columns-wrf-{tag}-levels.csv", _OUT_LEV, nz, ncol)
    out.update(_surface(f"nt-columns-wrf-{tag}-surface.csv", ("raincv",),
                        ncol))
    return out


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.int32)


def _differing_columns(a, b):
    cols = np.zeros(a["raincv"].shape[0], dtype=bool)
    for f in _OUT_LEV:
        cols |= (_bits(a[f]) != _bits(b[f])).any(axis=0)
    cols |= _bits(a["raincv"]) != _bits(b["raincv"])
    return cols


def test_the_fixture_discriminates(case):
    """Each configuration must change WRF's answer on these columns, or the
    GPU test below could pass with the defect it names still present."""
    m, *_ = case
    nz, ncol = int(m["nz"]), int(m["ncol"])
    it2 = _wrf("zcbase-real-it2", nz, ncol)
    assert _differing_columns(it2, _wrf("zcbase-real-it1", nz, ncol)).any()
    assert _differing_columns(it2, _wrf("zcbase-real-stepcu2", nz, ncol)).any()
    assert _differing_columns(it2, _wrf("pinned-it2", nz, ncol)).any()
    # Deep columns that rain: the evaporation and CAPE paths are reached.
    assert (it2["raincv"] > 0).sum() >= 20


def _run(case, *, elapsed, stepcu=1, monkeypatch=None):
    cp = pytest.importorskip("cupy")
    import gpuwm.core.ntiedtke as ntm

    m, lev, iface, sfc = case
    nz, ncol = int(m["nz"]), int(m["ncol"])

    def d3(a):
        return cp.asarray(a.reshape(a.shape[0], 1, ncol))

    atmosphere = {
        "temperature": d3(lev["t3d"]), "qv": d3(lev["qv3d"]),
        "qc": d3(lev["qc3d"]), "qi": d3(lev["qi3d"]),
        "u": d3(lev["u3d"]), "v": d3(lev["v3d"]),
        "pressure": d3(lev["pcps"]), "dz": d3(lev["dz8w"]),
        "rho": d3(lev["rho3d"]), "exner": d3(lev["exner"]),
        "p_interface": d3(iface["p8w"]),
    }
    driver = SimpleNamespace(**{a: d3(lev[n]) for a, n in _LANES.items()})
    fields = {n: cp.asarray(sfc[n].reshape(1, ncol))
              for n in ("xland", "hfx", "qfx")}
    state = SimpleNamespace(qc=atmosphere["qc"], qi=atmosphere["qi"],
                            p=atmosphere["pressure"], w=d3(iface["w"]),
                            elapsed_seconds=float(elapsed))
    cfg = SimpleNamespace(dx=float(m["dx_native"]), dt=float(m["dt"]),
                          clock_dt=0.0, ntiedtke_tiedtke_closure=False)
    if stepcu != 1:
        base = ntm.NtPipeline

        class Stepped(base):
            def __init__(self, **kw):
                kw["stepcu"] = stepcu
                super().__init__(**kw)

        monkeypatch.setattr(ntm, "NtPipeline", Stepped)
    scheme = ntm.NewTiedtke()
    scheme.bind_driver(driver)
    r = scheme(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    out = {f: cp.asnumpy(getattr(r, f)).reshape(nz, ncol) for f in _OUT_LEV}
    out["raincv"] = cp.asnumpy(r.rainc).reshape(ncol)
    scheme.release()
    return out


def _assert_bitwise(got, want):
    bad = {}
    for f in _OUT_LEV + ("raincv",):
        n = int((_bits(got[f]) != _bits(want[f])).sum())
        if n:
            bad[f] = n
    assert not bad, f"differing words against WRF: {bad}"


def test_a_later_step_is_bitwise(case):
    m, *_ = case
    _assert_bitwise(_run(case, elapsed=3600.0),
                    _wrf("zcbase-real-it2", int(m["nz"]), int(m["ncol"])))


def test_the_first_step_is_bitwise(case):
    """elapsed 0 is WRF's itimestep 1, where pre_run zeroes the forcing."""
    m, *_ = case
    _assert_bitwise(_run(case, elapsed=0.0),
                    _wrf("zcbase-real-it1", int(m["nz"]), int(m["ncol"])))


def test_stepcu_two_is_bitwise(case, monkeypatch):
    """Unreachable in a forecast (cudt_minutes is pinned to 0 for
    cu_physics = 16), so this grades the kernels' step arithmetic only."""
    m, *_ = case
    _assert_bitwise(
        _run(case, elapsed=3600.0, stepcu=2, monkeypatch=monkeypatch),
        _wrf("zcbase-real-stepcu2", int(m["nz"]), int(m["ncol"])))


def test_the_pinned_wrf_differs_only_where_zcbase_decides(case):
    m, *_ = case
    nz, ncol = int(m["nz"]), int(m["ncol"])
    got = _run(case, elapsed=3600.0)
    pinned = _wrf("pinned-it2", nz, ncol)
    bug = _differing_columns(pinned, _wrf("zcbase-real-it2", nz, ncol))
    ours = _differing_columns(got, pinned)
    assert ours.any()
    assert not (ours & ~bug).any(), (
        f"columns differing from the pinned build that the zcbase "
        f"declaration does not decide: {np.flatnonzero(ours & ~bug) + 1}")
