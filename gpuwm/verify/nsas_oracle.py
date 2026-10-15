"""Column cases and the file format of the NSAS (cu_physics = 96) oracle.

The WRF v4.6.1 referee is ``tools/nsas_wrf461_oracle``: a Fortran harness
linked against the ``module_cu_nsas.o`` object of the scalar WRF build (the
object inside the 0 ULP referee ``wrf.exe``), plus a second, independent
``-O0`` compile of the pinned source.  This module owns the two sides of its
interface so the harness, the fixture writer and the kernel tests cannot
disagree about a byte:

* :func:`make_cases` draws the columns: named families (deep, shallow-only,
  stable, saturated, dry, edge) and a seeded random sweep, each at
  ``nsas_dx_factor`` 0 and 1, with WRF's own driver inputs per column.
* :func:`write_cases` / :func:`read_cases` and :func:`read_outputs` are the
  stream layout the harness reads and writes (little endian, no records).

A case is one ``cu_nsas`` call.  Most cases are one column (``its = ite``),
which is what the per-column port computes.  A few cases are a two-column
ROW, because WRF's ``nsas2d`` derives ``kbmax``, ``kbm`` and ``kmax`` from
every column of the i-row at once (module_cu_nsas.F:619-631): those rows are
graded against the driver as well as the single-column stage calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

F32 = np.float32
I32 = np.int32

MAGIC = 0x4E534153          # "NSAS"

#: Per-case integer header: mp_physics, dx_factor_nsas, stepcu, itimestep,
#: ncol (columns in the i-row), family id, 0, 0.
CASE_INTS = 8
#: Per-case real header: dt, dx, pgcon, then padding.
CASE_REALS = 8
#: Per-column real header: xland, hpbl, hfx, qfx, then padding.
COL_REALS = 8
#: Interface arrays (nz + 1) then mass arrays (nz), in harness read order.
IFACE_FIELDS = ("p8w", "w")
MASS_FIELDS = ("p", "pi", "t", "qv", "qc", "qi", "rho", "dz8w", "u", "v")

#: Outputs per column, in harness write order.  Level outputs are nz long
#: except ``prsi`` (nz + 1); the four-value tails are float32 (integers are
#: written as their exact float32 value).
DRIVER_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
              "rucuten", "rvcuten")
DRIVER_SFC = ("raincv", "pratec", "hbot", "htop")
PREP_LEV = ("del", "prsl", "zl", "dot")
STAGE_LEV = ("t1", "q1", "u1", "v1", "qc2", "qi2")
STAGE_SFC = ("rain", "kbot", "ktop", "icps")

FAMILIES = {
    "deep": 1, "shallow": 2, "stable": 3, "saturated": 4, "dry": 5,
    "edge": 6, "random": 7, "row": 8, "row-solo": 9,
}

RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81


@dataclass
class Column:
    xland: float
    hpbl: float
    hfx: float
    qfx: float
    iface: dict[str, np.ndarray]
    mass: dict[str, np.ndarray]


@dataclass
class Case:
    mp_physics: int
    dx_factor_nsas: int
    stepcu: int
    itimestep: int
    family: int
    dt: float
    dx: float
    pgcon: float
    columns: list[Column] = field(default_factory=list)


def _qsat(t, p):
    """A smooth saturation mixing ratio for building soundings (not graded)."""
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def sounding(rng, nz, *, psfc, tsfc, lapse, ttrop, rh_sfc, rh_top,
             inversion=None, w_amp=0.0, w_top=12000.0, shear=(0.0, 0.0),
             u0=(0.0, 0.0), ptop=5000.0, cloud=0.0, xland=1.0, hpbl=1000.0,
             hfx=100.0, qfx=1.0e-4, saturate=False, tcap=None, hybrid=False):
    """One column of WRF driver inputs on a stretched eta grid (float64).

    ``hybrid`` uses a WRF-like hybrid vertical coordinate (terrain-following
    below, isobaric aloft, etac = 0.2), so columns of different surface
    pressure share upper-level pressures -- what makes nsas2d's row-wide
    level limits differ between neighbours.
    """
    x = np.linspace(0.0, 1.0, nz + 1)
    eta_w = 1.0 - x ** 1.35
    if hybrid:
        etac = 0.2
        b = np.clip((eta_w - etac) / (1.0 - etac), 0.0, 1.0) ** 2
        p_w = ptop + b * (psfc - ptop) + (eta_w - b) * (1.0e5 - ptop)
    else:
        p_w = ptop + eta_w * (psfc - ptop)
    p = 0.5 * (p_w[:-1] + p_w[1:])
    z_w = np.zeros(nz + 1)
    t = np.zeros(nz)
    qv = np.zeros(nz)
    tv_prev = tsfc
    for k in range(nz):
        # First guess with the layer below's virtual temperature, then one
        # correction with this layer's.
        for _ in range(2):
            dz = RD * tv_prev / G * np.log(p_w[k] / p_w[k + 1])
            zm = z_w[k] + 0.5 * dz
            tk = max(tsfc - lapse * 1.0e-3 * zm, ttrop)
            if zm > 16000.0:
                tk = ttrop + 1.5e-3 * (zm - 16000.0)
            if inversion is not None:
                z0, depth, dtinv = inversion
                if z0 <= zm <= z0 + depth:
                    tk += dtinv * (zm - z0) / depth
                elif zm > z0 + depth:
                    tk += dtinv * max(0.0, 1.0 - (zm - z0 - depth) / 1500.0)
            if tcap is not None:
                tk = min(tk, tcap)
            frac = min(zm / 12000.0, 1.0)
            rh = rh_sfc + (rh_top - rh_sfc) * frac
            if saturate:
                rh = 1.0
            qk = max(rh * _qsat(tk, p[k]), 1.0e-9)
            tv_prev = tk * (1.0 + 0.608 * qk)
        z_w[k + 1] = z_w[k] + dz
        t[k] = tk
        qv[k] = qk
    zm = 0.5 * (z_w[:-1] + z_w[1:])
    rho = p / (RD * t * (1.0 + 0.608 * qv))
    pi = (p / 1.0e5) ** (RD / CP)
    w = w_amp * np.sin(np.pi * np.clip(z_w / w_top, 0.0, 1.0))
    w[0] = 0.0
    u = u0[0] + shear[0] * zm * 1.0e-3 + rng.normal(0.0, 0.3, nz)
    v = u0[1] + shear[1] * zm * 1.0e-3 + rng.normal(0.0, 0.3, nz)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    if cloud > 0.0:
        lay = (zm > 1500.0) & (zm < 9000.0)
        qc[lay & (t > 253.0)] = cloud * rng.uniform(0.0, 1.0, nz)[lay & (t > 253.0)]
        qi[lay & (t < 263.0)] = cloud * rng.uniform(0.0, 1.0, nz)[lay & (t < 263.0)]
    iface = {"p8w": p_w, "w": w}
    mass = {"p": p, "pi": pi, "t": t, "qv": qv, "qc": qc, "qi": qi,
            "rho": rho, "dz8w": np.diff(z_w), "u": u, "v": v}
    return Column(xland=float(xland), hpbl=float(hpbl), hfx=float(hfx),
                  qfx=float(qfx),
                  iface={k: v.astype(F32) for k, v in iface.items()},
                  mass={k: v.astype(F32) for k, v in mass.items()})


def _family_kwargs(rng, family):
    """Sounding knobs for one named family."""
    if family == "deep":
        return dict(psfc=rng.uniform(98000, 101500), tsfc=rng.uniform(296, 306),
                    lapse=rng.uniform(6.2, 7.6), ttrop=rng.uniform(195, 212),
                    rh_sfc=rng.uniform(0.75, 0.95), rh_top=rng.uniform(0.35, 0.8),
                    w_amp=rng.uniform(-0.5, 3.0), w_top=rng.uniform(9e3, 14e3),
                    shear=(rng.uniform(-3, 6), rng.uniform(-3, 3)),
                    u0=(rng.uniform(-8, 8), rng.uniform(-8, 8)),
                    cloud=rng.choice([0.0, 2e-4]), xland=rng.choice([1.0, 2.0]),
                    hpbl=rng.uniform(300, 2500), hfx=rng.uniform(-20, 400),
                    qfx=rng.uniform(0, 3e-4))
    if family == "shallow":
        return dict(psfc=rng.uniform(98000, 102500), tsfc=rng.uniform(288, 302),
                    lapse=rng.uniform(7.5, 9.5), ttrop=rng.uniform(205, 218),
                    rh_sfc=rng.uniform(0.7, 0.92), rh_top=rng.uniform(0.05, 0.3),
                    inversion=(rng.uniform(900, 2200), rng.uniform(150, 500),
                               rng.uniform(3.0, 9.0)),
                    w_amp=rng.uniform(-1.0, 0.3), w_top=rng.uniform(3e3, 8e3),
                    shear=(rng.uniform(-2, 2), rng.uniform(-2, 2)),
                    cloud=0.0, xland=rng.choice([1.0, 2.0]),
                    hpbl=rng.uniform(400, 2000), hfx=rng.uniform(20, 350),
                    qfx=rng.uniform(2e-5, 2.5e-4))
    if family == "stable":
        return dict(psfc=rng.uniform(95000, 103000), tsfc=rng.uniform(255, 285),
                    lapse=rng.uniform(1.0, 4.0), ttrop=rng.uniform(210, 225),
                    rh_sfc=rng.uniform(0.3, 0.9), rh_top=rng.uniform(0.1, 0.5),
                    inversion=(0.0, rng.uniform(200, 800), rng.uniform(2, 10)),
                    w_amp=rng.uniform(-0.3, 0.3), xland=rng.choice([1.0, 2.0]),
                    hpbl=rng.uniform(0, 300), hfx=rng.uniform(-80, 10),
                    qfx=rng.uniform(-1e-5, 2e-5))
    if family == "saturated":
        return dict(psfc=rng.uniform(97000, 101500), tsfc=rng.uniform(285, 302),
                    lapse=rng.uniform(5.0, 7.0), ttrop=rng.uniform(195, 215),
                    rh_sfc=1.0, rh_top=1.0, saturate=True,
                    w_amp=rng.uniform(-1.0, 4.0), cloud=rng.choice([0.0, 5e-4]),
                    xland=rng.choice([1.0, 2.0]), hpbl=rng.uniform(100, 1500),
                    hfx=rng.uniform(-10, 200), qfx=rng.uniform(0, 2e-4))
    if family == "dry":
        return dict(psfc=rng.uniform(85000, 101000), tsfc=rng.uniform(290, 318),
                    lapse=rng.uniform(8.0, 9.8), ttrop=rng.uniform(205, 220),
                    rh_sfc=rng.uniform(0.01, 0.08), rh_top=rng.uniform(0.01, 0.05),
                    w_amp=rng.uniform(-1.0, 2.0), xland=1.0,
                    hpbl=rng.uniform(1000, 4500), hfx=rng.uniform(100, 600),
                    qfx=rng.uniform(0, 3e-5))
    raise ValueError(family)


def _edge_kwargs(rng, idx):
    """Named edge columns: the clamps, the cut-offs and the degenerate inputs."""
    base = _family_kwargs(rng, "deep")
    edges = [
        # fpvs's 330 K clamp at the surface
        dict(tsfc=338.0, lapse=9.0, rh_sfc=0.25),
        # fpvs's 180 K clamp aloft (cold tropopause)
        dict(ttrop=172.0, lapse=7.5),
        # high terrain: surface pressure 60 kPa
        dict(psfc=60000.0, tsfc=285.0),
        # a PBL height of zero beside a positive surface flux
        dict(hpbl=0.0, hfx=250.0, qfx=2e-4),
        # strong ascent at cloud base
        dict(w_amp=18.0),
        # strong descent
        dict(w_amp=-12.0),
        # negative buoyancy flux (shallow must not run)
        dict(hfx=-60.0, qfx=-1e-5),
        # calm, zero flux
        dict(w_amp=0.0, hfx=0.0, qfx=0.0, shear=(0.0, 0.0), u0=(0.0, 0.0)),
        # huge shear (downdraft efficiency clamp)
        dict(shear=(25.0, -18.0)),
        # supersaturated near the surface
        dict(rh_sfc=1.06, rh_top=0.95),
        # deep PBL top above the 0.7 sigma limit
        dict(hpbl=6000.0, hfx=500.0),
        # water point
        dict(xland=2.0, hfx=40.0, qfx=1.5e-4),
    ]
    base.update(edges[idx % len(edges)])
    return base


_MP_CHOICES = (0, 1, 3, 4, 6, 8, 28)
_DX_CHOICES = (1000.0, 3000.0, 4000.0, 9000.0, 12000.0, 27000.0, 36000.0,
               100000.0)
_DT_CHOICES = (6.0, 18.0, 30.0, 60.0, 90.0, 180.0)


def make_cases(nz: int, *, seed: int = 20261008, n_random: int = 400,
               n_per_family: int = 40, n_edge: int = 24,
               n_rows: int = 12) -> list[Case]:
    """The oracle's case list for one level count."""
    rng = np.random.default_rng(seed + nz)
    cases: list[Case] = []

    def header(family_id, dx_factor):
        return dict(mp_physics=int(rng.choice(_MP_CHOICES)),
                    dx_factor_nsas=int(dx_factor),
                    stepcu=int(rng.choice((1, 1, 1, 2, 5))),
                    itimestep=int(rng.integers(1, 500)), family=family_id,
                    dt=float(rng.choice(_DT_CHOICES)),
                    dx=float(rng.choice(_DX_CHOICES)),
                    pgcon=float(rng.choice((0.55, 0.55, 0.2, 0.7))))

    for name in ("deep", "shallow", "stable", "saturated", "dry"):
        for n in range(n_per_family):
            case = Case(**header(FAMILIES[name], n % 2))
            case.columns.append(sounding(rng, nz, **_family_kwargs(rng, name)))
            cases.append(case)
    for n in range(n_edge):
        case = Case(**header(FAMILIES["edge"], (n // 12) % 2))
        case.columns.append(sounding(rng, nz, **_edge_kwargs(rng, n)))
        cases.append(case)
    for n in range(n_random):
        family = ("deep", "shallow", "stable", "saturated", "dry",
                  "deep", "shallow")[n % 7]
        kw = _family_kwargs(rng, family)
        # widen every knob a little past its family
        kw["psfc"] = float(np.clip(kw["psfc"] * rng.uniform(0.85, 1.02),
                                   62000, 104000))
        kw["tsfc"] = float(kw["tsfc"] + rng.normal(0.0, 4.0))
        kw["hybrid"] = bool(n % 3 == 0) and kw["psfc"] > 70000.0
        case = Case(**header(FAMILIES["random"], n % 2))
        case.columns.append(sounding(rng, nz, **kw))
        cases.append(case)
    for n in range(n_rows):
        # Two columns in one i-row: a mountain beside a plain, so the
        # row-wide kbmax/kbm/kmax of nsas2d differs from each column's own.
        # Each member is ALSO run alone (family "row-solo"): that is the
        # per-column answer the port computes and is graded against; the
        # row call is the measurement of what WRF's tile coupling moves.
        case = Case(**header(FAMILIES["row"], n % 2))
        for m, psfc in enumerate((rng.uniform(98000, 101500),
                                  rng.uniform(66000, 80000))):
            kw = _family_kwargs(rng, "deep")
            kw["psfc"] = float(psfc)
            kw["hybrid"] = True
            if m == 0:
                # A near-dry-adiabatic, moist-aloft plain: its moist static
                # energy keeps falling to ~300 hPa, so the downdraft-origin
                # search (bounded by kbmax) reaches past its own limit only
                # when the mountain's row-wide kbmax lets it.
                kw["lapse"] = float(rng.uniform(8.6, 9.4))
                kw["rh_top"] = float(rng.uniform(0.5, 0.85))
                kw["ttrop"] = float(rng.uniform(185.0, 200.0))
            case.columns.append(sounding(rng, nz, **kw))
        cases.append(case)
        for col in case.columns:
            solo = Case(mp_physics=case.mp_physics,
                        dx_factor_nsas=case.dx_factor_nsas,
                        stepcu=case.stepcu, itimestep=case.itimestep,
                        family=FAMILIES["row-solo"], dt=case.dt, dx=case.dx,
                        pgcon=case.pgcon, columns=[col])
            cases.append(solo)
    return cases


def write_cases(path: Path, cases: list[Case], nz: int) -> None:
    with open(path, "wb") as f:
        np.array([MAGIC, len(cases), nz, 0], dtype="<i4").tofile(f)
        for c in cases:
            np.array([c.mp_physics, c.dx_factor_nsas, c.stepcu, c.itimestep,
                      len(c.columns), c.family, 0, 0], dtype="<i4").tofile(f)
            np.array([c.dt, c.dx, c.pgcon, 0, 0, 0, 0, 0],
                     dtype="<f4").tofile(f)
            for col in c.columns:
                np.array([col.xland, col.hpbl, col.hfx, col.qfx, 0, 0, 0, 0],
                         dtype="<f4").tofile(f)
                for name in IFACE_FIELDS:
                    a = np.asarray(col.iface[name], dtype="<f4")
                    assert a.shape == (nz + 1,)
                    a.tofile(f)
                for name in MASS_FIELDS:
                    a = np.asarray(col.mass[name], dtype="<f4")
                    assert a.shape == (nz,)
                    a.tofile(f)


def read_cases(path: Path) -> tuple[int, list[Case]]:
    raw = Path(path).read_bytes()
    pos = 0

    def take(n, dtype):
        nonlocal pos
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=pos)
        pos += 4 * n
        return a

    magic, ncase, nz, _ = take(4, "<i4")
    if magic != MAGIC:
        raise ValueError(f"{path} is not an NSAS case file")
    cases = []
    for _ in range(int(ncase)):
        ih = take(CASE_INTS, "<i4")
        rh = take(CASE_REALS, "<f4")
        case = Case(mp_physics=int(ih[0]), dx_factor_nsas=int(ih[1]),
                    stepcu=int(ih[2]), itimestep=int(ih[3]),
                    family=int(ih[5]), dt=float(rh[0]), dx=float(rh[1]),
                    pgcon=float(rh[2]))
        for _ in range(int(ih[4])):
            ch = take(COL_REALS, "<f4")
            iface = {n: take(nz + 1, "<f4").copy() for n in IFACE_FIELDS}
            mass = {n: take(nz, "<f4").copy() for n in MASS_FIELDS}
            case.columns.append(Column(xland=float(ch[0]), hpbl=float(ch[1]),
                                       hfx=float(ch[2]), qfx=float(ch[3]),
                                       iface=iface, mass=mass))
        cases.append(case)
    if pos != len(raw):
        raise ValueError(f"{path}: {len(raw) - pos} trailing bytes")
    return int(nz), cases


def output_layout(nz: int) -> list[tuple[str, int]]:
    """(name, length) per column in the harness's output order."""
    out = [(f"driver.{n}", nz) for n in DRIVER_LEV]
    out += [(f"driver.{n}", 1) for n in DRIVER_SFC]
    out += [(f"prep.{n}", nz) for n in PREP_LEV] + [("prep.prsi", nz + 1)]
    for stage in ("deep", "shallow"):
        out += [(f"{stage}.{n}", nz) for n in STAGE_LEV]
        out += [(f"{stage}.{n}", 1) for n in STAGE_SFC]
    return out


def read_outputs(path: Path, nz: int, ncolumns: int) -> dict[str, np.ndarray]:
    """Every output as an array of shape (ncolumns, length)."""
    layout = output_layout(nz)
    width = sum(n for _, n in layout)
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != width * ncolumns:
        raise ValueError(f"{path}: {raw.size} words, expected "
                         f"{width} x {ncolumns}")
    raw = raw.reshape(ncolumns, width)
    out, at = {}, 0
    for name, n in layout:
        out[name] = raw[:, at:at + n].copy()
        at += n
    return out


def flatten_columns(cases: list[Case]) -> list[tuple[int, int, Case, Column]]:
    """(case index, column index, case, column) in harness output order."""
    return [(ci, k, c, col) for ci, c in enumerate(cases)
            for k, col in enumerate(c.columns)]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--n-random", type=int, default=400)
    args = ap.parse_args(argv)
    cases = make_cases(args.nz, seed=args.seed, n_random=args.n_random)
    write_cases(args.out, cases, args.nz)
    print(f"{args.out}: {len(cases)} cases, "
          f"{sum(len(c.columns) for c in cases)} columns, nz={args.nz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
