"""mp_zero_out / mp_zero_out_all: WOOF's GPU pass against the branch's Fortran.

The HRRR door runs mp_zero_out = 2, mp_zero_out_thresh = 1e-12 and
mp_zero_out_all = 1 (request-defaults.v1.toml, source gsl_wrf39): the
operational RAP/HRRR branch calls microphysics_zero_outb and _outa on its
moist and scalar arrays after every microphysics call (NOAA-EMC/HRRR v4.1.21
WRFV3.9 dyn_em/solve_em.F:4070-4141).  This oracle compiles the branch's
phys/module_microphysics_zero_out.F unchanged (zero_out_stubs.F90 supplies
the three framework modules it USEs) with the column driver in
zero_out_driver.F90, and runs WOOF's production pass,
gpuwm.core.microphysics.microphysics_zero_out, on a real DomainState with
the same words.  Every output word of every species is compared as uint32.

Arrays follow the branch's Registry (Registry.EM_COMMON:2791 thompson,
:2804 thompsonaero): moist = qv qc qr qi qs qg, scalar = qni qnr (mp 8) or
qni qnr qnc qnwfa qnifa (mp 28), each behind WRF's dummy index 1.

    python zero_out_oracle.py build FORK_SRC BUILD_DIR      (CPU)
    python zero_out_oracle.py run BUILD_DIR OUT_DIR          (GPU, via the mutex)

``run`` writes OUT_DIR/zero_out-<mode>.json, mode = strict when
GPUWM_WRF_EXACT=1, default otherwise.  Analysis tooling only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = np.float32

#: gfortran flags: the -O0 arm is the referee; the second is WRF's own GNU
#: optimisation flags (arch/configure.defaults), a cross-check.
ARMS = {
    "o0": ["-O0", "-ffp-contract=off"],
    "wrf_o2": ["-O2", "-ftree-vectorize", "-funroll-loops"],
}
COMMON = ["-ffree-line-length-none", "-fconvert=little-endian"]

MOIST = ("qv", "qc", "qr", "qi", "qs", "qg")
#: WRF Registry scalar name -> WOOF state attribute, in Registry order.
SCALAR = {8: (("qni", "ni"), ("qnr", "nr")),
          28: (("qni", "ni"), ("qnr", "nr"), ("qnc", "nc"),
               ("qnwfa", "nwfa"), ("qnifa", "nifa"))}

#: name, mp_physics, mp_zero_out, thresh, specified (spec_zone 1)
CONFIGS = (
    ("hrrr_door", 28, 2, 1.0e-12, True),
    ("hrrr_door_unforced", 28, 2, 1.0e-12, False),
    ("mode1_thompson_aero", 28, 1, 1.0e-12, True),
    ("mode2_thompson", 8, 2, 1.0e-12, True),
    ("mode2_thresh_zero", 28, 2, 0.0, True),
    ("mode2_default_thresh", 28, 2, 1.0e-8, True),
)
NX, NY, NZ = 12, 10, 20

# Edge words (float32 bit patterns).
NEG_ZERO = np.uint32(0x80000000).view(F)
POS_SUB = np.uint32(0x000116C2).view(F)
NEG_SUB = np.uint32(0x800116C2).view(F)
QNAN = np.uint32(0x7FC00000).view(F)


def _regime(i, j):
    """A regime label per column; edge columns sit on the ring too."""
    k = (j * NX + i) % 8
    return ("convective", "stable_night", "snow_ice", "ocean_stratocu",
            "high_terrain", "threshold_edges", "negatives_signed_zero",
            "subnormal_nan")[k]


def make_fields(mp: int, thresh: float, seed: int = 20261007):
    """WOOF-order (nz, ny, nx) float32 arrays for every species."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 1.0, NZ)[:, None]
    names = MOIST + tuple(w for _, w in SCALAR[mp])
    out = {n: np.zeros((NZ, NY, NX), F) for n in names}
    t = F(thresh) if thresh > 0 else F(1.0e-12)
    edges = np.array([t, np.nextafter(t, F(1)), np.nextafter(t, F(0)),
                      F(0.0), NEG_ZERO, F(-1.0e-9), F(1.0e-13), F(9.99e-13),
                      F(1.0e-8), np.nextafter(F(1e-8), F(0)), F(-1.0e-12),
                      F(2.0e-12)], F)
    for j in range(NY):
        for i in range(NX):
            reg = _regime(i, j)
            col = {}
            qv = 0.016 * np.exp(-3.0 * z[:, 0])
            if reg == "convective":
                cloud = np.exp(-((z[:, 0] - 0.35) / 0.2) ** 2)
                col = dict(qv=qv, qc=2e-3 * cloud, qr=5e-3 * cloud ** 2,
                           qi=1e-4 * np.exp(-((z[:, 0] - 0.7) / 0.1) ** 2),
                           qs=2e-3 * cloud * z[:, 0], qg=3e-3 * cloud ** 3,
                           ni=1e5 * cloud, nr=1e4 * cloud, nc=1e8 * cloud,
                           nwfa=1e9 * (1 - z[:, 0]), nifa=1e6 * (1 - z[:, 0]))
            elif reg == "stable_night":
                tiny = 10.0 ** rng.uniform(-15, -9, NZ)
                col = dict(qv=0.006 * np.exp(-4 * z[:, 0]), qc=tiny,
                           qr=tiny * 0.3, qi=tiny * 0.1, qs=tiny * 2,
                           qg=tiny * 0.01, ni=tiny * 1e6, nr=tiny * 1e3,
                           nc=tiny * 1e9, nwfa=3e8 + 0 * tiny, nifa=tiny)
            elif reg == "snow_ice":
                col = dict(qv=0.002 * np.exp(-3 * z[:, 0]), qc=1e-5 * (z[:, 0] < .2),
                           qr=0 * z[:, 0], qi=5e-5 * (z[:, 0] > .3),
                           qs=8e-4 * (z[:, 0] < .6), qg=1e-6 + 0 * z[:, 0],
                           ni=2e5 * (z[:, 0] > .3), nr=0 * z[:, 0],
                           nc=5e7 * (z[:, 0] < .2), nwfa=1e8 + 0 * z[:, 0],
                           nifa=1e5 + 0 * z[:, 0])
            elif reg == "ocean_stratocu":
                deck = (z[:, 0] > .05) & (z[:, 0] < .15)
                col = dict(qv=0.012 * np.exp(-2 * z[:, 0]), qc=4e-4 * deck,
                           qr=1e-6 * deck, qi=0 * z[:, 0], qs=0 * z[:, 0],
                           qg=0 * z[:, 0], ni=0 * z[:, 0], nr=50 * deck,
                           nc=1e8 * deck, nwfa=5e8 + 0 * z[:, 0],
                           nifa=1e4 + 0 * z[:, 0])
            elif reg == "high_terrain":
                col = dict(qv=0.003 * np.exp(-3 * z[:, 0]),
                           qc=1e-11 * rng.uniform(-1, 1, NZ),
                           qr=1e-12 * rng.uniform(-2, 2, NZ),
                           qi=3e-5 * (z[:, 0] > .4), qs=1e-4 * (z[:, 0] > .2),
                           qg=2e-12 * rng.uniform(-1, 1, NZ),
                           ni=1e4 * (z[:, 0] > .4), nr=1e-13 + 0 * z[:, 0],
                           nc=1e-12 * rng.uniform(0, 2, NZ), nwfa=1e8 + 0 * z[:, 0],
                           nifa=1e5 + 0 * z[:, 0])
            elif reg == "threshold_edges":
                pick = edges[(np.arange(NZ) + i + j) % edges.size]
                col = {n: pick for n in names}
                col["qv"] = pick
            elif reg == "negatives_signed_zero":
                vals = np.array([F(-1e-9), NEG_ZERO, F(-1e-15), F(0.0),
                                 F(-3e-5), F(1e-12), F(-1e-12)], F)
                pick = vals[(np.arange(NZ) + i) % vals.size]
                col = {n: pick for n in names}
            else:  # subnormal_nan
                vals = np.array([POS_SUB, NEG_SUB, F(1e-38), F(-1e-38),
                                 NEG_ZERO, F(0.0), F(5e-13)], F)
                pick = vals[(np.arange(NZ) + j) % vals.size].copy()
                col = {n: pick for n in names}
                if i == 7 and j == 5:   # one NaN word, a moist and a scalar
                    col["qc"] = pick.copy()
                    col["qc"][3] = QNAN
                    col["ni"] = pick.copy()
                    col["ni"][4] = QNAN
            for n in names:
                out[n][:, j, i] = np.asarray(col.get(n, 0 * z[:, 0]), F)
    return out


def to_fortran(fields, names):
    """(n+1) species behind the dummy slot, Fortran (nx, nz, ny, n) order."""
    arr = np.zeros((len(names) + 1, NY, NZ, NX), F)
    for s, n in enumerate(names, start=1):
        arr[s] = fields[n].transpose(1, 0, 2)
    return np.ascontiguousarray(arr)


def build(fork_src: Path, build_dir: Path) -> dict:
    fork_src, build_dir = fork_src.resolve(), build_dir.resolve()
    sys.path.insert(0, str(HERE))
    from fetch_fork import fetch
    hashes = fetch(fork_src, ["phys/module_microphysics_zero_out.F",
                              "dyn_em/solve_em.F",
                              "Registry/Registry.EM_COMMON"])
    build_dir.mkdir(parents=True, exist_ok=True)
    for arm, flags in ARMS.items():
        b = build_dir / arm
        b.mkdir(exist_ok=True)
        run = lambda *a: subprocess.run(["gfortran", *flags, *COMMON, *a],
                                        cwd=b, check=True)
        run("-c", str(HERE / "zero_out_stubs.F90"))
        run("-c", "-cpp", "-ffree-form",
            str(fork_src / "phys/module_microphysics_zero_out.F"))
        run("-c", str(HERE / "zero_out_driver.F90"))
        run("-o", "zero_out_driver", "zero_out_stubs.o",
            "module_microphysics_zero_out.o", "zero_out_driver.o")
    version = subprocess.run(["gfortran", "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    receipt = {"sources": hashes, "gfortran": version,
               "arms": {a: f + COMMON for a, f in ARMS.items()}}
    (build_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def run_fortran(build_dir: Path, arm: str, work: Path, fields, mp, mode,
                thresh, sz):
    moist_names = MOIST
    scalar_names = tuple(w for _, w in SCALAR[mp])
    m = to_fortran(fields, moist_names)
    s = to_fortran(fields, scalar_names)
    inp = work / "in.bin"
    with inp.open("wb") as fh:
        np.array([NX, NY, NZ, m.shape[0], s.shape[0], mode, sz], "<i4").tofile(fh)
        np.array([thresh], "<f4").tofile(fh)
        m.tofile(fh)
        s.tofile(fh)
    out = work / f"out-{arm}.bin"
    subprocess.run([str(build_dir / arm / "zero_out_driver"), str(inp), str(out)],
                   check=True)
    data = np.fromfile(out, "<f4")
    m2 = data[:m.size].reshape(m.shape)
    s2 = data[m.size:].reshape(s.shape)
    res = {}
    for k, n in enumerate(moist_names, start=1):
        res[n] = np.ascontiguousarray(m2[k].transpose(1, 0, 2))
    for k, n in enumerate(scalar_names, start=1):
        res[n] = np.ascontiguousarray(s2[k].transpose(1, 0, 2))
    return res


def run_woof(fields, mp, mode, thresh, specified):
    import cupy as cp
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.state import DomainState
    from gpuwm.core.microphysics import microphysics_zero_out
    from gpuwm.core.moist import moist_species
    cfg = RunConfig(nx=NX, ny=NY, nz=NZ, dx=3000., dy=3000., ztop=20000.,
                    dt=20., run_seconds=0., moist=True, moist_cq=True,
                    mp_physics=mp, specified=specified, spec_zone=1,
                    relax_zone=4, spec_bdy_width=5,
                    mp_zero_out=mode, mp_zero_out_thresh=thresh,
                    mp_zero_out_all=1)
    validate_run_config(cfg)
    state = DomainState(cfg)
    names = moist_species(state)
    want = set(MOIST) | {w for _, w in SCALAR[mp]}
    if set(names) != want:
        raise AssertionError(f"WOOF species {sorted(names)} != Registry {sorted(want)}")
    for n in names:
        getattr(state, n)[...] = cp.asarray(fields[n])
    microphysics_zero_out(state, cfg)
    cp.cuda.Device().synchronize()
    out = {n: cp.asnumpy(getattr(state, n)) for n in names}
    del state
    return out


def compare(a, b):
    """Bitwise: words that differ, and the ULP distance on the ordered line."""
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    wa = a.view(np.int32)
    wb = b.view(np.int32)
    oa = monotone_fp32_key(a)
    ob = monotone_fp32_key(b)
    differ = wa != wb
    both_nan = np.isnan(a) & np.isnan(b)
    ulp = np.where(both_nan, 0, np.abs(oa - ob))
    return int(differ.sum()), int(ulp.max(initial=0))


def run(build_dir: Path, out_dir: Path) -> dict:
    build_dir, out_dir = build_dir.resolve(), out_dir.resolve()
    mode_name = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"mode": mode_name, "columns_per_config": NX * NY, "nz": NZ,
              "configs": {}}
    import cupy as cp
    report["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    total = differing = 0
    for name, mp, mode, thresh, specified in CONFIGS:
        fields = make_fields(mp, thresh)
        woof = run_woof(fields, mp, mode, thresh, specified)
        work = out_dir / name
        work.mkdir(exist_ok=True)
        sz = 1 if specified else 0
        cfg_rep = {"mp_physics": mp, "mp_zero_out": mode, "thresh": thresh,
                   "spec_zone": sz, "arms": {}}
        for arm in ARMS:
            ref = run_fortran(build_dir, arm, work, fields, mp, mode, F(thresh), sz)
            fields_rep = {}
            for n in ref:
                d, u = compare(woof[n], ref[n])
                changed = int((ref[n].view(np.uint32) != fields[n].view(np.uint32)).sum())
                fields_rep[n] = {"differing": d, "max_ulp": u,
                                 "words": int(ref[n].size),
                                 "words_the_pass_changed": changed}
                if d:
                    bad = np.argwhere(woof[n].view(np.uint32) != ref[n].view(np.uint32))
                    k, j, i = (int(x) for x in bad[0])
                    fields_rep[n]["first"] = {
                        "k": k, "j": j, "i": i, "regime": _regime(i, j),
                        "input": f"{fields[n][k, j, i].view(np.uint32):08x}",
                        "woof": f"{woof[n][k, j, i].view(np.uint32):08x}",
                        "wrf": f"{ref[n][k, j, i].view(np.uint32):08x}"}
                if arm == "o0":
                    total += ref[n].size
                    differing += d
            cfg_rep["arms"][arm] = fields_rep
        report["configs"][name] = cfg_rep
    report["words_compared_o0"] = total
    report["differing_o0"] = differing
    (out_dir / f"zero_out-{mode_name}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"zero_out {mode_name}: {differing} of {total} words differ (o0 arm)")
    for name, rep in report["configs"].items():
        for arm, fr in rep["arms"].items():
            bad = {n: (v["differing"], v["max_ulp"]) for n, v in fr.items() if v["differing"]}
            print(f"  {name} {arm}: {bad or 'bitwise'}")
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[0] == "build":
        print(json.dumps(build(Path(argv[1]), Path(argv[2])), indent=1))
    elif argv[0] == "run":
        rep = run(Path(argv[1]), Path(argv[2]))
        return 1 if rep["differing_o0"] else 0
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
