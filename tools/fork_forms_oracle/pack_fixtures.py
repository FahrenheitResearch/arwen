"""Pack the fork-form oracles' Fortran words into the regression fixtures.

    python pack_fixtures.py RUNS_DIR FIXTURE_DIR      (CPU, on the oracle box)

RUNS_DIR is the lane's run directory on the box (README.md): build/ trees
of zero_out_oracle.py, albsol_oracle.py, diff6_regimes.py and
upper_wind_regimes.py, the RRTMG fork-cloud outputs (rrtmg/wrf/fork-cloud)
and the RUC fork case directories (ruc/<run>/fork-*-strict).  The Fortran
side is rerun here from its compiled executables (-O0 arm), so the fixture
is what the branch's code computes for the inputs the generators make; the
inputs are not stored where a generator rebuilds them byte for byte.

Fixtures (tests/test_noaa_wrf39_fork_forms_oracle.py reads them):

* zero_out.npz   per config and species, the uint32 XOR of the branch's
  output with its input (zero wherever the pass left a word alone).
* albsol.npz     ALBSOL, ALBBCKSOL, COSZEN after every call of every case.
* upper_wind.npz per case, the XOR of the branch's u_save/v_save with the
  input winds.
* diff6.npz      the branch's tendency for every row in three regimes
  (FIXTURE_DIFF6_REGIMES), and the full steps its rk_scalar_tend forms.
* rrtmg_fork_cloud.npz  "<regime>/<field>" as tests/data/oracles/
  rrtmg_legacy/column/wrf461.npz, from the fork-cloud arm.
* ruc_fork_*.npz (tools/ruc_lsm_gpu_oracle/pack_fixture.py).
* provenance.json  receipts and sha256 of every fixture.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def _xor(out, inp):
    return np.bitwise_xor(np.ascontiguousarray(out, np.float32).view(np.uint32),
                          np.ascontiguousarray(inp, np.float32).view(np.uint32))


def pack_zero_out(runs: Path, dst: Path) -> dict:
    import zero_out_oracle as z
    build = runs / "zero_out" / "build"
    work = runs / "zero_out" / "pack"
    work.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, mp, mode, thresh, specified in z.CONFIGS:
        fields = z.make_fields(mp, thresh)
        ref = z.run_fortran(build, "o0", work, fields, mp, mode, np.float32(thresh),
                            1 if specified else 0)
        for species, out in ref.items():
            arrays[f"{name}/{species}"] = _xor(out, fields[species])
    np.savez_compressed(dst / "zero_out.npz", **arrays)
    return json.loads((build / "receipt.json").read_text())


def pack_albsol(runs: Path, dst: Path) -> dict:
    import albsol_oracle as a
    build = runs / "albsol" / "build"
    work = runs / "albsol" / "pack"
    work.mkdir(parents=True, exist_ok=True)
    inputs = a.inputs_for()
    out = np.zeros((len(a.STARTS), a.NCALLS, 3, a.NCOL), np.float32)
    for c in range(len(a.STARTS)):
        sub = dict(inputs, lat=inputs["lat_case"][c], lon=inputs["lon_case"][c])
        sub["scalars"] = [inputs["scalars"][c]]
        out[c] = a._one_case(build, "o0", work, sub)
    np.savez_compressed(dst / "albsol.npz", out=out,
                        scalars=np.asarray(inputs["scalars"], np.float32))
    return json.loads((build / "receipt.json").read_text())


def pack_upper_wind(runs: Path, dst: Path) -> dict:
    import upper_wind_regimes as u
    build = runs / "upper_wind" / "build"
    work = runs / "upper_wind" / "pack"
    work.mkdir(parents=True, exist_ok=True)
    arrays = {}
    seed = 0
    for specified in (True, False):
        for terrain3d in (False, True):
            for substeps in (1, 2, 4):
                seed += 1
                case = u.make_case(seed, specified, substeps, terrain3d)
                uw, vw = u.run_fortran(build, case, work)
                arrays[f"case{seed}/u"] = _xor(uw, case["u"])
                arrays[f"case{seed}/v"] = _xor(vw, case["v"])
    np.savez_compressed(dst / "upper_wind.npz", **arrays)
    return json.loads((build / "build.json").read_text())


#: The diff6 regimes kept in the fixture (1.5 MB for all six): convective
#: cells on gentle terrain, the steep ridge where the slope taper binds, and
#: the sharp edge structure the branch's halo read sees.  The run over all
#: six is in the lane's results.
FIXTURE_DIFF6_REGIMES = ("convective", "high_terrain", "edges")


def pack_diff6(runs: Path, dst: Path) -> dict:
    import diff6_regimes as d
    import diff6_fork_oracle as d6
    build = runs / "diff6" / "build"
    work = runs / "diff6" / "pack"
    work.mkdir(parents=True, exist_ok=True)
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor
    cfg = d._fork_cfg(20.0)
    arrays = {}
    seed = 0
    for regime in d.REGIMES:
        for slot, _array, st in d.ROWS:
            seed += 1
            if regime not in FIXTURE_DIFF6_REGIMES:
                continue
            c = d._case(regime, slot, st, seed)
            # The branch's own factor and step for this row (checked against
            # WOOF's in check_steps), so the fixture holds the branch's words.
            factor = float(np.float32(0.12 if slot in ("smag_ru", "smag_rv", "smag_rw",
                                                       "smag_rth", "smag_rtke") else 0.04))
            dt = float(np.float32(_diff6_dt(cfg, slot)))
            assert factor == float(np.float32(_diff6_factor(cfg, slot)))
            inp = work / "case.bin"
            d6._fortran_input(c, factor, dt, d.DX, d.DX, 0.05, inp)
            outp = work / "case.out"
            subprocess.run([str(build / "diff6_fork_driver"), str(inp), str(outp)],
                           check=True)
            want = np.fromfile(outp, np.float32).reshape(
                d.NY + 7, d.NZ + 1, d.NX + 7).transpose(1, 0, 2)
            nlev, nys, nxs = c["f"].shape
            arrays[f"{regime}/{slot}"] = np.ascontiguousarray(
                want[:nlev, 3:3 + nys, 3:3 + nxs])
    steps = d.check_steps(build)
    arrays["steps/dt"] = np.asarray(d.DTS, np.float32)
    arrays["steps/full"] = np.asarray([steps["full_steps"][str(float(np.float32(x)))]
                                       for x in d.DTS], np.float32)
    np.savez_compressed(dst / "diff6.npz", **arrays)
    return json.loads((build / "diff6-fork-build.json").read_text())


def pack_rrtmg(runs: Path, dst: Path) -> dict:
    sys.path.insert(0, str(ROOT / "tools" / "rrtmg_legacy_column_oracle"))
    from compare import PAIRS, PUBLISHED, read_records, to_cols
    cols = runs / "rrtmg" / "cols"
    meta = json.loads((cols / "regimes.json").read_text())
    fixture = ROOT / "tests" / "data" / "oracles" / "rrtmg_legacy" / "column"
    # The columns are the committed fixture's, word for word.
    for p in sorted(cols.glob("columns_*.npz")):
        a, b = np.load(p), np.load(fixture / p.name)
        assert sorted(a.files) == sorted(b.files), p.name
        for k in a.files:
            assert np.array_equal(np.asarray(a[k]).view(np.uint8),
                                  np.asarray(b[k]).view(np.uint8)), (p.name, k)
    out = {}
    for name in meta["regimes"]:
        rec = read_records(runs / "rrtmg" / "wrf" / "fork-cloud" / f"out_{name}.bin")
        for field in list(PAIRS) + list(PUBLISHED):
            if field in rec:
                out[f"{name}/{field}"] = np.ascontiguousarray(
                    to_cols(rec[field], meta["ncol"]), dtype=np.float32)
    np.savez_compressed(dst / "rrtmg_fork_cloud.npz", **out)
    b = runs / "rrtmg" / "build" / "fork-cloud"
    return {"build_receipt": (b / "build-receipt.txt").read_text().splitlines(),
            "units_receipt": json.loads((b / "units-receipt.json").read_text()),
            "cloud_form_receipt": json.loads((b / "cloud-form-receipt.json").read_text())}


def main():
    runs, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    ruc_cases = [Path(p) for p in sys.argv[3:]]
    dst.mkdir(parents=True, exist_ok=True)
    prov = {"zero_out": pack_zero_out(runs, dst), "albsol": pack_albsol(runs, dst),
            "upper_wind": pack_upper_wind(runs, dst), "diff6": pack_diff6(runs, dst),
            "rrtmg_fork_cloud": pack_rrtmg(runs, dst), "ruc": {}}
    for case in ruc_cases:
        name = "ruc_fork_" + case.name.replace("fork-", "").replace("-strict", "") + ".npz"
        subprocess.run([sys.executable, str(ROOT / "tools" / "ruc_lsm_gpu_oracle" / "pack_fixture.py"),
                        str(case), str(dst / name)], check=True)
        prov["ruc"][name] = {"case": case.name,
                             "compiler": (case / "compiler.txt").read_text().splitlines(),
                             "sha256": (case / "sha256.txt").read_text().splitlines()}
    prov["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(dst.glob("*.npz"))}
    (dst / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for p in sorted(dst.glob("*.npz")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
