"""Compiled WRF v4.7.1 ``cal_deform_and_div`` at both ``mix_full_fields`` values.

Operational HRRR runs ``mix_full_fields = .false.`` (the Registry default;
parm/conus/hrrr_wrf.nl omits the key).  Under ``diff_opt = 2`` with a PBL
scheme WRF reaches that branch only in ``cal_deform_and_div``, where du/dz
and dv/dz become ``u - u_base`` and ``v - v_base``
(dyn_em/module_diffusion_em.F:842-860 and :1017-1035, byte-identical
between the NOAA-EMC/HRRR v4.1.21 fork's WRFV3.9 and the pinned v4.7.1).
real.exe never assigns the 1-D base-state wind profiles and WRF's
allocation zero-fills them, so for a real-data run the two branches are
the same arithmetic.  This tool measures that claim on the compiled WRF
routine itself, on the same real-state fixtures the deformation parity
gate uses, without touching the pinned deformation oracle files.

``build``: compile the oracle with ``mix_full_fields_wrappers.F90``, which
adds ``oracle_deform_mix`` (the flag and the two profiles as arguments) to
the pinned wrapper set.  The four WRF sources must be the pinned v4.7.1
bytes (``build_common.SOURCE_PINS``).

``probe``: for every ``deformation-*.npz`` fixture, run the compiled
``cal_deform_and_div`` three ways and compare every tensor word against
the fixture's recorded reference (captured with ``.true.``):

* ``.true.`` with zero profiles: the sanity arm, must equal the reference;
* ``.false.`` with zero profiles: the real-data arm (operational HRRR);
* ``.false.`` with the level-mean wind as the profile: the arm that
  shows the branch is live when a profile is not zero.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_common import build, sha256  # noqa: E402
from cases import pad2, pad3  # noqa: E402
from deformation_build import ROUTINES  # noqa: E402
from deformation_reference import _invoke  # noqa: E402

WRAPPER = HERE / "mix_full_fields_wrappers.F90"
TENSORS = ("div", "d11", "d22", "d33", "d12", "d13", "d23")
STAGGER = {"div": 0, "d11": 0, "d22": 0, "d33": 0, "d12": 4, "d13": 5, "d23": 6}


def build_oracle(source_dir: Path, output: Path) -> Path:
    source_dir = Path(source_dir)
    return build(source_dir / "module_diffusion_em.F",
                 source_dir / "module_model_constants.F", WRAPPER, output,
                 ROUTINES,
                 extra_sources={source_dir / "module_bc.F": ["set_physical_bc3d"],
                                source_dir / "module_big_step_utilities_em.F": ["phy_prep"]},
                 module_declarations="   INTEGER, PARAMETER            :: bdyzone = 4\n")


def _fixture_slice(key, value, nx, ny, nz):
    if key == "d13":
        return np.ascontiguousarray(value[3:3 + nx + 1, :nz + 1, 3:3 + ny].transpose(1, 2, 0))
    if key == "d23":
        return np.ascontiguousarray(value[3:3 + nx, :nz + 1, 3:3 + ny + 1].transpose(1, 2, 0))
    if key == "d12":
        return np.ascontiguousarray(value[3:3 + nx + 1, :nz, 3:3 + ny + 1].transpose(1, 2, 0))
    return np.ascontiguousarray(value[3:3 + nx, :nz, 3:3 + ny].transpose(1, 2, 0))


def deform_arm(lib, arrays, meta, *, mix: int, u_base, v_base):
    """WRF-padded inputs through metrics, physics prep and the deformation."""
    nx, ny, nz, bx, by = [int(meta[k]) for k in ("nx", "ny", "nz", "bx", "by")]
    head = [nx, ny, nz, bx, by]
    shape = (nx + 6, nz + 1, ny + 6)
    out = {k: pad3(arrays[k], nx, ny, nz, bx, by) for k in
           ("u", "v", "w", "php", "phb", "p", "alt", "thp")}
    out.update({k: pad2(arrays[k], nx, ny, bx, by) for k in ("msfu", "msfv", "msft", "mut")})
    for k in ("dn", "dnw", "fnm", "fnp", "fzm", "fzp", "znw", "c1h", "c2h", "c1f", "c2f"):
        f = np.asarray(arrays.get(k, arrays[{"fzm": "fnm", "fzp": "fnp"}.get(k, k)]), dtype=np.float32)
        out[k] = np.asfortranarray(f[np.minimum(np.arange(nz + 1), f.size - 1)])
    moist = np.zeros((*shape, 4), dtype=np.float32, order="F")
    for slot, key in enumerate(("qv", "qc", "qi"), start=1):
        moist[:, :, :, slot] = pad3(arrays.get(key, np.zeros_like(arrays["alt"])), nx, ny, nz, bx, by)
    out["moist"] = moist
    for key in ("z", "rdz", "rdzw", "zx", "zy", "rho", "theta", "temp", "p8w", "t8w", "zw", *TENSORS):
        out[key] = np.zeros(shape, dtype=np.float32, order="F")
    profile = {}
    for key, values in (("ub", u_base), ("vb", v_base)):
        column = np.zeros(nz + 1, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        column[:min(values.size, nz + 1)] = values[:nz + 1]
        profile[key] = np.asfortranarray(column)

    def pointers(*keys):
        return [out[k] for k in keys]
    rdx, rdy = [float(np.float32(1.0 / meta[k])) for k in ("dx", "dy")]
    cf = [float(meta[k]) for k in ("cf1", "cf2", "cf3")]
    _invoke(lib, "oracle_metrics", head + pointers("php", "phb") + [rdx, rdy]
            + pointers("z", "rdz", "rdzw", "zx", "zy"))
    _invoke(lib, "oracle_phy", head + pointers("u", "v", "p", "alt", "php", "phb", "thp", "moist", "mut",
                                               "c1h", "c2h", "c1f", "c2f", "dnw", "fzm", "fzp", "znw")
            + [float(meta.get("p_top", 5000.))] + pointers("rho", "theta", "temp", "p8w", "t8w", "z", "zw"))
    _invoke(lib, "oracle_deform_mix", head + [int(mix), profile["ub"], profile["vb"]]
            + pointers("u", "v", "w", "msfu", "msfv", "msft", "rdz", "rdzw", "zx", "zy", "dn", "dnw", "fnm", "fnp")
            + [rdx, rdy, *cf] + pointers(*TENSORS))
    for key in TENSORS:
        _invoke(lib, "oracle_bc", head + [STAGGER[key], out[key]])
    return {key: _fixture_slice(key, out[key], nx, ny, nz) for key in TENSORS}


def _words(got, expected):
    a = np.ascontiguousarray(got, dtype=np.float32)
    b = np.ascontiguousarray(expected, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape {a.shape} versus {b.shape}")
    unequal = a.view(np.uint32) != b.view(np.uint32)
    signed_zero = unequal & (a == 0) & (b == 0)
    return {"words": int(a.size), "different_words": int(unequal.sum()),
            "signed_zero_only": int(signed_zero.sum()),
            "max_absolute": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))}


def level_mean_profiles(arrays):
    """A nonzero profile from the fixture itself: the level mean of u and v."""
    u = np.asarray(arrays["u"], dtype=np.float32)
    v = np.asarray(arrays["v"], dtype=np.float32)
    return (u.mean(axis=(1, 2)).astype(np.float32), v.mean(axis=(1, 2)).astype(np.float32))


def probe(library: Path, fixtures: Path):
    lib = ctypes.CDLL(str(library))
    result = {}
    for fixture in sorted(Path(fixtures).glob("deformation-*.npz")):
        with np.load(fixture) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
            reference = {key: data[f"ref__km4_iso0__{key}"] for key in TENSORS}
        nz = int(meta["nz"])
        zero = np.zeros(nz + 1, dtype=np.float32)
        ub, vb = level_mean_profiles(arrays)
        arms = {
            "true_zero_profile": deform_arm(lib, arrays, meta, mix=1, u_base=zero, v_base=zero),
            "false_zero_profile": deform_arm(lib, arrays, meta, mix=0, u_base=zero, v_base=zero),
            "false_level_mean_profile": deform_arm(lib, arrays, meta, mix=0, u_base=ub, v_base=vb),
        }
        result[fixture.name] = {
            arm: {key: _words(value[key], reference[key]) for key in TENSORS}
            for arm, value in arms.items()}
        result[fixture.name]["profile"] = {
            "u_base_max_abs": float(np.max(np.abs(ub))), "v_base_max_abs": float(np.max(np.abs(vb)))}
        print(fixture.name,
              {arm: {k: v["different_words"] for k, v in rows.items()}
               for arm, rows in result[fixture.name].items() if arm != "profile"}, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="compile the oracle with the mix_full_fields wrapper")
    b.add_argument("source_dir", type=Path, help="folder holding the four pinned v4.7.1 sources")
    b.add_argument("output", type=Path)
    p = sub.add_parser("probe", help="run every fixture at both flag values")
    p.add_argument("library", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build_oracle(args.source_dir, args.output))
        return 0
    measured = probe(args.library, args.fixtures)
    receipt = {"library_sha256": sha256(args.library), "wrapper_sha256": sha256(WRAPPER),
               "fixtures": str(args.fixtures), "cases": measured}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = {}
    for arm in ("true_zero_profile", "false_zero_profile", "false_level_mean_profile"):
        summary[arm] = {key: sum(case[arm][key]["different_words"] for case in measured.values())
                        for key in TENSORS}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
