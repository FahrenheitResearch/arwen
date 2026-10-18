"""Float32 surface/PBL boundary comparison. Missing dumps fail explicitly."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def wrf_array(directory, tag, name, shape, *, step=1):
    ids, _, jds, _, kds, *_ = map(int, (directory / "dims.txt").read_text().split())
    raw = (directory / f"s{step:03d}_rk1_it00_{tag}__{name}.bin").read_bytes()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    a = np.frombuffer(raw[24:], "<f4").reshape((i1-i0+1, k1-k0+1, j1-j0+1), order="F")
    nk, nj, ni = (1, *shape) if len(shape) == 2 else shape
    a = a[ids-i0:ids-i0+ni, kds-k0:kds-k0+nk, jds-j0:jds-j0+nj].transpose(1, 2, 0)
    if len(shape) == 2:
        a = a[0]
    if a.shape != shape:
        raise ValueError(f"{tag}/{name}: WRF crop {a.shape}, WOOF {shape}")
    return np.ascontiguousarray(a)


def compare(a, b):
    if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape:
        raise ValueError("equal-shape float32 arrays required")
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    bad = a.view(np.uint32) != b.view(np.uint32)
    indices = np.argwhere(bad)
    first = tuple(map(int, indices[0])) if len(indices) else None
    return dict(words=int(a.size), different_words=int(bad.sum()),
                wrf_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                woof_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
                first_index=first,
                first_wrf_word=None if first is None else hex(int(a.view(np.uint32)[first])),
                first_woof_word=None if first is None else hex(int(b.view(np.uint32)[first])))


def chain_pairs(components=False):
    atmosphere = ("theta", "exner", "temperature", "pressure", "p_interface", "u", "v", "rho", "dz", "z_interface")
    surface = ("ust", "hfx", "qfx", "wspd", "br", "fm", "fh", "znt", "u10", "v10", "tsk", "psfc", "pblh", "xland", "mavail")
    for n in atmosphere:
        yield "phy_prep", "output", "p00_phy_prep", n, "p00_phy_prep", "x_" + n
    for stage, tags in (("sfclay", ("p10_sfclay_in", "p11_sfclay_out")), ("noah", ("p20_noah_in", "p21_noah_out")), ("ysu", ("p30_ysu_in", "p31_ysu_out"))):
        for direction, tag in zip(("input", "output"), tags):
            for n in surface:
                yield stage, direction, tag, n, tag, "x_sf_" + n
            if stage == "noah":
                for n in ("glw", "swdown", "albedo", "emiss", "snow", "snowh", "canwat",
                          "smois", "tslb", "sh2o"):
                    yield stage, direction, tag, n, tag, "x_sf_" + n
    for wn, on in dict(rublten="du", rvblten="dv", rthblten="dtheta", rqvblten="dqv", rqcblten="dqc").items():
        yield "ysu", "output", "p31_ysu_out", wn, "p31_ysu_rates", "x_" + on
    if components:
        for wn, on in (("rthblten", "rtheta"), ("rqvblten", "rqv"), ("rqcblten", "rqc")):
            yield "coupled_pbl", "output", "p40_mass_coupled", wn, "p40_coupled_pbl", "x_" + on
        yield "coupled_radiation", "output", "p40_mass_coupled", "rthraten", "p41_components", "x_radiation_rtheta"
        yield "coupled_cumulus", "output", "p40_mass_coupled", "rthcuten", "p41_components", "x_cumulus_rtheta"
    for wn, on in (("ru_tendf", "ru"), ("rv_tendf", "rv"), ("t_tendf", "rtheta")):
        yield "held_fold", "input", "c0_phy", wn, "c_phys", "x_" + on


def supported_scalar(a, *, specified):
    """Support of WRF add_a2a for a nonperiodic specified domain.

    calculate_phy_tend dumps unmasked mass-coupled rates. WOOF's strict
    components already carry add_a2a's one-cell support and are undivided.
    Keep interior words untouched, including signed zeros.
    """
    a = a.copy()
    if specified:
        a[:, (0, -1), :] = np.float32(0)
        a[:, :, (0, -1)] = np.float32(0)
    return a


def run(wd, od, chain=False, *, components=False, specified=False):
    points, missing = [], []
    pairs = chain_pairs(components) if chain else (("held_fold", "input", "c0_phy", wn, "c_phys", "x_"+on) for wn, on in (("ru_tendf", "ru"), ("rv_tendf", "rv"), ("t_tendf", "rtheta")))
    for stage, direction, wt, wn, ot, on in pairs:
        point = dict(stage=stage, direction=direction, field=wn, wrf_tag=wt, woof_tag=ot)
        try:
            files = sorted(od.glob(f"*_s001_*_{ot}.npz"))
            if len(files) != 1:
                raise ValueError(f"{ot}: expected one dump, found {len(files)}")
            with np.load(files[0], allow_pickle=False) as z:
                b = z[on]
            a = wrf_array(wd, wt, wn, b.shape)
            if stage in ("coupled_pbl", "coupled_radiation", "coupled_cumulus"):
                a = supported_scalar(a, specified=specified)
                point["quantity"] = "mass-coupled scalar contribution before map division"
                point["specified_support"] = specified
            point.update(compare(a, b))
            points.append(point)
        except (OSError, KeyError, ValueError) as e:
            missing.append(dict(point, error=str(e)))
    return dict(schema="gpuwm.surface-pbl-trace/v1", step=1, points=points, missing=missing,
                all_dumped_identical=bool(points) and not missing and all(p["different_words"] == 0 for p in points))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wrf_dump", type=Path)
    p.add_argument("woof_dump", type=Path)
    p.add_argument("--chain", action="store_true")
    p.add_argument("--components", action="store_true", help="compare PBL and radiation theta contributions separately (strict only)")
    p.add_argument("--specified", action="store_true", help="apply nonperiodic specified-domain add_a2a support to WRF components")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.components and not a.chain:
        p.error("--components requires --chain")
    r = run(a.wrf_dump, a.woof_dump, a.chain, components=a.components, specified=a.specified)
    if a.out:
        a.out.write_text(json.dumps(r, indent=2) + "\n")
    for point in r["points"]:
        print(f'{point["wrf_tag"]}/{point["field"]}: {point["different_words"]}/{point["words"]}', point["first_index"])
    for point in r["missing"]:
        print("MISSING", point["wrf_tag"], point["field"], point["error"])
    return 0 if r["all_dumped_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
