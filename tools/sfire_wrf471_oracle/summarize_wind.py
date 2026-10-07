"""Grade every full wind output and write a numerical parity receipt."""
from pathlib import Path
import argparse
import hashlib
import json

import cupy as cp
import numpy as np

from gpuwm.core import sfire_wind
from tools.sfire_wrf471_oracle.fixture import words


def grade(destination):
    root = Path(__file__).parent / "fixtures/wind"
    reference = json.loads((root / "receipt.json").read_text())
    cases = {}
    for name, pinned in reference["cases"].items():
        path = root / (name + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != pinned["sha256"]:
            raise ValueError(f"native wind corpus differs from its receipt: {name}")
        with np.load(path, allow_pickle=False) as archive:
            f = dict(archive)
        xl, xh, yl, yh = map(int, f["domain"])
        fl, fh, fb, ft = map(int, f["fine_domain"])
        options = dict(fire_wind_height=float(f["fire_wind_height"]),
                       fire_lsm_zcoupling=bool(f["fire_lsm_zcoupling"]),
                       fire_lsm_zcoupling_ref=float(f["fire_lsm_zcoupling_ref"]))
        driver = name.startswith("wind/drivercompact_")
        compact = driver or name.startswith("wind/compact_")
        if compact:
            profile = int(name.rsplit("_", 1)[1]) >= 7
            xe, ye = xh + int(driver), yh + int(driver)
            result = sfire_wind.interpolate_native_atm2fire(
                f["u"][:-1, yl:ye, xl:xe+1], f["v"][:-1, yl:ye+1, xl:xe],
                f["ph"][:, yl:ye, xl:xe], f["phb"][:, yl, xl] if profile else f["phb"][:, yl:ye, xl:xe],
                f["z0"][yl:ye, xl:xe], f["zs"][yl:ye, xl:xe], f["z0f"],
                int(f["sr_x"]), int(f["sr_y"]), return_staggered_diagnostics=True,
                domain_includes_terminal_face=not driver, **options)
        else:
            options.update(domain=(xl, xh, yl, yh), fine_domain=(fl, fh, fb, ft))
            if "tiles" in f:
                options.update(tiles=tuple(map(tuple, f["tiles"])), fine_tiles=tuple(map(tuple, f["fine_tiles"])))
            result = sfire_wind.interpolate_atm2fire(
                f["u"][:-1], f["v"][:-1], *(f[k] for k in ("ph", "phb", "z0", "zs", "z0f")),
                int(f["sr_x"]), int(f["sr_y"]), **options)
        grades = {}
        for key in ("uf", "vf"):
            grades[key] = words(cp.asnumpy(result[key][fb:ft+1, fl:fh+1]), f[key][fb:ft+1, fl:fh+1])
        for key in ("uah", "vah"):
            actual = result[key] if compact else result[key][yl:yh+2, xl:xh+2]
            grades[key] = words(cp.asnumpy(actual), f[key + "_staggered"] if compact else f[key])
        cases[name] = grades
    outputs = [entry for case in cases.values() for entry in case.values()]
    summary = dict(cases=len(cases), graded_words=sum(g["words"] for g in outputs),
                   different_words=sum(g["different_words"] for g in outputs),
                   max_ulp=max(g["max_ulp"] for g in outputs),
                   nonfinite_differences=sum(g["nonfinite_differences"] for g in outputs))
    receipt = dict(reference="byte-extracted original WRF v4.7.1 interpolate_atm2fire",
                   reference_corpus_sha256=hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
                   device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                   cupy=cp.__version__, module_options=sfire_wind.MODULE_OPTIONS,
                   source_sha256=hashlib.sha256(sfire_wind.module_source().encode()).hexdigest(),
                   summary=summary, cases=cases)
    Path(destination).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary))
    if summary["different_words"]:
        raise SystemExit("wind replay differs from the native routine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    args = parser.parse_args()
    grade(args.destination)
