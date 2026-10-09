"""Grade independently generated MYNN family CSVs, keeping small receipts.

Run only under the GPU mutex. The Fortran harnesses generate the CSVs on the
same host; this tool performs numerical analysis and CUDA driver orchestration.
It does not read weather input files or render weather fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import _mynn_families as F
import _mynn_families_gsd41 as G


def load(path):
    steps = {}
    with Path(path).open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            families = steps.setdefault(int(row["step"]), {})
            families.setdefault(row["case"], []).append(row)
    return {step: [families[name] for name in F.FAMILIES]
            for step, families in steps.items()}


def digest(arrays):
    out = {}
    for name, value in arrays.items():
        out[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return out


def summary(module, results):
    bits = module.bit_mismatch_table(results)
    ulps = module.max_ulp_table(results)
    nonfinite = {}
    words = 0
    for _, actual, expected in results:
        for name, want in expected.items():
            words += want.size
            n = int(np.count_nonzero(~np.isfinite(actual[name])))
            if n:
                nonfinite[name] = nonfinite.get(name, 0) + n
    first = []
    for step, actual, expected in results:
        for name, want in expected.items():
            got = np.ascontiguousarray(actual[name]).view(np.uint32)
            want_words = np.ascontiguousarray(want).view(np.uint32)
            indices = np.argwhere(got != want_words)
            if indices.size:
                at = tuple(indices[0])
                first.append(dict(step=step, field=name, index=list(map(int, at)),
                                  got=f"{int(got[at]):08x}",
                                  want=f"{int(want_words[at]):08x}"))
        if first:
            break
    return dict(steps=len(results), words=words,
                mismatches={name: sum(v) for name, v in bits.items() if any(v)},
                max_ulp={name: max(v) for name, v in ulps.items()},
                nonfinite=nonfinite, first=first,
                last_actual_sha256=digest(results[-1][1]),
                last_expected_sha256=digest(results[-1][2]))


def grade(args):
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    steps = load(args.csv)
    module = G if args.fork else F

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda({n: cp.asarray(v) for n, v in values.items()},
                                   **kwargs)

    options = dict(bl_mynn_mixlength=args.mixlength)
    if args.fork:
        options.update(asis=args.as_written,
                       bl_mynn_cloud_tendency_form="gsd_41", cycling=args.cycling)
    rows = {}
    for replay in (True, False):
        result = module.integrate(driver, **options, steps=steps,
                                  replay=replay, to_host=cp.asnumpy) if args.fork else (
            module.integrate(driver, args.mixlength, steps=steps,
                             replay=replay, to_host=cp.asnumpy))
        rows["replay" if replay else "free"] = summary(module, result)
    return dict(csv=Path(args.csv).name,
                csv_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                options=options, results=rows)


def mutation(args):
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as D
    from gpuwm.core.kernels import module_options, module_source
    source = module_source("mynn_pbl")
    old = "        dqs[idx] = 0.0f;"
    new = "        dqs[idx] = __uint_as_float(0x00000001u);"
    assert source.count(old) == 1
    changed = source.replace(old, new)
    module = cp.RawModule(code=changed, options=module_options("mynn_pbl"))
    original = D.mynn_pbl_kernel
    D.mynn_pbl_kernel = lambda name, version="wrf_461": (
        module.get_function(name) if version == "wrf_461" else original(name, version))

    def driver(values, **kwargs):
        return D.mynn_bl_driver_cuda({n: cp.asarray(v) for n, v in values.items()},
                                    **kwargs)

    rows = []
    for length in (1, 2):
        for replay in (True, False):
            results = F.integrate(driver, length, replay=replay, to_host=cp.asnumpy)
            grade = summary(F, results)
            # This reconstructs the base gate's actual output selection.
            old_gate = {n: max(v) for n, v in F.max_ulp_table(results).items()
                        if n not in ("rqsblten", "dozone") and max(v)}
            rows.append(dict(mixlength=length, replay=replay,
                             base_gate_failures=old_gate, revised_gate=grade))
            assert not old_gate, old_gate
            assert grade["mismatches"].get("rqsblten", 0) > 0
    import pytest
    test_exit = pytest.main([
        str(Path(__file__).resolve().parents[1] / "tests" /
            "test_mynn_wrf461_exact_gpu.py"),
        "-k", "device", "-q", "--tb=short"])
    assert test_exit == pytest.ExitCode.TESTS_FAILED, test_exit
    return dict(mutation="snow tendency: 00000000 -> 00000001",
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                mutated_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                revised_pytest_exit=int(test_exit), results=rows)


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fork", action="store_true")
    parser.add_argument("--as-written", action="store_true")
    parser.add_argument("--cycling", action="store_true")
    parser.add_argument("--mixlength", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mutation", action="store_true")
    args = parser.parse_args()
    receipt = mutation(args) if args.mutation else grade(args)
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(args.out, flush=True)
    if not args.mutation and any(row["mismatches"] or row["nonfinite"]
                                 for row in receipt["results"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
