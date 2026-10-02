#!/usr/bin/env python3
"""Count native limiter branches in a separate, word-neutral instrumented build.

The pinned WRF source is never modified. The coverage copy only adds integer
counter calls and writes those counts after the native routine returns. Every
result is compared byte for byte to the existing unmodified WRF output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


DEFS = ["-Dwrfmodel", "-DEM_CORE=1", "-DNMM_CORE=0", "-DRWORDSIZE=4", "-DIWORDSIZE=4", "-DDWORDSIZE=8", "-DLWORDSIZE=4"]
FLAGS = ["-cpp", "-ffree-form", "-ffree-line-length-none", "-fallow-argument-mismatch", "-O0", "-ffp-contract=off", "-fcheck=all", "-fbacktrace"]
PHYSICAL = "i>=its .and. i<=min(ite,ide-1) .and. j>=jts .and. j<=min(jte,jde-1)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument(source: str) -> str:
    for routine, names in (("advect_scalar_pd", "hit_pd"), ("advect_scalar_mono", "hit_mono_in,hit_mono_out")):
        pattern = rf"(^SUBROUTINE {routine}\b.*?^END SUBROUTINE {routine}\b)"
        match = re.search(pattern, source, flags=re.M | re.S | re.I)
        if match is None:
            raise ValueError(f"native routine {routine} not found")
        body = match.group(1)
        if body.count("   IMPLICIT NONE") != 1:
            raise ValueError(f"native implicit declaration changed for {routine}")
        body = body.replace("   IMPLICIT NONE", f"   USE advect_limiter_spy, ONLY: {names}\n   IMPLICIT NONE", 1)
        if routine.endswith("_pd"):
            anchor = "     IF( flux_out(i,k,j) .gt. ph_low(i,k,j) ) THEN"
            if body.count(anchor) != 1:
                raise ValueError("native PD limiter branch changed")
            body = body.replace(anchor, anchor+f"\n       CALL hit_pd({PHYSICAL})", 1)
        else:
            for direction, condition, assignment in (
                ("in", "flux_in .gt. ph_hi", "scale_in(i,k,j) = max(0.,ph_hi/(flux_in+eps))"),
                ("out", "flux_out .gt. ph_low", "scale_out(i,k,j) = max(0.,ph_low/(flux_out+eps))"),
            ):
                anchor = f"     IF( {condition} ) {assignment}"
                if body.count(anchor) != 1:
                    raise ValueError(f"native monotonic {direction} limiter branch changed")
                replacement = f"     IF( {condition} ) THEN\n       CALL hit_mono_{direction}({PHYSICAL})\n       {assignment}\n     END IF"
                body = body.replace(anchor, replacement, 1)
        source = source[:match.start()]+body+source[match.end():]
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pins = [row.split(maxsplit=1) for row in (here/"SOURCES.sha256").read_text().splitlines() if row and not row.startswith("#")]
    for expected, relative in pins:
        if sha256(source/relative) != expected:
            raise ValueError(f"WRF source pin changed: {relative}")
    native = source/"dyn_em/module_advect_em.F"
    instrumented = output/"module_advect_em_spy.F"
    instrumented.write_text(instrument(native.read_text()), encoding="ascii")
    harness = (here/"run_advect.F90").read_text()
    harness = harness.replace("  use iso_fortran_env", "  use advect_limiter_spy, only: write_limiter_counts\n  use iso_fortran_env", 1)
    harness = harness.replace("  close(output)", "  close(output)\n  call write_limiter_counts(trim(out_path)//'.limiter.json')", 1)
    caller = output/"run_advect_spy.F90"
    caller.write_text(harness, encoding="ascii")
    objects = []
    for path in (here/"stub_wrf.F90", here/"limiter_spy.F90", source/"share/module_model_constants.F",
                 source/"frame/module_wrf_error.F", instrumented, caller):
        target = path.stem+".o"
        subprocess.run(["nice", "-n", "10", "gfortran", "-c", *FLAGS, *DEFS, "-o", target, str(path)], cwd=output, check=True)
        objects.append(target)
    executable = output/"run_advect"
    subprocess.run(["nice", "-n", "10", "gfortran", "-o", str(executable), *objects], cwd=output, check=True)
    report = {
        "schema_version": 1,
        "method": "integer-only native limiter branch counters in separate instrumented source copy",
        "reference_source_sha256": sha256(native),
        "instrumented_source_sha256": sha256(instrumented),
        "caller_sha256": sha256(caller),
        "executable_sha256": sha256(executable),
        "cases": {},
    }
    inputs = sorted(args.inputs.glob("*.in.bin"))
    if not inputs:
        raise ValueError("no native stream inputs found")
    for path in inputs:
        stem = path.name[:-7]
        target = output/(stem+".out.bin")
        reference = path.with_name(stem+".out.bin")
        subprocess.run(["nice", "-n", "10", str(executable), str(path.resolve()), str(target)], check=True)
        same = target.read_bytes() == reference.read_bytes()
        if not same:
            raise AssertionError(f"integer counters changed native output: {stem}")
        counts = json.loads(target.with_name(target.name+".limiter.json").read_text())
        report["cases"][stem] = {"input_sha256": sha256(path), "reference_sha256": sha256(reference),
                                  "output_sha256": sha256(target), "all_output_words_equal": same, **counts}
        if sum(counts.values()):
            print(f"{stem}: {counts}", flush=True)
    report["input_count"] = len(inputs)
    report["all_output_words_equal"] = True
    report["branch_totals"] = {name: sum(case[name] for case in report["cases"].values())
                                for name in ("pd_all", "pd_physical", "mono_in_all", "mono_in_physical", "mono_out_all", "mono_out_physical")}
    (output/"limiter-coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="ascii")


if __name__ == "__main__":
    main()
