"""Grade carried SFIRE regimes against source-pinned corrected WRF words.

Run strict and default in separate processes. Missing fixtures, missing fields,
nonfinite words and any unequal output word fail closed. No tolerance is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.sfire_wrf471_oracle.fixture import words

REGIMES = ("grass_flat", "timber_slope", "wet_rain", "grass_central",
           "grass_sethian", "grass_split_negative", "grass_cap")
FIELDS = ("lfn", "tign", "ros", "fuel_frac", "fire_area", "fmc_g",
          "fgrnhfx", "fgrnqfx", "grnhfx", "grnqfx", "avg_fuel_frac",
          "rthfrten", "rqvfrten", "fmc_gc", "rh_fire", "fmc_equi", "fmc_lag")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def grade(actual, expected):
    result = words(actual, expected)
    result.update(actual_sha256=digest(actual), expected_sha256=digest(expected),
                  nonfinite_words=int(np.count_nonzero(~np.isfinite(actual))) +
                  int(np.count_nonzero(~np.isfinite(expected))))
    if result["different_words"]:
        actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
        indexes = np.argwhere(actual.view(np.uint32) != expected.view(np.uint32))[:4]
        result["first_differences"] = [dict(index=index.tolist(), actual=float(actual[tuple(index)]),
            expected=float(expected[tuple(index)]), actual_bits=int(actual.view(np.uint32)[tuple(index)]),
            expected_bits=int(expected.view(np.uint32)[tuple(index)])) for index in indexes]
    return result


def passed(result):
    return result["different_words"] == 0 and result["nonfinite_words"] == 0


def read_case(root, name, receipt):
    path = root / (name + ".npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["cases"][name]["sha256"]:
        raise ValueError(f"SFIRE regime fixture hash differs: {name}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_regime_fuel(regime, fuel):
    expected = 8 if regime == "timber_slope" else 1
    if not np.all(np.asarray(fuel) == np.float32(expected)):
        raise ValueError(f"SFIRE {regime} requires fuel category {expected}; native initialization changed it")


def replay(root, *, plant_one_ulp=False):
    import cupy as cp
    from types import SimpleNamespace
    from cupy.cuda import compiler
    from gpuwm.core.sfire import FireState, FireOptions
    from gpuwm.core.sfire_core import CoreOptions, IgnitionLine
    from gpuwm.core.sfire_moisture import MoistureState, advance_moisture
    from gpuwm.core.sfire_phys import weighted_moisture
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.core import sfire_core, sfire_phys, sfire_moisture, sfire_atm, sfire_coupler, chem_sfire

    # A receipt owns a cold compile capture even if a test process previously
    # used these modules. State arrays are created only after these clears.
    for module in (sfire_core, sfire_phys, sfire_moisture, sfire_atm, sfire_coupler, chem_sfire):
        module._module.cache_clear()

    receipt = json.loads((root / "receipt.json").read_text())
    if receipt["reference_kind"] != "explicitly corrected SFIRE regimes":
        raise ValueError("SFIRE regimes require the corrected reference, including defect cells")
    compiles = []
    final_compiles = []
    capturing = False
    native_compile = compiler.compile_using_nvrtc
    native_program_compile = compiler._NVRTCProgram.compile

    def capture_program(program, options=(), *args, **kwargs):
        if capturing:
            flags = [str(v) for v in options if "ftz=" in str(v)]
            if flags != ["--ftz=false"]:
                raise ValueError(f"Final SFIRE NVRTC flags flush subnormals: {flags}")
            final_compiles.append(list(options))
        return native_program_compile(program, options, *args, **kwargs)

    def capture(source, options=(), *args, **kwargs):
        nonlocal capturing
        planted = False
        if plant_one_ulp and 'extern "C" __global__ void sfire_heat_fluxes(' in source:
            anchor = "    dry_burnt[idx]=dmass*(1.0f-bmst);"
            if source.count(anchor) != 1:
                raise ValueError("SFIRE planted kernel anchor differs")
            source = source.replace(anchor, anchor +
                "\n    if(idx==983) hfx[idx]=nextafterf(hfx[idx],__int_as_float(0x7f800000));", 1)
            planted = True
        flags = [str(v) for v in options if "ftz=" in str(v)]
        if flags != ["--ftz=false"]:
            raise ValueError(f"SFIRE compile does not preserve subnormals: {flags}")
        capturing = True
        try:
            ptx, mapping = native_compile(source, options, *args, **kwargs)
        finally:
            capturing = False
        compiles.append(dict(options=list(options), planted_one_ulp=planted,
                             source_sha256=digest(np.frombuffer(source.encode(), dtype=np.uint8)),
                             ptx_sha256=digest(np.frombuffer(ptx.encode() if isinstance(ptx, str) else ptx, dtype=np.uint8))))
        return ptx, mapping

    compiler.compile_using_nvrtc = capture
    compiler._NVRTCProgram.compile = capture_program
    results = {}
    try:
        for regime in REGIMES:
            initial = read_case(root, f"regimes/{regime}/initial", receipt)
            xlo, xhi, ylo, yhi, xm, _, ym, _ = map(int, initial["bounds"])
            domain = (xlo-xm, xhi-xm, ylo-ym, yhi-ym)
            validate_regime_fuel(regime, initial["nfuel_cat"])
            moisture = MoistureState.zeros(initial["nfuel_cat"].shape)
            fmc = initial["fmc_g"]
            if regime == "wet_rain":
                first = read_case(root, f"regimes/{regime}/step1", receipt)
                advance_moisture(moisture, 0., *[first[k+"_in"] for k in
                    ("rainc", "rainnc", "t2", "q2", "psfc")], initialize=True)
                fmc = weighted_moisture(moisture.fmc_gc, initial["nfuel_cat"])
            state = FireState.from_static(
                *[initial[k] for k in ("nfuel_cat", "zsf", "dzdxf", "dzdyf")],
                float(initial["dx"]), float(initial["dy"]), domain=domain,
                coord_xf=initial["coord_xf"], coord_yf=initial["coord_yf"],
                options=FireOptions(core=CoreOptions(upwinding=int(initial["core_upwinding"]),
                    upwind_split=int(initial["core_split"])), fire_fmc_read=0,
                    fmoist_run=regime == "wet_rain", fmoist_interp=regime == "wet_rain", fmoist_freq=1),
                fmc_g=fmc, moisture_state=moisture,
                ignitions=[IgnitionLine(*map(float, initial["ignition"]))])
            # Winds and static inputs are prescribed. No reference-produced
            # level set, fuel parameters or moisture state seed this replay.
            state.data.update(vx=cp.asarray(initial["vx"]), vy=cp.asarray(initial["vy"]))
            initialization = {name: grade(cp.asnumpy(state.data[name][state.interior]),
                initial[name][state.interior]) for name in
                ("lfn", "tign", "fuel_frac", "fire_area", "fmc_g", "fgip", "ischap",
                 "bbb", "betafl", "phiwc", "r_0", "iboros", "fuel_time")}
            initialization.update({"moisture."+name: grade(cp.asnumpy(value), initial[name])
                                   for name, value in moisture.arrays().items()})
            # Initialization is native and independently graded below. The next
            # moisture advance starts at t=0, with zero elapsed moisture time.
            state.moisture_initialized = regime == "wet_rain"
            ny, nx = initial["mu"].shape
            coupler = FireCoupler.__new__(FireCoupler)
            coupler.grid = state
            coupler.sr_x = coupler.sr_y = 2
            coupler.coarse_shape = (ny, nx)
            coupler.data = {k: cp.zeros((ny, nx), cp.float32) for k in
                ("grnhfx", "grnqfx", "grnhfx_fu", "grnqfx_fu", "avg_fuel_frac", "canhfx", "canqfx")}
            cfg = SimpleNamespace(fire_atm_feedback=1.0, fire_ext_grnd=60., fire_ext_crwn=90.,
                fire_crwn_hgt=20., fire_sfc_flx=1, fire_heat_peak=32., fire_tg_ub=1000.)
            atmosphere = {k: cp.asarray(initial[v]) for k, v in
                (("z_interface", "z_at_w"), ("dz", "dz8w"), ("rho", "rho"))}
            atm_state = SimpleNamespace(ht=cp.asarray(initial["terrain"]),
                c1h=cp.asarray(initial["c1h"]), c2h=cp.asarray(initial["c2h"]),
                total_mu=lambda: cp.asarray(initial["mu"]))
            steps = []
            for step in range(1, 21):
                expected = read_case(root, f"regimes/{regime}/step{step}", receipt)
                surface = {k: expected[k+"_in"] for k in ("rainc", "rainnc", "t2", "q2", "psfc")}
                state.advance(float(initial["dt"]), surface=surface)
                coupler._feedback(atm_state, cfg, atmosphere)
                actual = {**state.data, **state.moisture.arrays(), **coupler.data}
                grades = {}
                for field in FIELDS:
                    if field not in expected or field not in actual:
                        raise ValueError(f"SFIRE required field missing: {regime}/{step}/{field}")
                    value = cp.asnumpy(actual[field])
                    target = expected[field]
                    if field in state.data:
                        value, target = value[state.interior], target[state.interior]
                    grades[field] = grade(value, target)
                steps.append(dict(step=step, fields=grades,
                    cfl_bound=state.last_cfl_bound, cfl_exceeded=state.last_cfl_exceeded))
            last = expected
            results[regime] = dict(initialization=initialization, steps=steps, configuration=dict(
                fuel_category=float(initial["nfuel_cat"][state.interior][0, 0]),
                wind=[float(initial[k][state.interior][0, 0]) for k in ("vx", "vy")],
                slope=[float(initial[k][state.interior][0, 0]) for k in ("dzdxf", "dzdyf")],
                dt=float(initial["dt"]), upwinding=int(initial["core_upwinding"]),
                upwind_split=int(initial["core_split"])), activity=dict(
                initial_moisture=float(initial["fmc_g"][state.interior].mean()),
                final_moisture=float(last["fmc_g"][state.interior].mean()),
                rain_accumulation=float(last["rainnc_in"].max()),
                final_fuel_min=float(last["fuel_frac"][state.interior].min()),
                final_ros_max=float(last["ros"][state.interior].max()),
                final_heat_max=float(last["grnhfx"].max())))
        # Compile the smoke source module through its production entry point
        # too. Its tracer behavior remains in the existing smoke regressions.
        chem_sfire._module(cp.cuda.Device().id)
    finally:
        compiler.compile_using_nvrtc = native_compile
        compiler._NVRTCProgram.compile = native_program_compile
    ok = all(passed(g) for r in results.values() for s in r["steps"] for g in s["fields"].values())
    ok = ok and all(passed(g) for r in results.values() for g in r["initialization"].values())
    summary = {}
    for name, regime in results.items():
        summary[name] = {field: dict(max_ulp=max(s["fields"][field]["max_ulp"] for s in regime["steps"]),
            different_words=sum(s["fields"][field]["different_words"] for s in regime["steps"]),
            words=sum(s["fields"][field]["words"] for s in regime["steps"])) for field in FIELDS}
    return dict(status="PASS" if ok else "FAIL", regimes=results, compiles=compiles,
                final_nvrtc_options=final_compiles,
                summary=summary,
                planted_one_ulp=plant_one_ulp,
                fixture_receipt_sha256=hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
                reference={k: v for k, v in receipt.items() if k != "cases"},
                mode="strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("both", "strict", "default"), default="both")
    parser.add_argument("--plant-one-ulp", action="store_true",
                        help="Negative control: compile one ULP into one heat-flux kernel cell")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "both":
        modes = {}
        for mode in ("strict", "default"):
            output = args.out.with_name(args.out.stem + "-" + mode + ".json")
            env = dict(os.environ)
            env.pop("GPUWM_WRF_EXACT", None)
            if mode == "strict":
                env["GPUWM_WRF_EXACT"] = "1"
            command = [sys.executable, "-m", "tools.sfire_wrf471_oracle.regime_receipt",
                "--fixtures", str(args.fixtures), "--out", str(output), "--mode", mode]
            if args.plant_one_ulp:
                command.append("--plant-one-ulp")
            run = subprocess.run(command, env=env)
            modes[mode] = json.loads(output.read_text())
            modes[mode]["exit_code"] = run.returncode
        result = dict(status="PASS" if all(v["status"] == "PASS" and v["exit_code"] == 0
                                           for v in modes.values()) else "FAIL", modes=modes)
    else:
        try:
            expected_strict = args.mode == "strict"
            if (os.environ.get("GPUWM_WRF_EXACT") == "1") != expected_strict:
                raise ValueError("SFIRE mode and GPUWM_WRF_EXACT disagree")
            result = replay(args.fixtures, plant_one_ulp=args.plant_one_ulp)
        except Exception as exc:
            result = dict(status="FAIL", error=f"{type(exc).__name__}: {exc}", mode=args.mode)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], receipt=str(args.out))))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
