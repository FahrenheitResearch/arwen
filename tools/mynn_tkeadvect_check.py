"""Mechanical scalar-TKE qualification. Run on a sprint box under the GPU queue."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def oracle(csv, fork=False, mixlength=1, asis=False):
    import cupy as cp
    import _mynn_families as F
    import _mynn_families_gsd41 as G
    from tools.mynn_exact_review import load
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    family = G if fork else F
    results = []
    for step, blocks in load(csv).items():
        if fork:
            values = G.driver_values(G.step_inputs(blocks), np.zeros((6, 30), np.float32))
        else:
            values = F.step_inputs(blocks)
        values["qke_adv"] = F._f32(blocks, "qke_adv_in")
        options = dict(bl_mynn_tkeadvect=True, bl_mynn_mixlength=mixlength,
                       flag_qs=not fork)
        if fork:
            options.update(bl_mynn_version="gsd_41", bl_mynn_cloud_tendency_form="gsd_41",
                           bl_mynn_gsd41_unsquared_qtke=asis)
        actual = mynn_bl_driver_cuda({k: cp.asarray(v) for k, v in values.items()},
                    initflag=int(blocks[0][0]["initflag"]),
                    delt=np.float32(blocks[0][0]["delt"]), **options)
        expected = family.step_outputs(blocks)
        expected["qke_adv"] = F._f32(blocks, "qke_adv")
        for name, want in expected.items():
            got = cp.asnumpy(actual[name])
            assert got.shape == want.shape and got.dtype == want.dtype, name
            differences = int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32)))
            ulp = int(fp32_ulp_distance(got, want).max()) if want.dtype == np.float32 else 0
            results.append(dict(step=step, field=name, words=want.size,
                                different_words=differences, max_ulp=ulp))
    return dict(csv_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
                fork=fork, mixlength=mixlength, asis=asis, readings=results,
                words=sum(r["words"] for r in results),
                different_words=sum(r["different_words"] for r in results),
                max_ulp=max(r["max_ulp"] for r in results))


def transport(executable, folder, *, full_rk=False):
    """One final RK scalar stage on a 3-D patch, from compiled WRF PD tendency."""
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    from gpuwm.core.moist import advance_scalars_stage
    from gpuwm.verify.advect_oracle import (load_advect_cases, write_fortran_input,
        read_fortran_output, measure_words, defined_output_mask)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    case = next(c for c in load_advect_cases() if c.name == "real_interior")
    arrays = {name: value.copy() for name, value in case.inputs.items()}
    # The source update adds no scalar sources in this check.
    arrays["tend_pd"].fill(0.)
    case = replace(case, inputs=arrays)
    inp, out = folder / "pd.in.bin", folder / "pd.out.bin"
    write_fortran_input(case, "advect_scalar_pd", inp)
    subprocess.run([str(executable), str(inp), str(out)], check=True)
    reference = read_fortran_output(case, out, diagnostic=True)["tendency"]
    nz, ny, nx = case.shape
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=case.metadata["dx"], dy=case.metadata["dy"],
                    ztop=16000., dt=case.metadata["dt"], run_seconds=0.,
                    moist=False, bl_pbl_physics=5, bl_mynn_tkeadvect=True)
    state = DomainState(cfg)
    for name in ("c1h", "c2h", "rdnw", "rdn", "fnm", "fnp"):
        setattr(state, name, cp.asarray(arrays[name]))
    state.mub2d[...] = cp.asarray(arrays["mub"])
    state.mup0[...] = cp.asarray(arrays["mu_old"] - arrays["mub"])
    state.mup[...] = cp.asarray(arrays["muts"] - arrays["mub"])
    state.msft[...] = cp.asarray(arrays["msftx"])
    state.has_msf = True
    state.qke_adv[...] = cp.asarray(arrays["scalar_pd"])
    state.qke_adv0[...] = cp.asarray(arrays["q0"])
    advance_scalars_stage(state, cfg, *(cp.asarray(arrays[n]) for n in ("ru", "rv", "rw")),
                          cfg.dt, final=True)
    # WRF rk_update_scalar: multiply tendency by msfty, coupled multiply,
    # add, then divide by the new mass, with one float32 round per operation.
    tend = reference[:nz, 4:ny+4, 4:nx+4]
    c1, c2 = arrays["c1h"][:, None, None], arrays["c2h"][:, None, None]
    old_mass = c1 * arrays["mu_old"][None] + c2
    new_mass = c1 * arrays["muts"][None] + c2
    want = np.maximum((old_mass * arrays["q0"] + np.float32(cfg.dt) *
                       (tend * arrays["msfty"][None])) / new_mass, np.float32(0.))
    got = cp.asnumpy(state.qke_adv)
    metric = measure_words(got, want)
    stages = []
    if full_rk:
        want = arrays["q0"].copy()
        state.qke_adv[...] = cp.asarray(want)
        for stage, fraction in enumerate((1. / 3., .5, 1.), start=1):
            dt_stage = cfg.dt * fraction
            inputs = dict(arrays, scalar=want.copy(), scalar_pd=want.copy(),
                          tend_scalar=np.zeros_like(want), tend_pd=np.zeros_like(want))
            staged_case = replace(case, metadata=dict(case.metadata, dt=dt_stage), inputs=inputs)
            routine = "advect_scalar_pd" if stage == 3 else "advect_scalar"
            inp, out = folder / f"rk-{stage}.in.bin", folder / f"rk-{stage}.out.bin"
            write_fortran_input(staged_case, routine, inp)
            with inp.open("r+b") as stream:
                stream.seek(20 * 4)  # native stream header's rk_step
                stream.write(np.asarray([stage], dtype="<i4").tobytes())
            subprocess.run([str(executable), str(inp), str(out)], check=True)
            tend = read_fortran_output(staged_case, out, diagnostic=stage == 3)["tendency"][:nz, 4:ny+4, 4:nx+4]
            want = (old_mass * arrays["q0"] + np.float32(dt_stage) *
                    (tend * arrays["msfty"][None])) / new_mass
            if stage == 3:
                want = np.maximum(want, np.float32(0.))
            advance_scalars_stage(state, cfg, *(cp.asarray(arrays[n]) for n in ("ru", "rv", "rw")),
                                  dt_stage, final=stage == 3)
            got = cp.asnumpy(state.qke_adv)
            stage_metric = measure_words(got, want)
            stages.append(dict(stage=stage, dt=dt_stage, **stage_metric))
        metric = measure_words(got, want)
    planted = got.copy()
    at = tuple(np.argwhere(planted > 0.)[0])
    planted.view(np.uint32)[at] += np.uint32(1)
    negative = measure_words(planted, want)
    return dict(shape=list(got.shape), words=got.size, strict=metric, rk_stages=stages,
                planted_one_ulp=negative,
                expected_sha256=hashlib.sha256(want.tobytes()).hexdigest(),
                actual_sha256=hashlib.sha256(got.tobytes()).hexdigest())


def field_hashes(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: hashlib.sha256(data[name].tobytes()).hexdigest()
                for name in data.files if name != "__gpuwm_restart_header__"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--csv", type=Path)
    p.add_argument("--fork", action="store_true")
    p.add_argument("--mixlength", type=int, default=1)
    p.add_argument("--asis", action="store_true")
    p.add_argument("--advect-exe", type=Path)
    p.add_argument("--scratch", type=Path)
    p.add_argument("--full-rk", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = transport(a.advect_exe, a.scratch, full_rk=a.full_rk) if a.advect_exe else oracle(a.csv, a.fork, a.mixlength, a.asis)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if a.advect_exe:
        assert result["strict"]["different_words"] == 0, result["strict"]
        assert all(stage["different_words"] == 0 for stage in result["rk_stages"]), result["rk_stages"]
        assert result["planted_one_ulp"]["different_words"] == 1
        assert result["planted_one_ulp"]["max_ulp"] == 1
    else:
        assert result["different_words"] == 0, result
    print(a.out, flush=True)


if __name__ == "__main__":
    main()
