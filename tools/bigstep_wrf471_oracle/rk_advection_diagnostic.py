"""Attribute RK scalar differences without editing any forecast kernel.

The transient modules change only arithmetic evaluation order and contraction.
They are diagnostic controls, never replacements for the production launch.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cupy as cp
import gpuwm.core.advection as advection
from gpuwm.core.kernels import module_source
from gpuwm.verify.bigstep_momentum_oracle import measure_momentum_parity
from gpuwm.verify.bigstep_rk_oracle import load_rk_tendency_oracle, replay_rk_tendency_case


def replace_body(source, name, body):
    start = source.index("real " + name + "(")
    first = source.index("{", start)
    end = source.index("\n}", first) + 2
    return source[:first] + "{\n" + body + "\n}" + source[end:]


def wrf_flux_order(source):
    source = replace_body(source, "flux5", """    real central = __fdiv_rn(37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3), 60.0f);
    real dissipative = (qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1);
    real sign_velocity = copysignf(1.0f, vel);
    return vel * (central - __fdiv_rn(sign_velocity * dissipative, 60.0f));""")
    for name, sign in (("flux3", "-vel"), ("flux3h", "vel")):
        source = replace_body(source, name, f"""    real central = __fdiv_rn(7.0f * (q0 + qm1) - (qp1 + qm2), 12.0f);
    real dissipative = (qp1 - qm2) - 3.0f * (q0 - qm1);
    real sign_velocity = copysignf(1.0f, {sign});
    return vel * (central + __fdiv_rn(sign_velocity * dissipative, 12.0f));""")
    return source


def wrf_scalar_accumulation(source):
    start = source.index("    if (open_x || open_y) {", source.index("void flux_div_scalar("))
    end = source.index("\n    real fx[2]", start)
    # This control is only invoked on the fixture's specified boundary path.
    code = """    if (open_x || open_y) {
        real t = tend_out[IDX3(k, j, i)];
        if (!open_y || (j > 0 && j < ny - 1)) {
            real v0 = rv[I3(k, j, i, ny + 1, nx)];
            real v1 = rv[I3(k, j + 1, i, ny + 1, nx)];
            real fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
            real fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            real mrdy = msf[(size_t)j * nx + i] * dy_inv;
            t = t - mrdy * (fy1 - fy0);
        }
        if (!open_x || (i > 0 && i < nx - 1)) {
            real v0 = ru[I3(k, j, i, ny, nx + 1)];
            real v1 = ru[I3(k, j, i + 1, ny, nx + 1)];
            real fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
            real fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            real mrdx = msf[(size_t)j * nx + i] * dx_inv;
            t = t - mrdx * (fx1 - fx0);
        }
        real fz0 = zface_half(q, rw[I3(k, j, i, ny, nx)], k, j, i, nz, ny, nx, fnm, fnp);
        real fz1 = zface_half(q, rw[I3(k + 1, j, i, ny, nx)], k + 1, j, i, nz, ny, nx, fnm, fnp);
        t = t - rdnw[k] * (fz1 - fz0);
        tend_out[IDX3(k, j, i)] = t;
        return;
    }
"""
    return source[:start] + code + source[end:]


def main(output,fixture="rk-tendency.npz"):
    arrays, metadata = load_rk_tendency_oracle(fixture)
    baseline = module_source("advection")
    variants = {"production": None, "no_fma": baseline,
                "wrf_flux_order_no_fma": wrf_flux_order(baseline),
                "wrf_flux_and_accumulation_order_no_fma": wrf_scalar_accumulation(wrf_flux_order(baseline))}
    original = advection.get_kernel
    table = {}
    try:
        for variant, source in variants.items():
            if source is None:
                advection.get_kernel = original
            else:
                module = cp.RawModule(code=source, options=("-std=c++17", "--fmad=false"))
                advection.get_kernel = lambda name, function: (module.get_function(function) if name == "advection" else original(name, function))
            table[variant] = {}
            for case in metadata["cases"]:
                got = replay_rk_tendency_case(arrays, case)
                expected = arrays[f"tendency{case['id']}_tend_wrf"][4]
                table[variant][case["name"]] = measure_momentum_parity({"theta": expected}, {"theta": got["tend"][4]})["theta"]
    finally:
        advection.get_kernel = original
    output.write_text(json.dumps(table, indent=2) + "\n", encoding="ascii")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture",default="rk-tendency.npz")
    args=parser.parse_args()
    main(args.output,args.fixture)
