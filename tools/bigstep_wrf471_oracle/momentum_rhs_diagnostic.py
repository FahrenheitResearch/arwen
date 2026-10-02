"""Attribute RHS geopotential rounding with transient arithmetic controls.

Forecast source files are read, never modified. The full arithmetic control
retains the existing terms, stencils, indices, boundary conditions and helpers.
It restores the original Fortran multiplication/division and accumulation order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cupy as cp
import gpuwm.core.dycore as dycore
from gpuwm.core.kernels import module_source
from gpuwm.verify.bigstep_momentum_oracle import measure_momentum_parity
from gpuwm.verify.bigstep_prep_oracle import PREP_CASES, load_prep_fixture, prep_port_outputs


def change_body(source, transform):
    start = source.index("void slow_geopotential(")
    first = source.index("{", start)
    end = source.index("\n}", first) + 2
    return source[:first] + transform(source[first:end]) + source[end:]


def gw_and_accumulation_order(body):
    assert body.count("real gw = rn_mul(mass, rn_mul(G, w[ix]));") == 1
    body = body.replace("real gw = rn_mul(mass, rn_mul(G, w[ix]));", "real gw = rn_mul(rn_mul(mass, G), w[ix]);")
    target = "real tendency = (top && add_vertical) ? 0.0f : rph_t[ix];"
    body = body.replace(target, target + "\n    real hx_deferred = 0.0f;")
    body = body.replace("tendency = rn_sub(tendency, hx);", "hx_deferred = hx;")
    body = body.replace("tendency = rn_sub(tendency, hy);", "tendency = rn_sub(rn_sub(tendency, hy), hx_deferred);")
    body = body.replace("tendency = rn_sub(tendency, rn_add(hx, hy));", "tendency = rn_sub(rn_sub(tendency, hy), hx);")
    return body


def full_arithmetic_order(_body):
    return r"""{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = (top && add_vertical) ? 0.0f : rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(2.0f, quarter_rdx);
        quarter_rdy = rn_mul(2.0f, quarter_rdy);
    }
    if (add_vertical) {
        if (!top) {
            real ph_km = ph_total(php, phb, k - 1, c, st, base3d);
            real ph_k = ph_total(php, phb, k, c, st, base3d);
            real ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
            real wd_lo = rn_mul(rn_mul(rn_mul(0.5f, rn_add(ww[ix], ww[ix - st])), rdnw[k - 1]), rn_sub(ph_k, ph_km));
            real wd_hi = rn_mul(rn_mul(rn_mul(0.5f, rn_add(ww[ix + st], ww[ix])), rdnw[k]), rn_sub(ph_kp, ph_k));
            real omega = rn_add(rn_mul(fnm[k], wd_hi), rn_mul(fnp[k], wd_lo));
            tendency = rn_sub(tendency, omega);
        }
        real mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
        real gw = rn_mul(rn_mul(mass, G), w[ix]);
        if (has_msf) gw = rn_div(gw, msft[c]);
        tendency = rn_add(tendency, gw);
    }
    // WRF performs the y contribution first, then x. A direction's
    // map-factor division precedes the face sum and centered stencil.
    for (int axis = 0; axis < 2; ++axis) {
        int pos = axis ? i : j;
        int length = axis ? nx : ny;
        int boundary = axis ? boundary_x : boundary_y;
        real quarter_rd = axis ? quarter_rdx : quarter_rdy;
        real coefficient = has_msf ? rn_div(quarter_rd, msft[c]) : quarter_rd;
        int stencil = order == 2 ? 2 : 6;
        if (boundary && (pos == 0 || pos == length - 1)) continue;
        if (order != 2 && specified && (pos < 3 || pos >= length - 3)) {
            if (pos == 1 || pos == length - 2) stencil = 2;
            else if (!axis && (pos == 2 || pos == length - 3)) stencil = 4;
            else continue;
        }
        real face0 = axis ? fcx_value(i, j, k, st, ny, nx, u, mup, mub2d, c1f, c2f, msfu, has_msf, top, cfn, cfn1)
                          : fcy_value(j, i, k, st, ny, nx, v, mup, mub2d, c1f, c2f, msfv, has_msf, top, cfn, cfn1);
        real face1 = axis ? fcx_value(i + 1, j, k, st, ny, nx, u, mup, mub2d, c1f, c2f, msfu, has_msf, top, cfn, cfn1)
                          : fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d, c1f, c2f, msfv, has_msf, top, cfn, cfn1);
        real contribution;
        if (stencil == 2) {
            int pm = (pos - 1 + length) % length;
            int pp = (pos + 1) % length;
            size_t cm = axis ? (size_t)j * nx + pm : (size_t)pm * nx + i;
            size_t cp = axis ? (size_t)j * nx + pp : (size_t)pp * nx + i;
            real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d), ph_total(php, phb, k, cm, st, base3d));
            real d1 = rn_sub(ph_total(php, phb, k, cp, st, base3d), ph_total(php, phb, k, c, st, base3d));
            real sum = rn_add(rn_mul(face1, d1), rn_mul(face0, d0));
            contribution = rn_mul(coefficient, sum);
        } else {
            real d1 = axis ? x_difference(i, 1, j, k, st, nx, php, phb, base3d) : y_difference(j, 1, i, k, st, ny, nx, php, phb, base3d);
            real d2 = axis ? x_difference(i, 2, j, k, st, nx, php, phb, base3d) : y_difference(j, 2, i, k, st, ny, nx, php, phb, base3d);
            real raw;
            real reciprocal;
            if (stencil == 4) {
                raw = rn_sub(rn_mul(8.0f, d1), d2);
                reciprocal = __fdiv_rn(1.0f, 12.0f);
            } else {
                real d3 = axis ? x_difference(i, 3, j, k, st, nx, php, phb, base3d) : y_difference(j, 3, i, k, st, ny, nx, php, phb, base3d);
                raw = rn_add(rn_sub(rn_mul(45.0f, d1), rn_mul(9.0f, d2)), d3);
                reciprocal = __fdiv_rn(1.0f, 60.0f);
            }
            // WRF: (0.25*rd/msft) * ( (face1 + face0) * (1./60.) * (stencil) ),
            // evaluated left to right inside the parentheses.
            real weighted_faces = rn_mul(rn_add(face1, face0), reciprocal);
            contribution = rn_mul(coefficient, rn_mul(weighted_faces, raw));
        }
        tendency = rn_sub(tendency, contribution);
    }
    rph_t[ix] = tendency;
}"""


def main(output):
    baseline = module_source("dycore")
    variants = {"production": None,
                "gw_and_accumulation_order": change_body(baseline, gw_and_accumulation_order),
                "full_wrf_arithmetic_order": change_body(baseline, full_arithmetic_order)}
    original = dycore.get_kernel
    receipt = {"source_sha256": hashlib.sha256(baseline.encode()).hexdigest(), "measurements": {}}
    try:
        for variant, source in variants.items():
            if source is None:
                dycore.get_kernel = original
            else:
                module = cp.RawModule(code=source, options=("-std=c++17", "--fmad=false"))
                dycore.get_kernel = lambda name, function: module.get_function(function) if name == "dycore" else original(name, function)
            receipt["measurements"][variant] = {}
            for case in PREP_CASES:
                fixture = load_prep_fixture(case)
                got = prep_port_outputs(fixture)["ph_tend"]
                receipt["measurements"][variant][case] = {
                    "native": measure_momentum_parity({"ph_tend": fixture["ph_tend"]}, {"ph_tend": got})["ph_tend"],
                    "normalized_phi": measure_momentum_parity({"ph_tend": fixture["ph_tend_total_phi"]}, {"ph_tend": got})["ph_tend"]}
    finally:
        dycore.get_kernel = original
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="ascii")
    for variant, cases in receipt["measurements"].items():
        print(variant, {case: (metrics["normalized_phi"]["max_ulp"], metrics["normalized_phi"]["differing_words"]) for case, metrics in cases.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
