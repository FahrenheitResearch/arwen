"""Probe: WRF's damper also rewrites w below the layer; WOOF skips it.

WRF advance_w applies ``w = (w - dampwt*(c1f*mut+c2f)*w_save)/(1+dampwt)``
on EVERY level 2..kde, with ``dampwt = 0`` below ``hbot``.  There that is
``(w - 0*m*w_save)/1``: the identity, except that ``-0 - (-0) = +0``.  WOOF
updates only levels at or above hbot, so a w'' of exactly -0.0 below the
layer, under a negative w_save, would keep its sign in WOOF and lose it in
WRF.  This probe drives one column (j, i) = (7, 0) of the regime domain to
an all-negative-zero acoustic state with w_save = -1 m/s and reports, for
the damped and undamped cases, every w word where the two differ and
WRF's below-layer w'' signs.

    GPUWM_WRF_EXACT=1 python tools/upper_damping_wrf461_oracle/probe_signed_zero.py \
        --oracle OUT --output probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.upper_damping_wrf461_oracle import columns, run_oracle  # noqa: E402

J, I = 7, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from tools.smallstep_wrf471_oracle import vertical_workspace
    module, workspace = run_oracle._vertical_module()
    original = module.make_vertical_state
    nzero = np.float32(-0.0)

    def zeroed(raw, metadata):
        s, cfg = original(raw, metadata)
        for name in ("w_pp", "rw_t", "rph_t", "ph_pp", "th_pp"):
            getattr(s, name)[:, J, I] = nzero
        s.ww_pp[1:-1, J, I] = nzero
        s.mu_pp[J, I] = nzero
        s.scratch((cfg.ny, cfg.nx), "acoustic_mu_pp_old")[J, I] = nzero
        s.scratch((cfg.nz, cfg.ny, cfg.nx), "acoustic_th_pp_old")[:, J, I] = nzero
        s.w[:, J, I] = np.float32(-1.0)
        # Faces around the column, and their periodic aliases (face nx/ny is
        # face 0 in WRF's haloed memory; the CUDA kernel reads it directly).
        s.u_pp[:3, J, I:I + 2] = 0.0
        s.v_pp[:3, J:J + 2, I] = 0.0
        if I == 0:
            s.u_pp[:3, J, cfg.nx] = 0.0
        if J + 1 == cfg.ny:
            s.v_pp[:3, 0, I] = 0.0
        return s, cfg

    raw = columns.build_raw(ROOT)
    library = str(args.oracle / "O0" / "libsmallstep_oracle.so")
    result = {}
    edge, _ = columns.edge_zdamp(raw)
    for case, metadata in run_oracle.advance_w_cases(edge):
        if case not in ("hrrr_damp", "hrrr_control"):
            continue
        arrays = {}
        ctx = (patch.object(vertical_workspace, "workspace_source", workspace)
               if workspace else run_oracle._null())
        with ctx, patch.object(module, "make_vertical_state", zeroed):
            measured = module.measure_vertical_case(raw, metadata, library, output_arrays=arrays)
        wn, ww = arrays["w_native"][:, J, I], arrays["w_wrf"][:, J, I]
        differ = np.nonzero(wn.view(np.uint32) != ww.view(np.uint32))[0]
        result[case] = {
            "w_all_columns": measured["advance_w"]["w"],
            "column_w_levels_differing": [int(k) for k in differ],
            "column_w_native": [repr(float(x)) for x in wn],
            "column_w_wrf": [repr(float(x)) for x in ww],
            "column_w_wrf_negative_zero_levels": [int(k) for k in
                                                  np.nonzero((ww == 0) & np.signbit(ww))[0]]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("w_all_columns", "column_w_levels_differing",
                                            "column_w_wrf_negative_zero_levels")}
                      for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
