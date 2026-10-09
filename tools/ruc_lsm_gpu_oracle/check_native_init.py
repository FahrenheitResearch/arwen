"""Verify native initialization with full-field hashes and a before control."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.core.moist import init_moist_balanced
from gpuwm.ingest import wrfinput
from gpuwm.io.history_layout import live_state_history_fields
from test_ruc_init_history import _case
from test_wrfinput_cold_start import _noah_case
from types import SimpleNamespace


def state_for(cfg):
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), cfg.p_surf, cfg.ztop)
    return init_moist_balanced(cfg, coord, base, lambda z: np.full_like(z, .005))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    args = p.parse_args()
    source = subprocess.check_output(["git", "show", "08fa43b839bb5e2a41e08d7242cf0d513efd0fd3:gpuwm/ingest/wrfinput.py"], text=True)
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "initialize_wrfinput_physics")
    namespace = dict(wrfinput.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "before-wrfinput.py", "exec"), namespace)
    before = namespace[function.name]
    rows = []
    for scheme in (3, 2, 4):
        if scheme == 3:
            cfg, restored, expected = _case()
        else:
            raw = _noah_case()
            cfg = RunConfig(nx=5, ny=4, nz=10, ztop=16000., dx=3000., dy=3000., dt=20.,
                run_seconds=20., moist=True, mp_physics=8, sf_surface_physics=scheme,
                sf_sfclay_physics=1, bl_pbl_physics=1, ra_physics=0)
            restored = SimpleNamespace(raw=raw, global_attributes={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH"})
        arrays = []
        for initialize in (before, wrfinput.initialize_wrfinput_physics):
            state = state_for(cfg)
            state.physics = initialize(state, restored, cfg,
                radiation_start_time=datetime(2026, 7, 1, 18),
                radiation_latitude=np.full((cfg.ny, cfg.nx), 40., np.float32),
                radiation_longitude=np.full((cfg.ny, cfg.nx), -100., np.float32))
            arrays.append(cp.asnumpy(live_state_history_fields(state)["SH2O"]))
        row = {"scheme": scheme, "words": arrays[0].size,
            "before_sha256": hashlib.sha256(arrays[0].astype("<f4").tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(arrays[1].astype("<f4").tobytes()).hexdigest(),
            "changed_words": int(np.count_nonzero(arrays[0].view("u4") != arrays[1].view("u4")))}
        if scheme == 3:
            row["wrf_init_differing_words"] = int(np.count_nonzero(arrays[1].view("u4") != expected.view("u4")))
            assert row["wrf_init_differing_words"] == 0
        else:
            assert row["changed_words"] == 0, row
        rows.append(row)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
