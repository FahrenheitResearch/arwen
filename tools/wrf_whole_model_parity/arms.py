"""Write one namelist.input per scheme arm for a WRF-vs-WOOF whole-model run.

    python arms.py CASE_DIR ROOT [--steps N]

CASE_DIR is a real.exe pair (wrfinput_d01, wrfbdy_d01) with the
namelist.input real.exe ran; its domain block (grid, dt, start, levels) is
kept and the physics come from ARMS below.  Both sides run the namelist
written to ROOT/cases/<arm>/namelist.input (WOOF's copy differs only in
restart_interval, see run_woof_arms.sh).  Each arm changes only what it
tests from the classic WRF suite (YSU / MM5-revised surface layer / Noah /
Kain-Fritsch / RRTM LW / Dudhia SW).  The arms are table rows: a new scheme
is a new row.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE = dict(mp_physics=6, ra_lw_physics=1, ra_sw_physics=1, radt=15,
            sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
            bldt=0, cu_physics=1, cudt=5, num_soil_layers=4, icloud=1)
#: The wizard's default root (gpuwm.domain_wizard.DEFAULT_SUITE_PHYSICS,
#: profile morrison-mp10-ysu-mm5-noah-kf) with radiation as WRF's own RRTMG
#: (legacy port) so both sides run the same radiation; the profile's
#: RTE+RRTMGP substitution is a declared divergence and is not compared.
DEFAULT12_DYNAMICS = dict(epssm=0.5, diff_6th_opt=2, diff_6th_slopeopt=1,
                          diff_6th_factor=0.12)
ARMS = {
    "kf-default12": (dict(BASE, mp_physics=10, ra_lw_physics=4,
                          ra_sw_physics=4, radt=12, cudt=5),
                     DEFAULT12_DYNAMICS),
    "kf-cudt0": (dict(BASE, cudt=0), {}),
    "wsm6": (dict(BASE), {}),
    "kessler": (dict(BASE, mp_physics=1), {}),
    "milbrandt": (dict(BASE, mp_physics=9), {}),
    "wdm6": (dict(BASE, mp_physics=16), {}),
    "myj": (dict(BASE, bl_pbl_physics=2, sf_sfclay_physics=2), {}),
    "cu0": (dict(BASE, cu_physics=0), {}),
}
DYNAMICS = dict(w_damping=1, diff_opt=1, km_opt=4, khdif=0, kvdif=0,
                non_hydrostatic=True, moist_adv_opt=1, scalar_adv_opt=1,
                use_baseparam_fr_nml=True, top_lid=False)


def _fmt(value):
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def _case_domain(case_namelist: str) -> dict:
    def get(key):
        m = re.search(rf"^\s*{key}\s*=\s*([^,\n]+)", case_namelist,
                      re.M | re.I)
        if not m:
            raise SystemExit(f"case namelist has no {key}")
        return int(float(m.group(1)))
    return {k: get(k) for k in (
        "time_step", "e_we", "e_sn", "e_vert", "dx", "dy",
        "start_year", "start_month", "start_day", "start_hour",
        "interval_seconds", "num_metgrid_levels",
        "num_metgrid_soil_levels")}


def namelist(arm: str, dom: dict, steps: int) -> str:
    phys, dyn_extra = ARMS[arm]
    dt = dom["time_step"]
    seconds = steps * dt
    tc = dict(run_days=0, run_hours=0, run_minutes=seconds // 60,
              run_seconds=seconds % 60, start_year=dom["start_year"],
              start_month=dom["start_month"], start_day=dom["start_day"],
              start_hour=dom["start_hour"], start_minute=0, start_second=0,
              end_year=dom["start_year"], end_month=dom["start_month"],
              end_day=dom["start_day"], end_hour=dom["start_hour"],
              end_minute=seconds // 60, end_second=seconds % 60,
              interval_seconds=dom["interval_seconds"], input_from_file=True,
              history_interval_s=dt, frames_per_outfile=1000, restart=False,
              restart_interval=100000, io_form_history=2, io_form_restart=2,
              io_form_input=2, io_form_boundary=2, debug_level=0)
    domains = dict(time_step=dt, time_step_fract_num=0, time_step_fract_den=1,
                   max_dom=1, s_we=1, e_we=dom["e_we"], e_sn=dom["e_sn"],
                   e_vert=dom["e_vert"], dx=dom["dx"], dy=dom["dy"],
                   grid_id=1, parent_id=0, i_parent_start=1,
                   j_parent_start=1, parent_grid_ratio=1,
                   parent_time_step_ratio=1, feedback=0, smooth_option=0,
                   num_metgrid_levels=dom["num_metgrid_levels"],
                   num_metgrid_soil_levels=dom["num_metgrid_soil_levels"])
    bdy = dict(spec_bdy_width=5, spec_zone=1, relax_zone=4, specified=True,
               nested=False)
    out = []
    for name, group in (("time_control", tc), ("domains", domains),
                        ("physics", phys),
                        ("dynamics", dict(DYNAMICS, **dyn_extra)),
                        ("bdy_control", bdy),
                        ("namelist_quilt", dict(nio_tasks_per_group=0,
                                                nio_groups=1))):
        out.append(f" &{name}")
        out.extend(f" {k:<28}= {_fmt(v)}," for k, v in group.items())
        out.extend([" /", ""])
    # WOOF runs the matched arithmetic against WRF: correct mode (the 2.9.0
    # default) is named off in the selector comment, which wrf.exe reads as
    # a comment.
    from gpuwm.physics_source_defaults import (exact_mode_selectors,
                                               with_physics_selector_comment)
    return with_physics_selector_comment("\n".join(out), exact_mode_selectors())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--steps", type=int, default=15)
    args = ap.parse_args(argv)
    dom = _case_domain((args.case_dir / "namelist.input").read_text())
    for arm in ARMS:
        d = args.root / "cases" / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "namelist.input").write_text(namelist(arm, dom, args.steps))
    (args.root / "arms.json").write_text(json.dumps(
        {"steps": args.steps, "domain": dom,
         "arms": {k: {"physics": p, "dynamics": dict(DYNAMICS, **d)}
                  for k, (p, d) in ARMS.items()}}, indent=1))
    print("\n".join(ARMS))


if __name__ == "__main__":
    main()
