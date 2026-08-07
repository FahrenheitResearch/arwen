#!/usr/bin/env python3
"""Write the unchanged-stock-WRF acceptance namelist for one HRRR target."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm.ingest.hrrr_target import load_hrrr_target_domain  # noqa: E402
from gpuwm.vertical_contract import (  # noqa: E402
    explicit_vertical_from_wrf_namelist,
)


def _date_fields(value: datetime, prefix: str) -> str:
    return "\n".join((
        f" {prefix}_year                          = {value.year:04d},",
        f" {prefix}_month                         = {value.month:02d},",
        f" {prefix}_day                           = {value.day:02d},",
        f" {prefix}_hour                          = {value.hour:02d},",
        f" {prefix}_minute                        = {value.minute:02d},",
        f" {prefix}_second                        = {value.second:02d},",
    ))


def _eta_lines(eta) -> str:
    values = [f"{float(value):.5f}" for value in eta]
    rows = [", ".join(values[index:index + 5])
            for index in range(0, len(values), 5)]
    return "\n".join(
        (" eta_levels = " if index == 0 else "              ")
        + row + ("," if index < len(rows) - 1 else ",")
        for index, row in enumerate(rows))


def _time_step_lines(target) -> str:
    """The root clock, spelled the way stock WRF reads it.

    WRF's registry carries the clock as integer ``time_step`` plus an
    exact rational remainder (``time_step_fract_num/den``).  A 1.5 km
    ladder's 7.5 s clock therefore spells 7 + 1/2 -- the SAME spelling
    the experiment TOML uses -- and a whole-second clock stays the
    single line every prior emission wrote.
    """
    lines = [f" time_step                           = "
             f"{target.time_step_seconds},"]
    if getattr(target, "time_step_fract_num", 0):
        lines.append(f" time_step_fract_num                 = "
                     f"{target.time_step_fract_num},")
        lines.append(f" time_step_fract_den                 = "
                     f"{target.time_step_fract_den},")
    return "\n".join(lines)


def render_namelist(
        *, target, eta, valid_time, run_seconds: int,
        p_top: float = 10_000.0, hybrid_opt: int = 2,
        etac: float = 0.2) -> str:
    end_time = valid_time + timedelta(seconds=run_seconds)
    return f"""&time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = 0,
 run_seconds                         = {run_seconds},
{_date_fields(valid_time, "start")}
{_date_fields(end_time, "end")}
 interval_seconds                    = 3600,
 input_from_file                     = .true.,
 history_interval                    = 60,
 frames_per_outfile                  = 1,
 restart                             = .false.,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
 /

&domains
{_time_step_lines(target)}
 max_dom                             = 1,
 e_we                                = {target.nx + 1},
 e_sn                                = {target.ny + 1},
 e_vert                              = {target.nz + 1},
{_eta_lines(eta)}
 p_top_requested                     = {p_top:.12g},
 dx                                  = {target.dx_m:.12g},
 dy                                  = {target.dy_m:.12g},
 grid_id                             = 1,
 parent_id                           = 0,
 i_parent_start                      = 1,
 j_parent_start                      = 1,
 parent_grid_ratio                   = 1,
 parent_time_step_ratio              = 1,
 feedback                            = 0,
 smooth_option                       = 0,
 sfcp_to_sfcp                        = .true.,
 /

&physics
 mp_physics                          = 6,
 ra_lw_physics                       = 1,
 ra_sw_physics                       = 1,
 radt                                = 1,
 sf_sfclay_physics                   = 91,
 sf_surface_physics                  = 2,
 bl_pbl_physics                      = 1,
 bldt                                = 0,
 cu_physics                          = 0,
 cudt                                = 0,
 isfflx                              = 1,
 ifsnow                              = 1,
 icloud                              = 1,
 surface_input_source                = 1,
 num_soil_layers                     = 4,
 num_land_cat                        = 21,
 sf_urban_physics                    = 0,
 fractional_seaice                   = 1,
 sst_update                          = 0,
 ghg_input                           = 0,
 /

&fdda
 /

&dynamics
 hybrid_opt                          = {hybrid_opt},
 etac                                = {etac:.12g},
 w_damping                           = 1,
 epssm                               = 0.5,
 diff_opt                            = 2,
 km_opt                              = 4,
 mix_full_fields                     = .true.,
 diff_6th_opt                        = 2,
 diff_6th_factor                     = 0.08,
 diff_6th_slopeopt                   = 1,
 base_temp                           = 290.,
 damp_opt                            = 3,
 zdamp                               = 5000.,
 dampcoef                            = 0.2,
 khdif                               = 0,
 kvdif                               = 0,
 non_hydrostatic                     = .true.,
 moist_adv_opt                       = 1,
 scalar_adv_opt                      = 1,
 /

&bdy_control
 spec_bdy_width                      = {target.spec_bdy_width},
 spec_zone                           = {target.spec_zone},
 relax_zone                          = {target.relax_zone},
 specified                           = .true.,
 nested                              = .false.,
 /

&grib2
 /

&namelist_quilt
 nio_tasks_per_group                 = 0,
 nio_groups                          = 1,
 /
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-spec", type=Path, required=True)
    parser.add_argument("--gpuwm-namelist", type=Path, required=True)
    parser.add_argument("--valid-time", required=True)
    parser.add_argument("--run-seconds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.run_seconds <= 0:
        raise ValueError("run-seconds must be positive")
    target = load_hrrr_target_domain(args.domain_spec)
    vertical = explicit_vertical_from_wrf_namelist(
        args.gpuwm_namelist,
        expected_nz=target.nz,
        context="stock-WRF HRRR namelist writer",
    )
    valid_time = datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_namelist(
        target=target,
        eta=vertical.eta_levels,
        p_top=vertical.p_top,
        hybrid_opt=vertical.hybrid_opt,
        etac=vertical.etac,
        valid_time=valid_time,
        run_seconds=args.run_seconds,
    ), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
