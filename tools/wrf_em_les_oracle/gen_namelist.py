#!/usr/bin/env python3
"""Generate WRF v4.6.1 em_les namelists for the WP-L7 oracle.

Two families:
  stock  - the shipped test/em_les configuration, unmodified (native oracle)
  match  - matched to the ArWen idealized CBL case
           (gpuwm/verify/cases/convective_boundary_layer.py)

Every deviation from the stock em_les namelist is enumerated in the
matched-config receipt written alongside the runs.
"""
import sys
import json

# ArWen CBL case constants, read from convective_boundary_layer.py
ARWEN = dict(heat_flux=0.24, drag=0.0013, c_s=0.18, c_k=0.10, mix_isotropic=1,
             ztop=2400.0, epssm=0.1, smdiv=0.1, emdiv=0.0, tss=4)


def frac(dt):
    """WRF time_step + fraction for a float dt."""
    if abs(dt - round(dt)) < 1e-12:
        return int(round(dt)), 0, 1
    for den in (2, 4, 8, 10, 100):
        num = dt * den
        if abs(num - round(num)) < 1e-9:
            whole = int(dt // 1)
            return whole, int(round(num)) - whole * den, den
    raise ValueError(dt)


def namelist(km_opt, ncell, nz, dx, ztop, dt, hours, hist_min, stock=False):
    ts, fn, fd = frac(dt)
    e_we = ncell + 1
    e_sn = ncell + 1
    e_vert = nz + 1
    if stock:
        phys = dict(sf_sfclay_physics=1, isfflx=2, drag=None, use_theta_m=1,
                    khdif=1.0, kvdif=1.0, emdiv=0.01)
    else:
        phys = dict(sf_sfclay_physics=0, isfflx=0, drag=ARWEN["drag"],
                    use_theta_m=0, khdif=0.0, kvdif=0.0, emdiv=ARWEN["emdiv"])
    L = []
    A = L.append
    A(" &time_control")
    A(" run_days                            = 0,")
    A(" run_hours                           = %d," % hours)
    A(" run_minutes                         = 0,")
    A(" run_seconds                         = 0,")
    A(" start_year                          = 0001,")
    A(" start_month                         = 01,")
    A(" start_day                           = 01,")
    A(" start_hour                          = 00,")
    A(" start_minute                        = 00,")
    A(" start_second                        = 00,")
    A(" end_year                            = 0001,")
    A(" end_month                           = 01,")
    A(" end_day                             = 01,")
    A(" end_hour                            = %02d," % hours)
    A(" end_minute                          = 00,")
    A(" end_second                          = 00,")
    A(" history_interval_m                  = %d," % hist_min)
    A(" frames_per_outfile                  = 1000,")
    A(" restart                             = .false.,")
    A(" io_form_history                     = 2,")
    A(" io_form_restart                     = 2,")
    A(" io_form_input                       = 2,")
    A(" io_form_boundary                    = 2,")
    A(' iofields_filename                   = "iofields_les.txt",')
    A(" /")
    A("")
    A(" &domains")
    A(" time_step                           = %d," % ts)
    A(" time_step_fract_num                 = %d," % fn)
    A(" time_step_fract_den                 = %d," % fd)
    A(" max_dom                             = 1,")
    A(" s_we                                = 1,")
    A(" e_we                                = %d," % e_we)
    A(" s_sn                                = 1,")
    A(" e_sn                                = %d," % e_sn)
    A(" s_vert                              = 1,")
    A(" e_vert                              = %d," % e_vert)
    A(" dx                                  = %g," % dx)
    A(" dy                                  = %g," % dx)
    A(" ztop                                = %g," % ztop)
    A(" grid_id                             = 1,")
    A(" parent_id                           = 0,")
    A(" i_parent_start                      = 0,")
    A(" j_parent_start                      = 0,")
    A(" parent_grid_ratio                   = 1,")
    A(" parent_time_step_ratio              = 1,")
    A(" feedback                            = 0,")
    A(" smooth_option                       = 0,")
    A(" /")
    A("")
    A(" &physics")
    A(" mp_physics                          = 0,")
    A(" ra_lw_physics                       = 0,")
    A(" ra_sw_physics                       = 0,")
    A(" radt                                = 0,")
    A(" sf_sfclay_physics                   = %d," % phys["sf_sfclay_physics"])
    A(" sf_surface_physics                  = 0,")
    A(" bl_pbl_physics                      = 0,")
    A(" bldt                                = 0,")
    A(" cu_physics                          = 0,")
    A(" cudt                                = 0,")
    A(" isfflx                              = %d," % phys["isfflx"])
    A(" /")
    A("")
    A(" &fdda")
    A(" /")
    A("")
    A(" &dynamics")
    A(" hybrid_opt                          = 0,")
    A(" rk_ord                              = 3,")
    A(" diff_opt                            = 2,")
    A(" km_opt                              = %d," % km_opt)
    A(" damp_opt                            = 0,")
    A(" zdamp                               = 5000.,")
    A(" dampcoef                            = 0.1,")
    A(" khdif                               = %g," % phys["khdif"])
    A(" kvdif                               = %g," % phys["kvdif"])
    A(" c_s                                 = %g," % ARWEN["c_s"])
    A(" c_k                                 = %g," % ARWEN["c_k"])
    A(" mix_isotropic                       = %d," % ARWEN["mix_isotropic"])
    A(" mix_upper_bound                     = 0.1,")
    A(" smdiv                               = %g," % ARWEN["smdiv"])
    A(" emdiv                               = %g," % phys["emdiv"])
    A(" epssm                               = %g," % ARWEN["epssm"])
    A(" tke_heat_flux                       = %g," % ARWEN["heat_flux"])
    if phys["drag"] is not None:
        A(" tke_drag_coefficient                = %g," % phys["drag"])
    A(" time_step_sound                     = %d," % (6 if stock else ARWEN["tss"]))
    A(" h_mom_adv_order                     = 5,")
    A(" v_mom_adv_order                     = 3,")
    A(" h_sca_adv_order                     = 5,")
    A(" v_sca_adv_order                     = 3,")
    A(" mix_full_fields                     = .true.,")
    A(" non_hydrostatic                     = .true.,")
    A(" pert_coriolis                       = .true.,")
    A(" use_theta_m                         = %d," % phys["use_theta_m"])
    A(" diff_6th_opt                        = 0,")
    A(" w_damping                           = 0,")
    A(" /")
    A("")
    A(" &bdy_control")
    A(" periodic_x                          = .true.,")
    A(" symmetric_xs                        = .false.,")
    A(" symmetric_xe                        = .false.,")
    A(" open_xs                             = .false.,")
    A(" open_xe                             = .false.,")
    A(" periodic_y                          = .true.,")
    A(" symmetric_ys                        = .false.,")
    A(" symmetric_ye                        = .false.,")
    A(" open_ys                             = .false.,")
    A(" open_ye                             = .false.,")
    A(" /")
    A("")
    A(" &grib2")
    A(" /")
    A("")
    A(" &namelist_quilt")
    A(" nio_tasks_per_group = 0,")
    A(" nio_groups = 1,")
    A(" /")
    A("")
    A(" &ideal")
    A(" ideal_case = 9")
    A(" /")
    return "\n".join(L) + "\n"


RUNS = {
    # id                km  ncell  nz    dx    ztop     dt  h  hist  stock
    "stock_em_les":    (2,    39,  39, 100., 2000.,  1.0,  1,   10, True),
    "match_km3_100m":  (3,    96,  64, 100., 2400.,  0.5,  2,    1, False),
    "match_km2_100m":  (2,    96,  64, 100., 2400.,  0.5,  2,    1, False),
    "match_km3_50m":   (3,   192,  96,  50., 2400.,  0.25, 2,    1, False),
    "match_km2_50m":   (2,   192,  96,  50., 2400.,  0.25, 2,    1, False),
}


if __name__ == "__main__":
    outdir = sys.argv[1]
    meta = {}
    for rid, (km, nc, nz, dx, ztop, dt, hrs, hist, stock) in RUNS.items():
        txt = namelist(km, nc, nz, dx, ztop, dt, hrs, hist, stock=stock)
        with open("%s/namelist.%s" % (outdir, rid), "w") as fh:
            fh.write(txt)
        meta[rid] = dict(km_opt=km, ncell_x=nc, ncell_y=nc, nz=nz, dx_m=dx,
                         ztop_m=ztop, dt_s=dt, hours=hrs, hist_min=hist,
                         stock=stock, points=nc * nc * nz,
                         steps=int(hrs * 3600 / dt))
    with open("%s/run_matrix.json" % outdir, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    for k, v in sorted(meta.items()):
        print("%-16s km_opt=%d %4dx%4dx%3d dx=%5.1f dt=%4.2f %dh %9d pts %6d steps"
              % (k, v["km_opt"], v["ncell_x"], v["ncell_y"], v["nz"], v["dx_m"],
                 v["dt_s"], v["hours"], v["points"], v["steps"]))
