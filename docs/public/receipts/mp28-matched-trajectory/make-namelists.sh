#!/bin/bash
# usage: mknml.sh OUTFILE MP_PHYSICS NX NZ DX ZTOP DT RUN_MIN HIST_MIN
set -e
OUT=$1; MP=$2; NX=$3; NZ=$4; DX=$5; ZTOP=$6; DT=$7; RUNMIN=$8; HIST=$9
EWE=$((NX+1)); EVERT=$((NZ+1))
cat > "$OUT" <<EOF
 &time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = ${RUNMIN},
 run_seconds                         = 0,
 start_year                          = 0001,
 start_month                         = 01,
 start_day                           = 01,
 start_hour                          = 00,
 start_minute                        = 00,
 start_second                        = 00,
 end_year                            = 0001,
 end_month                           = 01,
 end_day                             = 01,
 end_hour                            = 00,
 end_minute                          = ${RUNMIN},
 end_second                          = 00,
 history_interval                    = ${HIST},
 frames_per_outfile                  = 1000,
 restart                             = .false.,
 restart_interval                    = 100000,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
 debug_level                         = 0,
 /

 &domains
 time_step                           = ${DT},
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 s_we                                = 1,
 e_we                                = ${EWE},
 s_sn                                = 1,
 e_sn                                = ${EWE},
 s_vert                              = 1,
 e_vert                              = ${EVERT},
 dx                                  = ${DX},
 dy                                  = ${DX},
 ztop                                = ${ZTOP},
 grid_id                             = 1,
 parent_id                           = 0,
 i_parent_start                      = 0,
 j_parent_start                      = 0,
 parent_grid_ratio                   = 1,
 parent_time_step_ratio              = 1,
 feedback                            = 0,
 smooth_option                       = 0,
 /

 &physics
 mp_physics                          = ${MP},
 ra_lw_physics                       = 0,
 ra_sw_physics                       = 0,
 radt                                = 30,
 sf_sfclay_physics                   = 0,
 sf_surface_physics                  = 0,
 bl_pbl_physics                      = 0,
 bldt                                = 0,
 cu_physics                          = 0,
 cudt                                = 0,
 /

 &fdda
 /

 &dynamics
 hybrid_opt                          = 0,
 rk_ord                              = 3,
 diff_opt                            = 2,
 km_opt                              = 4,
 c_s                                 = 0.25,
 damp_opt                            = 3,
 zdamp                               = 5000.,
 dampcoef                            = 0.2,
 w_damping                           = 0,
 khdif                               = 0,
 kvdif                               = 0,
 diff_6th_opt                        = 2,
 diff_6th_factor                     = 0.12,
 smdiv                               = 0.1,
 emdiv                               = 0.01,
 epssm                               = 0.1,
 time_step_sound                     = 6,
 h_mom_adv_order                     = 5,
 v_mom_adv_order                     = 3,
 h_sca_adv_order                     = 5,
 v_sca_adv_order                     = 3,
 moist_adv_opt                       = 1,
 scalar_adv_opt                      = 1,
 non_hydrostatic                     = .true.,
 mix_full_fields                     = .true.,
 /

 &bdy_control
 periodic_x                          = .true.,
 symmetric_xs                        = .false.,
 symmetric_xe                        = .false.,
 open_xs                             = .false.,
 open_xe                             = .false.,
 periodic_y                          = .true.,
 symmetric_ys                        = .false.,
 symmetric_ye                        = .false.,
 open_ys                             = .false.,
 open_ye                             = .false.,
 /

 &grib2
 /

 &namelist_quilt
 nio_tasks_per_group = 0,
 nio_groups = 1,
 /

 &ideal
 ideal_case = 2
 /
EOF
echo "wrote $OUT"
