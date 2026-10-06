# GSI namelist profile: the deterministic hybrid 3DEnVar analysis of RRFS v1
# (rrfs-workflow tag v1.0.25b, commit e4a100b), applied to a WRF-format
# background through GSI's WRF mass interface.
#
# A profile is a table of values. case/run_singleob.sh (and any later
# analysis runner) reads only these keys; another system's analysis is
# another profile file, not another script.
#
# Sources (PUBLIC, read 2026-10-03), cited per key as
#   EX  = scripts/exrrfs_analysis_gsi.sh   (script defaults, conv_dbz path)
#   NL  = ush/gsiparm.anl.sh               (namelist template)
#   CD  = ush/config_defaults.sh
#   SDL = ush/set_rrfs_config_SDL_VDL_MixEn.sh
#   FIX = NOAA-EMC/GSI-fix 3ccc7626b6 (the fix submodule of GSI db90edf)
PROFILE_NAME=rrfs-v1.0.25b
GSI_PIN=gsi-db90edf                  # rrfs-workflow v1.0.25b sorc/Externals.cfg [GSI] hash
STATIC_FIELDS="RDX,RDY,C3H,C4H,C3F,C4F,XLAND"  # GSI WRF converter: cplr_wrf_netcdf_interface.f90:263-433,703

# &SETUP
NHR_ASSIMILATION=3                   # NL:16
MITER=2; NITER1=50; NITER2=50        # EX:347-349
QOPTION=2                            # NL:7
GENCODE=78                           # NL:11
# &BKGERR
BKGERR_VS=1.0                        # CD:716 bkgerr_vs
BKGERR_HZSCL="0.7,1.4,2.80"          # EX:669
USENEWGFSBERROR=.true.               # EX:741, paired with RRFS's static B file
# &HYBRID_ENSEMBLE
L_HYB_ENS=.true.                     # EX:372 (ifhyb with RRFS members present)
BETA_S0=0.15                         # EX:658 beta1_inv
S_ENS_H="328.632,82.1580,4.10790,4.10790,82.1580"   # EX:671 = SDL:38
S_ENS_V="3,3,-0.30125,-0.30125,0.0"                 # EX:672 = SDL:39
NSCLGRP=2; NGVARLOC=2                # EX:675-676 = SDL:6-7
NAENSLOC=5                           # EX:731: nsclgrp*ngvarloc+nsclgrp-1
R_ENSLOCCOV4TIM=1.0                  # EX:677
R_ENSLOCCOV4VAR=0.05                 # EX:678 = SDL:40
R_ENSLOCCOV4SCL=1.0                  # EX:679
READIN_LOCALIZATION=.false.          # EX:670 = SDL:32
ASSIGN_VDL_NML=.false.               # EX:681
Q_HYB_ENS=.false.                    # EX:680
UV_HYB_ENS=.true.                    # NL:161
L_ENS_IN_DIFF_TIME=.true.            # NL:171
GRID_RATIO_ENS=1                     # EX:375 (RRFS members on the analysis grid)
# vloc_varlist is read only when assign_vdl_nml is true (NL:181-188).
# &GRIDOPTS
# RRFS runs grid_ratio_fv3_regional=2.0 (CD:738, a 6 km analysis grid on its
# 3 km model grid), and its FV3 member reader interpolates members onto that
# grid. GSI's WRF member reader does not: it stops unless each member file
# has the analysis-ensemble grid's size (MEASURED 2026-10-03 with
# grid_ratio_wrfmass=2 on 302x286 members: "incorrect grid size in netcdf
# file", 152x144 expected). So on WRF files the analysis grid is the model
# grid. A named difference from RRFS, not a value of RRFS's.
RRFS_GRID_RATIO=2
GRID_RATIO=1
# &OBSQC (NL:46-47), &OBS_INPUT (NL:50)
OBSQC="dfact=0.75,dfact1=3.0,noiqc=.false.,c_varqc=0.02,vadfile='prepbufr',vadwnd_l2rw_qc=.true.,"
OBS_INPUT="dmesh(1)=120.0,dmesh(2)=60.0,dmesh(3)=30,time_window_max=1.5,time_window_rad=1.0,ext_sonde=.true.,"
# &RAPIDREFRESH_CLDSURF (NL:191-225; EX:345 ifsoilnudge=.false.;
# EX:356-357 i_use_2mQ4B=2, i_use_2mT4B=1). One value differs from RRFS's,
# for the interface: RRFS sets i_gsdcldanal_type=0 (NL:220; its cloud
# analysis runs outside GSI) and its FV3 reader reads hydrometeors anyway.
# On the WRF interface type 0 makes GSI switch hydrometeor background I/O
# off (db90edf gsimod.F90:2141-2143), and the guess reader then expects
# reflectivity records the converter never wrote (MEASURED 2026-10-03:
# end-of-file on sigf03). Type 99 is GSI's "only read hydrometeor fields but
# no cloud analysis" (rapidrefresh_cldsurf_mod.f90:111-122): the same
# analysis without a cloud analysis, which is what RRFS's 0 means.
CLDSURF="dfi_radar_latent_heat_time_period=20.0,metar_impact_radius=10.0,metar_impact_radius_lowCloud=4.0,l_gsd_terrain_match_surfTobs=.true.,l_sfcobserror_ramp_t=.true.,l_sfcobserror_ramp_q=.true.,l_PBL_pseudo_SurfobsT=.false.,l_PBL_pseudo_SurfobsQ=.false.,l_PBL_pseudo_SurfobsUV=.false.,pblH_ration=0.4,pps_press_incr=40.0,l_gsd_limit_ocean_q=.true.,l_pw_hgt_adjust=.true.,l_limit_pw_innov=.true.,max_innov_pct=0.1,l_cleanSnow_WarmTs=.true.,r_cleanSnow_WarmTs_threshold=5.0,l_conserve_thetaV=.true.,i_conserve_thetaV_iternum=3,l_gsd_soilTQ_nudge=.false.,l_cld_bld=.true.,l_numconc=.true.,l_closeobs=.true.,cld_bld_hgt=1200.0,build_cloud_frac_p=0.50,clear_cloud_frac_p=0.10,iclean_hydro_withRef_allcol=1,i_use_2mQ4B=2,i_use_2mT4B=1,i_gsdcldanal_type=99,i_gsdsfc_uselist=1,i_lightpcp=1,i_sfct_gross=1,i_coastline=3,i_gsdqc=2,"
# &SINGLEOB_TEST defaults of the template (NL:234)
SINGLEOB_MAGINNOV=1.0; SINGLEOB_MAGOBERR=0.8; SINGLEOB_TYPE=t

# Variable table. RRFS's conv_dbz table (FIX anavinfo.rrfs_conv_dbz, EX:719)
# is written for FV3 restart names; anavinfo/arw_rrfs_conv.txt is the same
# control vector, groups and amplitudes on WRF names. One row of RRFS's
# table is not in it:
#   fed: GSI's WRF reader and writer carry no flash-extent density (only the
#        FV3 reader does), and RRFS v1's release default leaves it off.
# dbz stays: GSI's WRF hydrometeor-ensemble reader cannot run without it
# (db90edf cplr_get_wrf_mass_ensperts.f90:1738 deallocates the reflectivity
# work array whether or not it was allocated; MEASURED 2026-10-03, forrtl
# severe (153) there with dbz left out).
ANAVINFO_TABLE=profiles/anavinfo/arw_rrfs_conv.txt
# &SETUP keys of the conv_dbz path (NL:7-10; EX:720 if_model_dbz=.true.)
SETUP_EXTRA="if_model_dbz=.true.,static_gsi_nopcp_dbz=0.0,if_use_w_vr=.false.,rmesh_dbz=4.0,rmesh_vr=4.0,zmesh_dbz=1000.0,zmesh_vr=1000.0,inflate_dbz_obserr=.true.,missing_to_nopcp=.false.,radar_no_thinning=.true.,"
# What the files must carry for this table on GSI's WRF interface, and what
# WOOF's files lack (MEASURED 2026-10-03): QNCLOUD in members and background,
# REFL_10CM in the background. case/add_zero_fields.sh says why each is read
# and why zeros leave a temperature observation's increment unchanged.
ANAVINFO_DROP_MET_GUESS=""

# Fix files: <name GSI opens>=<file in the fix set>. RRFS's static B
# (rrfs_glb_berror.l127y770.f77, EX:739) and its workflow fix tree are not
# public; the public HRRR v4.1.21 static B stands in, with the flag that
# matches that file's format. Both are written into the run receipt.
FIX_MAP="convinfo=gsi-fix/convinfo.rrfs errtable=gsi-fix/errtable.rrfs satinfo=gsi-fix/global_satinfo.txt ozinfo=gsi-fix/global_ozinfo.txt pcpinfo=gsi-fix/global_pcpinfo.txt prepobs_prep.bufrtable=gsi-fix/prepobs_prep.bufrtable berror_stats=hrrr/hrrr_berror_stats_global"
SUBSTITUTE_USENEWGFSBERROR=.false.
