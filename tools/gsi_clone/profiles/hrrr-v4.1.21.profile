# GSI namelist profile: operational HRRR v4.1.21's hybrid analysis
# (NOAA-EMC/HRRR tag v4.1.21), the reference RRFS's localization is compared
# against. Same keys as rrfs-v1.0.25b.profile.
#
# Sources (PUBLIC, read 2026-10-03):
#   NL = parm/conus/hrrr_gsiparm.anl.sh (as run by the HRRR clone: receipt
#        gsiparm.anl_var of the da-gsi-clone lane)
#   EX = scripts/conus/exhrrr_analysis.sh, the values its HRRRDAS branch sets
#        (lines 296-337; beta1_inv at 329, grid_ratio at 333)
PROFILE_NAME=hrrr-v4.1.21
GSI_PIN=gsi-db90edf                  # run on the RRFS pin so both profiles share one executable
STATIC_FIELDS="RDX,RDY,C3H,C4H,C3F,C4F,XLAND"  # GSI WRF converter: cplr_wrf_netcdf_interface.f90:263-433,703

NHR_ASSIMILATION=3                   # NL
MITER=2; NITER1=50; NITER2=50        # NL:3
QOPTION=2; GENCODE=78                # NL
BKGERR_VS=1.0                        # NL:29
BKGERR_HZSCL="0.373,0.746,1.5"       # NL:30
USENEWGFSBERROR=.false.              # NL does not set it: GSI's default
L_HYB_ENS=.true.                     # EX (HRRRDAS members present)
BETA_S0=0.15                         # EX:329 beta1_inv
S_ENS_H="110"                        # NL:144
S_ENS_V="3"                          # NL:144
NSCLGRP=1; NGVARLOC=1; NAENSLOC=1    # GSI defaults: one localization group
R_ENSLOCCOV4TIM=1.0; R_ENSLOCCOV4VAR=1.0; R_ENSLOCCOV4SCL=1.0
READIN_LOCALIZATION=.false.          # NL:144 gives s_ens_h and s_ens_v in the namelist
ASSIGN_VDL_NML=.false.
Q_HYB_ENS=.false.
UV_HYB_ENS=.true.                    # NL:141
L_ENS_IN_DIFF_TIME=.true.            # NL
GRID_RATIO_ENS=1                     # EX
GRID_RATIO=1                         # EX:333 (with HRRRDAS)
OBSQC="dfact=0.75,dfact1=3.0,noiqc=.false.,c_varqc=0.02,vadfile='prepbufr',"
OBS_INPUT="dmesh(1)=120.0,dmesh(2)=60.0,dmesh(3)=30,time_window_max=1.5,time_window_rad=1.0,ext_sonde=.true.,"
# NL:155-174. HRRR runs its cloud analysis inside GSI (i_gsdcldanal_type=5)
# and soil nudging; the db90edf build has no cloud analysis compiled in, so
# the type is 0 here, and soil nudging is off because it reads SOILT1, which
# WOOF's WRF input export does not write.
CLDSURF="dfi_radar_latent_heat_time_period=10.0,metar_impact_radius=20.0,metar_impact_radius_lowCloud=8.0,l_gsd_terrain_match_surfTobs=.true.,l_sfcobserror_ramp_t=.true.,l_sfcobserror_ramp_q=.true.,l_PBL_pseudo_SurfobsT=.true.,l_PBL_pseudo_SurfobsQ=.true.,l_PBL_pseudo_SurfobsUV=.false.,pblH_ration=0.4,pps_press_incr=40.0,l_gsd_limit_ocean_q=.true.,l_pw_hgt_adjust=.true.,l_limit_pw_innov=.true.,max_innov_pct=0.1,l_cleanSnow_WarmTs=.true.,r_cleanSnow_WarmTs_threshold=5.0,l_conserve_thetaV=.true.,i_conserve_thetaV_iternum=3,l_gsd_soilTQ_nudge=.false.,l_cld_bld=.true.,cld_bld_hgt=1200.0,l_numconc=.true.,l_closeobs=.true.,build_cloud_frac_p=0.50,clear_cloud_frac_p=0.10,iclean_hydro_withRef_allcol=1,i_use_2mQ4B=2,i_use_2mT4B=1,i_gsdcldanal_type=0,i_gsdsfc_uselist=1,i_lightpcp=1,i_sfct_gross=1,i_coastline=3,i_gsdqc=2,"
SINGLEOB_MAGINNOV=1.0; SINGLEOB_MAGOBERR=0.8; SINGLEOB_TYPE=t

# HRRR's own table (fix/conus/hrrr_anavinfo_arw_netcdf) with two met_guess
# rows left out because WOOF's WRF input export does not write the field
# each reads: qnc (QNCLOUD) and tsoil (SOILT1). Neither is in the control
# vector, so the analysis variables are HRRR's.
ANAVINFO_TABLE=hrrr/hrrr_anavinfo_arw_netcdf
ANAVINFO_DROP_MET_GUESS="qnc tsoil"
# HRRR's table predates GSI's 2022 rename of the 2 m temperature guess from
# th2m to t2m (GSI db90edf src/gsi/cplr_read_wrf_mass_guess.f90:76, ":1981"),
# so that row takes db90edf's name; the field read is the same (TH2).
ANAVINFO_RENAME_MET_GUESS="th2m:t2m"
FIX_MAP="convinfo=hrrr/hrrr_nam_regional_convinfo errtable=hrrr/hrrr_nam_errtable.r3dv satinfo=hrrr/hrrr_global_satinfo.txt ozinfo=hrrr/hrrr_global_ozinfo.txt pcpinfo=hrrr/hrrr_global_pcpinfo.txt prepobs_prep.bufrtable=gsi-fix/prepobs_prep.bufrtable berror_stats=hrrr/hrrr_berror_stats_global"
SUBSTITUTE_USENEWGFSBERROR=""
