// Generated pointer aliases.  Regenerate with tools/ruc_fused/build_driver.py.
#define D_F(name) d_##name[i]
#define D_P(name,k) d_##name[(k)*n+i]
#define D_DECLARE_SCRATCH float* d_storage = (float*)sp[0]; \
float* d_snow = d_storage+(0+0*RUC_NZS)*n; \
float* d_snowh = d_storage+(1+0*RUC_NZS)*n; \
float* d_snowc = d_storage+(2+0*RUC_NZS)*n; \
float* d_canwat = d_storage+(3+0*RUC_NZS)*n; \
float* d_snoalb = d_storage+(4+0*RUC_NZS)*n; \
float* d_alb = d_storage+(5+0*RUC_NZS)*n; \
float* d_emiss = d_storage+(6+0*RUC_NZS)*n; \
float* d_lai = d_storage+(7+0*RUC_NZS)*n; \
float* d_mavail = d_storage+(8+0*RUC_NZS)*n; \
float* d_sfcexc = d_storage+(9+0*RUC_NZS)*n; \
float* d_z0 = d_storage+(10+0*RUC_NZS)*n; \
float* d_znt = d_storage+(11+0*RUC_NZS)*n; \
float* d_soilt = d_storage+(12+0*RUC_NZS)*n; \
float* d_hfx = d_storage+(13+0*RUC_NZS)*n; \
float* d_qfx = d_storage+(14+0*RUC_NZS)*n; \
float* d_lh = d_storage+(15+0*RUC_NZS)*n; \
float* d_sfcevp = d_storage+(16+0*RUC_NZS)*n; \
float* d_sfcrunoff = d_storage+(17+0*RUC_NZS)*n; \
float* d_udrunoff = d_storage+(18+0*RUC_NZS)*n; \
float* d_acrunoff = d_storage+(19+0*RUC_NZS)*n; \
float* d_grdflx = d_storage+(20+0*RUC_NZS)*n; \
float* d_acsnow = d_storage+(21+0*RUC_NZS)*n; \
float* d_snom = d_storage+(22+0*RUC_NZS)*n; \
float* d_qvg = d_storage+(23+0*RUC_NZS)*n; \
float* d_qcg = d_storage+(24+0*RUC_NZS)*n; \
float* d_dew = d_storage+(25+0*RUC_NZS)*n; \
float* d_qsfc = d_storage+(26+0*RUC_NZS)*n; \
float* d_qsg = d_storage+(27+0*RUC_NZS)*n; \
float* d_chklowq = d_storage+(28+0*RUC_NZS)*n; \
float* d_soilt1 = d_storage+(29+0*RUC_NZS)*n; \
float* d_tsnav = d_storage+(30+0*RUC_NZS)*n; \
float* d_smavail = d_storage+(31+0*RUC_NZS)*n; \
float* d_smmax = d_storage+(32+0*RUC_NZS)*n; \
float* d_rhosnf = d_storage+(33+0*RUC_NZS)*n; \
float* d_precipfr = d_storage+(34+0*RUC_NZS)*n; \
float* d_snowfallac = d_storage+(35+0*RUC_NZS)*n; \
float* d_z3d = d_storage+(36+0*RUC_NZS)*n; \
float* d_p8w = d_storage+(37+0*RUC_NZS)*n; \
float* d_t3d = d_storage+(38+0*RUC_NZS)*n; \
float* d_qv3d = d_storage+(39+0*RUC_NZS)*n; \
float* d_qc3d = d_storage+(40+0*RUC_NZS)*n; \
float* d_rho3d = d_storage+(41+0*RUC_NZS)*n; \
float* d_rainbl = d_storage+(42+0*RUC_NZS)*n; \
float* d_frzfrac = d_storage+(43+0*RUC_NZS)*n; \
float* d_glw = d_storage+(44+0*RUC_NZS)*n; \
float* d_gsw = d_storage+(45+0*RUC_NZS)*n; \
float* d_chs = d_storage+(46+0*RUC_NZS)*n; \
float* d_flqc = d_storage+(47+0*RUC_NZS)*n; \
float* d_flhc = d_storage+(48+0*RUC_NZS)*n; \
float* d_albbck = d_storage+(49+0*RUC_NZS)*n; \
float* d_xland = d_storage+(50+0*RUC_NZS)*n; \
float* d_xice = d_storage+(51+0*RUC_NZS)*n; \
float* d_tbot = d_storage+(52+0*RUC_NZS)*n; \
float* d_shdmin = d_storage+(53+0*RUC_NZS)*n; \
float* d_shdmax = d_storage+(54+0*RUC_NZS)*n; \
float* d_vegfra = d_storage+(55+0*RUC_NZS)*n; \
float* d_rainncv = d_storage+(56+0*RUC_NZS)*n; \
float* d_snowncv = d_storage+(57+0*RUC_NZS)*n; \
float* d_graupelncv = d_storage+(58+0*RUC_NZS)*n; \
float* d_lakemask = d_storage+(59+0*RUC_NZS)*n; \
float* d_psfc = d_storage+(60+0*RUC_NZS)*n; \
float* d_chs2 = d_storage+(61+0*RUC_NZS)*n; \
float* d_cqs2 = d_storage+(62+0*RUC_NZS)*n; \
float* d_cpm = d_storage+(63+0*RUC_NZS)*n; \
float* d_qgh = d_storage+(64+0*RUC_NZS)*n; \
float* d_tsk_save = d_storage+(65+0*RUC_NZS)*n; \
float* d_tsk_sea = d_storage+(66+0*RUC_NZS)*n; \
float* d_flhc_sea = d_storage+(67+0*RUC_NZS)*n; \
float* d_flqc_sea = d_storage+(68+0*RUC_NZS)*n; \
float* d_cpm_sea = d_storage+(69+0*RUC_NZS)*n; \
float* d_cqs2_sea = d_storage+(70+0*RUC_NZS)*n; \
float* d_chs2_sea = d_storage+(71+0*RUC_NZS)*n; \
float* d_chs_sea = d_storage+(72+0*RUC_NZS)*n; \
float* d_qsfc_sea = d_storage+(73+0*RUC_NZS)*n; \
float* d_qgh_sea = d_storage+(74+0*RUC_NZS)*n; \
float* d_hfx_sea = d_storage+(75+0*RUC_NZS)*n; \
float* d_qfx_sea = d_storage+(76+0*RUC_NZS)*n; \
float* d_lh_sea = d_storage+(77+0*RUC_NZS)*n; \
float* d_patm = d_storage+(78+0*RUC_NZS)*n; \
float* d_conflx = d_storage+(79+0*RUC_NZS)*n; \
float* d_prcpms = d_storage+(80+0*RUC_NZS)*n; \
float* d_newsnms = d_storage+(81+0*RUC_NZS)*n; \
float* d_snowrat = d_storage+(82+0*RUC_NZS)*n; \
float* d_grauprat = d_storage+(83+0*RUC_NZS)*n; \
float* d_icerat = d_storage+(84+0*RUC_NZS)*n; \
float* d_curat = d_storage+(85+0*RUC_NZS)*n; \
float* d_qkms = d_storage+(86+0*RUC_NZS)*n; \
float* d_tkms = d_storage+(87+0*RUC_NZS)*n; \
float* d_snwe = d_storage+(88+0*RUC_NZS)*n; \
float* d_snhei = d_storage+(89+0*RUC_NZS)*n; \
float* d_canwatr = d_storage+(90+0*RUC_NZS)*n; \
float* d_snowfrac = d_storage+(91+0*RUC_NZS)*n; \
float* d_rhosnfall = d_storage+(92+0*RUC_NZS)*n; \
float* d_rhosn = d_storage+(93+0*RUC_NZS)*n; \
float* d_emissl = d_storage+(94+0*RUC_NZS)*n; \
float* d_pc = d_storage+(95+0*RUC_NZS)*n; \
float* d_qwrtz = d_storage+(96+0*RUC_NZS)*n; \
float* d_rhocs = d_storage+(97+0*RUC_NZS)*n; \
float* d_bclh = d_storage+(98+0*RUC_NZS)*n; \
float* d_dqm = d_storage+(99+0*RUC_NZS)*n; \
float* d_ksat = d_storage+(100+0*RUC_NZS)*n; \
float* d_psis = d_storage+(101+0*RUC_NZS)*n; \
float* d_qmin = d_storage+(102+0*RUC_NZS)*n; \
float* d_ref = d_storage+(103+0*RUC_NZS)*n; \
float* d_wilt = d_storage+(104+0*RUC_NZS)*n; \
float* d_meltfactor = d_storage+(105+0*RUC_NZS)*n; \
float* d_lmavail = d_storage+(106+0*RUC_NZS)*n; \
float* d_sat = d_storage+(107+0*RUC_NZS)*n; \
float* d_cn = d_storage+(108+0*RUC_NZS)*n; \
float* d_snoh = d_storage+(109+0*RUC_NZS)*n; \
float* d_snflx = d_storage+(110+0*RUC_NZS)*n; \
float* d_s = d_storage+(111+0*RUC_NZS)*n; \
float* d_sublim = d_storage+(112+0*RUC_NZS)*n; \
float* d_evapl = d_storage+(113+0*RUC_NZS)*n; \
float* d_infiltr = d_storage+(114+0*RUC_NZS)*n; \
float* d_smelt = d_storage+(115+0*RUC_NZS)*n; \
float* d_runoff1 = d_storage+(116+0*RUC_NZS)*n; \
float* d_runoff2 = d_storage+(117+0*RUC_NZS)*n; \
float* d_t2 = d_storage+(118+0*RUC_NZS)*n; \
float* d_th2 = d_storage+(119+0*RUC_NZS)*n; \
float* d_q2 = d_storage+(120+0*RUC_NZS)*n; \
float* d_scale = d_storage+(121+0*RUC_NZS)*n; \
float* d_inverse = d_storage+(122+0*RUC_NZS)*n; \
float* d_soilmois = d_storage+(123+0*RUC_NZS)*n; \
float* d_sh2o = d_storage+(123+1*RUC_NZS)*n; \
float* d_tso = d_storage+(123+2*RUC_NZS)*n; \
float* d_smfr3d = d_storage+(123+3*RUC_NZS)*n; \
float* d_keepfr3dflag = d_storage+(123+4*RUC_NZS)*n; \
float* d_soilm1d = d_storage+(123+5*RUC_NZS)*n; \
float* d_tso1d = d_storage+(123+6*RUC_NZS)*n; \
float* d_smfrkeep = d_storage+(123+7*RUC_NZS)*n; \
float* d_keepfr = d_storage+(123+8*RUC_NZS)*n; \
float* d_soiliqw = d_storage+(123+9*RUC_NZS)*n; \
float* d_soilice = d_storage+(123+10*RUC_NZS)*n; \
float* d_seaice = d_storage+(123+11*RUC_NZS)*n;
#define D_DECLARE_OUTPUT const float* o_soilm1d = (const float*)op[0]; \
const float* o_ts1d = (const float*)op[1]; \
const float* o_smfrkeep = (const float*)op[2]; \
const float* o_keepfr = (const float*)op[3]; \
const float* o_soilice = (const float*)op[4]; \
const float* o_soiliqw = (const float*)op[5]; \
const float* o_iland = (const float*)op[6]; \
const float* o_snwe = (const float*)op[7]; \
const float* o_snhei = (const float*)op[8]; \
const float* o_snowfrac = (const float*)op[9]; \
const float* o_rhosn = (const float*)op[10]; \
const float* o_rhonewsn = (const float*)op[11]; \
const float* o_rhosnfall = (const float*)op[12]; \
const float* o_snowrat = (const float*)op[13]; \
const float* o_grauprat = (const float*)op[14]; \
const float* o_icerat = (const float*)op[15]; \
const float* o_curat = (const float*)op[16]; \
const float* o_alb_snow = (const float*)op[17]; \
const float* o_emiss = (const float*)op[18]; \
const float* o_mavail = (const float*)op[19]; \
const float* o_alb = (const float*)op[20]; \
const float* o_cst = (const float*)op[21]; \
const float* o_znt = (const float*)op[22]; \
const float* o_soilt = (const float*)op[23]; \
const float* o_soilt1 = (const float*)op[24]; \
const float* o_tsnav = (const float*)op[25]; \
const float* o_dew = (const float*)op[26]; \
const float* o_qvg = (const float*)op[27]; \
const float* o_qsg = (const float*)op[28]; \
const float* o_qcg = (const float*)op[29]; \
const float* o_smelt = (const float*)op[30]; \
const float* o_snoh = (const float*)op[31]; \
const float* o_snflx = (const float*)op[32]; \
const float* o_snom = (const float*)op[33]; \
const float* o_snowfallac = (const float*)op[34]; \
const float* o_acsnow = (const float*)op[35]; \
const float* o_edir1 = (const float*)op[36]; \
const float* o_ec1 = (const float*)op[37]; \
const float* o_ett1 = (const float*)op[38]; \
const float* o_eeta = (const float*)op[39]; \
const float* o_qfx = (const float*)op[40]; \
const float* o_hfx = (const float*)op[41]; \
const float* o_s = (const float*)op[42]; \
const float* o_sublim = (const float*)op[43]; \
const float* o_evapl = (const float*)op[44]; \
const float* o_prcpl = (const float*)op[45]; \
const float* o_fltot = (const float*)op[46]; \
const float* o_runoff1 = (const float*)op[47]; \
const float* o_runoff2 = (const float*)op[48]; \
const float* o_infiltr = (const float*)op[49]; \
const float* o_smf = (const float*)op[50]; \
const float* o_rsm = (const float*)op[51]; \
const float* o_snweprint = (const float*)op[52]; \
const float* o_snheiprint = (const float*)op[53]; \
const float* o_ilnb = (const float*)op[54];
#define D_COPY_INPUT D_F(snow)=((const float*)ip[0])[i]; \
D_F(snowh)=((const float*)ip[1])[i]; \
D_F(snowc)=((const float*)ip[2])[i]; \
D_F(canwat)=((const float*)ip[3])[i]; \
D_F(snoalb)=((const float*)ip[4])[i]; \
D_F(alb)=((const float*)ip[5])[i]; \
D_F(emiss)=((const float*)ip[6])[i]; \
D_F(lai)=((const float*)ip[7])[i]; \
D_F(mavail)=((const float*)ip[8])[i]; \
D_F(sfcexc)=((const float*)ip[9])[i]; \
D_F(z0)=((const float*)ip[10])[i]; \
D_F(znt)=((const float*)ip[11])[i]; \
D_F(soilt)=((const float*)ip[12])[i]; \
D_F(hfx)=((const float*)ip[13])[i]; \
D_F(qfx)=((const float*)ip[14])[i]; \
D_F(lh)=((const float*)ip[15])[i]; \
D_F(sfcevp)=((const float*)ip[16])[i]; \
D_F(sfcrunoff)=((const float*)ip[17])[i]; \
D_F(udrunoff)=((const float*)ip[18])[i]; \
D_F(acrunoff)=((const float*)ip[19])[i]; \
D_F(grdflx)=((const float*)ip[20])[i]; \
D_F(acsnow)=((const float*)ip[21])[i]; \
D_F(snom)=((const float*)ip[22])[i]; \
D_F(qvg)=((const float*)ip[23])[i]; \
D_F(qcg)=((const float*)ip[24])[i]; \
D_F(dew)=((const float*)ip[25])[i]; \
D_F(qsfc)=((const float*)ip[26])[i]; \
D_F(qsg)=((const float*)ip[27])[i]; \
D_F(chklowq)=((const float*)ip[28])[i]; \
D_F(soilt1)=((const float*)ip[29])[i]; \
D_F(tsnav)=((const float*)ip[30])[i]; \
D_F(smavail)=((const float*)ip[31])[i]; \
D_F(smmax)=((const float*)ip[32])[i]; \
D_F(rhosnf)=((const float*)ip[33])[i]; \
D_F(precipfr)=((const float*)ip[34])[i]; \
D_F(snowfallac)=((const float*)ip[35])[i]; \
D_F(z3d)=((const float*)ip[36])[i]; \
D_F(p8w)=((const float*)ip[37])[i]; \
D_F(t3d)=((const float*)ip[38])[i]; \
D_F(qv3d)=((const float*)ip[39])[i]; \
D_F(qc3d)=((const float*)ip[40])[i]; \
D_F(rho3d)=((const float*)ip[41])[i]; \
D_F(rainbl)=((const float*)ip[42])[i]; \
D_F(frzfrac)=((const float*)ip[43])[i]; \
D_F(glw)=((const float*)ip[44])[i]; \
D_F(gsw)=((const float*)ip[45])[i]; \
D_F(chs)=((const float*)ip[46])[i]; \
D_F(flqc)=((const float*)ip[47])[i]; \
D_F(flhc)=((const float*)ip[48])[i]; \
D_F(albbck)=((const float*)ip[49])[i]; \
D_F(xland)=((const float*)ip[50])[i]; \
D_F(xice)=((const float*)ip[51])[i]; \
D_F(tbot)=((const float*)ip[52])[i]; \
D_F(shdmin)=((const float*)ip[53])[i]; \
D_F(shdmax)=((const float*)ip[54])[i]; \
D_F(vegfra)=((const float*)ip[55])[i]; \
D_F(rainncv)=((const float*)ip[56])[i]; \
D_F(snowncv)=((const float*)ip[57])[i]; \
D_F(graupelncv)=((const float*)ip[58])[i]; \
D_F(lakemask)=((const float*)ip[59])[i]; \
D_F(psfc)=((const float*)ip[60])[i]; \
D_F(chs2)=((const float*)ip[61])[i]; \
D_F(cqs2)=((const float*)ip[62])[i]; \
D_F(cpm)=((const float*)ip[63])[i]; \
D_F(qgh)=((const float*)ip[64])[i]; \
D_F(tsk_save)=((const float*)ip[65])[i]; \
D_F(tsk_sea)=((const float*)ip[66])[i]; \
D_F(flhc_sea)=((const float*)ip[67])[i]; \
D_F(flqc_sea)=((const float*)ip[68])[i]; \
D_F(cpm_sea)=((const float*)ip[69])[i]; \
D_F(cqs2_sea)=((const float*)ip[70])[i]; \
D_F(chs2_sea)=((const float*)ip[71])[i]; \
D_F(chs_sea)=((const float*)ip[72])[i]; \
D_F(qsfc_sea)=((const float*)ip[73])[i]; \
D_F(qgh_sea)=((const float*)ip[74])[i]; \
D_F(hfx_sea)=((const float*)ip[75])[i]; \
D_F(qfx_sea)=((const float*)ip[76])[i]; \
D_F(lh_sea)=((const float*)ip[77])[i]; \
for (int k=0;k<RUC_NZS;++k) D_P(soilmois,k)=((const float*)ip[78])[k*n+i]; \
for (int k=0;k<RUC_NZS;++k) D_P(sh2o,k)=((const float*)ip[79])[k*n+i]; \
for (int k=0;k<RUC_NZS;++k) D_P(tso,k)=((const float*)ip[80])[k*n+i]; \
for (int k=0;k<RUC_NZS;++k) D_P(smfr3d,k)=((const float*)ip[81])[k*n+i]; \
for (int k=0;k<RUC_NZS;++k) D_P(keepfr3dflag,k)=((const float*)ip[82])[k*n+i];
#define D_ADMIT for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(soilmois,k))) { d_flag(flags,0); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(sh2o,k))) { d_flag(flags,1); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(tso,k))) { d_flag(flags,2); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(smfr3d,k))) { d_flag(flags,3); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(keepfr3dflag,k))) { d_flag(flags,4); admitted=false; } \
if (!isfinite(D_F(snow))) { d_flag(flags,5); admitted=false; } \
if (!isfinite(D_F(snowh))) { d_flag(flags,6); admitted=false; } \
if (!isfinite(D_F(snowc))) { d_flag(flags,7); admitted=false; } \
if (!isfinite(D_F(canwat))) { d_flag(flags,8); admitted=false; } \
if (!isfinite(D_F(snoalb))) { d_flag(flags,9); admitted=false; } \
if (!isfinite(D_F(alb))) { d_flag(flags,10); admitted=false; } \
if (!isfinite(D_F(emiss))) { d_flag(flags,11); admitted=false; } \
if (!isfinite(D_F(lai))) { d_flag(flags,12); admitted=false; } \
if (!isfinite(D_F(mavail))) { d_flag(flags,13); admitted=false; } \
if (!isfinite(D_F(sfcexc))) { d_flag(flags,14); admitted=false; } \
if (!isfinite(D_F(z0))) { d_flag(flags,15); admitted=false; } \
if (!isfinite(D_F(znt))) { d_flag(flags,16); admitted=false; } \
if (!isfinite(D_F(soilt))) { d_flag(flags,17); admitted=false; } \
if (!isfinite(D_F(hfx))) { d_flag(flags,18); admitted=false; } \
if (!isfinite(D_F(qfx))) { d_flag(flags,19); admitted=false; } \
if (!isfinite(D_F(lh))) { d_flag(flags,20); admitted=false; } \
if (!isfinite(D_F(sfcevp))) { d_flag(flags,21); admitted=false; } \
if (!isfinite(D_F(sfcrunoff))) { d_flag(flags,22); admitted=false; } \
if (!isfinite(D_F(udrunoff))) { d_flag(flags,23); admitted=false; } \
if (!isfinite(D_F(acrunoff))) { d_flag(flags,24); admitted=false; } \
if (!isfinite(D_F(grdflx))) { d_flag(flags,25); admitted=false; } \
if (!isfinite(D_F(acsnow))) { d_flag(flags,26); admitted=false; } \
if (!isfinite(D_F(snom))) { d_flag(flags,27); admitted=false; } \
if (!isfinite(D_F(qvg))) { d_flag(flags,28); admitted=false; } \
if (!isfinite(D_F(qcg))) { d_flag(flags,29); admitted=false; } \
if (!isfinite(D_F(dew))) { d_flag(flags,30); admitted=false; } \
if (!isfinite(D_F(qsfc))) { d_flag(flags,31); admitted=false; } \
if (!isfinite(D_F(qsg))) { d_flag(flags,32); admitted=false; } \
if (!isfinite(D_F(chklowq))) { d_flag(flags,33); admitted=false; } \
if (!isfinite(D_F(soilt1))) { d_flag(flags,34); admitted=false; } \
if (!isfinite(D_F(tsnav))) { d_flag(flags,35); admitted=false; } \
if (!isfinite(D_F(smavail))) { d_flag(flags,36); admitted=false; } \
if (!isfinite(D_F(smmax))) { d_flag(flags,37); admitted=false; } \
if (!isfinite(D_F(rhosnf))) { d_flag(flags,38); admitted=false; } \
if (!isfinite(D_F(precipfr))) { d_flag(flags,39); admitted=false; } \
if (!isfinite(D_F(snowfallac))) { d_flag(flags,40); admitted=false; } \
if (!isfinite(D_F(z3d))) { d_flag(flags,41); admitted=false; } \
if (!isfinite(D_F(p8w))) { d_flag(flags,42); admitted=false; } \
if (!isfinite(D_F(t3d))) { d_flag(flags,43); admitted=false; } \
if (!isfinite(D_F(qv3d))) { d_flag(flags,44); admitted=false; } \
if (!isfinite(D_F(qc3d))) { d_flag(flags,45); admitted=false; } \
if (!isfinite(D_F(rho3d))) { d_flag(flags,46); admitted=false; } \
if (!isfinite(D_F(rainbl))) { d_flag(flags,47); admitted=false; } \
if (!isfinite(D_F(frzfrac))) { d_flag(flags,48); admitted=false; } \
if (!isfinite(D_F(glw))) { d_flag(flags,49); admitted=false; } \
if (!isfinite(D_F(gsw))) { d_flag(flags,50); admitted=false; } \
if (!isfinite(D_F(chs))) { d_flag(flags,51); admitted=false; } \
if (!isfinite(D_F(flqc))) { d_flag(flags,52); admitted=false; } \
if (!isfinite(D_F(flhc))) { d_flag(flags,53); admitted=false; } \
if (!isfinite(D_F(albbck))) { d_flag(flags,54); admitted=false; } \
if (!isfinite(D_F(xland))) { d_flag(flags,55); admitted=false; } \
if (!isfinite(D_F(xice))) { d_flag(flags,56); admitted=false; } \
if (!isfinite(D_F(tbot))) { d_flag(flags,57); admitted=false; } \
if (!isfinite(D_F(shdmin))) { d_flag(flags,58); admitted=false; } \
if (!isfinite(D_F(shdmax))) { d_flag(flags,59); admitted=false; } \
if (!isfinite(D_F(vegfra))) { d_flag(flags,60); admitted=false; } \
if (!isfinite(D_F(rainncv))) { d_flag(flags,61); admitted=false; } \
if (!isfinite(D_F(snowncv))) { d_flag(flags,62); admitted=false; } \
if (!isfinite(D_F(graupelncv))) { d_flag(flags,63); admitted=false; } \
if (!isfinite(D_F(lakemask))) { d_flag(flags,64); admitted=false; }
#define D_CAT_INPUT 83
#define D_CAT_FLAG 65
#define D_CHECK_OUTPUT if (!isfinite(D_F(snow))) d_flag(flags,1056); \
if (!isfinite(D_F(snowh))) d_flag(flags,1057); \
if (!isfinite(D_F(snowc))) d_flag(flags,1058); \
if (!isfinite(D_F(canwat))) d_flag(flags,1059); \
if (!isfinite(D_F(snoalb))) d_flag(flags,1060); \
if (!isfinite(D_F(alb))) d_flag(flags,1061); \
if (!isfinite(D_F(emiss))) d_flag(flags,1062); \
if (!isfinite(D_F(lai))) d_flag(flags,1063); \
if (!isfinite(D_F(mavail))) d_flag(flags,1064); \
if (!isfinite(D_F(sfcexc))) d_flag(flags,1065); \
if (!isfinite(D_F(z0))) d_flag(flags,1066); \
if (!isfinite(D_F(znt))) d_flag(flags,1067); \
if (!isfinite(D_F(soilt))) d_flag(flags,1068); \
if (!isfinite(D_F(hfx))) d_flag(flags,1069); \
if (!isfinite(D_F(qfx))) d_flag(flags,1070); \
if (!isfinite(D_F(lh))) d_flag(flags,1071); \
if (!isfinite(D_F(sfcevp))) d_flag(flags,1072); \
if (!isfinite(D_F(sfcrunoff))) d_flag(flags,1073); \
if (!isfinite(D_F(udrunoff))) d_flag(flags,1074); \
if (!isfinite(D_F(acrunoff))) d_flag(flags,1075); \
if (!isfinite(D_F(grdflx))) d_flag(flags,1076); \
if (!isfinite(D_F(acsnow))) d_flag(flags,1077); \
if (!isfinite(D_F(snom))) d_flag(flags,1078); \
if (!isfinite(D_F(qvg))) d_flag(flags,1079); \
if (!isfinite(D_F(qcg))) d_flag(flags,1080); \
if (!isfinite(D_F(dew))) d_flag(flags,1081); \
if (!isfinite(D_F(qsfc))) d_flag(flags,1082); \
if (!isfinite(D_F(qsg))) d_flag(flags,1083); \
if (!isfinite(D_F(chklowq))) d_flag(flags,1084); \
if (!isfinite(D_F(soilt1))) d_flag(flags,1085); \
if (!isfinite(D_F(tsnav))) d_flag(flags,1086); \
if (!isfinite(D_F(smavail))) d_flag(flags,1087); \
if (!isfinite(D_F(smmax))) d_flag(flags,1088); \
if (!isfinite(D_F(rhosnf))) d_flag(flags,1089); \
if (!isfinite(D_F(precipfr))) d_flag(flags,1090); \
if (!isfinite(D_F(snowfallac))) d_flag(flags,1091); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(soilmois,k))) d_flag(flags,1092); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(sh2o,k))) d_flag(flags,1093); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(tso,k))) d_flag(flags,1094); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(smfr3d,k))) d_flag(flags,1095); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P(keepfr3dflag,k))) d_flag(flags,1096); \
if (!isfinite(D_F(infiltr))) d_flag(flags,1097); \
if (!isfinite(D_F(smelt))) d_flag(flags,1098); \
if (!isfinite(D_F(runoff1))) d_flag(flags,1099); \
if (!isfinite(D_F(runoff2))) d_flag(flags,1100);
#define D_COMMIT ((float*)cp[0])[i]=D_F(snow); \
((float*)cp[1])[i]=D_F(snowh); \
((float*)cp[2])[i]=D_F(snowc); \
((float*)cp[3])[i]=D_F(canwat); \
((float*)cp[4])[i]=D_F(snoalb); \
((float*)cp[5])[i]=D_F(alb); \
((float*)cp[6])[i]=D_F(emiss); \
((float*)cp[7])[i]=D_F(lai); \
((float*)cp[8])[i]=D_F(mavail); \
((float*)cp[9])[i]=D_F(sfcexc); \
((float*)cp[10])[i]=D_F(z0); \
((float*)cp[11])[i]=D_F(znt); \
((float*)cp[12])[i]=D_F(soilt); \
((float*)cp[13])[i]=D_F(hfx); \
((float*)cp[14])[i]=D_F(qfx); \
((float*)cp[15])[i]=D_F(lh); \
((float*)cp[16])[i]=D_F(sfcevp); \
((float*)cp[17])[i]=D_F(sfcrunoff); \
((float*)cp[18])[i]=D_F(udrunoff); \
((float*)cp[19])[i]=D_F(acrunoff); \
((float*)cp[20])[i]=D_F(grdflx); \
((float*)cp[21])[i]=D_F(acsnow); \
((float*)cp[22])[i]=D_F(snom); \
((float*)cp[23])[i]=D_F(qvg); \
((float*)cp[24])[i]=D_F(qcg); \
((float*)cp[25])[i]=D_F(dew); \
((float*)cp[26])[i]=D_F(qsfc); \
((float*)cp[27])[i]=D_F(qsg); \
((float*)cp[28])[i]=D_F(chklowq); \
((float*)cp[29])[i]=D_F(soilt1); \
((float*)cp[30])[i]=D_F(tsnav); \
((float*)cp[31])[i]=D_F(smavail); \
((float*)cp[32])[i]=D_F(smmax); \
((float*)cp[33])[i]=D_F(rhosnf); \
((float*)cp[34])[i]=D_F(precipfr); \
((float*)cp[35])[i]=D_F(snowfallac); \
((float*)cp[36])[i]=D_F(albbck); \
((float*)cp[37])[i]=D_F(chs); \
((float*)cp[38])[i]=D_F(flhc); \
((float*)cp[39])[i]=D_F(flqc); \
((float*)cp[40])[i]=D_F(psfc); \
((float*)cp[41])[i]=D_F(chs2); \
((float*)cp[42])[i]=D_F(cqs2); \
((float*)cp[43])[i]=D_F(cpm); \
((float*)cp[44])[i]=D_F(qgh); \
((float*)cp[45])[i]=D_F(tsk_save); \
((float*)cp[46])[i]=D_F(tsk_sea); \
((float*)cp[47])[i]=D_F(flhc_sea); \
((float*)cp[48])[i]=D_F(flqc_sea); \
((float*)cp[49])[i]=D_F(cpm_sea); \
((float*)cp[50])[i]=D_F(cqs2_sea); \
((float*)cp[51])[i]=D_F(chs2_sea); \
((float*)cp[52])[i]=D_F(chs_sea); \
((float*)cp[53])[i]=D_F(qsfc_sea); \
((float*)cp[54])[i]=D_F(qgh_sea); \
((float*)cp[55])[i]=D_F(hfx_sea); \
((float*)cp[56])[i]=D_F(qfx_sea); \
((float*)cp[57])[i]=D_F(lh_sea); \
for (int k=0;k<RUC_NZS;++k) ((float*)cp[58])[k*n+i]=D_P(soilmois,k); \
for (int k=0;k<RUC_NZS;++k) ((float*)cp[59])[k*n+i]=D_P(sh2o,k); \
for (int k=0;k<RUC_NZS;++k) ((float*)cp[60])[k*n+i]=D_P(tso,k); \
for (int k=0;k<RUC_NZS;++k) ((float*)cp[61])[k*n+i]=D_P(smfr3d,k); \
for (int k=0;k<RUC_NZS;++k) ((float*)cp[62])[k*n+i]=D_P(keepfr3dflag,k); \
((float*)cp[63])[i]=D_F(infiltr); \
((float*)cp[64])[i]=D_F(smelt); \
((float*)cp[65])[i]=D_F(runoff1); \
((float*)cp[66])[i]=D_F(runoff2); \
((float*)cp[67])[i]=D_F(t2); \
((float*)cp[68])[i]=D_F(th2); \
((float*)cp[69])[i]=D_F(q2);

// Full-width RUC driver.  Scratch arrays keep soil columns out of local frames.
#ifndef NAN
#define NAN __int_as_float(0x7fc00000)
#endif
__device__ __forceinline__ float d_min(float a,float b) {
    return (isnan(a) || isnan(b)) ? NAN : fminf(a,b);
}
__device__ __forceinline__ float d_max(float a,float b) {
    return (isnan(a) || isnan(b)) ? NAN : fmaxf(a,b);
}
__device__ __forceinline__ float d_npmin(float a,float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a<b ? a:b));
}
__device__ __forceinline__ float d_npmax(float a,float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a>b ? a:b));
}
__device__ __forceinline__ void d_flag(unsigned* flags,int bit) {
    atomicOr(flags+bit/32,1u<<(bit%32));
}
#define A(a,b) __fadd_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define B(a,b) __fsub_rn((a),(b))
#define Q(a,b) __fdiv_rn((a),(b))
__device__ __forceinline__ float d_qsn(float temperature,const float* tbq,
                                     unsigned* flags,int first,bool* admitted=nullptr) {
    if(!isfinite(temperature)) {
        d_flag(flags,first);
        if(admitted) *admitted=false;
    }
    float raw=A(Q(B(temperature,173.15f),0.05f),1.0f);
    if(!(fabsf(raw)<2147483648.0f)) {
        d_flag(flags,first+1);
        if(admitted) *admitted=false;
        return tbq[0];
    }
    return ruc_qsn_lookup(temperature,tbq);
}

extern "C" __global__ void ruc_driver_prologue(
    const unsigned long long* ip, const unsigned long long* sp,
    int* integer, bool* run, unsigned* flags,
    const int* ifortbl, const float* z0tbl, const float* lemitbl,
    const float* pctbl, const float* laitbl, const float* bb,
    const float* drysmc, const float* hc, const float* maxsmc,
    const float* refsmc, const float* satpsi, const float* satdk,
    const float* wltsmc, const float* qtz, const float* tbq,
    int n,int ktau,float dt,int iswater,int isice,int nv,int ns,
    float icealbedo,float cn,
    const float* landusef,const float* soilctop,int nlcat,int nscat,
    int mosaic_lu,int mosaic_soil,int lakemodel,
    int qvg_air,int rdlai2d,float xice_threshold) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    D_DECLARE_SCRATCH
    D_COPY_INPUT
    // xice_threshold: module_surface_driver.F:1365-1368, 0.5 or 0.02 by
    // fractional_seaice; the host hands the run's value to every ice test.
    bool component=D_F(xice)>=xice_threshold && D_F(xice)<=1.0f;
    if(component) {
        D_F(albbck)=icealbedo;
        D_F(alb)=Q(B(D_F(alb),M(B(1.0f,D_F(xice)),0.08f)),D_F(xice));
        D_F(emiss)=Q(B(D_F(emiss),M(B(1.0f,D_F(xice)),0.98f)),D_F(xice));
        D_F(soilt)=D_F(tsk_save);
    }
    bool admitted=true;
    D_ADMIT
    int veg=((const int*)ip[D_CAT_INPUT])[i];
    int soil=((const int*)ip[D_CAT_INPUT+1])[i];
    if(veg<1 || veg>nv) { d_flag(flags,D_CAT_FLAG); admitted=false; veg=1; }
    if(soil<1 || soil>ns) { d_flag(flags,D_CAT_FLAG+1); admitted=false; soil=1; }
    integer[i]=veg; integer[n+i]=soil;
    if(ktau==1) {
        for(int k=0;k<RUC_NZS;++k) D_P(keepfr3dflag,k)=0.0f;
        float blended=M(0.5f,A(D_F(soilt),D_P(tso,0)));
        // Snow by lineage (gpuwm.core.ruc_tier RUC_SNOW_FORMS).  wrf_45, the
        // operational RAP/HRRR branch's LSMRUC:459-465: snow water with no
        // cover starts at min(1, snow/32), written as the exact snow*2**-5,
        // and the inside-snow repair blends above 32 mm of snow water.
        // GPUWM_SNOW_WRF461: WRF v4.6.1 blends wherever snowc > 0.
#ifdef GPUWM_SNOW_WRF461
        bool inside_snow=D_F(snowc)>0.0f;
#else
        if(D_F(snow)>0.0f && D_F(snowc)<=0.0f) D_F(snowc)=d_min(1.0f,M(D_F(snow),0.03125f));
        bool inside_snow=D_F(snow)>32.0f;
#endif
        if(D_F(soilt1)<170.0f || D_F(soilt1)>400.0f)
            D_F(soilt1)=inside_snow ? blended:D_P(tso,0);
        D_F(tsnav)=B(blended,273.15f);
        D_F(qsg)=Q(d_qsn(D_F(soilt),tbq,flags,80,&admitted),M(D_F(p8w),1.0e-2f));
        // QVG/QCG cold start by lineage (gpuwm.core.ruc_tier
        // RUC_QVG_COLD_START_FORMS).  qvg_air: the operational RAP/HRRR branch's
        // LSMRUC:479-483, the lowest-level vapour with no ground condensate.
        // Otherwise public WRF v4.6.1 LSMRUC:505-514, saturation at the
        // skin times moisture availability, condensate from the air.
        if(qvg_air) {
            if(D_F(qvg)<=0.0f || D_F(qvg)>0.1f) { D_F(qvg)=D_F(qv3d); D_F(qcg)=0.0f; }
        } else {
            if(D_F(qcg)<0.0f || D_F(qcg)>0.1f) D_F(qcg)=D_F(qc3d);
            if(D_F(qvg)<=0.0f || D_F(qvg)>0.1f) D_F(qvg)=M(D_F(qsg),D_F(mavail));
        }
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg)));
        D_F(snom)=D_F(snowfallac)=D_F(precipfr)=D_F(dew)=0.0f;
        D_F(sfcrunoff)=D_F(udrunoff)=D_F(acrunoff)=0.0f;
        D_F(rhosnf)=-1000.0f; D_F(chklowq)=1.0f;
    }
    D_F(patm)=M(D_F(p8w),1.0e-5f);
    D_F(conflx)=M(D_F(z3d),0.5f);
    float resolved_liquid=M(D_F(rainncv),B(1.0f,D_F(frzfrac)));
    float resolved_frozen=M(D_F(rainncv),D_F(frzfrac));
    float convective=B(D_F(rainbl),D_F(rainncv));
    bool cold=D_F(t3d)<273.0f;
    bool mixed_cold=D_F(frzfrac)>0.0f && cold;
    float convective_liquid=mixed_cold ? d_max(0.0f,M(convective,B(1.0f,D_F(frzfrac)))) : (cold ? 0.0f:d_max(0.0f,convective));
    float convective_frozen=mixed_cold ? d_max(0.0f,M(convective,D_F(frzfrac))) : (cold ? d_max(0.0f,convective):0.0f);
    D_F(prcpms)=M(Q(A(resolved_liquid,convective_liquid),dt),1.0e-3f);
    D_F(newsnms)=M(Q(A(resolved_frozen,convective_frozen),dt),1.0e-3f);
    float frozen_total=A(resolved_frozen,convective_frozen);
    bool falling=frozen_total>0.0f;
    float denominator=falling ? frozen_total:1.0f;
    D_F(snowrat)=falling ? d_min(1.0f,d_max(0.0f,Q(D_F(snowncv),denominator))):0.0f;
    D_F(grauprat)=falling ? d_min(1.0f,d_max(0.0f,Q(D_F(graupelncv),denominator))):0.0f;
    D_F(icerat)=falling ? d_min(1.0f,d_max(0.0f,Q(B(B(resolved_frozen,D_F(snowncv)),D_F(graupelncv)),denominator))):0.0f;
    D_F(curat)=falling ? d_min(1.0f,d_max(0.0f,Q(convective_frozen,denominator))):0.0f;
    D_F(precipfr)=M(M(D_F(newsnms),dt),1000.0f);
    D_F(qkms)=Q(Q(D_F(flqc),D_F(rho3d)),D_F(mavail));
    D_F(tkms)=Q(Q(D_F(flhc),D_F(rho3d)),M(1004.5f,A(1.0f,M(0.84f,D_F(qv3d)))));
    D_F(snwe)=M(D_F(snow),1.0e-3f); D_F(snhei)=D_F(snowh);
    D_F(canwatr)=M(D_F(canwat),1.0e-3f); D_F(snowfrac)=D_F(snowc);
    D_F(rhosnfall)=D_F(rhosnf);
    D_F(rhosn)=D_F(snow)>0.0f && D_F(snowh)>0.0f ? Q(D_F(snow),D_F(snowh)):300.0f;
    // The driver version of SOILVEGIN.  The old leaf's fmin/fmax omit NaN
    // propagation.  These operands have been admitted, but derived NaNs must
    // still match the driver's CuPy minimum and maximum.
    int forest=ifortbl[veg-1];
    float range=B(D_F(shdmax),D_F(shdmin));
    float ratio=Q(B(D_F(vegfra),D_F(shdmin)),d_max(1.0f,range));
    float bounded=d_max(0.0f,d_min(1.0f,ratio));
    float factor=range<1.0f ? 1.0f:B(1.0f,bounded);
    float scaled=M(0.8f,laitbl[veg-1]), delta=0.0f;
    if(forest==1) delta=d_min(0.2f,scaled);
    if(forest==2 || forest==7) delta=d_min(0.5f,scaled);
    if(forest==3) delta=d_min(0.45f,scaled);
    if(forest==4) delta=d_min(0.75f,scaled);
    if(forest==5) delta=d_min(0.86f,scaled);
    float incoming_znt=D_F(znt);
    // module_sf_ruclsm.F:7075 ``if(.not.rdlai2d) LAI = LAItoday(IVGTYP)``:
    // under rdlai2d the prescribed 2-D LAI (the monthly field) stays.
    if(!rdlai2d) D_F(lai)=veg==iswater ? laitbl[veg-1]:B(laitbl[veg-1],M(delta,factor));
    if(veg!=iswater) D_F(znt)=forest==7 ? B(z0tbl[veg-1],M(0.125f,factor)):z0tbl[veg-1];
    D_F(emissl)=lemitbl[veg-1]; D_F(pc)=pctbl[veg-1];
    D_F(qwrtz)=D_F(rhocs)=D_F(bclh)=D_F(dqm)=D_F(ksat)=D_F(psis)=D_F(qmin)=D_F(ref)=D_F(wilt)=0.0f;
    if(soil!=14) {
        D_F(qwrtz)=qtz[soil-1]; D_F(rhocs)=M(hc[soil-1],1.0e6f);
        D_F(bclh)=bb[soil-1]; D_F(dqm)=B(maxsmc[soil-1],drysmc[soil-1]);
        D_F(ksat)=satdk[soil-1]; D_F(psis)=-satpsi[soil-1];
        D_F(qmin)=drysmc[soil-1]; D_F(ref)=refsmc[soil-1]; D_F(wilt)=wltsmc[soil-1];
    }
    ruc_mosaic_parameters(i,n,nlcat,nscat,mosaic_lu,mosaic_soil,
        landusef,soilctop,soil,iswater,rdlai2d!=0,factor,incoming_znt,
        ifortbl,z0tbl,lemitbl,pctbl,laitbl,bb,drysmc,hc,maxsmc,
        refsmc,satpsi,satdk,wltsmc,qtz,
        D_F(emissl),D_F(pc),D_F(znt),D_F(lai),D_F(qwrtz),
        D_F(rhocs),D_F(bclh),D_F(dqm),D_F(ksat),D_F(psis),
        D_F(qmin),D_F(ref),D_F(wilt));
    bool forested=forest>2;
    D_F(meltfactor)=forested ? 2.0f:0.85f;
    integer[3*n+i]=4;
    for(int k=1;k<RUC_NZS;++k) if(ruc_soil_layer_depth[k]>=(forested ? 0.4f:1.1f)) {
        integer[3*n+i]=k+1; break;
    }
    bool lake=lakemodel==1 && D_F(lakemask)==1.0f;
    bool water=B(D_F(xland),1.5f)>=0.0f && !lake;
    bool land=!(B(D_F(xland),1.5f)>=0.0f || lake);
    bool ice=land && D_F(xice)>=xice_threshold;
    D_F(seaice)=D_F(xice)>=xice_threshold ? 1.0f:0.0f;
    integer[2*n+i]=ice ? isice:veg; integer[4*n+i]=1;
    if(water) {
        D_F(smavail)=D_F(smmax)=1.0f; D_F(snow)=D_F(snowh)=D_F(snowc)=0.0f;
        D_F(qvg)=Q(d_qsn(D_F(soilt),tbq,flags,82,&admitted),M(D_F(p8w),1.0e-2f));
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg))); D_F(chklowq)=1.0f;
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilmois,k)=D_P(sh2o,k)=1.0f; D_P(tso,k)=D_F(soilt);
        }
    }
    if(ice) {
        D_F(znt)=0.011f; D_F(snoalb)=0.75f; D_F(dqm)=D_F(ref)=1.0f;
        D_F(qmin)=D_F(wilt)=0.0f; D_F(emissl)=0.98f;
        D_F(qvg)=D_F(qsg)=Q(d_qsn(D_F(soilt),tbq,flags,84,&admitted),M(D_F(p8w),1.0e-2f));
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg)));
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilmois,k)=D_P(smfr3d,k)=1.0f;
            D_P(sh2o,k)=D_P(keepfr3dflag,k)=0.0f;
            D_P(tso,k)=d_min(271.4f,D_P(tso,k));
        }
    }
    for(int k=0;k<RUC_NZS;++k) {
        D_P(soilm1d,k)=d_min(d_max(0.0f,B(D_P(soilmois,k),D_F(qmin))),D_F(dqm));
        D_P(tso1d,k)=D_P(tso,k);
        D_P(soiliqw,k)=d_min(d_max(0.0f,B(D_P(sh2o,k),D_F(qmin))),D_P(soilm1d,k));
        D_P(soilice,k)=Q(B(D_P(soilm1d,k),D_P(soiliqw,k)),0.9f);
        D_P(smfrkeep,k)=D_P(smfr3d,k); D_P(keepfr,k)=D_P(keepfr3dflag,k);
    }
    D_F(lmavail)=land ? d_max(0.00001f,d_min(1.0f,Q(D_P(soilm1d,0),B(D_F(ref),D_F(qmin))))):0.0f;
    D_F(sat)=5.0e-4f; D_F(cn)=cn;
    D_F(snoh)=D_F(snflx)=D_F(s)=D_F(sublim)=D_F(evapl)=0.0f;
    D_F(infiltr)=D_F(smelt)=D_F(runoff1)=D_F(runoff2)=0.0f;
    run[i]=land && admitted;
    atomicAdd(flags+36,land && !ice ? 1u:0u);
    atomicAdd(flags+37,water ? 1u:0u);
    atomicAdd(flags+38,lake ? 1u:0u);
    atomicAdd(flags+39,ice ? 1u:0u);
}

// NumPy float32 arithmetic retains an operand NaN's payload and quiets it.
// Invalid arithmetic without NaN operands produces its negative quiet NaN.
// CUDA arithmetic canonicalizes both, so retain these words explicitly in
// the host diagnostic transcription.  Finite arithmetic remains one IEEE op.
__device__ __forceinline__ float d_np_result(float a,float b,float result) {
    if(isnan(a)) return __int_as_float(__float_as_int(a) | 0x00400000);
    if(isnan(b)) return __int_as_float(__float_as_int(b) | 0x00400000);
    return isnan(result) ? __int_as_float(0xffc00000):result;
}
__device__ __forceinline__ float d_npadd(float a,float b) { return d_np_result(a,b,__fadd_rn(a,b)); }
__device__ __forceinline__ float d_npsub(float a,float b) { return d_np_result(a,b,__fsub_rn(a,b)); }
__device__ __forceinline__ float d_npmul(float a,float b) { return d_np_result(a,b,__fmul_rn(a,b)); }
__device__ __forceinline__ float d_npdiv(float a,float b) { return d_np_result(a,b,__fdiv_rn(a,b)); }
#define P_A(a,b) d_npadd((a),(b))
#define P_B(a,b) d_npsub((a),(b))
#define P_M(a,b) d_npmul((a),(b))
#define P_Q(a,b) d_npdiv((a),(b))
__device__ __forceinline__ float d_saturation(float pressure,float temperature) {
    float x=d_npmax(-80.0f,P_B(temperature,273.16f));
    float liquid=-.321582393e-13f;
    liquid=P_A(.379534310e-11f,P_M(x,liquid));
    liquid=P_A(.702620698e-8f,P_M(x,liquid));
    liquid=P_A(.203154182e-5f,P_M(x,liquid));
    liquid=P_A(.299291081e-3f,P_M(x,liquid));
    liquid=P_A(.264224321e-1f,P_M(x,liquid));
    liquid=P_A(.143177157e01f,P_M(x,liquid));
    liquid=P_A(.444606896e02f,P_M(x,liquid));
    liquid=P_A(.611583699e03f,P_M(x,liquid));
    float ice=.161444444e-12f;
    ice=P_A(.105785160e-9f,P_M(x,ice));
    ice=P_A(.307839583e-7f,P_M(x,ice));
    ice=P_A(.521693933e-5f,P_M(x,ice));
    ice=P_A(.565392987e-3f,P_M(x,ice));
    ice=P_A(.402737184e-1f,P_M(x,ice));
    ice=P_A(.184672631e01f,P_M(x,ice));
    ice=P_A(.499320233e02f,P_M(x,ice));
    ice=P_A(.609868993e03f,P_M(x,ice));
    return P_B(temperature,273.15f)<=0.0f ? P_Q(P_M(.622f,ice),P_B(pressure,ice)) : P_Q(P_M(.622f,liquid),P_B(pressure,liquid));
}
#define REBLEND(name,sea) if(component) D_F(name)=A(M(D_F(name),fraction),M(B(1.0f,fraction),sea))
#define FROM(name,out) D_F(name)=o_##out[i]
extern "C" __global__ void ruc_driver_epilogue(
    const unsigned long long* sp,const unsigned long long* op,
    const int* integer,const bool* run,unsigned* flags,
    const float* tbq,const float* lemitbl,const float* half,float dt,int n,
    const float* landusef,int nlcat,int mosaic_lu,int crop,int natural,
    int irrigation,int log_profile,float xice_threshold) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    D_DECLARE_SCRATCH
    D_DECLARE_OUTPUT
    float qsat=Q(d_qsn(D_F(t3d),tbq,flags,1024),M(D_F(p8w),1.0e-2f));
    if(run[i]) {
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilm1d,k)=o_soilm1d[k*n+i]; D_P(tso1d,k)=o_ts1d[k*n+i];
            D_P(smfrkeep,k)=o_smfrkeep[k*n+i]; D_P(keepfr,k)=o_keepfr[k*n+i];
            D_P(soiliqw,k)=o_soiliqw[k*n+i];
        }
        FROM(soilt,soilt); FROM(soilt1,soilt1); FROM(tsnav,tsnav); FROM(dew,dew);
        FROM(qvg,qvg); FROM(qsg,qsg); FROM(qcg,qcg); FROM(snom,snom);
        FROM(snowfallac,snowfallac); FROM(acsnow,acsnow); FROM(alb,alb);
        FROM(znt,znt); FROM(emissl,emiss); FROM(snwe,snwe); FROM(snhei,snhei);
        FROM(snowfrac,snowfrac); FROM(rhosnfall,rhosnfall); FROM(canwatr,cst);
        FROM(lmavail,mavail); FROM(smelt,smelt); FROM(runoff1,runoff1);
        FROM(runoff2,runoff2); FROM(infiltr,infiltr); FROM(qfx,eeta);
        FROM(lh,qfx); FROM(hfx,hfx); FROM(s,s);
        // Irrigation after SFCTMP and before soil diagnostics, by WRF
        // lineage (gpuwm.core.ruc_mosaic.IRRIGATION_FORMS; the host twin is
        // ruc_mosaic.irrigate).  irrigation==1: WRF v4.6.1 LSMRUC:985-1009
        // under mosaic_lu, a per-step relaxation.  irrigation==0: WRF v4.5.2
        // LSMRUC:970-999, a crop-fraction-scaled hard floor with no mosaic
        // gate.  It reads the LANDUSEF fractions when the run carries them
        // (nlcat>0); a run without them (mosaic_lu=0) gives the dominant
        // category the whole cell, which is WRF's arithmetic on a one-hot
        // LANDUSEF.  Its gates read the leaf area index SOILVEGIN left in
        // the scratch, table or 2-D, and the dominant category.
        if(irrigation==1) {
            if(mosaic_lu) {
                float croparea=landusef[(crop-1)*n+i];
                float naturalarea=landusef[(natural-1)*n+i];
                float factor=d_max(0.0f,d_min(1.0f,Q(B(D_F(vegfra),D_F(shdmin)),d_max(1.0f,B(D_F(shdmax),D_F(shdmin))))));
                if((croparea>0.0f || naturalarea>0.0f) && factor>0.75f) {
                    float cropsm=B(M(1.1f,D_F(wilt)),D_F(qmin));
                    float cropfr=d_min(1.0f,A(croparea,M(0.4f,naturalarea)));
                    for(int k=0;k<integer[3*n+i];++k) {
                        float newsm=A(M(cropsm,cropfr),M(B(1.0f,cropfr),D_P(soilm1d,k)));
                        if(D_P(soilm1d,k)<newsm) D_P(soilm1d,k)=newsm;
                    }
                }
            }
        } else {
            float croparea=nlcat>0 ? landusef[(crop-1)*n+i]:(integer[i]==crop ? 1.0f:0.0f);
            float naturalarea=nlcat>0 ? landusef[(natural-1)*n+i]:(integer[i]==natural ? 1.0f:0.0f);
            if(croparea>0.0f && D_F(lai)>1.1f) {
                float cropsm=B(M(1.1f,D_F(wilt)),D_F(qmin));
                float floor=M(cropsm,croparea);
                for(int k=0;k<integer[3*n+i];++k) {
                    if(D_P(soilm1d,k)<floor) D_P(soilm1d,k)=floor;
                }
            } else if(integer[i]==natural && D_F(lai)>0.7f) {
                float cropsm=B(M(1.2f,D_F(wilt)),D_F(qmin));
                float floor=M(M(cropsm,naturalarea),0.4f);
                for(int k=0;k<integer[3*n+i];++k) {
                    if(D_P(soilm1d,k)<floor) D_P(soilm1d,k)=floor;
                }
            }
        }
        float available=0.0f,maximum=0.0f;
        for(int k=0;k<RUC_NZS-1;++k) {
            float thickness=B(half[k+1],half[k]);
            available=A(available,M(A(D_F(qmin),D_P(soilm1d,k)),thickness));
            maximum=A(maximum,M(A(D_F(qmin),D_F(dqm)),thickness));
        }
        float bottom=B(ruc_soil_layer_depth[RUC_NZS-1],half[RUC_NZS-1]);
        available=A(available,M(A(D_F(qmin),D_P(soilm1d,RUC_NZS-1)),bottom));
        maximum=A(maximum,M(A(D_F(qmin),D_F(dqm)),bottom));
        float sr=M(M(D_F(runoff1),dt),1000.0f),ur=M(M(D_F(runoff2),dt),1000.0f);
        D_F(sfcrunoff)=A(D_F(sfcrunoff),sr); D_F(udrunoff)=A(D_F(udrunoff),ur);
        D_F(acrunoff)=A(D_F(acrunoff),sr); D_F(smavail)=M(available,1000.0f);
        D_F(smmax)=M(maximum,1000.0f);
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilmois,k)=A(D_P(soilm1d,k),D_F(qmin));
            D_P(sh2o,k)=d_min(A(D_P(soiliqw,k),D_F(qmin)),D_P(soilmois,k));
            D_P(tso,k)=k==RUC_NZS-1 ? D_F(tbot):D_P(tso1d,k);
            D_P(smfr3d,k)=D_P(smfrkeep,k); D_P(keepfr3dflag,k)=D_P(keepfr,k);
        }
        D_F(z0)=D_F(znt); D_F(sfcexc)=D_F(tkms);
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg)));
        D_F(chklowq)=D_F(qv3d)>=M(qsat,0.95f) && D_F(qv3d)<D_F(qvg) ? 0.0f:1.0f;
        D_F(emiss)=D_F(snow)==0.0f ? lemitbl[integer[i]-1]:D_F(emissl);
        D_F(snow)=M(D_F(snwe),1000.0f); D_F(snowh)=D_F(snhei);
        D_F(canwat)=M(D_F(canwatr),1000.0f); D_F(mavail)=D_F(lmavail);
        D_F(sfcevp)=A(D_F(sfcevp),M(D_F(qfx),dt)); D_F(grdflx)=M(-1.0f,D_F(s));
        D_F(snowc)=D_F(snowfrac)>0.0f && D_F(xice)>=xice_threshold ? M(D_F(snowfrac),D_F(xice)):D_F(snowfrac);
        D_F(rhosnf)=D_F(rhosnfall);
        D_F(sfcevp)=A(D_F(sfcevp),M(D_F(qfx),dt));
    }
    D_CHECK_OUTPUT
    float fraction=D_F(xice);
    bool component=fraction>=xice_threshold && fraction<=1.0f;
    REBLEND(alb,0.08f); REBLEND(emiss,0.98f);
    REBLEND(flhc,D_F(flhc_sea)); REBLEND(flqc,D_F(flqc_sea));
    REBLEND(cpm,D_F(cpm_sea)); REBLEND(cqs2,D_F(cqs2_sea));
    REBLEND(chs2,D_F(chs2_sea)); REBLEND(chs,D_F(chs_sea));
    REBLEND(qsfc,D_F(qsfc_sea)); REBLEND(qgh,D_F(qgh_sea));
    REBLEND(hfx,D_F(hfx_sea)); REBLEND(qfx,D_F(qfx_sea)); REBLEND(lh,D_F(lh_sea));
    if(component) D_F(tsk_save)=D_F(soilt);
    REBLEND(soilt,D_F(tsk_sea));
    float cqs=Q(D_F(flqc),M(D_F(mavail),D_F(rho3d)));
    D_F(chs)=Q(D_F(flhc),M(D_F(cpm),D_F(rho3d)));
    float th2=D_F(chs2)<1.0e-5f ? P_M(D_F(t3d),D_F(scale)) : P_B(P_M(D_F(soilt),D_F(scale)),P_Q(D_F(hfx),P_M(P_M(D_F(rho3d),1004.5f),D_F(chs2))));
    float t2=P_M(th2,D_F(inverse));
    float lower=d_npmin(D_F(soilt),D_F(t3d)),upper=d_npmax(D_F(soilt),D_F(t3d));
    t2=d_npmin(upper,d_npmax(lower,t2));
    D_F(t2)=t2; D_F(th2)=P_M(t2,D_F(scale));
    float qlev=d_npmin(d_saturation(D_F(p8w),D_F(t3d)),D_F(qv3d));
    float prox=P_A(qlev,P_Q(D_F(qfx),P_M(D_F(rho3d),cqs)));
    float qsfcmr=P_Q(D_F(qsfc),P_B(1.0f,D_F(qsfc)));
    float q2=D_F(cqs2)<1.0e-5f ? qlev:P_B(prox,P_Q(D_F(qfx),P_M(D_F(rho3d),D_F(cqs2))));
    q2=d_npmin(d_npmax(qsfcmr,qlev),d_npmax(d_npmin(qsfcmr,qlev),q2));
    D_F(q2)=d_npmin(d_saturation(D_F(psfc),t2),q2);
    // ruc_2m_diagnostic = log_profile: the operational RAP/HRRR branch's
    // module_sf_sfcdiags_ruclsm.F:150-179 (gpuwm.core.ruc_tier
    // RUC_2M_DIAGNOSTIC_FORMS; host twin ruc_runtime._sfcdiags_ruclsm).
    // dz1 is half the lowest layer, LSMRUC's conflx.
    if(log_profile) {
        float dz1=D_F(conflx);
        float dT=P_B(D_F(t3d),D_F(soilt));
        float dQ=P_B(qlev,qsfcmr);
        if(dT>0.0f) {
            float fh=d_npmin(d_npmax(P_B(1.0f,P_Q(dT,10.0f)),0.01f),1.0f);
            float fac=P_Q(gfk_log(P_Q(P_A(2.0f,0.05f),P_A(0.05f,fh))),
                          gfk_log(P_Q(P_A(dz1,0.05f),P_A(0.05f,fh))));
            float t2a=P_A(D_F(soilt),P_M(fac,P_B(D_F(t3d),D_F(soilt))));
            D_F(t2)=t2a; D_F(th2)=P_M(t2a,D_F(scale));
        }
        if(dQ>0.0f) {
            float fh=d_npmin(d_npmax(P_B(1.0f,P_Q(dQ,0.003f)),0.01f),1.0f);
            float fac=P_Q(gfk_log(P_Q(P_A(2.0f,0.05f),P_A(0.05f,fh))),
                          gfk_log(P_Q(P_A(dz1,0.05f),P_A(0.05f,fh))));
            D_F(q2)=P_A(qsfcmr,P_M(fac,P_B(qlev,qsfcmr)));
        }
    }
}

// The fields are written only when no check anywhere in the call failed:
// the driver's own words and the fused sfctmp's bit words (sflags).
extern "C" __global__ void ruc_driver_commit(
    const unsigned long long* sp,const unsigned long long* cp,const unsigned* flags,
    const unsigned long long* sflags,int swords,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int word=0;word<36;++word) if(flags[word]) return;
    for(int word=0;word<swords;++word) if(sflags[word]) return;
    D_DECLARE_SCRATCH
    D_COMMIT
}
#undef A
#undef M
#undef B
#undef Q
#undef REBLEND
#undef FROM

#undef P_A
#undef P_B
#undef P_M
#undef P_Q
