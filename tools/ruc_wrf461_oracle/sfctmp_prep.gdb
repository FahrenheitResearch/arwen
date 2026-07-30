# Pins the WRF v4.6.1 sfctmp snow-preparation block by reading it out of the
# running, UNMODIFIED module_sf_ruclsm.
#
# The block (phys/module_sf_ruclsm.F:1400-1766) is straight-line code inside
# sfctmp, and every result is a local or an inout dummy that the dispatch below
# it overwrites before sfctmp returns, so there is no argument list to observe
# it through.  Breakpoints bracket it exactly:
#
#   :1418  snhei_crit=0.01601*rhowater/rhosn - the first executable statement
#          of the block; nothing has been modified yet, so this is the
#          incoming state.
#   :1767  if (snow_mosaic==1.) then         - the first statement of the
#          dispatch, reached when snhei>0.
#   :2120  snheiprint=0.                     - the first statement of the
#          no-snow else branch at :2118, reached when snhei<=0.
#
# Exactly one of :1767 / :2120 is reached per call, and in both cases the
# preparation block has just finished and nothing else has run.
#
# One CSV row is emitted per (case, soil level).  Reals print with %.9g, which
# round-trips IEEE-754 binary32 exactly.
#
# keep_snow_albedo (:1658) and snowfrac2 (:1626) are only assigned inside the
# snhei>0. branch; when that branch does not run WRF leaves them undefined, so
# they are written as nan rather than as stack residue.

set confirm off
set pagination off
set height 0
set width 0
file ./run_sfctmp_prep

define emit_prep_rows
set $k = 1
while $k <= nzs
printf "%d,%d,", i, $k
printf "%.9g,%.9g,%.9g,%d,%d,", delt, c1sn, c2sn, isice, ivgtyp
printf "%.9g,%.9g,%.9g,%.9g,", seaice, gsw, tabs, tsnav
printf "%.9g,%.9g,%.9g,%.9g,", prcpms, newsnms, vegfra, lai
printf "%.9g,%.9g,%.9g,", sat, soilt, snowfallac
printf "%.9g,%.9g,", alb_snow, alb_snow_free
printf "%.9g,%.9g,%.9g,%.9g,", snowrat, grauprat, icerat, curat
printf "%d,%.9g,%.9g,%.9g,", $iland_in, $snwe_in, $snhei_in, $snowfrac_in
printf "%.9g,%.9g,%.9g,", $rhosn_in, $rhosnfall_in, $cst_in
printf "%.9g,%.9g,%.9g,", $alb_in, $emiss_in, $znt_in
printf "%.9g,", ts1d($k)
printf "%.9g,%.9g,%.9g,", snhei_crit, snhei_crit_newsn, zntsn
printf "%.9g,%.9g,%.9g,", snow_mosaic, snfr, newsn
printf "%.9g,%.9g,%.9g,", newsnowratio, snowfracnewsn, rhonewsn
printf "%.9g,%.9g,%.9g,", smelt, rainf, rsm
printf "%.9g,%.9g,%.9g,", dd1, infiltr, vegfrac
printf "%.9g,%.9g,%.9g,%.9g,", drip, dripsn, dripliq, smf
printf "%.9g,%.9g,%.9g,%.9g,", interw, intersn, infwater, intwratio
printf "%.9g,%.9g,", gswnew, gswin
printf "%.9g,%.9g,", albice, albsn
printf "%.9g,%.9g,", emissn, emiss_snowfree
if snhei > 0.
printf "%.9g,%.9g,", keep_snow_albedo, snowfrac2
else
printf "nan,nan,"
end
printf "%d,%.9g,%.9g,", iland, snwe, snhei
printf "%.9g,%.9g,%.9g,", snowfrac, rhosn, rhosnfall
printf "%.9g,%.9g,%.9g,%.9g,", cst, alb, emiss, znt
printf "%.9g,%.9g,", tice($k), rhosice($k)
printf "%.9g,%.9g\n", capice($k), thdifice($k)
set $k = $k + 1
end
end

break module_sf_ruclsm.F:1418
commands
silent
set $iland_in = iland
set $snwe_in = snwe
set $snhei_in = snhei
set $snowfrac_in = snowfrac
set $rhosn_in = rhosn
set $rhosnfall_in = rhosnfall
set $cst_in = cst
set $alb_in = alb
set $emiss_in = emiss
set $znt_in = znt
continue
end

break module_sf_ruclsm.F:1767
commands
silent
emit_prep_rows
continue
end

break module_sf_ruclsm.F:2120
commands
silent
emit_prep_rows
continue
end

set logging file sfctmp-prep-gdb.log
set logging redirect on
set logging overwrite on
set logging enabled on
printf "case_index,k,delt,c1sn,c2sn,isice,ivgtyp,seaice,gsw,tabs,tsnav,prcpms,newsnms,vegfra,lai,sat,soilt,snowfallac,alb_snow,alb_snow_free,snowrat,grauprat,icerat,curat,iland_before,snwe_before,snhei_before,snowfrac_before,rhosn_before,rhosnfall_before,cst_before,alb_before,emiss_before,znt_before,ts1d,snhei_crit,snhei_crit_newsn,zntsn,snow_mosaic,snfr,newsn,newsnowratio,snowfracnewsn,rhonewsn,smelt,rainf,rsm,dd1,infiltr,vegfrac,drip,dripsn,dripliq,smf,interw,intersn,infwater,intwratio,gswnew,gswin,albice,albsn,emissn,emiss_snowfree,keep_snow_albedo,snowfrac2,iland_after,snwe_after,snhei_after,snowfrac_after,rhosn_after,rhosnfall_after,cst_after,alb_after,emiss_after,znt_after,tice,rhosice,capice,thdifice\n"
run ruc-sfctmp-prep-inputs.csv
set logging enabled off
quit
