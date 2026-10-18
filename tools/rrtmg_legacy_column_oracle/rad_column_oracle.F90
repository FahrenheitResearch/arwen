! Column oracle for WRF v4.6.1 legacy RRTMG (ra_lw_physics = ra_sw_physics = 4)
! driven the way phys/module_radiation_driver.F drives it on one radiation
! step, over ncol independent columns (i = 1..ncol, j = 1).
!
! Physics comes only from unmodified WRF text:
!   RRTMG_LWINIT / RRTMG_LWRAD   phys/module_ra_rrtmg_lw.F (compiled whole)
!   RRTMG_SWINIT / RRTMG_SWRAD   phys/module_ra_rrtmg_sw.F (compiled whole)
!   radconst, calc_coszen, cal_cldfra1, ozn_time_int, ozn_p_int, oznini,
!   lin_interpol2                extracted verbatim (build_units.py)
!   gt_aod + calc_aerosol_rrtmg_sw   aer_opt = 3, stock 4.6.1 or the HRRR fork
!
! The glue statements below are the driver's own lines, in its order:
!   phy_prep t8w        dyn_em/module_big_step_utilities_em.F:4890-4935
!   radconst/coszen     module_radiation_driver.F:1188, :1206-1208
!   CLDFRA zero + cal_cldfra1 under ICLOUD == 1 and (F_QC or F_QI)  :1310-1332
!   ozone (o3input 2)   module_physics_init oznini; driver ozn_time_int/ozn_p_int
!   aer_opt = 3         :1651-1693 (taod5502d zero, gt_aod, k-sum), :2312-2335
!   GSW/GLW/SWDOWN/RTHRATEN zero   :1720-1737
!   RRTMG_LWRAD(RTHRATENLW=RTHRATEN) :2048-2094, RTHRATENLW = RTHRATEN :2256
!   RRTMG_SWRAD :2574-2682, RTHRATEN += RTHRATENSW :2690-2696
!   RTHRATENSW = RTHRATEN - RTHRATENLW, SWDOWN = GSW/(1-ALBEDO) :2866-2879
!
! Input (little-endian stream): see read section.  Output: named records
! (char*32 name, int32 count, float32 data) until EOF.
program rad_column_oracle
  use module_ra_rrtmg_lw, only: rrtmg_lwinit, rrtmg_lwrad
  use module_ra_rrtmg_sw, only: rrtmg_swinit, rrtmg_swrad
  use wrf461_raddrv_extract, only: radconst, calc_coszen, cal_cldfra1, &
                                   ozn_time_int, ozn_p_int, oznini
  use wrf_aer_gt, only: gt_aod
  use module_ra_aerosol, only: calc_aerosol_rrtmg_sw
  implicit none

  integer, parameter :: levsiz = 59, num_months = 13, nb = 14
  integer :: hdr(24)
  real :: rhdr(8)
  integer :: ncol, nz, mp_physics, has_reqc, has_reqi, has_reqs
  logical :: f_qv, f_qc, f_qr, f_qi, f_qs, f_qg, warm_rain
  integer :: o3input, aer_opt, sf_surface_physics, julday, yr, icloud
  integer :: cldovrlp, idcor, wif_input_opt
  real :: julian, xtime, gmt, radt, p_top
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte, i, k, j, kde_, k_end
  real :: declin, solcon, degrad, dpd, z0, z1, z2, w1, w2
  real, parameter :: piconst = 3.1415926535897932384626433
  character(len=1024) :: inpath, outpath
  integer :: iu, ou

  real, allocatable, dimension(:,:,:) :: p3d, t3d, pi3d, dz8w, rho, qv, qc, qr, &
       qi, qs, qg, re_cloud, re_ice, re_snow, nwfa, nifa, nbca, p8w, z_at_w, t8w, &
       zhalf, cldfra, o3rad, taod5503d, rthraten, rthratenlw, rthratensw, &
       rthratenlwc, rthratenswc, f_ice_phy, f_rain_phy, qndrop, &
       lwupflx, lwupflxc, lwdnflx, lwdnflxc, swupflx, swupflxc, swdnflx, swdnflxc
  integer, allocatable :: cldfra1_flag(:,:,:)
  real, allocatable, dimension(:,:) :: tsk, emiss, albedo, xland, xice, snow, &
       xlat, xlong, ht, coszen, hrang, glw, gsw, swdown, olr, lwcf, swcf, &
       lwupt, lwuptc, lwuptcln, lwdnt, lwdntc, lwdntcln, lwupb, lwupbc, lwupbcln, &
       lwdnb, lwdnbc, lwdnbcln, swupt, swuptc, swuptcln, swdnt, swdntc, swdntcln, &
       swupb, swupbc, swupbcln, swdnb, swdnbc, swdnbcln, coszr, swddir, swddni, &
       swddif, swdownc, swddnic, swddirc, obscur, swvisdir, swvisdif, swnirdir, &
       swnirdif, lradius, iradius, taod5502d, angexp2d, aerssa2d, aerasy2d
  real, allocatable :: fnm(:), fnp(:)
  ! POINTER, nullified unless aer_opt = 3, as the driver declares them
  real, pointer :: tauaer_sw(:,:,:,:) => null(), ssaaer_sw(:,:,:,:) => null(), &
                   asyaer_sw(:,:,:,:) => null()
  real, allocatable :: ozmixm(:,:,:,:), ozmixt(:,:,:), pin(:)

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(newunit=iu, file=trim(inpath), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(iu) hdr
  read(iu) rhdr
  ncol = hdr(1); nz = hdr(2); mp_physics = hdr(3)
  has_reqc = hdr(4); has_reqi = hdr(5); has_reqs = hdr(6)
  f_qv = hdr(7) /= 0; f_qc = hdr(8) /= 0; f_qr = hdr(9) /= 0
  f_qi = hdr(10) /= 0; f_qs = hdr(11) /= 0; f_qg = hdr(12) /= 0
  warm_rain = hdr(13) /= 0; o3input = hdr(14); aer_opt = hdr(15)
  sf_surface_physics = hdr(16); julday = hdr(17); yr = hdr(18); icloud = hdr(19)
  cldovrlp = hdr(20); idcor = hdr(21); wif_input_opt = hdr(22)
  julian = rhdr(1); xtime = rhdr(2); gmt = rhdr(3); radt = rhdr(4); p_top = rhdr(5)

  ! WRF patch bounds for one tile: kde is the staggered top (nz+1), kte the
  ! mass top (physics calls pass min(kte, kde-1)).
  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol; jts = 1; jte = 1; kts = 1; kte = nz

  call alloc3(p3d); call alloc3(t3d); call alloc3(pi3d); call alloc3(dz8w)
  call alloc3(rho); call alloc3(qv); call alloc3(qc); call alloc3(qr)
  call alloc3(qi); call alloc3(qs); call alloc3(qg); call alloc3(re_cloud)
  call alloc3(re_ice); call alloc3(re_snow); call alloc3(nwfa); call alloc3(nifa)
  call alloc3(nbca); call alloc3(p8w); call alloc3(z_at_w); call alloc3(t8w)
  call alloc3(zhalf); call alloc3(cldfra); call alloc3(o3rad); call alloc3(taod5503d)
  call alloc3(rthraten); call alloc3(rthratenlw); call alloc3(rthratensw)
  call alloc3(rthratenlwc); call alloc3(rthratenswc); call alloc3(f_ice_phy)
  call alloc3(f_rain_phy); call alloc3(qndrop)
  call alloc3(lwupflx); call alloc3(lwupflxc); call alloc3(lwdnflx); call alloc3(lwdnflxc)
  call alloc3(swupflx); call alloc3(swupflxc); call alloc3(swdnflx); call alloc3(swdnflxc)
  allocate(cldfra1_flag(ims:ime, kms:kme, jms:jme)); cldfra1_flag = 0
  call alloc2(tsk); call alloc2(emiss); call alloc2(albedo); call alloc2(xland)
  call alloc2(xice); call alloc2(snow); call alloc2(xlat); call alloc2(xlong)
  call alloc2(ht); call alloc2(coszen); call alloc2(hrang); call alloc2(glw)
  call alloc2(gsw); call alloc2(swdown); call alloc2(olr); call alloc2(lwcf)
  call alloc2(swcf); call alloc2(lwupt); call alloc2(lwuptc); call alloc2(lwuptcln)
  call alloc2(lwdnt); call alloc2(lwdntc); call alloc2(lwdntcln); call alloc2(lwupb)
  call alloc2(lwupbc); call alloc2(lwupbcln); call alloc2(lwdnb); call alloc2(lwdnbc)
  call alloc2(lwdnbcln); call alloc2(swupt); call alloc2(swuptc); call alloc2(swuptcln)
  call alloc2(swdnt); call alloc2(swdntc); call alloc2(swdntcln); call alloc2(swupb)
  call alloc2(swupbc); call alloc2(swupbcln); call alloc2(swdnb); call alloc2(swdnbc)
  call alloc2(swdnbcln); call alloc2(coszr); call alloc2(swddir); call alloc2(swddni)
  call alloc2(swddif); call alloc2(swdownc); call alloc2(swddnic); call alloc2(swddirc)
  call alloc2(obscur); call alloc2(swvisdir); call alloc2(swvisdif); call alloc2(swnirdir)
  call alloc2(swnirdif); call alloc2(lradius); call alloc2(iradius)
  call alloc2(taod5502d); call alloc2(angexp2d); call alloc2(aerssa2d); call alloc2(aerasy2d)
  allocate(fnm(kms:kme), fnp(kms:kme)); fnm = 0.; fnp = 0.

  call rd3(p3d, nz); call rd3(t3d, nz); call rd3(pi3d, nz); call rd3(dz8w, nz)
  call rd3(rho, nz); call rd3(qv, nz); call rd3(qc, nz); call rd3(qr, nz)
  call rd3(qi, nz); call rd3(qs, nz); call rd3(qg, nz); call rd3(re_cloud, nz)
  call rd3(re_ice, nz); call rd3(re_snow, nz); call rd3(nwfa, nz); call rd3(nifa, nz)
  call rd3(p8w, nz + 1); call rd3(z_at_w, nz + 1)
  read(iu) fnm(1:nz); read(iu) fnp(1:nz)
  call rd2(tsk); call rd2(emiss); call rd2(albedo); call rd2(xland); call rd2(xice)
  call rd2(snow); call rd2(xlat); call rd2(xlong); call rd2(ht)
  close(iu)
  nbca = 0.; obscur = 0.

  ! ---- phy_prep: t8w (dyn_em/module_big_step_utilities_em.F:4890-4935) ----
  ! k_start = kts, k_end = kte - 1 with phy_prep's own kte = kde: the mass
  ! levels are 1..nz, kde = nz + 1.
  kde_ = kde
  k_end = kde_ - 1
  do j = jts, jte
  do k = kts, k_end
  do i = its, ite
    zhalf(i,k,j) = 0.5*(z_at_w(i,k,j) +z_at_w(i,k+1,j) )
  enddo
  enddo
  enddo
  do j = jts, jte
  do k = 2, k_end
  do i = its, ite
    t8w(i,k,j) = fnm(k)*t3d(i,k,j)+fnp(k)*t3d(i,k-1,j)
  enddo
  enddo
  enddo
  do j = jts, jte
  do i = its, ite
    z0 = z_at_w(i,1,j)
    z1 = zhalf(i,1,j)
    z2 = zhalf(i,2,j)
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    t8w(i,1,j) = w1*t3d(i,1,j)+w2*t3d(i,2,j)
    z0 = z_at_w(i,kde_,j)
    z1 = zhalf(i,k_end,j)
    z2 = zhalf(i,k_end-1,j)
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    t8w(i,kde_,j) = w1*t3d(i,kde_-1,j)+w2*t3d(i,kde_-2,j)
  enddo
  enddo

  ! ---- radconst + calc_coszen (driver :1188, :1206-1208) ----
  ! share/module_model_constants.F: DEGRAD = piconst/180., DPD = 360./365.
  degrad = piconst/180.
  dpd = 360./365.
  call radconst(xtime, declin, solcon, julian, degrad, dpd)
  call calc_coszen(ims,ime,jms,jme,its,ite,jts,jte, &
                   julian,xtime+radt*0.5,gmt, &
                   declin,degrad,xlong,xlat,coszen,hrang)

  ! ---- cloud fraction (driver :1310-1332) ----
  cldfra = 0.
  if (icloud == 1) then
     if (f_qc .or. f_qi) then
        call cal_cldfra1(cldfra, qv, qc, qi, qs,                 &
                         f_qv, f_qc, f_qi, f_qs, t3d, p3d,        &
                         f_ice_phy, f_rain_phy,                   &
                         mp_physics, cldfra1_flag,                &
                         ids,ide, jds,jde, kds,kde,               &
                         ims,ime, jms,jme, kms,kme,               &
                         its,ite, jts,jte, kts,kte)
     end if
  end if

  ! ---- ozone, o3input = 2 (module_physics_init oznini; driver ozn_*) ----
  o3rad = 0.
  if (o3input == 2) then
     allocate(ozmixm(ims:ime, levsiz, jms:jme, num_months), pin(levsiz))
     allocate(ozmixt(ims:ime, levsiz, jms:jme))
     call oznini(ozmixm, pin, levsiz, num_months, xlat, &
                 ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
                 its,ite, jts,jte, kts,kte)
     call ozn_time_int(julday, julian, ozmixm, ozmixt, levsiz, num_months, &
                       ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
                       its,ite, jts,jte, kts,kte)
     call ozn_p_int(p3d, pin, levsiz, ozmixt, o3rad, &
                    ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
                    its,ite, jts,jte, kts,kte)
  end if

  ! ---- aer_opt = 3 (driver :1651-1693 then :2312-2335) ----
  taod5503d = 0.; taod5502d = 0.
  if (aer_opt == 3) then
     allocate(tauaer_sw(ims:ime, kms:kme, jms:jme, 1:nb))
     allocate(ssaaer_sw(ims:ime, kms:kme, jms:jme, 1:nb))
     allocate(asyaer_sw(ims:ime, kms:kme, jms:jme, 1:nb))
     tauaer_sw = 0.; ssaaer_sw = 0.; asyaer_sw = 0.
     do j = jts, jte
        do i = its, ite
           taod5502d(i,j) = 0.0
        end do
     end do
#ifdef FORK_AER
     call gt_aod(p3d, dz8w, t3d, qv, nwfa, nifa, taod5503d,           &
                 ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte)
#else
     call gt_aod(p3d, dz8w, t3d, qv, nwfa, nifa, nbca, taod5503d,     &
                 wif_input_opt, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte)
#endif
     do j = jts, jte
        do i = its, ite
           do k = kts, kte
              taod5502d(i,j) = taod5502d(i,j) + taod5503d(i,k,j)
           end do
        end do
     end do
     ! taer_type = 1, taer_aod550_opt = 2, taer_angexp_opt = taer_ssa_opt =
     ! taer_asy_opt = 3 (driver PARAMETERs :1014-1018); aer_*_val at their
     ! Registry defaults (unread on these options).
     call calc_aerosol_rrtmg_sw(ht, dz8w, p3d, t3d, qv, 1, 2, 3, 3, 3,  &
                                0.12, 1.3, 0.85, 0.9,                     &
                                taod5502d, angexp2d, aerssa2d, aerasy2d,  &
                                ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, &
                                tauaer_sw, ssaaer_sw, asyaer_sw, taod5503d)
  end if

  ! ---- driver zeroing (:1720-1737) ----
  gsw = 0.; glw = 0.; swdown = 0.; swddir = 0.; swddni = 0.; swddif = 0.
  rthraten = 0.; rthratenlw = 0.; rthratensw = 0.
  rthratenlwc = 0.; rthratenswc = 0.
  olr = 0.; lwcf = 0.; swcf = 0.; coszr = 0.
  lwupt = 0.; lwuptc = 0.; lwuptcln = 0.; lwdnt = 0.; lwdntc = 0.; lwdntcln = 0.
  lwupb = 0.; lwupbc = 0.; lwupbcln = 0.; lwdnb = 0.; lwdnbc = 0.; lwdnbcln = 0.
  swupt = 0.; swuptc = 0.; swuptcln = 0.; swdnt = 0.; swdntc = 0.; swdntcln = 0.
  swupb = 0.; swupbc = 0.; swupbcln = 0.; swdnb = 0.; swdnbc = 0.; swdnbcln = 0.
  swdownc = 0.; swddnic = 0.; swddirc = 0.
  swvisdir = 0.; swvisdif = 0.; swnirdir = 0.; swnirdif = 0.
  lwupflx = 0.; lwupflxc = 0.; lwdnflx = 0.; lwdnflxc = 0.
  swupflx = 0.; swupflxc = 0.; swdnflx = 0.; swdnflxc = 0.
  f_ice_phy = 0.; f_rain_phy = 0.; qndrop = 0.; lradius = 0.; iradius = 0.

  call rrtmg_lwinit(p_top, .true., ids,ide, jds,jde, kds,kde, &
                    ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte)
  call rrtmg_swinit(.true., ids,ide, jds,jde, kds,kde, &
                    ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte)

  call rrtmg_lwrad(rthratenlw=rthraten, rthratenlwc=rthratenlwc,        &
       lwupt=lwupt, lwuptc=lwuptc, lwuptcln=lwuptcln,                   &
       lwdnt=lwdnt, lwdntc=lwdntc, lwdntcln=lwdntcln,                   &
       lwupb=lwupb, lwupbc=lwupbc, lwupbcln=lwupbcln,                   &
       lwdnb=lwdnb, lwdnbc=lwdnbc, lwdnbcln=lwdnbcln,                   &
       glw=glw, olr=olr, lwcf=lwcf, emiss=emiss,                        &
       p8w=p8w, p3d=p3d, pi3d=pi3d, dz8w=dz8w, tsk=tsk, t3d=t3d,        &
       t8w=t8w, rho3d=rho, r=287., g=9.81,                              &
       icloud=icloud, warm_rain=warm_rain, cldfra3d=cldfra,             &
       cldovrlp=cldovrlp, idcor=idcor, xlat=xlat,                       &
       lradius=lradius, iradius=iradius,                                &
       is_cammgmp_used=.false.,                                         &
       f_ice_phy=f_ice_phy, f_rain_phy=f_rain_phy,                      &
       xland=xland, xice=xice, snow=snow,                               &
       qv3d=qv, qc3d=qc, qr3d=qr, qi3d=qi, qs3d=qs, qg3d=qg,            &
       o3input=o3input, o33d=o3rad,                                     &
       f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg,&
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,               &
       has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs,         &
       calc_clean_atm_diag=0,                                           &
       qndrop3d=qndrop, f_qndrop=.false.,                               &
       yr=yr, julian=julian, ghg_input=0,                               &
       ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,            &
       ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,            &
       its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte,            &
       lwupflx=lwupflx, lwupflxc=lwupflxc,                              &
       lwdnflx=lwdnflx, lwdnflxc=lwdnflxc,                              &
       mp_physics=mp_physics)
  do j = jts, jte
  do k = kts, kte
  do i = its, ite
     rthratenlw(i,k,j) = rthraten(i,k,j)
  end do
  end do
  end do

  call rrtmg_swrad(rthratensw=rthratensw, rthratenswc=rthratenswc,      &
       swupt=swupt, swuptc=swuptc, swuptcln=swuptcln, swdnt=swdnt,      &
       swdntc=swdntc, swdntcln=swdntcln, swupb=swupb, swupbc=swupbc,    &
       swupbcln=swupbcln, swdnb=swdnb, swdnbc=swdnbc, swdnbcln=swdnbcln,&
       swcf=swcf, gsw=gsw,                                              &
       xtime=xtime, gmt=gmt, xlat=xlat, xlong=xlong,                    &
       radt=radt, degrad=degrad, declin=declin,                         &
       coszr=coszr, julday=julday, solcon=solcon,                       &
       albedo=albedo, t3d=t3d, t8w=t8w, tsk=tsk,                        &
       p3d=p3d, p8w=p8w, pi3d=pi3d, rho3d=rho,                          &
       dz8w=dz8w, cldfra3d=cldfra, lradius=lradius, iradius=iradius,    &
       is_cammgmp_used=.false., r=287., g=9.81,                         &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,               &
       has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs,         &
       icloud=icloud, warm_rain=warm_rain,                              &
       cldovrlp=cldovrlp, idcor=idcor,                                  &
       f_ice_phy=f_ice_phy, f_rain_phy=f_rain_phy,                      &
       xland=xland, xice=xice, snow=snow,                               &
       qv3d=qv, qc3d=qc, qr3d=qr, qi3d=qi, qs3d=qs, qg3d=qg,            &
       o3input=o3input, o33d=o3rad,                                     &
       aer_opt=aer_opt, no_src=1,                                       &
       swvisdir=swvisdir, swvisdif=swvisdif,                            &
       swnirdir=swnirdir, swnirdif=swnirdif,                            &
       sf_surface_physics=sf_surface_physics,                           &
       f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg,&
       calc_clean_atm_diag=0,                                           &
       qndrop3d=qndrop, f_qndrop=.false.,                               &
       mp_physics=mp_physics,                                           &
       ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,            &
       ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,            &
       its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte,            &
       swupflx=swupflx, swupflxc=swupflxc,                              &
       swdnflx=swdnflx, swdnflxc=swdnflxc,                              &
       tauaer3d_sw=tauaer_sw, ssaaer3d_sw=ssaaer_sw, asyaer3d_sw=asyaer_sw, &
       swddir=swddir, swddni=swddni, swddif=swddif,                     &
       swdownc=swdownc, swddnic=swddnic, swddirc=swddirc,               &
       xcoszen=coszen, yr=yr, julian=julian, ghg_input=0,               &
       obscur=obscur, proceed_cmaq_sw=.false.)

  ! RTHRATEN = RTHRATEN + RTHRATENSW (:2690-2696), then the driver's
  ! RTHRATENSW = RTHRATEN - RTHRATENLW and SWDOWN (:2866-2879).
  do j = jts, jte
  do k = kts, kte
  do i = its, ite
     rthraten(i,k,j) = rthraten(i,k,j) + rthratensw(i,k,j)
  end do
  end do
  end do

  open(newunit=ou, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  ! the wrapper's own RTHRATENSW, before the driver rewrites it
  call put3('rthratensw_wrapper', rthratensw, nz)
  do j = jts, jte
  do k = kts, kte
  do i = its, ite
     rthratensw(i,k,j) = rthraten(i,k,j) - rthratenlw(i,k,j)
  end do
  end do
  end do
  do j = jts, jte
  do i = its, ite
     swdown(i,j) = gsw(i,j)/(1.-albedo(i,j))
  end do
  end do

  call put1('declin', declin); call put1('solcon', solcon)
  call put2('coszen', coszen); call put2('hrang', hrang)
  call put3('t8w', t8w, nz + 1); call put3('cldfra', cldfra, nz)
  call put3('o3rad', o3rad, nz)
  call put3('taod5503d', taod5503d, nz); call put2('taod5502d', taod5502d)
  if (aer_opt == 3) then
     call put4('tauaer_sw', tauaer_sw); call put4('ssaaer_sw', ssaaer_sw)
     call put4('asyaer_sw', asyaer_sw)
  end if
  call put3('rthratenlw', rthratenlw, nz); call put3('rthratenlwc', rthratenlwc, nz)
  call put2('glw', glw); call put2('olr', olr); call put2('lwcf', lwcf)
  call put2('lwupt', lwupt); call put2('lwuptc', lwuptc); call put2('lwdnt', lwdnt)
  call put2('lwdntc', lwdntc); call put2('lwupb', lwupb); call put2('lwupbc', lwupbc)
  call put2('lwdnb', lwdnb); call put2('lwdnbc', lwdnbc)
  call put3('rthratensw', rthratensw, nz); call put3('rthratenswc', rthratenswc, nz)
  call put3('rthraten', rthraten, nz)
  call put2('gsw', gsw); call put2('swdown', swdown); call put2('coszr', coszr)
  call put2('swddir', swddir); call put2('swddni', swddni); call put2('swddif', swddif)
  call put2('swcf', swcf); call put2('swupt', swupt); call put2('swuptc', swuptc)
  call put2('swdnt', swdnt); call put2('swdntc', swdntc); call put2('swupb', swupb)
  call put2('swupbc', swupbc); call put2('swdnb', swdnb); call put2('swdnbc', swdnbc)
  call put2('swdownc', swdownc)
  close(ou)
  write(*,'(A,I0,A)') 'rad_column_oracle: OK (', ncol, ' columns)'

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme)); a = 0.
  end subroutine alloc3

  subroutine alloc2(a)
    real, allocatable, intent(inout) :: a(:,:)
    allocate(a(ims:ime, jms:jme)); a = 0.
  end subroutine alloc2

  subroutine rd3(a, nk)
    real, intent(inout) :: a(ims:ime, kms:kme, jms:jme)
    integer, intent(in) :: nk
    real :: tmp(ncol, nk)
    read(iu) tmp
    a(1:ncol, 1:nk, 1) = tmp
  end subroutine rd3

  subroutine rd2(a)
    real, intent(inout) :: a(ims:ime, jms:jme)
    read(iu) a(1:ncol, 1)
  end subroutine rd2

  subroutine header(name, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    character(len=32) :: tag
    tag = name
    write(ou) tag, n
  end subroutine header

  subroutine put1(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v
    call header(name, 1)
    write(ou) v
  end subroutine put1

  subroutine put2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:ime, jms:jme)
    call header(name, ncol)
    write(ou) a(1:ncol, 1)
  end subroutine put2

  subroutine put3(name, a, nk)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:ime, kms:kme, jms:jme)
    integer, intent(in) :: nk
    call header(name, ncol*nk)
    write(ou) a(1:ncol, 1:nk, 1)
  end subroutine put3

  subroutine put4(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:ime, kms:kme, jms:jme, 1:nb)
    call header(name, ncol*(nz+1)*nb)
    write(ou) a(1:ncol, 1:nz+1, 1, 1:nb)
  end subroutine put4

end program rad_column_oracle
