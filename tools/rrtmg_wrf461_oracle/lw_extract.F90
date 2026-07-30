! Per-routine fixture extractor for WRF v4.6.1 legacy RRTMG longwave.
!
! Compiles against the UNMODIFIED phys/module_ra_rrtmg_lw.F.  For each input
! column this driver:
!
!   1. transcribes the RRTMG_LWRAD wrapper prep (EM_CORE=1, WRF_CHEM=0,
!      aer_ra_feedback=0, ghg_input=0, progn=0 path) statement for statement
!      to build the exact rrtmg_lw-level arguments the WRF driver builds,
!      using the module's own INIRAD/RELCALC/REICALC/mcica_subcol_lw;
!   2. dumps those arguments ("in/"), runs the DECOMPOSED chain
!      inatm -> cldprmc -> setcoef -> taumol -> (taut) -> rtrnmc calling the
!      unmodified module routines, dumping every intermediate stage;
!   3. calls the unmodified rrtmg_lw directly ("out/") -- Python asserts the
!      composed chain reproduces it bitwise;
!   4. calls the unmodified RRTMG_LWRAD wrapper on the same raw fields
!      ("wrap/") -- Python asserts the transcription in (1) is faithful.
!
! The MP option 5 branch (needs F_ICE_PHY present) and Ferrier branches are
! pinned off: the oracle never passes those arguments, matching the campaign
! schemes (WSM6/Thompson/Morrison/NSSL2).
!
! Input text format (list-directed):
!   line 1: ncase nz
!   per case:
!     A: caseid yr julday mp_physics icloud cldovrlp idcor o3input
!        has_reqc has_reqi has_reqs ifqv ifqc ifqr ifqi ifqs ifqg iwarm
!     B: julian xlat xlong emiss tsk psfc xland xice snow snowh p_top
!     nz lines:   p3d(Pa) t3d(K) dz8w(m) pi3d rho3d qv qc qr qi qs qg
!                 cldfra re_c(m) re_i(m) re_s(m) o33d(vmr)
!     nz+1 lines: p8w(Pa) t8w(K)
!
! Usage: lw_extract input.txt outdir

program lw_extract
  use lw_binio
  use lw_dump_state, only : dump_all_state
  use module_ra_rrtmg_lw, only : rrtmg_lwinit, rrtmg_lwrad, inirad, &
                                 relcalc, reicalc, retab, &
                                 mod_nlayers => nlayers, deltap
  use rrtmg_lw_rad,    only : rrtmg_lw, inatm
  use rrtmg_lw_cldprmc, only : cldprmc
  use rrtmg_lw_setcoef, only : setcoef
  use rrtmg_lw_taumol,  only : taumol
  use rrtmg_lw_rtrnmc,  only : rtrnmc
  use mcica_subcol_gen_lw, only : mcica_subcol_lw
  use parrrtm, only : nbndlw, ngptlw, mxmol, maxxsec
  use rrlw_wvn, only : ngb
  implicit none

  integer :: ncase, nz, icase, k, l, ll, nb, ig, iu
  character(len=1024) :: infile, outdir, path

  ! ---- case header ----
  integer :: caseid, yr, julday, mp_physics, icloud, cldovrlp, idcor, o3input
  integer :: has_reqc, has_reqi, has_reqs
  integer :: ifqv, ifqc, ifqr, ifqi, ifqs, ifqg, iwarm
  real :: julian, xlat0, xlong0, emiss0, tsk0, psfc0, xland0, xice0
  real :: snow0, snowh0, p_top, p_top_init
  logical :: warm_rain, f_qi_l
  logical :: inited

  ! ---- raw column inputs (wrapper-level) ----
  real, allocatable :: p3d_in(:), t3d_in(:), dz_in(:), pi3d_in(:), rho_in(:)
  real, allocatable :: qv_in(:), qc_in(:), qr_in(:), qi_in(:), qs_in(:), qg_in(:)
  real, allocatable :: cldfra_in(:), rec_in(:), rei_in(:), res_in(:), o3_in(:)
  real, allocatable :: p8w_in(:), t8w_in(:)

  ! ---- transcribed wrapper locals (names mirror RRTMG_LWRAD) ----
  integer :: nlayers, kts, kte, kme
  integer :: ncol, nlay, icld, juldat, inflglw, iceflglw, liqflglw
  integer :: iplon, irng, permuteseed, idx_rei, klev, nproflevs
  real :: lat, dzsum, dz, corr, gicewp, gliqwp, gsnowp, gravmks
  real :: snow_mass_factor, wght, vark, vark1
  real, allocatable :: pw1d(:), tw1d(:)
  real, allocatable :: tten1d(:), cldfra1d(:), dz1d(:), p1d(:), t1d(:)
  real, allocatable :: qv1d(:), qc1d(:), qr1d(:), qi1d(:), qs1d(:), qg1d(:), o31d(:)
  real, allocatable :: plev(:,:), tlev(:,:), play(:,:), tlay(:,:)
  real, allocatable :: h2ovmr(:,:), o3vmr(:,:), co2vmr(:,:), o2vmr(:,:)
  real, allocatable :: ch4vmr(:,:), n2ovmr(:,:)
  real, allocatable :: cfc11vmr(:,:), cfc12vmr(:,:), cfc22vmr(:,:), ccl4vmr(:,:)
  real, allocatable :: o3mmr(:), hgt(:,:), varint(:)
  real, allocatable :: emis(:,:)
  real, allocatable :: clwpth(:,:), ciwpth(:,:), cswpth(:,:)
  real, allocatable :: rel(:,:), rei(:,:), res(:,:), cldfrac(:,:)
  real, allocatable :: relqmcl(:,:), reicmcl(:,:), resnmcl(:,:)
  real, allocatable :: taucld(:,:,:), cldfmcl(:,:,:), clwpmcl(:,:,:)
  real, allocatable :: ciwpmcl(:,:,:), cswpmcl(:,:,:), taucmcl(:,:,:)
  real, allocatable :: tauaer(:,:,:)
  real, allocatable :: uflx(:,:), dflx(:,:), uflxc(:,:), dflxc(:,:)
  real, allocatable :: uflxcln(:,:), dflxcln(:,:), hr(:,:), hrc(:,:)
  real, allocatable :: pdel(:,:), cicewp(:,:), cliqwp(:,:), csnowp(:,:)
  real, allocatable :: reliq(:,:), reice(:,:)
  real, allocatable :: recloud1d(:,:), reice1d(:,:), resnow1d(:,:)
  real, dimension(1) :: tsfc, landfrac, landm, snowh, icefrac
  integer :: pcols, pver
  real(8) :: co2, n2o, ch4, cfc11, cfc12
  real :: cfc22, ccl4, o2
  real :: pprof(60), tprof(60)
  real :: amdw, amdo, amdo2
  real, parameter :: thresh = 1.e-9

  ! ---- rrtmg_lw-local mirrors for the decomposed chain ----
  integer :: istart, iend, iout, iaer, dnlayers
  integer :: inflag, iceflag, liqflag, ncbands, laytrop
  real, allocatable :: pavel(:), tavel(:), pz(:), tz(:)
  real :: tbound, pwvcm
  real, allocatable :: coldry(:), wbrodl(:), wkl(:,:), wx(:,:), semiss(:)
  real, allocatable :: fracs(:,:), taug(:,:), taut(:,:), taua(:,:)
  integer, allocatable :: jp(:), jt(:), jt1(:)
  real, allocatable :: planklay(:,:), planklev(:,:), plankbnd(:)
  real, allocatable :: colh2o(:), colco2(:), colo3(:), coln2o(:), colco(:)
  real, allocatable :: colch4(:), colo2(:), colbrd(:)
  integer, allocatable :: indself(:), indfor(:), indminor(:)
  real, allocatable :: selffac(:), selffrac(:), forfac(:), forfrac(:)
  real, allocatable :: minorfrac(:), scaleminor(:), scaleminorn2(:)
  real, allocatable :: fac00(:), fac01(:), fac10(:), fac11(:)
  real, allocatable :: rat_h2oco2(:), rat_h2oco2_1(:), rat_h2oo3(:), rat_h2oo3_1(:)
  real, allocatable :: rat_h2on2o(:), rat_h2on2o_1(:), rat_h2och4(:), rat_h2och4_1(:)
  real, allocatable :: rat_n2oco2(:), rat_n2oco2_1(:), rat_o3co2(:), rat_o3co2_1(:)
  real, allocatable :: cldfmc(:,:), taucmc(:,:), ciwpmc(:,:), clwpmc(:,:), cswpmc(:,:)
  real, allocatable :: reicmc(:), relqmc(:), resnmc(:)
  real, allocatable :: totuflux(:), totdflux(:), fnet(:), htr(:)
  real, allocatable :: totuclfl(:), totdclfl(:), fnetc(:), htrc(:)
  real, allocatable :: duflx(:,:), ddflx(:,:), duflxc(:,:), ddflxc(:,:)
  real, allocatable :: duflxcln(:,:), ddflxcln(:,:), dhr(:,:), dhrc(:,:)

  ! ---- direct rrtmg_lwrad cross-check (WRF-shaped 3D arrays) ----
  real, allocatable, dimension(:,:,:) :: w_p8w, w_p3d, w_pi3d, w_dz8w, w_t3d
  real, allocatable, dimension(:,:,:) :: w_t8w, w_rho3d
  real, allocatable, dimension(:,:,:) :: w_qv, w_qc, w_qr, w_qi, w_qs, w_qg
  real, allocatable, dimension(:,:,:) :: w_cldfra, w_o33d, w_rec, w_rei, w_res
  real, allocatable, dimension(:,:,:) :: w_rthlw, w_rthlwc
  real, dimension(1,1) :: w_glw, w_olr, w_lwcf, w_emiss, w_tsk, w_xland
  real, dimension(1,1) :: w_xice, w_snow, w_xlat
  real, dimension(1,1) :: w_lwupt, w_lwuptc, w_lwuptcln, w_lwdnt, w_lwdntc
  real, dimension(1,1) :: w_lwdntcln, w_lwupb, w_lwupbc, w_lwupbcln
  real, dimension(1,1) :: w_lwdnb, w_lwdnbc, w_lwdnbcln
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms_, kme_
  integer :: its, ite, jts, jte, kts_, kte_

  ! Weighted mean pressure/temperature standard profiles (RRTMG_LWRAD DATA)
  data pprof /1000.00,855.47,731.82,626.05,535.57,458.16,      &
              391.94,335.29,286.83,245.38,209.91,179.57,       &
              153.62,131.41,112.42,96.17,82.27,70.38,           &
              60.21,51.51,44.06,37.69,32.25,27.59,              &
              23.60,20.19,17.27,14.77,12.64,10.81,              &
              9.25,7.91,6.77,5.79,4.95,4.24,                    &
              3.63,3.10,2.65,2.27,1.94,1.66,                    &
              1.42,1.22,1.04,0.89,0.76,0.65,                    &
              0.56,0.48,0.41,0.35,0.30,0.26,                    &
              0.22,0.19,0.16,0.14,0.12,0.10/
  data tprof /286.96,281.07,275.16,268.11,260.56,253.02,        &
              245.62,238.41,231.57,225.91,221.72,217.79,        &
              215.06,212.74,210.25,210.16,210.69,212.14,        &
              213.74,215.37,216.82,217.94,219.03,220.18,        &
              221.37,222.64,224.16,225.88,227.63,229.51,        &
              231.50,233.73,236.18,238.78,241.60,244.44,        &
              247.35,250.33,253.32,256.30,259.22,262.12,        &
              264.80,266.50,267.59,268.44,268.69,267.76,        &
              266.13,263.96,261.54,258.93,256.15,253.23,        &
              249.89,246.67,243.48,240.25,236.66,233.86/
  data amdw / 1.607793 /
  data amdo / 0.603461 /
  data amdo2 / 0.905190 /
  data cfc22 / 0.169e-9 /
  data ccl4 / 0.093e-9 /
  data o2 / 0.209488 /

  nproflevs = 60

  if (command_argument_count() /= 2) then
     write(*,'(A)') 'usage: lw_extract input.txt outdir'
     error stop 4
  end if
  call get_command_argument(1, infile)
  call get_command_argument(2, outdir)

  open(newunit=iu, file=trim(infile), status='old', action='read')
  read(iu,*) ncase, nz

  kts = 1
  kte = nz
  kme = nz + 1

  allocate(p3d_in(nz), t3d_in(nz), dz_in(nz), pi3d_in(nz), rho_in(nz))
  allocate(qv_in(nz), qc_in(nz), qr_in(nz), qi_in(nz), qs_in(nz), qg_in(nz))
  allocate(cldfra_in(nz), rec_in(nz), rei_in(nz), res_in(nz), o3_in(nz))
  allocate(p8w_in(nz+1), t8w_in(nz+1))
  allocate(pw1d(nz+1), tw1d(nz+1))
  allocate(tten1d(nz), cldfra1d(nz), dz1d(nz), p1d(nz), t1d(nz))
  allocate(qv1d(nz), qc1d(nz), qr1d(nz), qi1d(nz), qs1d(nz), qg1d(nz), o31d(nz))
  allocate(pdel(1,nz), cicewp(1,nz), cliqwp(1,nz), csnowp(1,nz))
  allocate(reliq(1,nz), reice(1,nz))
  allocate(recloud1d(1,nz), reice1d(1,nz), resnow1d(1,nz))

  inited = .false.

  do icase = 1, ncase
     read(iu,*) caseid, yr, julday, mp_physics, icloud, cldovrlp, idcor, &
                o3input, has_reqc, has_reqi, has_reqs, ifqv, ifqc, ifqr, &
                ifqi, ifqs, ifqg, iwarm
     read(iu,*) julian, xlat0, xlong0, emiss0, tsk0, psfc0, xland0, xice0, &
                snow0, snowh0, p_top
     do k = 1, nz
        read(iu,*) p3d_in(k), t3d_in(k), dz_in(k), pi3d_in(k), rho_in(k), &
                   qv_in(k), qc_in(k), qr_in(k), qi_in(k), qs_in(k), &
                   qg_in(k), cldfra_in(k), rec_in(k), rei_in(k), res_in(k), &
                   o3_in(k)
     end do
     do k = 1, nz+1
        read(iu,*) p8w_in(k), t8w_in(k)
     end do
     warm_rain = (iwarm /= 0)
     f_qi_l = (ifqi /= 0)

     if (.not. inited) then
        p_top_init = p_top
        ids=1; ide=2; jds=1; jde=2; kds=1; kde=nz+1
        ims=1; ime=1; jms=1; jme=1; kms_=1; kme_=nz+1
        its=1; ite=1; jts=1; jte=1; kts_=1; kte_=nz
        call rrtmg_lwinit(p_top, .true., ids,ide,jds,jde,kds,kde, &
                          ims,ime,jms,jme,kms_,kme_, its,ite,jts,jte,kts_,kte_)
        nlayers = mod_nlayers
        write(*,'(A,I0)') 'lw_extract: nlayers = ', nlayers
        write(path,'(A,A)') trim(outdir), '/lw_coeffs.bin'
        call dump_all_state(trim(path))
        call alloc_layered()
        inited = .true.
     else if (p_top /= p_top_init) then
        write(*,'(A)') 'p_top changed between cases; single init broken'
        error stop 5
     end if

     ! =================================================================
     ! (1) Transcription of RRTMG_LWRAD prep (i=1, j=1, ncol=1).
     ! =================================================================
     if (yr < 0) then
        write(*,'(A)') 'ghg_input=1 path is not in oracle scope'
        error stop 6
     end if
     co2 = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6
     ch4 = 1774.e-9
     n2o = 319.e-9
     cfc11 = 0.251e-9
     cfc12 = 0.538e-9

     do k = kts, kte+1
        pw1d(k) = p8w_in(k)/100.
        tw1d(k) = t8w_in(k)
     end do
     do k = kts, kte
        qv1d(k) = 0.
        qc1d(k) = 0.
        qr1d(k) = 0.
        qi1d(k) = 0.
        qs1d(k) = 0.
        cldfra1d(k) = 0.
     end do
     do k = kts, kte
        qv1d(k) = qv_in(k)
        qv1d(k) = max(0., qv1d(k))
     end do
     if (o3input .eq. 2) then
        do k = kts, kte
           o31d(k) = o3_in(k)
        end do
     else
        do k = kts, kte
           o31d(k) = 0.0
        end do
     end if
     do k = kts, kte
        tten1d(k) = 0.
        t1d(k) = t3d_in(k)
        p1d(k) = p3d_in(k)/100.
        dz1d(k) = dz_in(k)
     end do

     if (icloud .ne. 0) then
        do k = kts, kte
           cldfra1d(k) = cldfra_in(k)
        end do
        if (ifqc /= 0) then
           do k = kts, kte
              qc1d(k) = qc_in(k)
              qc1d(k) = max(0., qc1d(k))
           end do
        end if
        if (ifqr /= 0) then
           do k = kts, kte
              qr1d(k) = qr_in(k)
              qr1d(k) = max(0., qr1d(k))
           end do
        end if
        ! qndrop path: F_QNDROP absent/false in oracle
        if (.not. f_qi_l .and. .not. warm_rain) then
           do k = kts, kte
              if (t1d(k) .lt. 273.15) then
                 qi1d(k) = qc1d(k)
                 qs1d(k) = qr1d(k)
                 qc1d(k) = 0.
                 qr1d(k) = 0.
              end if
           end do
        end if
        if (ifqi /= 0) then
           do k = kts, kte
              qi1d(k) = qi_in(k)
              qi1d(k) = max(0., qi1d(k))
           end do
        end if
        if (ifqs /= 0) then
           do k = kts, kte
              qs1d(k) = qs_in(k)
              qs1d(k) = max(0., qs1d(k))
           end do
        end if
        if (ifqg /= 0) then
           do k = kts, kte
              qg1d(k) = qg_in(k)
              qg1d(k) = max(0., qg1d(k))
           end do
        end if
        ! MP option 5 branch requires F_ICE_PHY present; oracle pins it off.
     end if
     ! Ferrier (mp 5/15) branch: campaign schemes only, pinned off.
     if (mp_physics == 5 .or. mp_physics == 15) then
        write(*,'(A)') 'Ferrier mp branch not in oracle scope'
        error stop 7
     end if

     do k = kts, kte
        qv1d(k) = amax1(qv1d(k), 1.e-12)
     end do

     ncol = 1
     nlay = nlayers
     icld = cldovrlp
     juldat = julian
     inflglw = 2
     iceflglw = 3
     liqflglw = 1

     if (icloud .ne. 0) then
        if (has_reqc .ne. 0) then
           inflglw = 3
           do k = kts, kte
              recloud1d(ncol,k) = max(2.5, rec_in(k)*1.e6)
              if (recloud1d(ncol,k).le.2.5 .and. cldfra_in(k).gt.0. &
                  .and. (xland0-1.5).gt.0.) then
                 recloud1d(ncol,k) = 10.5
              elseif (recloud1d(ncol,k).le.2.5 .and. cldfra_in(k).gt.0. &
                  .and. (xland0-1.5).lt.0.) then
                 recloud1d(ncol,k) = 7.5
              endif
           end do
        else
           do k = kts, kte
              recloud1d(ncol,k) = 5.0
           end do
        end if

        if (has_reqi .ne. 0) then
           inflglw = 4
           iceflglw = 4
           do k = kts, kte
              reice1d(ncol,k) = max(5., rei_in(k)*1.e6)
              if (reice1d(ncol,k).le.5. .and. cldfra_in(k).gt.0.) then
                 idx_rei = int(t3d_in(k)-179.)
                 idx_rei = min(max(idx_rei,1),75)
                 corr = t3d_in(k) - int(t3d_in(k))
                 reice1d(ncol,k) = retab(idx_rei)*(1.-corr) + &
                                   retab(idx_rei+1)*corr
                 reice1d(ncol,k) = max(reice1d(ncol,k), 5.0)
              endif
           end do
        else
           do k = kts, kte
              reice1d(ncol,k) = 10.0    ! EM_CORE==1 branch
           end do
        end if

        if (has_reqs .ne. 0) then
           inflglw = 5
           iceflglw = 5
           do k = kts, kte
              resnow1d(ncol,k) = max(10., res_in(k)*1.e6)
           end do
        else
           do k = kts, kte
              resnow1d(ncol,k) = 10.0
           end do
        end if

        ! special case for P3 microphysics
        if (has_reqs .eq. 0 .and. has_reqi .ne. 0 .and. has_reqc .ne. 0) then
           inflglw = 5
           iceflglw = 5
           do k = kts, kte
              resnow1d(ncol,k) = max(10., rei_in(k)*1.e6)
              qs1d(k) = qi_in(k)
              qi1d(k) = 0.
              reice1d(ncol,k) = 10.
           end do
        end if
     end if

     plev(ncol,1) = pw1d(1)
     tlev(ncol,1) = tw1d(1)
     tsfc(ncol) = tsk0
     do k = kts, kte
        play(ncol,k) = p1d(k)
        plev(ncol,k+1) = pw1d(k+1)
        pdel(ncol,k) = plev(ncol,k) - plev(ncol,k+1)
        tlay(ncol,k) = t1d(k)
        tlev(ncol,k+1) = tw1d(k+1)
        h2ovmr(ncol,k) = qv1d(k) * amdw
        co2vmr(ncol,k) = co2
        o2vmr(ncol,k) = o2
        ch4vmr(ncol,k) = ch4
        n2ovmr(ncol,k) = n2o
        cfc11vmr(ncol,k) = cfc11
        cfc12vmr(ncol,k) = cfc12
        cfc22vmr(ncol,k) = cfc22
        ccl4vmr(ncol,k) = ccl4
     end do

     dzsum = 0.0
     do k = kts, kte
        dz = dz1d(k)
        hgt(ncol,k) = dzsum + 0.5*dz
        dzsum = dzsum + dz
     end do

     ! Buffer levels above model top (Cavallo)
     do l = kte+1, nlayers, 1
        plev(ncol,l+1) = plev(ncol,l) - deltap
        play(ncol,l) = 0.5*(plev(ncol,l) + plev(ncol,l+1))
        hgt(ncol,l) = dzsum + 0.5*dz
        dzsum = dzsum + dz
     end do
     plev(ncol,nlayers+1) = 0.00
     play(ncol,nlayers) = 0.5*(plev(ncol,nlayers) + plev(ncol,nlayers+1))

     do l = 1, nlayers+1, 1
        if ( pprof(nproflevs) .lt. plev(ncol,l) ) then
           do ll = 2, nproflevs, 1
              if ( pprof(ll) .lt. plev(ncol,l) ) then
                 klev = ll - 1
                 exit
              endif
           end do
        else
           klev = nproflevs
        endif
        if (klev .ne. nproflevs) then
           vark  = tprof(klev)
           vark1 = tprof(klev+1)
           wght = (plev(ncol,l)-pprof(klev)) / (pprof(klev+1)-pprof(klev))
        else
           vark  = tprof(klev)
           vark1 = tprof(klev)
           wght = 0.0
        endif
        varint(l) = wght*(vark1-vark) + vark
     end do

     do l = kte+1, nlayers+1, 1
        tlev(ncol,l) = varint(l) + (tlev(ncol,kte) - varint(kte))
        tlay(ncol,l-1) = 0.5*(tlev(ncol,l) + tlev(ncol,l-1))
     end do

     do l = kte+1, nlayers, 1
        h2ovmr(ncol,l) = h2ovmr(ncol,kte)
        co2vmr(ncol,l) = co2vmr(ncol,kte)
        o2vmr(ncol,l) = o2vmr(ncol,kte)
        ch4vmr(ncol,l) = ch4vmr(ncol,kte)
        n2ovmr(ncol,l) = n2ovmr(ncol,kte)
        cfc11vmr(ncol,l) = cfc11vmr(ncol,kte)
        cfc12vmr(ncol,l) = cfc12vmr(ncol,kte)
        cfc22vmr(ncol,l) = cfc22vmr(ncol,kte)
        ccl4vmr(ncol,l) = ccl4vmr(ncol,kte)
     end do

     call inirad(o3mmr, plev, kts, nlay-1)

     if (o3input .eq. 2) then
        do k = kts, nlayers
           o3vmr(ncol,k) = o3mmr(k) * amdo
           if (k .le. kte) then
              o3vmr(ncol,k) = o31d(k)
           else
              o3vmr(ncol,k) = o31d(kte) - o3mmr(kte)*amdo + o3mmr(k)*amdo
              if (o3vmr(ncol,k) .le. 0.) o3vmr(ncol,k) = o3mmr(k)*amdo
           endif
        end do
     else
        do k = kts, nlayers
           o3vmr(ncol,k) = o3mmr(k) * amdo
           if (k .le. kte) o31d(k) = o3vmr(ncol,k)
        end do
     endif

     do nb = 1, nbndlw
        emis(ncol, nb) = emiss0
     end do

     ! inflglw==0 branch unreachable in WRF v4.6.1 (wrapper sets 2..5)
     if (inflglw .gt. 0) then
        do k = kts, kte
           cldfrac(ncol,k) = cldfra1d(k)
        end do

        pcols = ncol
        pver = kte - kts + 1
        gravmks = 9.81
        landfrac(ncol) = 2.-xland0
        landm(ncol) = landfrac(ncol)
        snowh(ncol) = 0.001*snow0
        icefrac(ncol) = xice0

        do k = kts, kte
           gicewp = (qi1d(k)+qs1d(k)) * pdel(ncol,k)*100.0 / gravmks * 1000.0
           gliqwp = qc1d(k) * pdel(ncol,k)*100.0 / gravmks * 1000.0
           cicewp(ncol,k) = gicewp / max(0.01,cldfrac(ncol,k))
           cliqwp(ncol,k) = gliqwp / max(0.01,cldfrac(ncol,k))
        end do

        if (iceflglw .ge. 4) then
           do k = kts, kte
              gicewp = qi1d(k) * pdel(ncol,k)*100.0 / gravmks * 1000.0
              cicewp(ncol,k) = gicewp / max(0.01,cldfrac(ncol,k))
           end do
        end if

        if (iceflglw .eq. 5) then
           do k = kts, kte
              snow_mass_factor = 0.99
              gicewp = gicewp + (qs1d(k)*(1.0-snow_mass_factor) * &
                       pdel(ncol,k)*100.0 / gravmks * 1000.0)
              if (resnow1d(ncol,k) .gt. 130.) then
                 snow_mass_factor = min(snow_mass_factor, &
                      (130.0/resnow1d(ncol,k))*(130.0/resnow1d(ncol,k)))
                 resnow1d(ncol,k) = 130.0
              endif
              gsnowp = qs1d(k) * snow_mass_factor * &
                       pdel(ncol,k)*100.0 / gravmks * 1000.0
              csnowp(ncol,k) = gsnowp / max(0.01,cldfrac(ncol,k))
           end do
        end if

        ! progn=0 path
        call relcalc(ncol, pcols, pver, tlay, landfrac, landm, icefrac, &
                     reliq, snowh)
        call reicalc(ncol, pcols, pver, tlay, reice)

        if (inflglw .ge. 3) then
           do k = kts, kte
              reliq(ncol,k) = recloud1d(ncol,k)
           end do
        endif
        if (iceflglw .ge. 4) then       ! EM_CORE==1
           do k = kts, kte
              reice(ncol,k) = reice1d(ncol,k)
           end do
        endif
        if (iceflglw .eq. 3) then
           do k = kts, kte
              reice(ncol,k) = reice(ncol,k) * 1.0315
              reice(ncol,k) = min(140.0,reice(ncol,k))
           end do
        endif
        ! is_CAMMGMP_used = .false. in oracle

        do k = kts, kte
           clwpth(ncol,k) = cliqwp(ncol,k)
           ciwpth(ncol,k) = cicewp(ncol,k)
           rel(ncol,k) = reliq(ncol,k)
           rei(ncol,k) = reice(ncol,k)
        end do

        if (inflglw .eq. 5) then
           do k = kts, kte
              cswpth(ncol,k) = csnowp(ncol,k)
              res(ncol,k) = resnow1d(ncol,k)
           end do
        else
           do k = kts, kte
              cswpth(ncol,k) = 0.
              res(ncol,k) = 10.
           end do
        endif

        do k = kts, kte
           do nb = 1, nbndlw
              taucld(nb,ncol,k) = 0.0
           end do
        end do
     end if

     do k = kte+1, nlayers
        clwpth(ncol,k) = 0.
        ciwpth(ncol,k) = 0.
        cswpth(ncol,k) = 0.
        rel(ncol,k) = 10.
        rei(ncol,k) = 10.
        res(ncol,k) = 10.
        cldfrac(ncol,k) = 0.
        do nb = 1, nbndlw
           taucld(nb,ncol,k) = 0.
        end do
     end do

     iplon = 1
     irng = 0
     permuteseed = 150
     lat = xlat0

     call mcica_subcol_lw(iplon, ncol, nlay, icld, permuteseed, irng, play, &
          cldfrac, ciwpth, clwpth, cswpth, rei, rel, res, taucld, &
          hgt, idcor, juldat, lat, &
          cldfmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, taucmcl)

     do nb = 1, nbndlw
        do k = kts, nlayers
           tauaer(ncol,k,nb) = 0.
        end do
     end do

     ! =================================================================
     ! (2) Dump the rrtmg_lw-level arguments, then run decomposed chain.
     ! =================================================================
     write(path,'(A,A,I4.4,A)') trim(outdir), '/lw_case_', caseid, '.bin'
     call bio_open(trim(path))

     call wr_i0('meta/caseid', caseid)
     call wr_i0('meta/yr', yr)
     call wr_r0('meta/julian', julian)
     call wr_i0('meta/mp_physics', mp_physics)
     call wr_i0('meta/icloud', icloud)
     call wr_i0('meta/cldovrlp', cldovrlp)
     call wr_i0('meta/idcor', idcor)
     call wr_i0('meta/o3input', o3input)
     call wr_i0('meta/has_reqc', has_reqc)
     call wr_i0('meta/has_reqi', has_reqi)
     call wr_i0('meta/has_reqs', has_reqs)
     call wr_i0('meta/nz', nz)
     call wr_i0('meta/nlayers', nlayers)
     call wr_i0('meta/ngptlw', ngptlw)
     call wr_i0('meta/nbndlw', nbndlw)
     call wr_i0('meta/mxmol', mxmol)
     call wr_i0('meta/maxxsec', maxxsec)
     call wr_r0('meta/xlat', xlat0)
     call wr_r0('meta/xlong', xlong0)
     call wr_r0('meta/p_top', p_top)

     ! Raw wrapper-level inputs, for the future driver-integration wave.
     call wr_r1('wrfin/p3d', p3d_in)
     call wr_r1('wrfin/t3d', t3d_in)
     call wr_r1('wrfin/dz8w', dz_in)
     call wr_r1('wrfin/pi3d', pi3d_in)
     call wr_r1('wrfin/rho3d', rho_in)
     call wr_r1('wrfin/qv', qv_in)
     call wr_r1('wrfin/qc', qc_in)
     call wr_r1('wrfin/qr', qr_in)
     call wr_r1('wrfin/qi', qi_in)
     call wr_r1('wrfin/qs', qs_in)
     call wr_r1('wrfin/qg', qg_in)
     call wr_r1('wrfin/cldfra', cldfra_in)
     call wr_r1('wrfin/re_cloud', rec_in)
     call wr_r1('wrfin/re_ice', rei_in)
     call wr_r1('wrfin/re_snow', res_in)
     call wr_r1('wrfin/o33d', o3_in)
     call wr_r1('wrfin/p8w', p8w_in)
     call wr_r1('wrfin/t8w', t8w_in)
     call wr_r0('wrfin/emiss', emiss0)
     call wr_r0('wrfin/tsk', tsk0)
     call wr_r0('wrfin/xland', xland0)
     call wr_r0('wrfin/xice', xice0)
     call wr_r0('wrfin/snow', snow0)

     ! Exact rrtmg_lw arguments (the frozen port-boundary interface).
     call wr_i0('in/ncol', ncol)
     call wr_i0('in/nlay', nlay)
     call wr_i0('in/icld', icld)
     call wr_r2('in/play', play)
     call wr_r2('in/plev', plev)
     call wr_r2('in/tlay', tlay)
     call wr_r2('in/tlev', tlev)
     call wr_r1('in/tsfc', tsfc)
     call wr_r2('in/h2ovmr', h2ovmr)
     call wr_r2('in/o3vmr', o3vmr)
     call wr_r2('in/co2vmr', co2vmr)
     call wr_r2('in/ch4vmr', ch4vmr)
     call wr_r2('in/n2ovmr', n2ovmr)
     call wr_r2('in/o2vmr', o2vmr)
     call wr_r2('in/cfc11vmr', cfc11vmr)
     call wr_r2('in/cfc12vmr', cfc12vmr)
     call wr_r2('in/cfc22vmr', cfc22vmr)
     call wr_r2('in/ccl4vmr', ccl4vmr)
     call wr_r2('in/emis', emis)
     call wr_i0('in/inflglw', inflglw)
     call wr_i0('in/iceflglw', iceflglw)
     call wr_i0('in/liqflglw', liqflglw)
     call wr_r3('in/cldfmcl', cldfmcl)
     call wr_r3('in/taucmcl', taucmcl)
     call wr_r3('in/ciwpmcl', ciwpmcl)
     call wr_r3('in/clwpmcl', clwpmcl)
     call wr_r3('in/cswpmcl', cswpmcl)
     call wr_r2('in/reicmcl', reicmcl)
     call wr_r2('in/relqmcl', relqmcl)
     call wr_r2('in/resnmcl', resnmcl)
     call wr_r3('in/tauaer', tauaer)

     ! ---- decomposed chain, mirroring the rrtmg_lw body ----
     istart = 1
     iend = 16
     iout = 0
     iaer = 10

     call inatm(iplon, nlay, icld, iaer, &
          play, plev, tlay, tlev, tsfc, h2ovmr, &
          o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr, cfc12vmr, &
          cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, &
          cldfmcl, taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, &
          resnmcl, tauaer, &
          dnlayers, pavel, pz, tavel, tz, tbound, semiss, coldry, &
          wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, &
          cldfmc, taucmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, taua)

     call wr_i0('inatm/nlayers', dnlayers)
     call wr_r1('inatm/pavel', pavel(1:dnlayers))
     call wr_r1('inatm/tavel', tavel(1:dnlayers))
     call wr_r1('inatm/pz', pz(0:dnlayers))
     call wr_r1('inatm/tz', tz(0:dnlayers))
     call wr_r0('inatm/tbound', tbound)
     call wr_r1('inatm/semiss', semiss)
     call wr_r1('inatm/coldry', coldry(1:dnlayers))
     call wr_r1('inatm/wbrodl', wbrodl(1:dnlayers))
     call wr_r2('inatm/wkl', wkl(:,1:dnlayers))
     call wr_r2('inatm/wx', wx(:,1:dnlayers))
     call wr_r0('inatm/pwvcm', pwvcm)
     call wr_i0('inatm/inflag', inflag)
     call wr_i0('inatm/iceflag', iceflag)
     call wr_i0('inatm/liqflag', liqflag)
     call wr_r2('inatm/cldfmc', cldfmc(:,1:dnlayers))
     call wr_r2('inatm/taucmc', taucmc(:,1:dnlayers))
     call wr_r2('inatm/ciwpmc', ciwpmc(:,1:dnlayers))
     call wr_r2('inatm/clwpmc', clwpmc(:,1:dnlayers))
     call wr_r2('inatm/cswpmc', cswpmc(:,1:dnlayers))
     call wr_r1('inatm/reicmc', reicmc(1:dnlayers))
     call wr_r1('inatm/relqmc', relqmc(1:dnlayers))
     call wr_r1('inatm/resnmc', resnmc(1:dnlayers))
     call wr_r2('inatm/taua', taua(1:dnlayers,:))

     call cldprmc(dnlayers, inflag, iceflag, liqflag, cldfmc, ciwpmc, &
                  clwpmc, cswpmc, reicmc, relqmc, resnmc, ncbands, taucmc)

     call wr_i0('cldprmc/ncbands', ncbands)
     call wr_r2('cldprmc/taucmc', taucmc(:,1:dnlayers))

     call setcoef(dnlayers, istart, pavel, tavel, tz, tbound, semiss, &
                  coldry, wkl, wbrodl, &
                  laytrop, jp, jt, jt1, planklay, planklev, plankbnd, &
                  colh2o, colco2, colo3, coln2o, colco, colch4, colo2, &
                  colbrd, fac00, fac01, fac10, fac11, &
                  rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
                  rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
                  rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
                  selffac, selffrac, indself, forfac, forfrac, indfor, &
                  minorfrac, scaleminor, scaleminorn2, indminor)

     call wr_i0('setcoef/laytrop', laytrop)
     call wr_i1('setcoef/jp', jp(1:dnlayers))
     call wr_i1('setcoef/jt', jt(1:dnlayers))
     call wr_i1('setcoef/jt1', jt1(1:dnlayers))
     call wr_r2('setcoef/planklay', planklay(1:dnlayers,:))
     call wr_r2('setcoef/planklev', planklev(0:dnlayers,:))
     call wr_r1('setcoef/plankbnd', plankbnd)
     call wr_r1('setcoef/colh2o', colh2o(1:dnlayers))
     call wr_r1('setcoef/colco2', colco2(1:dnlayers))
     call wr_r1('setcoef/colo3', colo3(1:dnlayers))
     call wr_r1('setcoef/coln2o', coln2o(1:dnlayers))
     call wr_r1('setcoef/colco', colco(1:dnlayers))
     call wr_r1('setcoef/colch4', colch4(1:dnlayers))
     call wr_r1('setcoef/colo2', colo2(1:dnlayers))
     call wr_r1('setcoef/colbrd', colbrd(1:dnlayers))
     call wr_r1('setcoef/fac00', fac00(1:dnlayers))
     call wr_r1('setcoef/fac01', fac01(1:dnlayers))
     call wr_r1('setcoef/fac10', fac10(1:dnlayers))
     call wr_r1('setcoef/fac11', fac11(1:dnlayers))
     call wr_r1('setcoef/rat_h2oco2', rat_h2oco2(1:dnlayers))
     call wr_r1('setcoef/rat_h2oco2_1', rat_h2oco2_1(1:dnlayers))
     call wr_r1('setcoef/rat_h2oo3', rat_h2oo3(1:dnlayers))
     call wr_r1('setcoef/rat_h2oo3_1', rat_h2oo3_1(1:dnlayers))
     call wr_r1('setcoef/rat_h2on2o', rat_h2on2o(1:dnlayers))
     call wr_r1('setcoef/rat_h2on2o_1', rat_h2on2o_1(1:dnlayers))
     call wr_r1('setcoef/rat_h2och4', rat_h2och4(1:dnlayers))
     call wr_r1('setcoef/rat_h2och4_1', rat_h2och4_1(1:dnlayers))
     call wr_r1('setcoef/rat_n2oco2', rat_n2oco2(1:dnlayers))
     call wr_r1('setcoef/rat_n2oco2_1', rat_n2oco2_1(1:dnlayers))
     call wr_r1('setcoef/rat_o3co2', rat_o3co2(1:dnlayers))
     call wr_r1('setcoef/rat_o3co2_1', rat_o3co2_1(1:dnlayers))
     call wr_r1('setcoef/selffac', selffac(1:dnlayers))
     call wr_r1('setcoef/selffrac', selffrac(1:dnlayers))
     call wr_i1('setcoef/indself', indself(1:dnlayers))
     call wr_r1('setcoef/forfac', forfac(1:dnlayers))
     call wr_r1('setcoef/forfrac', forfrac(1:dnlayers))
     call wr_i1('setcoef/indfor', indfor(1:dnlayers))
     call wr_r1('setcoef/minorfrac', minorfrac(1:dnlayers))
     call wr_r1('setcoef/scaleminor', scaleminor(1:dnlayers))
     call wr_r1('setcoef/scaleminorn2', scaleminorn2(1:dnlayers))
     call wr_i1('setcoef/indminor', indminor(1:dnlayers))

     call taumol(dnlayers, pavel, wx, coldry, &
                 laytrop, jp, jt, jt1, planklay, planklev, plankbnd, &
                 colh2o, colco2, colo3, coln2o, colco, colch4, colo2, &
                 colbrd, fac00, fac01, fac10, fac11, &
                 rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
                 rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
                 rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
                 selffac, selffrac, indself, forfac, forfrac, indfor, &
                 minorfrac, scaleminor, scaleminorn2, indminor, &
                 fracs, taug)

     call wr_r2('taumol/fracs', fracs(1:dnlayers,:))
     call wr_r2('taumol/taug', taug(1:dnlayers,:))

     do k = 1, dnlayers
        do ig = 1, ngptlw
           taut(k,ig) = taug(k,ig) + taua(k,ngb(ig))
        end do
     end do
     call wr_r2('taut/taut', taut(1:dnlayers,:))

     call rtrnmc(dnlayers, istart, iend, iout, pz, semiss, ncbands, &
                 cldfmc, taucmc, planklay, planklev, plankbnd, &
                 pwvcm, fracs, taut, &
                 totuflux, totdflux, fnet, htr, &
                 totuclfl, totdclfl, fnetc, htrc)

     call wr_r1('rtrnmc/totuflux', totuflux(0:dnlayers))
     call wr_r1('rtrnmc/totdflux', totdflux(0:dnlayers))
     call wr_r1('rtrnmc/fnet', fnet(0:dnlayers))
     call wr_r1('rtrnmc/htr', htr(0:dnlayers))
     call wr_r1('rtrnmc/totuclfl', totuclfl(0:dnlayers))
     call wr_r1('rtrnmc/totdclfl', totdclfl(0:dnlayers))
     call wr_r1('rtrnmc/fnetc', fnetc(0:dnlayers))
     call wr_r1('rtrnmc/htrc', htrc(0:dnlayers))

     ! =================================================================
     ! (3) Direct rrtmg_lw call -- composition authority.
     ! =================================================================
     call rrtmg_lw &
          (ncol, nlay, icld, &
           play, plev, tlay, tlev, tsfc, &
           h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, &
           cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, &
           inflglw, iceflglw, liqflglw, cldfmcl, &
           taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, &
           tauaer, &
           duflx, ddflx, dhr, duflxc, ddflxc, dhrc, &
           duflxcln, ddflxcln, 0)

     call wr_r2('out/uflx', duflx)
     call wr_r2('out/dflx', ddflx)
     call wr_r2('out/hr', dhr)
     call wr_r2('out/uflxc', duflxc)
     call wr_r2('out/dflxc', ddflxc)
     call wr_r2('out/hrc', dhrc)

     ! =================================================================
     ! (4) Direct RRTMG_LWRAD call -- wrapper-transcription authority.
     ! =================================================================
     call run_wrapper_check()

     call bio_close()
     write(*,'(A,I0,A)') 'lw_extract: case ', caseid, ' done'
  end do

  close(iu)
  write(*,'(A,I0,A)') 'lw_extract: ', ncase, ' cases complete'

contains

  subroutine alloc_layered()
    allocate(plev(1,nlayers+1), tlev(1,nlayers+1))
    allocate(play(1,nlayers), tlay(1,nlayers))
    allocate(h2ovmr(1,nlayers), o3vmr(1,nlayers), co2vmr(1,nlayers))
    allocate(o2vmr(1,nlayers), ch4vmr(1,nlayers), n2ovmr(1,nlayers))
    allocate(cfc11vmr(1,nlayers), cfc12vmr(1,nlayers))
    allocate(cfc22vmr(1,nlayers), ccl4vmr(1,nlayers))
    allocate(o3mmr(nlayers), hgt(1,nlayers), varint(nlayers+1))
    allocate(emis(1,nbndlw))
    allocate(clwpth(1,nlayers), ciwpth(1,nlayers), cswpth(1,nlayers))
    allocate(rel(1,nlayers), rei(1,nlayers), res(1,nlayers))
    allocate(cldfrac(1,nlayers))
    allocate(relqmcl(1,nlayers), reicmcl(1,nlayers), resnmcl(1,nlayers))
    allocate(taucld(nbndlw,1,nlayers))
    allocate(cldfmcl(ngptlw,1,nlayers), clwpmcl(ngptlw,1,nlayers))
    allocate(ciwpmcl(ngptlw,1,nlayers), cswpmcl(ngptlw,1,nlayers))
    allocate(taucmcl(ngptlw,1,nlayers))
    allocate(tauaer(1,nlayers,nbndlw))
    allocate(uflx(1,nlayers+1), dflx(1,nlayers+1))
    allocate(uflxc(1,nlayers+1), dflxc(1,nlayers+1))
    allocate(uflxcln(1,nlayers+1), dflxcln(1,nlayers+1))
    allocate(hr(1,nlayers), hrc(1,nlayers))
    ! rrtmg_lw local mirrors (declared sizes follow the rrtmg_lw body)
    allocate(pavel(nlayers+1), tavel(nlayers+1))
    allocate(pz(0:nlayers+1), tz(0:nlayers+1))
    allocate(coldry(nlayers+1), wbrodl(nlayers+1))
    allocate(wkl(mxmol,nlayers+1), wx(maxxsec,nlayers+1), semiss(nbndlw))
    allocate(fracs(nlayers+1,ngptlw), taug(nlayers+1,ngptlw))
    allocate(taut(nlayers+1,ngptlw), taua(nlayers+1,nbndlw))
    allocate(jp(nlayers+1), jt(nlayers+1), jt1(nlayers+1))
    allocate(planklay(nlayers+1,nbndlw), planklev(0:nlayers+1,nbndlw))
    allocate(plankbnd(nbndlw))
    allocate(colh2o(nlayers+1), colco2(nlayers+1), colo3(nlayers+1))
    allocate(coln2o(nlayers+1), colco(nlayers+1), colch4(nlayers+1))
    allocate(colo2(nlayers+1), colbrd(nlayers+1))
    allocate(indself(nlayers+1), indfor(nlayers+1), indminor(nlayers+1))
    allocate(selffac(nlayers+1), selffrac(nlayers+1))
    allocate(forfac(nlayers+1), forfrac(nlayers+1))
    allocate(minorfrac(nlayers+1), scaleminor(nlayers+1))
    allocate(scaleminorn2(nlayers+1))
    allocate(fac00(nlayers+1), fac01(nlayers+1))
    allocate(fac10(nlayers+1), fac11(nlayers+1))
    allocate(rat_h2oco2(nlayers+1), rat_h2oco2_1(nlayers+1))
    allocate(rat_h2oo3(nlayers+1), rat_h2oo3_1(nlayers+1))
    allocate(rat_h2on2o(nlayers+1), rat_h2on2o_1(nlayers+1))
    allocate(rat_h2och4(nlayers+1), rat_h2och4_1(nlayers+1))
    allocate(rat_n2oco2(nlayers+1), rat_n2oco2_1(nlayers+1))
    allocate(rat_o3co2(nlayers+1), rat_o3co2_1(nlayers+1))
    allocate(cldfmc(ngptlw,nlayers+1), taucmc(ngptlw,nlayers+1))
    allocate(ciwpmc(ngptlw,nlayers+1), clwpmc(ngptlw,nlayers+1))
    allocate(cswpmc(ngptlw,nlayers+1))
    allocate(reicmc(nlayers+1), relqmc(nlayers+1), resnmc(nlayers+1))
    allocate(totuflux(0:nlayers+1), totdflux(0:nlayers+1))
    allocate(fnet(0:nlayers+1), htr(0:nlayers+1))
    allocate(totuclfl(0:nlayers+1), totdclfl(0:nlayers+1))
    allocate(fnetc(0:nlayers+1), htrc(0:nlayers+1))
    allocate(duflx(1,nlayers+1), ddflx(1,nlayers+1))
    allocate(duflxc(1,nlayers+1), ddflxc(1,nlayers+1))
    allocate(duflxcln(1,nlayers+1), ddflxcln(1,nlayers+1))
    allocate(dhr(1,nlayers), dhrc(1,nlayers))
    allocate(w_p8w(1,kme,1), w_p3d(1,kme,1), w_pi3d(1,kme,1))
    allocate(w_dz8w(1,kme,1), w_t3d(1,kme,1), w_t8w(1,kme,1), w_rho3d(1,kme,1))
    allocate(w_qv(1,kme,1), w_qc(1,kme,1), w_qr(1,kme,1))
    allocate(w_qi(1,kme,1), w_qs(1,kme,1), w_qg(1,kme,1))
    allocate(w_cldfra(1,kme,1), w_o33d(1,kme,1))
    allocate(w_rec(1,kme,1), w_rei(1,kme,1), w_res(1,kme,1))
    allocate(w_rthlw(1,kme,1), w_rthlwc(1,kme,1))
  end subroutine alloc_layered

  subroutine run_wrapper_check()
    ! Feed the untouched RRTMG_LWRAD the same raw column and record its
    ! outputs; Python asserts they match the transcription-driven chain.
    w_p8w = 0.; w_p3d = 0.; w_pi3d = 0.; w_dz8w = 0.; w_t3d = 0.; w_t8w = 0.
    w_rho3d = 0.; w_qv = 0.; w_qc = 0.; w_qr = 0.; w_qi = 0.; w_qs = 0.
    w_qg = 0.; w_cldfra = 0.; w_o33d = 0.; w_rec = 0.; w_rei = 0.; w_res = 0.
    do k = 1, nz
       w_p3d(1,k,1) = p3d_in(k);  w_t3d(1,k,1) = t3d_in(k)
       w_dz8w(1,k,1) = dz_in(k);  w_pi3d(1,k,1) = pi3d_in(k)
       w_rho3d(1,k,1) = rho_in(k)
       w_qv(1,k,1) = qv_in(k);    w_qc(1,k,1) = qc_in(k)
       w_qr(1,k,1) = qr_in(k);    w_qi(1,k,1) = qi_in(k)
       w_qs(1,k,1) = qs_in(k);    w_qg(1,k,1) = qg_in(k)
       w_cldfra(1,k,1) = cldfra_in(k)
       w_o33d(1,k,1) = o3_in(k)
       w_rec(1,k,1) = rec_in(k);  w_rei(1,k,1) = rei_in(k)
       w_res(1,k,1) = res_in(k)
    end do
    do k = 1, nz+1
       w_p8w(1,k,1) = p8w_in(k);  w_t8w(1,k,1) = t8w_in(k)
    end do
    w_emiss = emiss0; w_tsk = tsk0; w_xland = xland0; w_xice = xice0
    w_snow = snow0; w_xlat = xlat0
    w_glw = 0.; w_olr = 0.; w_lwcf = 0.
    w_lwupt = 0.; w_lwuptc = 0.; w_lwuptcln = 0.
    w_lwdnt = 0.; w_lwdntc = 0.; w_lwdntcln = 0.
    w_lwupb = 0.; w_lwupbc = 0.; w_lwupbcln = 0.
    w_lwdnb = 0.; w_lwdnbc = 0.; w_lwdnbcln = 0.
    w_rthlw = 0.; w_rthlwc = 0.

    call rrtmg_lwrad( rthratenlw=w_rthlw, rthratenlwc=w_rthlwc,             &
         lwupt=w_lwupt, lwuptc=w_lwuptc, lwuptcln=w_lwuptcln,               &
         lwdnt=w_lwdnt, lwdntc=w_lwdntc, lwdntcln=w_lwdntcln,               &
         lwupb=w_lwupb, lwupbc=w_lwupbc, lwupbcln=w_lwupbcln,               &
         lwdnb=w_lwdnb, lwdnbc=w_lwdnbc, lwdnbcln=w_lwdnbcln,               &
         glw=w_glw, olr=w_olr, lwcf=w_lwcf, emiss=w_emiss,                  &
         p8w=w_p8w, p3d=w_p3d, pi3d=w_pi3d, dz8w=w_dz8w, tsk=w_tsk,         &
         t3d=w_t3d, t8w=w_t8w, rho3d=w_rho3d, r=287., g=9.81,               &
         icloud=icloud, warm_rain=warm_rain, cldfra3d=w_cldfra,             &
         cldovrlp=cldovrlp, idcor=idcor, xlat=w_xlat,                       &
         is_cammgmp_used=.false.,                                           &
         xland=w_xland, xice=w_xice, snow=w_snow,                           &
         qv3d=w_qv, qc3d=w_qc, qr3d=w_qr,                                   &
         qi3d=w_qi, qs3d=w_qs, qg3d=w_qg,                                   &
         o3input=o3input, o33d=w_o33d,                                      &
         f_qv=(ifqv/=0), f_qc=(ifqc/=0), f_qr=(ifqr/=0),                    &
         f_qi=(ifqi/=0), f_qs=(ifqs/=0), f_qg=(ifqg/=0),                    &
         re_cloud=w_rec, re_ice=w_rei, re_snow=w_res,                       &
         has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs,           &
         aer_ra_feedback=0, progn=0, calc_clean_atm_diag=0,                 &
         f_qndrop=.false.,                                                  &
         yr=yr, julian=julian, ghg_input=0,                                 &
         mp_physics=mp_physics,                                             &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,              &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms_, kme=kme_,            &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts_, kte=kte_ )

    call wr_r0('wrap/glw', w_glw(1,1))
    call wr_r0('wrap/olr', w_olr(1,1))
    call wr_r0('wrap/lwcf', w_lwcf(1,1))
    call wr_r0('wrap/lwupt', w_lwupt(1,1))
    call wr_r0('wrap/lwuptc', w_lwuptc(1,1))
    call wr_r0('wrap/lwdnt', w_lwdnt(1,1))
    call wr_r0('wrap/lwdntc', w_lwdntc(1,1))
    call wr_r0('wrap/lwupb', w_lwupb(1,1))
    call wr_r0('wrap/lwupbc', w_lwupbc(1,1))
    call wr_r0('wrap/lwdnb', w_lwdnb(1,1))
    call wr_r0('wrap/lwdnbc', w_lwdnbc(1,1))
    call wr_r1('wrap/rthratenlw', w_rthlw(1,1:nz,1))
    call wr_r1('wrap/rthratenlwc', w_rthlwc(1,1:nz,1))
    call wr_r1('wrap/pi3d', pi3d_in)
  end subroutine run_wrapper_check

end program lw_extract
