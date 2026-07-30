! Per-routine fixture extractor for WRF v4.6.1 legacy RRTMG SW (option 4).
!
! For each input column this program:
!   1. calls the UNMODIFIED RRTMG_SWRAD end-to-end and records its outputs;
!   2. replicates RRTMG_SWRAD's prep verbatim (the campaign path: icloud/=0,
!      o3input=2, aer_opt=0, ghg_input=0, progn=0, WRF_CHEM=0, EM_CORE=1,
!      sf_surface_physics/=8, no chem aerosols, mp_physics not Ferrier),
!      calls mcica_subcol_sw, then the UNMODIFIED rrtmg_sw one-shot, and
!      verifies every WRF-level output is BIT-IDENTICAL to step 1 - proving
!      the replicated prep is exact;
!   3. runs the chain inatm_sw -> cldprmc_sw -> setcoef_sw -> taumol_sw ->
!      (rrtmg_sw glue) -> spcvmc_sw with the UNMODIFIED routines, dumping
!      every boundary, and verifies the chain's fluxes are BIT-IDENTICAL to
!      the one-shot rrtmg_sw call - proving the glue replication is exact;
!   4. optionally (itap=1) re-runs the spcvmc interior loop with lifted
!      glue, calling the UNMODIFIED reftra_sw/vrtqdr_sw per g-point and
!      dumping their inputs/outputs, verified bitwise against step 3's
!      spcvmc totals.
!
! Any mismatch is fatal (error stop) - a fixture is only written after the
! run that produced it has been proven equivalent to the untouched driver.
!
! Input text format (list-directed; ES17.9E2 fields round-trip FP32 exactly):
!   line 1: ncase nz
!   per case:
!     A: caseid yr julday mp_physics icloud cldovrlp idcor o3input
!        has_reqc has_reqi has_reqs sf_surface_physics itap
!     B: julian gmt xtime radt declin solcon xcoszen xlat xlong albedo
!        tsk xland xice snow obscur
!     nz lines:   p3d(Pa) t3d dz8w pi3d rho3d qv qc qr qi qs qg cldfra
!                 re_cloud(m) re_ice(m) re_snow(m) o31d
!     nz+1 lines: p8w(Pa) t8w
!
! Output: SWD1 stream binary (see sw_dumpio.py), records "cNNNNN/...".

program sw_fixture_driver
  use module_ra_rrtmg_sw, only : rrtmg_swrad, rrtmg_swinit
  use rrtmg_sw_rad,     only : rrtmg_sw, inatm_sw
  use rrtmg_sw_cldprmc, only : cldprmc_sw
  use rrtmg_sw_setcoef, only : setcoef_sw
  use rrtmg_sw_taumol,  only : taumol_sw
  use rrtmg_sw_spcvmc,  only : spcvmc_sw
  use rrtmg_sw_reftra,  only : reftra_sw
  use rrtmg_sw_vrtqdr,  only : vrtqdr_sw
  use mcica_subcol_gen_sw, only : mcica_subcol_sw
  use module_ra_rrtmg_lw, only : inirad, relcalc, reicalc, retab
  use parrrsw, only : nbndsw, ngptsw, naerec, mxmol, jpband, jpb1, jpb2
  use rrsw_wvn, only : ngc, ngs
  use rrsw_tbl, only : od_lo, bpade, exp_tbl, tblint
  implicit none

  ! ---- program plumbing ----
  integer :: ncase, nz, icase, k, ig, nb, i
  character(len=1024) :: infile, outfile
  integer :: iu, ou
  character(len=8) :: cpre

  ! ---- case header ----
  integer :: caseid, yr, julday, mp_physics, icloud_in, cldovrlp, idcor
  integer :: o3input, has_reqc, has_reqi, has_reqs, sfphys, itap
  real :: julian, gmt, xtime, radt, declin, solcon
  real :: xcoszen0, xlat0, xlong0, albedo0, tsk0, xland0, xice0, snow0, obscur0

  ! ---- WRF-level fields (ims=ime=jms=jme=1, kms=1, kme=nz+1) ----
  integer :: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme
  integer :: its,ite,jts,jte,kts,kte
  real, allocatable, dimension(:,:,:) :: p3d, p8w, t3d, t8w, dz8w, pi3d, rho3d
  real, allocatable, dimension(:,:,:) :: qv3d, qc3d, qr3d, qi3d, qs3d, qg3d
  real, allocatable, dimension(:,:,:) :: cldfra3d, o33d, re_cloud, re_ice, re_snow
  real, allocatable, dimension(:,:,:) :: qndrop3d
  real, dimension(1,1) :: xland, xice, snow, tsk, albedo, xlat, xlong
  real, dimension(1,1) :: coszr, xcoszen, obscur
  real, dimension(1,1) :: gsw, swcf
  real, dimension(1,1) :: swupt, swuptc, swuptcln, swdnt, swdntc, swdntcln
  real, dimension(1,1) :: swupb, swupbc, swupbcln, swdnb, swdnbc, swdnbcln
  real, dimension(1,1) :: swddir, swddni, swddif, swdownc, swddnic, swddirc
  real, dimension(1,1) :: swvisdir, swvisdif, swnirdir, swnirdif
  real, allocatable, dimension(:,:,:) :: rthratensw, rthratenswc
  real, allocatable, dimension(:,:,:) :: swupflx, swupflxc, swdnflx, swdnflxc
  real, dimension(:,:,:,:), pointer :: tauaer3d_sw => null()
  real, dimension(:,:,:,:), pointer :: ssaaer3d_sw => null()
  real, dimension(:,:,:,:), pointer :: asyaer3d_sw => null()

  ! reference copies (step 1)
  real, allocatable, dimension(:,:,:) :: r_rthratensw, r_rthratenswc
  real, allocatable, dimension(:,:,:) :: r_swupflx, r_swupflxc, r_swdnflx, r_swdnflxc
  real, dimension(1,1) :: r_coszr, r_gsw, r_swcf
  real, dimension(1,1) :: r_swupt, r_swuptc, r_swdnt, r_swdntc
  real, dimension(1,1) :: r_swupb, r_swupbc, r_swdnb, r_swdnbc
  real, dimension(1,1) :: r_swddir, r_swddni, r_swddif, r_swdownc, r_swddnic, r_swddirc
  real, dimension(1,1) :: r_swvisdir, r_swvisdif, r_swnirdir, r_swnirdif

  ! ---- replicated RRTMG_SWRAD prep locals (verbatim names) ----
  real, allocatable, dimension(:) :: pw1d, tw1d
  real, allocatable, dimension(:) :: tten1d, cldfra1d, dz1d, p1d, t1d
  real, allocatable, dimension(:) :: qv1d, qc1d, qr1d, qi1d, qs1d, qg1d
  real, allocatable, dimension(:) :: o31d, rho1d, qndrop1d, o3mmr
  real, allocatable, dimension(:,:) :: plev, tlev, play, tlay
  real, allocatable, dimension(:,:) :: h2ovmr, o3vmr, co2vmr, o2vmr, ch4vmr, n2ovmr
  real, allocatable, dimension(:,:) :: hgt
  real, allocatable, dimension(:,:) :: clwpth, ciwpth, cswpth, rel, rei, res
  real, allocatable, dimension(:,:) :: cldfrac, relqmcl, reicmcl, resnmcl
  real, allocatable, dimension(:,:,:) :: taucld, ssacld, asmcld, fsfcld
  real, allocatable, dimension(:,:,:) :: cldfmcl, clwpmcl, ciwpmcl, cswpmcl
  real, allocatable, dimension(:,:,:) :: taucmcl, ssacmcl, asmcmcl, fsfcmcl
  real, allocatable, dimension(:,:,:) :: tauaer, ssaaer, asmaer
  real, allocatable, dimension(:,:,:) :: ecaer
  real, allocatable, dimension(:,:) :: pdel, cicewp, cliqwp, csnowp, reliq, reice
  real, allocatable, dimension(:,:) :: recloud1d, reice1d, resnow1d
  real, dimension(1) :: tsfc, coszen, asdir, asdif, aldir, aldif
  real, dimension(1) :: landfrac, landm, snowh, icefrac
  real :: coszrs, dz, dzsum, scon, adjes, gliqwp, gicewp, gsnowp
  real :: gravmks, snow_mass_factor, corr
  real(kind=8) :: co2, n2o, ch4
  real :: o2
  integer :: dyofyr, ncol, nlay, icld, juldat, inflgsw, iceflgsw, liqflgsw
  integer :: pcols, pver, iplon, irng, permuteseed, idx_rei
  real :: lat
  real :: amdw, amdo, amdo2
  logical :: dorrsw
  real, parameter :: r_dc = 287., gcon = 9.81

  ! ---- rrtmg_sw one-shot outputs (dimension (1, nlay+1/nlay+2)) ----
  real, allocatable, dimension(:,:) :: swuflx, swdflx, swuflxc, swdflxc
  real, allocatable, dimension(:,:) :: swuflxcln, swdflxcln
  real, allocatable, dimension(:,:) :: sibvisdir, sibvisdif, sibnirdir, sibnirdif
  real, allocatable, dimension(:,:) :: swdkdir, swdkdif, swdkdirc
  real, allocatable, dimension(:,:) :: swhr, swhrc
  real :: l_zbbcddir, l_dirdflux, l_difdflux

  ! ---- chain (rrtmg_sw interior) locals, shapes as in rrtmg_sw ----
  real, allocatable :: pavel(:), tavel(:), pz(:), tz(:), pdp(:), coldry(:)
  real, allocatable :: wkl(:,:)
  real :: tbound, cossza, zepzen
  real, allocatable :: adjflux(:), solvar(:)
  integer :: inflag, iceflag, liqflag, nlayers
  real, allocatable :: cldfmc(:,:), taucmc(:,:), ssacmc(:,:), asmcmc(:,:)
  real, allocatable :: fsfcmc(:,:), ciwpmc(:,:), clwpmc(:,:), cswpmc(:,:)
  real, allocatable :: reicmc(:), relqmc(:), resnmc(:)
  real, allocatable :: taua(:,:), ssaa(:,:), asma(:,:)
  real, allocatable :: taormc(:,:)
  integer :: laytrop, layswtch, laylow
  integer, allocatable :: jp(:), jt(:), jt1(:), indself(:), indfor(:)
  real, allocatable :: colh2o(:), colco2(:), colo3(:), coln2o(:), colch4(:)
  real, allocatable :: colo2(:), colmol(:), co2mult(:)
  real, allocatable :: selffac(:), selffrac(:), forfac(:), forfrac(:)
  real, allocatable :: fac00(:), fac01(:), fac10(:), fac11(:)
  real, allocatable :: albdir(:), albdif(:)
  real, allocatable :: zcldfmc(:,:), ztaucmc(:,:), ztaormc(:,:)
  real, allocatable :: zasycmc(:,:), zomgcmc(:,:)
  real, allocatable :: ztaua(:,:), zasya(:,:), zomga(:,:)
  real, allocatable :: zbbfu(:), zbbfd(:), zbbcu(:), zbbcd(:)
  real, allocatable :: zbbfddir(:), zbbcddir(:), zuvfd(:), zuvcd(:)
  real, allocatable :: zuvfddir(:), zuvcddir(:), znifd(:), znicd(:)
  real, allocatable :: znifddir(:), znicddir(:)
  real, allocatable :: zsflxzen(:), ztaug(:,:), ztaur(:,:)
  ! chain-assembled WRF-shape flux outputs
  real, allocatable :: c_swuflx(:), c_swdflx(:), c_swuflxc(:), c_swdflxc(:)
  real, allocatable :: c_swhr(:), c_swhrc(:)

  logical :: inited
  integer :: ib

  if (command_argument_count() /= 2) then
     write(*,'(A)') 'usage: sw_fixture_driver input.txt output.swd'
     error stop 4
  end if
  call get_command_argument(1, infile)
  call get_command_argument(2, outfile)

  open(newunit=iu, file=trim(infile), status='old', action='read')
  open(newunit=ou, file=trim(outfile), status='replace', access='stream', &
       form='unformatted')
  write(ou) 'SWD1'

  read(iu,*) ncase, nz

  ids=1; ide=2; jds=1; jde=2; kds=1; kde=nz+1
  ims=1; ime=1; jms=1; jme=1; kms=1; kme=nz+1
  its=1; ite=1; jts=1; jte=1; kts=1; kte=nz

  call alloc_all()
  inited = .false.

  ! WRF driver data statements
  amdw = 1.607793; amdo = 0.603461; amdo2 = 0.905190
  o2 = 0.209488

  do icase = 1, ncase
     call read_case()
     write(cpre,'(A,I5.5,A)') 'c', caseid, '/'

     if (mp_physics==5 .or. mp_physics==15 .or. mp_physics==85) then
        write(*,'(A)') 'Ferrier mp not supported by this oracle'
        error stop 5
     end if
     if (o3input /= 2 .or. sfphys == 8) then
        write(*,'(A)') 'oracle supports o3input=2 and sf_surface_physics/=8 only'
        error stop 5
     end if

     if (.not. inited) then
        call rrtmg_swinit(.true., ids,ide,jds,jde,kds,kde, &
                          ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte)
        inited = .true.
     end if

     call dump_inputs()

     ! ---------------- step 1: untouched WRF driver ----------------
     rthratensw = 0.; rthratenswc = 0.
     coszr=0.; gsw=0.; swcf=0.
     swupt=0.; swuptc=0.; swuptcln=0.; swdnt=0.; swdntc=0.; swdntcln=0.
     swupb=0.; swupbc=0.; swupbcln=0.; swdnb=0.; swdnbc=0.; swdnbcln=0.
     swddir=0.; swddni=0.; swddif=0.; swdownc=0.; swddnic=0.; swddirc=0.
     swvisdir=0.; swvisdif=0.; swnirdir=0.; swnirdif=0.
     swupflx=0.; swupflxc=0.; swdnflx=0.; swdnflxc=0.

     call rrtmg_swrad( rthratensw=rthratensw, rthratenswc=rthratenswc,     &
          swupt=swupt, swuptc=swuptc, swuptcln=swuptcln, swdnt=swdnt,      &
          swdntc=swdntc, swdntcln=swdntcln, swupb=swupb, swupbc=swupbc,    &
          swupbcln=swupbcln, swdnb=swdnb, swdnbc=swdnbc, swdnbcln=swdnbcln,&
          swcf=swcf, gsw=gsw,                                              &
          swvisdir=swvisdir, swvisdif=swvisdif,                            &
          swnirdir=swnirdir, swnirdif=swnirdif,                            &
          xtime=xtime, gmt=gmt, xlat=xlat, xlong=xlong,                    &
          radt=radt, degrad=3.1415926/180., declin=declin,                 &
          coszr=coszr, julday=julday, solcon=solcon,                       &
          albedo=albedo, t3d=t3d, t8w=t8w, tsk=tsk,                        &
          p3d=p3d, p8w=p8w, pi3d=pi3d, rho3d=rho3d,                        &
          dz8w=dz8w, cldfra3d=cldfra3d,                                    &
          is_cammgmp_used=.false., r=r_dc, g=gcon,                         &
          re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,               &
          has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs,         &
          icloud=icloud_in, warm_rain=.false.,                             &
          cldovrlp=cldovrlp, idcor=idcor,                                  &
          xland=xland, xice=xice, snow=snow,                               &
          qv3d=qv3d, qc3d=qc3d, qr3d=qr3d,                                 &
          qi3d=qi3d, qs3d=qs3d, qg3d=qg3d,                                 &
          o3input=o3input, o33d=o33d,                                      &
          aer_opt=0, no_src=1,                                             &
          sf_surface_physics=sfphys,                                       &
          f_qv=.true., f_qc=.true., f_qr=.true.,                           &
          f_qi=.true., f_qs=.true., f_qg=.true.,                           &
          aer_ra_feedback=0, progn=0, calc_clean_atm_diag=0,               &
          qndrop3d=qndrop3d, f_qndrop=.false.,                             &
          mp_physics=mp_physics,                                           &
          ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,            &
          ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,            &
          its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte,            &
          swupflx=swupflx, swupflxc=swupflxc,                              &
          swdnflx=swdnflx, swdnflxc=swdnflxc,                              &
          tauaer3d_sw=tauaer3d_sw, ssaaer3d_sw=ssaaer3d_sw,                &
          asyaer3d_sw=asyaer3d_sw,                                         &
          swddir=swddir, swddni=swddni, swddif=swddif,                     &
          swdownc=swdownc, swddnic=swddnic, swddirc=swddirc,               &
          xcoszen=xcoszen, yr=yr, julian=julian, ghg_input=0,              &
          obscur=obscur, proceed_cmaq_sw=.false. )

     r_rthratensw = rthratensw; r_rthratenswc = rthratenswc
     r_coszr=coszr; r_gsw=gsw; r_swcf=swcf
     r_swupt=swupt; r_swuptc=swuptc; r_swdnt=swdnt; r_swdntc=swdntc
     r_swupb=swupb; r_swupbc=swupbc; r_swdnb=swdnb; r_swdnbc=swdnbc
     r_swddir=swddir; r_swddni=swddni; r_swddif=swddif
     r_swdownc=swdownc; r_swddnic=swddnic; r_swddirc=swddirc
     r_swvisdir=swvisdir; r_swvisdif=swvisdif
     r_swnirdir=swnirdir; r_swnirdif=swnirdif
     r_swupflx=swupflx; r_swupflxc=swupflxc
     r_swdnflx=swdnflx; r_swdnflxc=swdnflxc

     call dump_wrf_outputs()

     if (xcoszen0 .le. 0.0) then
        call wi0(trim(cpre)//'night', 1)
        cycle
     end if
     call wi0(trim(cpre)//'night', 0)

     ! ---------------- step 2: replicated prep + one-shot rrtmg_sw ----
     call prep_replicated()
     call dump_mcica_in_and_entry()

     swuflx=0.; swdflx=0.; swuflxc=0.; swdflxc=0.
     swuflxcln=0.; swdflxcln=0.
     sibvisdir=0.; sibvisdif=0.; sibnirdir=0.; sibnirdif=0.
     swdkdir=0.; swdkdif=0.; swdkdirc=0.; swhr=0.; swhrc=0.
     icld = cldovrlp

     call rrtmg_sw &
        (ncol    ,nlay    ,icld    , &
         play    ,plev    ,tlay    ,tlev    ,tsfc   , &
         h2ovmr , o3vmr   ,co2vmr  ,ch4vmr  ,n2ovmr ,o2vmr , &
         asdir   ,asdif   ,aldir   ,aldif   , &
         coszen  ,adjes   ,dyofyr  ,scon    , &
         inflgsw ,iceflgsw,liqflgsw,cldfmcl , &
         taucmcl ,ssacmcl ,asmcmcl ,fsfcmcl , &
         ciwpmcl ,clwpmcl ,cswpmcl ,reicmcl ,relqmcl ,resnmcl, &
         tauaer  ,ssaaer  ,asmaer  ,ecaer   , &
         swuflx  ,swdflx  ,swhr    ,swuflxc ,swdflxc ,swhrc, &
         swuflxcln, swdflxcln, 0,  &
         sibvisdir, sibvisdif, sibnirdir, sibnirdif, &
         swdkdir, swdkdif, swdkdirc, 0, &
         l_zbbcddir, l_dirdflux, l_difdflux )

     call verify_wrf_level()

     ! ---------------- step 3: chained routines with dumps ------------
     call run_chain()

     ! ---------------- step 4: reftra/vrtqdr tap ----------------------
     if (itap /= 0) call run_rt_tap()
  end do

  close(iu); close(ou)
  write(*,'(A,I0,A)') 'sw_fixture_driver: ', ncase, ' cases done'

contains

  !=================== SWD1 record writers ===================
  subroutine wname(name, dtype, dims)
    character(len=*), intent(in) :: name
    integer, intent(in) :: dtype
    integer, intent(in) :: dims(:)
    write(ou) int(len_trim(name),4)
    write(ou) trim(name)
    write(ou) int(dtype,4)
    write(ou) int(size(dims),4)
    write(ou) int(dims,4)
  end subroutine wname

  subroutine wr0(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a
    call wname(name, 0, [integer ::]); write(ou) a
  end subroutine wr0

  subroutine wi0(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a
    call wname(name, 1, [integer ::]); write(ou) int(a,4)
  end subroutine wi0

  subroutine wr1(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:)
    call wname(name, 0, shape(a)); write(ou) a
  end subroutine wr1

  subroutine wi1(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    call wname(name, 1, shape(a)); write(ou) int(a,4)
  end subroutine wi1

  subroutine wr2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:)
    call wname(name, 0, shape(a)); write(ou) a
  end subroutine wr2

  !=================== bitwise verification ===================
  subroutine chkr(name, a, b)
    character(len=*), intent(in) :: name
    real, intent(in) :: a, b
    if (transfer(a,1_4) /= transfer(b,1_4)) then
       write(*,'(A,A,ES17.9E2,A,ES17.9E2)') 'BIT MISMATCH ', name, a, ' vs ', b
       error stop 6
    end if
  end subroutine chkr

  subroutine chk1(name, a, b)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:), b(:)
    integer :: j
    do j = 1, size(a)
       if (transfer(a(j),1_4) /= transfer(b(j),1_4)) then
          write(*,'(A,A,I5,ES17.9E2,A,ES17.9E2)') 'BIT MISMATCH ', name, j, &
               a(j), ' vs ', b(j)
          error stop 6
       end if
    end do
  end subroutine chk1

  !=================== input ===================
  subroutine alloc_all()
    allocate(p3d(1,kms:kme,1), p8w(1,kms:kme,1), t3d(1,kms:kme,1))
    allocate(t8w(1,kms:kme,1), dz8w(1,kms:kme,1), pi3d(1,kms:kme,1))
    allocate(rho3d(1,kms:kme,1))
    allocate(qv3d(1,kms:kme,1), qc3d(1,kms:kme,1), qr3d(1,kms:kme,1))
    allocate(qi3d(1,kms:kme,1), qs3d(1,kms:kme,1), qg3d(1,kms:kme,1))
    allocate(cldfra3d(1,kms:kme,1), o33d(1,kms:kme,1))
    allocate(re_cloud(1,kms:kme,1), re_ice(1,kms:kme,1), re_snow(1,kms:kme,1))
    allocate(qndrop3d(1,kms:kme,1))
    allocate(rthratensw(1,kms:kme,1), rthratenswc(1,kms:kme,1))
    allocate(r_rthratensw(1,kms:kme,1), r_rthratenswc(1,kms:kme,1))
    allocate(swupflx(1,kms:kme+2,1), swupflxc(1,kms:kme+2,1))
    allocate(swdnflx(1,kms:kme+2,1), swdnflxc(1,kms:kme+2,1))
    allocate(r_swupflx(1,kms:kme+2,1), r_swupflxc(1,kms:kme+2,1))
    allocate(r_swdnflx(1,kms:kme+2,1), r_swdnflxc(1,kms:kme+2,1))

    allocate(pw1d(kts:kte+1), tw1d(kts:kte+1))
    allocate(tten1d(kts:kte), cldfra1d(kts:kte), dz1d(kts:kte))
    allocate(p1d(kts:kte), t1d(kts:kte))
    allocate(qv1d(kts:kte), qc1d(kts:kte), qr1d(kts:kte))
    allocate(qi1d(kts:kte), qs1d(kts:kte), qg1d(kts:kte))
    allocate(o31d(kts:kte+1), rho1d(kts:kte), qndrop1d(kts:kte))
    allocate(o3mmr(kts:kte+1))
    allocate(plev(1,kts:kte+2), tlev(1,kts:kte+2))
    allocate(play(1,kts:kte+1), tlay(1,kts:kte+1))
    allocate(h2ovmr(1,kts:kte+1), o3vmr(1,kts:kte+1), co2vmr(1,kts:kte+1))
    allocate(o2vmr(1,kts:kte+1), ch4vmr(1,kts:kte+1), n2ovmr(1,kts:kte+1))
    allocate(hgt(1,kts:kte+1))
    allocate(clwpth(1,kts:kte+1), ciwpth(1,kts:kte+1), cswpth(1,kts:kte+1))
    allocate(rel(1,kts:kte+1), rei(1,kts:kte+1), res(1,kts:kte+1))
    allocate(cldfrac(1,kts:kte+1), relqmcl(1,kts:kte+1))
    allocate(reicmcl(1,kts:kte+1), resnmcl(1,kts:kte+1))
    allocate(taucld(nbndsw,1,kts:kte+1), ssacld(nbndsw,1,kts:kte+1))
    allocate(asmcld(nbndsw,1,kts:kte+1), fsfcld(nbndsw,1,kts:kte+1))
    allocate(cldfmcl(ngptsw,1,kts:kte+1), clwpmcl(ngptsw,1,kts:kte+1))
    allocate(ciwpmcl(ngptsw,1,kts:kte+1), cswpmcl(ngptsw,1,kts:kte+1))
    allocate(taucmcl(ngptsw,1,kts:kte+1), ssacmcl(ngptsw,1,kts:kte+1))
    allocate(asmcmcl(ngptsw,1,kts:kte+1), fsfcmcl(ngptsw,1,kts:kte+1))
    allocate(tauaer(1,kts:kte+1,nbndsw), ssaaer(1,kts:kte+1,nbndsw))
    allocate(asmaer(1,kts:kte+1,nbndsw))
    allocate(ecaer(1,kts:kte+1,naerec))
    allocate(pdel(1,1:kte-kts+1), cicewp(1,1:kte-kts+1), cliqwp(1,1:kte-kts+1))
    allocate(csnowp(1,1:kte-kts+1), reliq(1,1:kte-kts+1), reice(1,1:kte-kts+1))
    allocate(recloud1d(1,1:kte-kts+1), reice1d(1,1:kte-kts+1))
    allocate(resnow1d(1,1:kte-kts+1))
    allocate(swuflx(1,kts:kte+2), swdflx(1,kts:kte+2))
    allocate(swuflxc(1,kts:kte+2), swdflxc(1,kts:kte+2))
    allocate(swuflxcln(1,kts:kte+2), swdflxcln(1,kts:kte+2))
    allocate(sibvisdir(1,kts:kte+2), sibvisdif(1,kts:kte+2))
    allocate(sibnirdir(1,kts:kte+2), sibnirdif(1,kts:kte+2))
    allocate(swdkdir(1,kts:kte+2), swdkdif(1,kts:kte+2), swdkdirc(1,kts:kte+2))
    allocate(swhr(1,kts:kte+1), swhrc(1,kts:kte+1))

    nlay = kte - kts + 1 + 1     ! set for chain allocations (nz+1)
    allocate(pavel(nlay+1), tavel(nlay+1), pz(0:nlay+1), tz(0:nlay+1))
    allocate(pdp(nlay+1), coldry(nlay+1), wkl(mxmol,nlay+1))
    allocate(adjflux(jpband), solvar(jpband))
    allocate(cldfmc(ngptsw,nlay+1), taucmc(ngptsw,nlay+1))
    allocate(ssacmc(ngptsw,nlay+1), asmcmc(ngptsw,nlay+1))
    allocate(fsfcmc(ngptsw,nlay+1), ciwpmc(ngptsw,nlay+1))
    allocate(clwpmc(ngptsw,nlay+1), cswpmc(ngptsw,nlay+1))
    allocate(reicmc(nlay+1), relqmc(nlay+1), resnmc(nlay+1))
    allocate(taua(nlay+1,nbndsw), ssaa(nlay+1,nbndsw), asma(nlay+1,nbndsw))
    allocate(taormc(ngptsw,nlay+1))
    allocate(jp(nlay+1), jt(nlay+1), jt1(nlay+1))
    allocate(indself(nlay+1), indfor(nlay+1))
    allocate(colh2o(nlay+1), colco2(nlay+1), colo3(nlay+1), coln2o(nlay+1))
    allocate(colch4(nlay+1), colo2(nlay+1), colmol(nlay+1), co2mult(nlay+1))
    allocate(selffac(nlay+1), selffrac(nlay+1), forfac(nlay+1), forfrac(nlay+1))
    allocate(fac00(nlay+1), fac01(nlay+1), fac10(nlay+1), fac11(nlay+1))
    allocate(albdir(nbndsw), albdif(nbndsw))
    allocate(zcldfmc(nlay+1,ngptsw), ztaucmc(nlay+1,ngptsw))
    allocate(ztaormc(nlay+1,ngptsw), zasycmc(nlay+1,ngptsw))
    allocate(zomgcmc(nlay+1,ngptsw))
    allocate(ztaua(nlay+1,nbndsw), zasya(nlay+1,nbndsw), zomga(nlay+1,nbndsw))
    allocate(zbbfu(nlay+2), zbbfd(nlay+2), zbbcu(nlay+2), zbbcd(nlay+2))
    allocate(zbbfddir(nlay+2), zbbcddir(nlay+2), zuvfd(nlay+2), zuvcd(nlay+2))
    allocate(zuvfddir(nlay+2), zuvcddir(nlay+2), znifd(nlay+2), znicd(nlay+2))
    allocate(znifddir(nlay+2), znicddir(nlay+2))
    allocate(zsflxzen(ngptsw), ztaug(nlay,ngptsw), ztaur(nlay,ngptsw))
    allocate(c_swuflx(nlay+1), c_swdflx(nlay+1))
    allocate(c_swuflxc(nlay+1), c_swdflxc(nlay+1))
    allocate(c_swhr(nlay), c_swhrc(nlay))
  end subroutine alloc_all

  subroutine read_case()
    read(iu,*) caseid, yr, julday, mp_physics, icloud_in, cldovrlp, idcor, &
               o3input, has_reqc, has_reqi, has_reqs, sfphys, itap
    read(iu,*) julian, gmt, xtime, radt, declin, solcon, xcoszen0, xlat0, &
               xlong0, albedo0, tsk0, xland0, xice0, snow0, obscur0
    p3d=0.; p8w=0.; t3d=0.; t8w=0.; dz8w=0.; pi3d=0.; rho3d=0.
    qv3d=0.; qc3d=0.; qr3d=0.; qi3d=0.; qs3d=0.; qg3d=0.
    cldfra3d=0.; o33d=0.; re_cloud=0.; re_ice=0.; re_snow=0.; qndrop3d=0.
    do k = kts, kte
       read(iu,*) p3d(1,k,1), t3d(1,k,1), dz8w(1,k,1), pi3d(1,k,1), &
                  rho3d(1,k,1), qv3d(1,k,1), qc3d(1,k,1), qr3d(1,k,1), &
                  qi3d(1,k,1), qs3d(1,k,1), qg3d(1,k,1), cldfra3d(1,k,1), &
                  re_cloud(1,k,1), re_ice(1,k,1), re_snow(1,k,1), o33d(1,k,1)
    end do
    do k = kts, kte+1
       read(iu,*) p8w(1,k,1), t8w(1,k,1)
    end do
    xlat = xlat0; xlong = xlong0; albedo = albedo0; tsk = tsk0
    xland = xland0; xice = xice0; snow = snow0
    xcoszen = xcoszen0; obscur = obscur0
  end subroutine read_case

  subroutine dump_inputs()
    call wi0(trim(cpre)//'in/yr', yr)
    call wi0(trim(cpre)//'in/julday', julday)
    call wi0(trim(cpre)//'in/mp_physics', mp_physics)
    call wi0(trim(cpre)//'in/icloud', icloud_in)
    call wi0(trim(cpre)//'in/cldovrlp', cldovrlp)
    call wi0(trim(cpre)//'in/idcor', idcor)
    call wi0(trim(cpre)//'in/o3input', o3input)
    call wi0(trim(cpre)//'in/has_reqc', has_reqc)
    call wi0(trim(cpre)//'in/has_reqi', has_reqi)
    call wi0(trim(cpre)//'in/has_reqs', has_reqs)
    call wi0(trim(cpre)//'in/sf_surface_physics', sfphys)
    call wi0(trim(cpre)//'in/itap', itap)
    call wr0(trim(cpre)//'in/julian', julian)
    call wr0(trim(cpre)//'in/gmt', gmt)
    call wr0(trim(cpre)//'in/xtime', xtime)
    call wr0(trim(cpre)//'in/radt', radt)
    call wr0(trim(cpre)//'in/declin', declin)
    call wr0(trim(cpre)//'in/solcon', solcon)
    call wr0(trim(cpre)//'in/xcoszen', xcoszen0)
    call wr0(trim(cpre)//'in/xlat', xlat0)
    call wr0(trim(cpre)//'in/xlong', xlong0)
    call wr0(trim(cpre)//'in/albedo', albedo0)
    call wr0(trim(cpre)//'in/tsk', tsk0)
    call wr0(trim(cpre)//'in/xland', xland0)
    call wr0(trim(cpre)//'in/xice', xice0)
    call wr0(trim(cpre)//'in/snow', snow0)
    call wr0(trim(cpre)//'in/obscur', obscur0)
    call wr1(trim(cpre)//'in/p3d', p3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/t3d', t3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/dz8w', dz8w(1,kts:kte,1))
    call wr1(trim(cpre)//'in/pi3d', pi3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/rho3d', rho3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/qv3d', qv3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/qc3d', qc3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/qr3d', qr3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/qi3d', qi3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/qs3d', qs3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/qg3d', qg3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/cldfra3d', cldfra3d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/re_cloud', re_cloud(1,kts:kte,1))
    call wr1(trim(cpre)//'in/re_ice', re_ice(1,kts:kte,1))
    call wr1(trim(cpre)//'in/re_snow', re_snow(1,kts:kte,1))
    call wr1(trim(cpre)//'in/o33d', o33d(1,kts:kte,1))
    call wr1(trim(cpre)//'in/p8w', p8w(1,kts:kte+1,1))
    call wr1(trim(cpre)//'in/t8w', t8w(1,kts:kte+1,1))
  end subroutine dump_inputs

  subroutine dump_wrf_outputs()
    call wr1(trim(cpre)//'wrf/rthratensw', r_rthratensw(1,kts:kte,1))
    call wr1(trim(cpre)//'wrf/rthratenswc', r_rthratenswc(1,kts:kte,1))
    call wr0(trim(cpre)//'wrf/coszr', r_coszr(1,1))
    call wr0(trim(cpre)//'wrf/gsw', r_gsw(1,1))
    call wr0(trim(cpre)//'wrf/swcf', r_swcf(1,1))
    call wr0(trim(cpre)//'wrf/swupt', r_swupt(1,1))
    call wr0(trim(cpre)//'wrf/swuptc', r_swuptc(1,1))
    call wr0(trim(cpre)//'wrf/swdnt', r_swdnt(1,1))
    call wr0(trim(cpre)//'wrf/swdntc', r_swdntc(1,1))
    call wr0(trim(cpre)//'wrf/swupb', r_swupb(1,1))
    call wr0(trim(cpre)//'wrf/swupbc', r_swupbc(1,1))
    call wr0(trim(cpre)//'wrf/swdnb', r_swdnb(1,1))
    call wr0(trim(cpre)//'wrf/swdnbc', r_swdnbc(1,1))
    call wr0(trim(cpre)//'wrf/swddir', r_swddir(1,1))
    call wr0(trim(cpre)//'wrf/swddni', r_swddni(1,1))
    call wr0(trim(cpre)//'wrf/swddif', r_swddif(1,1))
    call wr0(trim(cpre)//'wrf/swdownc', r_swdownc(1,1))
    call wr0(trim(cpre)//'wrf/swddnic', r_swddnic(1,1))
    call wr0(trim(cpre)//'wrf/swddirc', r_swddirc(1,1))
    call wr0(trim(cpre)//'wrf/swvisdir', r_swvisdir(1,1))
    call wr0(trim(cpre)//'wrf/swvisdif', r_swvisdif(1,1))
    call wr0(trim(cpre)//'wrf/swnirdir', r_swnirdir(1,1))
    call wr0(trim(cpre)//'wrf/swnirdif', r_swnirdif(1,1))
    call wr1(trim(cpre)//'wrf/swupflx', r_swupflx(1,kts:kte+2,1))
    call wr1(trim(cpre)//'wrf/swupflxc', r_swupflxc(1,kts:kte+2,1))
    call wr1(trim(cpre)//'wrf/swdnflx', r_swdnflx(1,kts:kte+2,1))
    call wr1(trim(cpre)//'wrf/swdnflxc', r_swdnflxc(1,kts:kte+2,1))
  end subroutine dump_wrf_outputs

  !=================== replicated RRTMG_SWRAD prep (verbatim) ==========
  subroutine prep_replicated()
    ! trace gases (ghg_input=0 branch, note REAL(4) exp exactly as WRF)
    co2 = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6
    ch4 = 1774.e-9
    n2o = 319.e-9

    rho1d(kts:kte) = rho3d(1,kts:kte,1)
    dorrsw = .true.
    coszrs = xcoszen(1,1)
    if (coszrs.le.0.0) dorrsw = .false.
    if (.not. dorrsw) then
       write(*,'(A)') 'prep_replicated called on night column'
       error stop 7
    end if

    do k=kts,kte+1
       pw1d(k) = p8w(1,k,1)/100.
       tw1d(k) = t8w(1,k,1)
    end do
    do k=kts,kte
       qv1d(k)=0.; qc1d(k)=0.; qr1d(k)=0.; qi1d(k)=0.; qs1d(k)=0.
       cldfra1d(k)=0.; qndrop1d(k)=0.
    end do
    do k=kts,kte
       qv1d(k)=qv3d(1,k,1)
       qv1d(k)=max(0.,qv1d(k))
    end do
    do k=kts,kte
       o31d(k)=o33d(1,k,1)
    end do
    do k=kts,kte
       tten1d(k)=0.
       t1d(k)=t3d(1,k,1)
       p1d(k)=p3d(1,k,1)/100.
       dz1d(k)=dz8w(1,k,1)
    end do

    if (icloud_in .ne. 0) then
       do k=kts,kte
          cldfra1d(k)=cldfra3d(1,k,1)
       end do
       do k=kts,kte
          qc1d(k)=qc3d(1,k,1); qc1d(k)=max(0.,qc1d(k))
       end do
       do k=kts,kte
          qr1d(k)=qr3d(1,k,1); qr1d(k)=max(0.,qr1d(k))
       end do
       ! F_QI present and true -> no MP-3 fallback
       do k=kts,kte
          qi1d(k)=qi3d(1,k,1); qi1d(k)=max(0.,qi1d(k))
       end do
       do k=kts,kte
          qs1d(k)=qs3d(1,k,1); qs1d(k)=max(0.,qs1d(k))
       end do
       do k=kts,kte
          qg1d(k)=qg3d(1,k,1); qg1d(k)=max(0.,qg1d(k))
       end do
       ! mji MP option 5 block: F_QC=.true., F_QI=.true. -> condition false
    end if

    do k=kts,kte
       qv1d(k)=amax1(qv1d(k),1.e-12)
    end do

    ncol = 1
    nlay = (kte - kts + 1) + 1
    icld = cldovrlp
    juldat = julian

    inflgsw = 2
    iceflgsw = 3
    liqflgsw = 1

    if (icloud_in .ne. 0) then
       if (has_reqc .ne. 0) then
          inflgsw = 3
          do k=kts,kte
             recloud1d(ncol,k) = max(2.5, re_cloud(1,k,1)*1.e6)
             if (recloud1d(ncol,k).le.2.5.and.cldfra3d(1,k,1).gt.0. &
                  .and. (xland(1,1)-1.5).gt.0.) then
                recloud1d(ncol,k) = 10.5
             elseif (recloud1d(ncol,k).le.2.5.and.cldfra3d(1,k,1).gt.0. &
                  .and. (xland(1,1)-1.5).lt.0.) then
                recloud1d(ncol,k) = 7.5
             endif
          end do
       else
          do k=kts,kte
             recloud1d(ncol,k) = 5.0
          end do
       end if

       if (has_reqi .ne. 0) then
          inflgsw  = 4
          iceflgsw = 4
          do k=kts,kte
             reice1d(ncol,k) = max(5., re_ice(1,k,1)*1.e6)
             if (reice1d(ncol,k).le.5..and.cldfra3d(1,k,1).gt.0.) then
                idx_rei = int(t3d(1,k,1)-179.)
                idx_rei = min(max(idx_rei,1),75)
                corr = t3d(1,k,1) - int(t3d(1,k,1))
                reice1d(ncol,k) = retab(idx_rei)*(1.-corr) + &
                                  retab(idx_rei+1)*corr
                reice1d(ncol,k) = max(reice1d(ncol,k), 5.0)
             endif
          end do
       else
          do k=kts,kte
             reice1d(ncol,k) = 10.
          end do
       end if

       if (has_reqs .ne. 0) then
          inflgsw  = 5
          iceflgsw = 5
          do k=kts,kte
             resnow1d(ncol,k) = max(10., re_snow(1,k,1)*1.e6)
          end do
       else
          do k=kts,kte
             resnow1d(ncol,k) = 10.0
          end do
       end if

       ! special case for P3 microphysics
       if (has_reqs .eq. 0 .and. has_reqi .ne. 0 .and. has_reqc .ne. 0) then
          inflgsw  = 5
          iceflgsw = 5
          do k=kts,kte
             resnow1d(ncol,k) = max(10., re_ice(1,k,1)*1.e6)
             qs1d(k)=qi3d(1,k,1)
             qi1d(k)=0.
             reice1d(ncol,k)=10.
          end do
       end if
    end if

    coszen(ncol) = coszrs
    scon = solcon*(1-obscur(1,1))
    dyofyr = 0
    adjes = 1.0

    plev(ncol,1) = pw1d(1)
    tlev(ncol,1) = tw1d(1)
    tsfc(ncol) = tsk(1,1)
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
    end do

    dzsum = 0.0
    do k = kts, kte
       dz = dz1d(k)
       hgt(ncol,k) = dzsum + 0.5*dz
       dzsum = dzsum + dz
    end do

    play(ncol,kte+1) = 0.5 * plev(ncol,kte+1)
    tlay(ncol,kte+1) = tlev(ncol,kte+1) + 0.0
    plev(ncol,kte+2) = 1.0e-5
    tlev(ncol,kte+2) = tlev(ncol,kte+1) + 0.0
    tlev(ncol,kte+2) = tlev(ncol,kte+1) + 0.0
    h2ovmr(ncol,kte+1) = h2ovmr(ncol,kte)
    co2vmr(ncol,kte+1) = co2vmr(ncol,kte)
    o2vmr(ncol,kte+1) = o2vmr(ncol,kte)
    ch4vmr(ncol,kte+1) = ch4vmr(ncol,kte)
    n2ovmr(ncol,kte+1) = n2ovmr(ncol,kte)

    hgt(ncol,kte+1) = dzsum + 0.5*dz

    call inirad (o3mmr,plev,kts,kte)

    ! o3input == 2 branch
    do k = kts, kte+1
       o3vmr(ncol,k) = o3mmr(k) * amdo
       if(k.le.kte)then
          o3vmr(ncol,k) = o31d(k)
       else
          o3vmr(ncol,k) = o31d(kte) - o3mmr(kte)*amdo + o3mmr(k)*amdo
          if(o3vmr(ncol,k) .le. 0.)o3vmr(ncol,k) = o3mmr(k)*amdo
       endif
    end do

    ! sf_surface_physics /= 8 (or ocean) branch
    asdir(ncol) = albedo(1,1)
    asdif(ncol) = albedo(1,1)
    aldir(ncol) = albedo(1,1)
    aldif(ncol) = albedo(1,1)

    if (inflgsw .gt. 0) then
       do k = kts, kte
          cldfrac(ncol,k) = cldfra1d(k)
       end do

       pcols = ncol
       pver = kte - kts + 1
       gravmks = gcon
       landfrac(ncol) = 2.-xland(1,1)
       landm(ncol) = landfrac(ncol)
       snowh(ncol) = 0.001*snow(1,1)
       icefrac(ncol) = xice(1,1)

       do k = kts, kte
          gicewp = (qi1d(k)+qs1d(k)) * pdel(ncol,k)*100.0 / gravmks * 1000.0
          gliqwp = qc1d(k) * pdel(ncol,k)*100.0 / gravmks * 1000.0
          cicewp(ncol,k) = gicewp / max(0.01,cldfrac(ncol,k))
          cliqwp(ncol,k) = gliqwp / max(0.01,cldfrac(ncol,k))
       end do

       if(iceflgsw.ge.4)then
          do k = kts, kte
             gicewp = qi1d(k) * pdel(ncol,k)*100.0 / gravmks * 1000.0
             cicewp(ncol,k) = gicewp / max(0.01,cldfrac(ncol,k))
          end do
       end if

       if(iceflgsw.eq.5)then
          do k = kts, kte
             snow_mass_factor = 0.99
             gicewp = gicewp + (qs1d(k)*(1.0-snow_mass_factor) * &
                      pdel(ncol,k)*100.0 / gravmks * 1000.0)
             if (resnow1d(ncol,k) .gt. 130.)then
                snow_mass_factor = min(snow_mass_factor, &
                     (130.0/resnow1d(ncol,k))*(130.0/resnow1d(ncol,k)))
                resnow1d(ncol,k)   = 130.0
             endif
             gsnowp = qs1d(k) * snow_mass_factor * pdel(ncol,k)*100.0 / &
                      gravmks * 1000.0
             csnowp(ncol,k) = gsnowp / max(0.01,cldfrac(ncol,k))
          end do
       end if

       ! progn present and == 0 -> relcalc
       call relcalc(ncol, pcols, pver, tlay, landfrac, landm, icefrac, &
                    reliq, snowh)
       call reicalc(ncol, pcols, pver, tlay, reice)

       if (inflgsw .ge. 3) then
          do k = kts, kte
             reliq(ncol,k) = recloud1d(ncol,k)
          end do
       endif
       ! EM_CORE==1
       if (iceflgsw .ge. 4) then
          do k = kts, kte
             reice(ncol,k) = reice1d(ncol,k)
          end do
       endif

       if (iceflgsw .eq. 3) then
          do k = kts, kte
             reice(ncol,k) = reice(ncol,k) * 1.0315
             reice(ncol,k) = min(140.0,reice(ncol,k))
          end do
       endif

       ! is_CAMMGMP_used = .false.

       do k = kts, kte
          clwpth(ncol,k) = cliqwp(ncol,k)
          ciwpth(ncol,k) = cicewp(ncol,k)
          rel(ncol,k) = reliq(ncol,k)
          rei(ncol,k) = reice(ncol,k)
       end do

       if (inflgsw .eq. 5) then
          do k = kts, kte
             cswpth(ncol,k) = csnowp(ncol,k)
             res(ncol,k) = resnow1d(ncol,k)
          end do
       else
          do k = kts, kte
             cswpth(ncol,k) = 0.0
             res(ncol,k) = 10.0
          end do
       endif

       do k = kts, kte
          do nb = 1, nbndsw
             taucld(nb,ncol,k) = 0.0
             ssacld(nb,ncol,k) = 1.0
             asmcld(nb,ncol,k) = 0.0
             fsfcld(nb,ncol,k) = 0.0
          end do
       end do
    else
       ! inflgsw == 0 path is inactive in WRF option 4 (inflgsw >= 2 always)
       write(*,'(A)') 'unexpected inflgsw == 0'
       error stop 7
    end if

    clwpth(ncol,kte+1) = 0.
    ciwpth(ncol,kte+1) = 0.
    cswpth(ncol,kte+1) = 0.
    rel(ncol,kte+1) = 10.
    rei(ncol,kte+1) = 10.
    res(ncol,kte+1) = 10.
    cldfrac(ncol,kte+1) = 0.
    do nb = 1, nbndsw
       taucld(nb,ncol,kte+1) = 0.
       ssacld(nb,ncol,kte+1) = 1.
       asmcld(nb,ncol,kte+1) = 0.
       fsfcld(nb,ncol,kte+1) = 0.
    end do

    iplon = 1
    irng = 0
    permuteseed = 1

    lat = xlat(1,1)
    call mcica_subcol_sw(iplon, ncol, nlay, icld, permuteseed, irng, play, &
         cldfrac, ciwpth, clwpth, cswpth, rei, rel, res, taucld, ssacld, &
         asmcld, fsfcld, hgt, idcor, juldat, lat, &
         cldfmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, &
         taucmcl, ssacmcl, asmcmcl, fsfcmcl)

    ! WRF_CHEM == 0, no tauaer3d
    do nb = 1, nbndsw
       do k = kts,kte+1
          tauaer(ncol,k,nb) = 0.
          ssaaer(ncol,k,nb) = 1.
          asmaer(ncol,k,nb) = 0.
       end do
    end do

    do i = 1, naerec
       do k = kts, kte+1
          ecaer(ncol,k,i) = 0.
       end do
    end do
    ! aerod not present in oracle call; aer_opt=0 leaves ecaer zero
  end subroutine prep_replicated

  subroutine dump_mcica_in_and_entry()
    call wr1(trim(cpre)//'mcin/play', play(1,:))
    call wr1(trim(cpre)//'mcin/cldfrac', cldfrac(1,:))
    call wr1(trim(cpre)//'mcin/ciwpth', ciwpth(1,:))
    call wr1(trim(cpre)//'mcin/clwpth', clwpth(1,:))
    call wr1(trim(cpre)//'mcin/cswpth', cswpth(1,:))
    call wr1(trim(cpre)//'mcin/rei', rei(1,:))
    call wr1(trim(cpre)//'mcin/rel', rel(1,:))
    call wr1(trim(cpre)//'mcin/res', res(1,:))
    call wr1(trim(cpre)//'mcin/hgt', hgt(1,:))
    call wi0(trim(cpre)//'mcin/icld', icld)
    call wi0(trim(cpre)//'mcin/idcor', idcor)
    call wi0(trim(cpre)//'mcin/juldat', juldat)
    call wr0(trim(cpre)//'mcin/lat', lat)
    call wi0(trim(cpre)//'mcin/permuteseed', permuteseed)
    call wi0(trim(cpre)//'mcin/irng', irng)

    call wr1(trim(cpre)//'entry/play', play(1,:))
    call wr1(trim(cpre)//'entry/plev', plev(1,:))
    call wr1(trim(cpre)//'entry/tlay', tlay(1,:))
    call wr1(trim(cpre)//'entry/tlev', tlev(1,:))
    call wr0(trim(cpre)//'entry/tsfc', tsfc(1))
    call wr1(trim(cpre)//'entry/h2ovmr', h2ovmr(1,:))
    call wr1(trim(cpre)//'entry/o3vmr', o3vmr(1,:))
    call wr1(trim(cpre)//'entry/co2vmr', co2vmr(1,:))
    call wr1(trim(cpre)//'entry/ch4vmr', ch4vmr(1,:))
    call wr1(trim(cpre)//'entry/n2ovmr', n2ovmr(1,:))
    call wr1(trim(cpre)//'entry/o2vmr', o2vmr(1,:))
    call wr0(trim(cpre)//'entry/asdir', asdir(1))
    call wr0(trim(cpre)//'entry/asdif', asdif(1))
    call wr0(trim(cpre)//'entry/aldir', aldir(1))
    call wr0(trim(cpre)//'entry/aldif', aldif(1))
    call wr0(trim(cpre)//'entry/coszen', coszen(1))
    call wr0(trim(cpre)//'entry/adjes', adjes)
    call wr0(trim(cpre)//'entry/scon', scon)
    call wi0(trim(cpre)//'entry/dyofyr', dyofyr)
    call wi0(trim(cpre)//'entry/icld', icld)
    call wi0(trim(cpre)//'entry/inflgsw', inflgsw)
    call wi0(trim(cpre)//'entry/iceflgsw', iceflgsw)
    call wi0(trim(cpre)//'entry/liqflgsw', liqflgsw)
    call wi0(trim(cpre)//'entry/nlay', nlay)
    call wr2(trim(cpre)//'entry/cldfmcl', cldfmcl(:,1,:))
    call wr2(trim(cpre)//'entry/taucmcl', taucmcl(:,1,:))
    call wr2(trim(cpre)//'entry/ssacmcl', ssacmcl(:,1,:))
    call wr2(trim(cpre)//'entry/asmcmcl', asmcmcl(:,1,:))
    call wr2(trim(cpre)//'entry/fsfcmcl', fsfcmcl(:,1,:))
    call wr2(trim(cpre)//'entry/ciwpmcl', ciwpmcl(:,1,:))
    call wr2(trim(cpre)//'entry/clwpmcl', clwpmcl(:,1,:))
    call wr2(trim(cpre)//'entry/cswpmcl', cswpmcl(:,1,:))
    call wr1(trim(cpre)//'entry/reicmcl', reicmcl(1,:))
    call wr1(trim(cpre)//'entry/relqmcl', relqmcl(1,:))
    call wr1(trim(cpre)//'entry/resnmcl', resnmcl(1,:))
  end subroutine dump_mcica_in_and_entry

  !=================== verify one-shot vs WRF driver =================
  subroutine verify_wrf_level()
    real :: v_gsw, v_swcf, v_tten(kts:kte), v_ttenc(kts:kte)
    integer :: kk
    ! replicate RRTMG_SWRAD post-processing (verbatim)
    v_gsw = swdflx(1,1) - swuflx(1,1)
    v_swcf = (swdflx(1,kte+2) - swuflx(1,kte+2)) - &
             (swdflxc(1,kte+2) - swuflxc(1,kte+2))
    call chkr('gsw', v_gsw, r_gsw(1,1))
    call chkr('swcf', v_swcf, r_swcf(1,1))
    call chkr('swupt', swuflx(1,kte+2), r_swupt(1,1))
    call chkr('swuptc', swuflxc(1,kte+2), r_swuptc(1,1))
    call chkr('swdnt', swdflx(1,kte+2), r_swdnt(1,1))
    call chkr('swdntc', swdflxc(1,kte+2), r_swdntc(1,1))
    call chkr('swupb', swuflx(1,1), r_swupb(1,1))
    call chkr('swupbc', swuflxc(1,1), r_swupbc(1,1))
    call chkr('swdnb', swdflx(1,1), r_swdnb(1,1))
    call chkr('swdnbc', swdflxc(1,1), r_swdnbc(1,1))
    call chkr('swvisdir', sibvisdir(1,1), r_swvisdir(1,1))
    call chkr('swvisdif', sibvisdif(1,1), r_swvisdif(1,1))
    call chkr('swnirdir', sibnirdir(1,1), r_swnirdir(1,1))
    call chkr('swnirdif', sibnirdif(1,1), r_swnirdif(1,1))
    call chkr('swddir', swdkdir(1,1), r_swddir(1,1))
    call chkr('swddni', swdkdir(1,1)/coszrs, r_swddni(1,1))
    call chkr('swddif', swdkdif(1,1), r_swddif(1,1))
    call chkr('swdownc', swdflxc(1,1), r_swdownc(1,1))
    call chkr('swddirc', swdkdirc(1,1), r_swddirc(1,1))
    call chkr('swddnic', swdkdirc(1,1)/coszrs, r_swddnic(1,1))
    call chk1('swupflx', swuflx(1,kts:kte+2), r_swupflx(1,kts:kte+2,1))
    call chk1('swupflxc', swuflxc(1,kts:kte+2), r_swupflxc(1,kts:kte+2,1))
    call chk1('swdnflx', swdflx(1,kts:kte+2), r_swdnflx(1,kts:kte+2,1))
    call chk1('swdnflxc', swdflxc(1,kts:kte+2), r_swdnflxc(1,kts:kte+2,1))
    do kk=kts,kte
       v_tten(kk) = (swhr(1,kk)/86400.)/pi3d(1,kk,1)
       v_ttenc(kk) = (swhrc(1,kk)/86400.)/pi3d(1,kk,1)
    end do
    call chk1('rthratensw', v_tten, r_rthratensw(1,kts:kte,1))
    call chk1('rthratenswc', v_ttenc, r_rthratenswc(1,kts:kte,1))
  end subroutine verify_wrf_level

  !=================== step 3: chain with dumps ======================
  subroutine run_chain()
    integer :: iaer, lay
    real :: zdpgcp
    real :: swnflx(nlay+2), swnflxc(nlay+2)
    zepzen = 1.e-10
    iaer = 10   ! aer_opt == 0

    call inatm_sw (1, nlay, icld, iaer, &
         play, plev, tlay, tlev, tsfc, h2ovmr, &
         o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, &
         adjes, dyofyr, scon, inflgsw, iceflgsw, liqflgsw, &
         cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl, ciwpmcl, clwpmcl, &
         cswpmcl, reicmcl, relqmcl, resnmcl, tauaer, ssaaer, asmaer, &
         nlayers, pavel, pz, pdp, tavel, tz, tbound, coldry, wkl, &
         adjflux, solvar, inflag, iceflag, liqflag, cldfmc, taucmc, &
         ssacmc, asmcmc, fsfcmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, &
         resnmc, taua, ssaa, asma)

    call wi0(trim(cpre)//'inatm/nlayers', nlayers)
    call wr1(trim(cpre)//'inatm/pavel', pavel(1:nlayers))
    call wr1(trim(cpre)//'inatm/tavel', tavel(1:nlayers))
    call wr1(trim(cpre)//'inatm/pz', pz(0:nlayers))
    call wr1(trim(cpre)//'inatm/tz', tz(0:nlayers))
    call wr1(trim(cpre)//'inatm/pdp', pdp(1:nlayers))
    call wr0(trim(cpre)//'inatm/tbound', tbound)
    call wr1(trim(cpre)//'inatm/coldry', coldry(1:nlayers))
    call wr2(trim(cpre)//'inatm/wkl', wkl(:,1:nlayers))
    call wr1(trim(cpre)//'inatm/adjflux', adjflux(jpb1:jpb2))
    call wr1(trim(cpre)//'inatm/solvar', solvar(jpb1:jpb2))
    call wi0(trim(cpre)//'inatm/inflag', inflag)
    call wi0(trim(cpre)//'inatm/iceflag', iceflag)
    call wi0(trim(cpre)//'inatm/liqflag', liqflag)
    call wr2(trim(cpre)//'inatm/cldfmc', cldfmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/taucmc', taucmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/ssacmc', ssacmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/asmcmc', asmcmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/fsfcmc', fsfcmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/ciwpmc', ciwpmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/clwpmc', clwpmc(:,1:nlayers))
    call wr2(trim(cpre)//'inatm/cswpmc', cswpmc(:,1:nlayers))
    call wr1(trim(cpre)//'inatm/reicmc', reicmc(1:nlayers))
    call wr1(trim(cpre)//'inatm/relqmc', relqmc(1:nlayers))
    call wr1(trim(cpre)//'inatm/resnmc', resnmc(1:nlayers))
    call wr2(trim(cpre)//'inatm/taua', taua(1:nlayers,:))
    call wr2(trim(cpre)//'inatm/ssaa', ssaa(1:nlayers,:))
    call wr2(trim(cpre)//'inatm/asma', asma(1:nlayers,:))

    call cldprmc_sw(nlayers, inflag, iceflag, liqflag, cldfmc, &
                    ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, &
                    taormc, taucmc, ssacmc, asmcmc, fsfcmc)

    call wr2(trim(cpre)//'cldprmc/taormc', taormc(:,1:nlayers))
    call wr2(trim(cpre)//'cldprmc/taucmc', taucmc(:,1:nlayers))
    call wr2(trim(cpre)//'cldprmc/ssacmc', ssacmc(:,1:nlayers))
    call wr2(trim(cpre)//'cldprmc/asmcmc', asmcmc(:,1:nlayers))

    call setcoef_sw(nlayers, pavel, tavel, pz, tz, tbound, coldry, wkl, &
                    laytrop, layswtch, laylow, jp, jt, jt1, &
                    co2mult, colch4, colco2, colh2o, colmol, coln2o, &
                    colo2, colo3, fac00, fac01, fac10, fac11, &
                    selffac, selffrac, indself, forfac, forfrac, indfor)

    call wi0(trim(cpre)//'setcoef/laytrop', laytrop)
    call wi0(trim(cpre)//'setcoef/layswtch', layswtch)
    call wi0(trim(cpre)//'setcoef/laylow', laylow)
    call wi1(trim(cpre)//'setcoef/jp', jp(1:nlayers))
    call wi1(trim(cpre)//'setcoef/jt', jt(1:nlayers))
    call wi1(trim(cpre)//'setcoef/jt1', jt1(1:nlayers))
    call wi1(trim(cpre)//'setcoef/indself', indself(1:nlayers))
    call wi1(trim(cpre)//'setcoef/indfor', indfor(1:nlayers))
    call wr1(trim(cpre)//'setcoef/colh2o', colh2o(1:nlayers))
    call wr1(trim(cpre)//'setcoef/colco2', colco2(1:nlayers))
    call wr1(trim(cpre)//'setcoef/colo3', colo3(1:nlayers))
    call wr1(trim(cpre)//'setcoef/coln2o', coln2o(1:nlayers))
    call wr1(trim(cpre)//'setcoef/colch4', colch4(1:nlayers))
    call wr1(trim(cpre)//'setcoef/colo2', colo2(1:nlayers))
    call wr1(trim(cpre)//'setcoef/colmol', colmol(1:nlayers))
    call wr1(trim(cpre)//'setcoef/co2mult', co2mult(1:nlayers))
    call wr1(trim(cpre)//'setcoef/selffac', selffac(1:nlayers))
    call wr1(trim(cpre)//'setcoef/selffrac', selffrac(1:nlayers))
    call wr1(trim(cpre)//'setcoef/forfac', forfac(1:nlayers))
    call wr1(trim(cpre)//'setcoef/forfrac', forfrac(1:nlayers))
    call wr1(trim(cpre)//'setcoef/fac00', fac00(1:nlayers))
    call wr1(trim(cpre)//'setcoef/fac01', fac01(1:nlayers))
    call wr1(trim(cpre)//'setcoef/fac10', fac10(1:nlayers))
    call wr1(trim(cpre)//'setcoef/fac11', fac11(1:nlayers))

    ! taumol fixture (same call spcvmc makes internally)
    call taumol_sw(nlayers, &
                   colh2o, colco2, colch4, colo2, colo3, colmol, &
                   laytrop, jp, jt, jt1, &
                   fac00, fac01, fac10, fac11, &
                   selffac, selffrac, indself, forfac, forfrac, indfor, &
                   zsflxzen, ztaug, ztaur)
    call wr1(trim(cpre)//'taumol/sfluxzen', zsflxzen)
    call wr2(trim(cpre)//'taumol/taug', ztaug)
    call wr2(trim(cpre)//'taumol/taur', ztaur)

    ! rrtmg_sw glue (verbatim)
    cossza = coszen(1)
    if (cossza .le. zepzen) cossza = zepzen

    do ib=1,9
       albdir(ib) = aldir(1)
       albdif(ib) = aldif(1)
    enddo
    albdir(nbndsw) = aldir(1)
    albdif(nbndsw) = aldif(1)
    do ib=10,13
       albdir(ib) = asdir(1)
       albdif(ib) = asdif(1)
    enddo

    if (icld.eq.0) then
       zcldfmc(:,:) = 0.
       ztaucmc(:,:) = 0.
       ztaormc(:,:) = 0.
       zasycmc(:,:) = 0.
       zomgcmc(:,:) = 1.
    elseif (icld.ge.1) then
       do i=1,nlayers
          do ig=1,ngptsw
             zcldfmc(i,ig) = cldfmc(ig,i)
             ztaucmc(i,ig) = taucmc(ig,i)
             ztaormc(i,ig) = taormc(ig,i)
             zasycmc(i,ig) = asmcmc(ig,i)
             zomgcmc(i,ig) = ssacmc(ig,i)
          enddo
       enddo
    endif

    ! iaer == 10
    do ib = 1 ,nbndsw
       do i = 1 ,nlayers
          ztaua(i,ib) = taua(i,ib)
          zasya(i,ib) = asma(i,ib)
          zomga(i,ib) = ssaa(i,ib)
       enddo
    enddo

    call wr0(trim(cpre)//'spcin/cossza', cossza)
    call wr1(trim(cpre)//'spcin/albdir', albdir)
    call wr1(trim(cpre)//'spcin/albdif', albdif)
    call wr2(trim(cpre)//'spcin/zcldfmc', zcldfmc(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/ztaucmc', ztaucmc(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/ztaormc', ztaormc(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/zasycmc', zasycmc(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/zomgcmc', zomgcmc(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/ztaua', ztaua(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/zasya', zasya(1:nlayers,:))
    call wr2(trim(cpre)//'spcin/zomga', zomga(1:nlayers,:))

    do i=1,nlayers+1
       zbbcu(i) = 0.
       zbbcd(i) = 0.
       zbbfu(i) = 0.
       zbbfd(i) = 0.
       zbbcddir(i) = 0.
       zbbfddir(i) = 0.
       zuvcd(i) = 0.
       zuvfd(i) = 0.
       zuvcddir(i) = 0.
       zuvfddir(i) = 0.
       znicd(i) = 0.
       znifd(i) = 0.
       znicddir(i) = 0.
       znifddir(i) = 0.
    enddo

    call spcvmc_sw &
        (nlayers, jpb1, jpb2, 1, 0, &
         pavel, tavel, pz, tz, tbound, albdif, albdir, &
         zcldfmc, ztaucmc, zasycmc, zomgcmc, ztaormc, &
         ztaua, zasya, zomga, cossza, coldry, wkl, adjflux, &
         laytrop, layswtch, laylow, jp, jt, jt1, &
         co2mult, colch4, colco2, colh2o, colmol, coln2o, colo2, colo3, &
         fac00, fac01, fac10, fac11, &
         selffac, selffrac, indself, forfac, forfrac, indfor, &
         zbbfd, zbbfu, zbbcd, zbbcu, zuvfd, zuvcd, znifd, znicd, &
         zbbfddir, zbbcddir, zuvfddir, zuvcddir, znifddir, znicddir)

    call wr1(trim(cpre)//'spcout/zbbfd', zbbfd(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zbbfu', zbbfu(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zbbcd', zbbcd(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zbbcu', zbbcu(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zuvfd', zuvfd(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zuvcd', zuvcd(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/znifd', znifd(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/znicd', znicd(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zbbfddir', zbbfddir(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zbbcddir', zbbcddir(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zuvfddir', zuvfddir(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/zuvcddir', zuvcddir(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/znifddir', znifddir(1:nlayers+1))
    call wr1(trim(cpre)//'spcout/znicddir', znicddir(1:nlayers+1))

    ! assemble WRF-level fluxes exactly as rrtmg_sw does and verify the
    ! chain against the one-shot rrtmg_sw outputs
    do i = 1, nlayers+1
       c_swuflxc(i) = zbbcu(i)
       c_swdflxc(i) = zbbcd(i)
       c_swuflx(i) = zbbfu(i)
       c_swdflx(i) = zbbfd(i)
    end do
    do i = 1, nlayers+1
       swnflxc(i) = c_swdflxc(i) - c_swuflxc(i)
       swnflx(i) = c_swdflx(i) - c_swuflx(i)
    end do
    do i = 1, nlayers
       zdpgcp = heatfac_val() / pdp(i)
       c_swhrc(i) = (swnflxc(i+1) - swnflxc(i)) * zdpgcp
       c_swhr(i) = (swnflx(i+1) - swnflx(i)) * zdpgcp
    end do
    c_swhrc(nlayers) = 0.
    c_swhr(nlayers) = 0.

    call chk1('chain swuflx', c_swuflx, swuflx(1,1:nlayers+1))
    call chk1('chain swdflx', c_swdflx, swdflx(1,1:nlayers+1))
    call chk1('chain swuflxc', c_swuflxc, swuflxc(1,1:nlayers+1))
    call chk1('chain swdflxc', c_swdflxc, swdflxc(1,1:nlayers+1))
    call chk1('chain swhr', c_swhr, swhr(1,1:nlayers))
    call chk1('chain swhrc', c_swhrc, swhrc(1,1:nlayers))
    call chk1('chain sibvisdir', zuvfddir(1:nlayers+1), &
              rev_lev(sibvisdir(1,1:nlayers+1)))
  end subroutine run_chain

  ! zuvfddir is indexed bottom-to-top like sibvisdir; helper aligns the
  ! rrtmg_sw output (dirdnuv transferred at same index) - identity here.
  function rev_lev(a) result(b)
    real, intent(in) :: a(:)
    real :: b(size(a))
    b = a
  end function rev_lev

  real function heatfac_val()
    use rrsw_con, only : heatfac
    heatfac_val = heatfac
  end function heatfac_val

  !=================== step 4: reftra/vrtqdr tap =====================
  subroutine run_rt_tap()
    ! Lifted spcvmc_sw interior, calling the UNMODIFIED reftra_sw and
    ! vrtqdr_sw, dumping their per-g-point inputs and outputs.  The
    ! accumulated band fluxes are verified bitwise against the real
    ! spcvmc_sw outputs from run_chain.
    logical :: lrtchkclr(nlayers), lrtchkcld(nlayers)
    integer :: klev, ib1, ib2, ibm, igt, ikl, iw, jb, jg, jk, itind
    real :: tblind, ze1, zclear, zcloud
    real :: zdbt(nlayers+1), zdbt_nodel(nlayers+1)
    real :: zgcc(nlayers), zgco(nlayers)
    real :: zomcc(nlayers), zomco(nlayers)
    real :: zrdnd(nlayers+1), zrdndc(nlayers+1)
    real :: zref(nlayers+1), zrefc(nlayers+1), zrefo(nlayers+1)
    real :: zrefd(nlayers+1), zrefdc(nlayers+1), zrefdo(nlayers+1)
    real :: zrup(nlayers+1), zrupd(nlayers+1)
    real :: zrupc(nlayers+1), zrupdc(nlayers+1)
    real :: ztauc(nlayers), ztauo(nlayers)
    real :: ztdbt(nlayers+1)
    real :: ztra(nlayers+1), ztrac(nlayers+1), ztrao(nlayers+1)
    real :: ztrad(nlayers+1), ztradc(nlayers+1), ztrado(nlayers+1)
    real :: zdbtc(nlayers+1), ztdbtc(nlayers+1)
    real :: zincflx(ngptsw), zdbtc_nodel(nlayers+1)
    real :: ztdbt_nodel(nlayers+1), ztdbtc_nodel(nlayers+1)
    real :: zdbtmc, zdbtmo, zf, zwf, tauorig, repclc
    real :: zcd(nlayers+1,ngptsw), zcu(nlayers+1,ngptsw)
    real :: zfd(nlayers+1,ngptsw), zfu(nlayers+1,ngptsw)
    real :: a_pbbfd(nlayers+1), a_pbbfu(nlayers+1)
    real :: a_pbbcd(nlayers+1), a_pbbcu(nlayers+1)
    real :: a_pbbfddir(nlayers+1), a_pbbcddir(nlayers+1)
    real :: a_puvfd(nlayers+1), a_puvcd(nlayers+1)
    real :: a_puvfddir(nlayers+1), a_puvcddir(nlayers+1)
    real :: a_pnifd(nlayers+1), a_pnicd(nlayers+1)
    real :: a_pnifddir(nlayers+1), a_pnicddir(nlayers+1)
    ! tap accumulation arrays (layers x ngptsw)
    real :: t_ztauc(nlayers,ngptsw), t_zomcc(nlayers,ngptsw)
    real :: t_zgcc(nlayers,ngptsw), t_ztauo(nlayers,ngptsw)
    real :: t_zomco(nlayers,ngptsw), t_zgco(nlayers,ngptsw)
    integer :: t_lrtcld(nlayers,ngptsw)
    real :: t_zrefc(nlayers,ngptsw), t_zrefdc(nlayers,ngptsw)
    real :: t_ztrac(nlayers,ngptsw), t_ztradc(nlayers,ngptsw)
    real :: t_zrefo(nlayers,ngptsw), t_zrefdo(nlayers,ngptsw)
    real :: t_ztrao(nlayers,ngptsw), t_ztrado(nlayers,ngptsw)
    real :: t_zdbtc(nlayers+1,ngptsw), t_zdbt(nlayers+1,ngptsw)
    real :: t_ztdbtc(nlayers+1,ngptsw), t_ztdbt(nlayers+1,ngptsw)
    real :: t_zdbtcnd(nlayers+1,ngptsw), t_zdbtnd(nlayers+1,ngptsw)
    real :: t_ztdbtcnd(nlayers+1,ngptsw), t_ztdbtnd(nlayers+1,ngptsw)
    real :: t_zrdndc(nlayers+1,ngptsw), t_zrupc(nlayers+1,ngptsw)
    real :: t_zrupdc(nlayers+1,ngptsw)
    real :: t_zrdnd(nlayers+1,ngptsw), t_zrup(nlayers+1,ngptsw)
    real :: t_zrupd(nlayers+1,ngptsw)

    klev = nlayers
    ib1 = jpb1; ib2 = jpb2
    iw = 0
    repclc = 1.e-12
    a_pbbfd=0.; a_pbbfu=0.; a_pbbcd=0.; a_pbbcu=0.
    a_pbbfddir=0.; a_pbbcddir=0.
    a_puvfd=0.; a_puvcd=0.; a_puvfddir=0.; a_puvcddir=0.
    a_pnifd=0.; a_pnicd=0.; a_pnifddir=0.; a_pnicddir=0.

    do jb = ib1, ib2
       ibm = jb-15
       igt = ngc(ibm)
       do jg = 1, igt
          iw = iw+1
          zincflx(iw) = adjflux(jb) * zsflxzen(iw) * cossza

          ztdbtc(1)=1.0
          ztdbtc_nodel(1)=1.0
          zdbtc(klev+1) =0.0
          ztrac(klev+1) =0.0
          ztradc(klev+1)=0.0
          zrefc(klev+1) =albdir(ibm)
          zrefdc(klev+1)=albdif(ibm)
          zrupc(klev+1) =albdir(ibm)
          zrupdc(klev+1)=albdif(ibm)
          ztdbt(1)=1.0
          ztdbt_nodel(1)=1.0
          zdbt(klev+1) =0.0
          ztra(klev+1) =0.0
          ztrad(klev+1)=0.0
          zref(klev+1) =albdir(ibm)
          zrefd(klev+1)=albdif(ibm)
          zrup(klev+1) =albdir(ibm)
          zrupd(klev+1)=albdif(ibm)

          do jk=1,klev
             ikl=klev+1-jk
             lrtchkclr(jk)=.true.
             lrtchkcld(jk)=.false.
             lrtchkcld(jk)=(zcldfmc(ikl,iw) > repclc)

             ztauc(jk) = ztaur(ikl,iw) + ztaug(ikl,iw) + ztaua(ikl,ibm)
             zomcc(jk) = ztaur(ikl,iw) * 1.0 + ztaua(ikl,ibm) * zomga(ikl,ibm)
             zgcc(jk) = zasya(ikl,ibm) * zomga(ikl,ibm) * ztaua(ikl,ibm) / zomcc(jk)
             zomcc(jk) = zomcc(jk) / ztauc(jk)

             zclear = 1.0 - zcldfmc(ikl,iw)
             zcloud = zcldfmc(ikl,iw)

             ze1 = ztauc(jk) / cossza
             if (ze1 .le. od_lo) then
                zdbtmc = 1. - ze1 + 0.5 * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5
                zdbtmc = exp_tbl(itind)
             endif

             zdbtc_nodel(jk) = zdbtmc
             ztdbtc_nodel(jk+1) = zdbtc_nodel(jk) * ztdbtc_nodel(jk)

             tauorig = ztauc(jk) + ztaormc(ikl,iw)
             ze1 = tauorig / cossza
             if (ze1 .le. od_lo) then
                zdbtmo = 1. - ze1 + 0.5 * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5
                zdbtmo = exp_tbl(itind)
             endif

             zdbt_nodel(jk) = zclear*zdbtmc + zcloud*zdbtmo
             ztdbt_nodel(jk+1) = zdbt_nodel(jk) * ztdbt_nodel(jk)
          enddo

          do jk=1, klev
             zf = zgcc(jk) * zgcc(jk)
             zwf = zomcc(jk) * zf
             ztauc(jk) = (1.0 - zwf) * ztauc(jk)
             zomcc(jk) = (zomcc(jk) - zwf) / (1.0 - zwf)
             zgcc (jk) = (zgcc(jk) - zf) / (1.0 - zf)
          enddo

          ! icpr == 1
          do jk=1,klev
             ikl=klev+1-jk
             ztauo(jk) = ztauc(jk) + ztaucmc(ikl,iw)
             zomco(jk) = ztauc(jk) * zomcc(jk) + ztaucmc(ikl,iw) * zomgcmc(ikl,iw)
             zgco (jk) = (ztaucmc(ikl,iw) * zomgcmc(ikl,iw) * zasycmc(ikl,iw) + &
                         ztauc(jk) * zomcc(jk) * zgcc(jk)) / zomco(jk)
             zomco(jk) = zomco(jk) / ztauo(jk)
          enddo

          call reftra_sw (klev, lrtchkclr, zgcc, cossza, ztauc, zomcc, &
                          zrefc, zrefdc, ztrac, ztradc)
          call reftra_sw (klev, lrtchkcld, zgco, cossza, ztauo, zomco, &
                          zrefo, zrefdo, ztrao, ztrado)

          do jk=1,klev
             ikl = klev+1-jk
             zclear = 1.0 - zcldfmc(ikl,iw)
             zcloud = zcldfmc(ikl,iw)
             zref(jk) = zclear*zrefc(jk) + zcloud*zrefo(jk)
             zrefd(jk)= zclear*zrefdc(jk) + zcloud*zrefdo(jk)
             ztra(jk) = zclear*ztrac(jk) + zcloud*ztrao(jk)
             ztrad(jk)= zclear*ztradc(jk) + zcloud*ztrado(jk)

             ze1 = ztauc(jk) / cossza
             if (ze1 .le. od_lo) then
                zdbtmc = 1. - ze1 + 0.5 * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5
                zdbtmc = exp_tbl(itind)
             endif
             zdbtc(jk) = zdbtmc
             ztdbtc(jk+1) = zdbtc(jk)*ztdbtc(jk)

             ze1 = ztauo(jk) / cossza
             if (ze1 .le. od_lo) then
                zdbtmo = 1. - ze1 + 0.5 * ze1 * ze1
             else
                tblind = ze1 / (bpade + ze1)
                itind = tblint * tblind + 0.5
                zdbtmo = exp_tbl(itind)
             endif
             zdbt(jk) = zclear*zdbtmc + zcloud*zdbtmo
             ztdbt(jk+1) = zdbt(jk)*ztdbt(jk)
          enddo

          call vrtqdr_sw(klev, iw, zrefc, zrefdc, ztrac, ztradc, &
                         zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, zcd, zcu)
          call vrtqdr_sw(klev, iw, zref, zrefd, ztra, ztrad, &
                         zdbt, zrdnd, zrup, zrupd, ztdbt, zfd, zfu)

          do jk=1,klev+1
             ikl=klev+2-jk
             a_pbbfu(ikl) = a_pbbfu(ikl) + zincflx(iw)*zfu(jk,iw)
             a_pbbfd(ikl) = a_pbbfd(ikl) + zincflx(iw)*zfd(jk,iw)
             a_pbbcu(ikl) = a_pbbcu(ikl) + zincflx(iw)*zcu(jk,iw)
             a_pbbcd(ikl) = a_pbbcd(ikl) + zincflx(iw)*zcd(jk,iw)
             a_pbbfddir(ikl) = a_pbbfddir(ikl) + zincflx(iw)*ztdbt_nodel(jk)
             a_pbbcddir(ikl) = a_pbbcddir(ikl) + zincflx(iw)*ztdbtc_nodel(jk)
          enddo
          if (ibm >= 10 .and. ibm <= 13) then
             do jk=1,klev+1
                ikl=klev+2-jk
                a_puvcd(ikl) = a_puvcd(ikl) + zincflx(iw)*zcd(jk,iw)
                a_puvfd(ikl) = a_puvfd(ikl) + zincflx(iw)*zfd(jk,iw)
                a_puvcddir(ikl) = a_puvcddir(ikl) + zincflx(iw)*ztdbtc_nodel(jk)
                a_puvfddir(ikl) = a_puvfddir(ikl) + zincflx(iw)*ztdbt_nodel(jk)
             enddo
          else if (ibm == 14 .or. ibm <= 9) then
             do jk=1,klev+1
                ikl=klev+2-jk
                a_pnicd(ikl) = a_pnicd(ikl) + zincflx(iw)*zcd(jk,iw)
                a_pnifd(ikl) = a_pnifd(ikl) + zincflx(iw)*zfd(jk,iw)
                a_pnicddir(ikl) = a_pnicddir(ikl) + zincflx(iw)*ztdbtc_nodel(jk)
                a_pnifddir(ikl) = a_pnifddir(ikl) + zincflx(iw)*ztdbt_nodel(jk)
             enddo
          endif

          ! record tap
          do jk=1,klev
             t_ztauc(jk,iw)=ztauc(jk); t_zomcc(jk,iw)=zomcc(jk)
             t_zgcc(jk,iw)=zgcc(jk);   t_ztauo(jk,iw)=ztauo(jk)
             t_zomco(jk,iw)=zomco(jk); t_zgco(jk,iw)=zgco(jk)
             t_lrtcld(jk,iw)=merge(1,0,lrtchkcld(jk))
             t_zrefc(jk,iw)=zrefc(jk); t_zrefdc(jk,iw)=zrefdc(jk)
             t_ztrac(jk,iw)=ztrac(jk); t_ztradc(jk,iw)=ztradc(jk)
             t_zrefo(jk,iw)=zrefo(jk); t_zrefdo(jk,iw)=zrefdo(jk)
             t_ztrao(jk,iw)=ztrao(jk); t_ztrado(jk,iw)=ztrado(jk)
          enddo
          do jk=1,klev+1
             t_zdbtc(jk,iw)=zdbtc(jk); t_zdbt(jk,iw)=zdbt(jk)
             t_ztdbtc(jk,iw)=ztdbtc(jk); t_ztdbt(jk,iw)=ztdbt(jk)
             t_zdbtcnd(jk,iw)=zdbtc_nodel(jk); t_zdbtnd(jk,iw)=zdbt_nodel(jk)
             t_ztdbtcnd(jk,iw)=ztdbtc_nodel(jk); t_ztdbtnd(jk,iw)=ztdbt_nodel(jk)
             t_zrdndc(jk,iw)=zrdndc(jk); t_zrupc(jk,iw)=zrupc(jk)
             t_zrupdc(jk,iw)=zrupdc(jk)
             t_zrdnd(jk,iw)=zrdnd(jk); t_zrup(jk,iw)=zrup(jk)
             t_zrupd(jk,iw)=zrupd(jk)
          enddo
       enddo
    enddo

    ! verify the lifted loop against the real spcvmc_sw outputs
    call chk1('tap pbbfd', a_pbbfd, zbbfd(1:nlayers+1))
    call chk1('tap pbbfu', a_pbbfu, zbbfu(1:nlayers+1))
    call chk1('tap pbbcd', a_pbbcd, zbbcd(1:nlayers+1))
    call chk1('tap pbbcu', a_pbbcu, zbbcu(1:nlayers+1))
    call chk1('tap pbbfddir', a_pbbfddir, zbbfddir(1:nlayers+1))
    call chk1('tap pbbcddir', a_pbbcddir, zbbcddir(1:nlayers+1))
    call chk1('tap puvfd', a_puvfd, zuvfd(1:nlayers+1))
    call chk1('tap puvcd', a_puvcd, zuvcd(1:nlayers+1))
    call chk1('tap pnifd', a_pnifd, znifd(1:nlayers+1))
    call chk1('tap pnicd', a_pnicd, znicd(1:nlayers+1))

    call wr1(trim(cpre)//'rt/zincflx', zincflx)
    call wr2(trim(cpre)//'rt/ztauc', t_ztauc)
    call wr2(trim(cpre)//'rt/zomcc', t_zomcc)
    call wr2(trim(cpre)//'rt/zgcc', t_zgcc)
    call wr2(trim(cpre)//'rt/ztauo', t_ztauo)
    call wr2(trim(cpre)//'rt/zomco', t_zomco)
    call wr2(trim(cpre)//'rt/zgco', t_zgco)
    call wname(trim(cpre)//'rt/lrtchkcld', 1, shape(t_lrtcld))
    write(ou) int(t_lrtcld,4)
    call wr2(trim(cpre)//'rt/zrefc', t_zrefc)
    call wr2(trim(cpre)//'rt/zrefdc', t_zrefdc)
    call wr2(trim(cpre)//'rt/ztrac', t_ztrac)
    call wr2(trim(cpre)//'rt/ztradc', t_ztradc)
    call wr2(trim(cpre)//'rt/zrefo', t_zrefo)
    call wr2(trim(cpre)//'rt/zrefdo', t_zrefdo)
    call wr2(trim(cpre)//'rt/ztrao', t_ztrao)
    call wr2(trim(cpre)//'rt/ztrado', t_ztrado)
    call wr2(trim(cpre)//'rt/zdbtc', t_zdbtc)
    call wr2(trim(cpre)//'rt/zdbt', t_zdbt)
    call wr2(trim(cpre)//'rt/ztdbtc', t_ztdbtc)
    call wr2(trim(cpre)//'rt/ztdbt', t_ztdbt)
    call wr2(trim(cpre)//'rt/zdbtc_nodel', t_zdbtcnd)
    call wr2(trim(cpre)//'rt/zdbt_nodel', t_zdbtnd)
    call wr2(trim(cpre)//'rt/ztdbtc_nodel', t_ztdbtcnd)
    call wr2(trim(cpre)//'rt/ztdbt_nodel', t_ztdbtnd)
    call wr2(trim(cpre)//'rt/zcd', zcd)
    call wr2(trim(cpre)//'rt/zcu', zcu)
    call wr2(trim(cpre)//'rt/zfd', zfd)
    call wr2(trim(cpre)//'rt/zfu', zfu)
    call wr2(trim(cpre)//'rt/zrdndc', t_zrdndc)
    call wr2(trim(cpre)//'rt/zrupc', t_zrupc)
    call wr2(trim(cpre)//'rt/zrupdc', t_zrupdc)
    call wr2(trim(cpre)//'rt/zrdnd', t_zrdnd)
    call wr2(trim(cpre)//'rt/zrup', t_zrup)
    call wr2(trim(cpre)//'rt/zrupd', t_zrupd)
  end subroutine run_rt_tap

end program sw_fixture_driver
