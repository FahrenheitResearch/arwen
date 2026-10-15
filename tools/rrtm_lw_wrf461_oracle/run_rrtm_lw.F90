! Column oracle for WRF v4.6.1 RRTM longwave (ra_lw_physics = 1).
!
! Drives the UNMODIFIED phys/module_ra_rrtm.F twice per column set:
!
!   1. through its public entry RRTMLWRAD, exactly as module_radiation_driver.F
!      calls it (every optional argument present, ghg_input = 0), giving the
!      outputs WRF hands back to the model: RTHRATEN, RTHRATENC, GLW, OLR;
!   2. through the same subroutines RRTM calls (INIRAD, MM5ATM, SETCOEF,
!      GASABS, RTRN), with the 1-D packing copied from RRTMLWRAD, so every
!      intermediate the port reproduces can be dumped and compared stage by
!      stage.  The driver checks that this second route reproduces the first
!      bit for bit (GLW, OLR and both tendencies) before writing anything.
!
! The stage route's scratch arrays (TAUG, PFRAC, SELFFAC, SELFFRAC, INDSELF)
! are pre-filled with NaN / a poison integer, so a WRF read of an element its
! own code never wrote would surface as a non-finite output and fail the
! cross-check above.
!
! Files are little-endian stream I/O (CONVERT= on each OPEN): the main program
! is compiled with -fconvert=big-endian because that is how WRF reads
! RRTM_DATA, and gfortran takes the process default from the main program.
!
! usage: run_rrtm_lw INPUT.bin OUTPUT.bin

module oracle_io
  implicit none
  integer, parameter :: KIND_F32 = 0, KIND_I32 = 1, KIND_F64 = 2
contains
  subroutine put_header(u, name, kind, dims)
    integer, intent(in) :: u, kind
    character(len=*), intent(in) :: name
    integer, intent(in) :: dims(:)
    character(len=32) :: padded
    padded = name
    write(u) padded, int(kind, 4), int(size(dims), 4), int(dims, 4)
  end subroutine put_header

  subroutine put_f32(u, name, dims, values)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    integer, intent(in) :: dims(:)
    real(4), intent(in) :: values(*)
    call put_header(u, name, KIND_F32, dims)
    write(u) values(1:product(dims))
  end subroutine put_f32

  subroutine put_i32(u, name, dims, values)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    integer, intent(in) :: dims(:)
    integer(4), intent(in) :: values(*)
    call put_header(u, name, KIND_I32, dims)
    write(u) values(1:product(dims))
  end subroutine put_i32

  subroutine put_f64(u, name, dims, values)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    integer, intent(in) :: dims(:)
    real(8), intent(in) :: values(*)
    call put_header(u, name, KIND_F64, dims)
    write(u) values(1:product(dims))
  end subroutine put_f64
end module oracle_io

program run_rrtm_lw
  use, intrinsic :: ieee_arithmetic
  use module_ra_rrtm
  use oracle_io
  implicit none

  character(len=512) :: in_path, out_path, tag
  integer, parameter :: uin = 21, uout = 22
  integer :: magic, version, nsets, iset
  integer :: ncol, nz, yr, icloud, iwarm, flags(6)
  real :: p_top, julian
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
             its, ite, jts, jte, kts, kte
  real, allocatable, dimension(:,:,:) :: p8w, t8w, p3d, pi3d, dz8w, t3d, &
       rho3d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, cldfra3d, rthraten, rthratenc
  real, allocatable, dimension(:,:) :: tsk, emiss, glw, olr, buf2
  real, allocatable :: inbuf(:)
  logical :: f_qv, f_qc, f_qr, f_qi, f_qs, f_qg, warm_rain
  real :: qnan
  integer :: i, k, nk, nl, l, ngpt_
  ! stage arrays, column last so each writes as C-order (ncol, ...)
  real, allocatable :: s_o3prof(:,:), s_pavel(:,:), s_tavel(:,:), s_pz(:,:), &
       s_tz(:,:), s_cldfrac(:,:), s_taucloud(:,:), s_coldry(:,:), s_wkl(:,:,:), &
       s_tbound(:), s_colh2o(:,:), s_colco2(:,:), s_colo3(:,:), s_coln2o(:,:), &
       s_colch4(:,:), s_colo2(:,:), s_co2mult(:,:), s_fac00(:,:), s_fac01(:,:), &
       s_fac10(:,:), s_fac11(:,:), s_forfac(:,:), s_selffac(:,:), &
       s_selffrac(:,:), s_taug(:,:,:), s_pfrac(:,:,:), s_totdflux(:,:), &
       s_totuflux(:,:), s_htr(:,:), s_htrc(:,:), s_tten(:,:), s_ttenc(:,:), &
       s_glw(:), s_olr(:)
  integer, allocatable :: s_jp(:,:), s_jt(:,:), s_jt1(:,:), s_indself(:,:), &
       s_itr(:,:,:), s_laytrop(:), s_layswtch(:), s_laylow(:)
  ! RRTMLWRAD's 1-D packing, copied
  real, allocatable :: pw1d(:), tw1d(:), tten1d(:), tten1dc(:), cldfra1d(:), &
       dz1d(:), p1d(:), t1d(:), qv1d(:), qc1d(:), qr1d(:), qi1d(:), qs1d(:), &
       qg1d(:), o3prof(:)
  ! RRTM's locals, copied
  integer, allocatable :: itr(:,:), icldlyr(:), jp(:), jt(:), jt1(:), indself(:)
  real, allocatable :: pfrac(:,:), taug(:,:), wkl(:,:), wx(:,:), pavel(:), &
       tavel(:), cldfrac(:), taucloud(:), coldry(:), colh2o(:), colco2(:), &
       colo3(:), coln2o(:), colch4(:), colo2(:), co2mult(:), fac00(:), &
       fac01(:), fac10(:), fac11(:), forfac(:), selffac(:), selffrac(:), &
       pz(:), tz(:), totdflux(:), totuflux(:), htr(:), htrc(:)
  integer :: laytrop, layswtch, laylow, ktep1
  real :: tbound, tsfc, emiss0, glw0, olr0
  real :: semiss(NBANDS)
  real(8) :: co2vmr, n2ovmr, ch4vmr
  real(8) :: gases(3)
  real, parameter :: r_d = 287., g = 9.81
  logical :: first

  if (command_argument_count() /= 2) then
    write(*,*) 'usage: run_rrtm_lw INPUT.bin OUTPUT.bin'
    error stop 2
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  qnan = ieee_value(qnan, ieee_quiet_nan)
  ngpt_ = NGPT

  open(uin, file=trim(in_path), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(uout, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(uin) magic, version, nsets
  if (magic /= int(z'52524C57') .or. version /= 1) then
    write(*,*) 'bad input header', magic, version
    error stop 3
  end if
  write(uout) int(z'52524C57', 4), int(1, 4), int(nsets, 4)

  first = .true.
  do iset = 1, nsets
    read(uin) ncol, nz, yr, icloud, iwarm, flags, p_top, julian
    f_qv = flags(1) /= 0; f_qc = flags(2) /= 0; f_qr = flags(3) /= 0
    f_qi = flags(4) /= 0; f_qs = flags(5) /= 0; f_qg = flags(6) /= 0
    warm_rain = iwarm /= 0
    ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
    ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
    its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

    allocate(p8w(ncol,nz+1,1), t8w(ncol,nz+1,1), p3d(ncol,nz+1,1), &
             pi3d(ncol,nz+1,1), dz8w(ncol,nz+1,1), t3d(ncol,nz+1,1), &
             rho3d(ncol,nz+1,1), qv3d(ncol,nz+1,1), qc3d(ncol,nz+1,1), &
             qr3d(ncol,nz+1,1), qi3d(ncol,nz+1,1), qs3d(ncol,nz+1,1), &
             qg3d(ncol,nz+1,1), cldfra3d(ncol,nz+1,1), rthraten(ncol,nz+1,1), &
             rthratenc(ncol,nz+1,1), tsk(ncol,1), emiss(ncol,1), glw(ncol,1), &
             olr(ncol,1), inbuf(ncol*(nz+1)))
    ! Level kme of a mass array is never read by RRTMLWRAD; NaN proves it.
    call read_level_field(p8w, nz + 1); call read_level_field(t8w, nz + 1)
    call read_level_field(p3d, nz);     call read_level_field(pi3d, nz)
    call read_level_field(dz8w, nz);    call read_level_field(t3d, nz)
    call read_level_field(rho3d, nz);   call read_level_field(qv3d, nz)
    call read_level_field(qc3d, nz);    call read_level_field(qr3d, nz)
    call read_level_field(qi3d, nz);    call read_level_field(qs3d, nz)
    call read_level_field(qg3d, nz);    call read_level_field(cldfra3d, nz)
    read(uin) inbuf(1:ncol); tsk(:,1) = inbuf(1:ncol)
    read(uin) inbuf(1:ncol); emiss(:,1) = inbuf(1:ncol)

    if (first) then
      ! rrtm_lookuptable: reads RRTM_DATA and builds the combined tables.
      call rrtminit(p_top, .true., ids, ide, jds, jde, kds, kde, &
                    ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
      call dump_tables()
      first = .false.
    end if

    ! ---- route 1: RRTMLWRAD as module_radiation_driver.F:1848 calls it ----
    rthraten = 0.; rthratenc = 0.; glw = -1.; olr = -1.
    call RRTMLWRAD(p_top=p_top, rthraten=rthraten, rthratenc=rthratenc,  &
         glw=glw, olr=olr, emiss=emiss, qv3d=qv3d, qc3d=qc3d, qr3d=qr3d,  &
         qi3d=qi3d, qs3d=qs3d, qg3d=qg3d, p8w=p8w, p3d=p3d, pi3d=pi3d,    &
         dz8w=dz8w, tsk=tsk, t3d=t3d, t8w=t8w, rho3d=rho3d,                &
         cldfra3d=cldfra3d, r=r_d, g=g, f_qv=f_qv, f_qc=f_qc, f_qr=f_qr,  &
         f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, icloud=icloud,                   &
         warm_rain=warm_rain, yr=yr, julian=julian, ghg_input=0,           &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,             &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,             &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
    nl = NLAYERS

    write(tag, '(A,I0)') 'set', iset
    call put_header(uout, 'SET', KIND_I32, [6])
    write(uout) int(ncol, 4), int(nz, 4), int(nl, 4), int(yr, 4), &
                int(icloud, 4), int(iwarm, 4)
    call put_f32(uout, 'p_top', [1], [p_top])
    allocate(buf2(nz, ncol))
    do i = 1, ncol
      do k = 1, nz
        buf2(k, i) = rthraten(i, k, 1)
      end do
    end do
    call put_f32(uout, 'rthraten', [ncol, nz], buf2)
    do i = 1, ncol
      do k = 1, nz
        buf2(k, i) = rthratenc(i, k, 1)
      end do
    end do
    call put_f32(uout, 'rthratenc', [ncol, nz], buf2)
    call put_f32(uout, 'glw', [ncol], glw(:,1))
    call put_f32(uout, 'olr', [ncol], olr(:,1))
    deallocate(buf2)

    ! ---- route 2: the stages, column by column ----
    co2vmr = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6   ! RRTMLWRAD, ghg_input=0
    n2ovmr = 319.e-9
    ch4vmr = 1774.e-9
    gases = [co2vmr, n2ovmr, ch4vmr]
    call put_f64(uout, 'gases_f64', [3], gases)

    allocate(pw1d(nz+1), tw1d(nz+1), tten1d(nz), tten1dc(nz), cldfra1d(nz), &
             dz1d(nz), p1d(nz), t1d(nz), qv1d(nz), qc1d(nz), qr1d(nz), &
             qi1d(nz), qs1d(nz), qg1d(nz), o3prof(nz))
    allocate(itr(NGPT,nl), pfrac(NGPT,nl), taug(NGPT,nl), wkl(35,nl), &
             wx(MAXXSEC,nl), pavel(nl), tavel(nl), cldfrac(nl), taucloud(nl), &
             coldry(nl), colh2o(nl), colco2(nl), colo3(nl), coln2o(nl), &
             colch4(nl), colo2(nl), co2mult(nl), fac00(nl), fac01(nl), &
             fac10(nl), fac11(nl), forfac(nl), selffac(nl), selffrac(nl), &
             icldlyr(nl), jp(nl), jt(nl), jt1(nl), indself(nl), pz(0:nl), &
             tz(0:nl), totdflux(0:nl), totuflux(0:nl), htr(0:nl), htrc(0:nl))
    allocate(s_o3prof(nz,ncol), s_pavel(nl,ncol), s_tavel(nl,ncol), &
             s_pz(0:nl,ncol), s_tz(0:nl,ncol), s_cldfrac(nl,ncol), &
             s_taucloud(nl,ncol), s_coldry(nl,ncol), s_wkl(7,nl,ncol), &
             s_tbound(ncol), s_colh2o(nl,ncol), s_colco2(nl,ncol), &
             s_colo3(nl,ncol), s_coln2o(nl,ncol), s_colch4(nl,ncol), &
             s_colo2(nl,ncol), s_co2mult(nl,ncol), s_fac00(nl,ncol), &
             s_fac01(nl,ncol), s_fac10(nl,ncol), s_fac11(nl,ncol), &
             s_forfac(nl,ncol), s_selffac(nl,ncol), s_selffrac(nl,ncol), &
             s_taug(NGPT,nl,ncol), s_pfrac(NGPT,nl,ncol), &
             s_totdflux(0:nl,ncol), s_totuflux(0:nl,ncol), s_htr(0:nl,ncol), &
             s_htrc(0:nl,ncol), s_tten(nz,ncol), s_ttenc(nz,ncol), s_glw(ncol), &
             s_olr(ncol), s_jp(nl,ncol), s_jt(nl,ncol), s_jt1(nl,ncol), &
             s_indself(nl,ncol), s_itr(NGPT,nl,ncol), s_laytrop(ncol), &
             s_layswtch(ncol), s_laylow(ncol))

    do i = 1, ncol
      ! RRTMLWRAD F:1879-1981, verbatim logic
      do k = kts, kte + 1
        nk = kme - k + kms
        pw1d(k) = p8w(i, nk, 1)/100.
        tw1d(k) = t8w(i, nk, 1)
      end do
      qv1d = 0.; qc1d = 0.; qr1d = 0.; qi1d = 0.; qs1d = 0.; cldfra1d = 0.
      do k = kts, kte
        nk = kme - 1 - k + kms
        qv1d(k) = qv3d(i, nk, 1)
        qv1d(k) = max(0., qv1d(k))
      end do
      do k = kts, kte
        nk = kme - 1 - k + kms
        tten1d(k) = 0.; tten1dc(k) = 0.
        t1d(k) = t3d(i, nk, 1)
        p1d(k) = p3d(i, nk, 1)/100.
        dz1d(k) = dz8w(i, nk, 1)
      end do
      if (icloud /= 0) then
        do k = kts, kte
          nk = kme - 1 - k + kms
          cldfra1d(k) = cldfra3d(i, nk, 1)
        end do
        if (f_qc) then
          do k = kts, kte
            nk = kme - 1 - k + kms
            qc1d(k) = max(0., qc3d(i, nk, 1))
          end do
        end if
        if (f_qr) then
          do k = kts, kte
            nk = kme - 1 - k + kms
            qr1d(k) = max(0., qr3d(i, nk, 1))
          end do
        end if
        if (.not. f_qi .and. .not. warm_rain) then
          do k = kts, kte
            if (t1d(k) .lt. 273.15) then
              qi1d(k) = qc1d(k); qs1d(k) = qr1d(k); qc1d(k) = 0.; qr1d(k) = 0.
            end if
          end do
        end if
        do k = kts, kte
          nk = kme - 1 - k + kms
          qi1d(k) = max(0., qi3d(i, nk, 1))
        end do
        if (f_qs) then
          do k = kts, kte
            nk = kme - 1 - k + kms
            qs1d(k) = max(0., qs3d(i, nk, 1))
          end do
        end if
        if (f_qg) then
          do k = kts, kte
            nk = kme - 1 - k + kms
            qg1d(k) = max(0., qg3d(i, nk, 1))
          end do
        end if
      end if
      emiss0 = emiss(i, 1)
      tsfc = tsk(i, 1)

      ! RRTM F:2059-2298, with the scratch poisoned
      ktep1 = nl
      taug = qnan; pfrac = qnan; selffac = qnan; selffrac = qnan
      indself = -999999; itr = -999999
      call INIRAD(o3prof, pw1d, kts, kte)
      call MM5ATM(cldfra1d, o3prof, t1d, tw1d, tsfc, qv1d, qc1d, qr1d, qi1d, &
                  qs1d, qg1d, p1d, pw1d, dz1d, emiss0, r_d, g,               &
                  pavel, tavel, pz, tz, cldfrac, taucloud, coldry,           &
                  wkl, wx, tbound, semiss, co2vmr, n2ovmr, ch4vmr, kts, kte)
      call SETCOEF(kts, ktep1, pavel, tavel, coldry, colh2o, colco2, colo3,  &
                   coln2o, colch4, colo2, co2mult, fac00, fac01, fac10,      &
                   fac11, forfac, selffac, selffrac, jp, jt, jt1, indself,   &
                   wkl, laytrop, layswtch, laylow)
      call GASABS(kts, ktep1, coldry, colh2o, colco2, colo3, coln2o, colch4, &
                  colo2, co2mult, fac00, fac01, fac10, fac11, forfac,        &
                  selffac, selffrac, jp, jt, jt1, indself, itr, wx, pfrac,   &
                  taug, laytrop, layswtch, laylow)
      do k = 1, nl
        if (cldfrac(k) .gt. 0.) then
          icldlyr(k) = 1
        else
          icldlyr(k) = 0
        end if
      end do
      call RTRN(kts, ktep1, tavel, pz, tz, cldfrac, taucloud, totdflux,      &
                totuflux, htr, htrc, icldlyr, itr, pfrac, tbound, semiss)
      do k = 1, kte
        tten1d(k) = htr(kte - k)/86400.
        tten1dc(k) = htrc(kte - k)/86400.
      end do
      glw0 = totdflux(0)
      olr0 = totuflux(kte)

      ! The two routes must agree bit for bit, or the stage dump is not
      ! a dump of what RRTMLWRAD computed.
      if (transfer(glw0, 0) /= transfer(glw(i, 1), 0) .or. &
          transfer(olr0, 0) /= transfer(olr(i, 1), 0)) then
        write(*,*) 'route mismatch, set', iset, 'column', i, glw0, glw(i,1), &
                   olr0, olr(i,1)
        error stop 4
      end if
      do k = kts, kte
        nk = kme - 1 - k + kms
        if (transfer(0. + tten1d(nk)/pi3d(i, k, 1), 0) /=              &
            transfer(rthraten(i, k, 1), 0) .or.                          &
            transfer(0. + tten1dc(nk)/pi3d(i, k, 1), 0) /=             &
            transfer(rthratenc(i, k, 1), 0)) then
          write(*,*) 'tendency route mismatch', iset, i, k
          error stop 4
        end if
      end do
      if (.not. (ieee_is_finite(glw0) .and. ieee_is_finite(olr0))) then
        write(*,*) 'non-finite flux: an unwritten scratch element was read', &
                   iset, i
        error stop 5
      end if

      s_o3prof(:, i) = o3prof
      s_pavel(:, i) = pavel; s_tavel(:, i) = tavel
      s_pz(:, i) = pz; s_tz(:, i) = tz
      s_cldfrac(:, i) = cldfrac; s_taucloud(:, i) = taucloud
      s_coldry(:, i) = coldry
      do l = 1, nl
        s_wkl(:, l, i) = wkl(1:7, l)
      end do
      s_tbound(i) = tbound
      s_colh2o(:, i) = colh2o; s_colco2(:, i) = colco2; s_colo3(:, i) = colo3
      s_coln2o(:, i) = coln2o; s_colch4(:, i) = colch4; s_colo2(:, i) = colo2
      s_co2mult(:, i) = co2mult
      s_fac00(:, i) = fac00; s_fac01(:, i) = fac01
      s_fac10(:, i) = fac10; s_fac11(:, i) = fac11
      s_forfac(:, i) = forfac; s_selffac(:, i) = selffac
      s_selffrac(:, i) = selffrac
      s_jp(:, i) = jp; s_jt(:, i) = jt; s_jt1(:, i) = jt1
      s_indself(:, i) = indself
      s_laytrop(i) = laytrop; s_layswtch(i) = layswtch; s_laylow(i) = laylow
      s_taug(:, :, i) = taug; s_pfrac(:, :, i) = pfrac; s_itr(:, :, i) = itr
      s_totdflux(:, i) = totdflux; s_totuflux(:, i) = totuflux
      s_htr(:, i) = htr; s_htrc(:, i) = htrc
      s_tten(:, i) = tten1d; s_ttenc(:, i) = tten1dc
      s_glw(i) = glw0; s_olr(i) = olr0
    end do

    call put_f32(uout, 'o3prof', [ncol, nz], s_o3prof)
    call put_f32(uout, 'pavel', [ncol, nl], s_pavel)
    call put_f32(uout, 'tavel', [ncol, nl], s_tavel)
    call put_f32(uout, 'pz', [ncol, nl + 1], s_pz)
    call put_f32(uout, 'tz', [ncol, nl + 1], s_tz)
    call put_f32(uout, 'cldfrac', [ncol, nl], s_cldfrac)
    call put_f32(uout, 'taucloud', [ncol, nl], s_taucloud)
    call put_f32(uout, 'coldry', [ncol, nl], s_coldry)
    call put_f32(uout, 'wkl', [ncol, nl, 7], s_wkl)
    call put_f32(uout, 'tbound', [ncol], s_tbound)
    call put_f32(uout, 'colh2o', [ncol, nl], s_colh2o)
    call put_f32(uout, 'colco2', [ncol, nl], s_colco2)
    call put_f32(uout, 'colo3', [ncol, nl], s_colo3)
    call put_f32(uout, 'coln2o', [ncol, nl], s_coln2o)
    call put_f32(uout, 'colch4', [ncol, nl], s_colch4)
    call put_f32(uout, 'colo2', [ncol, nl], s_colo2)
    call put_f32(uout, 'co2mult', [ncol, nl], s_co2mult)
    call put_f32(uout, 'fac00', [ncol, nl], s_fac00)
    call put_f32(uout, 'fac01', [ncol, nl], s_fac01)
    call put_f32(uout, 'fac10', [ncol, nl], s_fac10)
    call put_f32(uout, 'fac11', [ncol, nl], s_fac11)
    call put_f32(uout, 'forfac', [ncol, nl], s_forfac)
    call put_f32(uout, 'selffac', [ncol, nl], s_selffac)
    call put_f32(uout, 'selffrac', [ncol, nl], s_selffrac)
    call put_i32(uout, 'jp', [ncol, nl], s_jp)
    call put_i32(uout, 'jt', [ncol, nl], s_jt)
    call put_i32(uout, 'jt1', [ncol, nl], s_jt1)
    call put_i32(uout, 'indself', [ncol, nl], s_indself)
    call put_i32(uout, 'laytrop', [ncol], s_laytrop)
    call put_i32(uout, 'layswtch', [ncol], s_layswtch)
    call put_i32(uout, 'laylow', [ncol], s_laylow)
    call put_f32(uout, 'taug', [ncol, nl, ngpt_], s_taug)
    call put_f32(uout, 'pfrac', [ncol, nl, ngpt_], s_pfrac)
    call put_i32(uout, 'itr', [ncol, nl, ngpt_], s_itr)
    call put_f32(uout, 'totdflux', [ncol, nl + 1], s_totdflux)
    call put_f32(uout, 'totuflux', [ncol, nl + 1], s_totuflux)
    call put_f32(uout, 'htr', [ncol, nl + 1], s_htr)
    call put_f32(uout, 'htrc', [ncol, nl + 1], s_htrc)
    call put_f32(uout, 'tten_topdown', [ncol, nz], s_tten)
    call put_f32(uout, 'ttenc_topdown', [ncol, nz], s_ttenc)
    call put_header(uout, 'ENDSET', KIND_I32, [1])
    write(uout) int(iset, 4)

    deallocate(p8w, t8w, p3d, pi3d, dz8w, t3d, rho3d, qv3d, qc3d, qr3d, qi3d, &
               qs3d, qg3d, cldfra3d, rthraten, rthratenc, tsk, emiss, glw, &
               olr, inbuf)
    deallocate(pw1d, tw1d, tten1d, tten1dc, cldfra1d, dz1d, p1d, t1d, qv1d, &
               qc1d, qr1d, qi1d, qs1d, qg1d, o3prof)
    deallocate(itr, pfrac, taug, wkl, wx, pavel, tavel, cldfrac, taucloud, &
               coldry, colh2o, colco2, colo3, coln2o, colch4, colo2, co2mult, &
               fac00, fac01, fac10, fac11, forfac, selffac, selffrac, icldlyr, &
               jp, jt, jt1, indself, pz, tz, totdflux, totuflux, htr, htrc)
    deallocate(s_o3prof, s_pavel, s_tavel, s_pz, s_tz, s_cldfrac, s_taucloud, &
               s_coldry, s_wkl, s_tbound, s_colh2o, s_colco2, s_colo3, &
               s_coln2o, s_colch4, s_colo2, s_co2mult, s_fac00, s_fac01, &
               s_fac10, s_fac11, s_forfac, s_selffac, s_selffrac, s_taug, &
               s_pfrac, s_totdflux, s_totuflux, s_htr, s_htrc, s_tten, &
               s_ttenc, s_glw, s_olr, s_jp, s_jt, s_jt1, s_indself, s_itr, &
               s_laytrop, s_layswtch, s_laylow)
  end do
  call put_header(uout, 'END', KIND_I32, [1])
  write(uout) int(nsets, 4)
  close(uin)
  close(uout)

contains

  subroutine read_level_field(a, nlev)
    real, intent(inout) :: a(:,:,:)
    integer, intent(in) :: nlev
    integer :: kk
    ! input is C-order (nlev, ncol): one level of every column at a time
    a = qnan
    do kk = 1, nlev
      read(uin) inbuf(1:ncol)
      a(:, kk, 1) = inbuf(1:ncol)
    end do
  end subroutine read_level_field

  subroutine dump_tables()
    call put_header(uout, 'TABLES', KIND_I32, [1])
    write(uout) int(0, 4)
    call put_f32(uout, 'FLUXFAC', [1], [FLUXFAC])
    call put_f32(uout, 'BPADE', [1], [BPADE])
    call put_f32(uout, 'HEATFAC', [1], [HEATFAC])
    call put_f32(uout, 'TAU', [5001], TAU)
    call put_f32(uout, 'TF', [5001], TF)
    call put_f32(uout, 'TRANS', [5001], TRANS)
    call put_f32(uout, 'CORR1', [201], CORR1)
    call put_f32(uout, 'CORR2', [201], CORR2)
    call put_f32(uout, 'RWGT', [MG*NBANDS], RWGT)
    ! combined coefficient tables, Fortran (rows, ng) => C (ng, rows)
    call put_f32(uout, 'ABSA1', [NG1, 5*13], ABSA1)
    call put_f32(uout, 'ABSB1', [NG1, 5*47], ABSB1)
    call put_f32(uout, 'SELFREFC1', [NG1, 10], SELFREFC1)
    call put_f32(uout, 'ABSA2', [NG2, 5*13], ABSA2)
    call put_f32(uout, 'ABSB2', [NG2, 5*47], ABSB2)
    call put_f32(uout, 'SELFREFC2', [NG2, 10], SELFREFC2)
    call put_f32(uout, 'ABSA3', [NG3, 10*5*13], ABSA3)
    call put_f32(uout, 'ABSB3', [NG3, 5*5*47], ABSB3)
    call put_f32(uout, 'SELFREFC3', [NG3, 10], SELFREFC3)
    call put_f32(uout, 'ABSA4', [NG4, 9*5*13], ABSA4)
    call put_f32(uout, 'ABSB4', [NG4, 6*5*47], ABSB4)
    call put_f32(uout, 'SELFREFC4', [NG4, 10], SELFREFC4)
    call put_f32(uout, 'ABSA5', [NG5, 9*5*13], ABSA5)
    call put_f32(uout, 'ABSB5', [NG5, 5*5*47], ABSB5)
    call put_f32(uout, 'SELFREFC5', [NG5, 10], SELFREFC5)
    call put_f32(uout, 'ABSA6', [NG6, 5*13], ABSA6)
    call put_f32(uout, 'SELFREFC6', [NG6, 10], SELFREFC6)
    call put_f32(uout, 'ABSA7', [NG7, 9*5*13], ABSA7)
    call put_f32(uout, 'ABSB7', [NG7, 5*47], ABSB7)
    call put_f32(uout, 'SELFREFC7', [NG7, 10], SELFREFC7)
    call put_f32(uout, 'ABSA8', [NG8, 5*7], ABSA8)
    call put_f32(uout, 'ABSB8', [NG8, 5*53], ABSB8)
    call put_f32(uout, 'SELFREFC8', [NG8, 10], SELFREFC8)
    call put_f32(uout, 'ABSA9', [NG9, 11*5*13], ABSA9)
    call put_f32(uout, 'ABSB9', [NG9, 5*47], ABSB9)
    call put_f32(uout, 'SELFREFC9', [NG9, 10], SELFREFC9)
    call put_f32(uout, 'ABSA10', [NG10, 5*13], ABSA10)
    call put_f32(uout, 'ABSB10', [NG10, 5*47], ABSB10)
    call put_f32(uout, 'ABSA11', [NG11, 5*13], ABSA11)
    call put_f32(uout, 'ABSB11', [NG11, 5*47], ABSB11)
    call put_f32(uout, 'SELFREFC11', [NG11, 10], SELFREFC11)
    call put_f32(uout, 'ABSA12', [NG12, 9*5*13], ABSA12)
    call put_f32(uout, 'SELFREFC12', [NG12, 10], SELFREFC12)
    call put_f32(uout, 'ABSA13', [NG13, 9*5*13], ABSA13)
    call put_f32(uout, 'SELFREFC13', [NG13, 10], SELFREFC13)
    call put_f32(uout, 'ABSA14', [NG14, 5*13], ABSA14)
    call put_f32(uout, 'ABSB14', [NG14, 5*47], ABSB14)
    call put_f32(uout, 'SELFREFC14', [NG14, 10], SELFREFC14)
    call put_f32(uout, 'ABSA15', [NG15, 9*5*13], ABSA15)
    call put_f32(uout, 'SELFREFC15', [NG15, 10], SELFREFC15)
    call put_f32(uout, 'ABSA16', [NG16, 9*5*13], ABSA16)
    call put_f32(uout, 'SELFREFC16', [NG16, 10], SELFREFC16)
    call put_f32(uout, 'FORREFC1', [NG1], FORREFC1)
    call put_f32(uout, 'FORREFC2', [NG2], FORREFC2)
    call put_f32(uout, 'FORREFC3', [NG3], FORREFC3)
    call put_f32(uout, 'ABSN2OAC3', [NG3], ABSN2OAC3)
    call put_f32(uout, 'ABSN2OBC3', [NG3], ABSN2OBC3)
    call put_f32(uout, 'CCL4C5', [NG5], CCL4C5)
    call put_f32(uout, 'ABSCO2C6', [NG6], ABSCO2C6)
    call put_f32(uout, 'CFC11ADJC6', [NG6], CFC11ADJC6)
    call put_f32(uout, 'CFC12C6', [NG6], CFC12C6)
    call put_f32(uout, 'ABSCO2C7', [NG7], ABSCO2C7)
    call put_f32(uout, 'ABSCO2AC8', [NG8], ABSCO2AC8)
    call put_f32(uout, 'ABSCO2BC8', [NG8], ABSCO2BC8)
    call put_f32(uout, 'ABSN2OAC8', [NG8], ABSN2OAC8)
    call put_f32(uout, 'ABSN2OBC8', [NG8], ABSN2OBC8)
    call put_f32(uout, 'CFC12C8', [NG8], CFC12C8)
    call put_f32(uout, 'CFC22ADJC8', [NG8], CFC22ADJC8)
    call put_f32(uout, 'ABSN2OC9', [3*NG9], ABSN2OC9)
    call put_f32(uout, 'FRACREFAC1', [NG1], FRACREFAC1)
    call put_f32(uout, 'FRACREFBC1', [NG1], FRACREFBC1)
    call put_f32(uout, 'FRACREFAC2', [13, NG2], FRACREFAC2)
    call put_f32(uout, 'FRACREFBC2', [NG2], FRACREFBC2)
    call put_f32(uout, 'FRACREFAC3', [10, NG3], FRACREFAC3)
    call put_f32(uout, 'FRACREFBC3', [5, NG3], FRACREFBC3)
    call put_f32(uout, 'FRACREFAC4', [9, NG4], FRACREFAC4)
    call put_f32(uout, 'FRACREFBC4', [6, NG4], FRACREFBC4)
    call put_f32(uout, 'FRACREFAC5', [9, NG5], FRACREFAC5)
    call put_f32(uout, 'FRACREFBC5', [5, NG5], FRACREFBC5)
    call put_f32(uout, 'FRACREFAC6', [NG6], FRACREFAC6)
    call put_f32(uout, 'FRACREFAC7', [9, NG7], FRACREFAC7)
    call put_f32(uout, 'FRACREFBC7', [NG7], FRACREFBC7)
    call put_f32(uout, 'FRACREFAC8', [NG8], FRACREFAC8)
    call put_f32(uout, 'FRACREFBC8', [NG8], FRACREFBC8)
    call put_f32(uout, 'FRACREFAC9', [9, NG9], FRACREFAC9)
    call put_f32(uout, 'FRACREFBC9', [NG9], FRACREFBC9)
    call put_f32(uout, 'FRACREFAC10', [NG10], FRACREFAC10)
    call put_f32(uout, 'FRACREFBC10', [NG10], FRACREFBC10)
    call put_f32(uout, 'FRACREFAC11', [NG11], FRACREFAC11)
    call put_f32(uout, 'FRACREFBC11', [NG11], FRACREFBC11)
    call put_f32(uout, 'FRACREFAC12', [9, NG12], FRACREFAC12)
    call put_f32(uout, 'FRACREFAC13', [9, NG13], FRACREFAC13)
    call put_f32(uout, 'FRACREFAC14', [NG14], FRACREFAC14)
    call put_f32(uout, 'FRACREFBC14', [NG14], FRACREFBC14)
    call put_f32(uout, 'FRACREFAC15', [9, NG15], FRACREFAC15)
    call put_f32(uout, 'FRACREFAC16', [9, NG16], FRACREFAC16)
  end subroutine dump_tables

end program run_rrtm_lw
