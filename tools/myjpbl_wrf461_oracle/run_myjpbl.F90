program run_myjpbl_oracle
  ! Column oracle for the MYJ PBL, WRF v4.6.1 phys/module_bl_myjpbl.F,
  ! compiled byte-unmodified at -O0 (build.sh).  This driver calls WRF's own
  ! MYJPBL (the routine module_pbl_driver.F calls for bl_pbl_physics = 2) on
  ! columns that make_columns.py generated, for NSTEP consecutive calls.
  !
  ! Usage:  run_myjpbl CASE_IN.bin CASE_OUT.bin CASE_OUT.txt
  !
  ! Input (stream, native little-endian):
  !   int32  ncol, nz, nstep, flqi, magic(=461)
  !   real32 dt
  !   (ncol, nz+1) real32, Fortran order: DZ PMID PINT TH T EXNER QV QCW QCI
  !                                       U V RHO TKE EXCH_H
  !   (ncol)       real32: HT TSK QSFC CHKLOWQ THZ0 QZ0 UZ0 VZ0 XLAND SICE
  !                        SNOW USTAR ZNT CT AKHS AKMS ELFLX
  !
  ! flqi = 1 passes QCI (WRF's PRESENT(QCI) arm, four mixed species and
  ! RQIBLTEN); flqi = 0 omits QCI and RQIBLTEN (three species).
  !
  ! Between calls the driver carries what WRF carries (TKE_MYJ, EXCH_H, QSFC,
  ! THZ0, QZ0, CT and the rest of the INOUT surface state) and also applies
  ! the returned tendencies to U, V, TH, QV, QCW, QCI as a forward step of
  ! length DT (T = TH*EXNER), so later calls see an evolved column.  Every
  ! call's complete input set is written before the call and every output
  ! after it, so a port can be replayed call by call.
  !
  ! Output: CASE_OUT.bin holds the raw records, CASE_OUT.txt one line per
  ! record: "<name> <r|i> <count>", in file order.
  use module_model_constants
  use module_bl_myjpbl
  implicit none

  integer :: ncol, nz, nstep, flqi, magic, istep, i, k
  real :: dt
  real, allocatable, dimension(:, :, :) :: dz, pmid, pint, th, t, exner, qv, qcw, qci
  real, allocatable, dimension(:, :, :) :: u, v, rho, tke, exch_h, el
  real, allocatable, dimension(:, :, :) :: rublten, rvblten, rthblten, rqvblten
  real, allocatable, dimension(:, :, :) :: rqcblten, rqiblten
  real, allocatable, dimension(:, :) :: ht, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0
  real, allocatable, dimension(:, :) :: xland, sice, snow, ustar, znt, ct, akhs, akms
  real, allocatable, dimension(:, :) :: elflx, pblh, mixht
  integer, allocatable, dimension(:, :) :: lowlyr, kpbl
  character(len=1024) :: fin, fout, fman
  character(len=32) :: tag
  integer :: uin, uout, uman

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, fman)
  open(newunit=uin, file=trim(fin), access='stream', form='unformatted', status='old')
  read(uin) ncol, nz, nstep, flqi, magic
  if (magic /= 461) stop 'bad magic'
  read(uin) dt

  allocate(dz(ncol, nz + 1, 1), pmid(ncol, nz + 1, 1), pint(ncol, nz + 1, 1))
  allocate(th(ncol, nz + 1, 1), t(ncol, nz + 1, 1), exner(ncol, nz + 1, 1))
  allocate(qv(ncol, nz + 1, 1), qcw(ncol, nz + 1, 1), qci(ncol, nz + 1, 1))
  allocate(u(ncol, nz + 1, 1), v(ncol, nz + 1, 1), rho(ncol, nz + 1, 1))
  allocate(tke(ncol, nz + 1, 1), exch_h(ncol, nz + 1, 1), el(ncol, nz + 1, 1))
  allocate(rublten(ncol, nz + 1, 1), rvblten(ncol, nz + 1, 1), rthblten(ncol, nz + 1, 1))
  allocate(rqvblten(ncol, nz + 1, 1), rqcblten(ncol, nz + 1, 1), rqiblten(ncol, nz + 1, 1))
  allocate(ht(ncol, 1), tsk(ncol, 1), qsfc(ncol, 1), chklowq(ncol, 1), thz0(ncol, 1))
  allocate(qz0(ncol, 1), uz0(ncol, 1), vz0(ncol, 1), xland(ncol, 1), sice(ncol, 1))
  allocate(snow(ncol, 1), ustar(ncol, 1), znt(ncol, 1), ct(ncol, 1), akhs(ncol, 1))
  allocate(akms(ncol, 1), elflx(ncol, 1), pblh(ncol, 1), mixht(ncol, 1))
  allocate(lowlyr(ncol, 1), kpbl(ncol, 1))

  read(uin) dz, pmid, pint, th, t, exner, qv, qcw, qci, u, v, rho, tke, exch_h
  read(uin) ht, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0, xland, sice, snow, &
            ustar, znt, ct, akhs, akms, elflx
  close(uin)
  lowlyr = 1

  open(newunit=uout, file=trim(fout), access='stream', form='unformatted', status='replace')
  open(newunit=uman, file=trim(fman), form='formatted', status='replace')
  write(uman, '(A,1X,I0,1X,I0,1X,I0,1X,I0)') '#header', ncol, nz, nstep, flqi
  call put_r('meta_dt', [dt])
  call put_constants()

  do istep = 1, nstep
    write(tag, '(A,I0,A)') 's', istep, '_in_'
    call put3(trim(tag)//'dz', dz); call put3(trim(tag)//'pmid', pmid)
    call put3(trim(tag)//'pint', pint); call put3(trim(tag)//'th', th)
    call put3(trim(tag)//'t', t); call put3(trim(tag)//'exner', exner)
    call put3(trim(tag)//'qv', qv); call put3(trim(tag)//'qcw', qcw)
    call put3(trim(tag)//'qci', qci); call put3(trim(tag)//'u', u)
    call put3(trim(tag)//'v', v); call put3(trim(tag)//'rho', rho)
    call put3(trim(tag)//'tke', tke); call put3(trim(tag)//'exch_h', exch_h)
    call put2(trim(tag)//'ht', ht); call put2(trim(tag)//'tsk', tsk)
    call put2(trim(tag)//'qsfc', qsfc); call put2(trim(tag)//'chklowq', chklowq)
    call put2(trim(tag)//'thz0', thz0); call put2(trim(tag)//'qz0', qz0)
    call put2(trim(tag)//'uz0', uz0); call put2(trim(tag)//'vz0', vz0)
    call put2(trim(tag)//'xland', xland); call put2(trim(tag)//'sice', sice)
    call put2(trim(tag)//'snow', snow); call put2(trim(tag)//'ustar', ustar)
    call put2(trim(tag)//'znt', znt); call put2(trim(tag)//'ct', ct)
    call put2(trim(tag)//'akhs', akhs); call put2(trim(tag)//'akms', akms)
    call put2(trim(tag)//'elflx', elflx)

    ! Sentinels: anything WRF does not write keeps one of these words.
    el = -999.; rublten = -999.; rvblten = -999.; rthblten = -999.
    rqvblten = -999.; rqcblten = -999.; rqiblten = -999.
    pblh = -999.; mixht = -999.; kpbl = -999

    if (flqi == 1) then
      call myjpbl(DT=dt, STEPBL=1, HT=ht, DZ=dz, PMID=pmid, PINT=pint,        &
                  TH=th, T=t, EXNER=exner, QV=qv, QCW=qcw, QCI=qci,           &
                  U=u, V=v, RHO=rho, TSK=tsk, QSFC=qsfc, CHKLOWQ=chklowq,     &
                  THZ0=thz0, QZ0=qz0, UZ0=uz0, VZ0=vz0, LOWLYR=lowlyr,        &
                  XLAND=xland, SICE=sice, SNOW=snow, TKE_MYJ=tke,             &
                  EXCH_H=exch_h, USTAR=ustar, ZNT=znt, EL_MYJ=el, PBLH=pblh,  &
                  KPBL=kpbl, CT=ct, AKHS=akhs, AKMS=akms, ELFLX=elflx,        &
                  MIXHT=mixht, RUBLTEN=rublten, RVBLTEN=rvblten,              &
                  RTHBLTEN=rthblten, RQVBLTEN=rqvblten, RQCBLTEN=rqcblten,    &
                  RQIBLTEN=rqiblten,                                          &
                  IDS=1, IDE=ncol + 1, JDS=1, JDE=2, KDS=1, KDE=nz + 1,       &
                  IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=nz + 1,           &
                  ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nz)
    else
      call myjpbl(DT=dt, STEPBL=1, HT=ht, DZ=dz, PMID=pmid, PINT=pint,        &
                  TH=th, T=t, EXNER=exner, QV=qv, QCW=qcw,                    &
                  U=u, V=v, RHO=rho, TSK=tsk, QSFC=qsfc, CHKLOWQ=chklowq,     &
                  THZ0=thz0, QZ0=qz0, UZ0=uz0, VZ0=vz0, LOWLYR=lowlyr,        &
                  XLAND=xland, SICE=sice, SNOW=snow, TKE_MYJ=tke,             &
                  EXCH_H=exch_h, USTAR=ustar, ZNT=znt, EL_MYJ=el, PBLH=pblh,  &
                  KPBL=kpbl, CT=ct, AKHS=akhs, AKMS=akms, ELFLX=elflx,        &
                  MIXHT=mixht, RUBLTEN=rublten, RVBLTEN=rvblten,              &
                  RTHBLTEN=rthblten, RQVBLTEN=rqvblten, RQCBLTEN=rqcblten,    &
                  IDS=1, IDE=ncol + 1, JDS=1, JDE=2, KDS=1, KDE=nz + 1,       &
                  IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=nz + 1,           &
                  ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nz)
    end if

    write(tag, '(A,I0,A)') 's', istep, '_out_'
    call put3(trim(tag)//'tke', tke); call put3(trim(tag)//'el_myj', el)
    call put3(trim(tag)//'exch_h', exch_h)
    call put3(trim(tag)//'rublten', rublten); call put3(trim(tag)//'rvblten', rvblten)
    call put3(trim(tag)//'rthblten', rthblten); call put3(trim(tag)//'rqvblten', rqvblten)
    call put3(trim(tag)//'rqcblten', rqcblten); call put3(trim(tag)//'rqiblten', rqiblten)
    call put2(trim(tag)//'pblh', pblh); call put2(trim(tag)//'mixht', mixht)
    call put2(trim(tag)//'qsfc', qsfc); call put2(trim(tag)//'thz0', thz0)
    call put2(trim(tag)//'qz0', qz0); call put2(trim(tag)//'ct', ct)
    call put2(trim(tag)//'uz0', uz0); call put2(trim(tag)//'vz0', vz0)
    call put2(trim(tag)//'akhs', akhs); call put2(trim(tag)//'akms', akms)
    call put2(trim(tag)//'ustar', ustar); call put2(trim(tag)//'znt', znt)
    call put_i(trim(tag)//'kpbl', kpbl(:, 1))

    ! Advance the column with the tendencies MYJ returned (forward step).
    do k = 1, nz
      do i = 1, ncol
        u(i, k, 1) = u(i, k, 1) + rublten(i, k, 1) * dt
        v(i, k, 1) = v(i, k, 1) + rvblten(i, k, 1) * dt
        th(i, k, 1) = th(i, k, 1) + rthblten(i, k, 1) * dt
        t(i, k, 1) = th(i, k, 1) * exner(i, k, 1)
        qv(i, k, 1) = max(qv(i, k, 1) + rqvblten(i, k, 1) * dt, 0.)
        qcw(i, k, 1) = max(qcw(i, k, 1) + rqcblten(i, k, 1) * dt, 0.)
        if (flqi == 1) qci(i, k, 1) = max(qci(i, k, 1) + rqiblten(i, k, 1) * dt, 0.)
      end do
    end do
  end do
  close(uout)
  close(uman)
  write(*, '(A,I0,A,I0,A,I0,A,I0)') 'myjpbl oracle: ncol=', ncol, ' nz=', nz, &
       ' nstep=', nstep, ' flqi=', flqi

contains

  subroutine put3(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :, :)
    write(uout) a(:, :, 1)
    write(uman, '(A,1X,A,1X,I0,1X,I0)') name, 'r', size(a, 1), size(a, 2)
  end subroutine put3

  subroutine put2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :)
    write(uout) a(:, 1)
    write(uman, '(A,1X,A,1X,I0,1X,I0)') name, 'r', size(a, 1), 1
  end subroutine put2

  subroutine put_r(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:)
    write(uout) a
    write(uman, '(A,1X,A,1X,I0,1X,I0)') name, 'r', size(a), 1
  end subroutine put_r

  subroutine put_i(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    write(uout) a
    write(uman, '(A,1X,A,1X,I0,1X,I0)') name, 'i', size(a), 1
  end subroutine put_i

  subroutine put_constants()
    ! The module's own folded PARAMETERs, as the -O0 object carries them,
    ! so a port can take its constants from the reference instead of
    ! re-deriving the folding.  B1**(2./3.) is the PRODQ2 :1164 operand.
    real :: c(48)
    c = 0.
    c(1) = g; c(2) = r_d; c(3) = cp; c(4) = xlv; c(5) = xls; c(6) = p608
    c(7) = pq0; c(8) = a2; c(9) = a3; c(10) = a4; c(11) = epsq2
    c(12) = vkarman; c(13) = capa; c(14) = rlivwv; c(15) = elocp
    c(16) = eps1; c(17) = eps2; c(18) = epsl; c(19) = epsru; c(20) = epsrs
    c(21) = epstrb; c(22) = fh; c(23) = alph; c(24) = beta; c(25) = el0max
    c(26) = el0min; c(27) = elfc; c(28) = a1; c(29) = a2x; c(30) = b1
    c(31) = b2; c(32) = c1; c(33) = elz0; c(34) = esq; c(35) = pq0sea
    c(36) = btg; c(37) = rb1; c(38) = adnh; c(39) = adnm; c(40) = anmh
    c(41) = anmm; c(42) = bdnh; c(43) = bdnm; c(44) = beqh; c(45) = beqm
    c(46) = bnmh; c(47) = bnmm; c(48) = b1**(2./3.)
    call put_r('const_a', c)
    c = 0.
    c(1) = bshh; c(2) = bshm; c(3) = bsmh; c(4) = cesh; c(5) = cesm
    c(6) = aeqh; c(7) = aeqm; c(8) = requ; c(9) = epsgh; c(10) = epsgm
    c(11) = ubryl; c(12) = ubry; c(13) = ubry3; c(14) = aubh; c(15) = aubm
    c(16) = bubh; c(17) = bubm; c(18) = cubr; c(19) = rcubr; c(20) = 0.5 * esq
    call put_r('const_b', c(1:20))
  end subroutine put_constants

end program run_myjpbl_oracle
