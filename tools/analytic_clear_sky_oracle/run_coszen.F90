! Column oracle driver for the WRF v4.6.1 side of ra_lw/ra_sw_physics = 90.
!
! WRF 4.6.1 has no analytic clear-sky scheme, so the only output of WOOF's
! ra=90 adapter that has a WRF definition is the radiation-call COSZEN
! carrier: radconst + calc_coszen (phys/module_radiation_driver.F
! :3469-3511, :3514-3541, called at :1188-1189 and :1206-1208 with
! xtime + radt*0.5).  Those two routines are compiled UNMODIFIED from the
! pinned WRF tree (build.sh extracts them by line range and checks the
! source hash); this driver only forms their scalar inputs the way WRF's
! own frame does:
!
!   julian  frame/module_domain.F:2157-2165  currentDayOfYearReal
!           = REAL(dayOfYear_r8) - 1.0, dayOfYear_r8 from WRF's own
!           ESMF time library (external/esmf_time_f90, linked as built)
!   gmt     frame/module_domain.F:2439-2447  domain_setgmtetc
!   xtime   frame/module_domain.F:2177-2191  minutesSinceSimulationStart
!
! cz_in.bin (stream, little-endian):
!   int32 ncol, ncase
!   real32 xlat(ncol), xlon(ncol)
!   per case: int32 yy, mm, dd, h, mi, s, elapsed_seconds; real32 radt
! cz_out.bin: per case
!   real32 julian, xtime, gmt, declin, solcon, coszen(ncol), hrang(ncol)
program run_coszen
  use module_utility
  use wrf461_model_constants, only: DEGRAD, DPD
  use wrf461_coszen, only: radconst, calc_coszen
  implicit none
  integer :: ncol, ncase, ic, rc
  integer :: yy, mm, dd, hh, mi, ss, esec
  integer :: hr, mn, sec, ms, days, seconds, Sn, Sd
  real :: radt, julian, xtime, gmt, declin, solcon
  real(WRFU_KIND_R8) :: currentDayOfYearR8
  real, allocatable :: xlat(:,:), xlon(:,:), coszen(:,:), hrang(:,:)
  type(WRFU_Time) :: tstart, tcur
  type(WRFU_TimeInterval) :: elapsed, since
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call WRFU_Initialize(defaultCalKind=WRFU_CAL_GREGORIAN, rc=rc)
  if (rc /= WRFU_SUCCESS) stop 'WRFU_Initialize'

  ! WRF's flags carry -fconvert=big-endian; the oracle files are little-endian
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', &
       convert='little_endian')
  read(10) ncol, ncase
  allocate(xlat(ncol,1), xlon(ncol,1), coszen(ncol,1), hrang(ncol,1))
  read(10) xlat, xlon
  open(11, file=trim(fout), access='stream', form='unformatted', status='replace', &
       convert='little_endian')
  do ic = 1, ncase
     read(10) yy, mm, dd, hh, mi, ss, esec, radt
     call WRFU_TimeSet(tstart, YY=yy, MM=mm, DD=dd, H=hh, M=mi, S=ss, rc=rc)
     if (rc /= WRFU_SUCCESS) stop 'WRFU_TimeSet'
     call WRFU_TimeIntervalSet(elapsed, S=esec, rc=rc)
     if (rc /= WRFU_SUCCESS) stop 'WRFU_TimeIntervalSet'
     tcur = tstart + elapsed

     ! frame/module_domain.F:2159-2165
     CALL WRFU_TimeGet( tcur, dayOfYear_r8=currentDayOfYearR8, rc=rc )
     if (rc /= WRFU_SUCCESS) stop 'WRFU_TimeGet dayOfYear_r8'
     julian = REAL( currentDayOfYearR8 ) - 1.0

     ! frame/module_domain.F:2439-2447
     CALL WRFU_TimeGet( tstart, H=hr, M=mn, S=sec, MS=ms, rc=rc)
     if (rc /= WRFU_SUCCESS) stop 'WRFU_TimeGet H/M/S'
     gmt=hr+real(mn)/60.+real(sec)/3600.+real(ms)/(1000*3600)

     ! frame/module_domain.F:2178-2191
     since = tcur - tstart
     CALL WRFU_TimeIntervalGet( since, D=days, S=seconds, Sn=Sn, Sd=Sd, rc=rc )
     if (rc /= WRFU_SUCCESS) stop 'WRFU_TimeIntervalGet'
     xtime = ( REAL( days ) * 24. * 60. ) + ( REAL( seconds ) / 60. )
     IF ( Sd /= 0 ) THEN
        xtime = xtime + ( ( REAL( Sn ) / REAL( Sd ) ) / 60. )
     ENDIF

     ! phys/module_radiation_driver.F:1188-1189 and :1206-1208
     CALL radconst(XTIME,DECLIN,SOLCON,JULIAN,DEGRAD,DPD)
     coszen = 0.; hrang = 0.
     call calc_coszen(1,ncol,1,1,1,ncol,1,1, &
                      julian,xtime+radt*0.5,gmt, &
                      declin,degrad,xlon,xlat,coszen,hrang)
     write(11) julian, xtime, gmt, declin, solcon, coszen, hrang
  end do
  close(10)
  close(11)
end program run_coszen
