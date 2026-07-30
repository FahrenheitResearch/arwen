! ==================================================================================================
! run_vegprecip -- bitwise oracle driver for WRF v4.6.1 Noah-MP PHENOLOGY and PRECIP_HEAT.
!
! Reads one case per line from stdin.  Every REAL is transported as an 8-digit
! hexadecimal IEEE-754 binary32 word so that no decimal rounding can sit between
! the case generator and the compiled WRF routine.  Every REAL that comes back
! is emitted the same way.
!
! The first token selects the leaf.
!
! PHEN line layout (whitespace separated, in this exact order):
!
!   PHEN caseid
!   dveg croptype vegtyp yearlen pgs iswater isbarren isice iurban   (9 integers)
!   34 hex words:
!     snowh tv lat julian troot lai sai hvt hvb tmin
!     laim(1..12)
!     saim(1..12)
!
!   Output: caseid followed by 6 hex words:  lai sai elai esai igs fb
!   (lai and sai are INTENT(INOUT); the driver reports what came back.)
!
! PRCP line layout:
!
!   PRCP caseid
!   iloc jloc vegtyp ist                                             (4 integers)
!   17 hex words:
!     dt uu vv elai esai fveg bdfall rain snow fp canliq canice tv sfctmp tg
!     ch2op tfrz_probe
!
!   tfrz_probe is not passed to the routine; the driver echoes it back so the
!   case generator can assert that its notion of TFRZ matches the module's.
!
!   Output: caseid followed by 16 hex words:
!     canliq canice qintr qdripr qthror qints qdrips qthros
!     pahv pahg pahb qrain qsnow snowhin fwet cmc
!
! The routines under test are reached through the visibility patch only; nothing
! about their bodies is altered.  See visibility_patch_vegprecip.py.
!
! Pinned option identity (WRF Registry defaults).  Only DVEG reaches PHENOLOGY;
! PRECIP_HEAT reads no option variable at all.  The driver asserts the identity
! rather than assuming it:
!   dveg     = 4  -> table LAI/SAI; the DVEG==7/8/9 "use input LAI" block is dead
!   opt_crop = 0  -> croptype is identically 0, so the crop/PGS disjunct is dead
! A case may carry dveg /= 4 only when it is explicitly marked as a negative
! control (caseid beginning with "NEGCTL"); the fixture builder refuses to write
! such a row into the CSV.
! ==================================================================================================

program run_vegprecip

  use module_sf_noahmplsm

  implicit none

  type (noahmp_parameters) :: parameters

  character(len=16384) :: line
  character(len=32)    :: leaf
  character(len=64)    :: caseid
  character(len=64)    :: tok

  integer :: ios, k

  ! --- PHENOLOGY -----------------------------------------------------------
  integer :: p_dveg, p_croptype, p_vegtyp, p_yearlen, p_pgs
  integer :: p_iswater, p_isbarren, p_isice, p_iurban
  real    :: p_snowh, p_tv, p_lat, p_julian, p_troot, p_lai, p_sai
  real    :: p_hvt, p_hvb, p_tmin
  real    :: p_laim(12), p_saim(12)
  real    :: p_elai, p_esai, p_igs, p_fb

  ! --- PRECIP_HEAT ---------------------------------------------------------
  integer :: q_iloc, q_jloc, q_vegtyp, q_ist
  real    :: q_dt, q_uu, q_vv, q_elai, q_esai, q_fveg, q_bdfall
  real    :: q_rain, q_snow, q_fp, q_canliq, q_canice
  real    :: q_tv, q_sfctmp, q_tg, q_ch2op, q_tfrz_probe
  real    :: q_qintr, q_qdripr, q_qthror, q_qints, q_qdrips, q_qthros
  real    :: q_pahv, q_pahg, q_pahb, q_qrain, q_qsnow, q_snowhin
  real    :: q_fwet, q_cmc

  do
     read(*,'(A)',iostat=ios) line
     if (ios /= 0) exit
     if (len_trim(line) == 0) cycle
     if (line(1:1) == '#') cycle

     call token(line, 1, leaf)

     select case (trim(leaf))

     ! ----------------------------------------------------------------------
     case ('PHEN')
        call token(line, 2, caseid)
        call token(line, 3,  tok); read(tok,*) p_dveg
        call token(line, 4,  tok); read(tok,*) p_croptype
        call token(line, 5,  tok); read(tok,*) p_vegtyp
        call token(line, 6,  tok); read(tok,*) p_yearlen
        call token(line, 7,  tok); read(tok,*) p_pgs
        call token(line, 8,  tok); read(tok,*) p_iswater
        call token(line, 9,  tok); read(tok,*) p_isbarren
        call token(line, 10, tok); read(tok,*) p_isice
        call token(line, 11, tok); read(tok,*) p_iurban

        p_snowh  = hexword(line, 12)
        p_tv     = hexword(line, 13)
        p_lat    = hexword(line, 14)
        p_julian = hexword(line, 15)
        p_troot  = hexword(line, 16)
        p_lai    = hexword(line, 17)
        p_sai    = hexword(line, 18)
        p_hvt    = hexword(line, 19)
        p_hvb    = hexword(line, 20)
        p_tmin   = hexword(line, 21)
        do k = 1, 12
           p_laim(k) = hexword(line, 21 + k)
           p_saim(k) = hexword(line, 33 + k)
        end do

        ! Pinned option identity.  DVEG is the module variable PHENOLOGY reads.
        if (p_dveg /= 4 .and. caseid(1:6) /= 'NEGCTL') then
           write(0,'(A)') 'run_vegprecip: dveg /= 4 outside a NEGCTL case'
           stop 2
        end if
        DVEG = p_dveg

        parameters%ISWATER    = p_iswater
        parameters%ISBARREN   = p_isbarren
        parameters%ISICE      = p_isice
        parameters%URBAN_FLAG = (p_iurban /= 0)
        parameters%HVT        = p_hvt
        parameters%HVB        = p_hvb
        parameters%TMIN       = p_tmin
        parameters%LAIM       = p_laim
        parameters%SAIM       = p_saim

        call PHENOLOGY (parameters, p_vegtyp, p_croptype, p_snowh, p_tv,   &
                        p_lat, p_yearlen, p_julian,                        &
                        p_lai, p_sai, p_troot, p_elai, p_esai, p_igs,      &
                        p_pgs, p_fb)

        write(*,'(A,6(1X,Z8.8))') trim(caseid),                            &
             real2hex(p_lai),  real2hex(p_sai),  real2hex(p_elai),         &
             real2hex(p_esai), real2hex(p_igs),  real2hex(p_fb)

     ! ----------------------------------------------------------------------
     case ('PRCP')
        call token(line, 2, caseid)
        call token(line, 3, tok); read(tok,*) q_iloc
        call token(line, 4, tok); read(tok,*) q_jloc
        call token(line, 5, tok); read(tok,*) q_vegtyp
        call token(line, 6, tok); read(tok,*) q_ist

        q_dt         = hexword(line, 7)
        q_uu         = hexword(line, 8)
        q_vv         = hexword(line, 9)
        q_elai       = hexword(line, 10)
        q_esai       = hexword(line, 11)
        q_fveg       = hexword(line, 12)
        q_bdfall     = hexword(line, 13)
        q_rain       = hexword(line, 14)
        q_snow       = hexword(line, 15)
        q_fp         = hexword(line, 16)
        q_canliq     = hexword(line, 17)
        q_canice     = hexword(line, 18)
        q_tv         = hexword(line, 19)
        q_sfctmp     = hexword(line, 20)
        q_tg         = hexword(line, 21)
        q_ch2op      = hexword(line, 22)
        q_tfrz_probe = hexword(line, 23)

        if (q_tfrz_probe /= TFRZ) then
           write(0,'(A)') 'run_vegprecip: TFRZ probe disagrees with the module'
           stop 3
        end if

        parameters%CH2OP = q_ch2op

        call PRECIP_HEAT (parameters, q_iloc, q_jloc, q_vegtyp, q_dt,      &
                          q_uu, q_vv, q_elai, q_esai, q_fveg, q_ist,       &
                          q_bdfall, q_rain, q_snow, q_fp,                  &
                          q_canliq, q_canice, q_tv, q_sfctmp, q_tg,        &
                          q_qintr, q_qdripr, q_qthror,                     &
                          q_qints, q_qdrips, q_qthros,                     &
                          q_pahv, q_pahg, q_pahb, q_qrain, q_qsnow,        &
                          q_snowhin, q_fwet, q_cmc)

        write(*,'(A,16(1X,Z8.8))') trim(caseid),                           &
             real2hex(q_canliq), real2hex(q_canice),                       &
             real2hex(q_qintr),  real2hex(q_qdripr), real2hex(q_qthror),   &
             real2hex(q_qints),  real2hex(q_qdrips), real2hex(q_qthros),   &
             real2hex(q_pahv),   real2hex(q_pahg),   real2hex(q_pahb),     &
             real2hex(q_qrain),  real2hex(q_qsnow),  real2hex(q_snowhin),  &
             real2hex(q_fwet),   real2hex(q_cmc)

     case default
        write(0,'(A)') 'run_vegprecip: unknown leaf tag '//trim(leaf)
        stop 4
     end select
  end do

contains

  ! Return the n-th whitespace-separated token of str.
  subroutine token(str, n, out)
    character(len=*), intent(in)  :: str
    integer,          intent(in)  :: n
    character(len=*), intent(out) :: out
    integer :: i, count, first, last
    i = 1
    count = 0
    out = ' '
    do
       do while (i <= len(str))
          if (str(i:i) /= ' ' .and. str(i:i) /= char(9)) exit
          i = i + 1
       end do
       if (i > len(str)) return
       first = i
       do while (i <= len(str))
          if (str(i:i) == ' ' .or. str(i:i) == char(9)) exit
          i = i + 1
       end do
       last = i - 1
       count = count + 1
       if (count == n) then
          out = str(first:last)
          return
       end if
    end do
  end subroutine token

  real function hexword(str, n)
    character(len=*), intent(in) :: str
    integer,          intent(in) :: n
    character(len=64) :: w
    integer :: ibits
    call token(str, n, w)
    read(w,'(Z8)') ibits
    hexword = transfer(ibits, hexword)
  end function hexword

  integer function real2hex(x)
    real, intent(in) :: x
    real2hex = transfer(x, real2hex)
  end function real2hex

end program run_vegprecip
