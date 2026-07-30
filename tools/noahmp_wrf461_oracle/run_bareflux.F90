! ==================================================================================================
! run_bareflux -- bitwise oracle driver for WRF v4.6.1 Noah-MP BARE_FLUX.
!
! Reads one case per line from stdin.  Every REAL is transported as an 8-digit
! hexadecimal IEEE-754 binary32 word so that no decimal rounding can sit between
! the case generator and the compiled WRF routine.  Every REAL that comes back
! is emitted the same way.
!
! Line layout (whitespace separated, in this exact order):
!
!   caseid                 (character, no spaces)
!   iopt_sfc iopt_stc      (integers -- pinned identity assertion is done here)
!   isnow ivgtyp iloc jloc iurban      (integers; iurban 0/1 -> parameters%urban_flag)
!   53 hex words:
!     dt sag lwdn ur uu vv sfctmp thair qair eair rhoair snowh zlvl zpd z0m
!     fsno emg rsurf lathea gamma rhsur q2 pahb dx dz8w qc psfc sfcprs
!     tgb cm ch qsfc
!     dzsnso(-2..4)  stc(-2..4)  df(-2..4)
!
! Output line: caseid followed by 13 hex words:
!     tgb cm ch qsfc tauxb tauyb irb shb evb ghb t2mb q2b ehb2
!
! The routine under test is reached through the visibility patch only; nothing
! about its body is altered.  See visibility_patch_bareflux.py.
! ==================================================================================================

program run_bareflux

  use module_sf_noahmplsm

  implicit none

  integer, parameter :: NSNOW = 3
  integer, parameter :: NSOIL = 4
  integer, parameter :: NREAL_IN = 53

  type (noahmp_parameters) :: parameters

  character(len=64)   :: caseid
  character(len=64)   :: mode
  character(len=8192) :: line
  character(len=8)    :: word
  real :: tesat, esw, esi, desw, desi

  integer :: iopt_sfc_in, iopt_stc_in
  integer :: isnow, ivgtyp, iloc, jloc, iurban
  integer :: ios, k, pos, nread

  real :: rin(NREAL_IN)

  real :: dt, sag, lwdn, ur, uu, vv, sfctmp, thair, qair, eair, rhoair
  real :: snowh, zlvl, zpd, z0m, fsno, emg, rsurf, lathea, gamma_, rhsur
  real :: q2, pahb, dx, dz8w, qc, psfc, sfcprs
  real :: tgb, cm, ch, qsfc
  real :: dzsnso(-NSNOW+1:NSOIL), stc(-NSNOW+1:NSOIL), df(-NSNOW+1:NSOIL)
  real :: tauxb, tauyb, irb, shb, evb, ghb, t2mb, q2b, ehb2

  ! ---------------------------------------------------------------------------
  ! Mode "esat": read one hex float32 T (degrees C) per line and print
  ! ESW ESI DESW DESI.  This exposes ESAT on its own so the CUDA half can be
  ! checked against the Fortran directly rather than against a Python
  ! transcription of it.
  ! ---------------------------------------------------------------------------
  if (command_argument_count() >= 1) then
     call get_command_argument(1, mode)
     if (trim(mode) == 'esat') then
        do
           read(*,'(A)',iostat=ios) line
           if (ios /= 0) exit
           if (len_trim(line) == 0) exit
           pos = 1
           call next_token(line, pos, word)
           tesat = hex2real(word)
           call ESAT(tesat, esw, esi, desw, desi)
           write(*,'(5(1X,Z8.8))') real2hex(tesat), real2hex(esw), &
                real2hex(esi), real2hex(desw), real2hex(desi)
        end do
        stop
     end if
  end if

  ! ---------------------------------------------------------------------------
  ! Pin the option identity that the WRF Registry defaults select.  Everything
  ! not on this list is dead for BARE_FLUX and is asserted off, never ported.
  !   opt_sfc = 1  -> M-O (SFCDIF1); SFCDIF2 is dead
  !   opt_stc = 1  -> semi-implicit; the opt_stc==3 snow-melt blend is dead
  ! The remaining noahmp_options arguments do not reach BARE_FLUX at all; they
  ! are set to the Registry defaults for completeness.
  ! ---------------------------------------------------------------------------

  do
     read(*,'(A)',iostat=ios) line
     if (ios /= 0) exit
     if (len_trim(line) == 0) exit

     pos = 1
     call next_token(line, pos, caseid)

     call next_int(line, pos, iopt_sfc_in)
     call next_int(line, pos, iopt_stc_in)
     if (iopt_sfc_in /= 1) stop 'run_bareflux: opt_sfc must be 1 (pinned identity)'
     if (iopt_stc_in /= 1) stop 'run_bareflux: opt_stc must be 1 (pinned identity)'

     ! WRF v4.6.1 Registry.EM_COMMON defaults, in declaration order:
     !   dveg opt_crs opt_btr opt_run opt_sfc opt_frz opt_inf opt_rad opt_alb
     !   opt_snf opt_tbot opt_stc opt_rsf opt_soil opt_pedo opt_crop opt_irr
     !   opt_irrm opt_infdv opt_tdrn
     call noahmp_options(4, 1, 1, 3, iopt_sfc_in, 1,          &
                         1, 3, 2, 1, 2, iopt_stc_in,          &
                         1, 1, 1, 0, 0, 0,                    &
                         0, 0)

     call next_int(line, pos, isnow)
     call next_int(line, pos, ivgtyp)
     call next_int(line, pos, iloc)
     call next_int(line, pos, jloc)
     call next_int(line, pos, iurban)

     do k = 1, NREAL_IN
        call next_token(line, pos, word)
        rin(k) = hex2real(word)
     end do

     dt     = rin( 1); sag    = rin( 2); lwdn   = rin( 3); ur     = rin( 4)
     uu     = rin( 5); vv     = rin( 6); sfctmp = rin( 7); thair  = rin( 8)
     qair   = rin( 9); eair   = rin(10); rhoair = rin(11); snowh  = rin(12)
     zlvl   = rin(13); zpd    = rin(14); z0m    = rin(15); fsno   = rin(16)
     emg    = rin(17); rsurf  = rin(18); lathea = rin(19); gamma_ = rin(20)
     rhsur  = rin(21); q2     = rin(22); pahb   = rin(23); dx     = rin(24)
     dz8w   = rin(25); qc     = rin(26); psfc   = rin(27); sfcprs = rin(28)
     tgb    = rin(29); cm     = rin(30); ch     = rin(31); qsfc   = rin(32)

     nread = 32
     do k = -NSNOW+1, NSOIL
        nread = nread + 1
        dzsnso(k) = rin(nread)
     end do
     do k = -NSNOW+1, NSOIL
        nread = nread + 1
        stc(k) = rin(nread)
     end do
     do k = -NSNOW+1, NSOIL
        nread = nread + 1
        df(k) = rin(nread)
     end do

     parameters%urban_flag = (iurban /= 0)

     call BARE_FLUX (parameters, NSNOW, NSOIL, isnow, dt, sag,       &
                     lwdn, ur, uu, vv, sfctmp,                       &
                     thair, qair, eair, rhoair, snowh,               &
                     dzsnso, zlvl, zpd, z0m, fsno,                   &
                     emg, stc, df, rsurf, lathea,                    &
                     gamma_, rhsur, iloc, jloc, q2, pahb,            &
                     tgb, cm, ch,                                    &
                     tauxb, tauyb, irb, shb, evb,                    &
                     ghb, t2mb, dx, dz8w, ivgtyp,                    &
                     qc, qsfc, psfc,                                 &
                     sfcprs, q2b, ehb2)

     write(*,'(A,13(1X,Z8.8))') trim(caseid),                        &
          real2hex(tgb),   real2hex(cm),    real2hex(ch),             &
          real2hex(qsfc),  real2hex(tauxb), real2hex(tauyb),          &
          real2hex(irb),   real2hex(shb),   real2hex(evb),            &
          real2hex(ghb),   real2hex(t2mb),  real2hex(q2b),            &
          real2hex(ehb2)
  end do

contains

  subroutine next_token(str, pos, tok)
    character(len=*), intent(in)    :: str
    integer,          intent(inout) :: pos
    character(len=*), intent(out)   :: tok
    integer :: i, j
    i = pos
    do while (i <= len(str))
       if (str(i:i) /= ' ' .and. str(i:i) /= char(9)) exit
       i = i + 1
    end do
    j = i
    do while (j <= len(str))
       if (str(j:j) == ' ' .or. str(j:j) == char(9)) exit
       j = j + 1
    end do
    tok = str(i:j-1)
    pos = j
  end subroutine next_token

  subroutine next_int(str, pos, val)
    character(len=*), intent(in)    :: str
    integer,          intent(inout) :: pos
    integer,          intent(out)   :: val
    character(len=64) :: tok
    call next_token(str, pos, tok)
    read(tok,*) val
  end subroutine next_int

  real function hex2real(hexword)
    character(len=*), intent(in) :: hexword
    integer :: ibits
    read(hexword,'(Z8)') ibits
    hex2real = transfer(ibits, hex2real)
  end function hex2real

  integer function real2hex(x)
    real, intent(in) :: x
    real2hex = transfer(x, real2hex)
  end function real2hex

end program run_bareflux
