! Auto-assembled dump driver for the WRF v4.6.1 RRTMG SW
! coefficient modules.  Dumps every kg-module array that the WRF init path
! initializes (the RRTMG_SW_DATA READ lists plus every cmbgb reduction
! target, including the ABSA/ABSB equivalence views), after running the
! exact WRF init entry point.  Output is a tagged stream of
! (name, dtype, rank, dims, payload) entries; with the oracle's
! -fconvert=big-endian build every integer and real in the stream is
! big-endian.  See tools/rrtmg_wrf461_oracle/build.sh.
program coeffs_dump_sw
   use module_ra_rrtmg_sw, only: rrtmg_swinit
   use dump_kit, only: check_kinds
   implicit none
   integer, parameter :: kme = 50
   real, parameter :: p_top = 5000.0
   integer :: u
   character(len=1024) :: outfile
   if (command_argument_count() /= 1) then
      write(*, '(A)') 'usage: coeffs_dump_sw OUTFILE'
      error stop 2
   end if
   call get_command_argument(1, outfile)
   call check_kinds()
   call rrtmg_swinit(.true., 1, 2, 1, 2, 1, kme, 1, 2, 1, 2, 1, kme, 1, 1, 1, 1, 1, kme - 1)
   open(newunit=u, file=trim(outfile), form='unformatted', &
        access='stream', status='replace')
   call dump_rrsw_kg16(u)
   call dump_rrsw_kg17(u)
   call dump_rrsw_kg18(u)
   call dump_rrsw_kg19(u)
   call dump_rrsw_kg20(u)
   call dump_rrsw_kg21(u)
   call dump_rrsw_kg22(u)
   call dump_rrsw_kg23(u)
   call dump_rrsw_kg24(u)
   call dump_rrsw_kg25(u)
   call dump_rrsw_kg26(u)
   call dump_rrsw_kg27(u)
   call dump_rrsw_kg28(u)
   call dump_rrsw_kg29(u)
   close(u)
   write(*, '(A)') 'coeffs_dump_sw: wrote ' // trim(outfile)
end program coeffs_dump_sw

subroutine dump_rrsw_kg16(u)
   use dump_kit
   use rrsw_kg16
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg16/rayl', rayl)
   call wr0(u, 'rrsw_kg16/strrat1', strrat1)
   call wi0(u, 'rrsw_kg16/layreffr', layreffr)
   call wr4(u, 'rrsw_kg16/kao', kao)
   call wr3(u, 'rrsw_kg16/kbo', kbo)
   call wr2(u, 'rrsw_kg16/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg16/forrefo', forrefo)
   call wr1(u, 'rrsw_kg16/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg16/ka', ka)
   call wr2(u, 'rrsw_kg16/absa', absa)
   call wr3(u, 'rrsw_kg16/kb', kb)
   call wr2(u, 'rrsw_kg16/absb', absb)
   call wr2(u, 'rrsw_kg16/selfref', selfref)
   call wr2(u, 'rrsw_kg16/forref', forref)
   call wr1(u, 'rrsw_kg16/sfluxref', sfluxref)
end subroutine dump_rrsw_kg16

subroutine dump_rrsw_kg17(u)
   use dump_kit
   use rrsw_kg17
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg17/rayl', rayl)
   call wr0(u, 'rrsw_kg17/strrat', strrat)
   call wi0(u, 'rrsw_kg17/layreffr', layreffr)
   call wr4(u, 'rrsw_kg17/kao', kao)
   call wr4(u, 'rrsw_kg17/kbo', kbo)
   call wr2(u, 'rrsw_kg17/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg17/forrefo', forrefo)
   call wr2(u, 'rrsw_kg17/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg17/ka', ka)
   call wr2(u, 'rrsw_kg17/absa', absa)
   call wr4(u, 'rrsw_kg17/kb', kb)
   call wr2(u, 'rrsw_kg17/absb', absb)
   call wr2(u, 'rrsw_kg17/selfref', selfref)
   call wr2(u, 'rrsw_kg17/forref', forref)
   call wr2(u, 'rrsw_kg17/sfluxref', sfluxref)
end subroutine dump_rrsw_kg17

subroutine dump_rrsw_kg18(u)
   use dump_kit
   use rrsw_kg18
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg18/rayl', rayl)
   call wr0(u, 'rrsw_kg18/strrat', strrat)
   call wi0(u, 'rrsw_kg18/layreffr', layreffr)
   call wr4(u, 'rrsw_kg18/kao', kao)
   call wr3(u, 'rrsw_kg18/kbo', kbo)
   call wr2(u, 'rrsw_kg18/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg18/forrefo', forrefo)
   call wr2(u, 'rrsw_kg18/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg18/ka', ka)
   call wr2(u, 'rrsw_kg18/absa', absa)
   call wr3(u, 'rrsw_kg18/kb', kb)
   call wr2(u, 'rrsw_kg18/absb', absb)
   call wr2(u, 'rrsw_kg18/selfref', selfref)
   call wr2(u, 'rrsw_kg18/forref', forref)
   call wr2(u, 'rrsw_kg18/sfluxref', sfluxref)
end subroutine dump_rrsw_kg18

subroutine dump_rrsw_kg19(u)
   use dump_kit
   use rrsw_kg19
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg19/rayl', rayl)
   call wr0(u, 'rrsw_kg19/strrat', strrat)
   call wi0(u, 'rrsw_kg19/layreffr', layreffr)
   call wr4(u, 'rrsw_kg19/kao', kao)
   call wr3(u, 'rrsw_kg19/kbo', kbo)
   call wr2(u, 'rrsw_kg19/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg19/forrefo', forrefo)
   call wr2(u, 'rrsw_kg19/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg19/ka', ka)
   call wr2(u, 'rrsw_kg19/absa', absa)
   call wr3(u, 'rrsw_kg19/kb', kb)
   call wr2(u, 'rrsw_kg19/absb', absb)
   call wr2(u, 'rrsw_kg19/selfref', selfref)
   call wr2(u, 'rrsw_kg19/forref', forref)
   call wr2(u, 'rrsw_kg19/sfluxref', sfluxref)
end subroutine dump_rrsw_kg19

subroutine dump_rrsw_kg20(u)
   use dump_kit
   use rrsw_kg20
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg20/rayl', rayl)
   call wi0(u, 'rrsw_kg20/layreffr', layreffr)
   call wr1(u, 'rrsw_kg20/absch4o', absch4o)
   call wr3(u, 'rrsw_kg20/kao', kao)
   call wr3(u, 'rrsw_kg20/kbo', kbo)
   call wr2(u, 'rrsw_kg20/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg20/forrefo', forrefo)
   call wr1(u, 'rrsw_kg20/sfluxrefo', sfluxrefo)
   call wr3(u, 'rrsw_kg20/ka', ka)
   call wr2(u, 'rrsw_kg20/absa', absa)
   call wr3(u, 'rrsw_kg20/kb', kb)
   call wr2(u, 'rrsw_kg20/absb', absb)
   call wr2(u, 'rrsw_kg20/selfref', selfref)
   call wr2(u, 'rrsw_kg20/forref', forref)
   call wr1(u, 'rrsw_kg20/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg20/absch4', absch4)
end subroutine dump_rrsw_kg20

subroutine dump_rrsw_kg21(u)
   use dump_kit
   use rrsw_kg21
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg21/rayl', rayl)
   call wr0(u, 'rrsw_kg21/strrat', strrat)
   call wi0(u, 'rrsw_kg21/layreffr', layreffr)
   call wr4(u, 'rrsw_kg21/kao', kao)
   call wr4(u, 'rrsw_kg21/kbo', kbo)
   call wr2(u, 'rrsw_kg21/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg21/forrefo', forrefo)
   call wr2(u, 'rrsw_kg21/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg21/ka', ka)
   call wr2(u, 'rrsw_kg21/absa', absa)
   call wr4(u, 'rrsw_kg21/kb', kb)
   call wr2(u, 'rrsw_kg21/absb', absb)
   call wr2(u, 'rrsw_kg21/selfref', selfref)
   call wr2(u, 'rrsw_kg21/forref', forref)
   call wr2(u, 'rrsw_kg21/sfluxref', sfluxref)
end subroutine dump_rrsw_kg21

subroutine dump_rrsw_kg22(u)
   use dump_kit
   use rrsw_kg22
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg22/rayl', rayl)
   call wr0(u, 'rrsw_kg22/strrat', strrat)
   call wi0(u, 'rrsw_kg22/layreffr', layreffr)
   call wr4(u, 'rrsw_kg22/kao', kao)
   call wr3(u, 'rrsw_kg22/kbo', kbo)
   call wr2(u, 'rrsw_kg22/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg22/forrefo', forrefo)
   call wr2(u, 'rrsw_kg22/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg22/ka', ka)
   call wr2(u, 'rrsw_kg22/absa', absa)
   call wr3(u, 'rrsw_kg22/kb', kb)
   call wr2(u, 'rrsw_kg22/absb', absb)
   call wr2(u, 'rrsw_kg22/selfref', selfref)
   call wr2(u, 'rrsw_kg22/forref', forref)
   call wr2(u, 'rrsw_kg22/sfluxref', sfluxref)
end subroutine dump_rrsw_kg22

subroutine dump_rrsw_kg23(u)
   use dump_kit
   use rrsw_kg23
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrsw_kg23/raylo', raylo)
   call wr0(u, 'rrsw_kg23/givfac', givfac)
   call wi0(u, 'rrsw_kg23/layreffr', layreffr)
   call wr3(u, 'rrsw_kg23/kao', kao)
   call wr2(u, 'rrsw_kg23/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg23/forrefo', forrefo)
   call wr1(u, 'rrsw_kg23/sfluxrefo', sfluxrefo)
   call wr3(u, 'rrsw_kg23/ka', ka)
   call wr2(u, 'rrsw_kg23/absa', absa)
   call wr2(u, 'rrsw_kg23/selfref', selfref)
   call wr2(u, 'rrsw_kg23/forref', forref)
   call wr1(u, 'rrsw_kg23/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg23/rayl', rayl)
end subroutine dump_rrsw_kg23

subroutine dump_rrsw_kg24(u)
   use dump_kit
   use rrsw_kg24
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrsw_kg24/raylao', raylao)
   call wr1(u, 'rrsw_kg24/raylbo', raylbo)
   call wr0(u, 'rrsw_kg24/strrat', strrat)
   call wi0(u, 'rrsw_kg24/layreffr', layreffr)
   call wr1(u, 'rrsw_kg24/abso3ao', abso3ao)
   call wr1(u, 'rrsw_kg24/abso3bo', abso3bo)
   call wr4(u, 'rrsw_kg24/kao', kao)
   call wr3(u, 'rrsw_kg24/kbo', kbo)
   call wr2(u, 'rrsw_kg24/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg24/forrefo', forrefo)
   call wr2(u, 'rrsw_kg24/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg24/ka', ka)
   call wr2(u, 'rrsw_kg24/absa', absa)
   call wr3(u, 'rrsw_kg24/kb', kb)
   call wr2(u, 'rrsw_kg24/absb', absb)
   call wr2(u, 'rrsw_kg24/selfref', selfref)
   call wr2(u, 'rrsw_kg24/forref', forref)
   call wr2(u, 'rrsw_kg24/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg24/abso3a', abso3a)
   call wr1(u, 'rrsw_kg24/abso3b', abso3b)
   call wr2(u, 'rrsw_kg24/rayla', rayla)
   call wr1(u, 'rrsw_kg24/raylb', raylb)
end subroutine dump_rrsw_kg24

subroutine dump_rrsw_kg25(u)
   use dump_kit
   use rrsw_kg25
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrsw_kg25/raylo', raylo)
   call wi0(u, 'rrsw_kg25/layreffr', layreffr)
   call wr1(u, 'rrsw_kg25/abso3ao', abso3ao)
   call wr1(u, 'rrsw_kg25/abso3bo', abso3bo)
   call wr3(u, 'rrsw_kg25/kao', kao)
   call wr1(u, 'rrsw_kg25/sfluxrefo', sfluxrefo)
   call wr3(u, 'rrsw_kg25/ka', ka)
   call wr2(u, 'rrsw_kg25/absa', absa)
   call wr1(u, 'rrsw_kg25/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg25/abso3a', abso3a)
   call wr1(u, 'rrsw_kg25/abso3b', abso3b)
   call wr1(u, 'rrsw_kg25/rayl', rayl)
end subroutine dump_rrsw_kg25

subroutine dump_rrsw_kg26(u)
   use dump_kit
   use rrsw_kg26
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrsw_kg26/raylo', raylo)
   call wr1(u, 'rrsw_kg26/sfluxrefo', sfluxrefo)
   call wr1(u, 'rrsw_kg26/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg26/rayl', rayl)
end subroutine dump_rrsw_kg26

subroutine dump_rrsw_kg27(u)
   use dump_kit
   use rrsw_kg27
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrsw_kg27/raylo', raylo)
   call wr0(u, 'rrsw_kg27/scalekur', scalekur)
   call wi0(u, 'rrsw_kg27/layreffr', layreffr)
   call wr3(u, 'rrsw_kg27/kao', kao)
   call wr3(u, 'rrsw_kg27/kbo', kbo)
   call wr1(u, 'rrsw_kg27/sfluxrefo', sfluxrefo)
   call wr3(u, 'rrsw_kg27/ka', ka)
   call wr2(u, 'rrsw_kg27/absa', absa)
   call wr3(u, 'rrsw_kg27/kb', kb)
   call wr2(u, 'rrsw_kg27/absb', absb)
   call wr1(u, 'rrsw_kg27/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg27/rayl', rayl)
end subroutine dump_rrsw_kg27

subroutine dump_rrsw_kg28(u)
   use dump_kit
   use rrsw_kg28
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg28/rayl', rayl)
   call wr0(u, 'rrsw_kg28/strrat', strrat)
   call wi0(u, 'rrsw_kg28/layreffr', layreffr)
   call wr4(u, 'rrsw_kg28/kao', kao)
   call wr4(u, 'rrsw_kg28/kbo', kbo)
   call wr2(u, 'rrsw_kg28/sfluxrefo', sfluxrefo)
   call wr4(u, 'rrsw_kg28/ka', ka)
   call wr2(u, 'rrsw_kg28/absa', absa)
   call wr4(u, 'rrsw_kg28/kb', kb)
   call wr2(u, 'rrsw_kg28/absb', absb)
   call wr2(u, 'rrsw_kg28/sfluxref', sfluxref)
end subroutine dump_rrsw_kg28

subroutine dump_rrsw_kg29(u)
   use dump_kit
   use rrsw_kg29
   implicit none
   integer, intent(in) :: u
   call wr0(u, 'rrsw_kg29/rayl', rayl)
   call wi0(u, 'rrsw_kg29/layreffr', layreffr)
   call wr1(u, 'rrsw_kg29/absh2oo', absh2oo)
   call wr1(u, 'rrsw_kg29/absco2o', absco2o)
   call wr3(u, 'rrsw_kg29/kao', kao)
   call wr3(u, 'rrsw_kg29/kbo', kbo)
   call wr2(u, 'rrsw_kg29/selfrefo', selfrefo)
   call wr2(u, 'rrsw_kg29/forrefo', forrefo)
   call wr1(u, 'rrsw_kg29/sfluxrefo', sfluxrefo)
   call wr3(u, 'rrsw_kg29/ka', ka)
   call wr2(u, 'rrsw_kg29/absa', absa)
   call wr3(u, 'rrsw_kg29/kb', kb)
   call wr2(u, 'rrsw_kg29/absb', absb)
   call wr2(u, 'rrsw_kg29/selfref', selfref)
   call wr2(u, 'rrsw_kg29/forref', forref)
   call wr1(u, 'rrsw_kg29/sfluxref', sfluxref)
   call wr1(u, 'rrsw_kg29/absh2o', absh2o)
   call wr1(u, 'rrsw_kg29/absco2', absco2)
end subroutine dump_rrsw_kg29

