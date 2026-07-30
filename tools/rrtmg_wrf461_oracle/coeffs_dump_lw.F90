! Auto-assembled dump driver for the WRF v4.6.1 RRTMG LW
! coefficient modules.  Dumps every kg-module array that the WRF init path
! initializes (the RRTMG_LW_DATA READ lists plus every cmbgb reduction
! target, including the ABSA/ABSB equivalence views), after running the
! exact WRF init entry point.  Output is a tagged stream of
! (name, dtype, rank, dims, payload) entries; with the oracle's
! -fconvert=big-endian build every integer and real in the stream is
! big-endian.  See tools/rrtmg_wrf461_oracle/build.sh.
program coeffs_dump_lw
   use module_ra_rrtmg_lw, only: rrtmg_lwinit
   use dump_kit, only: check_kinds
   implicit none
   integer, parameter :: kme = 50
   real, parameter :: p_top = 5000.0
   integer :: u
   character(len=1024) :: outfile
   if (command_argument_count() /= 1) then
      write(*, '(A)') 'usage: coeffs_dump_lw OUTFILE'
      error stop 2
   end if
   call get_command_argument(1, outfile)
   call check_kinds()
   call rrtmg_lwinit(p_top, .true., 1, 2, 1, 2, 1, kme, 1, 2, 1, 2, 1, kme, 1, 1, 1, 1, 1, kme - 1)
   open(newunit=u, file=trim(outfile), form='unformatted', &
        access='stream', status='replace')
   call dump_rrlw_kg01(u)
   call dump_rrlw_kg02(u)
   call dump_rrlw_kg03(u)
   call dump_rrlw_kg04(u)
   call dump_rrlw_kg05(u)
   call dump_rrlw_kg06(u)
   call dump_rrlw_kg07(u)
   call dump_rrlw_kg08(u)
   call dump_rrlw_kg09(u)
   call dump_rrlw_kg10(u)
   call dump_rrlw_kg11(u)
   call dump_rrlw_kg12(u)
   call dump_rrlw_kg13(u)
   call dump_rrlw_kg14(u)
   call dump_rrlw_kg15(u)
   call dump_rrlw_kg16(u)
   close(u)
   write(*, '(A)') 'coeffs_dump_lw: wrote ' // trim(outfile)
end program coeffs_dump_lw

subroutine dump_rrlw_kg01(u)
   use dump_kit
   use rrlw_kg01
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg01/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg01/fracrefbo', fracrefbo)
   call wr3(u, 'rrlw_kg01/kao', kao)
   call wr3(u, 'rrlw_kg01/kbo', kbo)
   call wr2(u, 'rrlw_kg01/kao_mn2', kao_mn2)
   call wr2(u, 'rrlw_kg01/kbo_mn2', kbo_mn2)
   call wr2(u, 'rrlw_kg01/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg01/forrefo', forrefo)
   call wr1(u, 'rrlw_kg01/fracrefa', fracrefa)
   call wr1(u, 'rrlw_kg01/fracrefb', fracrefb)
   call wr3(u, 'rrlw_kg01/ka', ka)
   call wr2(u, 'rrlw_kg01/absa', absa)
   call wr3(u, 'rrlw_kg01/kb', kb)
   call wr2(u, 'rrlw_kg01/absb', absb)
   call wr2(u, 'rrlw_kg01/ka_mn2', ka_mn2)
   call wr2(u, 'rrlw_kg01/kb_mn2', kb_mn2)
   call wr2(u, 'rrlw_kg01/selfref', selfref)
   call wr2(u, 'rrlw_kg01/forref', forref)
end subroutine dump_rrlw_kg01

subroutine dump_rrlw_kg02(u)
   ! not dumped (never initialized by the WRF init path): refparam
   use dump_kit
   use rrlw_kg02
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg02/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg02/fracrefbo', fracrefbo)
   call wr3(u, 'rrlw_kg02/kao', kao)
   call wr3(u, 'rrlw_kg02/kbo', kbo)
   call wr2(u, 'rrlw_kg02/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg02/forrefo', forrefo)
   call wr1(u, 'rrlw_kg02/fracrefa', fracrefa)
   call wr1(u, 'rrlw_kg02/fracrefb', fracrefb)
   call wr3(u, 'rrlw_kg02/ka', ka)
   call wr2(u, 'rrlw_kg02/absa', absa)
   call wr3(u, 'rrlw_kg02/kb', kb)
   call wr2(u, 'rrlw_kg02/absb', absb)
   call wr2(u, 'rrlw_kg02/selfref', selfref)
   call wr2(u, 'rrlw_kg02/forref', forref)
end subroutine dump_rrlw_kg02

subroutine dump_rrlw_kg03(u)
   use dump_kit
   use rrlw_kg03
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg03/fracrefao', fracrefao)
   call wr2(u, 'rrlw_kg03/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg03/kao', kao)
   call wr4(u, 'rrlw_kg03/kbo', kbo)
   call wr3(u, 'rrlw_kg03/kao_mn2o', kao_mn2o)
   call wr3(u, 'rrlw_kg03/kbo_mn2o', kbo_mn2o)
   call wr2(u, 'rrlw_kg03/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg03/forrefo', forrefo)
   call wr2(u, 'rrlw_kg03/fracrefa', fracrefa)
   call wr2(u, 'rrlw_kg03/fracrefb', fracrefb)
   call wr4(u, 'rrlw_kg03/ka', ka)
   call wr2(u, 'rrlw_kg03/absa', absa)
   call wr4(u, 'rrlw_kg03/kb', kb)
   call wr2(u, 'rrlw_kg03/absb', absb)
   call wr3(u, 'rrlw_kg03/ka_mn2o', ka_mn2o)
   call wr3(u, 'rrlw_kg03/kb_mn2o', kb_mn2o)
   call wr2(u, 'rrlw_kg03/selfref', selfref)
   call wr2(u, 'rrlw_kg03/forref', forref)
end subroutine dump_rrlw_kg03

subroutine dump_rrlw_kg04(u)
   use dump_kit
   use rrlw_kg04
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg04/fracrefao', fracrefao)
   call wr2(u, 'rrlw_kg04/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg04/kao', kao)
   call wr4(u, 'rrlw_kg04/kbo', kbo)
   call wr2(u, 'rrlw_kg04/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg04/forrefo', forrefo)
   call wr2(u, 'rrlw_kg04/fracrefa', fracrefa)
   call wr2(u, 'rrlw_kg04/fracrefb', fracrefb)
   call wr4(u, 'rrlw_kg04/ka', ka)
   call wr2(u, 'rrlw_kg04/absa', absa)
   call wr4(u, 'rrlw_kg04/kb', kb)
   call wr2(u, 'rrlw_kg04/absb', absb)
   call wr2(u, 'rrlw_kg04/selfref', selfref)
   call wr2(u, 'rrlw_kg04/forref', forref)
end subroutine dump_rrlw_kg04

subroutine dump_rrlw_kg05(u)
   use dump_kit
   use rrlw_kg05
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg05/fracrefao', fracrefao)
   call wr2(u, 'rrlw_kg05/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg05/kao', kao)
   call wr4(u, 'rrlw_kg05/kbo', kbo)
   call wr3(u, 'rrlw_kg05/kao_mo3', kao_mo3)
   call wr1(u, 'rrlw_kg05/ccl4o', ccl4o)
   call wr2(u, 'rrlw_kg05/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg05/forrefo', forrefo)
   call wr2(u, 'rrlw_kg05/fracrefa', fracrefa)
   call wr2(u, 'rrlw_kg05/fracrefb', fracrefb)
   call wr4(u, 'rrlw_kg05/ka', ka)
   call wr2(u, 'rrlw_kg05/absa', absa)
   call wr4(u, 'rrlw_kg05/kb', kb)
   call wr2(u, 'rrlw_kg05/absb', absb)
   call wr3(u, 'rrlw_kg05/ka_mo3', ka_mo3)
   call wr2(u, 'rrlw_kg05/selfref', selfref)
   call wr2(u, 'rrlw_kg05/forref', forref)
   call wr1(u, 'rrlw_kg05/ccl4', ccl4)
end subroutine dump_rrlw_kg05

subroutine dump_rrlw_kg06(u)
   use dump_kit
   use rrlw_kg06
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg06/fracrefao', fracrefao)
   call wr3(u, 'rrlw_kg06/kao', kao)
   call wr2(u, 'rrlw_kg06/kao_mco2', kao_mco2)
   call wr1(u, 'rrlw_kg06/cfc11adjo', cfc11adjo)
   call wr1(u, 'rrlw_kg06/cfc12o', cfc12o)
   call wr2(u, 'rrlw_kg06/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg06/forrefo', forrefo)
   call wr1(u, 'rrlw_kg06/fracrefa', fracrefa)
   call wr3(u, 'rrlw_kg06/ka', ka)
   call wr2(u, 'rrlw_kg06/absa', absa)
   call wr2(u, 'rrlw_kg06/ka_mco2', ka_mco2)
   call wr2(u, 'rrlw_kg06/selfref', selfref)
   call wr2(u, 'rrlw_kg06/forref', forref)
   call wr1(u, 'rrlw_kg06/cfc11adj', cfc11adj)
   call wr1(u, 'rrlw_kg06/cfc12', cfc12)
end subroutine dump_rrlw_kg06

subroutine dump_rrlw_kg07(u)
   use dump_kit
   use rrlw_kg07
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg07/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg07/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg07/kao', kao)
   call wr3(u, 'rrlw_kg07/kbo', kbo)
   call wr3(u, 'rrlw_kg07/kao_mco2', kao_mco2)
   call wr2(u, 'rrlw_kg07/kbo_mco2', kbo_mco2)
   call wr2(u, 'rrlw_kg07/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg07/forrefo', forrefo)
   call wr1(u, 'rrlw_kg07/fracrefb', fracrefb)
   call wr2(u, 'rrlw_kg07/fracrefa', fracrefa)
   call wr4(u, 'rrlw_kg07/ka', ka)
   call wr2(u, 'rrlw_kg07/absa', absa)
   call wr3(u, 'rrlw_kg07/kb', kb)
   call wr2(u, 'rrlw_kg07/absb', absb)
   call wr3(u, 'rrlw_kg07/ka_mco2', ka_mco2)
   call wr2(u, 'rrlw_kg07/kb_mco2', kb_mco2)
   call wr2(u, 'rrlw_kg07/selfref', selfref)
   call wr2(u, 'rrlw_kg07/forref', forref)
end subroutine dump_rrlw_kg07

subroutine dump_rrlw_kg08(u)
   use dump_kit
   use rrlw_kg08
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg08/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg08/fracrefbo', fracrefbo)
   call wr3(u, 'rrlw_kg08/kao', kao)
   call wr3(u, 'rrlw_kg08/kbo', kbo)
   call wr2(u, 'rrlw_kg08/kao_mco2', kao_mco2)
   call wr2(u, 'rrlw_kg08/kbo_mco2', kbo_mco2)
   call wr2(u, 'rrlw_kg08/kao_mn2o', kao_mn2o)
   call wr2(u, 'rrlw_kg08/kbo_mn2o', kbo_mn2o)
   call wr2(u, 'rrlw_kg08/kao_mo3', kao_mo3)
   call wr1(u, 'rrlw_kg08/cfc12o', cfc12o)
   call wr1(u, 'rrlw_kg08/cfc22adjo', cfc22adjo)
   call wr2(u, 'rrlw_kg08/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg08/forrefo', forrefo)
   call wr1(u, 'rrlw_kg08/fracrefa', fracrefa)
   call wr1(u, 'rrlw_kg08/fracrefb', fracrefb)
   call wr1(u, 'rrlw_kg08/cfc12', cfc12)
   call wr1(u, 'rrlw_kg08/cfc22adj', cfc22adj)
   call wr3(u, 'rrlw_kg08/ka', ka)
   call wr2(u, 'rrlw_kg08/absa', absa)
   call wr3(u, 'rrlw_kg08/kb', kb)
   call wr2(u, 'rrlw_kg08/absb', absb)
   call wr2(u, 'rrlw_kg08/ka_mco2', ka_mco2)
   call wr2(u, 'rrlw_kg08/ka_mn2o', ka_mn2o)
   call wr2(u, 'rrlw_kg08/ka_mo3', ka_mo3)
   call wr2(u, 'rrlw_kg08/kb_mco2', kb_mco2)
   call wr2(u, 'rrlw_kg08/kb_mn2o', kb_mn2o)
   call wr2(u, 'rrlw_kg08/selfref', selfref)
   call wr2(u, 'rrlw_kg08/forref', forref)
end subroutine dump_rrlw_kg08

subroutine dump_rrlw_kg09(u)
   use dump_kit
   use rrlw_kg09
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg09/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg09/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg09/kao', kao)
   call wr3(u, 'rrlw_kg09/kbo', kbo)
   call wr3(u, 'rrlw_kg09/kao_mn2o', kao_mn2o)
   call wr2(u, 'rrlw_kg09/kbo_mn2o', kbo_mn2o)
   call wr2(u, 'rrlw_kg09/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg09/forrefo', forrefo)
   call wr1(u, 'rrlw_kg09/fracrefb', fracrefb)
   call wr2(u, 'rrlw_kg09/fracrefa', fracrefa)
   call wr4(u, 'rrlw_kg09/ka', ka)
   call wr2(u, 'rrlw_kg09/absa', absa)
   call wr3(u, 'rrlw_kg09/kb', kb)
   call wr2(u, 'rrlw_kg09/absb', absb)
   call wr3(u, 'rrlw_kg09/ka_mn2o', ka_mn2o)
   call wr2(u, 'rrlw_kg09/kb_mn2o', kb_mn2o)
   call wr2(u, 'rrlw_kg09/selfref', selfref)
   call wr2(u, 'rrlw_kg09/forref', forref)
end subroutine dump_rrlw_kg09

subroutine dump_rrlw_kg10(u)
   use dump_kit
   use rrlw_kg10
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg10/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg10/fracrefbo', fracrefbo)
   call wr3(u, 'rrlw_kg10/kao', kao)
   call wr3(u, 'rrlw_kg10/kbo', kbo)
   call wr2(u, 'rrlw_kg10/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg10/forrefo', forrefo)
   call wr1(u, 'rrlw_kg10/fracrefa', fracrefa)
   call wr1(u, 'rrlw_kg10/fracrefb', fracrefb)
   call wr3(u, 'rrlw_kg10/ka', ka)
   call wr2(u, 'rrlw_kg10/absa', absa)
   call wr3(u, 'rrlw_kg10/kb', kb)
   call wr2(u, 'rrlw_kg10/absb', absb)
   call wr2(u, 'rrlw_kg10/selfref', selfref)
   call wr2(u, 'rrlw_kg10/forref', forref)
end subroutine dump_rrlw_kg10

subroutine dump_rrlw_kg11(u)
   use dump_kit
   use rrlw_kg11
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg11/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg11/fracrefbo', fracrefbo)
   call wr3(u, 'rrlw_kg11/kao', kao)
   call wr3(u, 'rrlw_kg11/kbo', kbo)
   call wr2(u, 'rrlw_kg11/kao_mo2', kao_mo2)
   call wr2(u, 'rrlw_kg11/kbo_mo2', kbo_mo2)
   call wr2(u, 'rrlw_kg11/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg11/forrefo', forrefo)
   call wr1(u, 'rrlw_kg11/fracrefa', fracrefa)
   call wr1(u, 'rrlw_kg11/fracrefb', fracrefb)
   call wr3(u, 'rrlw_kg11/ka', ka)
   call wr2(u, 'rrlw_kg11/absa', absa)
   call wr3(u, 'rrlw_kg11/kb', kb)
   call wr2(u, 'rrlw_kg11/absb', absb)
   call wr2(u, 'rrlw_kg11/ka_mo2', ka_mo2)
   call wr2(u, 'rrlw_kg11/kb_mo2', kb_mo2)
   call wr2(u, 'rrlw_kg11/selfref', selfref)
   call wr2(u, 'rrlw_kg11/forref', forref)
end subroutine dump_rrlw_kg11

subroutine dump_rrlw_kg12(u)
   use dump_kit
   use rrlw_kg12
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg12/fracrefao', fracrefao)
   call wr4(u, 'rrlw_kg12/kao', kao)
   call wr2(u, 'rrlw_kg12/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg12/forrefo', forrefo)
   call wr2(u, 'rrlw_kg12/fracrefa', fracrefa)
   call wr4(u, 'rrlw_kg12/ka', ka)
   call wr2(u, 'rrlw_kg12/absa', absa)
   call wr2(u, 'rrlw_kg12/selfref', selfref)
   call wr2(u, 'rrlw_kg12/forref', forref)
end subroutine dump_rrlw_kg12

subroutine dump_rrlw_kg13(u)
   use dump_kit
   use rrlw_kg13
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg13/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg13/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg13/kao', kao)
   call wr3(u, 'rrlw_kg13/kao_mco2', kao_mco2)
   call wr3(u, 'rrlw_kg13/kao_mco', kao_mco)
   call wr2(u, 'rrlw_kg13/kbo_mo3', kbo_mo3)
   call wr2(u, 'rrlw_kg13/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg13/forrefo', forrefo)
   call wr1(u, 'rrlw_kg13/fracrefb', fracrefb)
   call wr2(u, 'rrlw_kg13/fracrefa', fracrefa)
   call wr4(u, 'rrlw_kg13/ka', ka)
   call wr2(u, 'rrlw_kg13/absa', absa)
   call wr3(u, 'rrlw_kg13/ka_mco2', ka_mco2)
   call wr3(u, 'rrlw_kg13/ka_mco', ka_mco)
   call wr2(u, 'rrlw_kg13/kb_mo3', kb_mo3)
   call wr2(u, 'rrlw_kg13/selfref', selfref)
   call wr2(u, 'rrlw_kg13/forref', forref)
end subroutine dump_rrlw_kg13

subroutine dump_rrlw_kg14(u)
   use dump_kit
   use rrlw_kg14
   implicit none
   integer, intent(in) :: u
   call wr1(u, 'rrlw_kg14/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg14/fracrefbo', fracrefbo)
   call wr3(u, 'rrlw_kg14/kao', kao)
   call wr3(u, 'rrlw_kg14/kbo', kbo)
   call wr2(u, 'rrlw_kg14/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg14/forrefo', forrefo)
   call wr1(u, 'rrlw_kg14/fracrefa', fracrefa)
   call wr1(u, 'rrlw_kg14/fracrefb', fracrefb)
   call wr3(u, 'rrlw_kg14/ka', ka)
   call wr2(u, 'rrlw_kg14/absa', absa)
   call wr3(u, 'rrlw_kg14/kb', kb)
   call wr2(u, 'rrlw_kg14/absb', absb)
   call wr2(u, 'rrlw_kg14/selfref', selfref)
   call wr2(u, 'rrlw_kg14/forref', forref)
end subroutine dump_rrlw_kg14

subroutine dump_rrlw_kg15(u)
   use dump_kit
   use rrlw_kg15
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg15/fracrefao', fracrefao)
   call wr4(u, 'rrlw_kg15/kao', kao)
   call wr3(u, 'rrlw_kg15/kao_mn2', kao_mn2)
   call wr2(u, 'rrlw_kg15/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg15/forrefo', forrefo)
   call wr2(u, 'rrlw_kg15/fracrefa', fracrefa)
   call wr4(u, 'rrlw_kg15/ka', ka)
   call wr2(u, 'rrlw_kg15/absa', absa)
   call wr3(u, 'rrlw_kg15/ka_mn2', ka_mn2)
   call wr2(u, 'rrlw_kg15/selfref', selfref)
   call wr2(u, 'rrlw_kg15/forref', forref)
end subroutine dump_rrlw_kg15

subroutine dump_rrlw_kg16(u)
   use dump_kit
   use rrlw_kg16
   implicit none
   integer, intent(in) :: u
   call wr2(u, 'rrlw_kg16/fracrefao', fracrefao)
   call wr1(u, 'rrlw_kg16/fracrefbo', fracrefbo)
   call wr4(u, 'rrlw_kg16/kao', kao)
   call wr3(u, 'rrlw_kg16/kbo', kbo)
   call wr2(u, 'rrlw_kg16/selfrefo', selfrefo)
   call wr2(u, 'rrlw_kg16/forrefo', forrefo)
   call wr1(u, 'rrlw_kg16/fracrefb', fracrefb)
   call wr2(u, 'rrlw_kg16/fracrefa', fracrefa)
   call wr4(u, 'rrlw_kg16/ka', ka)
   call wr2(u, 'rrlw_kg16/absa', absa)
   call wr3(u, 'rrlw_kg16/kb', kb)
   call wr2(u, 'rrlw_kg16/absb', absb)
   call wr2(u, 'rrlw_kg16/selfref', selfref)
   call wr2(u, 'rrlw_kg16/forref', forref)
end subroutine dump_rrlw_kg16

