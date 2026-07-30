! Dump every post-init RRTMG SW coefficient table from the UNMODIFIED
! WRF v4.6.1 module_ra_rrtmg_sw.F, after calling rrtmg_swinit exactly as
! WRF does (rrtmg_swlookuptable reads RRTMG_SW_DATA big-endian, then
! rrtmg_sw_ini(cp) performs the 224->112 g-point reduction and builds
! exp_tbl, heatfac, cloud optics tables, and reference atmosphere).
!
! Also dumps the RAW (pre-reduction) arrays read from RRTMG_SW_DATA so the
! gpuwm port of the cmbgb## reduction can be gated against raw -> reduced.
!
! Output: self-describing stream binary ("SWD1" format, see sw_dumpio.py):
!   magic "SWD1"
!   per record: name_len int32 | name bytes | dtype int32 (0=f32,1=i32)
!               | rank int32 | dims int32(rank), Fortran order | payload
!
! Build/run via sw_build.sh.  FP32 throughout (kind_rb = kind(1.0)).

program sw_dump_tables
  use module_ra_rrtmg_sw, only : rrtmg_swinit
  implicit none

  integer :: ou
  character(len=1024) :: outfile

  if (command_argument_count() /= 1) then
     write(*,'(A)') 'usage: sw_dump_tables out.swd'
     error stop 4
  end if
  call get_command_argument(1, outfile)

  ! Init exactly as WRF: single call, any consistent WRF-ish dims.
  call rrtmg_swinit(.true., 1,2,1,2,1,50, 1,1,1,1,1,50, 1,1,1,1,1,49)

  open(newunit=ou, file=trim(outfile), status='replace', access='stream', &
       form='unformatted')
  write(ou) 'SWD1'

  call dump_all(ou)

  close(ou)
  write(*,'(A)') 'sw_dump_tables: done'

contains

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
    call wname(name, 0, [integer ::])
    write(ou) a
  end subroutine wr0

  subroutine wi0(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a
    call wname(name, 1, [integer ::])
    write(ou) int(a,4)
  end subroutine wi0

  subroutine wr1(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr1

  subroutine wi1(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    call wname(name, 1, shape(a))
    write(ou) int(a,4)
  end subroutine wi1

  subroutine wr2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr2

  subroutine wr3(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr3

  subroutine wr4(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:,:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr4

  subroutine dump_all(unit_)
    use parrrsw
    use rrsw_con
    use rrsw_ref, only : pref_ => pref, preflog_ => preflog, tref_ => tref
    use rrsw_tbl, only : exp_tbl, bpade, tau_tbl, od_lo, tblint
    use rrsw_wvn
    use rrsw_cld
    use rrsw_aer
    integer, intent(in) :: unit_

    call wr0('con/heatfac', heatfac)
    call wr0('con/oneminus', oneminus)
    call wr0('con/pi', pi)
    call wr0('con/grav', grav)
    call wr0('con/avogad', avogad)
    call wr0('con/fluxfac', fluxfac)
    call wr0('parrrsw/rrsw_scon', rrsw_scon)

    call wr1('ref/pref', pref_)
    call wr1('ref/preflog', preflog_)
    call wr1('ref/tref', tref_)

    call wr1('tbl/exp_tbl', exp_tbl)
    call wr0('tbl/bpade', bpade)
    call wr0('tbl/tau_tbl', tau_tbl)
    call wr0('tbl/od_lo', od_lo)
    call wr0('tbl/tblint', tblint)

    call wi1('wvn/ng', ng)
    call wi1('wvn/nspa', nspa)
    call wi1('wvn/nspb', nspb)
    call wr1('wvn/wavenum1', wavenum1)
    call wr1('wvn/wavenum2', wavenum2)
    call wr1('wvn/delwave', delwave)
    call wi1('wvn/ngc', ngc)
    call wi1('wvn/ngs', ngs)
    call wi1('wvn/ngn', ngn)
    call wi1('wvn/ngb', ngb)
    call wi1('wvn/ngm', ngm)
    call wr1('wvn/wt', wt)
    call wr1('wvn/rwgt', rwgt)

    call wr2('cld/extliq1', extliq1)
    call wr2('cld/ssaliq1', ssaliq1)
    call wr2('cld/asyliq1', asyliq1)
    call wr2('cld/extice2', extice2)
    call wr2('cld/ssaice2', ssaice2)
    call wr2('cld/asyice2', asyice2)
    call wr2('cld/extice3', extice3)
    call wr2('cld/ssaice3', ssaice3)
    call wr2('cld/asyice3', asyice3)
    call wr2('cld/fdlice3', fdlice3)
    call wr1('cld/abari', abari)
    call wr1('cld/bbari', bbari)
    call wr1('cld/cbari', cbari)
    call wr1('cld/dbari', dbari)
    call wr1('cld/ebari', ebari)
    call wr1('cld/fbari', fbari)

    call wr2('aer/rsrtaua', rsrtaua)
    call wr2('aer/rsrpiza', rsrpiza)
    call wr2('aer/rsrasya', rsrasya)

    call dump_kg16(); call dump_kg17(); call dump_kg18(); call dump_kg19()
    call dump_kg20(); call dump_kg21(); call dump_kg22(); call dump_kg23()
    call dump_kg24(); call dump_kg25(); call dump_kg26(); call dump_kg27()
    call dump_kg28(); call dump_kg29()
  end subroutine dump_all

  subroutine dump_kg16()
    use rrsw_kg16
    call wr4('kg16/kao', kao);           call wr3('kg16/kbo', kbo)
    call wr2('kg16/selfrefo', selfrefo); call wr2('kg16/forrefo', forrefo)
    call wr1('kg16/sfluxrefo', sfluxrefo)
    call wr2('kg16/absa', absa);         call wr2('kg16/absb', absb)
    call wr2('kg16/selfref', selfref);   call wr2('kg16/forref', forref)
    call wr1('kg16/sfluxref', sfluxref)
    call wr0('kg16/rayl', rayl);         call wr0('kg16/strrat1', strrat1)
    call wi0('kg16/layreffr', layreffr)
  end subroutine dump_kg16

  subroutine dump_kg17()
    use rrsw_kg17
    call wr4('kg17/kao', kao);           call wr4('kg17/kbo', kbo)
    call wr2('kg17/selfrefo', selfrefo); call wr2('kg17/forrefo', forrefo)
    call wr2('kg17/sfluxrefo', sfluxrefo)
    call wr2('kg17/absa', absa);         call wr2('kg17/absb', absb)
    call wr2('kg17/selfref', selfref);   call wr2('kg17/forref', forref)
    call wr2('kg17/sfluxref', sfluxref)
    call wr0('kg17/rayl', rayl);         call wr0('kg17/strrat', strrat)
    call wi0('kg17/layreffr', layreffr)
  end subroutine dump_kg17

  subroutine dump_kg18()
    use rrsw_kg18
    call wr4('kg18/kao', kao);           call wr3('kg18/kbo', kbo)
    call wr2('kg18/selfrefo', selfrefo); call wr2('kg18/forrefo', forrefo)
    call wr2('kg18/sfluxrefo', sfluxrefo)
    call wr2('kg18/absa', absa);         call wr2('kg18/absb', absb)
    call wr2('kg18/selfref', selfref);   call wr2('kg18/forref', forref)
    call wr2('kg18/sfluxref', sfluxref)
    call wr0('kg18/rayl', rayl);         call wr0('kg18/strrat', strrat)
    call wi0('kg18/layreffr', layreffr)
  end subroutine dump_kg18

  subroutine dump_kg19()
    use rrsw_kg19
    call wr4('kg19/kao', kao);           call wr3('kg19/kbo', kbo)
    call wr2('kg19/selfrefo', selfrefo); call wr2('kg19/forrefo', forrefo)
    call wr2('kg19/sfluxrefo', sfluxrefo)
    call wr2('kg19/absa', absa);         call wr2('kg19/absb', absb)
    call wr2('kg19/selfref', selfref);   call wr2('kg19/forref', forref)
    call wr2('kg19/sfluxref', sfluxref)
    call wr0('kg19/rayl', rayl);         call wr0('kg19/strrat', strrat)
    call wi0('kg19/layreffr', layreffr)
  end subroutine dump_kg19

  subroutine dump_kg20()
    use rrsw_kg20
    call wr3('kg20/kao', kao);           call wr3('kg20/kbo', kbo)
    call wr2('kg20/selfrefo', selfrefo); call wr2('kg20/forrefo', forrefo)
    call wr1('kg20/sfluxrefo', sfluxrefo)
    call wr1('kg20/absch4o', absch4o)
    call wr2('kg20/absa', absa);         call wr2('kg20/absb', absb)
    call wr2('kg20/selfref', selfref);   call wr2('kg20/forref', forref)
    call wr1('kg20/sfluxref', sfluxref)
    call wr1('kg20/absch4', absch4)
    call wr0('kg20/rayl', rayl)
    call wi0('kg20/layreffr', layreffr)
  end subroutine dump_kg20

  subroutine dump_kg21()
    use rrsw_kg21
    call wr4('kg21/kao', kao);           call wr4('kg21/kbo', kbo)
    call wr2('kg21/selfrefo', selfrefo); call wr2('kg21/forrefo', forrefo)
    call wr2('kg21/sfluxrefo', sfluxrefo)
    call wr2('kg21/absa', absa);         call wr2('kg21/absb', absb)
    call wr2('kg21/selfref', selfref);   call wr2('kg21/forref', forref)
    call wr2('kg21/sfluxref', sfluxref)
    call wr0('kg21/rayl', rayl);         call wr0('kg21/strrat', strrat)
    call wi0('kg21/layreffr', layreffr)
  end subroutine dump_kg21

  subroutine dump_kg22()
    use rrsw_kg22
    call wr4('kg22/kao', kao);           call wr3('kg22/kbo', kbo)
    call wr2('kg22/selfrefo', selfrefo); call wr2('kg22/forrefo', forrefo)
    call wr2('kg22/sfluxrefo', sfluxrefo)
    call wr2('kg22/absa', absa);         call wr2('kg22/absb', absb)
    call wr2('kg22/selfref', selfref);   call wr2('kg22/forref', forref)
    call wr2('kg22/sfluxref', sfluxref)
    call wr0('kg22/rayl', rayl);         call wr0('kg22/strrat', strrat)
    call wi0('kg22/layreffr', layreffr)
  end subroutine dump_kg22

  subroutine dump_kg23()
    use rrsw_kg23
    call wr3('kg23/kao', kao)
    call wr2('kg23/selfrefo', selfrefo); call wr2('kg23/forrefo', forrefo)
    call wr1('kg23/sfluxrefo', sfluxrefo)
    call wr1('kg23/raylo', raylo)
    call wr2('kg23/absa', absa)
    call wr2('kg23/selfref', selfref);   call wr2('kg23/forref', forref)
    call wr1('kg23/sfluxref', sfluxref)
    call wr1('kg23/rayl', rayl)
    call wr0('kg23/givfac', givfac)
    call wi0('kg23/layreffr', layreffr)
  end subroutine dump_kg23

  subroutine dump_kg24()
    use rrsw_kg24
    call wr4('kg24/kao', kao);           call wr3('kg24/kbo', kbo)
    call wr2('kg24/selfrefo', selfrefo); call wr2('kg24/forrefo', forrefo)
    call wr2('kg24/sfluxrefo', sfluxrefo)
    call wr1('kg24/abso3ao', abso3ao);   call wr1('kg24/abso3bo', abso3bo)
    call wr2('kg24/raylao', raylao);     call wr1('kg24/raylbo', raylbo)
    call wr2('kg24/absa', absa);         call wr2('kg24/absb', absb)
    call wr2('kg24/selfref', selfref);   call wr2('kg24/forref', forref)
    call wr2('kg24/sfluxref', sfluxref)
    call wr1('kg24/abso3a', abso3a);     call wr1('kg24/abso3b', abso3b)
    call wr2('kg24/rayla', rayla);       call wr1('kg24/raylb', raylb)
    call wr0('kg24/strrat', strrat)
    call wi0('kg24/layreffr', layreffr)
  end subroutine dump_kg24

  subroutine dump_kg25()
    use rrsw_kg25
    call wr3('kg25/kao', kao)
    call wr1('kg25/sfluxrefo', sfluxrefo)
    call wr1('kg25/abso3ao', abso3ao);   call wr1('kg25/abso3bo', abso3bo)
    call wr1('kg25/raylo', raylo)
    call wr2('kg25/absa', absa)
    call wr1('kg25/sfluxref', sfluxref)
    call wr1('kg25/abso3a', abso3a);     call wr1('kg25/abso3b', abso3b)
    call wr1('kg25/rayl', rayl)
    call wi0('kg25/layreffr', layreffr)
  end subroutine dump_kg25

  subroutine dump_kg26()
    use rrsw_kg26
    call wr1('kg26/sfluxrefo', sfluxrefo)
    call wr1('kg26/raylo', raylo)
    call wr1('kg26/sfluxref', sfluxref)
    call wr1('kg26/rayl', rayl)
  end subroutine dump_kg26

  subroutine dump_kg27()
    use rrsw_kg27
    call wr3('kg27/kao', kao);           call wr3('kg27/kbo', kbo)
    call wr1('kg27/sfluxrefo', sfluxrefo)
    call wr1('kg27/raylo', raylo)
    call wr2('kg27/absa', absa);         call wr2('kg27/absb', absb)
    call wr1('kg27/sfluxref', sfluxref)
    call wr1('kg27/rayl', rayl)
    call wr0('kg27/scalekur', scalekur)
    call wi0('kg27/layreffr', layreffr)
  end subroutine dump_kg27

  subroutine dump_kg28()
    use rrsw_kg28
    call wr4('kg28/kao', kao);           call wr4('kg28/kbo', kbo)
    call wr2('kg28/sfluxrefo', sfluxrefo)
    call wr2('kg28/absa', absa);         call wr2('kg28/absb', absb)
    call wr2('kg28/sfluxref', sfluxref)
    call wr0('kg28/rayl', rayl);         call wr0('kg28/strrat', strrat)
    call wi0('kg28/layreffr', layreffr)
  end subroutine dump_kg28

  subroutine dump_kg29()
    use rrsw_kg29
    call wr3('kg29/kao', kao);           call wr3('kg29/kbo', kbo)
    call wr2('kg29/selfrefo', selfrefo); call wr2('kg29/forrefo', forrefo)
    call wr1('kg29/sfluxrefo', sfluxrefo)
    call wr1('kg29/absh2oo', absh2oo);   call wr1('kg29/absco2o', absco2o)
    call wr2('kg29/absa', absa);         call wr2('kg29/absb', absb)
    call wr2('kg29/selfref', selfref);   call wr2('kg29/forref', forref)
    call wr1('kg29/sfluxref', sfluxref)
    call wr1('kg29/absh2o', absh2o);     call wr1('kg29/absco2', absco2)
    call wr0('kg29/rayl', rayl)
    call wi0('kg29/layreffr', layreffr)
  end subroutine dump_kg29

end program sw_dump_tables
