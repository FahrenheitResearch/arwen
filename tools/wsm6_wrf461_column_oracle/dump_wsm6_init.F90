program dump_wsm6_init
  ! Print the float32 words mp_wsm6_init leaves in WRF v4.6.1's mp_wsm6
  ! module, for hail_opt 0 and 1, as hexadecimal bit patterns.
  !
  ! The module's derived constants are PRIVATE, so build.sh links this
  ! program against a COPY of mp_wsm6.F90 whose single default-accessibility
  ! line ` private` reads ` public` (nothing else differs; build.sh checks
  ! the diff is exactly that one line).  The column oracle itself links the
  ! unmodified file.  These words are what gpuwm/core/wsm6_constants.py
  ! freezes for the CUDA kernel; the column oracle then proves them, because
  ! a wrong word cannot reproduce WRF's outputs bit for bit.
  use ccpp_kind_types, only: kind_phys
  use module_model_constants, only: rhoair0, rhowater, rhosnow, cliq, cpv
  use mp_wsm6
  implicit none
  integer :: hail, errflg, n
  character(len=256) :: errmsg

  do hail = 0, 1
    call mp_wsm6_init(rhoair0, rhowater, rhosnow, cliq, cpv, hail, errmsg, errflg)
    if (errflg /= 0) error stop 'mp_wsm6_init failed'
    write(*,'(a,i0)') 'hail_opt ', hail
    call w('pi', pi);           call w('xlv1', xlv1)
    call w('qc0', qc0);         call w('qck1', qck1)
    call w('pidnc', pidnc);     call w('pidn0r', pidn0r)
    call w('pidn0s', pidn0s);   call w('pidn0g', pidn0g)
    call w('pvtr', pvtr);       call w('pvts', pvts);       call w('pvtg', pvtg)
    call w('pacrr', pacrr);     call w('pacrc', pacrc);     call w('pacrg', pacrg)
    call w('precr1', precr1);   call w('precr2', precr2)
    call w('precs1', precs1);   call w('precs2', precs2)
    call w('precg1', precg1);   call w('precg2', precg2)
    call w('g6pbr', g6pbr);     call w('roqimax', roqimax)
    call w('n0g', n0g);         call w('bvtg', bvtg)
    call w('rslopermax', rslopermax);   call w('rslopesmax', rslopesmax)
    call w('rslopegmax', rslopegmax);   call w('rsloperbmax', rsloperbmax)
    call w('rslopesbmax', rslopesbmax); call w('rslopegbmax', rslopegbmax)
    call w('rsloper2max', rsloper2max); call w('rslopes2max', rslopes2max)
    call w('rslopeg2max', rslopeg2max); call w('rsloper3max', rsloper3max)
    call w('rslopes3max', rslopes3max); call w('rslopeg3max', rslopeg3max)
    ! mp_radar's radar_init words (refl10cm_wsm6), float32 then float64.
    do n = 1, 4
      call w('xcre', xcre(n)); call w('xcse', xcse(n)); call w('xcge', xcge(n))
      call w('xcrg', xcrg(n)); call w('xcsg', xcsg(n)); call w('xcgg', xcgg(n))
    enddo
    call w('xam_r', xam_r); call w('xam_s', xam_s); call w('xam_g', xam_g)
    call w('xbm_s', xbm_s); call w('xbm_g', xbm_g)
    call w('xmu_s', xmu_s); call w('xmu_g', xmu_g)
    call w('xocms', xocms); call w('xobms', xobms)
    call w('xocmg', xocmg); call w('xobmg', xobmg)
    call d('k_w', k_w); call d('pi5', pi5); call d('lamda4', lamda4)
    call d('m_w_0_re', real(m_w_0, kind=8)); call d('m_w_0_im', aimag(m_w_0))
    call d('m_i_0_re', real(m_i_0, kind=8)); call d('m_i_0_im', aimag(m_i_0))
    do n = 1, nrbins
      call d('xxds', xxds(n)); call d('xdts', xdts(n))
      call d('xxdg', xxdg(n)); call d('xdtg', xdtg(n))
    enddo
    do n = 1, nrbins + 1
      call d('simpson', simpson(n))
    enddo
  enddo

contains

  subroutine w(name, value)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: value
    write(*,'(a,1x,z8.8,1x,es16.9)') name, transfer(value, 0), value
  end subroutine w

  subroutine d(name, value)
    character(len=*), intent(in) :: name
    real(kind=8), intent(in) :: value
    write(*,'(a,1x,z16.16,1x,es24.17)') name, transfer(value, 0_8), value
  end subroutine d

end program dump_wsm6_init
