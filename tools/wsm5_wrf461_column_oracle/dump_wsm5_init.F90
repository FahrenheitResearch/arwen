program dump_wsm5_init
  ! Print the float32 words wsm5init leaves in WRF v4.6.1's module_mp_wsm5,
  ! and the radar_init words refl10cm_wsm5 reads from module_mp_radar, as
  ! hexadecimal bit patterns.  Linked against the build tree's own objects
  ! (module_mp_wsm5's SAVE variables and module_mp_radar's are public).
  ! These words are what gpuwm/core/wsm6_constants.py freezes for the CUDA
  ! kernel; the column oracle then proves them, because a wrong word cannot
  ! reproduce WRF's outputs bit for bit.
  use module_model_constants, only: rhoair0, rhowater, rhosnow, cliq, cpv
  use module_mp_wsm5
  use module_mp_radar
  implicit none
  integer :: n

  call wsm5init(rhoair0, rhowater, rhosnow, cliq, cpv, .true.)
  call w('pi', pi);             call w('xlv1', xlv1)
  call w('qc0', qc0);           call w('qck1', qck1)
  call w('pidnc', pidnc);       call w('pidn0r', pidn0r)
  call w('pidn0s', pidn0s)
  call w('pvtr', pvtr);         call w('pvts', pvts)
  call w('pacrr', pacrr);       call w('pacrc', pacrc)
  call w('precr1', precr1);     call w('precr2', precr2)
  call w('precs1', precs1);     call w('precs2', precs2)
  call w('roqimax', roqimax)
  call w('rslopermax', rslopermax);   call w('rslopesmax', rslopesmax)
  call w('rsloperbmax', rsloperbmax); call w('rslopesbmax', rslopesbmax)
  call w('rsloper2max', rsloper2max); call w('rslopes2max', rslopes2max)
  call w('rsloper3max', rsloper3max); call w('rslopes3max', rslopes3max)
  ! module_mp_radar's radar_init words, float32 then float64.
  do n = 1, 4
    call w('xcre', xcre(n)); call w('xcse', xcse(n))
    call w('xcrg', xcrg(n)); call w('xcsg', xcsg(n))
  enddo
  call w('xam_r', xam_r); call w('xam_s', xam_s)
  call w('xbm_s', xbm_s); call w('xmu_s', xmu_s)
  call w('xocms', xocms); call w('xobms', xobms)
  call d('k_w', k_w); call d('pi5', pi5); call d('lamda4', lamda4)
  call d('m_w_0_re', real(m_w_0, kind=8)); call d('m_w_0_im', aimag(m_w_0))
  call d('m_i_0_re', real(m_i_0, kind=8)); call d('m_i_0_im', aimag(m_i_0))
  do n = 1, nrbins
    call d('xxds', xxds(n)); call d('xdts', xdts(n))
  enddo
  do n = 1, nrbins + 1
    call d('simpson', simpson(n))
  enddo

contains

  subroutine w(name, value)
    character(len=*), intent(in) :: name
    real, intent(in) :: value
    write(*,'(a,1x,z8.8,1x,es16.9)') name, transfer(value, 0), value
  end subroutine w

  subroutine d(name, value)
    character(len=*), intent(in) :: name
    real(kind=8), intent(in) :: value
    write(*,'(a,1x,z16.16,1x,es24.17)') name, transfer(value, 0_8), value
  end subroutine d

end program dump_wsm5_init
