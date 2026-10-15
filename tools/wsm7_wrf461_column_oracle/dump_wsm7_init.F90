program dump_wsm7_init
  ! Print the float32 words wsm7init leaves in WRF v4.6.1's module_mp_wsm7
  ! (its REAL, SAVE coefficients are public) and the radar_init words that
  ! follow from wsm7init's xam/xbm/xmu, as hexadecimal bit patterns.
  !
  ! These are the words gpuwm/data/microphysics/wsm7-wrf461-words.json
  ! freezes for the CUDA kernel; the column oracle then proves them, because a
  ! wrong word cannot reproduce WRF's outputs bit for bit.
  use module_model_constants, only: rhoair0, rhowater, rhosnow, cliq, cpv
  use module_mp_wsm7
  use module_mp_radar, only: xcre, xcse, xcge, xcrg, xcsg, xcgg, xam_r, &
       xam_s, xam_g, xbm_s, xbm_g, xmu_s, xmu_g, xocms, xobms, xocmg, xobmg, &
       k_w, pi5, lamda4, m_w_0, m_i_0, nrbins, xxds, xdts, xxdg, xdtg, simpson
  implicit none
  integer :: n

  call wsm7init(rhoair0, rhowater, rhosnow, cliq, cpv, .true.)
  call w('pi', pi);           call w('xlv1', xlv1)
  call w('qc0', qc0);         call w('qck1', qck1)
  call w('pidnc', pidnc)
  call w('pidn0r', pidn0r);   call w('pidn0s', pidn0s)
  call w('pidn0g', pidn0g);   call w('pidn0h', pidn0h)
  call w('pvtr', pvtr);       call w('pvts', pvts)
  call w('pvtg', pvtg);       call w('pvth', pvth)
  call w('pacrr', pacrr);     call w('pacrc', pacrc)
  call w('pacrg', pacrg);     call w('pacrh', pacrh)
  call w('precr1', precr1);   call w('precr2', precr2)
  call w('precs1', precs1);   call w('precs2', precs2)
  call w('precg1', precg1);   call w('precg2', precg2)
  call w('prech1', prech1);   call w('prech2', prech2);   call w('prech3', prech3)
  call w('g6pbr', g6pbr);     call w('roqimax', roqimax)
  call w('rslopermax', rslopermax);   call w('rslopesmax', rslopesmax)
  call w('rslopegmax', rslopegmax);   call w('rslopehmax', rslopehmax)
  call w('rsloperbmax', rsloperbmax); call w('rslopesbmax', rslopesbmax)
  call w('rslopegbmax', rslopegbmax); call w('rslopehbmax', rslopehbmax)
  call w('rsloper2max', rsloper2max); call w('rslopes2max', rslopes2max)
  call w('rslopeg2max', rslopeg2max); call w('rslopeh2max', rslopeh2max)
  call w('rsloper3max', rsloper3max); call w('rslopes3max', rslopes3max)
  call w('rslopeg3max', rslopeg3max); call w('rslopeh3max', rslopeh3max)
  ! radar_init's words (refl10cm_wsm7), float32 then float64.
  call w('xam_r', xam_r); call w('xam_s', xam_s); call w('xam_g', xam_g)
  do n = 1, 4
    call w('xcre', xcre(n)); call w('xcse', xcse(n)); call w('xcge', xcge(n))
    call w('xcrg', xcrg(n)); call w('xcsg', xcsg(n)); call w('xcgg', xcgg(n))
  enddo
  call w('xbm_s', xbm_s); call w('xbm_g', xbm_g)
  call w('xmu_s', xmu_s); call w('xmu_g', xmu_g)
  call w('xocms', xocms); call w('xobms', xobms)
  call w('xocmg', xocmg); call w('xobmg', xobmg)
  call d('k_w', k_w); call d('pi5', pi5); call d('lamda4', lamda4)
  call d('m_w_0_re', real(m_w_0, kind=8)); call d('m_w_0_im', aimag(m_w_0))
  call d('m_i_0_re', real(m_i_0, kind=8)); call d('m_i_0_im', aimag(m_i_0))
  do n = 1, nrbins
    call d('xxds', xxds(n))
  enddo
  do n = 1, nrbins
    call d('xdts', xdts(n))
  enddo
  do n = 1, nrbins
    call d('xxdg', xxdg(n))
  enddo
  do n = 1, nrbins
    call d('xdtg', xdtg(n))
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

end program dump_wsm7_init
