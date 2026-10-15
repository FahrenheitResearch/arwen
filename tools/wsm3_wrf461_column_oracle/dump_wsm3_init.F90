program dump_wsm3_init
  ! Print the float32 words wsm3init leaves in WRF v4.6.1's module_mp_wsm3,
  ! as hexadecimal bit patterns, after the call module_physics_init.F:4483
  ! makes (rhoair0, rhowater, rhosnow, cliq, cpv).
  !
  ! Unlike mp_wsm6.F90, module_mp_wsm3.F declares its derived constants
  ! with no PRIVATE attribute (only its PARAMETERs are private), so this
  ! program links the UNMODIFIED module.  These words are what
  ! gpuwm/core/wsm3_constants.py freezes for the CUDA kernel; the column
  ! oracle then proves them, because a wrong word cannot reproduce WRF's
  ! outputs bit for bit.
  use module_model_constants, only: rhoair0, rhowater, rhosnow, cliq, cpv
  use module_mp_wsm3
  implicit none

  call wsm3init(rhoair0, rhowater, rhosnow, cliq, cpv, .true.)
  call w('pi', pi);             call w('xlv1', xlv1)
  call w('qc0', qc0);           call w('qck1', qck1)
  call w('pidnc', pidnc);       call w('pidn0r', pidn0r)
  call w('pidn0s', pidn0s)
  call w('pvtr', pvtr);         call w('pvts', pvts)
  call w('pacrr', pacrr);       call w('pacrs', pacrs)
  call w('precr1', precr1);     call w('precr2', precr2)
  call w('precs1', precs1);     call w('precs2', precs2)
  call w('roqimax', roqimax);   call w('xmmax', xmmax)
  call w('rslopermax', rslopermax);   call w('rslopesmax', rslopesmax)
  call w('rsloperbmax', rsloperbmax); call w('rslopesbmax', rslopesbmax)
  call w('rsloper2max', rsloper2max); call w('rslopes2max', rslopes2max)
  call w('rsloper3max', rsloper3max); call w('rslopes3max', rslopes3max)
  ! Gamma-product intermediates, for the record (the kernel reads only the
  ! derived words above).
  call w('g1pbr', g1pbr);   call w('g3pbr', g3pbr)
  call w('g4pbr', g4pbr);   call w('g5pbro2', g5pbro2)
  call w('g1pbs', g1pbs);   call w('g3pbs', g3pbs)
  call w('g4pbs', g4pbs);   call w('g5pbso2', g5pbso2)

contains

  subroutine w(name, value)
    character(len=*), intent(in) :: name
    real, intent(in) :: value
    write(*,'(a,1x,z8.8,1x,es16.9)') name, transfer(value, 0), value
  end subroutine w

end program dump_wsm3_init
