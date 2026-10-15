program dump_wsm6r_init
  ! Print the float32 words wsm6rinit leaves in WRF v4.6.1's module_mp_wsm6r,
  ! as hexadecimal bit patterns, in the order gpuwm/core/wsm6_constants.py
  ! WSM6R_INIT_F32 and kernels/wsm6r.cu's coefficient enum use.
  !
  ! module_mp_wsm6r has no PRIVATE statement, so its SAVE words are public
  ! and this program links the UNMODIFIED source.  wsm62D calls wsm6rinit
  ! at the top of every call; the words are the same every time (it reads
  ! only PARAMETERs).  The last two words are mp_wsm6_init's (WSM6's own
  ! module), read by the radius diagnosis.  The column oracle then proves
  ! the frozen words: a wrong word cannot reproduce WRF's outputs bit for
  ! bit.  module_model_constants is renamed on USE because module_mp_wsm6r
  ! declares PARAMETERs of the same names (cliq, cpv).
  use module_mp_wsm6r
  use module_model_constants, only: mc_rhoair0 => rhoair0, mc_rhowater => rhowater, &
       mc_rhosnow => rhosnow, mc_cliq => cliq, mc_cpv => cpv
  use mp_wsm6, only: mp_wsm6_init, wsm6_pidnc => pidnc, wsm6_pidn0s => pidn0s
  implicit none
  integer :: errflg
  character(len=256) :: errmsg

  call wsm6rinit
  call w('pi', pi)
  call w('qc0', qc0);         call w('qck1', qck1)
  call w('roqimax', roqimax); call w('xlv1', xlv1)
  call w('vt2r_a', vt2r_a);   call w('vt2s_a', vt2s_a)
  call w('vt2g_a', vt2g_a);   call w('vt2i_a', vt2i_a)
  call w('falli_a', falli_a)
  call w('prevp_a', prevp_a); call w('prevp_b', prevp_b)
  call w('psdep_a', psdep_a); call w('psdep_b', psdep_b)
  call w('psevp_a', psevp_a); call w('psevp_b', psevp_b)
  call w('pgdep_a', pgdep_a); call w('pgdep_b', pgdep_b)
  call w('pgevp_a', pgevp_a); call w('pgevp_b', pgevp_b)
  call w('psmlt_a', psmlt_a); call w('psmlt_b', psmlt_b)
  call w('pgmlt_a', pgmlt_a); call w('pgmlt_b', pgmlt_b)
  call w('praci_a', praci_a); call w('praci_b', praci_b)
  call w('praci_c', praci_c); call w('praci_d', praci_d)
  call w('psaci_a', psaci_a); call w('psaci_b', psaci_b)
  call w('psaci_c', psaci_c); call w('psaci_d', psaci_d)
  call w('pgaci_a', pgaci_a); call w('pgaci_b', pgaci_b)
  call w('pgaci_c', pgaci_c); call w('pgaci_d', pgaci_d)
  call w('pracs_a', pracs_a); call w('pracs_b', pracs_b)
  call w('pracs_c', pracs_c); call w('pracs_d', pracs_d)
  call w('psacr_a', psacr_a); call w('psacr_b', psacr_b)
  call w('psacr_c', psacr_c); call w('psacr_d', psacr_d)
  call w('pgacr_a', pgacr_a); call w('pgacr_b', pgacr_b)
  call w('pgacr_c', pgacr_c); call w('pgacr_d', pgacr_d)
  call w('pgacs_a', pgacs_a); call w('pgacs_b', pgacs_b)
  call w('pgacs_c', pgacs_c); call w('pgacs_d', pgacs_d)
  call w('pidep_a', pidep_a)
  call w('diffac_a', diffac_a); call w('diffac_b', diffac_b)
  call w('pgfrz_a', pgfrz_a);   call w('piacr_a', piacr_a)
  call w('pracw_a', pracw_a);   call w('psacw_a', psacw_a)
  call w('pgacw_a', pgacw_a)
  ! mp_wsm6_init's two words the WSM6 effective-radius diagnosis reads
  ! (mp_wsm6_effectRad.F90 USEs pidnc and pidn0s from mp_wsm6), with the
  ! arguments module_physics_init.F passes for WSM6, hail_opt 0.
  call mp_wsm6_init(mc_rhoair0, mc_rhowater, mc_rhosnow, mc_cliq, mc_cpv, 0, errmsg, errflg)
  if (errflg /= 0) error stop 'mp_wsm6_init failed'
  call w('wsm6_pidnc', wsm6_pidnc)
  call w('wsm6_pidn0s', wsm6_pidn0s)

contains

  subroutine w(name, value)
    character(len=*), intent(in) :: name
    real, intent(in) :: value
    write(*,'(a,1x,z8.8,1x,es16.9)') name, transfer(value, 0), value
  end subroutine w

end program dump_wsm6r_init
