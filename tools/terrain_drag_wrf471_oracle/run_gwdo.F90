program run_gwdo
  ! gwd_opt = 1: WRF v4.7.1's own module_bl_gwdo (gwdo -> bl_gwdo_run in
  ! phys/physics_mmm/bl_gwdo.F90, both byte-unmodified), called exactly as
  ! module_pbl_driver.F:2131-2151 calls it, on the gwd_cases columns at three
  ! grid lengths.  WRF passes DX=dx2d, which compute_2d_dx_area sets to the
  ! namelist dx (module_physics_init.F:5683-5688, the #else branch), and
  ! PI=3.141592653.
  !
  ! Writes fixtures/gwdo/inputs and fixtures/gwdo/dx<dx>: RUBLTEN/RVBLTEN after
  ! the call (the PBL tendencies it adds to, which the inputs carry before it),
  ! DTAUX3D/DTAUY3D and DUSFCG/DVSFCG.
  use gwd_cases
  use module_bl_gwdo, only: gwdo
  use oracle_io
  implicit none
  real :: rublten(ncol, nzi, 1), rvblten(ncol, nzi, 1)
  real :: dtaux3d(ncol, nzi, 1), dtauy3d(ncol, nzi, 1)
  real :: dusfcg(ncol, 1), dvsfcg(ncol, 1), dx2d(ncol, 1)
  real, parameter :: dts(3) = (/ 3000.0, 12000.0, 30000.0 /)
  real, parameter :: dtbl = 20.0
  character(len=1024) :: root
  character(len=256) :: errmsg
  character(len=64) :: name
  integer :: errflg, n

  call oracle_root(root)
  call build_columns()
  call put_inputs('gwdo/inputs')

  do n = 1, size(dts)
    rublten = rub0; rvblten = rvb0
    dtaux3d = 0.0; dtauy3d = 0.0; dusfcg = 0.0; dvsfcg = 0.0
    dx2d = dts(n)
    errmsg = ''; errflg = -1
    call gwdo(u3d=u3d, v3d=v3d, t3d=t3d, qv3d=qv3d,                         &
              p3d=p3d, p3di=p3di, pi3d=pi3d, z=z,                           &
              rublten=rublten, rvblten=rvblten,                             &
              dtaux3d=dtaux3d, dtauy3d=dtauy3d,                             &
              dusfcg=dusfcg, dvsfcg=dvsfcg,                                 &
              var2d=var2d, oc12d=oc12d,                                     &
              oa2d1=oa(:, :, 1), oa2d2=oa(:, :, 2),                         &
              oa2d3=oa(:, :, 3), oa2d4=oa(:, :, 4),                         &
              ol2d1=ol(:, :, 1), ol2d2=ol(:, :, 2),                         &
              ol2d3=ol(:, :, 3), ol2d4=ol(:, :, 4),                         &
              sina=sina, cosa=cosa, znu=znu, znw=znw, p_top=p_top,          &
              cp=cp, g=g, rd=r_d, rv=r_v, ep1=ep_1, pi=3.141592653,         &
              dt=dtbl, dx=dx2d, kpbl2d=kpbl, itimestep=7,                   &
              ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=nzi,            &
              ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nzi,                &
              its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz,                 &
              errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) then
      write(*, '(A,A)') 'gwdo errmsg: ', trim(errmsg)
      error stop 3
    end if
    write(name, '(A,I0)') 'gwdo/dx', nint(dts(n))
    call oracle_open(trim(name))
    call oracle_put('dx', dts(n))
    call oracle_put('dt', dtbl)
    call oracle_put('rublten', rublten(:, 1:nz, 1))
    call oracle_put('rvblten', rvblten(:, 1:nz, 1))
    call oracle_put('dtaux3d', dtaux3d(:, 1:nz, 1))
    call oracle_put('dtauy3d', dtauy3d(:, 1:nz, 1))
    call oracle_put('dusfcg', dusfcg(:, 1))
    call oracle_put('dvsfcg', dvsfcg(:, 1))
    call oracle_close()
  end do

contains

  subroutine put_inputs(case)
    character(len=*), intent(in) :: case
    call oracle_open(case)
    call oracle_put('u3d', u3d(:, 1:nz, 1)); call oracle_put('v3d', v3d(:, 1:nz, 1))
    call oracle_put('t3d', t3d(:, 1:nz, 1)); call oracle_put('qv3d', qv3d(:, 1:nz, 1))
    call oracle_put('p3d', p3d(:, 1:nz, 1)); call oracle_put('p3di', p3di(:, 1:nzi, 1))
    call oracle_put('pi3d', pi3d(:, 1:nz, 1)); call oracle_put('z', z(:, 1:nz, 1))
    call oracle_put('dz', dz(:, 1:nz, 1))
    call oracle_put('rublten0', rub0(:, 1:nz, 1)); call oracle_put('rvblten0', rvb0(:, 1:nz, 1))
    call oracle_put('rthblten0', rthb0(:, 1:nz, 1))
    call oracle_put('znu', znu(1:nz)); call oracle_put('znw', znw)
    call oracle_put('var2d', var2d(:, 1)); call oracle_put('oc12d', oc12d(:, 1))
    call oracle_put('oa', oa(:, 1, :)); call oracle_put('ol', ol(:, 1, :))
    call oracle_put('var2dss', varss(:, 1)); call oracle_put('oc12dss', ocss(:, 1))
    call oracle_put('oass', oass(:, 1, :)); call oracle_put('olss', olss(:, 1, :))
    call oracle_put('sina', sina(:, 1)); call oracle_put('cosa', cosa(:, 1))
    call oracle_put('xland', xland(:, 1)); call oracle_put('br', br(:, 1))
    call oracle_put('pblh', pblh(:, 1)); call oracle_put('kpbl', kpbl(:, 1))
    call oracle_put('p_top', p_top)
    call oracle_close()
  end subroutine put_inputs

end program run_gwdo
