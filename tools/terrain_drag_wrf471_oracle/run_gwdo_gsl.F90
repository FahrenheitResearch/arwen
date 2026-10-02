program run_gwdo_gsl
  ! gwd_opt = 3: WRF v4.7.1's own module_bl_gwdo_gsl (byte-unmodified), the
  ! GSL drag suite, called exactly as module_pbl_driver.F:2152-2185 calls it,
  ! on the gwd_cases columns at five grid lengths that reach every taper
  ! branch: 1 km (both tapers 0: nothing runs), 3 km (large-scale 0,
  ! small-scale 0.727), 5 km and 9 km (both partial), 15 km (both 1).
  ! WRF passes the namelist DX (a scalar here), PI=3.141592653, spp_pbl = 0
  ! and gwd_diags from the namelist; the oracle sets gwd_diags = 1 so the
  ! four components are recorded separately.
  !
  ! Writes fixtures/gwdo_gsl/dx<dx>: RUBLTEN/RVBLTEN/RTHBLTEN after the call
  ! (the inputs carry them before it, in fixtures/gwdo/inputs), the eight
  ! DTAUX3D/DTAUY3D components and the eight column stresses.
  use gwd_cases
  use module_bl_gwdo_gsl, only: gwdo_gsl
  use oracle_io
  implicit none
  real :: rublten(ncol, nzi, 1), rvblten(ncol, nzi, 1), rthblten(ncol, nzi, 1)
  real, dimension(ncol, nzi, 1) :: dtaux_ls, dtauy_ls, dtaux_bl, dtauy_bl, &
                                   dtaux_ss, dtauy_ss, dtaux_fd, dtauy_fd
  real, dimension(ncol, 1) :: dus_ls, dvs_ls, dus_bl, dvs_bl, dus_ss, dvs_ss, &
                              dus_fd, dvs_fd
  real :: pattern(ncol, nzi, 1)
  real, parameter :: dxs(5) = (/ 1000.0, 3000.0, 5000.0, 9000.0, 15000.0 /)
  real, parameter :: dtbl = 20.0
  character(len=1024) :: root
  character(len=64) :: name
  integer :: n

  call oracle_root(root)
  call build_columns()
  pattern = 0.0

  do n = 1, size(dxs)
    rublten = rub0; rvblten = rvb0; rthblten = rthb0
    dtaux_ls = 0.0; dtauy_ls = 0.0; dtaux_bl = 0.0; dtauy_bl = 0.0
    dtaux_ss = 0.0; dtauy_ss = 0.0; dtaux_fd = 0.0; dtauy_fd = 0.0
    dus_ls = 0.0; dvs_ls = 0.0; dus_bl = 0.0; dvs_bl = 0.0
    dus_ss = 0.0; dvs_ss = 0.0; dus_fd = 0.0; dvs_fd = 0.0
    call gwdo_gsl(u3d=u3d, v3d=v3d, t3d=t3d, qv3d=qv3d,                     &
                  p3d=p3d, p3di=p3di, pi3d=pi3d, z=z,                       &
                  rublten=rublten, rvblten=rvblten, rthblten=rthblten,      &
                  dtaux3d_ls=dtaux_ls, dtauy3d_ls=dtauy_ls,                 &
                  dtaux3d_bl=dtaux_bl, dtauy3d_bl=dtauy_bl,                 &
                  dtaux3d_ss=dtaux_ss, dtauy3d_ss=dtauy_ss,                 &
                  dtaux3d_fd=dtaux_fd, dtauy3d_fd=dtauy_fd,                 &
                  dusfcg_ls=dus_ls, dvsfcg_ls=dvs_ls,                       &
                  dusfcg_bl=dus_bl, dvsfcg_bl=dvs_bl,                       &
                  dusfcg_ss=dus_ss, dvsfcg_ss=dvs_ss,                       &
                  dusfcg_fd=dus_fd, dvsfcg_fd=dvs_fd,                       &
                  xland=xland, br=br,                                       &
                  var2d=var2d, oc12d=oc12d,                                 &
                  oa2d1=oa(:, :, 1), oa2d2=oa(:, :, 2),                     &
                  oa2d3=oa(:, :, 3), oa2d4=oa(:, :, 4),                     &
                  ol2d1=ol(:, :, 1), ol2d2=ol(:, :, 2),                     &
                  ol2d3=ol(:, :, 3), ol2d4=ol(:, :, 4),                     &
                  var2dss=varss, oc12dss=ocss,                              &
                  oa2d1ss=oass(:, :, 1), oa2d2ss=oass(:, :, 2),             &
                  oa2d3ss=oass(:, :, 3), oa2d4ss=oass(:, :, 4),             &
                  ol2d1ss=olss(:, :, 1), ol2d2ss=olss(:, :, 2),             &
                  ol2d3ss=olss(:, :, 3), ol2d4ss=olss(:, :, 4),             &
                  sina=sina, cosa=cosa, znu=znu, znw=znw, p_top=p_top,      &
                  dz=dz, pblh=pblh,                                         &
                  cp=cp, g=g, rd=r_d, rv=r_v, ep1=ep_1, pi=3.141592653,     &
                  dt=dtbl, dx=dxs(n), kpbl2d=kpbl, itimestep=7,             &
                  gwd_opt=3, gwd_diags=1,                                   &
                  spp_pbl=0, pattern_spp_pbl=pattern,                       &
                  ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=nzi,        &
                  ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nzi,            &
                  its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)
    write(name, '(A,I0)') 'gwdo_gsl/dx', nint(dxs(n))
    call oracle_open(trim(name))
    call oracle_put('dx', dxs(n))
    call oracle_put('dt', dtbl)
    call oracle_put('rublten', rublten(:, 1:nz, 1))
    call oracle_put('rvblten', rvblten(:, 1:nz, 1))
    call oracle_put('rthblten', rthblten(:, 1:nz, 1))
    call oracle_put('dtaux3d_ls', dtaux_ls(:, 1:nz, 1))
    call oracle_put('dtauy3d_ls', dtauy_ls(:, 1:nz, 1))
    call oracle_put('dtaux3d_bl', dtaux_bl(:, 1:nz, 1))
    call oracle_put('dtauy3d_bl', dtauy_bl(:, 1:nz, 1))
    call oracle_put('dtaux3d_ss', dtaux_ss(:, 1:nz, 1))
    call oracle_put('dtauy3d_ss', dtauy_ss(:, 1:nz, 1))
    call oracle_put('dtaux3d_fd', dtaux_fd(:, 1:nz, 1))
    call oracle_put('dtauy3d_fd', dtauy_fd(:, 1:nz, 1))
    call oracle_put('dusfcg_ls', dus_ls(:, 1)); call oracle_put('dvsfcg_ls', dvs_ls(:, 1))
    call oracle_put('dusfcg_bl', dus_bl(:, 1)); call oracle_put('dvsfcg_bl', dvs_bl(:, 1))
    call oracle_put('dusfcg_ss', dus_ss(:, 1)); call oracle_put('dvsfcg_ss', dvs_ss(:, 1))
    call oracle_put('dusfcg_fd', dus_fd(:, 1)); call oracle_put('dvsfcg_fd', dvs_fd(:, 1))
    call oracle_close()
  end do

end program run_gwdo_gsl
