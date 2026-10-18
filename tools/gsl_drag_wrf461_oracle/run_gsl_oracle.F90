program run_gsl_oracle
  ! gwd_opt = 3: WRF v4.6.1's own module_bl_gwdo_gsl (byte-unmodified, pinned
  ! in SOURCES.sha256), called exactly as module_pbl_driver.F:2157-2185 of
  ! WRF v4.6.1 calls it: PI=3.141592653, the namelist DX as a scalar, the
  ! PBL time step, spp_pbl with its pattern.  Columns: gsl_columns.F90.
  !
  ! For every (grid, dx, spp_pbl) case the routine runs twice on the same
  ! inputs, gwd_diags = 1 (the research diagnostics: the eight component
  ! tendencies and eight column stresses) and gwd_diags = 0 (what WRF runs by
  ! default).  The tendencies of the two calls are compared word for word
  ! here and the result is written (diags_agree = 1), so the fixture shows
  ! the diagnostics switch changes nothing else.
  !
  ! Writes <root>/<grid>/inputs and <root>/<grid>/dx<dx>[_spp]/ via oracle_io.
  use gsl_columns
  use module_bl_gwdo_gsl, only: gwdo_gsl
  use oracle_io
  implicit none
  type(column_set) :: a, b
  character(len=1024) :: root
  ! dx and the PBL time step of each case.  Grid A reaches every taper
  ! branch and both 0.01 thresholds: small-scale 0 at 1000 m, 0.00973 at
  ! 1009 m (off), 0.0108 at 1010 m (on); large-scale 0 at 3000 m, below 0.01
  ! at 3637 m (off), 0.0100098 at 3638 m (on); both partial at 5 and 9 km;
  ! small-scale 1 from 12 km, large-scale 1 from 13 km.
  real, parameter :: dx_a(12) = (/ 1000.0, 1009.0, 1010.0, 3000.0, 3637.0, &
       3638.0, 5000.0, 9000.0, 12000.0, 13000.0, 25000.0, 750.0 /)
  real, parameter :: dt_a(12) = (/ 6.0, 6.0, 6.0, 20.0, 20.0, 20.0, 30.0, &
       54.0, 72.0, 78.0, 150.0, 4.5 /)
  real, parameter :: dx_b(4) = (/ 3000.0, 9000.0, 15000.0, 30000.0 /)
  real, parameter :: dt_b(4) = (/ 18.0, 50.0, 90.0, 180.0 /)
  integer :: n

  call oracle_root(root)
  call build_grid_a(a)
  call build_grid_b(b)
  call write_inputs(a, 'A')
  call write_inputs(b, 'B')
  do n = 1, size(dx_a)
    call run_case(a, 'A', dx_a(n), dt_a(n), 0)
  end do
  call run_case(a, 'A', 3000.0, 20.0, 1)
  call run_case(a, 'A', 5000.0, 30.0, 1)
  call run_case(a, 'A', 13000.0, 78.0, 1)
  do n = 1, size(dx_b)
    call run_case(b, 'B', dx_b(n), dt_b(n), 0)
  end do
  call run_case(b, 'B', 9000.0, 50.0, 1)

contains

  subroutine write_inputs(s, grid)
    type(column_set), intent(in) :: s
    character(len=*), intent(in) :: grid
    integer :: nz
    nz = s%nz
    call oracle_open(grid // '/inputs')
    call oracle_put('u3d', s%u3d(:, 1:nz, 1))
    call oracle_put('v3d', s%v3d(:, 1:nz, 1))
    call oracle_put('t3d', s%t3d(:, 1:nz, 1))
    call oracle_put('qv3d', s%qv3d(:, 1:nz, 1))
    call oracle_put('p3d', s%p3d(:, 1:nz, 1))
    call oracle_put('p3di', s%p3di(:, 1:nz + 1, 1))
    call oracle_put('pi3d', s%pi3d(:, 1:nz, 1))
    call oracle_put('z', s%z(:, 1:nz, 1))
    call oracle_put('dz', s%dz(:, 1:nz, 1))
    call oracle_put('rublten0', s%rub0(:, 1:nz, 1))
    call oracle_put('rvblten0', s%rvb0(:, 1:nz, 1))
    call oracle_put('rthblten0', s%rthb0(:, 1:nz, 1))
    call oracle_put('pattern_spp_pbl', s%spp(:, 1:nz, 1))
    call oracle_put('znu', s%znu(1:nz))
    call oracle_put('znw', s%znw(1:nz + 1))
    call oracle_put('p_top', s%p_top)
    call oracle_put('var2d', s%var2d(:, 1))
    call oracle_put('oc12d', s%oc12d(:, 1))
    call oracle_put('oa', s%oa(:, 1, :))
    call oracle_put('ol', s%ol(:, 1, :))
    call oracle_put('var2dss', s%varss(:, 1))
    call oracle_put('oc12dss', s%ocss(:, 1))
    call oracle_put('oass', s%oass(:, 1, :))
    call oracle_put('olss', s%olss(:, 1, :))
    call oracle_put('sina', s%sina(:, 1))
    call oracle_put('cosa', s%cosa(:, 1))
    call oracle_put('xland', s%xland(:, 1))
    call oracle_put('br', s%br(:, 1))
    call oracle_put('pblh', s%pblh(:, 1))
    call oracle_put('kpbl', s%kpbl(:, 1))
    call oracle_put('ht', s%ht)
    call oracle_put('family', s%family)
    call oracle_close()
  end subroutine write_inputs

  subroutine call_gsl(s, dx, dtbl, spp_pbl, diags, rub, rvb, rthb, dtx, dty, &
                      dus, dvs)
    type(column_set), intent(in) :: s
    real, intent(in) :: dx, dtbl
    integer, intent(in) :: spp_pbl, diags
    real, intent(inout) :: rub(:,:,:), rvb(:,:,:), rthb(:,:,:)
    real, intent(inout) :: dtx(:,:,:,:), dty(:,:,:,:)   ! (ncol, kme, 1, 4)
    real, intent(inout) :: dus(:,:,:), dvs(:,:,:)       ! (ncol, 1, 4)
    integer :: ncol, nzi
    ncol = s%ncol; nzi = s%nz + 1
    call gwdo_gsl(u3d=s%u3d, v3d=s%v3d, t3d=s%t3d, qv3d=s%qv3d,             &
                  p3d=s%p3d, p3di=s%p3di, pi3d=s%pi3d, z=s%z,               &
                  rublten=rub, rvblten=rvb, rthblten=rthb,                  &
                  dtaux3d_ls=dtx(:,:,:,1), dtauy3d_ls=dty(:,:,:,1),         &
                  dtaux3d_bl=dtx(:,:,:,2), dtauy3d_bl=dty(:,:,:,2),         &
                  dtaux3d_ss=dtx(:,:,:,3), dtauy3d_ss=dty(:,:,:,3),         &
                  dtaux3d_fd=dtx(:,:,:,4), dtauy3d_fd=dty(:,:,:,4),         &
                  dusfcg_ls=dus(:,:,1), dvsfcg_ls=dvs(:,:,1),               &
                  dusfcg_bl=dus(:,:,2), dvsfcg_bl=dvs(:,:,2),               &
                  dusfcg_ss=dus(:,:,3), dvsfcg_ss=dvs(:,:,3),               &
                  dusfcg_fd=dus(:,:,4), dvsfcg_fd=dvs(:,:,4),               &
                  xland=s%xland, br=s%br,                                   &
                  var2d=s%var2d, oc12d=s%oc12d,                             &
                  oa2d1=s%oa(:,:,1), oa2d2=s%oa(:,:,2),                     &
                  oa2d3=s%oa(:,:,3), oa2d4=s%oa(:,:,4),                     &
                  ol2d1=s%ol(:,:,1), ol2d2=s%ol(:,:,2),                     &
                  ol2d3=s%ol(:,:,3), ol2d4=s%ol(:,:,4),                     &
                  var2dss=s%varss, oc12dss=s%ocss,                          &
                  oa2d1ss=s%oass(:,:,1), oa2d2ss=s%oass(:,:,2),             &
                  oa2d3ss=s%oass(:,:,3), oa2d4ss=s%oass(:,:,4),             &
                  ol2d1ss=s%olss(:,:,1), ol2d2ss=s%olss(:,:,2),             &
                  ol2d3ss=s%olss(:,:,3), ol2d4ss=s%olss(:,:,4),             &
                  sina=s%sina, cosa=s%cosa, znu=s%znu, znw=s%znw,           &
                  p_top=s%p_top, dz=s%dz, pblh=s%pblh,                      &
                  cp=cp, g=g, rd=r_d, rv=r_v, ep1=ep_1, pi=3.141592653,     &
                  dt=dtbl, dx=dx, kpbl2d=s%kpbl, itimestep=7,               &
                  gwd_opt=3, gwd_diags=diags,                               &
                  spp_pbl=spp_pbl, pattern_spp_pbl=s%spp,                   &
                  ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=nzi,        &
                  ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nzi,            &
                  its=1, ite=ncol, jts=1, jte=1, kts=1, kte=s%nz)
  end subroutine call_gsl

  subroutine run_case(s, grid, dx, dtbl, spp_pbl)
    type(column_set), intent(in) :: s
    character(len=*), intent(in) :: grid
    real, intent(in) :: dx, dtbl
    integer, intent(in) :: spp_pbl
    real, allocatable :: rub(:,:,:), rvb(:,:,:), rthb(:,:,:)
    real, allocatable :: rub0(:,:,:), rvb0(:,:,:), rthb0(:,:,:)
    real, allocatable :: dtx(:,:,:,:), dty(:,:,:,:), dus(:,:,:), dvs(:,:,:)
    real, allocatable :: dtx0(:,:,:,:), dty0(:,:,:,:), dus0(:,:,:), dvs0(:,:,:)
    character(len=8), parameter :: comp(4) = (/ 'ls', 'bl', 'ss', 'fd' /)
    character(len=128) :: name
    integer :: ncol, nz, m, agree
    ncol = s%ncol; nz = s%nz
    allocate(rub, mold=s%rub0); allocate(rvb, mold=s%rub0)
    allocate(rthb, mold=s%rub0)
    allocate(rub0, mold=s%rub0); allocate(rvb0, mold=s%rub0)
    allocate(rthb0, mold=s%rub0)
    allocate(dtx(ncol, nz + 1, 1, 4), dty(ncol, nz + 1, 1, 4))
    allocate(dtx0(ncol, nz + 1, 1, 4), dty0(ncol, nz + 1, 1, 4))
    allocate(dus(ncol, 1, 4), dvs(ncol, 1, 4), dus0(ncol, 1, 4), dvs0(ncol, 1, 4))

    ! gwd_diags = 1
    rub = s%rub0; rvb = s%rvb0; rthb = s%rthb0
    dtx = 0.0; dty = 0.0; dus = 0.0; dvs = 0.0
    call call_gsl(s, dx, dtbl, spp_pbl, 1, rub, rvb, rthb, dtx, dty, dus, dvs)
    ! gwd_diags = 0 on the same inputs
    rub0 = s%rub0; rvb0 = s%rvb0; rthb0 = s%rthb0
    dtx0 = 0.0; dty0 = 0.0; dus0 = 0.0; dvs0 = 0.0
    call call_gsl(s, dx, dtbl, spp_pbl, 0, rub0, rvb0, rthb0, dtx0, dty0, &
                  dus0, dvs0)
    agree = 1
    if (any(transfer(rub(:, 1:nz, 1), [0]) /= transfer(rub0(:, 1:nz, 1), [0]))) agree = 0
    if (any(transfer(rvb(:, 1:nz, 1), [0]) /= transfer(rvb0(:, 1:nz, 1), [0]))) agree = 0
    if (any(transfer(rthb(:, 1:nz, 1), [0]) /= transfer(rthb0(:, 1:nz, 1), [0]))) agree = 0

    if (spp_pbl == 1) then
      write(name, '(A,A,I0,A)') grid, '/dx', nint(dx), '_spp'
    else
      write(name, '(A,A,I0)') grid, '/dx', nint(dx)
    end if
    call oracle_open(trim(name))
    call oracle_put('dx', dx)
    call oracle_put('dt', dtbl)
    call oracle_put('spp_pbl', spp_pbl)
    call oracle_put('diags_agree', agree)
    call oracle_put('rublten', rub(:, 1:nz, 1))
    call oracle_put('rvblten', rvb(:, 1:nz, 1))
    call oracle_put('rthblten', rthb(:, 1:nz, 1))
    do m = 1, 4
      call oracle_put('dtaux3d_' // trim(comp(m)), dtx(:, 1:nz, 1, m))
      call oracle_put('dtauy3d_' // trim(comp(m)), dty(:, 1:nz, 1, m))
      call oracle_put('dusfcg_' // trim(comp(m)), dus(:, 1, m))
      call oracle_put('dvsfcg_' // trim(comp(m)), dvs(:, 1, m))
    end do
    call oracle_close()
  end subroutine run_case

end program run_gsl_oracle
