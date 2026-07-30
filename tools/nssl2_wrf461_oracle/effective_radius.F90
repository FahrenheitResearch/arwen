program nssl2_effective_radius_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, calc_eff_radius
  implicit none

  integer, parameter :: nx = 4, ny = 1, nz = 12
  integer :: i, k, unit, case_id
  real :: nssl_params(20)
  real :: density(nx,ny,nz)
  real :: qc(nx,nz), nc(nx,nz), qi(nx,nz), ni(nx,nz)
  real :: qs(nx,nz), ns(nx,nz)
  real :: raw_cloud(nx,ny,nz), raw_ice(nx,ny,nz), raw_snow(nx,ny,nz)
  real :: re_cloud(nx,ny,nz), re_ice(nx,ny,nz), re_snow(nx,ny,nz)
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_effective_radius_oracle OUTPUT.csv'
  endif

  ! Exact WRF v4.6.1 Registry defaults forwarded by module_physics_init.F.
  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  qc = 0.0
  nc = 0.0
  qi = 0.0
  ni = 0.0
  qs = 0.0
  ns = 0.0
  raw_cloud = 2.51e-6
  raw_ice = 10.01e-6
  raw_snow = 25.0e-6

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k - 1) * nx + i - 1, 8)
        density(i,1,k) = 1.225 * exp(-real((k - 1) * 500) / 8000.0)
        select case (case_id)
        case (0)
           ! Hydrometeor-free cells retain the driver's background radii.
        case (1)
           ! Positive but sub-threshold mass also retains background values.
           qc(i,k) = 0.5e-13
           nc(i,k) = 1.0e8
           qi(i,k) = 0.5e-13
           ni(i,k) = 1.0e8
           qs(i,k) = 0.5e-13
           ns(i,k) = 1.0e8
        case default
           qc(i,k) = 10.0 ** (-6.0 + 0.45 * real(case_id))
           nc(i,k) = 10.0 ** (3.0 + 1.25 * real(case_id))
           qi(i,k) = 0.7 * qc(i,k)
           ni(i,k) = 0.18 * nc(i,k)
           qs(i,k) = 1.6 * qc(i,k)
           ns(i,k) = 0.012 * nc(i,k)
        end select
     enddo
  enddo

  call calc_eff_radius(nx,ny,nz,1,1,0,0,                        &
       t1=raw_cloud,t2=raw_ice,t3=raw_snow,                     &
       qcw=qc,qci=qi,qsw=qs,ccw=nc,cci=ni,csw=ns,dn=density)

  re_cloud = max(2.51e-6, min(raw_cloud, 50.0e-6))
  re_ice = max(10.01e-6, min(raw_ice, 125.0e-6))
  re_snow = max(25.0e-6, min(raw_snow, 999.0e-6))

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,rho_kg_m3,qc,nc_per_kg,qi,ni_per_kg,qs,ns_per_kg,raw_cloud_m,raw_ice_m,raw_snow_m,re_cloud_m,re_ice_m,re_snow_m'
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k - 1) * nx + i - 1, 8)
        write(unit,'(I0,2(",",I0),13(",",ES24.16E3))')             &
             case_id, i, k, density(i,1,k), qc(i,k), nc(i,k),       &
             qi(i,k), ni(i,k), qs(i,k), ns(i,k),                    &
             raw_cloud(i,1,k), raw_ice(i,1,k), raw_snow(i,1,k),    &
             re_cloud(i,1,k), re_ice(i,1,k), re_snow(i,1,k)
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_EFFECTIVE_RADIUS_ORACLE_COMPLETE', &
       trim(output_path)
end program nssl2_effective_radius_oracle
