program nssl2_snow_sedimentation_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, sediment1d
  implicit none

  integer, parameter :: nx = 6, ny = 1, nz = 12, na = 40
  integer, parameter :: ls = 6, lns = 13
  integer :: i, k, icase, unit
  real :: nssl_params(20), dt, diameter, volume, amplitude
  real :: an(nx,ny,nz,na), density(nx,ny,nz)
  real :: layer_depth(nx,ny,nz), inverse_depth(nx,ny,nz)
  real :: temperature(nx,ny,nz), ice_number(nx,ny,nz)
  real :: xfall(nx,ny,na)
  real :: before_qs(nx,ny,nz), before_ns(nx,ny,nz)
  real, parameter :: steps(4) = [0.5, 10.0, 60.0, 300.0]
  double precision :: timesed1, timesed2, timesed3, zmaxsed, timesetvt
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_snow_sedimentation_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,column,k,dt_s,rho_kg_m3,dz_m,qs_before,ns_before_per_kg,qs_after,ns_after_per_kg,snowncv_kg_m2'

  do icase = 1, size(steps)
     dt = steps(icase)
     an = 0.0
     density = 1.0
     layer_depth = 100.0
     inverse_depth = 0.01
     temperature = 258.0
     ice_number = 0.0
     xfall = 0.0

     do k = 1, nz
        do i = 1, nx
           density(i,1,k) = 1.28 - 0.052*real(k-1) + 0.007*real(i-1)
           layer_depth(i,1,k) = 31.0 + 17.0*real(k) + 3.5*real(i-1)
           inverse_depth(i,1,k) = 1.0/layer_depth(i,1,k)
           temperature(i,1,k) = 262.0 - 0.9*real(k-1)

           select case (i)
           case (1)
              amplitude = max(0.0, 1.0 - abs(real(k)-6.5)/4.0)
              an(i,1,k,ls) = 7.0e-4*amplitude
              diameter = 15.0e-6 + 160.0e-6*real(k-1)
           case (2)
              amplitude = exp(-0.27*real(k-1))
              an(i,1,k,ls) = 2.0e-3*amplitude
              diameter = 0.25e-3 + 0.15e-3*real(mod(k,5))
           case (3)
              if (k >= 7 .and. k <= 10) then
                 an(i,1,k,ls) = 2.0e-4*real(k-5)
              endif
              diameter = 1.5e-3 + 0.30e-3*real(k-7)
           case (4)
              if (mod(k,3) /= 0) an(i,1,k,ls) = 4.0e-4*real(mod(k,4)+1)
              diameter = merge(5.0e-6, 12.0e-3, mod(k,2) == 0)
           case (5)
              an(i,1,k,ls) = 10.0**(-14.0 + real(mod(k,7)))
              diameter = 10.0e-6 + 0.80e-3*real(mod(k,6))
           case (6)
              an(i,1,k,ls) = 2.0e-7*1.7**real(k-1)
              diameter = 0.05e-3*1.45**real(k-1)
           end select

           if (an(i,1,k,ls) > 0.0) then
              volume = 0.523599*diameter**3
              before_ns(i,1,k) = an(i,1,k,ls)/(100.0*volume)
              ! Exercise positive snow with an absent number moment; SETVTZ
              ! diagnoses a bounded local distribution before sedimentation.
              if (i == 6 .and. mod(k,5) == 0) before_ns(i,1,k) = 0.0
              an(i,1,k,lns) = before_ns(i,1,k)*density(i,1,k)
           else
              before_ns(i,1,k) = 0.0
           endif
        enddo
     enddo
     before_qs = an(:,:,:,ls)
     timesed1 = 0.0d0
     timesed2 = 0.0d0
     timesed3 = 0.0d0
     zmaxsed = 0.0d0
     timesetvt = 0.0d0

     call sediment1d(dt,nx,ny,nz,an,na,0,0,xfall,density,      &
          layer_depth,inverse_depth,temperature,ice_number,2,1, &
          1,1,timesed1,timesed2,timesed3,zmaxsed,timesetvt)

     do k = 1, nz
        do i = 1, nx
           write(unit,'(3(I0,","),7(ES24.16E3,","),ES24.16E3)') &
                icase, i, k, dt, density(i,1,k), layer_depth(i,1,k), &
                before_qs(i,1,k), before_ns(i,1,k), an(i,1,k,ls),  &
                an(i,1,k,lns)/density(i,1,k),                     &
                dt*density(i,1,1)*xfall(i,1,ls)
        enddo
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_SNOW_SEDIMENTATION_ORACLE_COMPLETE', &
       trim(output_path)
end program nssl2_snow_sedimentation_oracle
