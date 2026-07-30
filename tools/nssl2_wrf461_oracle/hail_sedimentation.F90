program nssl2_hail_sedimentation_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, sediment1d
  implicit none

  integer, parameter :: nx = 6, ny = 1, nz = 12, na = 40
  integer, parameter :: lhl = 8, lnhl = 15, lvhl = 17
  integer :: i, k, icase, unit
  real :: nssl_params(20), dt, diameter, volume, amplitude
  real :: particle_density
  real :: an(nx,ny,nz,na), density(nx,ny,nz)
  real :: layer_depth(nx,ny,nz), inverse_depth(nx,ny,nz)
  real :: temperature(nx,ny,nz), ice_number(nx,ny,nz)
  real :: xfall(nx,ny,na)
  real :: before_qh(nx,ny,nz), before_nh(nx,ny,nz)
  real :: before_volh(nx,ny,nz)
  real, parameter :: steps(4) = [0.5, 10.0, 60.0, 300.0]
  double precision :: timesed1, timesed2, timesed3, zmaxsed, timesetvt
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_hail_sedimentation_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 800.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,column,k,dt_s,rho_kg_m3,dz_m,qh_before,nh_before_per_kg,qvolh_before_m3_per_kg,qh_after,nh_after_per_kg,qvolh_after_m3_per_kg,hailncv_kg_m2'

  do icase = 1, size(steps)
     dt = steps(icase)
     an = 0.0
     density = 1.0
     layer_depth = 100.0
     inverse_depth = 0.01
     temperature = 245.0
     ice_number = 0.0
     xfall = 0.0

     do k = 1, nz
        do i = 1, nx
           density(i,1,k) = 1.28 - 0.052*real(k-1) + 0.007*real(i-1)
           layer_depth(i,1,k) = 31.0 + 17.0*real(k) + 3.5*real(i-1)
           inverse_depth(i,1,k) = 1.0/layer_depth(i,1,k)
           temperature(i,1,k) = 258.0 - 1.2*real(k-1)
           particle_density = 500.0 + 100.0*real(mod(i+k,5))

           select case (i)
           case (1)
              amplitude = max(0.0, 1.0 - abs(real(k)-6.5)/4.0)
              an(i,1,k,lhl) = 1.2e-3*amplitude
              diameter = 0.15e-3 + 0.22e-3*real(k-1)
           case (2)
              amplitude = exp(-0.24*real(k-1))
              an(i,1,k,lhl) = 3.5e-3*amplitude
              diameter = 1.5e-3 + 0.70e-3*real(mod(k,5))
           case (3)
              if (k >= 7 .and. k <= 10) then
                 an(i,1,k,lhl) = 3.0e-4*real(k-5)
              endif
              diameter = 8.0e-3 + 3.0e-3*real(k-7)
           case (4)
              if (mod(k,3) /= 0) an(i,1,k,lhl) = 7.0e-4*real(mod(k,4)+1)
              diameter = merge(0.15e-3, 50.0e-3, mod(k,2) == 0)
           case (5)
              an(i,1,k,lhl) = 10.0**(-12.0 + real(mod(k,7)))
              diameter = 0.45e-3 + 7.0e-3*real(mod(k,6))
           case (6)
              an(i,1,k,lhl) = 3.0e-7*1.9**real(k-1)
              diameter = 0.35e-3*1.52**real(k-1)
           end select

           if (an(i,1,k,lhl) > 0.0) then
              volume = 0.523599*diameter**3
              before_nh(i,1,k) = an(i,1,k,lhl)/(particle_density*volume)
              if (i == 6 .and. mod(k,5) == 0) before_nh(i,1,k) = 0.0
              before_volh(i,1,k) = an(i,1,k,lhl)/particle_density
              an(i,1,k,lnhl) = before_nh(i,1,k)*density(i,1,k)
              an(i,1,k,lvhl) = before_volh(i,1,k)*density(i,1,k)
           else
              before_nh(i,1,k) = 0.0
              before_volh(i,1,k) = 0.0
           endif
        enddo
     enddo
     before_qh = an(:,:,:,lhl)
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
           write(unit,'(3(I0,","),9(ES24.16E3,","),ES24.16E3)') &
                icase, i, k, dt, density(i,1,k), layer_depth(i,1,k), &
                before_qh(i,1,k), before_nh(i,1,k),                  &
                before_volh(i,1,k), an(i,1,k,lhl),                  &
                an(i,1,k,lnhl)/density(i,1,k),                      &
                an(i,1,k,lvhl)/density(i,1,k),                      &
                dt*density(i,1,1)*xfall(i,1,lhl)
        enddo
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_HAIL_SEDIMENTATION_ORACLE_COMPLETE', &
       trim(output_path)
end program nssl2_hail_sedimentation_oracle
