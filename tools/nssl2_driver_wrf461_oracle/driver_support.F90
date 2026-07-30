program nssl2_driver_support_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, calcnfromq, &
       calcnfromcuten, sediment1d
  implicit none

  integer, parameter :: nx = 6, ny = 1, nz = 14, na = 40
  integer, parameter :: nfield = 16, nruns = 3
  integer, parameter :: fqv = 1, fqc = 2, fqr = 3, fqi = 4
  integer, parameter :: fqs = 5, fqg = 6, fqh = 7, fnc = 8
  integer, parameter :: fnr = 9, fni = 10, fns = 11, fng = 12
  integer, parameter :: fnh = 13, fnn = 14, fvg = 15, fvh = 16
  integer, parameter :: lv = 2, lc = 3, lr = 4, li = 5
  integer, parameter :: ls = 6, lh = 7, lhl = 8, lccn = 9
  integer, parameter :: lnc = 10, lnr = 11, lni = 12, lns = 13
  integer, parameter :: lnh = 14, lnhl = 15, lvh = 16, lvhl = 17

  integer :: i, k, irun, unit, first_step, cu_used
  real :: dt, amplitude, diameter, volume, particle_density
  real :: nssl_params(20)
  real :: registry(nx,nz,nfield), rates(nx,nz,4)
  real :: an(nx,ny,nz,na), ancuten(nx,ny,nz,na)
  real :: density2(nx,nz+1), density3(nx,ny,nz)
  real :: layer_depth(nx,ny,nz), inverse_depth(nx,ny,nz)
  real :: temperature(nx,ny,nz), ice_nucleation(nx,ny,nz)
  real :: xfall(nx,ny,na)
  real :: rainnc(nx), rainncv(nx), snownc(nx), snowncv(nx)
  real :: graupelnc(nx), graupelncv(nx), hailnc(nx), hailncv(nx)
  real :: sr(nx), ice_export(nx)
  real, parameter :: steps(nruns) = [0.5, 60.0, 240.0]
  double precision :: timesed1, timesed2, timesed3, zmaxsed, timesetvt
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_driver_support_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'phase,run,case,column,k,first_step,cu_used,dt_s,' // &
       'rho_kg_m3,dz_m,temperature_k,qrcuten,qscuten,qicuten,qccuten,' // &
       'qv,qc,qr,qi,qs,qg,qh,qndrop_per_kg,qnr_per_kg,qni_per_kg,' // &
       'qns_per_kg,qng_per_kg,qnh_per_kg,qnn_per_kg,' // &
       'qvolg_m3_per_kg,qvolh_m3_per_kg,' // &
       'rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,' // &
       'hailnc,hailncv,sr,ice_surface_export'

  do irun = 1, nruns
     dt = steps(irun)
     first_step = merge(1, 0, irun /= 2)
     cu_used = merge(1, 0, irun >= 2)
     registry = 0.0
     rates = 0.0
     an = 0.0
     ancuten = 0.0
     density2 = 0.0
     density3 = 0.0
     layer_depth = 0.0
     inverse_depth = 0.0
     temperature = 260.0
     ice_nucleation = 0.0
     xfall = 0.0

     do k = 1, nz + 1
        do i = 1, nx
           density2(i,k) = 1.26 - 0.035*real(k-1) + 0.006*real(i-1)
        enddo
     enddo
     do k = 1, nz
        do i = 1, nx
           density3(i,1,k) = density2(i,k)
           if (irun == 1) then
              layer_depth(i,1,k) = 240.0 + 13.0*real(k) + 2.0*real(i)
           else
              layer_depth(i,1,k) = 20.0 + 7.0*real(mod(k+i,5))
           endif
           inverse_depth(i,1,k) = 1.0/layer_depth(i,1,k)
           registry(i,k,fqv) = 0.008 + 0.00005*real(k)
           registry(i,k,fnn) = 4.5e8
           amplitude = exp(-0.18*(real(k)-8.0)**2)

           select case (i)
           case (1)
              ! Empty/no-op column, including zero KF rates.

           case (2)
              ! Mass-only initial state. calcnfromq must diagnose every number
              ! moment and both missing density moments on first-step runs.
              registry(i,k,fqc) = 1.5e-4*amplitude
              registry(i,k,fqr) = 3.0e-4*amplitude
              registry(i,k,fqi) = 1.2e-4*amplitude
              registry(i,k,fqs) = 2.4e-4*amplitude
              registry(i,k,fqg) = 3.2e-4*amplitude
              registry(i,k,fqh) = 2.1e-4*amplitude

           case (3)
              ! Fully initialized, variable-density graupel/hail column.
              registry(i,k,fqc) = 1.0e-4*amplitude
              registry(i,k,fqr) = 6.0e-4*amplitude
              registry(i,k,fqi) = 1.5e-4*amplitude
              registry(i,k,fqs) = 4.0e-4*amplitude
              registry(i,k,fqg) = 7.0e-4*amplitude
              registry(i,k,fqh) = 8.0e-4*amplitude
              diameter = 0.20e-3 + 0.025e-3*real(k)
              volume = 0.523599*diameter**3
              registry(i,k,fnc) = registry(i,k,fqc)/(1000.0*volume)
              registry(i,k,fni) = registry(i,k,fqi)/(900.0*volume)
              diameter = 0.35e-3 + 0.06e-3*real(k)
              volume = 0.523599*diameter**3
              registry(i,k,fnr) = registry(i,k,fqr)/(1000.0*volume)
              registry(i,k,fns) = registry(i,k,fqs)/(100.0*volume)
              particle_density = 260.0 + 23.0*real(k)
              registry(i,k,fng) = registry(i,k,fqg)/(particle_density*volume)
              registry(i,k,fvg) = registry(i,k,fqg)/particle_density
              particle_density = 610.0 + 17.0*real(k)
              registry(i,k,fnh) = registry(i,k,fqh)/(particle_density*volume)
              registry(i,k,fvh) = registry(i,k,fqh)/particle_density

           case (4)
              ! Dynamics-applied KF mass with all four q*cuten rates available.
              registry(i,k,fqc) = 2.0e-5 + 0.5e-5*amplitude
              registry(i,k,fqr) = 4.0e-5 + 2.0e-5*amplitude
              registry(i,k,fqi) = 3.0e-5 + 0.7e-5*amplitude
              registry(i,k,fqs) = 5.0e-5 + 1.5e-5*amplitude
              registry(i,k,fnc) = 3.0e7
              registry(i,k,fnr) = 8.0e4
              registry(i,k,fni) = 1.5e5
              registry(i,k,fns) = 6.0e4
              rates(i,k,1) = (1.8e-5 + 0.2e-5*amplitude)/dt
              rates(i,k,2) = (1.4e-5 + 0.1e-5*amplitude)/dt
              rates(i,k,3) = (0.9e-5 + 0.1e-5*amplitude)/dt
              rates(i,k,4) = (1.1e-5 + 0.1e-5*amplitude)/dt

           case (5)
              ! Large particles over thin layers force adaptive CFL > 1.
              if (k >= 6) then
                 registry(i,k,fqr) = 8.0e-4*(1.0 + 0.04*real(k))
                 registry(i,k,fqi) = 2.0e-4*(1.0 + 0.02*real(k))
                 registry(i,k,fqs) = 6.0e-4*(1.0 + 0.03*real(k))
                 registry(i,k,fqg) = 9.0e-4*(1.0 + 0.03*real(k))
                 registry(i,k,fqh) = 1.1e-3*(1.0 + 0.02*real(k))
                 diameter = 2.2e-3 + 0.12e-3*real(k-6)
                 volume = 0.523599*diameter**3
                 registry(i,k,fnr) = registry(i,k,fqr)/(1000.0*volume)
                 registry(i,k,fni) = registry(i,k,fqi)/(900.0*volume)
                 registry(i,k,fns) = registry(i,k,fqs)/(100.0*volume)
                 registry(i,k,fng) = registry(i,k,fqg)/(480.0*volume)
                 registry(i,k,fnh) = registry(i,k,fqh)/(820.0*volume)
                 registry(i,k,fvg) = registry(i,k,fqg)/480.0
                 registry(i,k,fvh) = registry(i,k,fqh)/820.0
              endif

           case (6)
              ! Sparse mixed-phase conservation column with alternating gaps.
              if (mod(k,3) /= 0) then
                 registry(i,k,fqr) = 1.0e-5*1.45**real(k-1)
                 registry(i,k,fqs) = 0.7*registry(i,k,fqr)
                 registry(i,k,fqg) = 0.8*registry(i,k,fqr)
                 registry(i,k,fqh) = 0.6*registry(i,k,fqr)
                 registry(i,k,fqi) = 0.3*registry(i,k,fqr)
                 diameter = 0.25e-3*1.12**real(k-1)
                 volume = 0.523599*diameter**3
                 registry(i,k,fnr) = registry(i,k,fqr)/(1000.0*volume)
                 registry(i,k,fni) = registry(i,k,fqi)/(900.0*volume)
                 registry(i,k,fns) = registry(i,k,fqs)/(100.0*volume)
                 registry(i,k,fng) = registry(i,k,fqg)/(520.0*volume)
                 registry(i,k,fnh) = registry(i,k,fqh)/(850.0*volume)
                 registry(i,k,fvg) = registry(i,k,fqg)/520.0
                 registry(i,k,fvh) = registry(i,k,fqh)/850.0
              endif
           end select
        enddo
     enddo

     do i = 1, nx
        rainnc(i) = 0.20*real(irun) + 0.01*real(i)
        snownc(i) = 0.10*real(irun) + 0.01*real(i)
        graupelnc(i) = 0.05*real(irun) + 0.01*real(i)
        hailnc(i) = 0.03*real(irun) + 0.01*real(i)
        rainncv(i) = -9.0
        snowncv(i) = -9.0
        graupelncv(i) = -9.0
        hailncv(i) = -9.0
        sr(i) = -9.0
        ice_export(i) = 0.0
     enddo
     call write_rows(unit, 'before', irun, first_step, cu_used, dt)

     ! Exact default-driver gather and concentration conversion.
     do k = 1, nz
        do i = 1, nx
           an(i,1,k,lv) = registry(i,k,fqv)
           an(i,1,k,lc) = registry(i,k,fqc)
           an(i,1,k,lr) = registry(i,k,fqr)
           an(i,1,k,li) = registry(i,k,fqi)
           an(i,1,k,ls) = registry(i,k,fqs)
           an(i,1,k,lh) = registry(i,k,fqg)
           an(i,1,k,lhl) = registry(i,k,fqh)
           an(i,1,k,lnc) = registry(i,k,fnc)*density2(i,k)
           an(i,1,k,lnr) = registry(i,k,fnr)*density2(i,k)
           an(i,1,k,lni) = registry(i,k,fni)*density2(i,k)
           an(i,1,k,lns) = registry(i,k,fns)*density2(i,k)
           an(i,1,k,lnh) = registry(i,k,fng)*density2(i,k)
           an(i,1,k,lnhl) = registry(i,k,fnh)*density2(i,k)
           an(i,1,k,lccn) = registry(i,k,fnn)*density2(i,k)
           an(i,1,k,lvh) = registry(i,k,fvg)*density2(i,k)
           an(i,1,k,lvhl) = registry(i,k,fvh)*density2(i,k)
        enddo
     enddo

     if (first_step == 1) then
        call calcnfromq(nx,ny,nz,an,na,0,0,density2)
     endif

     if (cu_used == 1) then
        do k = 1, nz
           do i = 1, nx
              ancuten(i,1,k,lr) = dt*rates(i,k,1)
              ancuten(i,1,k,ls) = dt*rates(i,k,2)
              ancuten(i,1,k,li) = dt*rates(i,k,3)
              ancuten(i,1,k,lc) = dt*rates(i,k,4)
           enddo
        enddo
        call calcnfromcuten(nx,ny,nz,ancuten,an,na,0,0,density2)
     endif

     timesed1 = 0.0d0
     timesed2 = 0.0d0
     timesed3 = 0.0d0
     zmaxsed = 0.0d0
     timesetvt = 0.0d0
     call sediment1d(dt,nx,ny,nz,an,na,0,0,xfall,density3, &
          layer_depth,inverse_depth,temperature,ice_nucleation,2,1, &
          1,1,timesed1,timesed2,timesed3,zmaxsed,timesetvt)

     ! Exact default four-category reducer. Cloud-ice xfall is diagnostic only.
     rainncv = 0.0
     snowncv = 0.0
     graupelncv = 0.0
     hailncv = 0.0
     do i = 1, nx
        rainncv(i) = dt*density2(i,1) * &
             (xfall(i,1,lr) + xfall(i,1,ls) + &
              xfall(i,1,lh) + xfall(i,1,lhl))
        snowncv(i) = dt*density2(i,1)*xfall(i,1,ls)
        graupelncv(i) = dt*density2(i,1)*xfall(i,1,lh)
        hailncv(i) = dt*density2(i,1)*xfall(i,1,lhl)
        ice_export(i) = dt*density2(i,1)*xfall(i,1,li)
        rainnc(i) = rainnc(i) + rainncv(i)
        snownc(i) = snownc(i) + snowncv(i)
        graupelnc(i) = graupelnc(i) + graupelncv(i)
        hailnc(i) = hailnc(i) + hailncv(i)
        sr(i) = (snowncv(i) + graupelncv(i) + hailncv(i)) / &
             (rainncv(i) + 1.0e-12)
     enddo

     ! Exact one-time final denscale/scatter.
     do k = 1, nz
        do i = 1, nx
           registry(i,k,fqv) = an(i,1,k,lv)
           registry(i,k,fqc) = an(i,1,k,lc)
           registry(i,k,fqr) = an(i,1,k,lr)
           registry(i,k,fqi) = an(i,1,k,li)
           registry(i,k,fqs) = an(i,1,k,ls)
           registry(i,k,fqg) = an(i,1,k,lh)
           registry(i,k,fqh) = an(i,1,k,lhl)
           registry(i,k,fnc) = an(i,1,k,lnc)/density2(i,k)
           registry(i,k,fnr) = an(i,1,k,lnr)/density2(i,k)
           registry(i,k,fni) = an(i,1,k,lni)/density2(i,k)
           registry(i,k,fns) = an(i,1,k,lns)/density2(i,k)
           registry(i,k,fng) = an(i,1,k,lnh)/density2(i,k)
           registry(i,k,fnh) = an(i,1,k,lnhl)/density2(i,k)
           registry(i,k,fnn) = an(i,1,k,lccn)/density2(i,k)
           registry(i,k,fvg) = an(i,1,k,lvh)/density2(i,k)
           registry(i,k,fvh) = an(i,1,k,lvhl)/density2(i,k)
        enddo
     enddo
     call write_rows(unit, 'after', irun, first_step, cu_used, dt)
  enddo

  close(unit)
  print '(A,1X,A)', 'NSSL2_DRIVER_SUPPORT_ORACLE_COMPLETE', trim(output_path)

contains

  subroutine write_rows(output_unit, phase, run_id, first_flag, cu_flag, step)
    integer, intent(in) :: output_unit, run_id, first_flag, cu_flag
    character(len=*), intent(in) :: phase
    real, intent(in) :: step
    integer :: ii, kk

    do kk = 1, nz
       do ii = 1, nx
          write(output_unit,'(A,6(",",I0),34(",",ES24.16E3))') &
               trim(phase), run_id, ii, ii, kk, first_flag, cu_flag, &
               step, density2(ii,kk), layer_depth(ii,1,kk), &
               temperature(ii,1,kk), &
               rates(ii,kk,1), rates(ii,kk,2), rates(ii,kk,3), &
               rates(ii,kk,4), registry(ii,kk,1:nfield), &
               rainnc(ii), rainncv(ii), snownc(ii), snowncv(ii), &
               graupelnc(ii), graupelncv(ii), hailnc(ii), hailncv(ii), &
               sr(ii), ice_export(ii)
       enddo
    enddo
  end subroutine write_rows
end program nssl2_driver_support_oracle
