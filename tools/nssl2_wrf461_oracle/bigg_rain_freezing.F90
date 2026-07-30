program nssl2_bigg_rain_freezing_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnr = 11, lnh = 14, lnhl = 15
  integer, parameter :: lvh = 16, lvhl = 17
  integer :: i, k, unit, nml_unit, case_id, repetition, ltemq
  real :: nssl_params(20), dt, pressure, exner, rho, temperature
  real :: table_temperature, qvs, rain_diameter, rain_volume, scale
  real :: before_qv, before_theta, before_qr, before_nr
  real :: before_qg, before_ng, before_volg
  real :: graupel_density, graupel_diameter, graupel_volume
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_bigg_rain_freezing_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]

  ! Isolate default Bigg option-2 freezing.  Liquid saturation suppresses
  ! rain evaporation, absent neighbors suppress collection, and disabling
  ! the optional small-rain/splinter routes sends frozen rain to graupel.
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  dmrauto = -2,'
  write(nml_unit,'(A)') '  icracr = 0,'
  write(nml_unit,'(A)') '  ibiggopt = 2,'
  write(nml_unit,'(A)') '  ibiggsnow = 0,'
  write(nml_unit,'(A)') '  ibiggsmallrain = 0,'
  write(nml_unit,'(A)') '  nsplinter = 0,'
  write(nml_unit,'(A)') '  ifrzg = 1.0,'
  write(nml_unit,'(A)') '  depfac = 0.0,'
  write(nml_unit,'(A)') '  ehr0 = 0.0,'
  write(nml_unit,'(A)') '/'
  close(nml_unit)

  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  xdnmx = 900.0; xdnmx(lc) = 1000.0; xdnmx(lr) = 1000.0
  xdnmx(li) = 917.0; xdnmx(ls) = 300.0
  xdnmn = 900.0; xdnmn(lc) = 1000.0; xdnmn(lr) = 1000.0
  xdnmn(li) = 100.0; xdnmn(ls) = 100.0
  xdnmn(lh) = 170.0; xdnmn(lhl) = 500.0
  xdn0 = 900.0; xdn0(lc) = 1000.0; xdn0(lr) = 1000.0
  xdn0(li) = 900.0; xdn0(ls) = 100.0
  xdn0(lh) = 500.0; xdn0(lhl) = 900.0
  cdx = 0.6; cdx(ls) = 2.0; cdx(lh) = 0.8; cdx(lhl) = 0.45
  ido = 1

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,dt_s,rho_kg_m3,pressure_pa,exner,temperature_k,target_rain_diameter_m,theta_before_k,qv_before,qr_before,qnr_before_per_kg,qg_before,qng_before_per_kg,qvolg_before_m3_per_kg,theta_after_k,qv_after,qr_after,qnr_after_per_kg,qg_after,qng_after_per_kg,qvolg_after_m3_per_kg,qh_after,qnh_after_per_kg,qvolh_after_m3_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        repetition = ((k-1)*nx+i-1)/12
        pressure = 100000.0 - 9000.0*real(repetition) &
             - 350.0*real(case_id)
        scale = 10.0**(0.5*real(repetition)-0.5)

        select case (case_id)
        case (0)
           temperature = 260.0; rain_diameter = 1.0e-3
           before_qr = 0.0; dt = 1.0
        case (1)
           temperature = 260.0; rain_diameter = 1.0e-3
           before_qr = 1.0e-13; dt = 1.0
        case (2)
           temperature = 270.0; rain_diameter = 0.50e-3
           before_qr = 1.0e-4*scale; dt = 1.0
        case (3)
           temperature = 268.149; rain_diameter = 0.30e-3
           before_qr = 1.0e-4*scale; dt = 0.1
        case (4)
           temperature = 267.0; rain_diameter = 0.30e-3
           before_qr = 2.0e-4*scale; dt = 0.5
        case (5)
           temperature = 265.0; rain_diameter = 0.50e-3
           before_qr = 4.0e-4*scale; dt = 1.0
        case (6)
           temperature = 260.0; rain_diameter = 1.00e-3
           before_qr = 6.0e-4*scale; dt = 5.0
        case (7)
           temperature = 255.0; rain_diameter = 2.00e-3
           before_qr = 8.0e-4*scale; dt = 15.0
        case (8)
           temperature = 250.0; rain_diameter = 4.00e-3
           before_qr = 1.0e-3*scale; dt = 30.0
        case (9)
           temperature = 240.0; rain_diameter = 6.00e-3
           before_qr = 1.0e-3*scale; dt = 60.0
        case (10)
           temperature = 235.0; rain_diameter = 0.080e-3
           before_qr = 2.0e-4*scale; dt = 300.0
        case default
           temperature = 230.0; rain_diameter = 3.00e-3
           before_qr = 1.0e-3*scale; dt = 1000.0
        end select

        ! Exercise the exact temperature-table and pressure/rho rounding at
        ! four nearby real states without moving the -5 C branch case.
        if (case_id /= 3) temperature = temperature - 0.037*real(repetition)
        ltemq = int((temperature-163.15)/0.002 + 1.5)
        ltemq = min(1000001, max(1, ltemq))
        table_temperature = 163.15 + real(ltemq-1)*0.002
        qvs = (380.0/pressure)*exp(17.2693882* &
             (table_temperature-273.15)/(table_temperature-35.86))
        before_qv = qvs
        exner = (pressure/100000.0)**(287.04/1004.0)
        before_theta = temperature/exner
        rho = pressure/(287.04*temperature*(1.0 + 0.608*before_qv))
        rain_volume = 0.523599*rain_diameter**3
        if (before_qr > 0.0) then
           before_nr = before_qr/(1000.0*rain_volume)
        else
           before_nr = 0.0
        endif

        ! Repetition zero preserves the new-category oracle.  The remaining
        ! repetitions carry existing predicted-density graupel through an
        ! in-range, below-minimum, or above-final-maximum mean-volume state.
        ! Adjacent vapor exchange and graupel-rain collection are disabled by
        ! native namelist controls above, leaving Bigg transfer plus native
        ! density/number/volume reconstruction and final bounds.
        select case (repetition)
        case (0)
           before_qg = 0.0
           before_ng = 0.0
           before_volg = 0.0
        case (1)
           before_qg = 2.0e-4 + 1.0e-5*real(case_id)
           graupel_density = 170.0 + 365.0*real(mod(case_id,3))
           graupel_diameter = 1.0e-3
           graupel_volume = 0.523599*graupel_diameter**3
           before_ng = before_qg/(graupel_density*graupel_volume)
           before_volg = before_qg/graupel_density
        case (2)
           before_qg = 5.0e-4 + 2.0e-5*real(case_id)
           graupel_density = 300.0 + 100.0*real(mod(case_id,4))
           graupel_diameter = 0.15e-3
           graupel_volume = 0.523599*graupel_diameter**3
           before_ng = before_qg/(graupel_density*graupel_volume)
           before_volg = before_qg/graupel_density
        case default
           before_qg = 1.0e-3 + 4.0e-5*real(case_id)
           graupel_density = 650.0 + 100.0*real(mod(case_id,3))
           graupel_diameter = 15.0e-3
           graupel_volume = 0.523599*graupel_diameter**3
           before_ng = before_qg/(graupel_density*graupel_volume)
           before_volg = before_qg/graupel_density
        end select

        call run_case(unit)
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_BIGG_RAIN_FREEZING_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine run_case(output_unit)
    integer, intent(in) :: output_unit
    real :: a1(1,1,3,na)
    real :: u0(1,1,3), u1(1,1,3), u2(1,1,3), u3(1,1,3)
    real :: u4(1,1,3), u5(1,1,3), u6(1,1,3), u7(1,1,3)
    real :: u8(1,1,3), u9(1,1,3), uu0(1,1,3), uu7(1,1,3)
    real :: zg(1,1,3), dd(1,1,3), pp2(1,1,3), ppn(1,1,3)
    real :: ww(1,1,3), tt3(1,1,3), tee(1,3), aa(1,1,3,nxtra)
    real :: rp(1,3), ep(1,3), alp(1,3,3), el(1,1,3), th(3,1)

    a1 = 0.0
    a1(:,:,:,lt) = before_theta
    a1(:,:,:,lv) = before_qv
    a1(:,:,:,lr) = before_qr
    a1(:,:,:,lnr) = before_nr*rho
    a1(:,:,:,lh) = before_qg
    a1(:,:,:,lnh) = before_ng*rho
    a1(:,:,:,lvh) = before_volg*rho
    u0 = temperature
    u1 = 0.0; u2 = 0.0; u3 = 0.0; u4 = 0.0; u5 = 0.0
    u6 = 0.0; u7 = 0.0; u8 = 0.0; u9 = 0.0
    uu0 = 380.0/pressure; uu7 = exner; zg = 1000.0; dd = rho
    pp2 = exner; ppn = pressure; ww = 0.0; tt3 = 0.0; tee = 0.0
    aa = 0.0; rp = 0.0; ep = 0.0; alp = 0.0; el = 0.0; th = 0.0
    timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(3(I0,","),22(ES24.16E3,","),ES24.16E3)') &
         case_id, i, k, dt, rho, pressure, exner, temperature, &
         rain_diameter, before_theta, before_qv, before_qr, before_nr, &
         before_qg, before_ng, before_volg, &
         a1(1,1,2,lt), a1(1,1,2,lv), a1(1,1,2,lr), &
         a1(1,1,2,lnr)/rho, a1(1,1,2,lh), a1(1,1,2,lnh)/rho, &
         a1(1,1,2,lvh)/rho, a1(1,1,2,lhl), &
         a1(1,1,2,lnhl)/rho, a1(1,1,2,lvhl)/rho
  end subroutine run_case
end program nssl2_bigg_rain_freezing_oracle
