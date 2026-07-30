program nssl2_rain_cloud_accretion_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1
  integer, parameter :: lnc = 10, lnr = 11
  integer :: i, k, unit, nml_unit, case_id
  real :: nssl_params(20), dt, cloud_radius, rain_diameter
  real :: cloud_mass, rain_volume, rho, before_qc, before_qr
  real :: before_nc, before_nr
  real :: with_qc, with_qr, with_nc, with_nr
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_rain_cloud_accretion_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]

  ! Disable the neighboring warm-rain processes while preserving the exact
  ! default accretion equations.  This avoids subtracting two nearly equal
  ! full-process calls in the generated oracle values.
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  dmrauto = -2,'
  write(nml_unit,'(A)') '  icracr = 0,'
  write(nml_unit,'(A)') '  evapfac = 0.0,'
  write(nml_unit,'(A)') '/'
  close(nml_unit)

  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  xdnmx = 900.0
  xdnmx(lc) = 1000.0
  xdnmx(lr) = 1000.0
  xdnmx(li) = 917.0
  xdnmx(ls) = 300.0
  xdnmn = 900.0
  xdnmn(lc) = 1000.0
  xdnmn(lr) = 1000.0
  xdnmn(li) = 100.0
  xdnmn(ls) = 100.0
  xdnmn(lh) = 170.0
  xdnmn(lhl) = 500.0
  xdn0 = 900.0
  xdn0(lc) = 1000.0
  xdn0(lr) = 1000.0
  xdn0(li) = 900.0
  xdn0(ls) = 100.0
  xdn0(lh) = 500.0
  xdn0(lhl) = 900.0
  cdx = 0.6
  cdx(ls) = 2.0
  cdx(lh) = 0.8
  cdx(lhl) = 0.45
  ido = 1

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,dt_s,rho_kg_m3,cloud_radius_m,rain_volume_diameter_m,qc_before,nc_before_per_kg,qr_before,nr_before_per_kg,qc_after_accretion,qr_after_accretion,nc_after_accretion_per_kg,nr_after_accretion_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        select case (case_id)
        case (0)
           cloud_radius = 5.0e-6; rain_diameter = 0.50e-3; dt = 0.1
        case (1)
           cloud_radius = 5.0e-6; rain_diameter = 0.83e-3; dt = 1.0
        case (2)
           cloud_radius = 5.0e-6; rain_diameter = 0.90e-3; dt = 5.0
        case (3)
           cloud_radius = 5.0e-6; rain_diameter = 2.10e-3; dt = 10.0
        case (4)
           cloud_radius = 6.0e-6; rain_diameter = 2.60e-3; dt = 30.0
        case (5)
           cloud_radius = 7.5e-6; rain_diameter = 2.72e-3; dt = 60.0
        case (6)
           cloud_radius = 7.5e-6; rain_diameter = 0.35e-3; dt = 10.0
        case (7)
           cloud_radius = 10.0e-6; rain_diameter = 0.21e-3; dt = 5.0
        case (8)
           cloud_radius = 20.0e-6; rain_diameter = 85.0e-6; dt = 0.5
        case (9)
           cloud_radius = 25.0e-6; rain_diameter = 99.0e-6; dt = 1.0
        case (10)
           cloud_radius = 30.0e-6; rain_diameter = 101.0e-6; dt = 2.0
        case default
           cloud_radius = 50.0e-6; rain_diameter = 0.30e-3; dt = 10.0
        end select

        rho = 1.30 - 0.075*real(k-1) + 0.01*real(i-1)
        before_qc = 2.5e-5*1.65**real(mod(case_id,6))
        before_qr = 2.0e-5*1.55**real(mod(case_id+2,6))
        cloud_mass = 1000.0*(4.0/3.0)*acos(-1.0)*cloud_radius**3
        rain_volume = 0.523599*rain_diameter**3
        before_nc = before_qc/cloud_mass
        before_nr = before_qr/(1000.0*rain_volume)

        call run_variant(before_qr, before_nr, with_qc, with_qr, with_nc, with_nr)

        write(unit,'(3(I0,","),11(ES24.16E3,","),ES24.16E3)') &
             case_id, i, k, dt, rho, cloud_radius, rain_diameter, &
             before_qc, before_nc, before_qr, before_nr,          &
             with_qc, with_qr, with_nc, with_nr
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_RAIN_CLOUD_ACCRETION_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine run_variant(input_qr, input_nr, output_qc, output_qr, output_nc, &
                         output_nr)
    real, intent(in) :: input_qr, input_nr
    real, intent(out) :: output_qc, output_qr, output_nc, output_nr
    real :: a1(1,1,3,na)
    real :: u0(1,1,3), u1(1,1,3), u2(1,1,3), u3(1,1,3)
    real :: u4(1,1,3), u5(1,1,3), u6(1,1,3), u7(1,1,3)
    real :: u8(1,1,3), u9(1,1,3), uu0(1,1,3), uu7(1,1,3)
    real :: zg(1,1,3), dd(1,1,3), pp2(1,1,3), ppn(1,1,3)
    real :: ww(1,1,3), tt3(1,1,3), tee(1,3), aa(1,1,3,nxtra)
    real :: rp(1,3), ep(1,3), alp(1,3,3), el(1,1,3), th(3,1)

    a1 = 0.0
    a1(:,:,:,lt) = 300.0
    a1(:,:,:,2) = 0.1
    a1(:,:,:,lc) = before_qc
    a1(:,:,:,lr) = input_qr
    a1(:,:,:,lnc) = before_nc*rho
    a1(:,:,:,lnr) = input_nr*rho
    u0 = 300.0
    u1 = 0.0
    u2 = 0.0
    u3 = 0.0
    u4 = 0.0
    u5 = 0.0
    u6 = 0.0
    u7 = 0.0
    u8 = 0.0
    u9 = 0.0
    uu0 = 380.0/100000.0
    uu7 = 1.0
    zg = 1000.0
    dd = rho
    pp2 = 1.0
    ppn = 100000.0
    ww = 0.0
    tt3 = 0.0
    tee = 0.0
    aa = 0.0
    rp = 0.0
    ep = 0.0
    alp = 0.0
    el = 0.0
    th = 0.0
    timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    output_qc = a1(1,1,2,lc)
    output_qr = a1(1,1,2,lr)
    output_nc = a1(1,1,2,lnc)/rho
    output_nr = a1(1,1,2,lnr)/rho
  end subroutine run_variant
end program nssl2_rain_cloud_accretion_oracle
