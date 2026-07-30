program nssl2_fused_gs_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs, &
       ido, xdnmx, xdnmn, cdx, xdn0, ventr, ventc, c1sw, nxtra, &
       tabqvs, tabqis, nqsat, fqsat
  implicit none

  integer, parameter :: nx = 1, ny = 1, nz = 4, na = 40
  integer, parameter :: numproc = 1, ncases = 30, nstate = 16
  integer, parameter :: lt = 1, lv = 2, lc = 3, lr = 4
  integer, parameter :: li = 5, ls = 6, lg = 7, lh = 8
  integer, parameter :: lccn = 9, lnc = 10, lnr = 11
  integer, parameter :: lni = 12, lns = 13, lng = 14, lnh = 15
  integer, parameter :: lvg = 16, lvh = 17
  real, parameter :: rd = 287.04, cp = 1004.0, rho00 = 1.225
  real, parameter :: base_rho_g(nz) = [170.0,300.0,500.0,850.0]
  real, parameter :: base_rho_h(nz) = [500.0,650.0,800.0,900.0]
  real, parameter :: threshold_mass(nz) = [0.0,5.0e-13,5.0e-12,1.0e-8]
  real, parameter :: velocity_base(nz) = [-4.0,0.0,8.0,18.0]
  real, parameter :: velocity_wet(nz) = [4.0,10.0,18.0,24.0]
  real, parameter :: dz_base(nz) = [125.0,250.0,500.0,1000.0]
  character(len=*), parameter :: schema = 'gpuwm.nssl2.fused-gs.v1'

  integer :: case_index, repetition, k, output_unit, diag_unit
  real :: nssl_params(20), dt_s
  real :: an(nx,ny,nz,na), density(nx,ny,nz), pressure(nx,ny,nz)
  real :: exner(nx,ny,nz), velocity(nx,ny,nz), dz(nx,ny,nz)
  real :: t0(nx,ny,nz), t1(nx,ny,nz), t2(nx,ny,nz)
  real :: t3(nx,ny,nz), t4(nx,ny,nz), t5(nx,ny,nz)
  real :: t6(nx,ny,nz), t7(nx,ny,nz), t8(nx,ny,nz)
  real :: t9(nx,ny,nz), t00(nx,ny,nz), t77(nx,ny,nz)
  real :: tmp3d(nx,ny,nz), tkediss(nx,ny,nz)
  real, allocatable :: axtra(:,:,:,:)
  real :: elec(nx,ny,nz)
  real :: rainprod(nx,nz), evapprod(nx,nz), alpha2d(nx,nz,3)
  real :: thproc(nz,numproc), before(nz,nstate), after(nz,nstate)
  real :: temperature_input(nz), theta_input(nz)
  real(kind=8) :: timevtcalc
  character(len=64) :: case_name, case_family
  character(len=512) :: output_path, trace_path
  logical :: emit_trace

  call get_command_argument(1, output_path)
  call get_command_argument(2, trace_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_fused_gs_oracle OUTPUT.csv [TRACE.csv]'
  endif
  emit_trace = len_trim(trace_path) > 0

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)
  allocate(axtra(nx,ny,nz,nxtra))

  open(newunit=output_unit, file=trim(output_path), status='replace', &
       action='write', form='formatted')
  call write_header(output_unit)
  diag_unit = 0
  if (emit_trace) then
     open(newunit=diag_unit, file=trim(trace_path), status='replace', &
          action='write', form='formatted')
     write(diag_unit,'(A)') 'record,case,case_family,repetition,k,values'
  endif

  do case_index = 1, ncases
     do repetition = 0, 1
        call setup_case(case_index, case_name, case_family, dt_s)
        call compute_primary_ice_target()
        call snapshot(before)

        if (emit_trace) then
           write(diag_unit,'(A,",",A,",",A,",",I0)') &
                'CASE', trim(case_name), trim(case_family), repetition
        endif

        call nssl_2mom_gs(nx,ny,nz,na,1,0,0,dt_s,dz,            &
             t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,an,density,t77,      &
             pressure,velocity,diag_unit,t00,t77,ventr,ventc,   &
             c1sw,1,ido,xdnmx(3:8),xdnmn(3:8),cdx(3:8),         &
             xdn0(3:8),tmp3d,tkediss,thproc,numproc,1000.0,      &
             1000.0,64,timevtcalc,axtra,.false.,.false.,         &
             rainprod,evapprod,alpha2d,elec,1,1,nx,1,ny)

        call snapshot(after)
        do k = 1, nz
           call write_row(output_unit, case_name, case_family, &
                repetition, k, dt_s, before(k,:), after(k,:))
        enddo
     enddo
  enddo

  if (emit_trace) close(diag_unit)
  close(output_unit)
  print '(A,1X,A)', 'NSSL2_FUSED_GS_ORACLE_COMPLETE', trim(output_path)

contains

  subroutine setup_case(index, name, family, dt)
    integer, intent(in) :: index
    character(len=*), intent(out) :: name, family
    real, intent(out) :: dt
    integer :: kk
    real :: temp(nz), qvs, qis, scale, rho_g, rho_h, mass_floor
    real :: qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, nh, qnn

    call case_metadata(index, name, family, dt, temp)
    an = 0.0
    density = 0.0
    pressure = 0.0
    exner = 0.0
    velocity = 0.0
    dz = 0.0
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    t4 = 0.0
    t5 = 0.0
    t6 = 0.0
    t7 = 0.0
    t8 = 0.0
    t9 = 0.0
    t00 = 0.0
    t77 = 0.0
    tmp3d = 0.0
    tkediss = 0.0
    axtra = 0.0
    elec = 0.0
    rainprod = 0.0
    evapprod = 0.0
    alpha2d = 0.0
    thproc = 0.0
    timevtcalc = 0.0d0

    do kk = 1, nz
       pressure(1,1,kk) = 97500.0 - 10500.0*real(kk-1) &
            - 37.0*real(mod(index,5))
       exner(1,1,kk) = (pressure(1,1,kk)/100000.0)**(rd/cp)
       t0(1,1,kk) = temp(kk)
       t00(1,1,kk) = 380.0/pressure(1,1,kk)
       t77(1,1,kk) = exner(1,1,kk)
       call saturation(temp(kk), pressure(1,1,kk), qvs, qis)

       scale = 1.0 + 0.15*real(kk-1)
       qc = 2.0e-4*scale
       qr = 3.0e-4*scale
       qi = 1.5e-4*scale
       qs = 2.5e-4*scale
       qg = 3.5e-4*scale
       qh = 4.5e-4*scale
       nc = 8.0e7/scale
       nr = 3.0e4/scale
       ni = 2.0e5/scale
       ns = 5.0e3/scale
       ng = 2.0e3/scale
       nh = 5.0e2/scale
       qnn = 3.5e8
       rho_g = base_rho_g(kk)
       rho_h = base_rho_h(kk)
       an(1,1,kk,lv) = merge(1.04*qis, 0.98*qvs, temp(kk) < 273.15)

       select case (index)
       case (1)
          qc = 0.0; qr = 0.0; qi = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; nr = 0.0; ni = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          an(1,1,kk,lv) = 0.80*merge(qis, qvs, temp(kk) < 273.15)
       case (2)
          mass_floor = threshold_mass(kk)
          qc = mass_floor; qr = mass_floor; qi = mass_floor
          qs = mass_floor; qg = mass_floor; qh = mass_floor
          nc = 0.0; nr = 0.0; ni = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          an(1,1,kk,lv) = 0.95*merge(qis, qvs, temp(kk) < 273.15)
       case (7)
          qc = 8.0e-3; qr = 1.5e-3; qi = 1.2e-3
          qs = 1.5e-3; qg = 1.8e-3; qh = 2.0e-3
          nc = 2.0e8; nr = 8.0e4; ni = 8.0e5
          ns = 2.0e4; ng = 8.0e3; nh = 2.0e3
          an(1,1,kk,lv) = 1.02*merge(qis, qvs, temp(kk) < 273.15)
       case (8)
          qc = 1.5e-3; qr = 8.0e-3; qi = 1.8e-3
          qs = 2.0e-3; qg = 2.2e-3; qh = 2.4e-3
          nr = 2.0e4; ni = 1.0e6; ns = 2.0e4; ng = 8.0e3; nh = 2.0e3
          an(1,1,kk,lv) = 0.98*qis
       case (9)
          qc = 1.2e-3; qr = 1.5e-3; qi = 5.0e-3
          qs = 2.5e-3; qg = 2.5e-3; qh = 2.5e-3
          ni = 3.0e6; ns = 2.5e4; ng = 1.0e4; nh = 2.5e3
          select case (kk)
          case (1); an(1,1,kk,lv) = 0.88*qis
          case (2); an(1,1,kk,lv) = 1.02*qis
          case (3); an(1,1,kk,lv) = 1.08*qis
          case (4); an(1,1,kk,lv) = 0.96*qis
          end select
       case (10)
          qc = 1.2e-3; qr = 1.8e-3; qi = 1.8e-3
          qs = 6.0e-3; qg = 2.8e-3; qh = 2.8e-3
          ns = 3.0e4; ng = 1.2e4; nh = 3.0e3
          select case (kk)
          case (1); an(1,1,kk,lv) = 0.90*merge(qis,qvs,temp(kk)<273.15)
          case (2); an(1,1,kk,lv) = 0.99*merge(qis,qvs,temp(kk)<273.15)
          case (3); an(1,1,kk,lv) = 1.04*merge(qis,qvs,temp(kk)<273.15)
          case (4); an(1,1,kk,lv) = 0.96*merge(qis,qvs,temp(kk)<273.15)
          end select
       case (11)
          qc = 0.0; qr = 2.0e-4; qi = 1.5e-3
          qs = 2.0e-3; qg = 2.5e-3; qh = 3.0e-3
          nc = 0.0
          select case (kk)
          case (1); an(1,1,kk,lv) = 0.70*qis
          case (2); an(1,1,kk,lv) = 0.97*qis
          case (3); an(1,1,kk,lv) = 1.05*qis
          case (4); an(1,1,kk,lv) = 1.18*qis
          end select
       case (12)
          qc = 0.0; qr = 5.0e-4; qi = 3.0e-3
          qs = 4.0e-3; qg = 5.0e-3; qh = 5.0e-3
          nc = 0.0
          select case (kk)
          case (1); an(1,1,kk,lv) = 0.55*qis
          case (2); an(1,1,kk,lv) = 0.80*qis
          case (3); an(1,1,kk,lv) = 1.25*qis
          case (4); an(1,1,kk,lv) = 1.60*qis
          end select
       case (13)
          qc = 5.0e-3; qr = 3.0e-3; qi = 8.0e-4
          qs = 2.0e-3; qg = 4.0e-3; qh = 4.0e-3
          nc = 1.5e8; nr = 5.0e4; ng = 6.0e3; nh = 1.5e3
          an(1,1,kk,lv) = 1.01*merge(qis, qvs, temp(kk) < 273.15)
       case (14)
          qc = 4.0e-3; qr = 4.0e-3; qi = 5.0e-4
          qs = 1.5e-3; qg = 6.0e-3; qh = 3.0e-3
          nc = 1.2e8; nr = 4.0e4; ng = 2.5e3; nh = 1.0e3
          an(1,1,kk,lv) = 0.97*qvs
       case (15)
          qc = 2.0e-4; qr = 3.0e-4; qi = 2.0e-4
          qs = 3.0e-4; qg = 4.0e-4; qh = 5.0e-4
          select case (kk)
          case (1)
             nc=0.0; nr=0.0; ni=0.0; ns=0.0; ng=0.0; nh=0.0
          case (2)
             nc=1.0e-9; nr=1.0e-9; ni=1.0e-9
             ns=1.0e-9; ng=1.0e-9; nh=1.0e-9
          case (3)
             nc=1.0e12; nr=1.0e10; ni=1.0e11
             ns=1.0e10; ng=1.0e9; nh=1.0e8
          case (4)
             nc=8.0e7; nr=3.0e4; ni=2.0e5
             ns=5.0e3; ng=2.0e3; nh=5.0e2
          end select
       case (16)
          qc = 1.0e-3; qr = 1.2e-3; qi = 1.1e-3
          qs = 1.4e-3; qg = 1.6e-3; qh = 1.8e-3
          select case (kk)
          case (1)
             nc=1.0e-6; nr=1.0e-6; ni=1.0e-6
             ns=1.0e-6; ng=1.0e-6; nh=1.0e-6
          case (2)
             nc=1.0e4; nr=1.0e2; ni=1.0e3
             ns=1.0e2; ng=1.0e2; nh=1.0e1
          case (3)
             nc=1.0e11; nr=1.0e9; ni=1.0e10
             ns=1.0e9; ng=1.0e8; nh=1.0e7
          case (4)
             nc=1.0e8; nr=3.0e4; ni=2.0e5
             ns=5.0e3; ng=2.0e3; nh=5.0e2
          end select
       case (23:25)
          ! Rain-only Bigg/heat-budget slabs.  These retain option-18's
          ! default ibiggsnow=3 routing and exact native moment bounds.
          qc = 0.0; qi = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; ni = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          qr = 2.0e-3; nr = 1.0e4
          an(1,1,kk,lv) = qvs
       case (26)
          ! qxmin(r)=1e-12 is strict.  Level 3 makes the native ten-percent
          ! donor caps land at dt*q=1e-12 and dt*c=1e-8.
          qc = 0.0; qi = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; ni = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          select case (kk)
          case (1); qr=1.0e-12;    nr=1.0e-8
          case (2); qr=1.0001e-12; nr=1.0001e-8
          case (3); qr=1.0e-11;    nr=1.0e-7
          case (4); qr=1.0e-8;     nr=1.0e-4
          end select
          an(1,1,kk,lv) = qvs
       case (27)
          ! Deliberately malformed q/N pairs exercise mass-only, number-only,
          ! number-rich, and number-poor donor cleanup before freezing.
          qc = 0.0; qi = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; ni = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          select case (kk)
          case (1); qr=1.0e-3; nr=0.0
          case (2); qr=0.0;    nr=1.0e4
          case (3); qr=1.0e-3; nr=1.0e10
          case (4); qr=1.0e-3; nr=1.0e-6
          end select
          an(1,1,kk,lv) = qvs
       case (28)
          ! Span negative/near-zero/positive fwet1 with both Bigg and qiacr
          ! present, exposing the pre-cold-override heat cap and final factor.
          qc = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          qr = 2.0e-3; nr = 2.0e4; qi = 2.0e-3; ni = 4.0e5
          select case (kk)
          case (1); an(1,1,kk,lv) = 2.00*qvs
          case (2); an(1,1,kk,lv) = 1.20*qvs
          case (3); an(1,1,kk,lv) = 1.00*qvs
          case (4); an(1,1,kk,lv) = 0.70*qvs
          end select
       case (29)
          ! Simultaneous Bigg and rain/cloud-ice collection share the rain
          ! heat and donor limits; hail starts empty and must remain empty.
          qc = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          qr = 3.0e-3; nr = 3.0e4
          select case (kk)
          case (1); qi=0.0;    ni=0.0
          case (2); qi=1.0e-6; ni=2.0e2
          case (3); qi=1.0e-3; ni=2.0e5
          case (4); qi=5.0e-3; ni=1.0e6
          end select
          an(1,1,kk,lv) = qvs
       case (30)
          ! Four orders of rain and ice donor pressure at long dt force the
          ! shared heat cap, qr/dt cap, and post-process moment clamps.
          qc = 0.0; qs = 0.0; qg = 0.0; qh = 0.0
          nc = 0.0; ns = 0.0; ng = 0.0; nh = 0.0
          select case (kk)
          case (1); qr=1.0e-11; nr=1.0e-7; qi=1.0e-8; ni=1.0e1
          case (2); qr=1.0e-8;  nr=1.0e-4; qi=1.0e-6; ni=1.0e3
          case (3); qr=1.0e-5;  nr=1.0e1;  qi=1.0e-3; ni=2.0e5
          case (4); qr=1.0e-2;  nr=1.0e5;  qi=1.0e-2; ni=2.0e6
          end select
          an(1,1,kk,lv) = qvs
       case default
          ! The all-active and exact gate-scan cases retain the coupled base state.
       end select

       density(1,1,kk) = pressure(1,1,kk)/(rd*temp(kk)* &
            (1.0 + 0.608*an(1,1,kk,lv)))
       an(1,1,kk,lt) = temp(kk)/exner(1,1,kk)
       an(1,1,kk,lc) = qc
       an(1,1,kk,lr) = qr
       an(1,1,kk,li) = qi
       an(1,1,kk,ls) = qs
       an(1,1,kk,lg) = qg
       an(1,1,kk,lh) = qh
       an(1,1,kk,lnc) = nc
       an(1,1,kk,lnr) = nr
       an(1,1,kk,lni) = ni
       an(1,1,kk,lns) = ns
       an(1,1,kk,lng) = ng
       an(1,1,kk,lnh) = nh
       an(1,1,kk,lccn) = qnn
       if (qg > 0.0) an(1,1,kk,lvg) = density(1,1,kk)*qg/rho_g
       if (qh > 0.0) an(1,1,kk,lvh) = density(1,1,kk)*qh/rho_h

       if (index == 2 .or. (index == 15 .and. kk <= 2)) then
          an(1,1,kk,lvg) = 0.0
          an(1,1,kk,lvh) = 0.0
       elseif (index == 15 .and. kk == 3) then
          an(1,1,kk,lvg) = density(1,1,kk)*qg/50.0
          an(1,1,kk,lvh) = density(1,1,kk)*qh/1500.0
       elseif (index == 16) then
          select case (kk)
          case (1); rho_g=50.0; rho_h=250.0
          case (2); rho_g=170.0; rho_h=500.0
          case (3); rho_g=900.0; rho_h=900.0
          case (4); rho_g=1500.0; rho_h=1600.0
          end select
          an(1,1,kk,lvg) = density(1,1,kk)*qg/rho_g
          an(1,1,kk,lvh) = density(1,1,kk)*qh/rho_h
       endif

       velocity(1,1,kk) = velocity_base(kk) &
            + 0.25*real(mod(index,4))
       if (index == 13 .or. index == 14) then
          velocity(1,1,kk) = velocity_wet(kk)
       endif
       dz(1,1,kk) = dz_base(kk) &
            + 5.0*real(mod(index,3))
       temperature_input(kk) = temp(kk)
       theta_input(kk) = an(1,1,kk,lt)
    enddo
  end subroutine setup_case

  subroutine case_metadata(index, name, family, dt, temp)
    integer, intent(in) :: index
    character(len=*), intent(out) :: name, family
    real, intent(out) :: dt, temp(nz)
    real :: gate

    family = 'coupled_all_active'
    select case (index)
    case (1)
       name='zero_clear_dt0p1'; family='zero_threshold'; dt=0.1
       temp=[285.0,275.0,265.0,245.0]
    case (2)
       name='threshold_cleanup_dt1'; family='zero_threshold'; dt=1.0
       temp=[273.15,268.15,243.15,233.15]
    case (3)
       name='all_active_dt0p1'; dt=0.1; temp=[238.0,258.0,268.0,278.0]
    case (4)
       name='all_active_dt1'; dt=1.0; temp=[238.0,258.0,268.0,278.0]
    case (5)
       name='all_active_dt10'; dt=10.0; temp=[238.0,258.0,268.0,278.0]
    case (6)
       name='all_active_dt60'; dt=60.0; temp=[238.0,258.0,268.0,278.0]
    case (7)
       name='cloud_donor_compete_dt60'; family='cloud_donor_competition'
       dt=60.0; temp=[232.0,240.0,260.0,272.0]
    case (8)
       name='rain_donor_compete_dt60'; family='rain_donor_competition'
       dt=60.0; temp=[250.0,260.0,265.149,265.151]
    case (9)
       name='ice_donor_compete_dt60'; family='ice_donor_competition'
       dt=60.0; temp=[238.0,243.15,258.0,268.15]
    case (10)
       name='snow_donor_compete_dt60'; family='snow_donor_competition'
       dt=60.0; temp=[248.0,260.0,269.0,274.0]
    case (11)
       name='frozen_vapor_signed_dt60'; family='shared_frozen_vapor'
       dt=60.0; temp=[235.0,250.0,265.0,270.0]
    case (12)
       name='frozen_vapor_limiter_dt300'; family='shared_frozen_vapor'
       dt=300.0; temp=[235.0,243.15,258.0,268.15]
    case (13)
       name='wetgrowth_shedding_hm_dt10'; family='wetgrowth_shedding_hm'
       dt=10.0; temp=[268.0,270.149,270.15,270.151]
    case (14)
       name='wetgrowth_g2h_melt_dt60'; family='wetgrowth_shedding_hm'
       dt=60.0; temp=[271.0,273.149,273.15,273.151]
    case (15)
       name='moment_bounds_dt0p1'; family='moment_cleanup_bounds'
       dt=0.1; temp=[243.0,265.0,273.0,280.0]
    case (16)
       name='moment_bounds_dt60'; family='moment_cleanup_bounds'
       dt=60.0; temp=[243.0,265.0,273.0,280.0]
    case (17:22)
       family='exact_temperature_gate'
       dt=10.0
       select case (index)
       case (17); name='gate_243p15'; gate=243.15
       case (18); name='gate_265p15'; gate=265.15
       case (19); name='gate_268p15'; gate=268.15
       case (20); name='gate_270p15'; gate=270.15
       case (21); name='gate_271p15'; gate=271.15
       case (22); name='gate_273p15'; gate=273.15
       end select
       temp=[gate-0.001,gate,gate+0.001,gate+0.010]
    case (23)
       name='bigg_strict_temp_diameter'; family='rain_freezing_bigg_gates'
       dt=1.0; temp=[268.15,268.149,255.6345,255.6325]
    case (24)
       name='bigg_default_snow_split'; family='rain_freezing_bigg_gates'
       dt=1.0; temp=[245.7843,245.7833,245.7823,245.0]
    case (25)
       name='rain_heat_cold_override'; family='rain_freezing_heat_limiter'
       dt=10.0; temp=[243.151,243.15,243.149,240.0]
    case (26)
       name='rain_freezing_transfer_thresholds'; family='rain_freezing_thresholds'
       dt=1.0; temp=[240.0,240.0,240.0,240.0]
    case (27)
       name='rain_freezing_moment_donors'; family='rain_freezing_thresholds'
       dt=10.0; temp=[240.0,240.0,240.0,240.0]
    case (28)
       name='rain_heat_fwet_span'; family='rain_freezing_heat_limiter'
       dt=60.0; temp=[250.0,250.0,250.0,250.0]
    case (29)
       name='combined_bigg_qiacr'; family='rain_freezing_shared_donor'
       dt=10.0; temp=[255.0,250.0,245.0,240.0]
    case (30)
       name='rain_freezing_long_dt_caps'; family='rain_freezing_shared_donor'
       dt=60.0; temp=[240.0,240.0,240.0,240.0]
    case default
       error stop 'unknown fused-GS oracle case'
    end select
  end subroutine case_metadata

  subroutine saturation(temperature, press, qvs, qis)
    real, intent(in) :: temperature, press
    real, intent(out) :: qvs, qis
    integer :: ltemq

    ltemq = int((temperature-163.15)/fqsat + 1.5)
    ltemq = min(nqsat, max(1,ltemq))
    qvs = (380.0/press)*tabqvs(ltemq)
    qis = (380.0/press)*tabqis(ltemq)
  end subroutine saturation

  subroutine compute_primary_ice_target()
    integer :: kk, ltemq
    real :: qvs, qis, ssival, target

    t7 = 0.0
    do kk = 1, nz
       ltemq = int((t0(1,1,kk)-163.15)/fqsat + 1.5)
       ltemq = min(nqsat, max(1,ltemq))
       qvs = t00(1,1,kk)*tabqvs(ltemq)
       qis = t00(1,1,kk)*tabqis(ltemq)
       ssival = min(qvs,max(an(1,1,kk,lv),0.0))/qis
       if (ssival > 1.0 .and. t0(1,1,kk) <= 268.15) then
          target = density(1,1,kk)/rho00*1.0e3* &
               exp(min(57.0,12.96*(ssival-1.0)-0.639))
          t7(1,1,kk) = min(target,1.0e30)
       endif
    enddo
  end subroutine compute_primary_ice_target

  subroutine snapshot(values)
    real, intent(out) :: values(nz,nstate)
    integer :: kk

    do kk = 1, nz
       values(kk,:) = [an(1,1,kk,lv),an(1,1,kk,lc), &
            an(1,1,kk,lr),an(1,1,kk,li),an(1,1,kk,ls), &
            an(1,1,kk,lg),an(1,1,kk,lh),an(1,1,kk,lnc), &
            an(1,1,kk,lnr),an(1,1,kk,lni),an(1,1,kk,lns), &
            an(1,1,kk,lng),an(1,1,kk,lnh),an(1,1,kk,lccn), &
            an(1,1,kk,lvg),an(1,1,kk,lvh)]
    enddo
  end subroutine snapshot

  subroutine write_header(unit)
    integer, intent(in) :: unit
    write(unit,'(A)') &
      'schema_version,case,case_family,repetition,k,dt_s,dz_m,' // &
      'rho_kg_m3,pressure_pa,exner,w_lower_m_s,w_upper_m_s,' // &
      'w_center_m_s,primary_ice_target_m3,temperature_before_k,' // &
      'theta_before_k,qv_before,qc_before,qr_before,qi_before,' // &
      'qs_before,qg_before,qh_before,qndrop_before,qnr_before,' // &
      'qni_before,qns_before,qng_before,qnh_before,qnn_before,' // &
      'qvolg_before,qvolh_before,temperature_after_k,theta_after_k,' // &
      'qv_after,qc_after,qr_after,qi_after,qs_after,qg_after,qh_after,' // &
      'qndrop_after,qnr_after,qni_after,qns_after,qng_after,qnh_after,' // &
      'qnn_after,qvolg_after,qvolh_after'
  end subroutine write_header

  subroutine write_row(unit, name, family, rep, level, dt, before_row, after_row)
    integer, intent(in) :: unit, rep, level
    character(len=*), intent(in) :: name, family
    real, intent(in) :: dt, before_row(nstate), after_row(nstate)
    integer :: field
    real :: w_upper

    w_upper = velocity(1,1,min(level+1,nz))
    write(unit,'(A,",",A,",",A,",",I0,",",I0)',advance='no') &
         schema, trim(name), trim(family), rep, level
    call append_real(unit,dt)
    call append_real(unit,dz(1,1,level))
    call append_real(unit,density(1,1,level))
    call append_real(unit,pressure(1,1,level))
    call append_real(unit,exner(1,1,level))
    call append_real(unit,velocity(1,1,level))
    call append_real(unit,w_upper)
    call append_real(unit,0.5*(velocity(1,1,level)+w_upper))
    call append_real(unit,t7(1,1,level))
    call append_real(unit,temperature_input(level))
    call append_real(unit,theta_input(level))
    do field = 1, nstate
       call append_real(unit,before_row(field))
    enddo
    call append_real(unit,t0(1,1,level))
    call append_real(unit,an(1,1,level,lt))
    do field = 1, nstate
       call append_real(unit,after_row(field))
    enddo
    write(unit,'()')
  end subroutine write_row

  subroutine append_real(unit, value)
    integer, intent(in) :: unit
    real, intent(in) :: value
    write(unit,'(",",ES24.16E3)',advance='no') value
  end subroutine append_real

end program nssl2_fused_gs_oracle
