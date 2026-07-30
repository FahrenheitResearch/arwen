program run_morrison_oracle
  ! Drive the public WRF Morrison wrapper from the byte-unmodified v4.6.1
  ! module.  Every expected value written here is a value returned by
  ! MP_MORR_TWO_MOMENT; this program constructs inputs but never reimplements
  ! Morrison arithmetic.
  use ccpp_kind_types, only: kind_phys
  use module_model_constants, only: cp, ep_2, p1000mb, r_d, r_v
  use module_mp_morr_two_moment, only: morr_two_moment_init, &
                                        mp_morr_two_moment, polysvp
  implicit none

  integer, parameter :: nz = 32
  integer, parameter :: ncase = 14

  real, dimension(1, nz, 1) :: th, qv, qc, qr, qi, qs, qg
  real, dimension(1, nz, 1) :: ni, ns, nr, ng
  real, dimension(1, nz, 1) :: rho, pii, pressure, dz, w
  real, dimension(1, nz, 1) :: qrcuten, qscuten, qicuten
  real, dimension(1, nz, 1) :: refl
  real, dimension(1, nz, 1) :: th_in, qv_in, qc_in, qr_in, qi_in
  real, dimension(1, nz, 1) :: qs_in, qg_in, ni_in, ns_in, nr_in, ng_in
  real, dimension(1, nz, 1) :: rho_in, pii_in, pressure_in, dz_in
  real, dimension(1, nz, 1) :: qrcuten_in, qscuten_in, qicuten_in
  real, dimension(1, 1) :: ht
  real, dimension(1, 1) :: rainnc, rainncv, sr
  real, dimension(1, 1) :: snownc, snowncv, graupelnc, graupelncv
  real :: rainnc_in, rainncv_in, sr_in
  real :: snownc_in, snowncv_in, graupelnc_in, graupelncv_in
  real :: dt

  character(len=1024) :: level_path, surface_path
  integer :: icase, mode, k, ulev, usfc

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0) then
    write(*, '(A)') 'usage: run_morrison LEVELS.csv SURFACE.csv'
    error stop 2
  end if

  ! Morrison uses default REAL.  Pin the reference's precision explicitly:
  ! WRF's own ccpp kind declaration under -DRWORDSIZE=4 must be the same kind.
  if (kind_phys /= kind(1.0) .or. storage_size(1.0) /= 32) then
    write(*, '(A,I0,A,I0,A,I0)') 'kind_phys=', kind_phys, &
      ' kind(1.0)=', kind(1.0), ' storage_size=', storage_size(1.0)
    error stop 3
  end if

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') &
    'case,mode,k,nz,dt,theta_in,qv_in,qc_in,qr_in,qi_in,qs_in,qg_in,' // &
    'ni_in,ns_in,nr_in,ng_in,rho_in,pii_in,pressure_in,dz_in,' // &
    'qrcuten_in,qscuten_in,qicuten_in,theta_out,qv_out,qc_out,qr_out,' // &
    'qi_out,qs_out,qg_out,ni_out,ns_out,nr_out,ng_out,refl_10cm_out'

  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') &
    'case,mode,dt,rainnc_in,rainncv_in,snownc_in,snowncv_in,' // &
    'graupelnc_in,graupelncv_in,sr_in,rainnc_out,rainncv_out,' // &
    'snownc_out,snowncv_out,graupelnc_out,graupelncv_out,sr_out'

  do mode = 0, 1
    do icase = 1, ncase
      call morr_two_moment_init(mode)
      call build_case(icase, mode, dt)
      call snapshot_inputs

      call mp_morr_two_moment(                                             &
        itimestep=1, th=th, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,      &
        ni=ni, ns=ns, nr=nr, ng=ng, rho=rho, pii=pii, p=pressure,          &
        dt_in=dt, dz=dz, ht=ht, w=w, rainnc=rainnc, rainncv=rainncv,       &
        sr=sr, snownc=snownc, snowncv=snowncv, graupelnc=graupelnc,        &
        graupelncv=graupelncv, refl_10cm=refl, diagflag=.true.,            &
        do_radar_ref=1, qrcuten=qrcuten, qscuten=qscuten,                  &
        qicuten=qicuten, ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz,        &
        ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz,                        &
        its=1, ite=1, jts=1, jte=1, kts=1, kte=nz)

      do k = 1, nz
        write(ulev, '(4(I0,","))', advance='no') icase, mode, k, nz
        write(ulev, '(31(ES24.16E3,:,","))')                               &
          dt, th_in(1,k,1), qv_in(1,k,1), qc_in(1,k,1), qr_in(1,k,1),      &
          qi_in(1,k,1), qs_in(1,k,1), qg_in(1,k,1), ni_in(1,k,1),          &
          ns_in(1,k,1), nr_in(1,k,1), ng_in(1,k,1), rho_in(1,k,1),         &
          pii_in(1,k,1), pressure_in(1,k,1), dz_in(1,k,1),                 &
          qrcuten_in(1,k,1), qscuten_in(1,k,1), qicuten_in(1,k,1),         &
          th(1,k,1), qv(1,k,1), qc(1,k,1), qr(1,k,1), qi(1,k,1),           &
          qs(1,k,1), qg(1,k,1), ni(1,k,1), ns(1,k,1), nr(1,k,1),           &
          ng(1,k,1), refl(1,k,1)
      end do

      write(usfc, '(2(I0,","))', advance='no') icase, mode
      write(usfc, '(15(ES24.16E3,:,","))')                                 &
        dt, rainnc_in, rainncv_in, snownc_in, snowncv_in,                  &
        graupelnc_in, graupelncv_in, sr_in, rainnc(1,1), rainncv(1,1),     &
        snownc(1,1), snowncv(1,1), graupelnc(1,1), graupelncv(1,1),        &
        sr(1,1)
    end do
  end do

contains

  subroutine snapshot_inputs
    th_in = th
    qv_in = qv
    qc_in = qc
    qr_in = qr
    qi_in = qi
    qs_in = qs
    qg_in = qg
    ni_in = ni
    ns_in = ns
    nr_in = nr
    ng_in = ng
    rho_in = rho
    pii_in = pii
    pressure_in = pressure
    dz_in = dz
    qrcuten_in = qrcuten
    qscuten_in = qscuten
    qicuten_in = qicuten
    rainnc_in = rainnc(1,1)
    rainncv_in = rainncv(1,1)
    snownc_in = snownc(1,1)
    snowncv_in = snowncv(1,1)
    graupelnc_in = graupelnc(1,1)
    graupelncv_in = graupelncv(1,1)
    sr_in = sr(1,1)
  end subroutine snapshot_inputs

  subroutine build_case(which, rimed_mode, dt_out)
    integer, intent(in) :: which, rimed_mode
    real, intent(out) :: dt_out
    real :: z, dz_base, temperature, ew, qsat, shape_cloud, shape_precip
    real :: nzero, subn, minnorm, threshold
    integer :: kk, selector

    nzero = sign(0.0, -1.0)
    subn = transfer(1, 0.0)
    minnorm = transfer(8388608, 0.0)

    dt_out = 45.0
    dz_base = 400.0
    if (which == 9) then
      dt_out = 90.0
      dz_base = 150.0
    else if (which == 11 .or. which == 12) then
      dt_out = 10.0
    else if (which == 13) then
      dt_out = 30.0
    end if

    ht = 0.0
    w = 0.0
    qrcuten = 0.0
    qscuten = 0.0
    qicuten = 0.0
    refl = nzero
    qc = 0.0
    qr = 0.0
    qi = 0.0
    qs = 0.0
    qg = 0.0
    ni = 0.0
    ns = 0.0
    nr = 0.0
    ng = 0.0

    do kk = 1, nz
      z = (real(kk) - 0.5) * dz_base
      dz(1,kk,1) = dz_base
      pressure(1,kk,1) = 98000.0 * exp(-z / 8000.0)

      select case (which)
      case (1)
        temperature = max(276.0, 299.0 - 0.0030 * z)
      case (2, 10, 11, 12, 14)
        temperature = 290.0 - 0.0063 * z
      case (3, 7, 8)
        temperature = 263.0 - 0.0030 * z
      case (4, 5)
        temperature = 286.0 - 0.0055 * z
      case (6)
        temperature = 278.0 + 0.0005 * z
      case (9)
        temperature = 288.0 - 0.0065 * z
      case (13)
        temperature = 205.0 - 0.0012 * z
      end select

      pii(1,kk,1) = (pressure(1,kk,1) / p1000mb) ** (r_d / cp)
      th(1,kk,1) = temperature / pii(1,kk,1)
      rho(1,kk,1) = pressure(1,kk,1) / (r_d * temperature)
      ew = min(0.99 * pressure(1,kk,1), polysvp(temperature, 0))
      qsat = ep_2 * ew / (pressure(1,kk,1) - ew)
      qv(1,kk,1) = 0.85 * qsat
      shape_cloud = exp(-((z - 3500.0) / 2300.0) ** 2)
      shape_precip = exp(-((z - 2400.0) / 2800.0) ** 2)

      select case (which)
      case (1)
        qv(1,kk,1) = 1.01 * qsat
        qc(1,kk,1) = 1.5e-3 * shape_cloud
        qr(1,kk,1) = 6.0e-4 * shape_precip
        nr(1,kk,1) = 1.5e6 / rho(1,kk,1)
      case (2)
        qv(1,kk,1) = merge(1.08, 1.015, temperature < 273.15) * qsat
        qc(1,kk,1) = 1.4e-3 * shape_cloud
        qr(1,kk,1) = 5.0e-4 * shape_precip
        nr(1,kk,1) = 1.4e6 / rho(1,kk,1)
        if (temperature < 273.15) then
          qi(1,kk,1) = 2.5e-4 * shape_cloud
          qs(1,kk,1) = 7.0e-4 * shape_precip
          qg(1,kk,1) = 3.0e-4 * shape_precip
          ni(1,kk,1) = 8.0e4 / rho(1,kk,1)
          ns(1,kk,1) = 2.0e5 / rho(1,kk,1)
          ng(1,kk,1) = 8.0e4 / rho(1,kk,1)
        end if
      case (3)
        qv(1,kk,1) = 1.08 * qsat
        qi(1,kk,1) = 6.0e-4 * shape_cloud
        qs(1,kk,1) = 1.1e-3 * shape_precip
        qg(1,kk,1) = 4.0e-4 * shape_precip
        ni(1,kk,1) = 1.0e5 / rho(1,kk,1)
        ns(1,kk,1) = 2.5e5 / rho(1,kk,1)
        ng(1,kk,1) = 9.0e4 / rho(1,kk,1)
      case (4)
        qv(1,kk,1) = 0.25 * qsat
      case (5)
        qv(1,kk,1) = 0.45 * qsat
        qr(1,kk,1) = 8.0e-4 * shape_precip
        nr(1,kk,1) = 1.2e6 / rho(1,kk,1)
      case (6)
        qv(1,kk,1) = 0.98 * qsat
        qs(1,kk,1) = 8.0e-4 * shape_precip
        qg(1,kk,1) = 5.0e-4 * shape_precip
        ns(1,kk,1) = 2.0e5 / rho(1,kk,1)
        ng(1,kk,1) = 1.0e5 / rho(1,kk,1)
      case (7)
        qv(1,kk,1) = 1.10 * qsat
        qc(1,kk,1) = 1.8e-3 * shape_cloud
        qi(1,kk,1) = 2.0e-4 * shape_cloud
        qs(1,kk,1) = 1.0e-3 * shape_precip
        qg(1,kk,1) = 2.0e-4 * shape_precip
        ni(1,kk,1) = 7.0e4 / rho(1,kk,1)
        ns(1,kk,1) = 2.0e5 / rho(1,kk,1)
        ng(1,kk,1) = 7.0e4 / rho(1,kk,1)
      case (8)
        qv(1,kk,1) = 1.06 * qsat
        qi(1,kk,1) = 9.0e-4 * shape_cloud
        ni(1,kk,1) = qi(1,kk,1) * 1.0e7
        qs(1,kk,1) = 4.0e-4 * shape_precip
        ns(1,kk,1) = 1.5e5 / rho(1,kk,1)
      case (9)
        qv(1,kk,1) = 1.04 * qsat
        qc(1,kk,1) = 1.0e-3 * shape_cloud
        qr(1,kk,1) = 2.0e-3 * shape_precip
        qi(1,kk,1) = 4.0e-4 * shape_cloud
        qs(1,kk,1) = 2.0e-3 * shape_precip
        qg(1,kk,1) = 1.5e-3 * shape_precip
        nr(1,kk,1) = 2.0e6 / rho(1,kk,1)
        ni(1,kk,1) = 1.0e5 / rho(1,kk,1)
        ns(1,kk,1) = 3.0e5 / rho(1,kk,1)
        ng(1,kk,1) = 1.5e5 / rho(1,kk,1)
      case (10)
        qv(1,kk,1) = 1.03 * qsat
        qr(1,kk,1) = 3.0e-4 * shape_precip
        qi(1,kk,1) = 2.0e-4 * shape_cloud
        qs(1,kk,1) = 4.0e-4 * shape_precip
        nr(1,kk,1) = 1.0e6 / rho(1,kk,1)
        ni(1,kk,1) = 5.0e4 / rho(1,kk,1)
        ns(1,kk,1) = 1.0e5 / rho(1,kk,1)
        selector = mod(kk - 1, 7)
        select case (selector)
        case (0)
          qrcuten(1,kk,1) = 0.0
          qscuten(1,kk,1) = nzero
          qicuten(1,kk,1) = subn
        case (1)
          qrcuten(1,kk,1) = nearest(1.0e-10, -1.0)
          qscuten(1,kk,1) = 1.0e-10
          qicuten(1,kk,1) = nearest(1.0e-10, 1.0)
        case (2)
          qrcuten(1,kk,1) = 2.0e-7
          qscuten(1,kk,1) = 3.0e-8
          qicuten(1,kk,1) = 4.0e-9
        case default
          qrcuten(1,kk,1) = 2.0e-8 * real(selector)
          qscuten(1,kk,1) = 1.0e-8 * real(selector)
          qicuten(1,kk,1) = 0.5e-8 * real(selector)
        end select
      case (11)
        qv(1,kk,1) = 0.90 * qsat
        selector = mod(kk - 1, 8)
        select case (selector)
        case (0)
          threshold = 0.0
        case (1)
          threshold = nzero
        case (2)
          threshold = subn
        case (3)
          threshold = nearest(1.0e-14, -1.0)
        case (4)
          threshold = 1.0e-14
        case (5)
          threshold = nearest(1.0e-14, 1.0)
        case (6)
          threshold = 1.0e-8
        case default
          threshold = 1.0e-6
        end select
        qc(1,kk,1) = threshold
        qr(1,kk,1) = threshold
        qi(1,kk,1) = threshold
        qs(1,kk,1) = threshold
        qg(1,kk,1) = threshold
        ni(1,kk,1) = merge(5.0e4 / rho(1,kk,1), threshold, &
                            threshold >= 1.0e-14)
        ns(1,kk,1) = ni(1,kk,1)
        nr(1,kk,1) = ni(1,kk,1)
        ng(1,kk,1) = ni(1,kk,1)
      case (12)
        selector = mod(kk - 1, 6)
        select case (selector)
        case (0)
          qv(1,kk,1) = 0.0
          qc(1,kk,1) = nzero
          nr(1,kk,1) = -subn
        case (1)
          qv(1,kk,1) = nzero
          qr(1,kk,1) = subn
          ni(1,kk,1) = nzero
        case (2)
          qv(1,kk,1) = subn
          qi(1,kk,1) = nzero
          ns(1,kk,1) = subn
        case (3)
          qv(1,kk,1) = minnorm
          qs(1,kk,1) = subn
          ng(1,kk,1) = nzero
        case (4)
          qv(1,kk,1) = nearest(minnorm, -1.0)
          qg(1,kk,1) = -subn
        case default
          qv(1,kk,1) = nearest(minnorm, 1.0)
          qc(1,kk,1) = subn
          ng(1,kk,1) = subn
        end select
      case (13)
        qv(1,kk,1) = 0.80 * qsat
        if (mod(kk, 5) == 0) then
          qi(1,kk,1) = 2.0e-6 * shape_cloud
          ni(1,kk,1) = 3.0e4 / rho(1,kk,1)
        end if
      case (14)
        qv(1,kk,1) = 0.75 * qsat
        if (mod(kk, 2) == 0) then
          qc(1,kk,1) = 4.0e-7
          qr(1,kk,1) = 7.0e-7
          nr(1,kk,1) = 8.0e4 / rho(1,kk,1)
        else
          qi(1,kk,1) = 4.0e-7
          qs(1,kk,1) = 7.0e-7
          ni(1,kk,1) = 5.0e4 / rho(1,kk,1)
          ns(1,kk,1) = 8.0e4 / rho(1,kk,1)
        end if
      end select
    end do

    rainnc(1,1) = 0.125 * real(which) + 0.5 * real(rimed_mode)
    rainncv(1,1) = nzero
    snownc(1,1) = 0.25 * real(which) + 0.75 * real(rimed_mode)
    snowncv(1,1) = 0.0
    graupelnc(1,1) = 0.375 * real(which) + real(rimed_mode)
    graupelncv(1,1) = nzero
    sr(1,1) = nzero
  end subroutine build_case

end program run_morrison_oracle
