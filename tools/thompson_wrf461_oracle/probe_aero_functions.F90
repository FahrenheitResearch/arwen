! Pointwise probes of the scalar aerosol-aware helper functions inside
! unmodified WRF v4.6.1 module_mp_thompson.F, so ArWen's __device__
! transcriptions can be gated before any network kernel exists.
!
! Covers: iceDeMott, iceKoop, activ_ncloud, Eff_aero, and calc_effectRad.
!
! calc_effectRad is probed directly rather than only through mp_gt_driver
! because two of its three droplet-shape branches are not reachable from
! the driver:
!   * nc > 1.0e10 is DEAD CODE in v4.6.1 - the immediately preceding line
!     clamps nc to Nt_c_max = 1.999e9 (module_mp_thompson.F:5626), so the
!     branch at 5638 can never be taken from any caller.  The probe records
!     that fact as data rather than prose.
!   * nc < 100 is reachable only for cloud mixing ratios barely above R1,
!     because mp_thompson's terminal rediagnosis rebuilds nc from the
!     droplet size clamp.
!
! Note on iceDeMott: in v4.6.1 the Phillips (2008) branch is entirely
! commented out (5474-5505), so the function is a pure function of
! (tempc, rho, nifa).  qv, qvs and qvsi are still formal arguments and are
! passed here as zeros to make that independence explicit and testable.

program probe_aero_functions
  use module_mp_thompson, only: thompson_init, iceDeMott, iceKoop,      &
       activ_ncloud, Eff_aero, calc_effectRad, RSLF, RSIF
  implicit none

  integer, parameter :: nx = 2, ny = 2, nzi = 2
  real :: hgt(nx,nzi,ny)
  real :: nwfa_init(nx,nzi,ny), nifa_init(nx,nzi,ny)
  real :: nwfa2d_init(nx,ny)
  integer :: probe_unit, free_unit
  logical :: unit_opened
  character(len=512) :: output_dir

  call get_command_argument(1, output_dir)
  if (len_trim(output_dir) == 0) then
     error stop 'usage: probe_aero_functions OUTPUT_DIRECTORY'
  endif

  ! build_aero.sh scopes GFORTRAN_CONVERT_UNIT to unit 20 only, on the
  ! premise that table_ccnAct's 20..99 search (5123-5133) lands there in a
  ! fresh process.  Assert it rather than assume it.
  free_unit = -1
  do probe_unit = 20, 99
     inquire(unit=probe_unit, opened=unit_opened)
     if (.not. unit_opened) then
        free_unit = probe_unit
        exit
     endif
  enddo
  if (free_unit /= 20) then
     print '(A,I0)', 'FATAL: lowest free Fortran unit is not 20 but ', free_unit
     error stop 'CCN_ACTIVATE.BIN endian conversion assumption violated'
  endif

  hgt = 0.0
  hgt(:,1,:) = 100.0
  hgt(:,2,:) = 200.0
  nwfa_init = 3.0e8
  nifa_init = 1.0e6
  nwfa2d_init = 0.0

  ! Aerosol-aware init is required: activ_ncloud reads tnccn_act, which is
  ! only filled from CCN_ACTIVATE.BIN when is_aerosol_aware is true, and
  ! calc_effectRad branches on is_aerosol_aware at line 5627.
  call thompson_init(                                                &
       hgt=hgt,                                                      &
       nwfa2d=nwfa2d_init, nwfa=nwfa_init, nifa=nifa_init,           &
       wif_input_opt=0,                                              &
       ids=1, ide=3, jds=1, jde=3, kds=1, kde=2,                    &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nzi,                &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nzi)

  call probe_demott()
  call probe_koop()
  call probe_activ()
  call probe_eff_aero()
  call probe_effect_rad()

  print '(A)', 'THOMPSON_AERO_PROBE_COMPLETE'

contains

  subroutine probe_demott()
    integer :: unit, a, b, c
    real :: tempc, rho, nifa, value
    real, parameter :: tc(10) = (/-0.1, -2.0, -5.0, -10.0, -15.0,       &
                                  -20.0, -25.0, -30.0, -35.0, -40.0/)
    real, parameter :: rr(4) = (/0.35, 0.65, 1.0, 1.25/)
    real, parameter :: nn(8) = (/5.0e3, 5.0e4, 5.0e5, 5.0e6, 5.0e7,     &
                                 5.0e8, 5.0e9, 9.999e9/)

    open(newunit=unit, file=trim(output_dir)//'/probe-icedemott.csv',   &
         status='replace', action='write')
    write(unit,'(A)') 'tempc_c,rho_kg_m3,nifa_per_m3,xni_per_m3'
    do a = 1, size(tc)
       do b = 1, size(rr)
          do c = 1, size(nn)
             tempc = tc(a)
             rho = rr(b)
             nifa = nn(c)
             ! qv/qvs/qvsi are unused in v4.6.1; pass zeros deliberately.
             value = iceDeMott(tempc, 0.0, 0.0, 0.0, rho, nifa)
             write(unit,'(4(ES24.16E3,:,","))') tempc, rho, nifa, value
          enddo
       enddo
    enddo
    close(unit)
  end subroutine probe_demott

  subroutine probe_koop()
    integer :: unit, a, b, c
    real :: temp, qvs, qv, naero, dtv, value
    real, parameter :: tt(8) = (/200.0, 210.0, 220.0, 225.0, 230.0,     &
                                 233.0, 235.0, 237.9/)
    ! iceKoop is extremely steep in liquid saturation: below roughly 0.96
    ! the freezing probability underflows to zero and above roughly 0.99 it
    ! saturates the 1000 per litre cap, so most of the useful information
    ! lives in a narrow band that a coarse grid would miss entirely.
    real, parameter :: sat(12) = (/0.80, 0.90, 0.955, 0.960, 0.965,     &
                                   0.970, 0.975, 0.980, 0.985, 0.990,   &
                                   1.000, 1.100/)
    real, parameter :: na(5) = (/1.11e7, 1.0e8, 3.0e8, 3.0e9, 9.999e9/)
    real, parameter :: pref = 50000.0

    open(newunit=unit, file=trim(output_dir)//'/probe-icekoop.csv',     &
         status='replace', action='write')
    write(unit,'(A)') 'temp_k,qv,qvs,nwfa_per_m3,dt_s,xni_per_m3'
    dtv = 10.0
    do a = 1, size(tt)
       do b = 1, size(sat)
          do c = 1, size(na)
             temp = tt(a)
             qvs = RSLF(pref, temp)
             qv = sat(b) * qvs
             naero = na(c)
             value = iceKoop(temp, qv, qvs, naero, dtv)
             write(unit,'(6(ES24.16E3,:,","))') temp, qv, qvs, naero,   &
                  dtv, value
          enddo
       enddo
    enddo
    close(unit)
  end subroutine probe_koop

  subroutine probe_activ()
    integer :: unit, a, b, c
    real :: tt, ww, nccn, value
    ! Temperatures straddle every ta_Tk nearest-neighbour bin edge; note
    ! activ_ncloud does NOT interpolate in temperature.
    real, parameter :: tk(11) = (/238.0, 243.15, 248.0, 253.15, 258.0,  &
                                  263.15, 273.15, 283.15, 293.15,       &
                                  303.15, 310.0/)
    real, parameter :: wl(12) = (/0.005, 0.01, 0.02, 0.05, 0.1, 0.316,  &
                                  0.5, 1.0, 3.16, 10.0, 100.0, 200.0/)
    real, parameter :: nl(10) = (/5.0e6, 1.0e7, 1.11e7, 3.16e7, 1.0e8,  &
                                  3.16e8, 1.0e9, 3.16e9, 1.0e10,        &
                                  2.0e10/)

    open(newunit=unit, file=trim(output_dir)//'/probe-activncloud.csv', &
         status='replace', action='write')
    write(unit,'(A)') 'temp_k,w_m_s,nccn_per_m3,activated_per_m3'
    do a = 1, size(tk)
       do b = 1, size(wl)
          do c = 1, size(nl)
             tt = tk(a)
             ww = wl(b)
             nccn = nl(c)
             value = activ_ncloud(tt, ww, nccn)
             write(unit,'(4(ES24.16E3,:,","))') tt, ww, nccn, value
          enddo
       enddo
    enddo
    close(unit)
  end subroutine probe_activ

  subroutine probe_eff_aero()
    integer :: unit, a, b, c
    real :: dcol, daer, visco, rhoa, temp, value
    character(len=1) :: sp
    character(len=1), parameter :: species(3) = (/'r', 's', 'g'/)
    real, parameter :: dd(8) = (/5.0e-5, 1.0e-4, 3.0e-4, 5.0e-4,        &
                                 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3/)
    real, parameter :: da(2) = (/0.04e-6, 0.8e-6/)
    real, parameter :: tset(3) = (/240.0, 265.0, 285.0/)

    open(newunit=unit, file=trim(output_dir)//'/probe-effaero.csv',     &
         status='replace', action='write')
    write(unit,'(A)') 'species,d_collector_m,d_aerosol_m,visco,rho_kg_m3,temp_k,eff'
    do a = 1, size(species)
       do b = 1, size(dd)
          do c = 1, size(da)
             sp = species(a)
             dcol = dd(b)
             daer = da(c)
             temp = tset(mod(b - 1, size(tset)) + 1)
             ! Same viscosity law mp_thompson uses at 3199-3204.
             if (temp - 273.15 >= 0.0) then
                visco = (1.718 + 0.0049*(temp-273.15))*1.0e-5
             else
                visco = (1.718 + 0.0049*(temp-273.15)                   &
                         - 1.2e-5*(temp-273.15)*(temp-273.15))*1.0e-5
             endif
             rhoa = 1.0
             value = Eff_aero(dcol, daer, visco, rhoa, temp, sp)
             write(unit,'(A,6(",",ES24.16E3))') sp, dcol, daer, visco,  &
                  rhoa, temp, value
          enddo
       enddo
    enddo
    close(unit)
  end subroutine probe_eff_aero

  ! WIDENED FROM 14 ROWS TO 50.
  !
  ! Rows 1-14 are the ORIGINAL droplet ladder, unchanged and still first, so
  ! every committed value stays bit-identical: calc_effectRad's per-level
  ! work is independent once has_qc/has_qi/has_qs are set (:5636-5641), and
  ! all three are already true for the original 14, so lengthening the
  ! column cannot move them.
  !
  ! What those 14 could NOT measure: all of them carry t1d = 285 K and
  ! qs1d = 2e-4, so tc0 = MIN(-0.1, t-273.15) (:5662) is pinned at its clamp
  ! and the effs_m column is ONE state repeated fourteen times -- the snow
  ! branch's sa/sb polynomials (:5668-5690) and its 10.0**loga_ were
  ! untouched by this probe.  qi1d/ni1d were likewise a single pair, so
  ! neither of :5658's clamps was approached.
  !
  ! Rows 15-50 add four sweeps: temperature (12), snow mixing ratio (9),
  ! cloud mixing ratio including the rc <= R1 CYCLE (8), and ice mass/number
  ! including the 125 um clamp (7).
  subroutine probe_effect_rad()
    integer, parameter :: kts = 1, kte = 50
    integer :: unit, k, slot
    real :: t1d(kts:kte), p1d(kts:kte), qv1d(kts:kte), qc1d(kts:kte)
    real :: nc1d(kts:kte), qi1d(kts:kte), ni1d(kts:kte), qs1d(kts:kte)
    real :: nc_target(kts:kte)
    real :: re_qc1d(kts:kte), re_qi1d(kts:kte), re_qs1d(kts:kte)
    real :: rho
    ! Droplet number per cubic metre, chosen to straddle every branch in
    ! calc_effectRad including the two that mp_gt_driver cannot reach.
    real, parameter :: nc_m3(14) = (/2.0, 50.0, 99.0, 100.0, 101.0,     &
         1.0e3, 1.0e5, 1.0e7, 5.0e7, 1.0e8, 5.0e8, 1.999e9, 1.0e10,     &
         5.0e10/)
    real, parameter :: qc_set(14) = (/2.0e-12, 2.0e-12, 2.0e-12,        &
         1.0e-6, 1.0e-6, 1.0e-5, 1.0e-4, 3.0e-4, 3.0e-4, 3.0e-4,        &
         3.0e-4, 3.0e-4, 3.0e-4, 3.0e-4/)
    ! 273.05 sits just inside the tc0 clamp; the rest walk the snow fit down
    ! to -53 C, past the -40 C the committed after-columns reach.
    real, parameter :: t_sweep(12) = (/273.05, 272.0, 270.0, 265.0,     &
         260.0, 255.0, 250.0, 245.0, 240.0, 235.0, 228.0, 220.0/)
    ! The last rung is deliberately unphysical: it is the only way to reach
    ! :5697's 999 um upper clamp, which no committed column approaches.
    real, parameter :: qs_sweep(9) = (/1.0e-12, 1.0e-8, 1.0e-6, 1.0e-5, &
         5.0e-5, 2.0e-4, 1.0e-3, 5.0e-3, 5.0e-2/)
    real, parameter :: qc_sweep(8) = (/1.0e-12, 1.0e-9, 1.0e-7, 1.0e-6, &
         1.0e-5, 1.0e-4, 1.0e-3, 5.0e-3/)
    real, parameter :: qi_sweep(7) = (/1.0e-12, 1.0e-9, 1.0e-7, 1.0e-6, &
         1.0e-5, 1.0e-4, 1.0e-3/)
    ! effi goes as (ri/ni)**(1/bm_i), so the last pair is chosen to put lami
    ! past :5658's 125 um upper clamp -- the other end of the same MIN/MAX
    ! the first pair reaches from below through the ri <= R1 CYCLE.
    real, parameter :: ni_sweep(7) = (/1.0e2, 1.0e3, 1.0e4, 5.0e4,      &
         2.0e5, 1.0e6, 3.0e5/)

    do k = kts, kte
       if (k <= 14) then
          ! Block 1: the original droplet ladder, byte for byte.
          t1d(k) = 285.0
          p1d(k) = 90000.0
          qv1d(k) = 0.008
          qc1d(k) = qc_set(k)
          nc_target(k) = nc_m3(k)
          qi1d(k) = 1.0e-5
          ni1d(k) = 5.0e4
          qs1d(k) = 2.0e-4
       elseif (k <= 26) then
          ! Block 2: TEMPERATURE.  Everything else held fixed so the whole
          ! level-to-level variation in effs_m is tc0's.
          slot = k - 14
          t1d(k) = t_sweep(slot)
          p1d(k) = 70000.0
          qv1d(k) = 0.004
          qc1d(k) = 3.0e-4
          nc_target(k) = 1.0e8
          qi1d(k) = 1.0e-5
          ni1d(k) = 5.0e4
          qs1d(k) = 2.0e-4
       elseif (k <= 35) then
          ! Block 3: SNOW MASS at tc0 = -20 C, from the rs <= R1 CYCLE
          ! (:5661) through to the 999 um clamp (:5697).
          slot = k - 26
          t1d(k) = 253.15
          p1d(k) = 70000.0
          qv1d(k) = 0.001
          qc1d(k) = 3.0e-4
          nc_target(k) = 1.0e8
          qi1d(k) = 1.0e-5
          ni1d(k) = 5.0e4
          qs1d(k) = qs_sweep(slot)
       elseif (k <= 43) then
          ! Block 4: CLOUD MASS at fixed droplet number, from the rc <= R1
          ! CYCLE (:5638) through both of :5651's clamps.
          slot = k - 35
          t1d(k) = 285.0
          p1d(k) = 90000.0
          qv1d(k) = 0.008
          qc1d(k) = qc_sweep(slot)
          nc_target(k) = 1.0e8
          qi1d(k) = 1.0e-5
          ni1d(k) = 5.0e4
          qs1d(k) = 2.0e-4
       else
          ! Block 5: ICE MASS AND NUMBER together, so lami spans both of
          ! :5658's clamps.
          slot = k - 43
          t1d(k) = 253.15
          p1d(k) = 70000.0
          qv1d(k) = 0.001
          qc1d(k) = 3.0e-4
          nc_target(k) = 1.0e8
          qi1d(k) = qi_sweep(slot)
          ni1d(k) = ni_sweep(slot)
          qs1d(k) = 2.0e-4
       endif
       rho = 0.622*p1d(k)/(287.04*t1d(k)*(qv1d(k)+0.622))
       ! nc is seeded per cubic metre and divided by the local density, the
       ! way :5637 reads it back; ni1d is already the per-kilogram value the
       ! original 14 rows used, so it is passed straight through.
       nc1d(k) = nc_target(k)/rho
       re_qc1d(k) = 2.49e-6
       re_qi1d(k) = 4.99e-6
       re_qs1d(k) = 9.99e-6
    enddo

    call calc_effectRad(t1d, p1d, qv1d, qc1d, nc1d, qi1d, ni1d, qs1d,   &
         re_qc1d, re_qi1d, re_qs1d, kts, kte)

    open(newunit=unit, file=trim(output_dir)//'/probe-effectrad.csv',   &
         status='replace', action='write')
    write(unit,'(A)') 'k,temp_k,p_pa,qv,qc,nc_per_kg,nc_target_per_m3,qi,ni_per_kg,qs,effc_m,effi_m,effs_m'
    do k = kts, kte
       write(unit,'(I0,12(",",ES24.16E3))') k, t1d(k), p1d(k), qv1d(k), &
            qc1d(k), nc1d(k), nc_target(k), qi1d(k), ni1d(k), qs1d(k),  &
            re_qc1d(k), re_qi1d(k), re_qs1d(k)
    enddo
    close(unit)
  end subroutine probe_effect_rad

end program probe_aero_functions
