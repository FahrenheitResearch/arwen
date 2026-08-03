! Per-kernel Fortran reference for the SUB-FREEZING half of WRF's warm-rain
! loop (WP-06 wave 3).
!
! WHY THIS FILE EXISTS.  module_mp_thompson.F's warm-rain block at :2157-2234
! has NO temperature guard: every rate in it runs at every level, including
! levels far below freezing.  gpuwm gives those levels to
! thompson_aa_cold_network, so the cold kernel owns four rates
! (pnc_wau, pnc_rcw, pna_rca, pnd_rcd) plus both nu_c stages and mvd_c that
! WP-07's own oracle ladder -- entirely at or above 275 K -- never reaches.
! tests/test_thompson_aerosol_cold_gpu.py gates them against the table this
! program emits.  Until 2026-07-31 the program existed only in an agent
! scratch directory and was subsequently lost, which made the embedded
! _WRF_COLD_WARM_LOOP table in that test unverifiable by a reader.
!
! PROVENANCE HONESTY -- READ BEFORE TRUSTING THE ROW COUNT.  This file is a
! RECONSTRUCTION, not a recovery.  The original driver was not on disk.  What
! is recovered with certainty, because every one of them is printed in the
! committed test table, is:
!   * the five (p, T) environments and the qv law,           see p_ladder
!   * dt = 30 s,                                             see dt
!   * nine of the twelve entry droplet numbers,              see nc_ladder
!   * seven of the nine cloud contents,                      see qc_ladder
!   * all three rain contents and all three rain MVDs,       see qr/mvdr
!   * all three aerosol loadings.                            see nwfa/nifa
! The remaining five ladder entries (marked RECONSTRUCTED below) are pinned
! only indirectly, by the statistics tests/test_thompson_aerosol_cold_gpu.py
! records in prose: the sweep is 11340 rows, 3528 of them have
! nu_c(:1832) /= nu_c(:2170), 2058 of those also have prr_wau > 0, and the
! GPU-against-Fortran maxima over the full sweep are
!     nu_c_entry, nu_c_working, mvd_r   exact
!     pna_rca, pnd_rcd                  <= 7.5e-16
!     mvd_c                             <= 1.3e-7
!     nc_m3, pnc_rcw                    <= 3.3e-7
!     pnc_wau                           1.973e-6
!     prr_wau, pnr_wau                  2.311e-6
! The three row counts force the ladder SHAPE uniquely --
!     11340 = 5 env x 12 nc x 9 qc x (7 rain states x 3 aerosol loadings)
! -- and by themselves admit 46750 distinct ladders.  Requiring the residual
! maxima as well cuts that to 6060, and requiring every reconstructed entry
! to be one the sibling probe_warm_rates_aero.F90 ladder also uses leaves
! NINE, differing only in which two of {1.0e9, 1.999e9, 5.0e9} and which of
! {50.0, 2.0e7, 7.69e7} are taken.  This file takes 50.0, 1.0e9 and 1.999e9,
! matching the head and tail of that sibling ladder.  Every one of the nine
! reproduces every statistic above; nothing available distinguishes them.
!
! WHAT THAT DOES AND DOES NOT WEAKEN.  Every row of the committed
! _WRF_COLD_WARM_LOOP table is regenerated BIT-FOR-BIT by this program (see
! check_probe_oracles_aero.py), because every one of those rows lands on a
! ladder entry that is directly recovered.  Each of the eight full-sweep
! maxima above is likewise attained on the directly-recovered 9 x 7 sub-grid,
! so no quoted number depends on a reconstructed entry.  What a reconstructed
! entry could still change is a number nobody has quoted -- and the first
! candidate tried, qc = 5.0e-3, was REJECTED for exactly that reason: it
! raised the mvd_c maximum to 1.801e-7, above the 1.3e-7 the test records.
!
! HOW IT IS BUILT AND RUN.  tools/thompson_wrf461_oracle/build_aero_probes.sh.
! It links against the SAME compiled module_mp_thompson.o that
! tools/thompson_wrf461_oracle/build_aero.sh produces from
! wrf461-pristine/phys/module_mp_thompson.F (WRF v4.6.1, commit
! d66e442, zero local modifications) and calls thompson_init exactly as
! run_column_aero.F90 does, so t_Efrw and Dr are WRF's own arrays.
!
! WHAT IT EVALUATES.  module_mp_thompson.F:1826-1842 (BOTH nu_c stages and
! the droplet rediagnosis) and :2144-2232, verbatim, with the module's own
! public WGAMMA, Eff_aero, t_Efrw and Dr.  ccg/ocg1/ocg2/cce are PRIVATE in
! the module and are rebuilt from thompson_init's own expressions at :671-685.
!
! REAL(4) vs DOUBLE PRECISION follows WRF's declarations at :1579-1600.  nc is
! REAL: declaring it DOUBLE moves nc_m3 in the seventh digit and breaks the
! bit-for-bit reproduction of the committed table.

program probe_cold_warm_loop_aero
  use module_mp_thompson, only: thompson_init, WGAMMA, Eff_aero, t_Efrw, Dr
  implicit none

  integer, parameter :: nz = 24, nx = 2, ny = 2
  real, parameter :: PI = 3.1415926536
  real, parameter :: rho_w = 1000.0
  real, parameter :: am_r = PI*rho_w/6.0
  real, parameter :: bm_r = 3.0
  real, parameter :: mu_r = 0.0
  real, parameter :: av_r = 4854.0
  real, parameter :: bv_r = 1.0
  real, parameter :: fv_r = 195.0
  real, parameter :: D0c = 1.E-6
  real, parameter :: D0r = 50.E-6
  real, parameter :: R1 = 1.E-12
  real, parameter :: R2 = 1.E-6
  real, parameter :: Nt_c_max = 1999.E6
  real, parameter :: naIN1 = 5.0E5
  real, parameter :: rho_not = 101325.0/(287.05*298.0)
  real, parameter :: R = 287.04
  integer, parameter :: nbr = 100

  real :: cce(5,15), ccg(5,15), ocg1(15), ocg2(15)
  real :: cre(16), crg(16)
  real :: org2, org3, obmr, t1_qr_qc

  ! thompson_init scaffolding (mirrors run_column_aero.F90)
  real :: hgt(nx,nz,ny), nwfa(nx,nz,ny), nifa(nx,nz,ny), nbca(nx,nz,ny)
  real :: nwfa2d(nx,ny), nbca2d(nx,ny)

  integer :: n, i, j, ic, unit_c, free_unit
  logical :: unit_opened
  character(len=512) :: outdir

  real :: pres, temp, qv1, qc1, ncpk, qr1, nrpk, nwfapk, nifapk, dt
  real :: rho, rhof, visco, tempc, orho, odts
  real :: rc, nc, rr, nr, nwfa_m3, nifa_m3
  real :: xDc, mvd_c, mvd_r, Ef_rw, Ef_ra, Ef_rr
  real :: Dc_g, Dc_b, zeta1, zeta, taud, tau
  double precision :: lamc, lamr, ilamr, N0_r
  double precision :: prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw
  double precision :: pnr_rcr, pna_rca, pnd_rcd
  integer :: nu_c, nu_c_entry, nu_c_working, idx
  logical :: L_qc, L_qr

  ! ------------------------------------------------------------------
  ! Ladders.  Entries marked RECONSTRUCTED are the five that the committed
  ! test table does not print; see the header.
  ! ------------------------------------------------------------------
  integer, parameter :: n_env = 5
  real, parameter :: p_ladder(n_env) = (/ 95000.0, 80000.0, 60000.0,     &
                                          40000.0, 25000.0 /)
  real, parameter :: t_ladder(n_env) = (/ 272.0, 265.0, 255.0, 245.0,    &
                                          232.0 /)
  integer, parameter :: n_nc = 12
  real, parameter :: nc_ladder(n_nc) = (/                                &
       2.0,                                                              &
       50.0,                                                             &  ! RECONSTRUCTED
       1.0e4, 1.0e6, 3.0e7, 1.0e8, 1.5e8, 2.0e8, 3.0e8, 5.0e8,           &
       1.0e9,                                                            &  ! RECONSTRUCTED
       1.999e9 /)                                                           ! RECONSTRUCTED
  integer, parameter :: n_qc = 9
  real, parameter :: qc_ladder(n_qc) = (/                                &
       0.0, 2.0e-12,                                                     &
       1.0e-8,                                                           &  ! RECONSTRUCTED
       5.0e-6,                                                           &  ! RECONSTRUCTED
       1.0e-4, 1.0e-3, 4.0e-2, 8.0e-2, 1.5e-1 /)
  ! Seven rain states: one with no rain, then 2 rain contents x 3 rain MVDs.
  integer, parameter :: n_rain = 7
  real, parameter :: qr_state(n_rain) = (/ 0.0,                          &
       1.0e-5, 1.0e-5, 1.0e-5, 2.0e-3, 2.0e-3, 2.0e-3 /)
  real, parameter :: mvdr_state(n_rain) = (/ 0.0,                        &
       6.0e-5, 4.0e-4, 1.5e-3, 6.0e-5, 4.0e-4, 1.5e-3 /)
  integer, parameter :: n_aer = 3
  real, parameter :: nwfa_ladder(n_aer) = (/ 1.0e7, 3.0e8, 3.0e9 /)
  real, parameter :: nifa_ladder(n_aer) = (/ 5.0e3, 1.0e6, 5.0e9 /)

  integer :: i_env, i_nc, i_qc, i_rain, i_aer

  call get_command_argument(1, outdir)
  if (len_trim(outdir) == 0) then
     error stop 'usage: probe_cold_warm_loop_aero OUTDIR'
  endif

  ! build_aero_probes.sh scopes GFORTRAN_CONVERT_UNIT to unit 20 only, on the
  ! premise that table_ccnAct's 20..99 search lands there in a fresh process.
  free_unit = -1
  do i = 20, 99
     inquire(unit=i, opened=unit_opened)
     if (.not. unit_opened) then
        free_unit = i
        exit
     endif
  enddo
  if (free_unit /= 20) error stop 'unit 20 assumption violated'

  hgt = 0.0
  do j = 1, ny
     do i = 1, nx
        do n = 1, nz
           hgt(i,n,j) = (real(n) - 0.5) * 500.0
        enddo
     enddo
  enddo
  nwfa = 0.0
  nifa = 0.0
  nbca = 0.0
  nwfa2d = 0.0
  nbca2d = 0.0

  call thompson_init(hgt=hgt, nwfa2d=nwfa2d, nbca2d=nbca2d,             &
       nwfa=nwfa, nifa=nifa, nbca=nbca, wif_input_opt=0,                &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                       &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                     &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  ! module_mp_thompson.F:671-685, verbatim.
  do n = 1, 15
     cce(1,n) = n + 1.
     cce(2,n) = bm_r + n + 1.
     cce(3,n) = bm_r + n + 4.
     cce(4,n) = n + 2.0 + 1.          ! bv_c = 2.0
     cce(5,n) = bm_r + n + 2.0 + 1.
     ccg(1,n) = WGAMMA(cce(1,n))
     ccg(2,n) = WGAMMA(cce(2,n))
     ccg(3,n) = WGAMMA(cce(3,n))
     ccg(4,n) = WGAMMA(cce(4,n))
     ccg(5,n) = WGAMMA(cce(5,n))
     ocg1(n) = 1./ccg(1,n)
     ocg2(n) = 1./ccg(2,n)
  enddo

  ! module_mp_thompson.F:705-726, 786, only the rain entries used here.
  cre(1) = bm_r + 1.
  cre(2) = mu_r + 1.
  cre(3) = bm_r + mu_r + 1.
  cre(9) = mu_r + bv_r + 3.
  do n = 1, 16
     if (n == 1 .or. n == 2 .or. n == 3 .or. n == 9) then
        crg(n) = WGAMMA(cre(n))
     else
        crg(n) = 0.0
     endif
  enddo
  org2 = 1./crg(2)
  org3 = 1./crg(3)
  obmr = 1./bm_r
  t1_qr_qc = PI*.25*av_r * crg(9)

  open(newunit=unit_c, file=trim(outdir)//'/aero-cold-warm-loop.csv',   &
       status='replace', action='write')
  write(unit_c,'(A)') 'case,p_pa,temp_k,qv,qc,nc_per_kg,qr,'//          &
       'nr_per_kg,nwfa_per_kg,nifa_per_kg,dt,rho,rhof,visco,'//         &
       'nu_c_entry,nu_c_working,nc_m3,mvd_c,mvd_r,prr_wau,pnr_wau,'//   &
       'pnc_wau,pnc_rcw,pna_rca,pnd_rcd,lamc,xDc,nwfa_m3,nifa_m3,'//    &
       'nr_m3,lamr,N0_r'

  ic = 0
  dt = 30.0
  do i_env = 1, n_env
   do i_nc = 1, n_nc
    do i_qc = 1, n_qc
     do i_rain = 1, n_rain
      do i_aer = 1, n_aer

        pres = p_ladder(i_env)
        temp = t_ladder(i_env)
        ! Order-1e-3 vapour.  The association order matters: this is one
        ! REAL(4) ulp below 55.98/pres, and it is what the committed test
        ! table's qv column prints.  probe_warm_rates_aero.F90 uses the same
        ! line with 1000.0 for its warmer ladder.
        qv1  = 0.9 * 0.622 * 100.0 / pres
        qv1  = max(1.0e-5, min(qv1, 0.02))
        qc1  = qc_ladder(i_qc)
        qr1  = qr_state(i_rain)

        rho = 0.622*pres/(R*temp*(max(1.e-10,qv1)+0.622))
        ncpk = nc_ladder(i_nc)/rho
        nwfapk = nwfa_ladder(i_aer)/rho
        nifapk = nifa_ladder(i_aer)/rho
        if (qr1 > R1) then
           nrpk = qr1 * (3.672/mvdr_state(i_rain))**3 / (PI*rho_w)
        else
           nrpk = 0.0
        endif

        ic = ic + 1
        call cold_warm_loop()
        ! REAL(4) state columns are written from the REAL variables; every
        ! DOUBLE PRECISION rate is written as a DOUBLE so the GPU comparison
        ! is not floored by a float32 round trip.
        write(unit_c,'(I0,13(",",ES24.16E3),2(",",I0),16(",",ES24.16E3))') &
             ic,                                                        &
             dble(pres), dble(temp), dble(qv1), dble(qc1), dble(ncpk),  &
             dble(qr1), dble(nrpk), dble(nwfapk), dble(nifapk),         &
             dble(dt), dble(rho), dble(rhof), dble(visco),              &
             nu_c_entry, nu_c_working,                                  &
             dble(nc), dble(mvd_c), dble(mvd_r),                        &
             prr_wau, pnr_wau, pnc_wau, pnc_rcw, pna_rca, pnd_rcd,      &
             lamc, dble(xDc), dble(nwfa_m3), dble(nifa_m3),             &
             dble(nr), lamr, N0_r
      enddo
     enddo
    enddo
   enddo
  enddo
  close(unit_c)
  print '(A,1X,I0)', 'AERO_COLD_WARM_LOOP_ROWS', ic

contains

  ! module_mp_thompson.F:1798-1900 (cloud + rain entry diagnosis only),
  ! then :2144-2232.  Identical to probe_warm_rates_aero.F90's warm_rates
  ! except that both nu_c stages are reported separately.
  subroutine cold_warm_loop()

    orho = 1./rho
    odts = 1./dt
    tempc = temp - 273.15
    rhof = SQRT(rho_not/rho)
    if (tempc .ge. 0.0) then
       visco = (1.718+0.0049*tempc)*1.0E-5
    else
       visco = (1.718+0.0049*tempc-1.2E-5*tempc*tempc)*1.0E-5
    endif

    ! :1803-1804
    nwfa_m3 = MAX(11.1E6, MIN(9999.E6, nwfapk*rho))
    nifa_m3 = MAX(naIN1*0.01, MIN(9999.E6, nifapk*rho))

    prr_wau = 0.d0
    pnr_wau = 0.d0
    pnc_wau = 0.d0
    prr_rcw = 0.d0
    pnc_rcw = 0.d0
    pnr_rcr = 0.d0
    pna_rca = 0.d0
    pnd_rcd = 0.d0
    xDc = 0.0
    nu_c = 0
    nu_c_entry = 0
    nu_c_working = 0
    lamc = 0.d0

    ! :1826-1842
    if (qc1 .gt. R1) then
       rc = qc1*rho
       nc = MAX(2., MIN(ncpk*rho, Nt_c_max))
       L_qc = .true.
       nu_c = MIN(15, NINT(1000.E6/nc) + 2)
       nu_c_entry = nu_c
       lamc = (nc*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc)**obmr
       xDc = (bm_r + nu_c + 1.) / lamc
       if (xDc.lt. D0c) then
          lamc = cce(2,nu_c)/D0c
       elseif (xDc.gt. D0r*2.) then
          lamc = cce(2,nu_c)/(D0r*2.)
       endif
       nc = MIN( DBLE(Nt_c_max), ccg(1,nu_c)*ocg2(nu_c)*rc              &
            / am_r*lamc**bm_r)
    else
       rc = R1
       nc = 2.
       L_qc = .false.
    endif

    ! :1845-1870, rain entry diagnosis
    if (qr1 .gt. R1) then
       rr = qr1*rho
       nr = MAX(R2, nrpk*rho)
       if (nr .le. R2) then
          mvd_r = 1.0E-3
          lamr = (3.0 + mu_r + 0.672) / mvd_r
          nr = crg(2)*org3*rr*lamr**bm_r / am_r
       endif
       L_qr = .true.
       lamr = (am_r*crg(3)*org2*nr/rr)**obmr
       mvd_r = (3.0 + mu_r + 0.672) / lamr
       if (mvd_r .gt. 2.5E-3) then
          mvd_r = 2.5E-3
          lamr = (3.0 + mu_r + 0.672) / mvd_r
          nr = crg(2)*org3*rr*lamr**bm_r / am_r
       elseif (mvd_r .lt. D0r*0.75) then
          mvd_r = D0r*0.75
          lamr = (3.0 + mu_r + 0.672) / mvd_r
          nr = crg(2)*org3*rr*lamr**bm_r / am_r
       endif
    else
       rr = R1
       nr = R2
       L_qr = .false.
       mvd_r = 0.0
       lamr = 0.d0
    endif

    ! :2146-2151 (unconditional over k in WRF)
    lamr = (am_r*crg(3)*org2*nr/rr)**obmr
    ilamr = 1./lamr
    mvd_r = (3.0 + mu_r + 0.672) / lamr
    N0_r = nr*org2*lamr**cre(2)

    ! :2159-2166
    if (L_qr .and. mvd_r .gt. D0r) then
       Ef_rr = 1.0 - EXP(2300.0*(mvd_r-1950.0E-6))
       pnr_rcr = Ef_rr * 2.0*nr*rr
    endif

    ! :2168-2175
    mvd_c = D0c
    if (L_qc) then
       nu_c = MIN(15, NINT(1000.E6/nc) + 2)
       nu_c_working = nu_c
       xDc = MAX(D0c*1.E6, ((rc/(am_r*nc))**obmr) * 1.E6)
       lamc = (nc*am_r* ccg(2,nu_c) * ocg1(nu_c) / rc)**obmr
       mvd_c = (3.0+nu_c+0.672) / lamc
       mvd_c = MAX(D0c, MIN(mvd_c, D0r))
    endif

    ! :2179-2194
    if (rc .gt. 0.01e-3) then
       Dc_g = ((ccg(3,nu_c)*ocg2(nu_c))**obmr / lamc) * 1.E6
       Dc_b = (xDc*xDc*xDc*Dc_g*Dc_g*Dc_g - xDc*xDc*xDc*xDc*xDc*xDc)    &
            **(1./6.)
       zeta1 = 0.5*((6.25E-6*xDc*Dc_b*Dc_b*Dc_b - 0.4)                  &
            + abs(6.25E-6*xDc*Dc_b*Dc_b*Dc_b - 0.4))
       zeta = 0.027*rc*zeta1
       taud = 0.5*((0.5*Dc_b - 7.5) + abs(0.5*Dc_b - 7.5)) + R1
       tau  = 3.72/(rc*taud)
       prr_wau = zeta/tau
       prr_wau = MIN(DBLE(rc*odts), prr_wau)
       pnr_wau = prr_wau / (am_r*nu_c*10.*D0r*D0r*D0r)
       pnc_wau = MIN(DBLE(nc*odts), prr_wau                             &
            / (am_r*mvd_c*mvd_c*mvd_c))
    endif

    ! :2197-2208
    if (L_qr .and. mvd_r.gt. D0r .and. mvd_c.gt. D0c) then
       lamr = 1./ilamr
       idx = 1 + INT(nbr*DLOG(DBLE(mvd_r)/Dr(1))/DLOG(Dr(nbr)/Dr(1)))
       idx = MIN(idx, nbr)
       Ef_rw = t_Efrw(idx, INT(mvd_c*1.E6))
       prr_rcw = rhof*t1_qr_qc*Ef_rw*rc*N0_r                            &
            *((lamr+fv_r)**(-cre(9)))
       prr_rcw = MIN(DBLE(rc*odts), prr_rcw)
       pnc_rcw = rhof*t1_qr_qc*Ef_rw*nc*N0_r                            &
            *((lamr+fv_r)**(-cre(9)))
       pnc_rcw = MIN(DBLE(nc*odts), pnc_rcw)
    endif

    ! :2211-2222
    if (L_qr .and. mvd_r.gt. D0r) then
       Ef_ra = Eff_aero(mvd_r,0.04E-6,visco,rho,temp,'r')
       lamr = 1./ilamr
       pna_rca = rhof*t1_qr_qc*Ef_ra*nwfa_m3*N0_r                       &
            *((lamr+fv_r)**(-cre(9)))
       pna_rca = MIN(DBLE(nwfa_m3*odts), pna_rca)

       Ef_ra = Eff_aero(mvd_r,0.8E-6,visco,rho,temp,'r')
       pnd_rcd = rhof*t1_qr_qc*Ef_ra*nifa_m3*N0_r                       &
            *((lamr+fv_r)**(-cre(9)))
       pnd_rcd = MIN(DBLE(nifa_m3*odts), pnd_rcd)
    endif

  end subroutine cold_warm_loop

end program probe_cold_warm_loop_aero
