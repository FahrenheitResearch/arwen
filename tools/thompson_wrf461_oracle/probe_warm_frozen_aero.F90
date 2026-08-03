! WP-07 aerosol oracle: WRF's ALWAYS-RUN frozen collection block, evaluated
! ABOVE FREEZING.
!
! WHY THIS PROGRAM EXISTS
! -----------------------
! iiwarm is a PARAMETER .false. (module_mp_thompson.F:59), so the "frozen
! hydrometeor species" loop opened at :2239 has NO temperature guard and runs
! at EVERY model level, ambient-warm ones included.  Six of its rates have no
! mp=8 counterpart and are new in mp_physics=28:
!
!   pnc_scw   :2411-2412   droplets collected by snow          -> ncten
!   pnc_gcw   :2436-2437   droplets collected by graupel       -> ncten
!   pna_sca   :2444-2446   CCN scavenged by snow               -> nwfaten
!   pnd_scd   :2448-2450   IN  scavenged by snow               -> nifaten
!   pna_gca   :2462-2467   CCN scavenged by graupel            -> nwfaten
!   pnd_gcd   :2468-2471   IN  scavenged by graupel            -> nifaten
!
! ArWen puts them in gpuwm/core/kernels/thompson_aerosol_warm.cu for levels
! whose entry temperature is >= 273.15 K, because WP-06's cold kernel returns
! early there.  A MELTING LAYER is exactly that state -- snow and graupel
! falling through air warmer than 0 C -- so these are the mp=28 droplet and
! aerosol sinks that matter most in a real forecast, and until this program
! existed they were gated against no Fortran reference at all.
!
! This program also emits the two MASS companions prs_scw (:2407-2410) and
! prg_gcw (:2433-2435) plus every intermediate the rates are built from
! (twet, xDs, smoe, ilamg, N0_g, xDg, vtg, stoke_g, Ef_sw, Ef_gw), so a
! disagreement localizes to one WRF line instead of to a summed qc delta.
!
! WHAT IT IS
! ----------
! A scratch-free, deterministic driver that USEs the SAME compiled
! module_mp_thompson.o that build_aero.sh produces from
! wrf461-pristine/phys/module_mp_thompson.F (WRF v4.6.1, commit
! d66e442, zero local modifications).  It calls thompson_init exactly as
! run_column_aero.F90 does -- so t_Efsw, Ds and Eff_aero are WRF's own -- and
! then evaluates :1826-1949, :1972-2012, :2027-2139, :2168-2175 and
! :2402-2471 VERBATIM for a ladder of above-freezing states.
!
! cce/ccg/ocg1/ocg2, cse/csg, cge/cgg, sa/sb and the r_s/r_g tables are
! PRIVATE in the module, so they are rebuilt here with the module's own
! public WGAMMA using thompson_init's exact expressions (:671-685, :727-746,
! :753-770).  The rebuilt cge/cgg values are echoed on stdout so a reviewer
! can confirm they are the module's, not a transcription.
!
! HOW TO BUILD AND RUN
! --------------------
!   ./build_aero.sh /path/to/WRF-v4.6.1 /some/build-dir /path/CCN_ACTIVATE.BIN
!   ./build_probe_warm_frozen.sh /some/build-dir /out/dir
!
! build_probe_warm_frozen.sh is the sibling script; it reuses the objects
! build_aero.sh already produced and never rebuilds a .dat.  Compiler and
! flags are build_aero.sh's: gfortran -O2 -ffree-form, baseline x86-64, i.e.
! no FMA instruction.  That is load-bearing: nvrtc defaults to --fmad=true,
! so the CUDA side must contraction-pin every chain it compares here.
!
! TYPE FIDELITY IS THE POINT
! --------------------------
! WRF's declarations at :1578-1600 mix precisions deliberately and this
! program reproduces that mixing exactly:
!   ilamg, N0_g, lamg, lamc  DOUBLE PRECISION
!   smob..smog, mvd_c, mvd_g, xDs, xDg, vtg, stoke_g, Ef_*  REAL(4)
!   every p* rate                                           DOUBLE PRECISION
! so e.g. prg_gcw = rhof*t1_qg_qc*Ef_gw*rc  (a REAL(4) chain)  * N0_g
! (promoting to double) * ilamg**cge(9,idx_bg) -- and cge(9,idx_bg) is a
! REAL(4) 3.8899998664855957, NOT the decimal 3.89.  Every REAL(4) column is
! written as DBLE(x) so the GPU comparison is not floored by a text round
! trip.

program probe_warm_frozen_aero
  use module_mp_thompson, only: thompson_init, WGAMMA, Eff_aero,        &
       t_Efsw, Ds, t_Efrw, Dr, RSLF, t_dew, t_lcl, theta_e,       &
       compT_fr_The
  implicit none

  integer, parameter :: nz = 24, nx = 2, ny = 2
  real, parameter :: PI = 3.1415926536
  real, parameter :: rho_w = 1000.0
  real, parameter :: rho_i = 890.0
  real, parameter :: T_0 = 273.15

  real, parameter :: am_r = PI*rho_w/6.0
  real, parameter :: bm_r = 3.0
  real, parameter :: am_s = 0.069
  real, parameter :: bm_s = 2.0
  real, parameter :: bm_g = 3.0
  real, parameter :: mu_g = 0.0
  ! :77-79.  is_hail_aware is .false. for mp=8 and mp=28, so idx_bg is
  ! pinned to idx_bg1 = 5 and rho_g(5) = 400.
  integer, parameter :: idx_bg1 = 5
  real, parameter :: rho_g5 = 400.0
  real, parameter :: am_g5 = PI*rho_g5/6.0
  ! :463-464.  thompson_init restores the pre-hail fall-speed law into slot
  ! idx_bg1 whenever the optional ng argument is absent, which it is for
  ! both mp=8 and mp=28.
  real, parameter :: av_g5 = 442.0
  real, parameter :: bv_g5 = 0.89
  real, parameter :: av_s = 40.0
  real, parameter :: bv_s = 0.55
  real, parameter :: D0c = 1.E-6
  real, parameter :: D0r = 50.E-6
  real, parameter :: D0s = 300.E-6
  real, parameter :: R1 = 1.E-12
  real, parameter :: R2 = 1.E-6
  real, parameter :: Nt_c_max = 1999.E6
  real, parameter :: naIN1 = 0.5E6
  real, parameter :: rho_not = 101325.0/(287.05*298.0)
  real, parameter :: Rd = 287.04
  ! r_s(1) and r_g(1), the first entry of each content lookup table (:293,
  ! :301).  Both are 1.e-6.
  real, parameter :: r_s1 = 1.e-6
  real, parameter :: r_g1 = 1.e-6
  integer, parameter :: nbs = 100

  ! Field et al. (2005) snow moment fit, :358-363.
  real, dimension(10), parameter :: sa = (/                             &
       5.065339, -0.062659, -3.032362, 0.029469, -0.000285,             &
       0.31255,   0.000204,  0.003199, 0.0,      -0.015952 /)
  real, dimension(10), parameter :: sb = (/                             &
       0.476221, -0.015896,  0.165977, 0.007468, -0.000141,             &
       0.060366,  0.000079,  0.000594, 0.0,      -0.003577 /)

  real :: cce(5,15), ccg(5,15), ocg1(15), ocg2(15)
  real :: cse(17), csg(17)
  real :: cge(12), cgg(12)
  real :: oams, obmr, obmg, ogg1, ogg2, ogg3
  real :: t1_qs_qc, t1_qg_qc

  ! thompson_init scaffolding (mirrors run_column_aero.F90)
  real :: hgt(nx,nz,ny), nwfa3(nx,nz,ny), nifa3(nx,nz,ny), nbca3(nx,nz,ny)
  real :: nwfa2d(nx,ny), nbca2d(nx,ny)

  integer :: n, i, j, ic, unit_c, free_unit
  logical :: unit_opened
  character(len=512) :: outdir

  ! --- state ------------------------------------------------------------
  real :: pres, temp, qv1, qc1, ncpk, qs1, qg1, ngpk, nwfapk, nifapk, dt
  real :: rho, rhof, visco, tempc, orho, odts, qvs, satw, twet
  real :: dew_t, Tlcl, The
  real :: rc, nc, rs, rg, ng, nwfa_m3, nifa_m3
  real :: xDc, mvd_c, mvd_g, xDs, xDg, vtg, stoke_g
  real :: smob, smo2, smoc, smoe, tc0, a_, b_, loga_
  real :: Ef_sw, Ef_gw, Ef_sa, Ef_ga
  double precision :: lamc, lamg, ilamg, N0_g
  double precision :: prs_scw, pnc_scw, prg_gcw, pnc_gcw
  double precision :: pna_sca, pnd_scd, pna_gca, pnd_gcd
  ! --- :2144-2222, evaluated in replay mode ------------------------------
  real :: rr, nr, mvd_r, Dc_g, Dc_b, zeta1, zeta, taud, tau
  real :: Ef_rw, Ef_rr, Ef_ra
  double precision :: lamr, ilamr, N0_r
  double precision :: prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw
  double precision :: pnr_rcr, pna_rca, pnd_rcd
  real :: cre(16), crg(16), org2, org3, t1_qr_qc
  integer :: nu_c, idx, idx_snow, idx_cloud
  logical :: L_qc, L_qs, L_qg

  ! --- ladders ----------------------------------------------------------
  integer, parameter :: n_env = 4
  real, parameter :: p_ladder(n_env) =                                  &
       (/ 100000.0, 95000.0, 85000.0, 70000.0 /)
  ! Every temperature is ABOVE 273.15 K: this program exists to cover the
  ! states no committed fixture reaches.  273.16 is the tightest melting
  ! layer float32 can express one ulp-ish above T_0.
  real, parameter :: t_ladder(n_env) =                                  &
       (/ 280.0, 273.16, 276.5, 285.0 /)
  ! Relative humidity drives the twet branch at :2006: below 0.999 WRF
  ! iterates Bolton's wet bulb, at or above it twet stays at temp.
  integer, parameter :: n_rh = 3
  real, parameter :: rh_ladder(n_rh) = (/ 0.35, 0.90, 1.02 /)
  integer, parameter :: n_nc = 6
  real, parameter :: nc_ladder(n_nc) = (/                               &
       2.0, 7.0e7, 1.0e8, 2.5e8, 1.0e9, 5.0e9 /)
  integer, parameter :: n_qc = 4
  real, parameter :: qc_ladder(n_qc) = (/                               &
       2.0e-12, 1.0e-5, 3.0e-4, 5.0e-3 /)
  ! Paired frozen ladder.  Index 1 has neither species (L_qs/L_qg false),
  ! 2 sits in WRF's L_q* window but below ArWen's mass-content gate (see
  ! below), 3 has both below r_s(1)/r_g(1), 4-7 straddle the scavenging
  ! gates and the D0s riming gate.
  !
  ! INDEX 2 IS DELIBERATE.  WRF's L_qs(k)/L_qg(k) test the MIXING RATIO
  ! against R1 (:1906, :1915); thompson.cu -- and therefore this port's warm
  ! network, which copies it -- tests the mass content qs*rho instead.  At
  ! rho < 1 the two disagree over qs in (1e-12, 1e-12/rho], a window at most
  ! 18 percent wide around WRF's own R1 floor.  1.1e-12 with rho ~ 0.86 at
  ! (70000 Pa, 285 K) lands inside it, so the gap is MEASURED rather than
  ! assumed to be harmless.
  integer, parameter :: n_frz = 7
  real, parameter :: qs_ladder(n_frz) = (/                              &
       0.0, 1.1e-12, 5.0e-10, 2.0e-6, 5.0e-5, 5.0e-4, 4.0e-3 /)
  real, parameter :: qg_ladder(n_frz) = (/                              &
       0.0, 1.1e-12, 5.0e-10, 2.0e-6, 3.0e-4, 5.0e-5, 3.0e-3 /)
  ! Graupel mean-volume diameter targets: below D0r (clamps low), in
  ! range, above 25.4 mm (clamps high).  Index 4 is the sentinel ng1d = 0,
  ! which drives WRF's ng(k) <= R2 branch at :1924-1928 -- a whole branch
  ! thompson.cu has no counterpart for.
  integer, parameter :: n_mvdg = 4
  real, parameter :: mvdg_ladder(n_mvdg) =                              &
       (/ 3.0e-5, 1.5e-3, 4.0e-2, -1.0 /)
  integer, parameter :: n_aer = 2
  real, parameter :: nwfa_ladder(n_aer) = (/ 1.0e7, 3.0e9 /)
  real, parameter :: nifa_ladder(n_aer) = (/ 5.0e3, 5.0e8 /)

  integer :: i_env, i_rh, i_nc, i_qc, i_frz, i_mvdg, i_aer

  ! --- optional replay of arbitrary states (see REPLAY MODE below) ------
  character(len=512) :: statefile, header
  integer :: unit_s, unit_r, ios
  real :: s_qr, s_nrpk

  call get_command_argument(1, outdir)
  if (len_trim(outdir) == 0)                                            &
       error stop 'usage: probe_warm_frozen_aero OUTDIR [STATES.csv]'
  call get_command_argument(2, statefile)

  ! table_ccnAct OPENs CCN_ACTIVATE.BIN on the lowest free unit in 20..99
  ! and build_aero.sh scopes GFORTRAN_CONVERT_UNIT to unit 20 alone.  Abort
  ! rather than silently read the big-endian table as native.
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
  nwfa3 = 0.0
  nifa3 = 0.0
  nbca3 = 0.0
  nwfa2d = 0.0
  nbca2d = 0.0

  call thompson_init(hgt=hgt, nwfa2d=nwfa2d, nbca2d=nbca2d,             &
       nwfa=nwfa3, nifa=nifa3, nbca=nbca3, wif_input_opt=0,             &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                       &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                     &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  ! :671-685, verbatim (bv_c = 2.0).
  do n = 1, 15
     cce(1,n) = n + 1.
     cce(2,n) = bm_r + n + 1.
     cce(3,n) = bm_r + n + 4.
     cce(4,n) = n + 2.0 + 1.
     cce(5,n) = bm_r + n + 2.0 + 1.
     ccg(1,n) = WGAMMA(cce(1,n))
     ccg(2,n) = WGAMMA(cce(2,n))
     ccg(3,n) = WGAMMA(cce(3,n))
     ccg(4,n) = WGAMMA(cce(4,n))
     ccg(5,n) = WGAMMA(cce(5,n))
     ocg1(n) = 1./ccg(1,n)
     ocg2(n) = 1./ccg(2,n)
  enddo

  ! :727-746, only the snow entries this program reads.
  cse(1) = bm_s + 1.
  cse(13) = bv_s + 2.
  csg(1) = WGAMMA(cse(1))
  csg(13) = WGAMMA(cse(13))
  oams = 1./am_s

  ! :753-770 for m = idx_bg1.  cge(9) and cge(11) are REAL(4) sums, which
  ! is exactly why they are printed below: 3.8899998664855957 is NOT the
  ! decimal 3.89 and the difference is 9e-7 of prg_gcw at ilamg ~ 1e-3.
  cge(1) = bm_g + 1.
  cge(2) = mu_g + 1.
  cge(3) = bm_g + mu_g + 1.
  cge(6) = bm_g + mu_g + bv_g5 + 1.
  cge(9) = mu_g + bv_g5 + 3.
  cge(11) = 0.5*(bv_g5 + 5. + 2.*mu_g)
  do n = 1, 12
     if (n == 1 .or. n == 2 .or. n == 3 .or. n == 6                     &
          .or. n == 9 .or. n == 11) then
        cgg(n) = WGAMMA(cge(n))
     else
        cgg(n) = 0.0
     endif
  enddo
  obmr = 1./bm_r
  obmg = 1./bm_g
  ogg1 = 1./cgg(1)
  ogg2 = 1./cgg(2)
  ogg3 = 1./cgg(3)
  t1_qs_qc = PI*.25*av_s
  t1_qg_qc = PI*.25*av_g5*cgg(9)

  ! :705-726 and :786, only the rain entries the replay block reads.
  cre(1) = bm_r + 1.
  cre(2) = 0.0 + 1.                 ! mu_r + 1., mu_r = 0 (:103)
  cre(3) = bm_r + 0.0 + 1.
  cre(9) = 0.0 + 1.0 + 3.           ! mu_r + bv_r + 3., bv_r = 1 (:144)
  do n = 1, 16
     if (n == 1 .or. n == 2 .or. n == 3 .or. n == 9) then
        crg(n) = WGAMMA(cre(n))
     else
        crg(n) = 0.0
     endif
  enddo
  org2 = 1./crg(2)
  org3 = 1./crg(3)
  t1_qr_qc = PI*.25*4854.0 * crg(9)

  print '(A,1X,ES24.16E3)', 'WP07F_CGE9   ', dble(cge(9))
  print '(A,1X,ES24.16E3)', 'WP07F_CGE6   ', dble(cge(6))
  print '(A,1X,ES24.16E3)', 'WP07F_CGE11  ', dble(cge(11))
  print '(A,1X,ES24.16E3)', 'WP07F_CGG9   ', dble(cgg(9))
  print '(A,1X,ES24.16E3)', 'WP07F_CGG6   ', dble(cgg(6))
  print '(A,1X,ES24.16E3)', 'WP07F_CGG11  ', dble(cgg(11))
  print '(A,1X,ES24.16E3)', 'WP07F_CGG2   ', dble(cgg(2))
  print '(A,1X,ES24.16E3)', 'WP07F_CGG3   ', dble(cgg(3))
  print '(A,1X,ES24.16E3)', 'WP07F_OGG2   ', dble(ogg2)
  print '(A,1X,ES24.16E3)', 'WP07F_OGG3   ', dble(ogg3)
  print '(A,1X,ES24.16E3)', 'WP07F_AMG5   ', dble(am_g5)
  print '(A,1X,ES24.16E3)', 'WP07F_MVDGNUM', dble(3.0 + mu_g + 0.672)
  print '(A,1X,ES24.16E3)', 'WP07F_BVG    ', dble(bv_g5)
  print '(A,1X,ES24.16E3)', 'WP07F_T1QSQC ', dble(t1_qs_qc)
  print '(A,1X,ES24.16E3)', 'WP07F_T1QGQC ', dble(t1_qg_qc)

  open(newunit=unit_c,                                                  &
       file=trim(outdir)//'/wp07-warm-frozen-rates.csv',                &
       status='replace', action='write')
  write(unit_c,'(A)') 'case,pres,temp,qv,qc,nc_per_kg,qs,qg,'//         &
       'ng_per_kg,nwfa_per_kg,nifa_per_kg,dt,rho,rhof,visco,twet,'//    &
       'nc_m3,mvd_c,nwfa_m3,nifa_m3,xDs,smoe,ilamg,N0_g,xDg,vtg,'//     &
       'stoke_g,Ef_sw,Ef_gw,prs_scw,pnc_scw,prg_gcw,pnc_gcw,'//         &
       'pna_sca,pnd_scd,pna_gca,pnd_gcd'

  ic = 0
  dt = 20.0
  do i_env = 1, n_env
   do i_rh = 1, n_rh
    do i_nc = 1, n_nc
     do i_qc = 1, n_qc
      do i_frz = 1, n_frz
       do i_mvdg = 1, n_mvdg
        do i_aer = 1, n_aer
         ! Without graupel the mvd_g axis is degenerate.
         if (qg_ladder(i_frz) <= 0.0 .and. i_mvdg > 1) cycle

         pres = p_ladder(i_env)
         temp = t_ladder(i_env)
         qv1 = rh_ladder(i_rh) * RSLF(pres, temp)
         qv1 = max(1.0e-6, min(qv1, 0.05))
         qc1 = qc_ladder(i_qc)
         qs1 = qs_ladder(i_frz)
         qg1 = qg_ladder(i_frz)

         rho = 0.622*pres/(Rd*temp*(max(1.e-10,qv1)+0.622))
         ncpk = nc_ladder(i_nc)/rho
         nwfapk = nwfa_ladder(i_aer)/rho
         nifapk = nifa_ladder(i_aer)/rho
         if (qg1 > R1 .and. mvdg_ladder(i_mvdg) > 0.0) then
            ngpk = qg1 * (3.672/mvdg_ladder(i_mvdg))**3                 &
                 / (PI*rho_g5)
         else
            ! mvdg_ladder < 0 is the ng1d = 0 sentinel for WRF's
            ! ng(k) <= R2 branch (:1924-1928).
            ngpk = 0.0
         endif

         ic = ic + 1
         call frozen_rates()
         write(unit_c,'(I0,36(",",ES24.16E3))') ic,                     &
              dble(pres), dble(temp), dble(qv1), dble(qc1),             &
              dble(ncpk), dble(qs1), dble(qg1), dble(ngpk),             &
              dble(nwfapk), dble(nifapk), dble(dt),                     &
              dble(rho), dble(rhof), dble(visco), dble(twet),           &
              dble(nc), dble(mvd_c), dble(nwfa_m3), dble(nifa_m3),      &
              dble(xDs), dble(smoe), ilamg, N0_g,                       &
              dble(xDg), dble(vtg), dble(stoke_g),                      &
              dble(Ef_sw), dble(Ef_gw),                                 &
              prs_scw, pnc_scw, prg_gcw, pnc_gcw,                       &
              pna_sca, pnd_scd, pna_gca, pnd_gcd
        enddo
       enddo
      enddo
     enddo
    enddo
   enddo
  enddo
  close(unit_c)
  print '(A,1X,I0)', 'WP07F_WARM_FROZEN_ROWS', ic

  ! ---------------------------------------------------------------------
  ! REPLAY MODE.
  ! ---------------------------------------------------------------------
  ! With a second argument the program ALSO evaluates every rate for a
  ! caller-supplied list of states, which is how a committed column fixture's
  ! own entry column becomes an oracle row.  That is the only way to say
  ! which package owns a fixture's qr/nr residual without arguing from a
  ! hand-written float32 emulation: WRF evaluates :2144-2232 at EVERY level
  ! (the k-loop at :2156 sits ahead of both the `.not. iiwarm` branch at
  ! :2239 and the `temp(k) .lt. T_0` guard at :2554), so the autoconversion
  ! and accretion rates below are defined for a 240 K fixture level too.
  !
  ! Input CSV header, exactly:
  !   pres,temp,qv,qc,nc_per_kg,qr,nr_per_kg,qs,qg,ng_per_kg,nwfa_per_kg,
  !   nifa_per_kg,dt
  if (len_trim(statefile) > 0) then
     open(newunit=unit_s, file=trim(statefile), status='old',            &
          action='read', iostat=ios)
     if (ios /= 0) error stop 'cannot open states file'
     read(unit_s,'(A)') header
     open(newunit=unit_r,                                                &
          file=trim(outdir)//'/wp07-warm-frozen-replay.csv',             &
          status='replace', action='write')
     write(unit_r,'(A)') 'case,pres,temp,qv,qc,nc_per_kg,qr,nr_per_kg,'//&
          'qs,qg,ng_per_kg,nwfa_per_kg,nifa_per_kg,dt,rho,rhof,visco,'// &
          'twet,nc_m3,nu_c,lamc,mvd_c,xDc,nr_m3,lamr,mvd_r,N0_r,'//      &
          'prr_wau,pnr_wau,pnc_wau,prr_rcw,pnc_rcw,pnr_rcr,pna_rca,'//   &
          'pnd_rcd,xDs,smoe,ilamg,N0_g,prs_scw,pnc_scw,prg_gcw,'//       &
          'pnc_gcw,pna_sca,pnd_scd,pna_gca,pnd_gcd,xDg,vtg,'//        &
          'stoke_g,Ef_sw,Ef_gw'
     ic = 0
     do
        read(unit_s,*,iostat=ios) pres, temp, qv1, qc1, ncpk, s_qr,      &
             s_nrpk, qs1, qg1, ngpk, nwfapk, nifapk, dt
        if (ios /= 0) exit
        ! :1802.  The ladder loop forms rho itself; replay mode must too.
        rho = 0.622*pres/(Rd*temp*(max(1.e-10,qv1)+0.622))
        ic = ic + 1
        call frozen_rates()
        call warm_rain_rates(s_qr, s_nrpk)
        write(unit_r,'(I0,18(",",ES24.16E3),",",I0,32(",",ES24.16E3))')  &
             ic, dble(pres), dble(temp), dble(qv1), dble(qc1),           &
             dble(ncpk), dble(s_qr), dble(s_nrpk), dble(qs1), dble(qg1), &
             dble(ngpk), dble(nwfapk), dble(nifapk), dble(dt),           &
             dble(rho), dble(rhof), dble(visco), dble(twet), dble(nc),   &
             nu_c,                                                       &
             lamc, dble(mvd_c), dble(xDc), dble(nr), lamr, dble(mvd_r),  &
             N0_r, prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw,          &
             pnr_rcr, pna_rca, pnd_rcd, dble(xDs), dble(smoe), ilamg,    &
             N0_g, prs_scw, pnc_scw, prg_gcw, pnc_gcw, pna_sca,          &
             pnd_scd, pna_gca, pnd_gcd, dble(xDg), dble(vtg),            &
             dble(stoke_g), dble(Ef_sw), dble(Ef_gw)
     enddo
     close(unit_s)
     close(unit_r)
     print '(A,1X,I0)', 'WP07F_REPLAY_ROWS', ic
  endif

contains

  subroutine frozen_rates()

    orho = 1./rho
    odts = 1./dt
    tempc = temp - 273.15

    ! :1972-1996
    rhof = SQRT(rho_not/rho)
    qvs = RSLF(pres, temp)
    satw = qv1/qvs
    if (tempc .ge. 0.0) then
       visco = (1.718+0.0049*tempc)*1.0E-5
    else
       visco = (1.718+0.0049*tempc-1.2E-5*tempc*tempc)*1.0E-5
    endif

    ! :2001-2012.  Every state in this program's ladder has tempc > 0, so
    ! k_melting >= k and this level always takes the wet-bulb treatment.
    twet = temp
    if (satw .lt. 0.999) then
       dew_t = MIN(temp-0.001, t_dew(pres, qv1))
       Tlcl = t_lcl(temp, dew_t)
       The = theta_e(pres, temp, qv1, Tlcl)
       twet = MIN(temp, compT_fr_The(The, pres))
    endif

    ! :1804-1806, the aer_init_opt = 0 entry aerosol clamps.
    nwfa_m3 = MAX(11.1E6, MIN(9999.E6, nwfapk*rho))
    nifa_m3 = MAX(naIN1*0.01, MIN(9999.E6, nifapk*rho))

    ! :1826-1842 then :2168-2175.
    if (qc1 .gt. R1) then
       rc = qc1*rho
       nc = MAX(2., MIN(ncpk*rho, Nt_c_max))
       L_qc = .true.
       nu_c = MIN(15, NINT(1000.E6/nc) + 2)
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
    mvd_c = D0c
    if (L_qc) then
       nu_c = MIN(15, NINT(1000.E6/nc) + 2)
       xDc = MAX(D0c*1.E6, ((rc/(am_r*nc))**obmr) * 1.E6)
       lamc = (nc*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc)**obmr
       mvd_c = (3.0+nu_c+0.672) / lamc
       mvd_c = MAX(D0c, MIN(mvd_c, D0r))
    endif

    ! :1906-1913
    if (qs1 .gt. R1) then
       rs = qs1*rho
       L_qs = .true.
    else
       rs = R1
       L_qs = .false.
    endif

    ! :1915-1949, with idx_bg pinned to idx_bg1 because is_hail_aware is
    ! .false. (:1950).
    if (qg1 .gt. R1) then
       L_qg = .true.
       rg = qg1*rho
       ng = MAX(R2, ngpk*rho)
       if (ng .le. R2) then
          mvd_g = 1.5E-3
          lamg = (3.0 + mu_g + 0.672) / mvd_g
          ng = cgg(2)*ogg3*rg*lamg**bm_g / am_g5
       endif
       lamg = (am_g5*cgg(3)*ogg2*ng/rg)**obmg
       mvd_g = (3.0 + mu_g + 0.672) / lamg
       if (mvd_g .gt. 25.4E-3) then
          mvd_g = 25.4E-3
          lamg = (3.0 + mu_g + 0.672) / mvd_g
          ng = cgg(2)*ogg3*rg*lamg**bm_g / am_g5
       elseif (mvd_g .lt. D0r) then
          mvd_g = D0r
          lamg = (3.0 + mu_g + 0.672) / mvd_g
          ng = cgg(2)*ogg3*rg*lamg**bm_g / am_g5
       endif
    else
       rg = R1
       ng = R2
       mvd_g = 0.0
       L_qg = .false.
    endif

    ! :2027-2101, only the moments the frozen collection block reads.
    ! bm_s is 2.0 exactly, so smo2 == smob (:2033-2034).
    xDs = 0.0
    smob = 0.0
    smo2 = 0.0
    smoc = 0.0
    smoe = 0.0
    if (L_qs) then
       tc0 = MIN(-0.1, temp-273.15)
       smob = rs*oams
       smo2 = smob
       smoc = field_moment(tc0, smo2, cse(1))
       smoe = field_moment(tc0, smo2, cse(13))
       ! :2245-2250
       xDs = smoc/smob
    endif

    ! :2135-2139.  Unconditional over k; with no graupel it runs on the
    ! rg = R1 / ng = R2 placeholders and the rate gates below reject it.
    lamg = (am_g5*cgg(3)*ogg2*ng/rg)**obmg
    ilamg = 1./lamg
    N0_g = ng*ogg2*lamg**cge(2)

    prs_scw = 0.d0
    pnc_scw = 0.d0
    prg_gcw = 0.d0
    pnc_gcw = 0.d0
    pna_sca = 0.d0
    pnd_scd = 0.d0
    pna_gca = 0.d0
    pnd_gcd = 0.d0
    Ef_sw = 0.0
    Ef_gw = 0.0
    stoke_g = 0.0
    xDg = 0.0
    vtg = 0.0

    ! :2402-2440
    if (L_qc .and. mvd_c .gt. D0c) then
       if (xDs .gt. D0s) then
          idx_snow = 1 + INT(nbs*DLOG(DBLE(xDs)/Ds(1))                  &
               /DLOG(Ds(nbs)/Ds(1)))
          idx_snow = MIN(idx_snow, nbs)
          idx_cloud = INT(mvd_c*1.E6)
          Ef_sw = t_Efsw(idx_snow, idx_cloud)
          prs_scw = rhof*t1_qs_qc*Ef_sw*rc*smoe
          prs_scw = MIN(DBLE(rc*odts), prs_scw)
          pnc_scw = rhof*t1_qs_qc*Ef_sw*nc*smoe
          pnc_scw = MIN(DBLE(nc*odts), pnc_scw)
       endif
       if (rg .ge. r_g1 .and. mvd_c .gt. D0c) then
          xDg = (bm_g + mu_g + 1.) * ilamg
          vtg = rhof*av_g5*cgg(6)*ogg3 * ilamg**bv_g5
          stoke_g = mvd_c*mvd_c*vtg*rho_w/(9.*visco*xDg)
          if (stoke_g.ge.0.4 .and. stoke_g.le.10.) then
             Ef_gw = 0.55*ALOG10(2.51*stoke_g)
          elseif (stoke_g.lt.0.4) then
             Ef_gw = 0.0
          elseif (stoke_g.gt.10) then
             Ef_gw = 0.77
          endif
          if (twet .gt. T_0) Ef_gw = Ef_gw*0.1
          prg_gcw = rhof*t1_qg_qc*Ef_gw*rc*N0_g*ilamg**cge(9)
          pnc_gcw = rhof*t1_qg_qc*Ef_gw*nc*N0_g*ilamg**cge(9)
          pnc_gcw = MIN(DBLE(nc*odts), pnc_gcw)
       endif
    endif

    ! :2443-2450
    if (rs .gt. r_s1) then
       Ef_sa = Eff_aero(xDs, 0.04E-6, visco, rho, temp, 's')
       pna_sca = rhof*t1_qs_qc*Ef_sa*nwfa_m3*smoe
       pna_sca = MIN(DBLE(nwfa_m3*odts), pna_sca)

       Ef_sa = Eff_aero(xDs, 0.8E-6, visco, rho, temp, 's')
       pnd_scd = rhof*t1_qs_qc*Ef_sa*nifa_m3*smoe
       pnd_scd = MIN(DBLE(nifa_m3*odts), pnd_scd)
    endif

    ! :2460-2471
    if (rg .gt. r_g1) then
       xDg = (bm_g + mu_g + 1.) * ilamg
       Ef_ga = Eff_aero(xDg, 0.04E-6, visco, rho, temp, 'g')
       pna_gca = rhof*t1_qg_qc*Ef_ga*nwfa_m3*N0_g*ilamg**cge(9)
       pna_gca = MIN(DBLE(nwfa_m3*odts), pna_gca)

       Ef_ga = Eff_aero(xDg, 0.8E-6, visco, rho, temp, 'g')
       pnd_gcd = rhof*t1_qg_qc*Ef_ga*nifa_m3*N0_g*ilamg**cge(9)
       pnd_gcd = MIN(DBLE(nifa_m3*odts), pnd_gcd)
    endif

  end subroutine frozen_rates

  ! module_mp_thompson.F:1878-1898 then :2144-2222, VERBATIM.  Consumes the
  ! rc / nc / nu_c / lamc / mvd_c / xDc that frozen_rates already diagnosed,
  ! because WRF diagnoses them once per level and every rate here reads the
  ! same ones.  WRF evaluates this block at EVERY level: the k-loop at :2156
  ! opens before the `.not. iiwarm` branch (:2239) and before the
  ! `temp(k) .lt. T_0` guard (:2554).
  subroutine warm_rain_rates(qr_in, nr_in)
    real, intent(in) :: qr_in, nr_in
    logical :: L_qr

    prr_wau = 0.d0
    pnr_wau = 0.d0
    pnc_wau = 0.d0
    prr_rcw = 0.d0
    pnc_rcw = 0.d0
    pnr_rcr = 0.d0
    pna_rca = 0.d0
    pnd_rcd = 0.d0

    ! :1878-1898
    if (qr_in .gt. R1) then
       rr = qr_in*rho
       nr = MAX(R2, nr_in*rho)
       if (nr .le. R2) then
          mvd_r = 1.0E-3
          lamr = (3.0 + 0.0 + 0.672) / mvd_r
          nr = crg(2)*org3*rr*lamr**bm_r / am_r
       endif
       L_qr = .true.
       lamr = (am_r*crg(3)*org2*nr/rr)**obmr
       mvd_r = (3.0 + 0.0 + 0.672) / lamr
       if (mvd_r .gt. 2.5E-3) then
          mvd_r = 2.5E-3
          lamr = (3.0 + 0.0 + 0.672) / mvd_r
          nr = crg(2)*org3*rr*lamr**bm_r / am_r
       elseif (mvd_r .lt. D0r*0.75) then
          mvd_r = D0r*0.75
          lamr = (3.0 + 0.0 + 0.672) / mvd_r
          nr = crg(2)*org3*rr*lamr**bm_r / am_r
       endif
    else
       rr = R1
       nr = R2
       L_qr = .false.
    endif

    ! :2146-2151, unconditional over k
    lamr = (am_r*crg(3)*org2*nr/rr)**obmr
    ilamr = 1./lamr
    mvd_r = (3.0 + 0.0 + 0.672) / lamr
    N0_r = nr*org2*lamr**cre(2)

    ! :2159-2166
    if (L_qr .and. mvd_r .gt. D0r) then
       Ef_rr = 1.0 - EXP(2300.0*(mvd_r-1950.0E-6))
       pnr_rcr = Ef_rr * 2.0*nr*rr
    endif

    ! :2179-2194
    if (rc .gt. 0.01e-3 .and. L_qc) then
       Dc_g = ((ccg(3,nu_c)*ocg2(nu_c))**obmr / lamc) * 1.E6
       Dc_b = (xDc*xDc*xDc*Dc_g*Dc_g*Dc_g - xDc*xDc*xDc*xDc*xDc*xDc)     &
            **(1./6.)
       zeta1 = 0.5*((6.25E-6*xDc*Dc_b*Dc_b*Dc_b - 0.4)                   &
            + abs(6.25E-6*xDc*Dc_b*Dc_b*Dc_b - 0.4))
       zeta = 0.027*rc*zeta1
       taud = 0.5*((0.5*Dc_b - 7.5) + abs(0.5*Dc_b - 7.5)) + R1
       tau  = 3.72/(rc*taud)
       prr_wau = zeta/tau
       prr_wau = MIN(DBLE(rc*odts), prr_wau)
       pnr_wau = prr_wau / (am_r*nu_c*10.*D0r*D0r*D0r)
       pnc_wau = MIN(DBLE(nc*odts), prr_wau                              &
            / (am_r*mvd_c*mvd_c*mvd_c))
    endif

    ! :2197-2208
    if (L_qr .and. mvd_r.gt. D0r .and. mvd_c.gt. D0c) then
       idx = 1 + INT(100*DLOG(DBLE(mvd_r)/Dr(1))/DLOG(Dr(100)/Dr(1)))
       idx = MIN(idx, 100)
       Ef_rw = t_Efrw(idx, INT(mvd_c*1.E6))
       prr_rcw = rhof*t1_qr_qc*Ef_rw*rc*N0_r*((lamr+195.0)**(-cre(9)))
       prr_rcw = MIN(DBLE(rc*odts), prr_rcw)
       pnc_rcw = rhof*t1_qr_qc*Ef_rw*nc*N0_r*((lamr+195.0)**(-cre(9)))
       pnc_rcw = MIN(DBLE(nc*odts), pnc_rcw)
    endif

    ! :2211-2222
    if (L_qr .and. mvd_r.gt. D0r) then
       Ef_ra = Eff_aero(mvd_r,0.04E-6,visco,rho,temp,'r')
       pna_rca = rhof*t1_qr_qc*Ef_ra*nwfa_m3*N0_r                        &
            *((lamr+195.0)**(-cre(9)))
       pna_rca = MIN(DBLE(nwfa_m3*odts), pna_rca)

       Ef_ra = Eff_aero(mvd_r,0.8E-6,visco,rho,temp,'r')
       pnd_rcd = rhof*t1_qr_qc*Ef_ra*nifa_m3*N0_r                        &
            *((lamr+195.0)**(-cre(9)))
       pnd_rcd = MIN(DBLE(nifa_m3*odts), pnd_rcd)
    endif
  end subroutine warm_rain_rates

  ! Field et al. (2005) power-law moment, module_mp_thompson.F:2069-2101.
  ! One body for every moment because WRF's blocks differ only in which
  ! cse(n) is substituted for the moment exponent.
  real function field_moment(tc0_in, smo2_in, moment)
    real, intent(in) :: tc0_in, smo2_in, moment
    real :: la_, aa_, bb_
    la_ = sa(1) + sa(2)*tc0_in + sa(3)*moment                           &
        + sa(4)*tc0_in*moment + sa(5)*tc0_in*tc0_in                     &
        + sa(6)*moment*moment + sa(7)*tc0_in*tc0_in*moment              &
        + sa(8)*tc0_in*moment*moment + sa(9)*tc0_in*tc0_in*tc0_in       &
        + sa(10)*moment*moment*moment
    aa_ = 10.0**la_
    bb_ = sb(1) + sb(2)*tc0_in + sb(3)*moment                           &
        + sb(4)*tc0_in*moment + sb(5)*tc0_in*tc0_in                     &
        + sb(6)*moment*moment + sb(7)*tc0_in*tc0_in*moment              &
        + sb(8)*tc0_in*moment*moment + sb(9)*tc0_in*tc0_in*tc0_in       &
        + sb(10)*moment*moment*moment
    field_moment = aa_ * smo2_in**bb_
  end function field_moment

end program probe_warm_frozen_aero
