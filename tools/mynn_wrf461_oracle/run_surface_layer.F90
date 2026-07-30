program run_mynn_surface_layer_oracle
  use module_sf_mynn, only: mynn_sf_init_driver, SFCLAY1D_mynn
  implicit none

  integer, parameter :: ncase = 6
  integer, parameter :: ids = 1, ide = ncase + 1
  integer, parameter :: jds = 1, jde = 2, kds = 1, kde = 3
  integer, parameter :: ims = 1, ime = ncase
  integer, parameter :: jms = 1, jme = 1, kms = 1, kme = 2
  integer, parameter :: its = 1, ite = ncase
  integer, parameter :: jts = 1, jte = 1, kts = 1, kte = 2
  integer, parameter :: j = 1, itimestep = 1
  integer, parameter :: isfflx = 1, isftcflx = 0, iz0tlnd = 0
  integer, parameter :: spp_pbl = 0
  real, parameter :: cp = 1004.5, grav = 9.81, r = 287.0
  real, parameter :: rovcp = r / cp, xlv = 2.5e6, dx = 3000.0
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, rv = 461.6
  real, parameter :: ep1 = rv / r - 1.0, ep2 = r / rv
  real, parameter :: karman = 0.4

  character(len=32) :: case_name(ncase)
  character(len=1024) :: output_path
  integer :: i, unit
  real :: u1d(ncase), v1d(ncase), t1d(ncase), qv1d(ncase)
  real :: p1d(ncase), dz8w1d(ncase), rho1d(ncase)
  real :: u1d2(ncase), v1d2(ncase), dz2w1d(ncase)
  real :: psfcpa(ncase), chs(ncase), chs2(ncase), cqs2(ncase)
  real :: cpm(ncase), pblh(ncase), rmol(ncase), znt(ncase)
  real :: ust(ncase), mavail(ncase), zol(ncase), mol(ncase)
  real :: regime(ncase), psim(ncase), psih(ncase), xland(ncase)
  real :: hfx(ncase), qfx(ncase), tsk(ncase), u10(ncase), v10(ncase)
  real :: th2(ncase), t2(ncase), q2(ncase), flhc(ncase), flqc(ncase)
  real :: snowh(ncase), qgh(ncase), qsfc(ncase), lh(ncase)
  real :: gz1oz0(ncase), wspd(ncase), br(ncase), ch(ncase), qcg(ncase)
  real :: wstar(ncase), qstar(ncase), rstoch1d(ncase)
  real :: ustm(ncase), ck(ncase), cka(ncase), cd(ncase), cda(ncase)
  real :: hfx_input(ncase), qfx_input(ncase), znt_input(ncase)
  real :: qsfc_input(ncase), ust_input(ncase)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_surface_layer OUTPUT.csv'
    error stop 2
  end if

  case_name = [character(len=32) :: &
      'stable_land', 'unstable_land', 'neutral_land', 'snow_land', &
      'stable_water', 'unstable_water']

  u1d = [5.0, 6.0, 4.0, 3.0, 8.0, 10.0]
  v1d = [1.0, 2.0, 0.0, 1.0, 3.0, 2.0]
  u1d2 = [7.0, 8.0, 5.0, 4.0, 10.0, 12.0]
  v1d2 = [1.5, 2.5, 0.0, 1.5, 3.5, 2.5]
  t1d = [290.0, 300.0, 295.0, 268.0, 291.0, 298.0]
  tsk = [286.0, 307.0, 295.0, 265.0, 288.0, 303.0]
  qv1d = [0.0060, 0.0110, 0.0080, 0.0020, 0.0080, 0.0120]
  p1d = [99000.0, 98500.0, 100000.0, 99500.0, 99000.0, 98500.0]
  psfcpa = 100000.0
  dz8w1d = [40.0, 44.0, 40.0, 30.0, 36.0, 42.0]
  dz2w1d = [60.0, 66.0, 60.0, 45.0, 54.0, 63.0]
  pblh = [300.0, 1400.0, 600.0, 200.0, 350.0, 1100.0]
  xland = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
  snowh = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0]
  mavail = [0.45, 0.70, 0.55, 0.30, 1.0, 1.0]
  znt = [0.08, 0.12, 0.10, 0.03, 0.0002, 0.0002]
  hfx = [-30.0, 220.0, 0.0, -20.0, -10.0, 120.0]
  qfx = [-1.0e-6, 1.2e-4, 0.0, -2.0e-7, 2.0e-5, 8.0e-5]

  do i = 1, ncase
    rho1d(i) = p1d(i) / (r * t1d(i) * &
        (1.0 + ep1 * qv1d(i) / (1.0 + qv1d(i))))
  end do

  qsfc = qv1d / (1.0 + qv1d)
  ! Force water points through WRF's saturation-specific-humidity branch.
  qsfc(5:6) = 0.0
  qcg = qsfc
  qcg(5:6) = qv1d(5:6) / (1.0 + qv1d(5:6))
  ust = max(0.04 * sqrt(u1d * u1d + v1d * v1d), 0.001)
  ustm = ust

  chs = 0.0
  chs2 = 0.0
  cqs2 = 0.0
  cpm = 0.0
  rmol = 0.0
  zol = 0.0
  mol = 0.0
  regime = 0.0
  psim = 0.0
  psih = 0.0
  lh = qfx * xlv
  qgh = 0.0
  gz1oz0 = 0.0
  wspd = 0.0
  br = 0.0
  ch = 0.0
  flhc = 0.0
  flqc = 0.0
  u10 = 0.0
  v10 = 0.0
  th2 = 0.0
  t2 = 0.0
  q2 = 0.0
  wstar = 0.0
  qstar = 0.0
  ck = 0.0
  cka = 0.0
  cd = 0.0
  cda = 0.0
  rstoch1d = 0.0

  hfx_input = hfx
  qfx_input = qfx
  znt_input = znt
  qsfc_input = qsfc
  ust_input = ust

  call mynn_sf_init_driver(.false.)
  call SFCLAY1D_mynn( &
      j, u1d, v1d, t1d, qv1d, p1d, dz8w1d, rho1d, &
      u1d2, v1d2, dz2w1d, &
      cp, grav, rovcp, r, xlv, psfcpa, chs, chs2, cqs2, cpm, &
      pblh, rmol, znt, ust, mavail, zol, mol, regime, &
      psim, psih, xland, hfx, qfx, tsk, &
      u10, v10, th2, t2, q2, flhc, flqc, snowh, qgh, &
      qsfc, lh, gz1oz0, wspd, br, isfflx, dx, &
      svp1, svp2, svp3, svpt0, ep1, ep2, &
      karman, ch, qcg, itimestep, wstar, qstar, &
      spp_pbl, rstoch1d, &
      ids, ide, jds, jde, kds, kde, &
      ims, ime, jms, jme, kms, kme, &
      its, ite, jts, jte, kts, kte, &
      isftcflx, iz0tlnd, ustm, ck, cka, cd, cda)

  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,xland,snowh,u1,v1,t1,qv1,p1,rho1,dz1,' // &
      'u2,v2,dz2,psfc,tsk,pblh,mavail,hfx_input,qfx_input,znt_input,' // &
      'qsfc_input,ust_input,regime,zol,rmol,ust,ustm,mol,psim,psih,' // &
      'chs,chs2,cqs2,ch,flhc,flqc,qgh,qsfc,hfx,qfx,lh,u10,v10,th2,' // &
      't2,q2,gz1oz0,wspd,br,ck,cka,cd,cda,wstar,qstar,cpm'
  do i = 1, ncase
    write(unit, '(A,55(",",ES24.16E3))') trim(case_name(i)), &
        xland(i), snowh(i), u1d(i), v1d(i), t1d(i), qv1d(i), &
        p1d(i), rho1d(i), dz8w1d(i), u1d2(i), v1d2(i), &
        dz2w1d(i), psfcpa(i), tsk(i), pblh(i), mavail(i), &
        hfx_input(i), qfx_input(i), znt_input(i), qsfc_input(i), &
        ust_input(i), regime(i), zol(i), rmol(i), ust(i), ustm(i), &
        mol(i), psim(i), psih(i), chs(i), chs2(i), cqs2(i), ch(i), &
        flhc(i), flqc(i), qgh(i), qsfc(i), hfx(i), qfx(i), lh(i), &
        u10(i), v10(i), th2(i), t2(i), q2(i), gz1oz0(i), wspd(i), &
        br(i), ck(i), cka(i), cd(i), cda(i), wstar(i), qstar(i), cpm(i)
  end do
  close(unit)
end program run_mynn_surface_layer_oracle
