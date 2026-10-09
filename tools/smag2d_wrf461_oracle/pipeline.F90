! C ABI driver for the WRF v4.6.1 km_opt=4, diff_opt=2 column oracle.
!
! Storage only: every number is produced by an unmodified WRF routine, called
! in the order dyn_em/module_first_rk_step_part1/part2.F call them for one
! serial tile covering the whole domain.  WRF's own set_physical_bc2d/3d fills
! every halo, so the halo contents are exactly what a serial WRF run holds.
!
! Memory: i = -4:nx+5, k = 1:nz+1, j = -4:ny+5.  Domain: ids=1, ide=nx+1,
! jds=1, jde=ny+1, kds=1, kde=nz+1.  Tile = patch = domain.
! boundary: 0 periodic, 1 open, 2 specified (WRF real-data lateral boundaries).
module smag2d_wrf461_pipeline
use iso_c_binding
use module_oracle_config
use module_diffusion_em
implicit none
contains

subroutine pipeline_config(cfg, boundary, mix_full)
  type(grid_config_rec_type), intent(out) :: cfg
  integer, intent(in) :: boundary, mix_full
  cfg%open_xs = boundary == 1; cfg%open_xe = boundary == 1
  cfg%open_ys = boundary == 1; cfg%open_ye = boundary == 1
  cfg%specified = boundary == 2
  cfg%periodic_x = boundary == 0; cfg%periodic_y = boundary == 0
  cfg%nested = .false.; cfg%polar = .false.
  cfg%mix_full_fields = mix_full /= 0
  cfg%km_opt = 4; cfg%diff_opt = 2; cfg%sfs_opt = 0; cfg%m_opt = 0
  cfg%bl_pbl_physics = 1; cfg%cu_physics = 0; cfg%shcu_physics = 0
  cfg%use_theta_m = 0
end subroutine

subroutine smag2d_km4_pipeline(nx, ny, nz, boundary, mix_full, n_moist, n_scalar, &
    u, v, w, ph, phb, t2, alt, p, pb, moist, scalar, tke, msfu, msfv, msft, mut, &
    c1h, c2h, c1f, c2f, dn, dnw, fnm, fnp, znw, &
    dx, dy, rdx, rdy, cf1, cf2, cf3, ptop, cs, dt, &
    z, rdz, rdzw, zx, zy, rho, div, d11, d22, d33, d12, d13, d23, &
    kmh, kmv, khh, khv, bn2, tu, tv, tw, tth, ttke, tmoist, tscalar) bind(C)
  integer(c_int), value :: nx, ny, nz, boundary, mix_full, n_moist, n_scalar
  real(c_float), dimension(-4:nx+5, 1:nz+1, -4:ny+5), intent(inout) :: &
    u, v, w, ph, phb, t2, alt, p, pb, tke
  real(c_float), intent(inout) :: moist(-4:nx+5, 1:nz+1, -4:ny+5, n_moist)
  real(c_float), intent(inout) :: scalar(-4:nx+5, 1:nz+1, -4:ny+5, n_scalar)
  real(c_float), dimension(-4:nx+5, -4:ny+5), intent(inout) :: msfu, msfv, msft, mut
  real(c_float), dimension(nz+1), intent(in) :: c1h, c2h, c1f, c2f, dn, dnw, fnm, fnp, znw
  real(c_float), value :: dx, dy, rdx, rdy, cf1, cf2, cf3, ptop, cs, dt
  real(c_float), dimension(-4:nx+5, 1:nz+1, -4:ny+5), intent(inout) :: &
    z, rdz, rdzw, zx, zy, rho, div, d11, d22, d33, d12, d13, d23, &
    kmh, kmv, khh, khv, bn2, tu, tv, tw, tth, ttke
  real(c_float), intent(inout) :: tmoist(-4:nx+5, 1:nz+1, -4:ny+5, n_moist)
  real(c_float), intent(inout) :: tscalar(-4:nx+5, 1:nz+1, -4:ny+5, n_scalar)

  type(grid_config_rec_type) :: cfg
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, n
  real, dimension(-4:nx+5, 1:nz+1, -4:ny+5) :: th_phy, th_phy_m_t0, p_phy, pi_phy, &
    u_phy, v_phy, p8w, t_phy, t8w, z_phy, z_at_w, dz8w, p_hyd, p_hyd_w, &
    rublten, rvblten, rucuten, rvcuten, rushten, rvshten, xkmv_meso, dlk
  real, dimension(-4:nx+5, -4:ny+5) :: gamu, gamv, hpbl
  real :: ub(nz+1), vb(nz+1), fzm(nz+1), fzp(nz+1)
  real :: nba(-4:nx+5, 1:nz+1, -4:ny+5, 9)
  real :: chem(-4:nx+5, 1:nz+1, -4:ny+5, 1), tchem(-4:nx+5, 1:nz+1, -4:ny+5, 1)
  real :: tracer(-4:nx+5, 1:nz+1, -4:ny+5, 1), ttracer(-4:nx+5, 1:nz+1, -4:ny+5, 1)

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -4; ime = nx + 5; jms = -4; jme = ny + 5; kms = 1; kme = nz + 1
  call pipeline_config(cfg, boundary, mix_full)
  cfg%c_s = cs

  ! Halo state of the prognostic and static fields, as WRF's boundary
  ! routines leave it after each step (solve_em rk_phys_bc / start_em).
  call bc3(u, 'u'); call bc3(v, 'v'); call bc3(w, 'w')
  call bc3(ph, 'w'); call bc3(phb, 'w'); call bc3(t2, 't'); call bc3(alt, 't')
  call bc3(p, 'p'); call bc3(pb, 'p'); call bc3(tke, 't')
  do n = 2, n_moist
    call bc3(moist(:, :, :, n), 'p')
  end do
  do n = 2, n_scalar
    call bc3(scalar(:, :, :, n), 'p')
  end do
  call bc2(msfu, 'u'); call bc2(msfv, 'v'); call bc2(msft, 't'); call bc2(mut, 't')

  ! module_first_rk_step_part1: phy_prep.  The diffusion uses its rho;
  ! calculate_km_kh takes its th_phy/t_phy/p_phy/p8w/t8w.  WRF's FZM/FZP
  ! physics interpolation weights are the dycore FNM/FNP here; only the
  ! surface/top extrapolations read them and km_opt=4 consumes neither.
  fzm = fnm; fzp = fnp
  th_phy = 0.; th_phy_m_t0 = 0.; p_phy = 0.; pi_phy = 0.; u_phy = 0.; v_phy = 0.
  p8w = 0.; t_phy = 0.; t8w = 0.; z_phy = 0.; z_at_w = 0.; dz8w = 0.; p_hyd = 0.; p_hyd_w = 0.
  call phy_prep(cfg, mut, mut, mut, c1h, c2h, c1f, c2f, u, v, p, pb, alt, ph, phb, t2, &
    moist, n_moist, rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, &
    t8w, z_phy, z_at_w, dz8w, p_hyd, p_hyd_w, dnw, fzm, fzp, znw, ptop, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)

  ! module_first_rk_step_part2: metrics, their boundary fill, deformation,
  ! coefficients, phy_bc, then horizontal_diffusion_2.
  call compute_diff_metrics(cfg, ph, phb, z, rdz, rdzw, zx, zy, rdx, rdy, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)
  call bc3(rdzw, 'w'); call bc3(rdz, 'w'); call bc3(z, 'w')
  call bc3(zx, 'e'); call bc3(zy, 'f')

  ub = 0.; vb = 0.; nba = 0.
  call cal_deform_and_div(cfg, u, v, w, div, d11, d22, d33, d12, d13, d23, nba, 9, &
    ub, vb, msfu, msfu, msfv, msfv, msft, msft, rdx, rdy, dn, dnw, rdz, rdzw, &
    fnm, fnp, cf1, cf2, cf3, zx, zy, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)

  hpbl = 0.; dlk = 0.; xkmv_meso = 0.
  call calculate_km_kh(cfg, dt, 0., 0., 0, kmh, kmv, khh, khv, bn2, 0., 0., div, &
    d11, d22, d33, d12, d13, d23, tke, p8w, t8w, th_phy, t_phy, p_phy, moist, &
    dn, dnw, dx, dy, rdz, rdzw, 0, n_moist, cf1, cf2, cf3, .false., 0.1, &
    msft, msft, zx, zy, hpbl, dlk, xkmv_meso, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)

  rublten = 0.; rvblten = 0.; rucuten = 0.; rvcuten = 0.; rushten = 0.; rvshten = 0.
  gamu = 0.; gamv = 0.
  call phy_bc(cfg, div, d11, d22, d33, d12, d13, d23, kmh, kmv, khh, khv, tke, rho, &
    rublten, rvblten, rucuten, rvcuten, rushten, rvshten, gamu, gamv, xkmv_meso, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kde)

  chem = 0.; tchem = 0.; tracer = 0.; ttracer = 0.
  call horizontal_diffusion_2(tth, tu, tv, tw, ttke, tmoist, n_moist, tchem, 1, &
    tscalar, n_scalar, ttracer, 1, t2, th_phy, tke, cfg, d11, d22, d12, d13, d23, &
    nba, 9, div, moist, chem, scalar, tracer, msfu, msfu, msfv, msfv, msft, msft, &
    kmh, kmv, khh, 4, rdx, rdy, rdz, rdzw, fnm, fnp, cf1, cf2, cf3, zx, zy, dn, dnw, rho, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)

contains

  subroutine bc3(field, stagger)
    real, intent(inout) :: field(-4:nx+5, 1:nz+1, -4:ny+5)
    character, intent(in) :: stagger
    call set_physical_bc3d(field, stagger, cfg, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
      ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kde)
  end subroutine

  subroutine bc2(field, stagger)
    real, intent(inout) :: field(-4:nx+5, -4:ny+5)
    character, intent(in) :: stagger
    call set_physical_bc2d(field, stagger, cfg, ids, ide, jds, jde, ims, ime, jms, jme, &
      ids, ide, jds, jde, ids, ide, jds, jde)
  end subroutine

end subroutine

end module
