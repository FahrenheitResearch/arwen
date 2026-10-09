! C ABI driver for the WRF v4.6.1 diff_opt=1 column oracle.
!
! Storage only: every number is produced by an unmodified WRF routine, called
! in the order a serial WRF run calls them on RK step 1 for one tile covering
! the whole domain:
!   module_first_rk_step_part1   phy_prep
!   module_first_rk_step_part2   compute_diff_metrics, set_physical_bc3d on
!                                the metrics, cal_deform_and_div,
!                                calculate_km_kh, phy_bc
!   module_em rk_tendency        diff_opt1 block (:802-878): horizontal_diffusion
!                                for u, v, w with xkmh; horizontal_diffusion_3dmp
!                                for theta with xkhh and t_init; under
!                                bl_pbl_physics = 0 the constant-kvdif vertical
!                                routines
!   module_em rk_scalar_tend     diff_opt1 block (:1380-1419) for every moist
!                                species, every number scalar and (km_opt = 2)
!                                TKE, with xkhh
! WRF's own set_physical_bc2d/3d fills every halo.
!
! Memory: i = -4:nx+5, k = 1:nz+1, j = -4:ny+5.  Domain: ids=1, ide=nx+1,
! jds=1, jde=ny+1, kds=1, kde=nz+1.  Tile = patch = domain.
! boundary: 0 periodic, 1 open, 2 specified (WRF real-data lateral boundaries).
! khdif = kvdif = 0: WOOF refuses khdif/kvdif > 0 with km_opt /= 1 under
! diff_opt = 1 (gpuwm/config.py), so the oracle covers exactly what WOOF runs.
module diffopt1_wrf461_pipeline
use iso_c_binding
use module_oracle_config
use module_diffusion_em
use module_model_constants, only: prandtl
implicit none
contains

subroutine pipeline_config(cfg, boundary, km_opt, isfflx, pbl, cs, ck)
  type(grid_config_rec_type), intent(out) :: cfg
  integer, intent(in) :: boundary, km_opt, isfflx, pbl
  real, intent(in) :: cs, ck
  cfg%open_xs = boundary == 1; cfg%open_xe = boundary == 1
  cfg%open_ys = boundary == 1; cfg%open_ye = boundary == 1
  cfg%specified = boundary == 2
  cfg%periodic_x = boundary == 0; cfg%periodic_y = boundary == 0
  cfg%nested = .false.; cfg%polar = .false.
  cfg%mix_full_fields = .false.
  cfg%km_opt = km_opt; cfg%diff_opt = 1; cfg%sfs_opt = 0; cfg%m_opt = 0
  cfg%bl_pbl_physics = pbl; cfg%isfflx = isfflx
  cfg%cu_physics = 0; cfg%shcu_physics = 0; cfg%use_theta_m = 0
  cfg%c_s = cs; cfg%c_k = ck
  ! diff_opt=2 surface constants; tke_km ignores them under diff_opt=1.
  cfg%tke_drag_coefficient = 0.0013; cfg%tke_heat_flux = 0.24
end subroutine

subroutine diffopt1_pipeline(nx, ny, nz, boundary, km_opt, isotropic, isfflx, pbl, &
    n_moist, n_scalar, &
    u, v, w, ph, phb, t2, t_init, alt, p, pb, moist, scalar, tke, msfu, msfv, msft, mut, &
    c1h, c2h, c1f, c2f, dn, dnw, fnm, fnp, znw, &
    dx, dy, rdx, rdy, cf1, cf2, cf3, ptop, cs, ck, dt, mix_upper_bound, &
    z, rdz, rdzw, zx, zy, div, d11, d22, d33, d12, d13, d23, &
    kmh, kmv, khh, khv, bn2, tu, tv, tw, tth, ttke, tmoist, tscalar) bind(C)
  integer(c_int), value :: nx, ny, nz, boundary, km_opt, isotropic, isfflx, pbl
  integer(c_int), value :: n_moist, n_scalar
  real(c_float), dimension(-4:nx+5, 1:nz+1, -4:ny+5), intent(inout) :: &
    u, v, w, ph, phb, t2, t_init, alt, p, pb, tke
  real(c_float), intent(inout) :: moist(-4:nx+5, 1:nz+1, -4:ny+5, n_moist)
  real(c_float), intent(inout) :: scalar(-4:nx+5, 1:nz+1, -4:ny+5, n_scalar)
  real(c_float), dimension(-4:nx+5, -4:ny+5), intent(inout) :: msfu, msfv, msft, mut
  real(c_float), dimension(nz+1), intent(in) :: c1h, c2h, c1f, c2f, dn, dnw, fnm, fnp, znw
  real(c_float), value :: dx, dy, rdx, rdy, cf1, cf2, cf3, ptop, cs, ck, dt, mix_upper_bound
  real(c_float), dimension(-4:nx+5, 1:nz+1, -4:ny+5), intent(inout) :: &
    z, rdz, rdzw, zx, zy, div, d11, d22, d33, d12, d13, d23, &
    kmh, kmv, khh, khv, bn2, tu, tv, tw, tth, ttke
  real(c_float), intent(inout) :: tmoist(-4:nx+5, 1:nz+1, -4:ny+5, n_moist)
  real(c_float), intent(inout) :: tscalar(-4:nx+5, 1:nz+1, -4:ny+5, n_scalar)

  type(grid_config_rec_type) :: cfg
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, n
  real, dimension(-4:nx+5, 1:nz+1, -4:ny+5) :: th_phy, th_phy_m_t0, p_phy, pi_phy, &
    u_phy, v_phy, p8w, t_phy, t8w, z_phy, z_at_w, dz8w, p_hyd, p_hyd_w, rho, &
    rublten, rvblten, rucuten, rvcuten, rushten, rvshten, xkmv_meso, dlk
  real, dimension(-4:nx+5, -4:ny+5) :: gamu, gamv, hpbl, msfvinv
  real :: ub(nz+1), vb(nz+1), qvb(nz+1), fzm(nz+1), fzp(nz+1), rdn(nz+1), rdnw(nz+1)
  real :: nba(-4:nx+5, 1:nz+1, -4:ny+5, 9)
  real, parameter :: khdif = 0., kvdif = 0.
  real :: khdq, kvdq

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -4; ime = nx + 5; jms = -4; jme = ny + 5; kms = 1; kme = nz + 1
  call pipeline_config(cfg, boundary, km_opt, isfflx, pbl, cs, ck)

  ! Halo state of the prognostic and static fields, as WRF's boundary
  ! routines leave it (solve_em rk_phys_bc / start_em).
  call bc3(u, 'u'); call bc3(v, 'v'); call bc3(w, 'w')
  call bc3(ph, 'w'); call bc3(phb, 'w'); call bc3(t2, 't'); call bc3(t_init, 't')
  call bc3(alt, 't'); call bc3(p, 'p'); call bc3(pb, 'p'); call bc3(tke, 't')
  do n = 2, n_moist
    call bc3(moist(:, :, :, n), 'p')
  end do
  do n = 2, n_scalar
    call bc3(scalar(:, :, :, n), 'p')
  end do
  call bc2(msfu, 'u'); call bc2(msfv, 'v'); call bc2(msft, 't'); call bc2(mut, 't')
  ! start_em: grid%msfvx_inv = 1./grid%msfvx (Lambert: msfvx = msfvy = msfv).
  msfvinv = 1. / msfv
  ! start_em: rdnw = 1./dnw, rdn = 1./dn (only the kvdif=0 vertical routines read them).
  rdnw = 1. / dnw
  rdn = 0.; rdn(2:nz) = 1. / dn(2:nz)

  ! module_first_rk_step_part1: phy_prep.  calculate_km_kh takes its
  ! th_phy/t_phy/p_phy/p8w/t8w.  FZM/FZP are the dycore FNM/FNP here.
  fzm = fnm; fzp = fnp
  th_phy = 0.; th_phy_m_t0 = 0.; p_phy = 0.; pi_phy = 0.; u_phy = 0.; v_phy = 0.
  p8w = 0.; t_phy = 0.; t8w = 0.; z_phy = 0.; z_at_w = 0.; dz8w = 0.; p_hyd = 0.; p_hyd_w = 0.
  rho = 0.
  call phy_prep(cfg, mut, mut, mut, c1h, c2h, c1f, c2f, u, v, p, pb, alt, ph, phb, t2, &
    moist, n_moist, rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, &
    t8w, z_phy, z_at_w, dz8w, p_hyd, p_hyd_w, dnw, fzm, fzp, znw, ptop, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)

  ! module_first_rk_step_part2 (diff_opt >= 1).
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
  call calculate_km_kh(cfg, dt, 0., 0., 0, kmh, kmv, khh, khv, bn2, khdif, kvdif, div, &
    d11, d22, d33, d12, d13, d23, tke, p8w, t8w, th_phy, t_phy, p_phy, moist, &
    dn, dnw, dx, dy, rdz, rdzw, isotropic, n_moist, cf1, cf2, cf3, .false., mix_upper_bound, &
    msft, msft, zx, zy, hpbl, dlk, xkmv_meso, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde)

  rublten = 0.; rvblten = 0.; rucuten = 0.; rvcuten = 0.; rushten = 0.; rvshten = 0.
  gamu = 0.; gamv = 0.
  call phy_bc(cfg, div, d11, d22, d33, d12, d13, d23, kmh, kmv, khh, khv, tke, rho, &
    rublten, rvblten, rucuten, rvcuten, rushten, rvshten, gamu, gamv, xkmv_meso, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
    ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kde)

  ! module_em.F rk_tendency, forward_step diff_opt1 block (rk_step = 1).
  call horizontal_diffusion('u', u, tu, mut, c1h, c2h, cfg, &
    msfu, msfu, msfv, msfvinv, msfv, msft, msft, khdif, kmh, rdx, rdy, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  call horizontal_diffusion('v', v, tv, mut, c1h, c2h, cfg, &
    msfu, msfu, msfv, msfvinv, msfv, msft, msft, khdif, kmh, rdx, rdy, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  call horizontal_diffusion('w', w, tw, mut, c1f, c2f, cfg, &
    msfu, msfu, msfv, msfvinv, msfv, msft, msft, khdif, kmh, rdx, rdy, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  khdq = 3.*khdif
  call horizontal_diffusion_3dmp('m', t2, tth, mut, c1h, c2h, cfg, t_init, &
    msfu, msfu, msfv, msfvinv, msfv, msft, msft, khdq, khh, rdx, rdy, &
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  if (pbl == 0) then
    call vertical_diffusion_u(u, tu, cfg, ub, c1h, c2h, alt, mut, rdn, rdnw, kvdif, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    call vertical_diffusion_v(v, tv, cfg, vb, c1h, c2h, alt, mut, rdn, rdnw, kvdif, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    call vertical_diffusion('w', w, tw, cfg, c1f, c2f, alt, mut, rdn, rdnw, kvdif, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    kvdq = 3.*kvdif
    call vertical_diffusion_3dmp(t2, tth, cfg, t_init, c1h, c2h, alt, mut, rdn, rdnw, kvdq, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  end if

  ! module_em.F rk_scalar_tend diff_opt1 block: khdq = khdif/prandtl, xkmhd = xkhh.
  khdq = khdif/prandtl
  kvdq = kvdif/prandtl
  qvb = 0.
  do n = 2, n_moist
    call scalar_mix(moist(:, :, :, n), tmoist(:, :, :, n), n == p_qv)
  end do
  do n = 2, n_scalar
    call scalar_mix(scalar(:, :, :, n), tscalar(:, :, :, n), .false.)
  end do
  if (km_opt == 2) call scalar_mix(tke, ttke, .false.)

contains

  subroutine scalar_mix(field, tend, is_qv)
    real, intent(in) :: field(-4:nx+5, 1:nz+1, -4:ny+5)
    real, intent(inout) :: tend(-4:nx+5, 1:nz+1, -4:ny+5)
    logical, intent(in) :: is_qv
    call horizontal_diffusion('m', field, tend, mut, c1h, c2h, cfg, &
      msfu, msfu, msfv, msfvinv, msfv, msft, msft, khdq, khh, rdx, rdy, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    if (pbl == 0) then
      if (is_qv) then
        call vertical_diffusion_mp(field, tend, cfg, qvb, c1h, c2h, alt, mut, rdn, rdnw, kvdq, &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
      else
        call vertical_diffusion('m', field, tend, cfg, c1h, c2h, alt, mut, rdn, rdnw, kvdq, &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
      end if
    end if
  end subroutine

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
