! Oracle driver: runs WRF v4.7.1's own ndown steps on the cases make_cases.py
! writes, one case file in, one output file out.
!
!   run_ndown rebalance CASE.bin OUT.bin
!   run_ndown blend     CASE.bin OUT.bin
!
! Every number is float32 (WRF's RWORDSIZE=4), little-endian, unformatted
! stream; arrays are Fortran order, i fastest, so numpy writes them C-order
! as (ny, nz, nx) for the 3-D fields and (ny, nx) for the 2-D ones.  The
! memory and patch bounds are WRF's for a one-patch domain: i = 1..nx+1,
! k = 1..nz+1, j = 1..ny+1 (the staggered ends), mass loops stopping at
! nx, nz, ny inside the routines themselves.
!
! rebalance case: int32 nx, ny, nz, hypsometric_opt; real t00, p00, tlp,
!   tiso, p_strat, tlp_strat, p_top; c1f, c2f, c3f, c4f (nz+1); c1h, c2h,
!   c3h, c4h, dnw, rdnw, rdn (nz); ht (the parent's terrain interpolated to
!   the child, ndown's ht_int), ht_fine (the blended child terrain), mu_2
!   (nx, ny); t_2, qv (nx, nz, ny).  Output: pb, t_init, alb (nx, nz, ny);
!   phb (nx, nz+1, ny); mub, ht (nx, ny); t_2, p, alt, al, p_hyd
!   (nx, nz, ny); ph_2, ph0 (nx, nz+1, ny); psfc (nx, ny).
! blend case: int32 nx, ny, spec_bdy_width, blend_width; real
!   ter_interpolated, ter_input (nx, ny).  Output: ter_input after the
!   blend (nx, ny).
PROGRAM run_ndown
   USE module_configure
   USE module_domain
   USE module_state_description, ONLY : P_QV, num_moist
   USE ndown_rebalance_host, ONLY : rebalance
   IMPLICIT NONE
   CHARACTER(LEN=512) :: mode, case_path, out_path
   CALL get_command_argument(1, mode)
   CALL get_command_argument(2, case_path)
   CALL get_command_argument(3, out_path)
   SELECT CASE (TRIM(mode))
   CASE ('rebalance')
      CALL do_rebalance()
   CASE ('blend')
      CALL do_blend()
   CASE DEFAULT
      WRITE (0, '(A)') 'usage: run_ndown rebalance|blend CASE OUT'
      STOP 2
   END SELECT
CONTAINS

   SUBROUTINE do_rebalance()
      TYPE(domain) :: grid
      INTEGER :: nx, ny, nz, hyps, u, i, j, k
      REAL, ALLOCATABLE :: moist(:,:,:,:)
      REAL, ALLOCATABLE :: buf2(:,:), buf3(:,:,:), buf1f(:), buf1h(:)
      OPEN (NEWUNIT=u, FILE=TRIM(case_path), ACCESS='stream', FORM='unformatted', STATUS='old')
      READ (u) nx, ny, nz, hyps
      READ (u) grid%t00, grid%p00, grid%tlp, grid%tiso, grid%p_strat, grid%tlp_strat, grid%p_top
      grid%hypsometric_opt = hyps
      grid%sd31 = 1 ; grid%ed31 = nx + 1
      grid%sd32 = 1 ; grid%ed32 = nz + 1
      grid%sd33 = 1 ; grid%ed33 = ny + 1
      grid%sm31 = 1 ; grid%em31 = nx + 1
      grid%sm32 = 1 ; grid%em32 = nz + 1
      grid%sm33 = 1 ; grid%em33 = ny + 1
      grid%sp31 = 1 ; grid%ep31 = nx + 1
      grid%sp32 = 1 ; grid%ep32 = nz + 1
      grid%sp33 = 1 ; grid%ep33 = ny + 1
      ALLOCATE (buf1f(nz + 1), buf1h(nz))
      ALLOCATE (grid%c1f(nz + 1), grid%c2f(nz + 1), grid%c3f(nz + 1), grid%c4f(nz + 1))
      ALLOCATE (grid%c1h(nz + 1), grid%c2h(nz + 1), grid%c3h(nz + 1), grid%c4h(nz + 1))
      ALLOCATE (grid%dnw(nz + 1), grid%rdnw(nz + 1), grid%rdn(nz + 1))
      grid%c1h = 0. ; grid%c2h = 0. ; grid%c3h = 0. ; grid%c4h = 0.
      grid%dnw = 0. ; grid%rdnw = 0. ; grid%rdn = 0.
      READ (u) buf1f ; grid%c1f = buf1f
      READ (u) buf1f ; grid%c2f = buf1f
      READ (u) buf1f ; grid%c3f = buf1f
      READ (u) buf1f ; grid%c4f = buf1f
      READ (u) buf1h ; grid%c1h(1:nz) = buf1h
      READ (u) buf1h ; grid%c2h(1:nz) = buf1h
      READ (u) buf1h ; grid%c3h(1:nz) = buf1h
      READ (u) buf1h ; grid%c4h(1:nz) = buf1h
      READ (u) buf1h ; grid%dnw(1:nz) = buf1h
      READ (u) buf1h ; grid%rdnw(1:nz) = buf1h
      READ (u) buf1h ; grid%rdn(1:nz) = buf1h
      ALLOCATE (grid%ht(nx + 1, ny + 1), grid%ht_fine(nx + 1, ny + 1))
      ALLOCATE (grid%mub(nx + 1, ny + 1), grid%mu_2(nx + 1, ny + 1), grid%psfc(nx + 1, ny + 1))
      ALLOCATE (buf2(nx, ny), buf3(nx, nz, ny))
      grid%ht = 0. ; grid%ht_fine = 0. ; grid%mub = 0. ; grid%mu_2 = 0. ; grid%psfc = 0.
      READ (u) buf2 ; grid%ht(1:nx, 1:ny) = buf2
      READ (u) buf2 ; grid%ht_fine(1:nx, 1:ny) = buf2
      READ (u) buf2 ; grid%mu_2(1:nx, 1:ny) = buf2
      ALLOCATE (grid%pb(nx + 1, nz + 1, ny + 1), grid%t_init(nx + 1, nz + 1, ny + 1))
      ALLOCATE (grid%alb(nx + 1, nz + 1, ny + 1), grid%phb(nx + 1, nz + 1, ny + 1))
      ALLOCATE (grid%t_2(nx + 1, nz + 1, ny + 1), grid%p(nx + 1, nz + 1, ny + 1))
      ALLOCATE (grid%alt(nx + 1, nz + 1, ny + 1), grid%al(nx + 1, nz + 1, ny + 1))
      ALLOCATE (grid%p_hyd(nx + 1, nz + 1, ny + 1), grid%ph_2(nx + 1, nz + 1, ny + 1))
      ALLOCATE (grid%ph0(nx + 1, nz + 1, ny + 1))
      grid%pb = 0. ; grid%t_init = 0. ; grid%alb = 0. ; grid%phb = 0.
      grid%t_2 = 0. ; grid%p = 0. ; grid%alt = 0. ; grid%al = 0.
      grid%p_hyd = 0. ; grid%ph_2 = 0. ; grid%ph0 = 0.
      ALLOCATE (moist(nx + 1, nz + 1, ny + 1, num_moist))
      moist = 0.
      READ (u) buf3 ; grid%t_2(1:nx, 1:nz, 1:ny) = buf3
      READ (u) buf3 ; moist(1:nx, 1:nz, 1:ny, P_QV) = buf3
      CLOSE (u)
      model_config_rec%use_theta_m = 0
      model_config_rec%use_baseparam_fr_nml = .FALSE.
      CALL rebalance ( grid, moist )
      OPEN (NEWUNIT=u, FILE=TRIM(out_path), ACCESS='stream', FORM='unformatted', STATUS='replace')
      WRITE (u) grid%pb(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%t_init(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%alb(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%phb(1:nx, 1:nz + 1, 1:ny)
      WRITE (u) grid%mub(1:nx, 1:ny)
      WRITE (u) grid%ht(1:nx, 1:ny)
      WRITE (u) grid%t_2(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%p(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%alt(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%al(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%p_hyd(1:nx, 1:nz, 1:ny)
      WRITE (u) grid%ph_2(1:nx, 1:nz + 1, 1:ny)
      WRITE (u) grid%ph0(1:nx, 1:nz + 1, 1:ny)
      WRITE (u) grid%psfc(1:nx, 1:ny)
      CLOSE (u)
   END SUBROUTINE do_rebalance

   SUBROUTINE do_blend()
      INTEGER :: nx, ny, sbw, bw, u
      REAL, ALLOCATABLE :: ter_int(:,:,:), ter_in(:,:,:), buf2(:,:)
      EXTERNAL blend_terrain
      OPEN (NEWUNIT=u, FILE=TRIM(case_path), ACCESS='stream', FORM='unformatted', STATUS='old')
      READ (u) nx, ny, sbw, bw
      ALLOCATE (ter_int(nx + 1, 1, ny + 1), ter_in(nx + 1, 1, ny + 1), buf2(nx, ny))
      ter_int = 0. ; ter_in = 0.
      READ (u) buf2 ; ter_int(1:nx, 1, 1:ny) = buf2
      READ (u) buf2 ; ter_in(1:nx, 1, 1:ny) = buf2
      CLOSE (u)
      harness_spec_bdy_width = sbw
      harness_blend_width = bw
      CALL blend_terrain ( ter_int , ter_in , &
                           1 , nx + 1 , 1 , ny + 1 , 1 , 1 , &
                           1 , nx + 1 , 1 , ny + 1 , 1 , 1 , &
                           1 , nx + 1 , 1 , ny + 1 , 1 , 1 )
      OPEN (NEWUNIT=u, FILE=TRIM(out_path), ACCESS='stream', FORM='unformatted', STATUS='replace')
      WRITE (u) ter_in(1:nx, 1, 1:ny)
      CLOSE (u)
   END SUBROUTINE do_blend

END PROGRAM run_ndown
