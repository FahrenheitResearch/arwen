! Oracle driver for WRF v4.7.1's slope_rad / topo_shading code.
!
! Reads one case from in.bin (stream, native byte order), runs the WRF
! statements extract.py cut out of the WRF sources, writes out.bin.  One
! process = one patch, exactly as a serial wrf.exe runs a domain.
!
! Layout of in.bin: int32 mode, nx, ny, then per mode (all float32
! arrays nx*ny, Fortran order, i fastest; mass points only):
!   mode 1  slope:     ht msftx msfty sina cosa, rdx rdy
!   mode 2  toposhad:  int32 nested; ht xlat xlong sina cosa ht_shad_in,
!                      xtime gmt radt declin dx dy shadlen
!   mode 3  adjust:    xlat xlong coszen diffuse_frac swdown gsw hrang
!                      slope slp_azi, int32 shadowmask(nx*ny), declin solcon
!   mode 4  diffuse:   int32 ruiz; coszen swdown ht swddif, solcon
!   mode 5  blend:     int32 spec_bdy_width blend_width; toposoil ht
!                      (smooth_cg_topo's blend_terrain on d01)
PROGRAM wrf_topo_oracle
  USE wrf_topo_blocks
  USE wrf_topo_inline
  USE module_configure, ONLY : stub_spec_bdy_width, stub_blend_width
  IMPLICIT NONE
  INTEGER :: mode, nx, ny, u, v, i, j, nested_i, ruiz_i
  INTEGER :: ids, ide, jds, jde, ims, ime, jms, jme
  REAL, ALLOCATABLE :: a(:,:), b(:,:), c(:,:), d(:,:), e(:,:), f(:,:)
  REAL, ALLOCATABLE :: g(:,:), h(:,:), p(:,:), q(:,:), r(:,:)
  INTEGER, ALLOCATABLE :: mask(:,:)
  REAL :: s1, s2, s3, s4, s5, s6, s7
  TYPE(grid_t) :: grid
  TYPE(config_t) :: config_flags

  OPEN(NEWUNIT=u, FILE='in.bin', ACCESS='STREAM', FORM='UNFORMATTED', &
       STATUS='OLD')
  OPEN(NEWUNIT=v, FILE='out.bin', ACCESS='STREAM', FORM='UNFORMATTED', &
       STATUS='REPLACE')
  READ(u) mode, nx, ny
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1

  SELECT CASE (mode)
  CASE (1)
    ALLOCATE(grid%ht(nx,ny), grid%msftx(nx,ny), grid%msfty(nx,ny), &
             grid%sina(nx,ny), grid%cosa(nx,ny), grid%toposlpx(nx,ny), &
             grid%toposlpy(nx,ny), grid%lap_hgt(nx,ny), grid%slope(nx,ny), &
             grid%slp_azi(nx,ny))
    READ(u) grid%ht, grid%msftx, grid%msfty, grid%sina, grid%cosa, &
            grid%rdx, grid%rdy
    ! start_em.F runs the block on the whole tile, its..ite = ids..ide.
    CALL slope_block(grid, config_flags, ids, ide, jds, jde, &
                     ids, ide, jds, jde)
    WRITE(v) grid%slope, grid%slp_azi, grid%toposlpx, grid%toposlpy

  CASE (2)
    ! Memory reaches two cells past the patch on every side: the scan
    ! reads that far (the jpe+3 / ipe+3 / ips-3 bounds of iteration 1).
    ims = -2; ime = nx + 3; jms = -2; jme = ny + 3
    ALLOCATE(a(nx,ny), b(nx,ny), c(nx,ny), d(nx,ny), e(nx,ny), f(nx,ny))
    READ(u) nested_i
    READ(u) a, b, c, d, e, f, s1, s2, s3, s4, s5, s6, s7
    ALLOCATE(g(ims:ime,jms:jme), h(ims:ime,jms:jme), p(ims:ime,jms:jme), &
             q(ims:ime,jms:jme), r(ims:ime,jms:jme), &
             mask(ims:ime,jms:jme))
    ! HT with WRF's specified/nested halo: set_physical_bc2d copies the
    ! edge mass value outward for a mass ('r') field (module_bc.F).
    DO j = jms, jme
      DO i = ims, ime
        g(i,j) = a(MIN(MAX(i,1),nx), MIN(MAX(j,1),ny))
      ENDDO
    ENDDO
    p = 0.; q = 1.; r = 0.; mask = 0; h = 0.
    p(1:nx,1:ny) = b          ! xlat
    r(1:nx,1:ny) = c          ! xlong (reuse r below after the call)
    ! sina/cosa and the shadow height need their own arrays
    CALL run_shad(nested_i /= 0, g, p, r, d, e, f, s1, s2, s3, s4, s5, s6, &
                  s7)
  CASE (3)
    ALLOCATE(a(nx,ny), b(nx,ny), c(nx,ny), d(nx,ny), e(nx,ny), f(nx,ny), &
             g(nx,ny), h(nx,ny), p(nx,ny), q(nx,ny), r(nx,ny), mask(nx,ny))
    READ(u) a, b, c, d, e, f, g, h, p, mask, s1, s2
    q = 0.; r = 0.
    ! xlat xlong coszen diffuse_frac swdown gsw hrang slope slp_azi
    CALL TOPO_RAD_ADJ_DRVR(a, b, c, mask, d, s1, e, f, q, r, s2, g, h, p, &
                           1, nx + 1, 1, ny + 1, 1, 1, &
                           1, nx, 1, ny, 1, 1, &
                           1, nx, 1, ny, 1, 1)
    WRITE(v) e, f, q, r
  CASE (4)
    ALLOCATE(a(nx,ny), b(nx,ny), c(nx,ny), d(nx,ny), e(nx,ny), f(nx,ny), &
             g(nx,ny))
    READ(u) ruiz_i
    READ(u) a, b, c, d, s1
    e = 0.; f = 0.
    CALL diffuse_block(a, b, c, s1, d, e, f, g, ruiz_i /= 0, 1, nx, 1, ny)
    WRITE(v) d, g
  CASE (5)
    ALLOCATE(a(nx,ny), b(nx,ny))
    READ(u) stub_spec_bdy_width, stub_blend_width
    READ(u) a, b
    ! real.exe on d01: CALL blend_terrain(grid%toposoil, grid%ht, ids,
    ! ide, jds, jde, 1, 1, ims, ime, jms, jme, 1, 1, ips, ipe, jps, jpe,
    ! 1, 1) -- one patch, ipe = ide.
    CALL blend_terrain(a, b, 1, nx + 1, 1, ny + 1, 1, 1, &
                       1, nx, 1, ny, 1, 1, &
                       1, nx + 1, 1, ny + 1, 1, 1)
    WRITE(v) b
  CASE DEFAULT
    STOP 'unknown mode'
  END SELECT
  CLOSE(u); CLOSE(v)

CONTAINS

  SUBROUTINE run_shad(nested, ht_loc_in, xlat_in, xlong_in, sina_in, &
                      cosa_in, ht_shad_in, xtime, gmt, radt, declin, dx, &
                      dy, shadlen)
    LOGICAL, INTENT(IN) :: nested
    REAL, INTENT(IN) :: ht_loc_in(ims:ime,jms:jme)
    REAL, INTENT(IN) :: xlat_in(ims:ime,jms:jme), xlong_in(ims:ime,jms:jme)
    REAL, INTENT(IN) :: sina_in(nx,ny), cosa_in(nx,ny), ht_shad_in(nx,ny)
    REAL, INTENT(IN) :: xtime, gmt, radt, declin, dx, dy, shadlen
    REAL :: ht_loc(ims:ime,jms:jme), ht_shad(ims:ime,jms:jme)
    REAL :: xlat(ims:ime,jms:jme), xlong(ims:ime,jms:jme)
    REAL :: sina(ims:ime,jms:jme), cosa(ims:ime,jms:jme)
    INTEGER :: shadowmask(ims:ime,jms:jme)
    ! pre_radiation_driver: ht_loc = ht over the whole memory, then (for a
    ! nest) spec_bdyfield has written the parent's shadow height into the
    ! outer rows of ht_shad before toposhad_init runs.
    ht_loc = ht_loc_in
    xlat = xlat_in; xlong = xlong_in
    sina = 0.; cosa = 1.
    sina(1:nx,1:ny) = sina_in; cosa(1:nx,1:ny) = cosa_in
    ht_shad = 0.; shadowmask = 0
    ht_shad(1:nx,1:ny) = ht_shad_in
    ! One patch covering the domain: niter = 1, patch and tile bounds
    ! ips..min(ipe,ide-1) = 1..nx (pre_radiation_driver:3429-3452).
    CALL toposhad_init(ht_shad, ht_loc, shadowmask, nested, 1, &
                       ids, ide, jds, jde, 1, 1, &
                       ims, ime, jms, jme, 1, 1, &
                       1, nx, 1, ny, 1, 1, &
                       1, nx, 1, ny, 1, 1)
    CALL toposhad(xlat, xlong, sina, cosa, xtime, gmt, radt, declin, &
                  dx, dy, ht_shad, ht_loc, 1, shadowmask, shadlen, &
                  ids, ide, jds, jde, 1, 1, &
                  ims, ime, jms, jme, 1, 1, &
                  1, nx, 1, ny, 1, 1, &
                  1, nx, 1, ny, 1, 1)
    WRITE(v) shadowmask(1:nx,1:ny), ht_shad(1:nx,1:ny), ht_loc(1:nx,1:ny)
  END SUBROUTINE run_shad

END PROGRAM wrf_topo_oracle
