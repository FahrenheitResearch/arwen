program run_drag_suite
  ! NOAA's CCPP GSL drag suite (ccpp-physics edd85a55, physics/GWD/
  ! drag_suite.F90, byte-unmodified, the RRFS v1 pin) driven on the very
  ! columns WOOF's gwd_opt = 3 kernel is held to: the WRF oracle's published
  ! inputs, gpuwm/data/terrain_drag/oracle/gwdo/inputs.  Nothing here
  ! computes an expected value; it maps WRF's column layout onto CCPP's
  ! interface the way each host model hands its scheme a column, calls
  ! drag_suite_run once per grid length, and writes what CCPP returns.
  !
  ! The mapping, host by host (WRF: module_bl_gwdo_gsl.F gwdo_gsl; FV3:
  ! the CCPP argument table drag_suite.meta):
  !   winds    WRF rotates grid winds to zonal/meridional with sina/cosa and
  !            rotates the tendencies back; this driver does the same.
  !   heights  WRF hands gwdo2d heights above sea level (z) and layer depths
  !            (dz); FV3 hands geopotential above the surface (phil, phii).
  !            phil = g*(z - zsfc), phii = g*sum(dz), zsfc = z(1) - dz(1)/2.
  !   moisture WRF passes the water vapour mixing ratio; FV3 passes specific
  !            humidity, q = qv/(1+qv).
  !   mask     WRF xland 1 land / 2 water; FV3 slmsk 1 land / 0 sea.
  !   del      prsi(k) - prsi(k+1), both hosts.
  ! Switches as RRFS v1 sets them (det/input.nml:175-177, :203): gwd_opt 3,
  ! do_gsl_drag_ls_bl, do_gsl_drag_ss and do_gsl_drag_tofd all true.  Two
  ! cdmbgwd settings: (1, 1), which leaves the scheme at WRF's constants,
  ! and (3.5, 1.0), RRFS's own value (det/input.nml:164).
  !
  ! Two host values the scheme reads are inputs, never constants:
  !   IMX     the host's global x count, which scales the mesoscale
  !           coefficient: cleff = 0.5e-5 / sqrt(imx / 192)
  !           (drag_suite.F90:576-586).  FV3 hands lonr = gnx = 4 * cnx
  !           (atmos_model.F90:683, GFS_typedefs.F90:4622); for RRFS v1's
  !           North American tile cnx = npx - 1 = 3950 (det/input.nml:117),
  !           so IMX = 15800.
  !   DELTIM  the physics step, which the semi-implicit form drag and
  !           blocking read (drag_suite.F90:1371-1372).  RRFS v1 runs 36 s
  !           (dt_inner, det/input.nml:189); the WRF fixture runs 20 s.
  !
  ! usage: run_drag_suite INPUT_DIR OUTPUT_DIR IMX DELTIM
  use machine, only: kind_phys
  use drag_suite, only: drag_suite_run
  implicit none
  integer, parameter :: ncol = 64, nz = 40, nzi = nz + 1
  real(4) :: u3d(ncol, nz), v3d(ncol, nz), t3d(ncol, nz), qv3d(ncol, nz)
  real(4) :: p3d(ncol, nz), p3di(ncol, nzi), pi3d(ncol, nz), z(ncol, nz)
  real(4) :: dz(ncol, nz)
  real(4) :: var2d(ncol), oc12d(ncol), oa(ncol, 4), ol(ncol, 4)
  real(4) :: var2dss(ncol), oc12dss(ncol), oass(ncol, 4), olss(ncol, 4)
  real(4) :: sina(ncol), cosa(ncol), xland(ncol), br(ncol), pblh(ncol)
  integer :: kpbl(ncol)

  real(kind_phys) :: u1(ncol, nz), v1(ncol, nz), t1(ncol, nz), q1(ncol, nz)
  real(kind_phys) :: prsi(ncol, nzi), del(ncol, nz), prsl(ncol, nz)
  real(kind_phys) :: prslk(ncol, nz), phii(ncol, nzi), phil(ncol, nz)
  real(kind_phys) :: dudt(ncol, nz), dvdt(ncol, nz), dtdt(ncol, nz)
  real(kind_phys) :: rdxzb(ncol), dxa(ncol)
  real(kind_phys) :: var(ncol), oc1(ncol), oa4(ncol, 4), ol4(ncol, 4)
  real(kind_phys) :: varss(ncol), oc1ss(ncol), oa4ss(ncol, 4), ol4ss(ncol, 4)
  real(kind_phys) :: theta(ncol), sigma(ncol), gamma(ncol), elvmax(ncol)
  real(kind_phys), dimension(ncol, nz) :: dtaux_ms, dtauy_ms, dtaux_bl, &
       dtauy_bl, dtaux_ss, dtauy_ss, dtaux_fd, dtauy_fd
  real(kind_phys), dimension(ncol) :: dusfc, dvsfc, dus_ms, dvs_ms, dus_bl, &
       dvs_bl, dus_ss, dvs_ss, dus_fd, dvs_fd
  real(kind_phys) :: slmsk(ncol), br1(ncol), hpbl(ncol)
  real(kind_phys) :: dtend(ncol, nz, 1), spp_wts(ncol, nz), cdmbgwd(2)
  integer :: dtidx(1, 1)
  real(kind_phys), parameter :: g = 9.81_kind_phys, rd = 287.0_kind_phys
  real(kind_phys), parameter :: rv = 461.6_kind_phys
  real(kind_phys), parameter :: cp = 7.0_kind_phys * rd / 2.0_kind_phys
  real(kind_phys), parameter :: fv = rv / rd - 1.0_kind_phys
  real(kind_phys), parameter :: pi = 3.141592653_kind_phys
  real(kind_phys), parameter :: dxs(5) = (/ 1000.0_kind_phys, 3000.0_kind_phys, &
       5000.0_kind_phys, 9000.0_kind_phys, 15000.0_kind_phys /)
  real(kind_phys), parameter :: cdmbs(2, 2) = reshape((/ 1.0_kind_phys, &
       1.0_kind_phys, 3.5_kind_phys, 1.0_kind_phys /), (/ 2, 2 /))
  real(kind_phys) :: zsfc, zq
  character(len=1024) :: indir, outdir
  character(len=64) :: arg
  integer :: imx
  real(kind_phys) :: deltim
  character(len=64) :: tag
  character(len=512) :: errmsg
  integer :: errflg, i, k, n, m

  if (command_argument_count() /= 4) then
    write(*, *) 'usage: run_drag_suite INPUT_DIR OUTPUT_DIR IMX DELTIM'
    stop 1
  end if
  call get_command_argument(1, indir)
  call get_command_argument(2, outdir)
  call get_command_argument(3, arg); read(arg, *) imx
  call get_command_argument(4, arg); read(arg, *) deltim
  call read4(indir, 'u3d', u3d, ncol * nz); call read4(indir, 'v3d', v3d, ncol * nz)
  call read4(indir, 't3d', t3d, ncol * nz); call read4(indir, 'qv3d', qv3d, ncol * nz)
  call read4(indir, 'p3d', p3d, ncol * nz); call read4(indir, 'p3di', p3di, ncol * nzi)
  call read4(indir, 'pi3d', pi3d, ncol * nz); call read4(indir, 'z', z, ncol * nz)
  call read4(indir, 'dz', dz, ncol * nz)
  call read4(indir, 'var2d', var2d, ncol); call read4(indir, 'oc12d', oc12d, ncol)
  call read4(indir, 'oa', oa, ncol * 4); call read4(indir, 'ol', ol, ncol * 4)
  call read4(indir, 'var2dss', var2dss, ncol); call read4(indir, 'oc12dss', oc12dss, ncol)
  call read4(indir, 'oass', oass, ncol * 4); call read4(indir, 'olss', olss, ncol * 4)
  call read4(indir, 'sina', sina, ncol); call read4(indir, 'cosa', cosa, ncol)
  call read4(indir, 'xland', xland, ncol); call read4(indir, 'br', br, ncol)
  call read4(indir, 'pblh', pblh, ncol)
  call readi(indir, 'kpbl', kpbl, ncol)

  do i = 1, ncol
    zsfc = real(z(i, 1), kind_phys) - 0.5_kind_phys * real(dz(i, 1), kind_phys)
    zq = 0.0_kind_phys
    phii(i, 1) = 0.0_kind_phys
    do k = 1, nz
      u1(i, k) = real(u3d(i, k), kind_phys) * real(cosa(i), kind_phys) &
               - real(v3d(i, k), kind_phys) * real(sina(i), kind_phys)
      v1(i, k) = real(u3d(i, k), kind_phys) * real(sina(i), kind_phys) &
               + real(v3d(i, k), kind_phys) * real(cosa(i), kind_phys)
      t1(i, k) = real(t3d(i, k), kind_phys)
      q1(i, k) = real(qv3d(i, k), kind_phys) / (1.0_kind_phys + real(qv3d(i, k), kind_phys))
      prsl(i, k) = real(p3d(i, k), kind_phys)
      prslk(i, k) = real(pi3d(i, k), kind_phys)
      phil(i, k) = g * (real(z(i, k), kind_phys) - zsfc)
      zq = zq + real(dz(i, k), kind_phys)
      phii(i, k + 1) = g * zq
    end do
    do k = 1, nzi
      prsi(i, k) = real(p3di(i, k), kind_phys)
    end do
    do k = 1, nz
      del(i, k) = prsi(i, k) - prsi(i, k + 1)
    end do
    var(i) = var2d(i); oc1(i) = oc12d(i); varss(i) = var2dss(i); oc1ss(i) = oc12dss(i)
    do m = 1, 4
      oa4(i, m) = oa(i, m); ol4(i, m) = ol(i, m)
      oa4ss(i, m) = oass(i, m); ol4ss(i, m) = olss(i, m)
    end do
    slmsk(i) = merge(1.0_kind_phys, 0.0_kind_phys, xland(i) < 1.5)
    br1(i) = br(i); hpbl(i) = pblh(i)
  end do
  theta = 0.0_kind_phys; sigma = 0.0_kind_phys; gamma = 0.0_kind_phys
  elvmax = 0.0_kind_phys; spp_wts = 0.0_kind_phys; dtend = 0.0_kind_phys
  dtidx = 1

  do m = 1, 2
    cdmbgwd = cdmbs(:, m)
    do n = 1, size(dxs)
      dudt = 0.0_kind_phys; dvdt = 0.0_kind_phys; dtdt = 0.0_kind_phys
      dtaux_ms = 0.0_kind_phys; dtauy_ms = 0.0_kind_phys
      dtaux_bl = 0.0_kind_phys; dtauy_bl = 0.0_kind_phys
      dtaux_ss = 0.0_kind_phys; dtauy_ss = 0.0_kind_phys
      dtaux_fd = 0.0_kind_phys; dtauy_fd = 0.0_kind_phys
      dusfc = 0.0_kind_phys; dvsfc = 0.0_kind_phys
      dus_ms = 0.0_kind_phys; dvs_ms = 0.0_kind_phys; dus_bl = 0.0_kind_phys
      dvs_bl = 0.0_kind_phys; dus_ss = 0.0_kind_phys; dvs_ss = 0.0_kind_phys
      dus_fd = 0.0_kind_phys; dvs_fd = 0.0_kind_phys
      rdxzb = 0.0_kind_phys
      dxa = dxs(n)
      errflg = 0; errmsg = ''
      call drag_suite_run(ncol, nz, dvdt, dudt, dtdt, u1, v1, t1, q1, kpbl, &
           prsi, del, prsl, prslk, phii, phil, deltim, 7,                  &
           var, oc1, oa4, ol4, varss, oc1ss, oa4ss, ol4ss,                 &
           theta, sigma, gamma, elvmax,                                    &
           dtaux_ms, dtauy_ms, dtaux_bl, dtauy_bl,                         &
           dtaux_ss, dtauy_ss, dtaux_fd, dtauy_fd,                         &
           dusfc, dvsfc, dus_ms, dvs_ms, dus_bl, dvs_bl,                   &
           dus_ss, dvs_ss, dus_fd, dvs_fd,                                 &
           slmsk, br1, hpbl,                                               &
           g, cp, rd, rv, fv, pi, imx, cdmbgwd, 0, 0,                      &
           .false., 1, rdxzb, dxa, 3,                                      &
           .true., .true., .true.,                                         &
           dtend, dtidx, 1, 1, 1, 1, .false., .true., .false.,             &
           spp_wts, 0, errmsg, errflg)
      if (errflg /= 0) then
        write(*, *) 'drag_suite_run error: ', trim(errmsg)
        stop 2
      end if
      write(tag, '(A,I0,A,I0)') 'dx', nint(dxs(n)), '_cdmb', m
      ! Rotate back to the grid exactly as WRF's wrapper does.
      call write8(outdir, trim(tag)//'_du', dudt * spread(real(cosa, kind_phys), 2, nz) &
                  + dvdt * spread(real(sina, kind_phys), 2, nz), ncol * nz)
      call write8(outdir, trim(tag)//'_dv', -dudt * spread(real(sina, kind_phys), 2, nz) &
                  + dvdt * spread(real(cosa, kind_phys), 2, nz), ncol * nz)
      call write8(outdir, trim(tag)//'_dt', dtdt, ncol * nz)
      call write8(outdir, trim(tag)//'_dtaux_ms', rot_x(dtaux_ms, dtauy_ms), ncol * nz)
      call write8(outdir, trim(tag)//'_dtaux_bl', rot_x(dtaux_bl, dtauy_bl), ncol * nz)
      call write8(outdir, trim(tag)//'_dtaux_ss', rot_x(dtaux_ss, dtauy_ss), ncol * nz)
      call write8(outdir, trim(tag)//'_dtaux_fd', rot_x(dtaux_fd, dtauy_fd), ncol * nz)
      call write8(outdir, trim(tag)//'_dtauy_ms', rot_y(dtaux_ms, dtauy_ms), ncol * nz)
      call write8(outdir, trim(tag)//'_dtauy_bl', rot_y(dtaux_bl, dtauy_bl), ncol * nz)
      call write8(outdir, trim(tag)//'_dtauy_ss', rot_y(dtaux_ss, dtauy_ss), ncol * nz)
      call write8(outdir, trim(tag)//'_dtauy_fd', rot_y(dtaux_fd, dtauy_fd), ncol * nz)
    end do
  end do
  call write8(outdir, 'del', del, ncol * nz)
  write(*, '(A,I0,A,I0,A,F6.1)') 'drag_suite_run columns written, kind_phys = ', &
       kind_phys, ', imx = ', imx, ', deltim = ', deltim

contains

  function rot_x(ax, ay) result(r)
    real(kind_phys), intent(in) :: ax(ncol, nz), ay(ncol, nz)
    real(kind_phys) :: r(ncol, nz)
    r = ax * spread(real(cosa, kind_phys), 2, nz) + ay * spread(real(sina, kind_phys), 2, nz)
  end function rot_x

  function rot_y(ax, ay) result(r)
    real(kind_phys), intent(in) :: ax(ncol, nz), ay(ncol, nz)
    real(kind_phys) :: r(ncol, nz)
    r = -ax * spread(real(sina, kind_phys), 2, nz) + ay * spread(real(cosa, kind_phys), 2, nz)
  end function rot_y

  subroutine read4(dir, name, a, n)
    character(len=*), intent(in) :: dir, name
    integer, intent(in) :: n
    real(4), intent(out) :: a(n)
    integer :: u
    open(newunit=u, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old', action='read')
    read(u) a
    close(u)
  end subroutine read4

  subroutine readi(dir, name, a, n)
    character(len=*), intent(in) :: dir, name
    integer, intent(in) :: n
    integer, intent(out) :: a(n)
    integer :: u
    open(newunit=u, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old', action='read')
    read(u) a
    close(u)
  end subroutine readi

  subroutine write8(dir, name, a, n)
    character(len=*), intent(in) :: dir, name
    integer, intent(in) :: n
    real(kind_phys), intent(in) :: a(n)
    integer :: u
    open(newunit=u, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='replace', action='write')
    write(u) real(a, 8)
    close(u)
  end subroutine write8

end program run_drag_suite
