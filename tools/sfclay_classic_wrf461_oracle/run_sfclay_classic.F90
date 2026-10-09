! Column driver for WRF v4.6.1 phys/module_sf_sfclay.F (sf_sfclay_physics=91).
!
! Calls the byte-unmodified SFCLAY wrapper exactly as
! phys/module_surface_driver.F does for option 91 on an EM build: every
! optional argument present (ISFTCFLX, IZ0TLND, SCM_FORCE_FLUX=0, USTM, CK,
! CKA, CD, CDA), the constants from share/module_model_constants.F passed in
! the same order, and SFCLAYINIT run first, as phys_init does.
!
! One SFCLAY call per group and step covers the group's columns as one i-row
! (its=1..ncol, j=1, k=1..2), which is the shape a WRF tile hands it.  The
! inout state (ZNT UST USTM MOL HFX QFX QSFC ZOL and the rest) persists
! between the NSTEP calls exactly as WRF's state arrays do.
!
! Input: the text file make_columns.py writes (float32 words as unsigned
! integers).  Output, to the second argument:
!   TABLE n psimtb_word psihtb_word            (n = 0..1000)
!   ROW group step column  34 output words     (SFCLAY_OUTPUTS order)
! Words are written as signed 32-bit integers (TRANSFER of the REAL).
program run_sfclay_classic
   use module_sf_sfclay, only : sfclay, sfclayinit, psimtb, psihtb
   implicit none

   ! share/module_model_constants.F, v4.6.1
   real, parameter :: g = 9.81
   real, parameter :: r_d = 287.
   real, parameter :: cp = 7.*r_d/2.
   real, parameter :: r_v = 461.6
   real, parameter :: rcp = r_d/cp
   real, parameter :: xlv = 2.5e6
   real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65, svpt0 = 273.15
   real, parameter :: ep_1 = r_v/r_d - 1., ep_2 = r_d/r_v
   real, parameter :: karman = 0.4
   real, parameter :: eomeg = 7.2921e-5
   real, parameter :: stbolt = 5.67051e-8
   real, parameter :: p1000mb = 100000.

   integer, parameter :: nin = 20, nout = 34
   character(len=256) :: inpath, outpath
   character(len=64) :: gname
   integer :: ngroup, nstep, ig, istep, ncol, isfflx, isftcflx, iz0tlnd, i, k, n
   integer(8) :: dxw, wbuf(nin)
   real :: dxv
   integer :: scm_force_flux

   real, allocatable, dimension(:,:,:) :: u3d, v3d, t3d, qv3d, p3d, dz8w
   real, allocatable, dimension(:,:) :: psfc, chs, chs2, cqs2, cpm, znt, ust, &
        pblh, mavail, zol, mol, regime, psim, psih, fm, fh, xland, hfx, qfx, &
        lh, tsk, flhc, flqc, qgh, qsfc, rmol, u10, v10, th2, t2, q2, gz1oz0, &
        wspd, br, lakemask, dx, ustm, ck, cka, cd, cda
   real :: inval(nin)

   call get_command_argument(1, inpath)
   call get_command_argument(2, outpath)
   open(10, file=trim(inpath), status='old', action='read')
   open(20, file=trim(outpath), status='replace', action='write')

   call sfclayinit(.true.)
   do n = 0, 1000
      write(20, '(a,1x,i0,2(1x,i0))') 'TABLE', n, transfer(psimtb(n), 0), &
           transfer(psihtb(n), 0)
   end do

   scm_force_flux = 0
   read(10, *) ngroup, nstep
   do ig = 1, ngroup
      read(10, *) gname, ncol, isfflx, isftcflx, iz0tlnd, dxw
      dxv = word_to_real(dxw)
      allocate(u3d(ncol,2,1), v3d(ncol,2,1), t3d(ncol,2,1), qv3d(ncol,2,1), &
               p3d(ncol,2,1), dz8w(ncol,2,1))
      allocate(psfc(ncol,1), chs(ncol,1), chs2(ncol,1), cqs2(ncol,1), &
               cpm(ncol,1), znt(ncol,1), ust(ncol,1), pblh(ncol,1), &
               mavail(ncol,1), zol(ncol,1), mol(ncol,1), regime(ncol,1), &
               psim(ncol,1), psih(ncol,1), fm(ncol,1), fh(ncol,1), &
               xland(ncol,1), hfx(ncol,1), qfx(ncol,1), lh(ncol,1), &
               tsk(ncol,1), flhc(ncol,1), flqc(ncol,1), qgh(ncol,1), &
               qsfc(ncol,1), rmol(ncol,1), u10(ncol,1), v10(ncol,1), &
               th2(ncol,1), t2(ncol,1), q2(ncol,1), gz1oz0(ncol,1), &
               wspd(ncol,1), br(ncol,1), lakemask(ncol,1), dx(ncol,1), &
               ustm(ncol,1), ck(ncol,1), cka(ncol,1), cd(ncol,1), cda(ncol,1))
      ! Diagnostics that SFCLAY only writes on some paths start at zero, the
      ! value a WRF state array holds before its first physics call.
      chs = 0.; chs2 = 0.; cqs2 = 0.; cpm = 0.; regime = 0.; psim = 0.
      psih = 0.; fm = 0.; fh = 0.; lh = 0.; flhc = 0.; flqc = 0.; qgh = 0.
      rmol = 0.; u10 = 0.; v10 = 0.; th2 = 0.; t2 = 0.; q2 = 0.; gz1oz0 = 0.
      wspd = 0.; br = 0.; ck = 0.; cka = 0.; cd = 0.; cda = 0.
      u3d = 0.; v3d = 0.; t3d = 0.; qv3d = 0.; p3d = 0.; dz8w = 0.
      do i = 1, ncol
         read(10, *) wbuf
         do k = 1, nin
            inval(k) = word_to_real(wbuf(k))
         end do
         u3d(i,1,1) = inval(1);  v3d(i,1,1) = inval(2)
         t3d(i,1,1) = inval(3);  qv3d(i,1,1) = inval(4)
         p3d(i,1,1) = inval(5);  dz8w(i,1,1) = inval(6)
         psfc(i,1) = inval(7);   tsk(i,1) = inval(8)
         znt(i,1) = inval(9);    ust(i,1) = inval(10)
         ustm(i,1) = inval(11);  pblh(i,1) = inval(12)
         mavail(i,1) = inval(13); xland(i,1) = inval(14)
         lakemask(i,1) = inval(15); mol(i,1) = inval(16)
         hfx(i,1) = inval(17);   qfx(i,1) = inval(18)
         qsfc(i,1) = inval(19);  zol(i,1) = inval(20)
         dx(i,1) = dxv
      end do

      do istep = 1, nstep
         call sfclay(u3d, v3d, t3d, qv3d, p3d, dz8w,                    &
              cp, g, rcp, r_d, xlv, psfc, chs, chs2, cqs2, cpm,          &
              znt, ust, pblh, mavail, zol, mol, regime, psim, psih,      &
              fm, fh,                                                    &
              xland, hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, rmol,     &
              u10, v10, th2, t2, q2,                                     &
              gz1oz0, wspd, br, isfflx, dx,                              &
              svp1, svp2, svp3, svpt0, ep_1, ep_2,                       &
              karman, eomeg, stbolt,                                     &
              p1000mb, lakemask,                                         &
              1, ncol+1, 1, 2, 1, 2,                                     &
              1, ncol, 1, 1, 1, 2,                                       &
              1, ncol, 1, 1, 1, 1,                                       &
              ustm, ck, cka, cd, cda,                                    &
              isftcflx, iz0tlnd, scm_force_flux)
         do i = 1, ncol
            write(20, '(a,3(1x,i0),34(1x,i0))') 'ROW', ig, istep, i,     &
                 transfer(znt(i,1), 0), transfer(ust(i,1), 0),           &
                 transfer(ustm(i,1), 0), transfer(mol(i,1), 0),          &
                 transfer(hfx(i,1), 0), transfer(qfx(i,1), 0),           &
                 transfer(qsfc(i,1), 0), transfer(zol(i,1), 0),          &
                 transfer(regime(i,1), 0), transfer(psim(i,1), 0),       &
                 transfer(psih(i,1), 0), transfer(fm(i,1), 0),           &
                 transfer(fh(i,1), 0), transfer(lh(i,1), 0),             &
                 transfer(u10(i,1), 0), transfer(v10(i,1), 0),           &
                 transfer(th2(i,1), 0), transfer(t2(i,1), 0),            &
                 transfer(q2(i,1), 0), transfer(chs(i,1), 0),            &
                 transfer(chs2(i,1), 0), transfer(cqs2(i,1), 0),         &
                 transfer(flhc(i,1), 0), transfer(flqc(i,1), 0),         &
                 transfer(qgh(i,1), 0), transfer(rmol(i,1), 0),          &
                 transfer(wspd(i,1), 0), transfer(br(i,1), 0),           &
                 transfer(gz1oz0(i,1), 0), transfer(cpm(i,1), 0),        &
                 transfer(ck(i,1), 0), transfer(cka(i,1), 0),            &
                 transfer(cd(i,1), 0), transfer(cda(i,1), 0)
         end do
      end do
      deallocate(u3d, v3d, t3d, qv3d, p3d, dz8w, psfc, chs, chs2, cqs2, cpm, &
           znt, ust, pblh, mavail, zol, mol, regime, psim, psih, fm, fh,     &
           xland, hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, rmol, u10, v10,  &
           th2, t2, q2, gz1oz0, wspd, br, lakemask, dx, ustm, ck, cka, cd, cda)
   end do
   close(10)
   close(20)

contains

   real function word_to_real(w)
      integer(8), intent(in) :: w
      integer(4) :: s
      if (w >= 2147483648_8) then
         s = int(w - 4294967296_8, 4)
      else
         s = int(w, 4)
      end if
      word_to_real = transfer(s, 0.0)
   end function word_to_real

end program run_sfclay_classic
