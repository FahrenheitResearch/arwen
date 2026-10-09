! Column driver for WRF v4.6.1's revised MM5 surface layer
! (sf_sfclay_physics = 1), compiled against the byte-unmodified
! phys/module_sf_sfclayrev.F and phys/physics_mmm/sf_sfclayrev.F90.
!
! It calls exactly what WRF's surface driver calls: sf_sfclayrev_init once
! (module_physics_init's sfclayrevinit), then the WRF wrapper SFCLAYREV
! (module_surface_driver.F:2116) with one j row, one k level, and the same
! model constants the driver passes (cp, g, rcp, r_d, xlv, svp1..svpt0,
! ep_1, ep_2, karman, p1000mb from share/module_model_constants.F, folded
! here by the same compiler in the same REAL kind).
!
! Input: one line per column, a case id then 21 float32 words as 8-digit hex
!   u v t qv p dz8w psfc tsk pblh mavail xland lakemask dx
!   znt ust ustm mol hfx qfx qsfc zol
! Output: per arm, per step, per column: arm id, the three switches, step,
! case id, then the 34 SFCLAY_OUTPUTS words as hex, in
! gpuwm.core.physics_inventory.SFCLAY_OUTPUTS order.
!
! Steps chain: step s+1 starts from every inout array step s left behind
! (WRF's own carry), with the atmosphere held fixed.  Every arm restarts
! from the input file's state.

program run_sfclayrev
  use ccpp_kind_types, only: kind_phys
  use sf_sfclayrev, only: sf_sfclayrev_init
  use module_sf_sfclayrev, only: sfclayrev
  implicit none

  ! share/module_model_constants.F, spelled identically.
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.
  real, parameter :: cp = 7.*r_d/2.
  real, parameter :: r_v = 461.6
  real, parameter :: rcp = r_d/cp
  real, parameter :: p1000mb = 100000.
  real, parameter :: xlv = 2.5E6
  real, parameter :: svp1 = 0.6112
  real, parameter :: svp2 = 17.67
  real, parameter :: svp3 = 29.65
  real, parameter :: svpt0 = 273.15
  real, parameter :: ep_1 = R_v/R_d-1.
  real, parameter :: ep_2 = R_d/R_v
  real, parameter :: karman = 0.4

  integer, parameter :: nin = 21, nout = 34, nsteps = 3, narms = 6
  ! arm: isfflx, isftcflx, iz0tlnd
  integer, parameter :: arms(3, narms) = reshape( &
       (/ 1, 0, 0,   1, 1, 0,   1, 2, 0,   1, 0, 1,   1, 0, 2,   0, 0, 0 /), &
       (/ 3, narms /))

  character(len=512) :: in_path, out_path, line
  character(len=32) :: seed_option
  integer :: n, i, ios, arm, step, unit_in, unit_out, errflg
  character(len=256) :: errmsg
  integer, allocatable :: case_id(:)
  integer :: words(nin)
  real(kind_phys), allocatable :: inp(:, :)

  real(kind_phys), allocatable, dimension(:, :, :) :: u3d, v3d, t3d, qv3d, p3d, dz8w
  real(kind_phys), allocatable, dimension(:, :) :: dx, mavail, pblh, psfc, tsk, xland, &
       lakemask, water_depth, lh, u10, v10, th2, t2, q2, ck, cka, cd, cda, &
       regime, hfx, qfx, qsfc, mol, rmol, gz1oz0, wspd, br, psim, psih, fm, fh, &
       znt, zol, ust, cpm, chs2, cqs2, chs, flhc, flqc, qgh, ustm
  real(kind_phys), allocatable :: outv(:, :)

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  call get_command_argument(3, seed_option)
  if (len_trim(seed_option) /= 0 .and. trim(seed_option) /= '--seed-exchange') &
       stop 'third argument must be --seed-exchange'

  open(newunit=unit_in, file=trim(in_path), status='old', action='read')
  n = 0
  do
     read(unit_in, '(A)', iostat=ios) line
     if (ios /= 0) exit
     if (line(1:1) == '#' .or. len_trim(line) == 0) cycle
     n = n + 1
  end do
  rewind(unit_in)
  allocate(case_id(n), inp(nin, n))
  i = 0
  do
     read(unit_in, '(A)', iostat=ios) line
     if (ios /= 0) exit
     if (line(1:1) == '#' .or. len_trim(line) == 0) cycle
     i = i + 1
     read(line, *) case_id(i)
     read(line(index(line, ' ') + 1:), '(21(Z8,1X))') words
     inp(:, i) = transfer(words, inp(:, i))
  end do
  close(unit_in)

  allocate(u3d(n,1,1), v3d(n,1,1), t3d(n,1,1), qv3d(n,1,1), p3d(n,1,1), dz8w(n,1,1))
  allocate(dx(n,1), mavail(n,1), pblh(n,1), psfc(n,1), tsk(n,1), xland(n,1), &
       lakemask(n,1), water_depth(n,1), lh(n,1), u10(n,1), v10(n,1), th2(n,1), &
       t2(n,1), q2(n,1), ck(n,1), cka(n,1), cd(n,1), cda(n,1), regime(n,1), &
       hfx(n,1), qfx(n,1), qsfc(n,1), mol(n,1), rmol(n,1), gz1oz0(n,1), &
       wspd(n,1), br(n,1), psim(n,1), psih(n,1), fm(n,1), fh(n,1), znt(n,1), &
       zol(n,1), ust(n,1), cpm(n,1), chs2(n,1), cqs2(n,1), chs(n,1), &
       flhc(n,1), flqc(n,1), qgh(n,1), ustm(n,1))
  allocate(outv(nout, n))

  call sf_sfclayrev_init(errmsg, errflg)
  if (errflg /= 0) stop 'sf_sfclayrev_init failed'

  open(newunit=unit_out, file=trim(out_path), status='replace', action='write')
  write(unit_out, '(A)') '# arm isfflx isftcflx iz0tlnd step case znt ust ustm mol hfx qfx ' // &
       'qsfc zol regime psim psih fm fh lh u10 v10 th2 t2 q2 chs chs2 cqs2 flhc ' // &
       'flqc qgh rmol wspd br gz1oz0 cpm ck cka cd cda'

  do arm = 1, narms
     u3d(:,1,1) = inp(1,:);  v3d(:,1,1) = inp(2,:);  t3d(:,1,1) = inp(3,:)
     qv3d(:,1,1) = inp(4,:); p3d(:,1,1) = inp(5,:);  dz8w(:,1,1) = inp(6,:)
     psfc(:,1) = inp(7,:);   tsk(:,1) = inp(8,:);    pblh(:,1) = inp(9,:)
     mavail(:,1) = inp(10,:); xland(:,1) = inp(11,:); lakemask(:,1) = inp(12,:)
     dx(:,1) = inp(13,:)
     znt(:,1) = inp(14,:);  ust(:,1) = inp(15,:);   ustm(:,1) = inp(16,:)
     mol(:,1) = inp(17,:);  hfx(:,1) = inp(18,:);   qfx(:,1) = inp(19,:)
     qsfc(:,1) = inp(20,:); zol(:,1) = inp(21,:)
     water_depth(:,1) = 50.
     ! Inout arrays the scheme overwrites on every isfflx=1 call start from
     ! zero, a cold start's registry value.  With isfflx=0 WRF leaves CHS,
     ! CHS2 and CQS2 untouched, so zero is also what they hold there.
     regime = 0.; rmol = 0.; gz1oz0 = 0.; wspd = 0.; br = 0.; psim = 0.
     psih = 0.; fm = 0.; fh = 0.; cpm = 0.; chs2 = 0.; cqs2 = 0.; chs = 0.
     flhc = 0.; flqc = 0.; qgh = 0.
     ! Nonzero carry exposes the flux-off preservation defect; the default
     ! keeps the original 66,990-word cold-start oracle reproducible.
     if (trim(seed_option) == '--seed-exchange') then
        chs = 0.125; chs2 = 0.25; cqs2 = 0.375
        flhc = 0.5; flqc = 0.001
     endif
     ! LH is intent(out) and SFCLAYREV copies an unassigned local into it
     ! when isfflx=0; seed it so that case writes a known word.
     lh = 0.; u10 = 0.; v10 = 0.; th2 = 0.; t2 = 0.; q2 = 0.
     ck = 0.; cka = 0.; cd = 0.; cda = 0.
     do step = 1, nsteps
        call sfclayrev(u3d, v3d, t3d, qv3d, p3d, dz8w,                    &
             cp, g, rcp, r_d, xlv, psfc, chs, chs2, cqs2, cpm,            &
             znt, ust, pblh, mavail, zol, mol, regime, psim, psih,        &
             fm, fh,                                                      &
             xland, hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, rmol,       &
             u10, v10, th2, t2, q2,                                       &
             gz1oz0, wspd, br, arms(1, arm), dx,                          &
             svp1, svp2, svp3, svpt0, ep_1, ep_2, karman,                 &
             p1000mb, lakemask,                                           &
             1, n, 1, 1, 1, 1,                                            &
             1, n, 1, 1, 1, 1,                                            &
             1, n, 1, 1, 1, 1,                                            &
             ustm, ck, cka, cd, cda, arms(2, arm), arms(3, arm),          &
             0, water_depth,                                              &
             0, errmsg, errflg)
        if (errflg /= 0) stop 'sfclayrev failed'
        outv(1,:) = znt(:,1);   outv(2,:) = ust(:,1);    outv(3,:) = ustm(:,1)
        outv(4,:) = mol(:,1);   outv(5,:) = hfx(:,1);    outv(6,:) = qfx(:,1)
        outv(7,:) = qsfc(:,1);  outv(8,:) = zol(:,1);    outv(9,:) = regime(:,1)
        outv(10,:) = psim(:,1); outv(11,:) = psih(:,1);  outv(12,:) = fm(:,1)
        outv(13,:) = fh(:,1);   outv(14,:) = lh(:,1);    outv(15,:) = u10(:,1)
        outv(16,:) = v10(:,1);  outv(17,:) = th2(:,1);   outv(18,:) = t2(:,1)
        outv(19,:) = q2(:,1);   outv(20,:) = chs(:,1);   outv(21,:) = chs2(:,1)
        outv(22,:) = cqs2(:,1); outv(23,:) = flhc(:,1);  outv(24,:) = flqc(:,1)
        outv(25,:) = qgh(:,1);  outv(26,:) = rmol(:,1);  outv(27,:) = wspd(:,1)
        outv(28,:) = br(:,1);   outv(29,:) = gz1oz0(:,1); outv(30,:) = cpm(:,1)
        outv(31,:) = ck(:,1);   outv(32,:) = cka(:,1);   outv(33,:) = cd(:,1)
        outv(34,:) = cda(:,1)
        do i = 1, n
           write(unit_out, '(I1,3(1X,I1),1X,I1,1X,I4,34(1X,Z8.8))') arm, arms(:, arm), &
                step, case_id(i), transfer(outv(:, i), words(1:1), nout)
        end do
     end do
  end do
  close(unit_out)
end program run_sfclayrev
