! Driver for the WRF v4.6.1 updraft-helicity oracle.
!
! Builds a deterministic 14x12x30 (mass) fixture with terrain, map factors,
! a strong rotating updraft, a downdraft column (trips the use_column
! abort), and a marginal column whose w changes sign INSIDE the 2-5 km band
! (exercises cal_helicity's mid-column zero-then-reaccumulate quirk), then
! drives THREE steps of the WRF sequence exactly as
! module_first_rk_step_part2.F does under nwp_diagnostics=1:
!
!     compute_diff_metrics (fresh zx/zy/rdz/rdzw from this step's ph)
!     cal_helicity          (uh + running up_heli_max)
!
! Step amplitudes 0.6 / 1.0 / 0.75 make the running max hold step-2 values
! through step 3.  Tile bounds are the single-tile real-case geometry
! (specified BCs): its=ids..ite=ide-1, kts=kds..kte=kde, i.e. k_end = kpe
! as solve_em.F:299-300 passes it.
!
! Every input the two subroutines read is echoed to the inputs CSV with its
! exact FP32 bit pattern; the outputs CSV carries uh and up_heli_max after
! each step.  Nothing here invents an expected value: the physics numbers
! all come from the unmodified extracted subroutines.
!
! usage: run_cal_helicity UH_INPUTS_CSV UH_OUTPUTS_CSV
program run_cal_helicity
  use uh_oracle_mod
  implicit none

  integer, parameter :: ids = 1, ide = 15, jds = 1, jde = 13
  integer, parameter :: kds = 1, kde = 31
  integer, parameter :: ims = ids - 3, ime = ide + 3
  integer, parameter :: jms = jds - 3, jme = jde + 3
  integer, parameter :: kms = 1, kme = kde
  integer, parameter :: its = ids, ite = ide - 1
  integer, parameter :: jts = jds, jte = jde - 1
  integer, parameter :: kts = kds, kte = kde
  integer, parameter :: nsteps = 3

  type(grid_config_rec_type) :: config_flags

  real, dimension(ims:ime, kms:kme, jms:jme) :: u, v, w, ph, phb
  real, dimension(ims:ime, kms:kme, jms:jme) :: z, rdz, rdzw, zx, zy
  real, dimension(ims:ime, jms:jme) :: msfux, msfuy, msfvx, msfvy, ht
  real, dimension(ims:ime, jms:jme) :: uh, up_heli_max
  real, dimension(kms:kme) :: znw, dnw, dn, fnm, fnp
  real :: rdx, rdy, cf1, cf2, cf3, cof1, cof2
  real :: dx, dy, ztop, amp, zwv, frac, r2a, r2b, r2c, rot
  real, parameter :: amps(nsteps) = (/ 0.6, 1.0, 0.75 /)
  real, parameter :: xca = 5.5, yca = 6.5    ! rotating updraft
  real, parameter :: xcb = 10.5, ycb = 4.5   ! downdraft column
  real, parameter :: xcc = 3.5, ycc = 10.5   ! marginal (sign flip in band)
  integer :: i, j, k, s, iu_in, iu_out
  character(len=1024) :: inputs_csv, outputs_csv

  if (command_argument_count() /= 2) then
     write(*,'(A)') 'usage: run_cal_helicity UH_INPUTS_CSV UH_OUTPUTS_CSV'
     error stop 2
  end if
  call get_command_argument(1, inputs_csv)
  call get_command_argument(2, outputs_csv)

  config_flags%specified = .true.   ! real-case root; children use nested,
                                    ! which selects the same bound logic

  ! ---- vertical coordinate (WRF conventions; index 1 endpoint zero) -------
  do k = kds, kde
     znw(k) = (real(kde - k) / real(kde - 1))**1.2
  end do
  dnw = 0.
  dn = 0.
  fnm = 0.
  fnp = 0.
  do k = kds, kde - 1
     dnw(k) = znw(k+1) - znw(k)
  end do
  do k = kds + 1, kde - 1
     dn(k) = 0.5 * (dnw(k) + dnw(k-1))
     fnp(k) = 0.5 * dnw(k)   / dn(k)
     fnm(k) = 0.5 * dnw(k-1) / dn(k)
  end do
  ! Surface extrapolation weights, dyn_em/module_initialize_real.F:3747-3752.
  cof1 = (2.*dn(2) + dn(3)) / (dn(2) + dn(3)) * dnw(1) / dn(2)
  cof2 = dn(2) / (dn(2) + dn(3)) * dnw(1) / dn(3)
  cf1 = fnp(2) + cof1
  cf2 = fnm(2) - cof1 - cof2
  cf3 = cof2

  dx = 1000.
  dy = 1000.
  rdx = 1. / dx
  rdy = 1. / dy
  ztop = 18000.

  ! ---- static 2-D fields ---------------------------------------------------
  ht = 0.
  msfux = 1.
  msfuy = 1.
  msfvx = 1.
  msfvy = 1.
  do j = jds, jde - 1
     do i = ids, ide - 1
        ht(i,j) = 250. * (sin(0.7*real(i)) * cos(0.5*real(j)))**2
     end do
  end do
  do j = jds, jde - 1
     do i = ids, ide
        msfux(i,j) = 1.0 + 0.02 * sin(0.3*real(i) + 0.2*real(j))
        msfuy(i,j) = msfux(i,j)
     end do
  end do
  do j = jds, jde
     do i = ids, ide - 1
        msfvy(i,j) = 1.0 + 0.02 * cos(0.25*real(i) - 0.15*real(j))
        msfvx(i,j) = msfvy(i,j)
     end do
  end do

  ! ---- base geopotential (terrain-following to a flat top) ------------------
  phb = 0.
  do j = jds, jde - 1
     do k = kds, kde
        do i = ids, ide - 1
           phb(i,k,j) = g * (ht(i,j) + (1. - znw(k)) * (ztop - ht(i,j)))
        end do
     end do
  end do

  uh = 0.
  up_heli_max = 0.

  open(newunit=iu_in, file=trim(inputs_csv), form='formatted', &
       status='replace', action='write')
  write(iu_in,'(A)') 'field,step,k,j,i,value,bits'
  open(newunit=iu_out, file=trim(outputs_csv), form='formatted', &
       status='replace', action='write')
  write(iu_out,'(A)') 'field,step,k,j,i,value,bits'

  ! static echoes (step 0)
  do k = kds, kde
     call put(iu_in, 'znw', 0, k, 0, 0, znw(k))
  end do
  do k = kds, kde - 1
     call put(iu_in, 'dnw', 0, k, 0, 0, dnw(k))
     call put(iu_in, 'dn',  0, k, 0, 0, dn(k))
     call put(iu_in, 'fnm', 0, k, 0, 0, fnm(k))
     call put(iu_in, 'fnp', 0, k, 0, 0, fnp(k))
  end do
  call put(iu_in, 'cf1', 0, 0, 0, 0, cf1)
  call put(iu_in, 'cf2', 0, 0, 0, 0, cf2)
  call put(iu_in, 'cf3', 0, 0, 0, 0, cf3)
  call put(iu_in, 'rdx', 0, 0, 0, 0, rdx)
  call put(iu_in, 'rdy', 0, 0, 0, 0, rdy)
  do j = jds, jde - 1
     do i = ids, ide - 1
        call put(iu_in, 'ht', 0, 0, j, i, ht(i,j))
     end do
  end do
  do j = jds, jde - 1
     do i = ids, ide
        call put(iu_in, 'msfux', 0, 0, j, i, msfux(i,j))
     end do
  end do
  do j = jds, jde
     do i = ids, ide - 1
        call put(iu_in, 'msfvy', 0, 0, j, i, msfvy(i,j))
     end do
  end do
  do j = jds, jde - 1
     do k = kds, kde
        do i = ids, ide - 1
           call put(iu_in, 'phb', 0, k, j, i, phb(i,k,j))
        end do
     end do
  end do

  do s = 1, nsteps
     amp = amps(s)
     u = 0.
     v = 0.
     w = 0.
     ph = 0.

     ! u at u-points, mass levels: westerly shear + mesocyclone rotation.
     do j = jds, jde - 1
        do k = kds, kde - 1
           frac = 1. - 0.5 * (znw(k) + znw(k+1))
           do i = ids, ide
              r2a = (real(i) - 0.5 - xca)**2 + (real(j) - yca)**2
              rot = amp * 18. * exp(-r2a / 8.)
              u(i,k,j) = 12. * frac &
                   - rot * (real(j) - yca) / sqrt(r2a + 0.25)
           end do
        end do
     end do
     ! v at v-points, mass levels: southerly shear + rotation.
     do j = jds, jde
        do k = kds, kde - 1
           frac = 1. - 0.5 * (znw(k) + znw(k+1))
           do i = ids, ide - 1
              r2a = (real(i) - xca)**2 + (real(j) - 0.5 - yca)**2
              rot = amp * 18. * exp(-r2a / 8.)
              v(i,k,j) = 4. + 8. * frac &
                   + rot * (real(i) - xca) / sqrt(r2a + 0.25)
           end do
        end do
     end do
     ! w at w-levels: updraft A, downdraft B, sign-flipping column C.
     do j = jds, jde - 1
        do k = kds, kde
           do i = ids, ide - 1
              zwv = ht(i,j) + (1. - znw(k)) * (ztop - ht(i,j))
              r2a = (real(i) - xca)**2 + (real(j) - yca)**2
              r2b = (real(i) - xcb)**2 + (real(j) - ycb)**2
              r2c = (real(i) - xcc)**2 + (real(j) - ycc)**2
              w(i,k,j) = amp * 38. * exp(-r2a / 6.) &
                            * exp(-((zwv - 4000.) / 2500.)**2) &
                       - amp * 12. * exp(-r2b / 4.) &
                            * exp(-((zwv - 3500.) / 2500.)**2) &
                       + amp * 8. * exp(-r2c / 3.) &
                            * exp(-((zwv - 3000.) / 3000.)**2) &
                            * ((3500. - zwv) / 3500.)
           end do
        end do
     end do
     ! ph perturbation, growing with height.
     do j = jds, jde - 1
        do k = kds, kde
           do i = ids, ide - 1
              ph(i,k,j) = amp * g * 25. * sin(0.4*real(i)) &
                          * cos(0.3*real(j)) * (1. - znw(k))
           end do
        end do
     end do

     do j = jds, jde - 1
        do k = kds, kde - 1
           do i = ids, ide
              call put(iu_in, 'u', s, k, j, i, u(i,k,j))
           end do
        end do
     end do
     do j = jds, jde
        do k = kds, kde - 1
           do i = ids, ide - 1
              call put(iu_in, 'v', s, k, j, i, v(i,k,j))
           end do
        end do
     end do
     do j = jds, jde - 1
        do k = kds, kde
           do i = ids, ide - 1
              call put(iu_in, 'w', s, k, j, i, w(i,k,j))
              call put(iu_in, 'ph', s, k, j, i, ph(i,k,j))
           end do
        end do
     end do

     ! WRF step sequence: metrics fresh from this step's ph (diff_opt 1/2
     ! path, module_first_rk_step_part2.F:419-434), then helicity.
     z = 0.
     rdz = 0.
     rdzw = 0.
     zx = 0.
     zy = 0.
     call compute_diff_metrics(config_flags, ph, phb, z, rdz, rdzw, &
                               zx, zy, rdx, rdy, &
                               ids, ide, jds, jde, kds, kde, &
                               ims, ime, jms, jme, kms, kme, &
                               its, ite, jts, jte, kts, kte)
     call cal_helicity(config_flags, u, v, w, uh, up_heli_max, &
                       ph, phb, msfux, msfuy, msfvx, msfvy, ht, &
                       rdx, rdy, dn, dnw, rdz, rdzw, &
                       fnm, fnp, cf1, cf2, cf3, zx, zy, &
                       ids, ide, jds, jde, kds, kde, &
                       ims, ime, jms, jme, kms, kme, &
                       its, ite, jts, jte, kts, kte)

     do j = jds, jde - 1
        do i = ids, ide - 1
           call put(iu_out, 'uh', s, 0, j, i, uh(i,j))
           call put(iu_out, 'up_heli_max', s, 0, j, i, up_heli_max(i,j))
        end do
     end do
  end do

  close(iu_in)
  close(iu_out)
  write(*,'(A)') 'run_cal_helicity: wrote inputs + outputs for 3 steps'

contains

  subroutine put(unit, name, s, k, j, i, x)
    integer, intent(in) :: unit, s, k, j, i
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    integer :: bits
    bits = transfer(x, bits)
    write(unit,'(A,",",I0,",",I0,",",I0,",",I0,",",ES17.9E3,",",I0)') &
         trim(name), s, k, j, i, x, bits
  end subroutine put

end program run_cal_helicity
