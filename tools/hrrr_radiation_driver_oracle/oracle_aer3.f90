! aer_opt = 3 oracle: the fork's gt_aod and calc_aerosol_rrtmg_sw, driven
! the way module_radiation_driver.F drives them on a radiation step
! (:950-1000 then :1883-1896): taod5502d zeroed, gt_aod, the column sum in
! k order, then calc_aerosol_rrtmg_sw with the driver's PARAMETERs
! taer_type = 1, taer_aod550_opt = 2, taer_angexp_opt = taer_ssa_opt =
! taer_asy_opt = 3 (:769-773) and the Registry defaults of the four
! aer_*_val namelist values (unread on these options).
!
! aer3_in.bin (stream, little-endian):
!   int32 ncol, nz
!   real32 p, t, qv, dz8w, nwfa, nifa: each (ncol, nz) in Fortran order
!   real32 ht(ncol)
! aer3_out.bin:
!   real32 tauaer, ssaaer, asyaer: each (ncol, nz, 14) in Fortran order
!   real32 taod5503d (ncol, nz), taod5502d (ncol)
program oracle_aer3
  use hrrr_radiation_driver_oracle
  use module_ra_aerosol
  implicit none
  integer, parameter :: nb = 14
  integer :: ncol, nz, kme, i, k
  real, allocatable :: tmp(:,:)
  real, allocatable :: p(:,:,:), t(:,:,:), qv(:,:,:), dz(:,:,:), nw(:,:,:), ni(:,:,:)
  real, allocatable :: taod3(:,:,:), ht(:,:), aod2(:,:), ang2(:,:), ssa2(:,:), asy2(:,:)
  real, allocatable :: tau(:,:,:,:), ssa(:,:,:,:), asy(:,:,:,:), out(:,:,:)

  open(10, file='aer3_in.bin', access='stream', form='unformatted', status='old')
  read(10) ncol, nz
  kme = nz + 1
  allocate(tmp(ncol, nz))
  allocate(p(ncol,kme,1), t(ncol,kme,1), qv(ncol,kme,1), dz(ncol,kme,1), &
           nw(ncol,kme,1), ni(ncol,kme,1), taod3(ncol,kme,1))
  allocate(ht(ncol,1), aod2(ncol,1), ang2(ncol,1), ssa2(ncol,1), asy2(ncol,1))
  allocate(tau(ncol,kme,1,nb), ssa(ncol,kme,1,nb), asy(ncol,kme,1,nb), out(ncol,nz,nb))
  ! the level above the last mass level is never read (kte = nz); keep it
  ! physical so nothing undefined reaches a debug trap
  p = 50000.; t = 250.; qv = 0.; dz = 100.; nw = 1.e9; ni = 1.e6; taod3 = 0.
  read(10) tmp; p(:,1:nz,1) = tmp
  read(10) tmp; t(:,1:nz,1) = tmp
  read(10) tmp; qv(:,1:nz,1) = tmp
  read(10) tmp; dz(:,1:nz,1) = tmp
  read(10) tmp; nw(:,1:nz,1) = tmp
  read(10) tmp; ni(:,1:nz,1) = tmp
  read(10) ht(:,1)
  close(10)

  do i = 1, ncol
     aod2(i,1) = 0.0
  end do
  call gt_aod(p, dz, t, qv, nw, ni, taod3,                    &
              1,ncol, 1,1, 1,kme, 1,ncol, 1,1, 1,nz)
  do i = 1, ncol
     do k = 1, nz
        aod2(i,1) = aod2(i,1) + taod3(i,k,1)
     end do
  end do
  tau = 0.; ssa = 0.; asy = 0.
  ang2 = 0.; ssa2 = 0.; asy2 = 0.
  call calc_aerosol_rrtmg_sw(ht, dz, p, t, qv, 1, 2, 3, 3, 3,       &
                             0.12, 1.3, 0.85, 0.9,                  &
                             aod2, ang2, ssa2, asy2,                &
                             1,ncol, 1,1, 1,kme, 1,ncol, 1,1, 1,nz, &
                             tau, ssa, asy, taod3)

  open(11, file='aer3_out.bin', access='stream', form='unformatted', status='replace')
  out = tau(:,1:nz,1,:); write(11) out
  out = ssa(:,1:nz,1,:); write(11) out
  out = asy(:,1:nz,1,:); write(11) out
  tmp = taod3(:,1:nz,1); write(11) tmp
  write(11) aod2(:,1)
  close(11)
end program oracle_aer3
