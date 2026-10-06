! Oracle harness for NOAA's radar latent heating builder (HRRR v4.1.21,
! sorc/hrrr_ref2tten.fd).  It reads raw float32 arrays, calls NOAA's own
! calc_pbl_height, build_missing_REFcone and radar_ref2tten, unchanged, in
! the order and with the constants of gsdcloudanalysis_ref2tten.f90
! (:155-162, :244, :347-348, :386-393), and writes what they return.
!
! Arrays are (nlon, nlat, nsig) in Fortran order, which is the same bytes
! as a C-order (nz, ny, nx) array.
!
! Usage: oracle_ref2tten <dir> <nlon> <nlat> <nsig> [convection_only 0|1]
!   reads  <dir>/ref.bin t.bin q.bin p.bin h.bin
!   writes <dir>/pblh.bin refcone.bin tten.bin
program oracle_ref2tten
  use kinds, only: r_single, r_kind, i_kind
  use constants, only: init_constants, init_constants_derived
  implicit none
  integer(i_kind) :: nlon, nlat, nsig, conv_flag, nargs
  character(len=1024) :: dir, arg
  real(r_single), allocatable :: ref_mos_3d(:,:,:), t_bk(:,:,:), q_bk(:,:,:)
  real(r_single), allocatable :: p_bk(:,:,:), h_bk(:,:,:), pblh(:,:)
  real(r_single), allocatable :: ges_tten(:,:,:)
  real(r_single) :: krad_bot
  real(r_single) :: dfi_radar_latent_heat_time_period
  real(r_kind) :: dfi_lhtp
  real(r_kind) :: convection_refl_threshold
  logical :: l_tten_for_convection_only

  nargs = command_argument_count()
  if (nargs < 4) stop 'usage: oracle_ref2tten dir nlon nlat nsig [conv 0|1]'
  call get_command_argument(1, dir)
  call get_command_argument(2, arg); read(arg, *) nlon
  call get_command_argument(3, arg); read(arg, *) nlat
  call get_command_argument(4, arg); read(arg, *) nsig
  conv_flag = 1
  if (nargs >= 5) then
     call get_command_argument(5, arg); read(arg, *) conv_flag
  endif

  ! gsdcloudanalysis_ref2tten.f90:155-162
  call init_constants(.true.)
  call init_constants_derived
  krad_bot = 7.0_r_single
  dfi_radar_latent_heat_time_period = 20.0_r_single
  convection_refl_threshold = 28.0_r_kind
  l_tten_for_convection_only = (conv_flag /= 0)

  allocate(ref_mos_3d(nlon,nlat,nsig), t_bk(nlon,nlat,nsig))
  allocate(q_bk(nlon,nlat,nsig), p_bk(nlon,nlat,nsig), h_bk(nlon,nlat,nsig))
  allocate(pblh(nlon,nlat), ges_tten(nlon,nlat,nsig))
  call slurp(trim(dir)//'/ref.bin', ref_mos_3d, nlon*nlat*nsig)
  call slurp(trim(dir)//'/t.bin', t_bk, nlon*nlat*nsig)
  call slurp(trim(dir)//'/q.bin', q_bk, nlon*nlat*nsig)
  call slurp(trim(dir)//'/p.bin', p_bk, nlon*nlat*nsig)
  call slurp(trim(dir)//'/h.bin', h_bk, nlon*nlat*nsig)

  ! :244 -- the driver passes (nlon, nlat) to a routine declared
  ! (nlat, nlon); kept exactly as NOAA calls it.
  call calc_pbl_height(nlon,nlat,nsig,q_bk,t_bk,p_bk,pblh)
  call dump(trim(dir)//'/pblh.bin', pblh, nlon*nlat)

  ! :347-348
  call build_missing_REFcone(nlon,nlat,nsig,krad_bot,ref_mos_3d,h_bk,pblh)
  call dump(trim(dir)//'/refcone.bin', ref_mos_3d, nlon*nlat*nsig)

  ! :386-393
  ges_tten = -20.0_r_kind
  ges_tten(:,:,nsig) = -10.0_r_kind
  dfi_lhtp = dfi_radar_latent_heat_time_period
  call radar_ref2tten(nlon,nlat,nsig,ref_mos_3d,p_bk,t_bk,ges_tten,dfi_lhtp, &
                      krad_bot,pblh,l_tten_for_convection_only, &
                      convection_refl_threshold)
  call dump(trim(dir)//'/tten.bin', ges_tten, nlon*nlat*nsig)

contains

  subroutine slurp(path, a, n)
    character(len=*), intent(in) :: path
    integer(i_kind), intent(in) :: n
    real(r_single), intent(out) :: a(n)
    integer :: u, ios
    open(newunit=u, file=path, access='stream', form='unformatted', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(0,*) 'cannot open ', path
       stop 2
    endif
    read(u) a
    close(u)
  end subroutine slurp

  subroutine dump(path, a, n)
    character(len=*), intent(in) :: path
    integer(i_kind), intent(in) :: n
    real(r_single), intent(in) :: a(n)
    integer :: u
    open(newunit=u, file=path, access='stream', form='unformatted', &
         status='replace', action='write')
    write(u) a
    close(u)
  end subroutine dump

end program oracle_ref2tten
