! Harness for the final precipitation analysis of NOAA's GSD cloud analysis
! (NOAA-EMC/HRRR tag v4.1.21, sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90).
!
! This file is ours.  NOAA's block reads GSI module arrays, so it cannot be
! compiled alone.  build.sh cuts lines 872-1049 (the three precipitation
! modes) and lines 1053-1066 (the clamp) out of NOAA's file by line number
! into two include files, and noaa_precip_block below includes them verbatim.
! The ONLY change is the array bindings: ges_qr(j,i,k), ges_qnr, ges_qs and
! ges_qg, which GSI reaches through bundle pointers, are dummy arguments
! here, and the namelist and module values the lines read
! (l_use_hydroretrieval_all, l_precip_clear_only,
! r_cleanSnow_WarmTs_threshold, i_lightpcp, iclean_hydro_withRef,
! iclean_hydro_withRef_allcol, t_bk, p_bk, ref_mos_3d, sat_ctp) are dummy
! arguments of the kinds NOAA declares (gsdcloudanalysis.F90:92-94, 137,
! 144, 173-181, 222-229, 233-250; rapidrefresh_cldsurf_mod.f90:218-248).
! The four cloud arrays the clamp also touches are local arrays of zeros.
!
! The driver does what gsdcloudanalysis.F90 does around the block: outputs
! start at miss_obs_real (:611-616), then PrecipMxR_radar (NOAA's file,
! unchanged) runs the Thompson retrieval (:778-780, opt_hydrometeor_retri =
! 3, :295), then the block.  constants.f90 and kinds.f90 are NOAA's,
! unchanged, with the regional constants GSI sets for a WRF run.
!
! It also writes the temperature and pressure the retrieval saw, evaluated
! with the same two expressions as PrecipMxr_radar.f90:106-107, so the
! device kernel can be fed bit-identical inputs.
!
! Input (stream, native endian): lon2, lat2, nsig (int32); theta (real32 K),
! p (real32 hPa), each (lon2, lat2, nsig); ref (real64 dBZ); sat_ctp (real32
! hPa, (lon2, lat2)); ges_qr, ges_qnr, ges_qs, ges_qg (real64, (lon2, lat2,
! nsig); the driver transposes them to NOAA's (lat2, lon2, nsig)).
! Arguments: input, output, mode (trim-build | clear | all), i_lightpcp,
! iclean_hydro_withRef, iclean_hydro_withRef_allcol,
! r_cleanSnow_WarmTs_threshold.
subroutine noaa_precip_block(lon2, lat2, nsig, l_use_hydroretrieval_all, &
     l_precip_clear_only, r_cleanSnow_WarmTs_threshold, i_lightpcp, &
     iclean_hydro_withRef, iclean_hydro_withRef_allcol, &
     ges_qr, ges_qnr, ges_qs, ges_qg, ref_mos_3d, t_bk, p_bk, sat_ctp, &
     rain_3d, nrain_3d, snow_3d, graupel_3d)
  use kinds, only: r_single, i_kind, r_kind
  use constants, only: zero, rd_over_cp, h1000
  implicit none
  integer(i_kind), intent(in) :: lon2, lat2, nsig
  logical, intent(in) :: l_use_hydroretrieval_all, l_precip_clear_only
  real(r_kind), intent(in) :: r_cleanSnow_WarmTs_threshold
  integer(i_kind), intent(in) :: i_lightpcp
  integer(i_kind), intent(in) :: iclean_hydro_withRef
  integer(i_kind), intent(in) :: iclean_hydro_withRef_allcol
  real(r_kind), intent(in) :: ges_qr(lat2,lon2,nsig)
  real(r_kind), intent(in) :: ges_qnr(lat2,lon2,nsig)
  real(r_kind), intent(in) :: ges_qs(lat2,lon2,nsig)
  real(r_kind), intent(in) :: ges_qg(lat2,lon2,nsig)
  real(r_kind), intent(in) :: ref_mos_3d(lon2,lat2,nsig)
  real(r_single), intent(in) :: t_bk(lon2,lat2,nsig)
  real(r_single), intent(in) :: p_bk(lon2,lat2,nsig)
  real(r_single), intent(in) :: sat_ctp(lon2,lat2)
  real(r_single), intent(inout) :: rain_3d(lon2,lat2,nsig)
  real(r_single), intent(inout) :: nrain_3d(lon2,lat2,nsig)
  real(r_single), intent(inout) :: snow_3d(lon2,lat2,nsig)
  real(r_single), intent(inout) :: graupel_3d(lon2,lat2,nsig)

  real(r_single), allocatable :: cldwater_3d(:,:,:), cldice_3d(:,:,:)
  real(r_single), allocatable :: nice_3d(:,:,:), nwater_3d(:,:,:)
  real(r_single), allocatable :: rain_1d_save(:), nrain_1d_save(:)
  real(r_single), allocatable :: snow_1d_save(:)
  real(r_kind), parameter :: miss_obs_real = -99999.0_r_kind
  integer(i_kind) :: i, j, k
  integer(i_kind) :: imaxlvl_ref
  real(r_kind) :: max_retrieved_qrqs, max_bk_qrqs, ratio_hyd_bk2obs
  real(r_kind) :: qrlimit, qrlimit_lightpcp
  real(r_kind) :: refmax, snowtemp, raintemp, nraintemp, graupeltemp
  real(r_kind) :: snowadd, ratio2
  real(r_kind) :: tsfc

  allocate(cldwater_3d(lon2,lat2,nsig), cldice_3d(lon2,lat2,nsig))
  allocate(nice_3d(lon2,lat2,nsig), nwater_3d(lon2,lat2,nsig))
  cldwater_3d = 0.0_r_single
  cldice_3d = 0.0_r_single
  nice_3d = 0.0_r_single
  nwater_3d = 0.0_r_single
  allocate(rain_1d_save(nsig), nrain_1d_save(nsig), snow_1d_save(nsig))
  rain_1d_save = miss_obs_real
  nrain_1d_save = miss_obs_real
  snow_1d_save = miss_obs_real

  include 'noaa_gsdcloudanalysis_0872_1049.inc'
  include 'noaa_gsdcloudanalysis_1053_1066.inc'
end subroutine noaa_precip_block

! Never reached with opt_hydrometeor_retri = 3; PrecipMxr_radar.f90 names
! them in its Kessler and Ferrier arms, so the link needs the symbols.
subroutine pcp_mxr()
  stop 'oracle_b: pcp_mxr (Kessler) is not part of this oracle'
end subroutine pcp_mxr

subroutine pcp_mxr_ferrier()
  stop 'oracle_b: pcp_mxr_ferrier is not part of this oracle'
end subroutine pcp_mxr_ferrier

program oracle_b
  use kinds, only: r_single, i_kind, r_kind
  use constants, only: init_constants, init_constants_derived, rd_over_cp, h1000
  implicit none
  integer(i_kind) :: lon2, lat2, nsig, u, i, j, k
  integer(i_kind) :: i_lightpcp, iclean_hydro_withRef
  integer(i_kind) :: iclean_hydro_withRef_allcol, opt_hydrometeor_retri
  logical :: l_use_hydroretrieval_all, l_precip_clear_only
  real(r_kind) :: r_cleanSnow_WarmTs_threshold
  real(r_single), allocatable :: t_bk(:,:,:), p_bk(:,:,:), sat_ctp(:,:)
  real(r_single), allocatable :: t_3d(:,:,:), p_3d(:,:,:)
  real(r_kind), allocatable :: ref_mos_3d(:,:,:), buffer(:,:,:)
  real(r_kind), allocatable :: ges_qr(:,:,:), ges_qnr(:,:,:)
  real(r_kind), allocatable :: ges_qs(:,:,:), ges_qg(:,:,:)
  real(r_single), allocatable :: rain_3d(:,:,:), nrain_3d(:,:,:)
  real(r_single), allocatable :: snow_3d(:,:,:), graupel_3d(:,:,:)
  integer(i_kind), allocatable :: pcp_type_3d(:,:,:)
  real(r_kind), parameter :: miss_obs_real = -99999.0_r_kind
  integer(i_kind), parameter :: miss_obs_int = -99999
  character(len=1024) :: fin, fout, text

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, text)
  l_use_hydroretrieval_all = (trim(text) == 'all')
  l_precip_clear_only = (trim(text) == 'clear')
  if (.not. (l_use_hydroretrieval_all .or. l_precip_clear_only &
             .or. trim(text) == 'trim-build')) stop 'oracle_b: mode is trim-build, clear or all'
  call get_command_argument(4, text)
  read(text, *) i_lightpcp
  call get_command_argument(5, text)
  read(text, *) iclean_hydro_withRef
  call get_command_argument(6, text)
  read(text, *) iclean_hydro_withRef_allcol
  call get_command_argument(7, text)
  read(text, *) r_cleanSnow_WarmTs_threshold

  ! GSI's own sequence for a regional (WRF mass core) run.
  call init_constants_derived
  call init_constants(.true.)

  open(newunit=u, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) lon2, lat2, nsig
  allocate(t_bk(lon2,lat2,nsig), p_bk(lon2,lat2,nsig), sat_ctp(lon2,lat2))
  allocate(ref_mos_3d(lon2,lat2,nsig), buffer(lon2,lat2,nsig))
  allocate(ges_qr(lat2,lon2,nsig), ges_qnr(lat2,lon2,nsig))
  allocate(ges_qs(lat2,lon2,nsig), ges_qg(lat2,lon2,nsig))
  read(u) t_bk
  read(u) p_bk
  read(u) ref_mos_3d
  read(u) sat_ctp
  read(u) buffer
  do k = 1, nsig; do j = 1, lat2; do i = 1, lon2
    ges_qr(j,i,k) = buffer(i,j,k)
  end do; end do; end do
  read(u) buffer
  do k = 1, nsig; do j = 1, lat2; do i = 1, lon2
    ges_qnr(j,i,k) = buffer(i,j,k)
  end do; end do; end do
  read(u) buffer
  do k = 1, nsig; do j = 1, lat2; do i = 1, lon2
    ges_qs(j,i,k) = buffer(i,j,k)
  end do; end do; end do
  read(u) buffer
  do k = 1, nsig; do j = 1, lat2; do i = 1, lon2
    ges_qg(j,i,k) = buffer(i,j,k)
  end do; end do; end do
  close(u)

  allocate(rain_3d(lon2,lat2,nsig), nrain_3d(lon2,lat2,nsig))
  allocate(snow_3d(lon2,lat2,nsig), graupel_3d(lon2,lat2,nsig))
  allocate(pcp_type_3d(lon2,lat2,nsig))
  allocate(t_3d(lon2,lat2,nsig), p_3d(lon2,lat2,nsig))
  ! gsdcloudanalysis.F90:613-616, :693
  rain_3d = miss_obs_real
  nrain_3d = miss_obs_real
  snow_3d = miss_obs_real
  graupel_3d = miss_obs_real
  pcp_type_3d = miss_obs_int

  ! gsdcloudanalysis.F90:295, :778-780
  opt_hydrometeor_retri = 3
  call PrecipMxR_radar(0, lat2, lon2, nsig, &
       t_bk, p_bk, ref_mos_3d, &
       pcp_type_3d, rain_3d, nrain_3d, snow_3d, graupel_3d, opt_hydrometeor_retri)

  call noaa_precip_block(lon2, lat2, nsig, l_use_hydroretrieval_all, &
       l_precip_clear_only, r_cleanSnow_WarmTs_threshold, i_lightpcp, &
       iclean_hydro_withRef, iclean_hydro_withRef_allcol, &
       ges_qr, ges_qnr, ges_qs, ges_qg, ref_mos_3d, t_bk, p_bk, sat_ctp, &
       rain_3d, nrain_3d, snow_3d, graupel_3d)

  ! The temperature and pressure the retrieval saw: the two expressions of
  ! PrecipMxr_radar.f90:106-107, evaluated again for export.
  do j = 1, lat2
    do i = 1, lon2
      do k = 1, nsig
        t_3d(i,j,k) = t_bk(i,j,k)*(p_bk(i,j,k)/h1000)**rd_over_cp
        p_3d(i,j,k) = p_bk(i,j,k)*100.0_r_single
      end do
    end do
  end do

  open(newunit=u, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  write(u) lon2, lat2, nsig
  write(u) rain_3d
  write(u) nrain_3d
  write(u) snow_3d
  write(u) graupel_3d
  write(u) t_3d
  write(u) p_3d
  close(u)
end program oracle_b
