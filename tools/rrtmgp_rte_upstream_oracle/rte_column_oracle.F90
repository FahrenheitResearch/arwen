! Column oracle for WOOF's RTE+RRTMGP: the upstream Fortran, run as is.
!
! Built by build.sh against earth-system-radiation/rte-rrtmgp at commit
! fa107a16120051c4124305c6b3d4c87059119f58 (the commit gpuwm/core/rrtmgp.py
! cites) in single precision (-DRTE_USE_SP, wp = c_float), gfortran -O0
! -ffp-contract=off, so every float operation is one IEEE binary32 rounding
! in source order.  Nothing in this file computes physics: it reads the
! columns, calls the upstream frontends in the order WOOF's production
! radiation call composes them, and writes every intermediate the GPU path
! also produces.
!
! Composition (WOOF's coupling, expressed in upstream routines):
!   LW  gas_optics (1scl + internal Planck sources, tlev supplied)
!       cloud_optics on bands (2str), ice roughness 2
!       McICA: g-point clouds = band clouds with tau * mask (mask 0 or 1)
!       clouds%increment(atmos)            -> increment_1scalar_by_2stream
!       rte_lw (one angle, broadband fluxes)
!   SW  gas_optics (2str + TOA source)
!       cloud_optics on bands, McICA as above
!       clouds%increment(atmos)            -> increment_2stream_by_2stream
!       atmos%delta_scale()                -> delta_scale_2str_k (f = g**2)
!       rte_sw (mu0 by column, alb_dir = alb_dif, broadband fluxes)
!
! Stream I/O, little-endian, Fortran column-major (the Python side writes
! numpy arrays transposed accordingly).  See make_columns.py for the layout.
program rte_column_oracle
  use mo_rte_kind,            only: wp
  use mo_optical_props,       only: ty_optical_props_1scl, ty_optical_props_2str
  use mo_gas_concentrations,  only: ty_gas_concs
  use mo_source_functions,    only: ty_source_func_lw
  use mo_fluxes,              only: ty_fluxes_broadband
  use mo_rte_lw,              only: rte_lw
  use mo_rte_sw,              only: rte_sw
  use mo_gas_optics_rrtmgp,   only: ty_gas_optics_rrtmgp
  use mo_cloud_optics_rrtmgp, only: ty_cloud_optics_rrtmgp
  use mo_optics_utils_rrtmgp, only: load_gas_optics, load_cloud_optics
  use mo_testing_utils,       only: stop_on_err
  implicit none

  character(len=1024) :: infile, outfile, datadir
  integer :: ncol, nlay, ngas, nsw, ngpt_lw, ngpt_sw, nbnd_lw, nbnd_sw
  integer :: i, g, uin, uout
  character(len=32), allocatable :: gnames(:)
  real(wp), allocatable :: play(:,:), plev(:,:), tlay(:,:), tlev(:,:), tsfc(:)
  real(wp), allocatable :: vmr(:,:,:)
  real(wp), allocatable :: clwp(:,:), ciwp(:,:), reliq(:,:), dgice(:,:)
  real(wp), allocatable :: mask_lw(:,:,:), mask_sw(:,:,:)
  real(wp), allocatable :: emis(:), mu0(:), alb(:), inc_gpt(:)
  integer,  allocatable :: sw_cols(:)
  real(wp), allocatable :: emis_b(:,:), alb_b(:,:), inc(:,:), toa(:,:)
  real(wp), allocatable, target :: flw_up(:,:), flw_dn(:,:)
  real(wp), allocatable, target :: fsw_up(:,:), fsw_dn(:,:), fsw_dir(:,:)
  integer,  allocatable :: bands(:)

  type(ty_gas_concs)           :: gas_concs, gas_concs_sw
  type(ty_gas_optics_rrtmgp)   :: kd_lw, kd_sw
  type(ty_cloud_optics_rrtmgp) :: co_lw, co_sw
  type(ty_optical_props_1scl)  :: atm_lw
  type(ty_optical_props_2str)  :: atm_sw, cld_lw, cld_sw, cg_lw, cg_sw
  type(ty_source_func_lw)      :: src
  type(ty_fluxes_broadband)    :: flw, fsw

  call get_command_argument(1, datadir)
  call get_command_argument(2, infile)
  call get_command_argument(3, outfile)

  open(newunit=uin, file=trim(infile), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(uin) ncol, nlay, ngas, nsw
  allocate(gnames(ngas))
  do i = 1, ngas
    read(uin) gnames(i)
  end do
  allocate(play(ncol,nlay), plev(ncol,nlay+1), tlay(ncol,nlay), &
           tlev(ncol,nlay+1), tsfc(ncol), vmr(ncol,nlay,ngas))
  read(uin) play, plev, tlay, tlev, tsfc, vmr
  allocate(clwp(ncol,nlay), ciwp(ncol,nlay), reliq(ncol,nlay), dgice(ncol,nlay))
  read(uin) clwp, ciwp, reliq, dgice
  read(uin) ngpt_lw, ngpt_sw
  allocate(mask_lw(ncol,nlay,ngpt_lw), emis(ncol))
  read(uin) mask_lw, emis
  allocate(sw_cols(nsw), mask_sw(nsw,nlay,ngpt_sw), mu0(nsw), alb(nsw), &
           inc_gpt(ngpt_sw))
  read(uin) sw_cols, mask_sw, mu0, alb, inc_gpt
  close(uin)

  call stop_on_err(gas_concs%init(gnames))
  do i = 1, ngas
    call stop_on_err(gas_concs%set_vmr(trim(gnames(i)), vmr(:,:,i)))
  end do
  call load_gas_optics(kd_lw, trim(datadir)//'/rrtmgp-gas-lw-g256.nc', gas_concs)
  call load_gas_optics(kd_sw, trim(datadir)//'/rrtmgp-gas-sw-g224.nc', gas_concs)
  call load_cloud_optics(co_lw, trim(datadir)//'/rrtmgp-clouds-lw-bnd.nc')
  call load_cloud_optics(co_sw, trim(datadir)//'/rrtmgp-clouds-sw-bnd.nc')
  call stop_on_err(co_lw%set_ice_roughness(2))
  call stop_on_err(co_sw%set_ice_roughness(2))
  if (kd_lw%get_ngpt() /= ngpt_lw .or. kd_sw%get_ngpt() /= ngpt_sw) &
    call stop_on_err("g-point count of the coefficient files differs from the input")
  nbnd_lw = kd_lw%get_nband()
  nbnd_sw = kd_sw%get_nband()

  open(newunit=uout, file=trim(outfile), access='stream', form='unformatted', &
       status='replace', convert='little_endian')

  ! ------------------------------------------------------------------ LW
  call stop_on_err(atm_lw%alloc_1scl(ncol, nlay, kd_lw))
  call stop_on_err(src%alloc(ncol, nlay, kd_lw))
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas_concs, &
                                    atm_lw, src, tlev=tlev))
  write(uout) atm_lw%tau, src%lay_source, src%lev_source, src%sfc_source

  call stop_on_err(cld_lw%alloc_2str(ncol, nlay, co_lw))
  call stop_on_err(co_lw%cloud_optics(clwp, ciwp, reliq, dgice, cld_lw))
  write(uout) cld_lw%tau, cld_lw%ssa, cld_lw%g

  call stop_on_err(cg_lw%alloc_2str(ncol, nlay, kd_lw))
  bands = kd_lw%get_gpoint_bands()
  do g = 1, ngpt_lw
    cg_lw%tau(:,:,g) = cld_lw%tau(:,:,bands(g)) * mask_lw(:,:,g)
    cg_lw%ssa(:,:,g) = cld_lw%ssa(:,:,bands(g))
    cg_lw%g  (:,:,g) = cld_lw%g  (:,:,bands(g))
  end do
  call stop_on_err(cg_lw%increment(atm_lw))
  write(uout) atm_lw%tau

  allocate(emis_b(nbnd_lw, ncol), flw_up(ncol,nlay+1), flw_dn(ncol,nlay+1))
  do i = 1, ncol
    emis_b(:, i) = emis(i)
  end do
  flw%flux_up => flw_up
  flw%flux_dn => flw_dn
  call stop_on_err(rte_lw(atm_lw, src, emis_b, flw))
  write(uout) flw_up, flw_dn

  ! ------------------------------------------------------------------ SW
  ! Daylit columns only, as WOOF's driver solves them.
  call stop_on_err(gas_concs_sw%init(gnames))
  do i = 1, ngas
    call stop_on_err(gas_concs_sw%set_vmr(trim(gnames(i)), vmr(sw_cols,:,i)))
  end do
  call stop_on_err(atm_sw%alloc_2str(nsw, nlay, kd_sw))
  allocate(toa(nsw, ngpt_sw))
  call stop_on_err(kd_sw%gas_optics(play(sw_cols,:), plev(sw_cols,:), &
                                    tlay(sw_cols,:), gas_concs_sw, atm_sw, toa))
  write(uout) atm_sw%tau, atm_sw%ssa, toa(1,:)

  call stop_on_err(cld_sw%alloc_2str(nsw, nlay, co_sw))
  call stop_on_err(co_sw%cloud_optics(clwp(sw_cols,:), ciwp(sw_cols,:), &
                                      reliq(sw_cols,:), dgice(sw_cols,:), cld_sw))
  write(uout) cld_sw%tau, cld_sw%ssa, cld_sw%g

  call stop_on_err(cg_sw%alloc_2str(nsw, nlay, kd_sw))
  bands = kd_sw%get_gpoint_bands()
  do g = 1, ngpt_sw
    cg_sw%tau(:,:,g) = cld_sw%tau(:,:,bands(g)) * mask_sw(:,:,g)
    cg_sw%ssa(:,:,g) = cld_sw%ssa(:,:,bands(g))
    cg_sw%g  (:,:,g) = cld_sw%g  (:,:,bands(g))
  end do
  call stop_on_err(cg_sw%increment(atm_sw))
  write(uout) atm_sw%tau, atm_sw%ssa, atm_sw%g
  call stop_on_err(atm_sw%delta_scale())
  write(uout) atm_sw%tau, atm_sw%ssa, atm_sw%g

  allocate(alb_b(nbnd_sw, nsw), inc(nsw, ngpt_sw))
  do i = 1, nsw
    alb_b(:, i) = alb(i)
    inc(i, :) = inc_gpt(:)
  end do
  allocate(fsw_up(nsw,nlay+1), fsw_dn(nsw,nlay+1), fsw_dir(nsw,nlay+1))
  fsw%flux_up => fsw_up
  fsw%flux_dn => fsw_dn
  fsw%flux_dn_dir => fsw_dir
  call stop_on_err(rte_sw(atm_sw, mu0, inc, alb_b, alb_b, fsw))
  write(uout) fsw_up, fsw_dn, fsw_dir
  close(uout)
end program rte_column_oracle
