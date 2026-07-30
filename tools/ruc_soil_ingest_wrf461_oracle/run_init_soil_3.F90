! Drive WRF v4.6.1 share/module_soil_pre.F init_soil_depth_3 + init_soil_3_real
! byte-unmodified and dump every input and output as CSV.
!
! This is the routine real.exe uses to build a RUC LSM initial soil column
! (share/module_soil_pre.F:817-830 selects it for RUCLSMSCHEME).  It is driven
! directly rather than mirrored, because a mirror agrees with itself.
!
! Two source geometries are exercised, both of which reach gpuwm:
!
!   flag_soil_layers = 1  a layer source (ERA5 / GFS through metgrid).  The
!                         n layer values arrive in st_input(2..n+1); WRF
!                         itself writes tsk into slot 1 and tmn into slot
!                         n+2, and the sample depths are the layer MIDPOINTS
!                         in centimetres, integer-truncated
!                         (share/module_optional_input.F:1949-1954,
!                         :1330-1355).
!   flag_soil_levels = 1  a node source already on RUC's own levels
!                         (HRRR / RUC-native).  The n node values arrive in
!                         st_input(1..n) and there is no surface or deep
!                         anchor at all.
!
! tslb and smois are pre-filled with a sentinel before every call.  They are
! INTENT(OUT) with no default initialisation, and init_soil_3_real assigns a
! target level ONLY when it falls inside the source bracket, so a sentinel
! surviving into the CSV is WRF leaving that level undefined -- which a
! consumer must know about rather than discover in a forecast.

program run_init_soil_3

   use module_soil_pre, only : init_soil_depth_3, init_soil_3_real, &
                               em_width, hold_ups

   implicit none

   integer, parameter :: nsoil = 9        ! RUC, num_soil_layers = 9
   integer, parameter :: ncol  = 20       ! independent columns per experiment
   integer, parameter :: nalloc = 12      ! >= max(nlev)+2
   integer, parameter :: nexp  = 7

   integer, parameter :: ids = 1, ide = ncol + 1, jds = 1, jde = 2
   integer, parameter :: kds = 1, kde = 2
   integer, parameter :: ims = ids, ime = ide, jms = jds, jme = jde
   integer, parameter :: kms = kds, kme = kde
   integer, parameter :: its = ids, ite = ide - 1, jts = jds, jte = jde - 1
   integer, parameter :: kts = kds, kte = kde - 1

   real, parameter :: sentinel = -1.0e30

   real, dimension(nsoil) :: zs, dzs
   real, dimension(ims:ime, 1:nalloc, jms:jme) :: st_input, sm_input
   real, dimension(ims:ime, jms:jme) :: landmask, sst, tsk, tmn
   real, dimension(ims:ime, 1:nsoil, jms:jme) :: tslb, smois
   integer, dimension(100) :: st_levels_input, sm_levels_input

   ! ---- the per-column source profiles ------------------------------------
   ! Four-layer source values (ERA5 0-7/7-28/28-100/100-289 cm and the
   ! GFS/Noah 0-10/10-40/40-100/100-200 cm sets both use these numbers; only
   ! the declared sample depths differ).  Chosen to span: moisture rising
   ! with depth (the only shape that arms init_soil_3_real's factorsm
   ! branch), moisture falling with depth, an exactly flat profile (where the
   ! strict > comparisons must NOT arm it), moisture small enough that the
   ! max(0.02,...) floor binds, and one exactly zero column.
   real, dimension(4, ncol) :: st4, sm4
   real, dimension(ncol) :: tsk0, tmn0, sst0, mask0

   integer :: iexp, c, k, l, nlev, unit_out
   integer :: flag_layers, flag_levels, flag_adj, flag_sst
   character(len=64) :: label
   character(len=256) :: outfile
   character(len=24) :: field

   data st4 / &
      300.0, 295.0, 290.0, 288.0, &   !  1 warm surface, cooling with depth
      265.0, 270.0, 275.0, 280.0, &   !  2 cold surface, warming with depth
      285.0, 286.0, 287.0, 288.0, &   !  3 near-isothermal
      310.0, 305.0, 300.0, 295.0, &   !  4 hot and dry
      301.5, 299.5, 297.5, 293.5, &   !  5 very dry, moisture rising
      250.0, 255.0, 260.0, 265.0, &   !  6 frozen
      290.0, 290.0, 290.0, 290.0, &   !  7 exactly uniform
      320.0, 300.0, 285.0, 280.0, &   !  8 steep near-surface gradient
      270.0, 280.0, 290.0, 300.0, &   !  9 inverted
      288.0, 288.5, 289.0, 289.5, &   ! 10 zero moisture everywhere
      275.3, 276.7, 278.1, 279.9, &   ! 11 tsk far above the profile
      298.4, 297.2, 296.1, 295.3, &   ! 12 tsk far below the profile
      283.0, 284.0, 285.0, 286.0, &   ! 13 very cold tmn
      283.0, 284.0, 285.0, 286.0, &   ! 14 very warm tmn
      292.25, 291.75, 291.25, 290.75, &   ! 15 saturated
      292.25, 291.75, 291.25, 290.75, &   ! 16 saturated, rising
      305.125, 302.375, 299.625, 296.875, & ! 17 non-round mantissas
      271.0625, 272.1875, 273.3125, 274.4375, & ! 18 non-round mantissas
      295.0, 294.0, 293.0, 292.0, &   ! 19 open water
      289.0, 288.0, 287.0, 286.0 /    ! 20 open water, sst != tsk

   data sm4 / &
      0.05,  0.10,   0.18,   0.25,  &   !  1 rising  -> factorsm armed
      0.30,  0.28,   0.25,   0.20,  &   !  2 falling
      0.45,  0.46,   0.47,   0.48,  &   !  3 rising
      0.005, 0.004,  0.003,  0.002, &   !  4 falling, floor binds
      0.001, 0.002,  0.003,  0.004, &   !  5 rising, floor binds
      0.15,  0.16,   0.14,   0.13,  &   !  6 mixed
      0.20,  0.20,   0.20,   0.20,  &   !  7 flat: > is false, factorsm off
      0.02,  0.30,   0.40,   0.45,  &   !  8 rising, large jump
      0.40,  0.30,   0.20,   0.10,  &   !  9 falling
      0.0,   0.0,    0.0,    0.0,   &   ! 10 exactly zero
      0.12,  0.14,   0.16,   0.18,  &   ! 11 rising
      0.33,  0.31,   0.29,   0.27,  &   ! 12 falling
      0.22,  0.24,   0.26,   0.28,  &   ! 13 rising
      0.28,  0.26,   0.24,   0.22,  &   ! 14 falling
      0.50,  0.50,   0.50,   0.50,  &   ! 15 flat at 0.5
      0.47,  0.48,   0.49,   0.50,  &   ! 16 rising to 0.5
      0.171875, 0.203125, 0.234375, 0.265625, & ! 17 rising, exact binary
      0.265625, 0.234375, 0.203125, 0.171875, & ! 18 falling, exact binary
      0.35,  0.35,   0.35,   0.35,  &   ! 19 open water
      0.10,  0.20,   0.30,   0.40 /     ! 20 open water, rising

   data tsk0 / 301.0, 263.0, 285.5, 312.0, 302.0, 248.0, 290.0, 325.0, &
               268.0, 287.5, 310.0, 260.0, 282.0, 282.0, 292.5, 292.5, &
               306.0, 270.0, 295.0, 290.0 /
   data tmn0 / 287.0, 279.0, 288.5, 293.0, 292.0, 268.0, 290.0, 278.0, &
               302.0, 290.0, 281.0, 294.5, 200.0, 380.0, 290.25, 290.25, &
               295.5, 275.5, 293.0, 285.0 /
   data sst0 / 301.0, 263.0, 285.5, 312.0, 302.0, 248.0, 290.0, 325.0, &
               268.0, 287.5, 310.0, 260.0, 282.0, 282.0, 292.5, 292.5, &
               306.0, 270.0, 291.0, 279.5 /
   data mask0 / 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0 /

   ! module_initialize_real.F:430 sets hold_ups = .TRUE. at the initial time,
   ! which is the only time RUC soil state is built.  skip_middle_points_t
   ! then returns .FALSE. everywhere, so every column is processed.
   hold_ups = .TRUE.
   em_width = 0

   if (command_argument_count() /= 1) then
      write(0, '(A)') 'usage: run_init_soil_3 OUTPUT.csv'
      error stop 2
   end if
   call get_command_argument(1, outfile)

   call init_soil_depth_3(zs, dzs, nsoil)

   open(newunit=unit_out, file=trim(outfile), status='replace', action='write')

   ! Header.  zs/dzs are constants of the scheme, so they are written once as
   ! a comment rather than repeated on every row.
   write(unit_out, '(A)', advance='no') '# zs_m='
   do k = 1, nsoil
      write(field, '(ES17.9E3)') zs(k)
      write(unit_out, '(A,A)', advance='no') trim(adjustl(field)), ';'
   end do
   write(unit_out, '(A)', advance='no') ' dzs_m='
   do k = 1, nsoil
      write(field, '(ES17.9E3)') dzs(k)
      write(unit_out, '(A,A)', advance='no') trim(adjustl(field)), ';'
   end do
   write(unit_out, '(A)') ''

   write(unit_out, '(A)', advance='no') &
      'experiment,flag_soil_layers,flag_soil_levels,flag_sm_adj,flag_sst,' // &
      'nlev,col,landmask,tsk_in,tmn,sst'
   do k = 1, nsoil
      write(field, '(I0)') k
      write(unit_out, '(A,A)', advance='no') ',lev_cm_', trim(field)
   end do
   do k = 1, nsoil
      write(field, '(I0)') k
      write(unit_out, '(A,A)', advance='no') ',st_src_', trim(field)
   end do
   do k = 1, nsoil
      write(field, '(I0)') k
      write(unit_out, '(A,A)', advance='no') ',sm_src_', trim(field)
   end do
   write(unit_out, '(A)', advance='no') ',tsk_out'
   do k = 1, nsoil
      write(field, '(I0)') k
      write(unit_out, '(A,A)', advance='no') ',tslb_', trim(field)
   end do
   do k = 1, nsoil
      write(field, '(I0)') k
      write(unit_out, '(A,A)', advance='no') ',smois_', trim(field)
   end do
   write(unit_out, '(A)') ''

   do iexp = 1, nexp

      call experiment_setup(iexp, label, nlev, flag_layers, flag_levels, &
                            flag_adj, flag_sst)

      ! ---- fill the WRF-side input arrays exactly as real.exe does --------
      st_input = sentinel
      sm_input = sentinel
      tslb = sentinel
      smois = sentinel
      st_levels_input = -1
      sm_levels_input = -1

      do k = 1, nlev
         st_levels_input(k) = source_level_cm(iexp, k)
         sm_levels_input(k) = source_level_cm(iexp, k)
      end do

      do c = 1, ncol
         landmask(c, jts) = mask0(c)
         tsk(c, jts) = tsk0(c)
         tmn(c, jts) = tmn0(c)
         sst(c, jts) = sst0(c)
         do k = 1, nlev
            if (flag_levels == 1) then
               ! share/module_optional_input.F:1313-1316 -- node source lands
               ! in slots 1..n.
               st_input(c, k, jts) = source_st(iexp, k, c)
               sm_input(c, k, jts) = source_sm(iexp, k, c)
            else
               ! share/module_optional_input.F:1364-1370 -- layer source lands
               ! in slots 2..n+1, leaving slot 1 for tsk and n+2 for tmn.
               st_input(c, k + 1, jts) = source_st(iexp, k, c)
               sm_input(c, k + 1, jts) = source_sm(iexp, k, c)
            end if
         end do
      end do

      write(0, '(A,A)') '=== experiment: ', trim(label)

      call init_soil_3_real(tsk, tmn, smois, tslb, &
                            st_input, sm_input, landmask, sst, &
                            zs, dzs, flag_adj, &
                            st_levels_input, sm_levels_input, &
                            nsoil, nlev, nlev, &
                            nalloc, nalloc, &
                            flag_sst, flag_layers, flag_levels, &
                            ids, ide, jds, jde, kds, kde, &
                            ims, ime, jms, jme, kms, kme, &
                            its, ite, jts, jte, kts, kte)

      do c = 1, ncol
         write(unit_out, '(A)', advance='no') trim(label)
         call put_int(unit_out, flag_layers)
         call put_int(unit_out, flag_levels)
         call put_int(unit_out, flag_adj)
         call put_int(unit_out, flag_sst)
         call put_int(unit_out, nlev)
         call put_int(unit_out, c)
         call put_real(unit_out, mask0(c))
         call put_real(unit_out, tsk0(c))
         call put_real(unit_out, tmn0(c))
         call put_real(unit_out, sst0(c))
         do k = 1, nsoil
            if (k <= nlev) then
               call put_int(unit_out, source_level_cm(iexp, k))
            else
               call put_int(unit_out, -1)
            end if
         end do
         do k = 1, nsoil
            if (k <= nlev) then
               call put_real(unit_out, source_st(iexp, k, c))
            else
               call put_real(unit_out, sentinel)
            end if
         end do
         do k = 1, nsoil
            if (k <= nlev) then
               call put_real(unit_out, source_sm(iexp, k, c))
            else
               call put_real(unit_out, sentinel)
            end if
         end do
         call put_real(unit_out, tsk(c, jts))
         do l = 1, nsoil
            call put_real(unit_out, tslb(c, l, jts))
         end do
         do l = 1, nsoil
            call put_real(unit_out, smois(c, l, jts))
         end do
         write(unit_out, '(A)') ''
      end do
   end do

   close(unit_out)

contains

   subroutine experiment_setup(iexp, label, nlev, flag_layers, flag_levels, &
                               flag_adj, flag_sst)
      integer, intent(in) :: iexp
      character(len=*), intent(out) :: label
      integer, intent(out) :: nlev, flag_layers, flag_levels, flag_adj, flag_sst

      select case (iexp)
      case (1)   ! ERA5 layers through metgrid, defaults
         label = 'era5_layers_noadj_nosst'
         nlev = 4 ; flag_layers = 1 ; flag_levels = 0 ; flag_adj = 0 ; flag_sst = 0
      case (2)   ! the same, with the Noah->RUC bucket adjustment armed
         label = 'era5_layers_adj_nosst'
         nlev = 4 ; flag_layers = 1 ; flag_levels = 0 ; flag_adj = 1 ; flag_sst = 0
      case (3)   ! the same, with an SST field present
         label = 'era5_layers_noadj_sst'
         nlev = 4 ; flag_layers = 1 ; flag_levels = 0 ; flag_adj = 0 ; flag_sst = 1
      case (4)   ! GFS/Noah layer depths
         label = 'noah_layers_adj_nosst'
         nlev = 4 ; flag_layers = 1 ; flag_levels = 0 ; flag_adj = 1 ; flag_sst = 0
      case (5)   ! node source already on RUC's nine levels
         label = 'ruc_levels_noadj_nosst'
         nlev = 9 ; flag_layers = 0 ; flag_levels = 1 ; flag_adj = 0 ; flag_sst = 0
      case (6)   ! node source, adjustment armed: it must be a no-op
         label = 'ruc_levels_adj_sst'
         nlev = 9 ; flag_layers = 0 ; flag_levels = 1 ; flag_adj = 1 ; flag_sst = 1
      case (7)   ! layer source handed in DEEPEST FIRST, to drive the sort
         label = 'era5_layers_reversed_adj'
         nlev = 4 ; flag_layers = 1 ; flag_levels = 0 ; flag_adj = 1 ; flag_sst = 0
      case default
         label = 'unreachable'
         nlev = 0 ; flag_layers = 0 ; flag_levels = 0 ; flag_adj = 0 ; flag_sst = 0
      end select
   end subroutine experiment_setup

   ! Sample depths in centimetres, as WRF's own reader computes them.
   ! char2int2 (share/module_optional_input.F:1949-1954) and the
   ! flag_soil_layers block (:1339-1352) both form the layer MIDPOINT with
   ! INTEGER division, so 0-7 cm becomes 3, not 7 and not 3.5.
   integer function source_level_cm(iexp, k) result(cm)
      integer, intent(in) :: iexp, k
      integer, dimension(4), parameter :: era5_bottom_cm = (/ 7, 28, 100, 289 /)
      integer, dimension(4), parameter :: noah_bottom_cm = (/ 10, 40, 100, 200 /)
      integer, dimension(9), parameter :: ruc_node_cm = &
         (/ 0, 1, 4, 10, 30, 60, 100, 160, 300 /)
      integer :: above, kk

      select case (iexp)
      case (5, 6)
         cm = ruc_node_cm(k)
      case (7)
         ! deepest first: st_levels_input arrives unsorted and
         ! init_soil_3_real:1881-1899 must sort it, and the profile with it.
         kk = 5 - k
         above = 0
         if (kk > 1) above = era5_bottom_cm(kk - 1)
         cm = (above + era5_bottom_cm(kk)) / 2
      case (4)
         above = 0
         if (k > 1) above = noah_bottom_cm(k - 1)
         cm = (above + noah_bottom_cm(k)) / 2
      case default
         above = 0
         if (k > 1) above = era5_bottom_cm(k - 1)
         cm = (above + era5_bottom_cm(k)) / 2
      end select
   end function source_level_cm

   real function source_st(iexp, k, c) result(value)
      integer, intent(in) :: iexp, k, c
      select case (iexp)
      case (5, 6)
         value = node_profile(st4(1, c), st4(4, c), k, c, 0.0)
      case (7)
         value = st4(5 - k, c)
      case default
         value = st4(k, c)
      end select
   end function source_st

   real function source_sm(iexp, k, c) result(value)
      integer, intent(in) :: iexp, k, c
      select case (iexp)
      case (5, 6)
         value = node_profile(sm4(1, c), sm4(4, c), k, c, 1.0)
      case (7)
         value = sm4(5 - k, c)
      case default
         value = sm4(k, c)
      end select
   end function source_sm

   ! Nine node values spanning the same endpoints as the four-layer table,
   ! with a per-column curvature so the node experiment is not a straight
   ! line the interpolation could reproduce by accident.  `cap` clamps
   ! moisture into [0,1]; temperature passes cap = 0 and is left alone.
   real function node_profile(top, bottom, k, c, cap) result(value)
      real, intent(in) :: top, bottom, cap
      integer, intent(in) :: k, c
      real :: frac, bend
      frac = real(k - 1) / 8.0
      bend = 0.05 * real(mod(c, 5) - 2) * (bottom - top)
      value = top + (bottom - top) * frac + bend * frac * (1.0 - frac)
      if (cap > 0.0) then
         value = max(0.0, min(cap, value))
      end if
   end function node_profile

   subroutine put_int(unit_out, value)
      integer, intent(in) :: unit_out, value
      character(len=16) :: buffer
      write(buffer, '(I0)') value
      write(unit_out, '(A,A)', advance='no') ',', trim(adjustl(buffer))
   end subroutine put_int

   subroutine put_real(unit_out, value)
      integer, intent(in) :: unit_out
      real, intent(in) :: value
      character(len=24) :: buffer
      ! ES17.9E3 round-trips every float32 exactly (9 significant digits).
      write(buffer, '(ES17.9E3)') value
      write(unit_out, '(A,A)', advance='no') ',', trim(adjustl(buffer))
   end subroutine put_real

end program run_init_soil_3
