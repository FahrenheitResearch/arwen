! Oracle driver for the WRF v4.6.1 map-projection authority.
!
! Compiles the pristine share/module_llxy.F (WRF v4.6.1, commit
! d66e442fccc04111067e29274c9f9eaccc3cef28) unmodified, promoted to
! double precision with -fdefault-real-8, and answers a query deck on
! stdin with IEEE-754 binary64 hex words on stdout so no decimal
! rounding sits between the Fortran authority and the Python gate.
!
! Deck grammar (one query per line, list-directed fields):
!   CASE id code lat1 lon1 knowni knownj dx stdlon truelat1 truelat2
!   LLIJ id lat lon
!   IJLL id i j
!   MAPF id lat
!   ROTA id lon
! CASE activates a projection via map_set (code: 1=PROJ_LC, 2=PROJ_PS,
! 3=PROJ_MERC) and emits the SETUP line of derived projection state.
! Every non-CASE line must name the active case id.
!
! The MAPF and ROTA queries answer with the WPS geogrid formulas that
! populate geo_em MAPFAC_* and SINALPHA/COSALPHA.  Those live outside
! module_llxy, in WPS v4.6.0 geogrid/src/process_tile_module.F
! (sha256 ef546e2747987948f1aa681566936713fffa6c2cf2542a804540033bf9b479c2):
! get_map_factor (:1735-1876) and get_rotang (:1920-2067).  The
! subroutines below transcribe those loop bodies per point, verbatim
! except for removing the array indexing; get_rotang's lc_cone
! dependency resolves to module_llxy's lc_cone, whose text is
! byte-identical to WPS module_map_utils.F:1124-1157.

program run_llxy
   use module_llxy
   implicit none

   character(len=1024) :: line
   character(len=8)    :: tag
   character(len=32)   :: id, active_id
   type(proj_info)     :: proj
   integer             :: ios, code, active_code
   real                :: lat1, lon1, knowni, knownj, dx, stdlon
   real                :: tl1, tl2
   real                :: lat, lon, xi, xj, mfac, sina, cosa
   logical             :: have_case

   have_case = .false.
   active_id = ''
   active_code = -1
   tl1 = 0.
   tl2 = 0.
   stdlon = 0.

   do
      read (*, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle
      read (line, *) tag
      select case (trim(tag))
      case ('CASE')
         read (line, *) tag, id, code, lat1, lon1, knowni, knownj, dx, &
                        stdlon, tl1, tl2
         call map_init(proj)
         call map_set(code, proj, lat1=lat1, lon1=lon1, knowni=knowni, &
                      knownj=knownj, dx=dx, stdlon=stdlon, &
                      truelat1=tl1, truelat2=tl2)
         active_id = id
         active_code = code
         have_case = .true.
         if (code == PROJ_MERC) then
            write (*, '(A,1X,A,1X,I3,7(1X,A))') 'SETUP', trim(id), code, &
               r2h(proj%hemi), r2h(0.), r2h(proj%rebydx), r2h(proj%rsw), &
               r2h(0.), r2h(0.), r2h(proj%dlon)
         else if (code == PROJ_LC) then
            write (*, '(A,1X,A,1X,I3,7(1X,A))') 'SETUP', trim(id), code, &
               r2h(proj%hemi), r2h(proj%cone), r2h(proj%rebydx), &
               r2h(proj%rsw), r2h(proj%polei), r2h(proj%polej), r2h(0.)
         else
            write (*, '(A,1X,A,1X,I3,7(1X,A))') 'SETUP', trim(id), code, &
               r2h(proj%hemi), r2h(0.), r2h(proj%rebydx), r2h(proj%rsw), &
               r2h(proj%polei), r2h(proj%polej), r2h(0.)
         end if
      case ('LLIJ')
         read (line, *) tag, id, lat, lon
         call require_case(id)
         call latlon_to_ij(proj, lat, lon, xi, xj)
         write (*, '(A,1X,A,4(1X,A))') 'LLIJ', trim(id), &
            r2h(lat), r2h(lon), r2h(xi), r2h(xj)
      case ('IJLL')
         read (line, *) tag, id, xi, xj
         call require_case(id)
         call ij_to_latlon(proj, xi, xj, lat, lon)
         write (*, '(A,1X,A,4(1X,A))') 'IJLL', trim(id), &
            r2h(xi), r2h(xj), r2h(lat), r2h(lon)
      case ('MAPF')
         read (line, *) tag, id, lat
         call require_case(id)
         call geogrid_map_factor(active_code, tl1, tl2, lat, mfac)
         write (*, '(A,1X,A,2(1X,A))') 'MAPF', trim(id), &
            r2h(lat), r2h(mfac)
      case ('ROTA')
         read (line, *) tag, id, lon
         call require_case(id)
         call geogrid_rotang(active_code, tl1, tl2, stdlon, lon, sina, cosa)
         write (*, '(A,1X,A,3(1X,A))') 'ROTA', trim(id), &
            r2h(lon), r2h(sina), r2h(cosa)
      case default
         write (0, '(A,A)') 'unknown deck tag: ', trim(tag)
         stop 2
      end select
   end do

contains

   function r2h(x) result(h)
      ! IEEE-754 binary64 hex word: no decimal rounding between the
      ! oracle and the fixture (pattern: noahmp oracle real2hex).
      real, intent(in) :: x
      character(len=16) :: h
      integer(kind=8) :: bits
      bits = transfer(x, bits)
      write (h, '(Z16.16)') bits
   end function r2h

   subroutine require_case(id)
      character(len=*), intent(in) :: id
      if (.not. have_case .or. trim(id) /= trim(active_id)) then
         write (0, '(A,A)') 'query names inactive case: ', trim(id)
         stop 3
      end if
   end subroutine require_case

   subroutine geogrid_map_factor(iproj_type, truelat1, truelat2, xlat, &
                                 mapfac)
      ! Per-point transcription of WPS v4.6.0 get_map_factor
      ! (geogrid/src/process_tile_module.F:1735-1876), PROJ_LC /
      ! PROJ_PS / PROJ_MERC branches, loop bodies verbatim.
      integer, intent(in) :: iproj_type
      real, intent(in)    :: truelat1, truelat2, xlat
      real, intent(out)   :: mapfac
      real :: n, colat, colat0, colat1, colat2

      if (iproj_type == PROJ_LC) then
         if (truelat1 /= truelat2) then
            colat1 = rad_per_deg*(90.0 - truelat1)
            colat2 = rad_per_deg*(90.0 - truelat2)
            n = (log(sin(colat1)) - log(sin(colat2))) &
                / (log(tan(colat1/2.0)) - log(tan(colat2/2.0)))
            colat = rad_per_deg*(90.0 - xlat)
            mapfac = sin(colat2)/sin(colat)*(tan(colat/2.0) &
                     /tan(colat2/2.0))**n
         else
            colat0 = rad_per_deg*(90.0 - truelat1)
            colat = rad_per_deg*(90.0 - xlat)
            mapfac = sin(colat0)/sin(colat)*(tan(colat/2.0) &
                     /tan(colat0/2.0))**cos(colat0)
         end if
      else if (iproj_type == PROJ_PS) then
         mapfac = (1.0 + sin(rad_per_deg*abs(truelat1)))/ &
                  (1.0 + sin(rad_per_deg*sign(1.,truelat1)*xlat))
      else if (iproj_type == PROJ_MERC) then
         colat0 = rad_per_deg*(90.0 - truelat1)
         colat = rad_per_deg*(90.0 - xlat)
         mapfac = sin(colat0) / sin(colat)
      else
         write (0, '(A)') 'geogrid_map_factor: unsupported projection'
         stop 4
      end if
   end subroutine geogrid_map_factor

   subroutine geogrid_rotang(iproj_type, truelat1, truelat2, stand_lon, &
                             xlon, sina, cosa)
      ! Per-point transcription of WPS v4.6.0 get_rotang
      ! (geogrid/src/process_tile_module.F:1920-2067), PROJ_LC /
      ! PROJ_PS / PROJ_MERC branches, loop bodies verbatim.  Note the
      ! authority applies no hemisphere factor: alpha is
      ! wrap(stand_lon - lon) * cone (cone from lc_cone, positive in
      ! both hemispheres) for PROJ_LC, wrap(stand_lon - lon) for
      ! PROJ_PS, and identically zero for PROJ_MERC.
      integer, intent(in) :: iproj_type
      real, intent(in)    :: truelat1, truelat2, stand_lon, xlon
      real, intent(out)   :: sina, cosa
      real :: cone, alpha, d_lon

      if (iproj_type == PROJ_LC) then
         call lc_cone(truelat1, truelat2, cone)
         d_lon = stand_lon - xlon
         if (d_lon > 180.) then
            d_lon = d_lon - 360.
         else if (d_lon < -180.) then
            d_lon = d_lon + 360.
         end if
         alpha = d_lon * cone * RAD_PER_DEG
         sina = sin(alpha)
         cosa = cos(alpha)
      else if (iproj_type == PROJ_PS) then
         d_lon = stand_lon - xlon
         if (d_lon > 180.) then
            d_lon = d_lon - 360.
         else if (d_lon < -180.) then
            d_lon = d_lon + 360.
         end if
         alpha = d_lon * RAD_PER_DEG
         sina = sin(alpha)
         cosa = cos(alpha)
      else if (iproj_type == PROJ_MERC) then
         sina = 0.0
         cosa = 1.0
      else
         write (0, '(A)') 'geogrid_rotang: unsupported projection'
         stop 5
      end if
   end subroutine geogrid_rotang

end program run_llxy
