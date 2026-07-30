! Fixture driver for WRF v4.6.1 mcica_subcol_lw (module mcica_subcol_gen_lw
! inside phys/module_ra_rrtmg_lw.F, compiled unmodified).
!
! Reads one case description from a raw stream input file (big-endian via
! the oracle build's -fconvert=big-endian):
!   int32  ncol, nlay, icld, permuteseed, irng, idcor, juldat
!   real32 lat
!   real32 play(ncol,nlay), hgt(ncol,nlay), cldfrac(ncol,nlay),
!          ciwp(ncol,nlay), clwp(ncol,nlay), cswp(ncol,nlay),
!          rei(ncol,nlay), rel(ncol,nlay), res(ncol,nlay),
!          tauc(nbndlw,ncol,nlay)
! all arrays in Fortran storage order.
!
! Calls the real WRF init first (rrtmg_lwinit reads RRTMG_LW_DATA and runs
! rrtmg_lw_ini, which fills rrlw_wvn's ngb used by the generator), exactly
! as wrf does before any radiation call, then calls mcica_subcol_lw with
! iplon = 1 as in the WRF driver, and dumps every output through dump_kit.
program mcica_fixture_lw
   use parkind, only: im => kind_im, rb => kind_rb
   use parrrtm, only: nbndlw, ngptlw
   use mcica_subcol_gen_lw, only: mcica_subcol_lw
   use module_ra_rrtmg_lw, only: rrtmg_lwinit
   use dump_kit
   implicit none

   integer, parameter :: kme = 50
   real, parameter :: p_top = 5000.0
   integer :: uin, uout
   character(len=1024) :: infile, outfile
   integer(kind=4) :: ncol4, nlay4, icld4, permuteseed4, irng4, idcor4, &
                      juldat4
   integer(kind=im) :: ncol, nlay, icld, permuteseed, irng, idcor, juldat
   integer(kind=im), parameter :: iplon = 1
   real(kind=rb) :: lat
   real(kind=rb), allocatable :: play(:, :), hgt(:, :), cldfrac(:, :)
   real(kind=rb), allocatable :: ciwp(:, :), clwp(:, :), cswp(:, :)
   real(kind=rb), allocatable :: rei(:, :), rel(:, :), res(:, :)
   real(kind=rb), allocatable :: tauc(:, :, :)
   real(kind=rb), allocatable :: cldfmcl(:, :, :), ciwpmcl(:, :, :)
   real(kind=rb), allocatable :: clwpmcl(:, :, :), cswpmcl(:, :, :)
   real(kind=rb), allocatable :: taucmcl(:, :, :)
   real(kind=rb), allocatable :: reicmcl(:, :), relqmcl(:, :), resnmcl(:, :)

   call check_kinds()
   if (command_argument_count() /= 2) then
      write (*, '(A)') 'usage: mcica_fixture_lw INFILE OUTFILE'
      error stop 2
   end if
   call get_command_argument(1, infile)
   call get_command_argument(2, outfile)

   call rrtmg_lwinit(p_top, .true., 1, 2, 1, 2, 1, kme, &
                     1, 2, 1, 2, 1, kme, 1, 1, 1, 1, 1, kme - 1)

   open (newunit=uin, file=trim(infile), form='unformatted', &
         access='stream', status='old')
   read (uin) ncol4, nlay4, icld4, permuteseed4, irng4, idcor4, juldat4
   read (uin) lat
   ncol = ncol4
   nlay = nlay4
   icld = icld4
   permuteseed = permuteseed4
   irng = irng4
   idcor = idcor4
   juldat = juldat4

   allocate (play(ncol, nlay), hgt(ncol, nlay), cldfrac(ncol, nlay))
   allocate (ciwp(ncol, nlay), clwp(ncol, nlay), cswp(ncol, nlay))
   allocate (rei(ncol, nlay), rel(ncol, nlay), res(ncol, nlay))
   allocate (tauc(nbndlw, ncol, nlay))
   read (uin) play
   read (uin) hgt
   read (uin) cldfrac
   read (uin) ciwp
   read (uin) clwp
   read (uin) cswp
   read (uin) rei
   read (uin) rel
   read (uin) res
   read (uin) tauc
   close (uin)

   allocate (cldfmcl(ngptlw, ncol, nlay), ciwpmcl(ngptlw, ncol, nlay))
   allocate (clwpmcl(ngptlw, ncol, nlay), cswpmcl(ngptlw, ncol, nlay))
   allocate (taucmcl(ngptlw, ncol, nlay))
   allocate (reicmcl(ncol, nlay), relqmcl(ncol, nlay), resnmcl(ncol, nlay))
   cldfmcl = -9999.0_rb
   ciwpmcl = -9999.0_rb
   clwpmcl = -9999.0_rb
   cswpmcl = -9999.0_rb
   taucmcl = -9999.0_rb
   reicmcl = -9999.0_rb
   relqmcl = -9999.0_rb
   resnmcl = -9999.0_rb

   call mcica_subcol_lw(iplon, ncol, nlay, icld, permuteseed, irng, play, &
                        cldfrac, ciwp, clwp, cswp, rei, rel, res, tauc, &
                        hgt, idcor, juldat, lat, &
                        cldfmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, &
                        relqmcl, resnmcl, taucmcl)

   open (newunit=uout, file=trim(outfile), form='unformatted', &
         access='stream', status='replace')
   call wi0(uout, 'mcica_lw/irng_out', irng)
   call wr3(uout, 'mcica_lw/cldfmcl', cldfmcl)
   call wr3(uout, 'mcica_lw/ciwpmcl', ciwpmcl)
   call wr3(uout, 'mcica_lw/clwpmcl', clwpmcl)
   call wr3(uout, 'mcica_lw/cswpmcl', cswpmcl)
   call wr3(uout, 'mcica_lw/taucmcl', taucmcl)
   call wr2(uout, 'mcica_lw/reicmcl', reicmcl)
   call wr2(uout, 'mcica_lw/relqmcl', relqmcl)
   call wr2(uout, 'mcica_lw/resnmcl', resnmcl)
   close (uout)
   write (*, '(A)') 'mcica_fixture_lw: wrote ' // trim(outfile)
end program mcica_fixture_lw
