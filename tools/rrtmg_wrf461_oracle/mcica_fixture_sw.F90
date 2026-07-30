! Fixture driver for WRF v4.6.1 mcica_subcol_sw (module mcica_subcol_gen_sw
! inside phys/module_ra_rrtmg_sw.F, compiled unmodified).
!
! Input stream layout (big-endian; see mcica_fixture_lw.F90):
!   int32  ncol, nlay, icld, permuteseed, irng, idcor, juldat
!   real32 lat
!   real32 play(ncol,nlay), hgt(ncol,nlay), cldfrac(ncol,nlay),
!          ciwp(ncol,nlay), clwp(ncol,nlay), cswp(ncol,nlay),
!          rei(ncol,nlay), rel(ncol,nlay), res(ncol,nlay),
!          tauc(nbndsw,ncol,nlay), ssac(nbndsw,ncol,nlay),
!          asmc(nbndsw,ncol,nlay), fsfc(nbndsw,ncol,nlay)
program mcica_fixture_sw
   use parkind, only: im => kind_im, rb => kind_rb
   use parrrsw, only: nbndsw, ngptsw
   use mcica_subcol_gen_sw, only: mcica_subcol_sw
   use module_ra_rrtmg_sw, only: rrtmg_swinit
   use dump_kit
   implicit none

   integer, parameter :: kme = 50
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
   real(kind=rb), allocatable :: tauc(:, :, :), ssac(:, :, :)
   real(kind=rb), allocatable :: asmc(:, :, :), fsfc(:, :, :)
   real(kind=rb), allocatable :: cldfmcl(:, :, :), ciwpmcl(:, :, :)
   real(kind=rb), allocatable :: clwpmcl(:, :, :), cswpmcl(:, :, :)
   real(kind=rb), allocatable :: taucmcl(:, :, :), ssacmcl(:, :, :)
   real(kind=rb), allocatable :: asmcmcl(:, :, :), fsfcmcl(:, :, :)
   real(kind=rb), allocatable :: reicmcl(:, :), relqmcl(:, :), resnmcl(:, :)

   call check_kinds()
   if (command_argument_count() /= 2) then
      write (*, '(A)') 'usage: mcica_fixture_sw INFILE OUTFILE'
      error stop 2
   end if
   call get_command_argument(1, infile)
   call get_command_argument(2, outfile)

   call rrtmg_swinit(.true., 1, 2, 1, 2, 1, kme, &
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
   allocate (tauc(nbndsw, ncol, nlay), ssac(nbndsw, ncol, nlay))
   allocate (asmc(nbndsw, ncol, nlay), fsfc(nbndsw, ncol, nlay))
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
   read (uin) ssac
   read (uin) asmc
   read (uin) fsfc
   close (uin)

   allocate (cldfmcl(ngptsw, ncol, nlay), ciwpmcl(ngptsw, ncol, nlay))
   allocate (clwpmcl(ngptsw, ncol, nlay), cswpmcl(ngptsw, ncol, nlay))
   allocate (taucmcl(ngptsw, ncol, nlay), ssacmcl(ngptsw, ncol, nlay))
   allocate (asmcmcl(ngptsw, ncol, nlay), fsfcmcl(ngptsw, ncol, nlay))
   allocate (reicmcl(ncol, nlay), relqmcl(ncol, nlay), resnmcl(ncol, nlay))
   cldfmcl = -9999.0_rb
   ciwpmcl = -9999.0_rb
   clwpmcl = -9999.0_rb
   cswpmcl = -9999.0_rb
   taucmcl = -9999.0_rb
   ssacmcl = -9999.0_rb
   asmcmcl = -9999.0_rb
   fsfcmcl = -9999.0_rb
   reicmcl = -9999.0_rb
   relqmcl = -9999.0_rb
   resnmcl = -9999.0_rb

   call mcica_subcol_sw(iplon, ncol, nlay, icld, permuteseed, irng, play, &
                        cldfrac, ciwp, clwp, cswp, rei, rel, res, &
                        tauc, ssac, asmc, fsfc, &
                        hgt, idcor, juldat, lat, &
                        cldfmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, &
                        relqmcl, resnmcl, taucmcl, ssacmcl, asmcmcl, &
                        fsfcmcl)

   open (newunit=uout, file=trim(outfile), form='unformatted', &
         access='stream', status='replace')
   call wi0(uout, 'mcica_sw/irng_out', irng)
   call wr3(uout, 'mcica_sw/cldfmcl', cldfmcl)
   call wr3(uout, 'mcica_sw/ciwpmcl', ciwpmcl)
   call wr3(uout, 'mcica_sw/clwpmcl', clwpmcl)
   call wr3(uout, 'mcica_sw/cswpmcl', cswpmcl)
   call wr3(uout, 'mcica_sw/taucmcl', taucmcl)
   call wr3(uout, 'mcica_sw/ssacmcl', ssacmcl)
   call wr3(uout, 'mcica_sw/asmcmcl', asmcmcl)
   call wr3(uout, 'mcica_sw/fsfcmcl', fsfcmcl)
   call wr2(uout, 'mcica_sw/reicmcl', reicmcl)
   call wr2(uout, 'mcica_sw/relqmcl', relqmcl)
   call wr2(uout, 'mcica_sw/resnmcl', resnmcl)
   close (uout)
   write (*, '(A)') 'mcica_fixture_sw: wrote ' // trim(outfile)
end program mcica_fixture_sw
