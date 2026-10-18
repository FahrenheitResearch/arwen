! Calls the fork's thompson_init exactly as run_columns_fork.F90 does
! (aerosol aware, is_start = .false.), so every derived constant takes the
! value the column oracle runs with, then writes them through the harness
! subroutine fork_dump_constants (make_dump_module.py) to
! fork_constants.txt.  Run in a directory holding the fork's tables, as the
! column driver is.
program dump_fork_constants
  use module_mp_thompson, only: thompson_init, fork_dump_constants
  implicit none
  integer, parameter:: nxp = 2, nz = 2
  real:: hgt(nxp,nz,2), nwfa(nxp,nz,2), nifa(nxp,nz,2), nwfa2d(nxp,2)
  integer:: u, i, free_unit
  logical:: unit_opened

  hgt = 0.0
  hgt(:,2,:) = 100.0
  nwfa = 1.0e9
  nifa = 1.0e6
  nwfa2d = 1.0e6
  free_unit = -1
  do i = 20, 99
     inquire(unit=i, opened=unit_opened)
     if (.not. unit_opened) then
        free_unit = i
        exit
     endif
  enddo
  if (free_unit /= 20) error stop 3
  call thompson_init(                                                 &
       hgt=hgt, nwfa2d=nwfa2d, nwfa=nwfa, nifa=nifa,                  &
       dx=3000.0, dy=3000.0, is_start=.false.,                        &
       ids=1, ide=nxp, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nxp, jms=1, jme=2, kms=1, kme=nz,                   &
       its=1, ite=nxp, jts=1, jte=2, kts=1, kte=nz)
  open(newunit=u, file='fork_constants.txt', status='replace',        &
       action='write')
  call fork_dump_constants(u)
  close(u)
  print '(A)', 'FORK_CONSTANTS_DUMPED'
end program dump_fork_constants
