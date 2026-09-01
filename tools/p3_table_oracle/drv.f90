program p3_table_oracle
  use microphy_p3, only: p3_init, p3_oracle_dump
  implicit none
  character(len=1024) :: tabdir, outdir
  integer :: nCat, stat, nargs
  logical :: trplMomI
  nargs = command_argument_count()
  if (nargs < 3) then
     print*, "usage: drv <tabledir> <outdir> <nCat>"
     stop 2
  end if
  call get_command_argument(1, tabdir)
  call get_command_argument(2, outdir)
  block
    character(len=32) :: s
    call get_command_argument(3, s)
    read(s,*) nCat
  end block
  trplMomI = .false.
  stat = -999
  print*, "ORACLE: calling p3_init(nCat=", nCat, ", trplMomI=F, model=WRF)"
  call p3_init(trim(tabdir), nCat, trplMomI, "WRF", stat, .false.)
  print*, "ORACLE: p3_init returned stat=", stat
  call p3_oracle_dump(trim(outdir))
  print*, "ORACLE: dump written"
end program p3_table_oracle
