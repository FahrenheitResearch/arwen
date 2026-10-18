  integer :: i
  character(len=24) :: nm
  call probe_init()
  do i = 0, 400
    write(nm,'(a,i0)') 'g4_', i; call prd(trim(nm), gamxinflu(i,0,4,1))
    write(nm,'(a,i0)') 'g1_', i; call prd(trim(nm), gamxinflu(i,0,1,1))
    write(nm,'(a,i0)') 'cr_', i; call pr(trim(nm), ciacrratio(i,0))
    write(nm,'(a,i0)') 'qr_', i; call pr(trim(nm), qiacrratio(i,0))
  end do
  do i = 0, 400
    write(nm,'(a,i0)') 'g4b_', i; call prd(trim(nm), gamxinflu(i,1,4,1))
    write(nm,'(a,i0)') 'g1b_', i; call prd(trim(nm), gamxinflu(i,1,1,1))
    write(nm,'(a,i0)') 'crb_', i; call pr(trim(nm), ciacrratio(i,1))
    write(nm,'(a,i0)') 'qrb_', i; call pr(trim(nm), qiacrratio(i,1))
  end do
  do i = lc, lhab
    write(nm,'(a,i0)') 'da0_', i; call pr(trim(nm), da0(i))
  end do
  call pr("c1sw", c1sw); call pr("ventrn", ventrn); call pr("ventc", ventc); call pr("ventr", ventr)
