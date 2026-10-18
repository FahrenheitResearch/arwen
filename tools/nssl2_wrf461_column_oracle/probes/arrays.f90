  integer :: il, j
  character(len=24) :: nm
  call probe_init()
  do il = lc, lhab
    write(nm,'(a,i0)') 'xdnmx_', il; call pr(trim(nm), xdnmx(il))
    write(nm,'(a,i0)') 'xdnmn_', il; call pr(trim(nm), xdnmn(il))
    write(nm,'(a,i0)') 'xdn0_', il; call pr(trim(nm), xdn0(il))
    write(nm,'(a,i0)') 'cdx_', il; call pr(trim(nm), cdx(il))
    write(nm,'(a,i0)') 'cno_', il; call pr(trim(nm), cno(il))
    write(nm,'(a,i0)') 'qxmin_', il; call pr(trim(nm), qxmin(il))
    write(nm,'(a,i0)') 'xvmn_', il; call pr(trim(nm), xvmn(il))
    write(nm,'(a,i0)') 'xvmx_', il; call pr(trim(nm), xvmx(il))
    write(nm,'(a,i0)') 'da0_', il; call pr(trim(nm), da0(il))
    write(nm,'(a,i0)') 'da1_', il; call pr(trim(nm), da1(il))
    write(nm,'(a,i0)') 'xnu_', il; call pr(trim(nm), xnu(il))
    write(nm,'(a,i0)') 'xmu_', il; call pr(trim(nm), xmu(il))
    write(nm,'(a,i0)') 'dnu_', il; call pr(trim(nm), dnu(il))
    write(nm,'(a,i0)') 'bx_', il; call pr(trim(nm), bx(il))
    write(nm,'(a,i0)') 'ax_', il; call pr(trim(nm), ax(il))
    write(nm,'(a,i0)') 'bb_', il; call pr(trim(nm), bb(il))
    write(nm,'(a,i0)') 'lvol_', il; call pri(trim(nm), lvol(il))
    write(nm,'(a,i0)') 'ipc_', il; call pri(trim(nm), ipc(il))
    write(nm,'(a,i0)') 'ln_', il; call pri(trim(nm), ln(il))
    write(nm,'(a,i0)') 'lz_', il; call pri(trim(nm), lz(il))
    do j = lc, lhab
      write(nm,'(a,i0,a,i0)') 'dab0_', il, '_', j; call pr(trim(nm), dab0(il,j))
      write(nm,'(a,i0,a,i0)') 'dab1_', il, '_', j; call pr(trim(nm), dab1(il,j))
    end do
  end do
  call pr("cnu", cnu)
