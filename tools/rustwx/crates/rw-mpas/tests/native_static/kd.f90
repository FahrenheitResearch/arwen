! Calls pristine MPAS: mpas_kd_tree.F:64-159,226-332.
program native_static_kd
 use mpas_kd_tree
 implicit none
 type(mpas_kd_type), allocatable,target::p(:)
 type(mpas_kd_type), pointer::tree,res
 integer::n,i,q,id,bits
 real::x(3),d,limit
 do n=1,37
  allocate(p(n))
  do i=1,n
   allocate(p(i)%point(3))
   p(i)%point=[real(modulo(i*17,23)-11)*0.125,real(modulo(i*11,19)-9)*0.25,real(modulo(i*13,29)-14)*0.125]
   p(i)%id=i
  end do
  tree=>mpas_kd_construct(p,3)
  do q=0,511
   x=[real(modulo(q*71,41)-20)*0.125,real(modulo(q*37,43)-21)*0.125,real(modulo(q*17,47)-23)*0.125]
   select case(modulo(q,4))
   case(0)
    limit=0.
   case(1)
    limit=0.5
   case(2)
    limit=8.
   case(3)
    limit=huge(limit)
   end select
   d=-1
   if(modulo(q,4)==3)then
    call mpas_kd_search(tree,x,res,d)
   else
    call mpas_kd_search(tree,x,res,d,limit)
   endif
   id=-1
   if(associated(res)) id=res%id
   bits=transfer(d,bits)
   write(*,'(3(i0,1x),z8.8)') n,q,id,bits
  end do
  ! mpas_kd_free owns point memory; construction rearranges p.
  call mpas_kd_free(tree)
  deallocate(p)
 end do
end program
