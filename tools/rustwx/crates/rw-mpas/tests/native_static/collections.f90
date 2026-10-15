! Complete pristine container API replay, not an extracted implementation.
! MPAS: mpas_init_atm_bitarray.F:30-213; mpas_init_atm_queue.F:44-245;
! mpas_stack.F:121-258. Empty queue result words are undefined and not printed.
program static_collections
 use init_atm_bitarray
 use init_atm_queue
 use mpas_stack
 implicit none
 type(bitarray_type)::a,b,c,unallocated
 type(queue_type)::q
 type(q_data)::item,peeked,removed
 type,extends(mpas_stack_payload_type)::payload
   integer::value
 end type
 type(payload),target::shared(8)
 type(payload),pointer::owned
 type(mpas_stack_type),pointer::stack=>null()
 class(mpas_stack_payload_type),pointer::top
 integer::nx,ny,i,j,k,n,ix,iy,bits
 integer,parameter::sizes(8)=[0,1,31,32,33,63,64,97]
 do n=1,size(sizes)
  nx=sizes(n)
  do ny=0,4
   call bitarray_create(a,nx,ny)
   call dump_bits(a)
   do j=1,ny
    do i=1,nx
     if(mod(i*7+j*3,5)<2)call bitarray_set(a,i,j)
    enddo
   enddo
   call bitarray_copy(a,b)
   do j=1,ny
    do i=1,nx
     if(mod(i*3+j*7,7)==0)call bitarray_clear(a,i,j)
    enddo
   enddo
   call dump_bits(a)
   call dump_bits(b)
   call bitarray_create(c,nx,ny)
   do j=1,ny
    do i=1,nx
     if(mod(i+j,3)==0)call bitarray_set(c,i,j)
    enddo
   enddo
   call bitarray_merge(a,c)
   call dump_bits(a)
   do j=1,ny
    do i=1,nx
     call word(merge(1,0,bitarray_test(a,i,j)))
    enddo
   enddo
   call bitarray_destroy(a)
   call bitarray_destroy(b)
   call bitarray_destroy(c)
  enddo
 enddo
 ! Native warning branch on an explicitly unassociated storage pointer.
 nullify(unallocated%iarray)
 call bitarray_destroy(unallocated)
 do n=0,64
  call q_init(q)
  call word(merge(1,0,q_isdata(q)))
  call word(q_length(q))
  do i=1,n
   item%x=i*17-99
   item%y=mod(i*23,71)-35
   bits=int(z'80000000')+i
   item%lat=transfer(bits,item%lat)
   bits=int(z'00000001')+i*12345
   item%lon=transfer(bits,item%lon)
   call q_insert(q,item)
   peeked=q_peek(q)
   call dump_item(peeked)
   call word(q_length(q))
  enddo
  do i=1,n/2
   removed=q_remove(q)
   call dump_item(removed)
   call word(q_length(q))
   call word(merge(1,0,q_isdata(q)))
  enddo
  call q_destroy(q)
  ! Destroy leaves native head/tail dangling; reinitialize before reuse.
  call q_init(q)
  call word(q_length(q))
  do i=1,n
   item=q_data(real(i),real(-i),i,-i)
   call q_insert(q,item)
  enddo
  do while(q_isdata(q))
   removed=q_remove(q)
   call dump_item(removed)
  enddo
  call word(merge(1,0,q_isdata(q)))
  ! Exercise error logging, without treating undefined returned words as proof.
  peeked=q_peek(q)
  removed=q_remove(q)
  call q_destroy(q)
 enddo
 do n=0,64
  call word(merge(1,0,mpas_stack_is_empty(stack)))
  top=>mpas_stack_pop(stack)
  call word(merge(1,0,associated(top)))
  do i=1,8
   shared(i)%value=i*11
  enddo
  do i=1,n
   k=mod(i-1,8)+1
   stack=>mpas_stack_push(stack,shared(k))
  enddo
  do i=1,8
   shared(i)%value=shared(i)%value+1000
  enddo
  call word(merge(1,0,mpas_stack_is_empty(stack)))
  do i=1,n/2
   top=>mpas_stack_pop(stack)
   select type(top)
   type is(payload)
    call word(top%value)
   class default
    stop 1
   end select
  enddo
  call mpas_stack_free(stack,.false.)
  call word(merge(1,0,mpas_stack_is_empty(stack)))
  do i=1,8
   call word(shared(i)%value)
  enddo
  do i=1,n
   allocate(owned)
   owned%value=i
   stack=>mpas_stack_push(stack,owned)
  enddo
  call mpas_stack_free(stack)
  call word(merge(1,0,mpas_stack_is_empty(stack)))
  do i=1,n
   allocate(owned)
   owned%value=i
   stack=>mpas_stack_push(stack,owned)
  enddo
  call mpas_stack_free(stack,.true.)
  call word(merge(1,0,mpas_stack_is_empty(stack)))
 enddo
contains
 subroutine word(value)
  integer,intent(in)::value
  write(*,'(z8.8)')value
 end subroutine
 subroutine dump_item(value)
  type(q_data),intent(in)::value
  call word(transfer(value%lat,0))
  call word(transfer(value%lon,0))
  call word(value%x)
  call word(value%y)
 end subroutine
 subroutine dump_bits(value)
  type(bitarray_type),intent(in)::value
  integer::i,j
  call word(value%nx)
  call word(value%ny)
  call word(value%x_int_dim)
  call word(value%y_int_dim)
  call word(value%integer_size)
  do j=1,value%y_int_dim
   do i=1,value%x_int_dim
    call word(value%iarray(i,j))
   enddo
  enddo
 end subroutine
end program
