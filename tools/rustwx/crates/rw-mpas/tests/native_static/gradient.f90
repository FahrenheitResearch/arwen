! Pristine atm_initialize_deformation_weights, mpas_atm_advection.F:745-925.
! Planar inputs exercise its common tangent-plane integration/division block,
! including areas below the retired cutoff, underflow, and zero area.
program gradient
 use atm_advection
 implicit none
 integer::q,i
 real::scale,x(6),y(6)
 type(mpas_pool_type)::mesh
 allocate(xCell(1),yCell(1),zCell(1),xVertex(6),yVertex(6),zVertex(6))
 allocate(gradient_x(6,1),gradient_y(6,1),counts(1),coc(6,1),coe(2,6),eoc(6,1),voc(6,1))
 counts=6;coc=1;coe=1;eoc(:,1)=[1,2,3,4,5,6];voc(:,1)=[1,2,3,4,5,6]
 zCell=0.;zVertex=0.
 do q=0,31
  select case(mod(q,4))
  case(0);scale=1.
  case(1);scale=1.e-11
  case(2);scale=1.e-23
  case(3);scale=0.
  end select
  x=[0.,2.,3.,2.,0.,-1.]*scale
  y=[0.,0.,1.,3.,2.,1.]*scale
  if(mod(q,8)>=4)y=0.
  xCell(1)=x(1);yCell(1)=y(1)
  xVertex(1:5)=x(2:6);yVertex(1:5)=y(2:6)
  xVertex(6)=0.;yVertex(6)=0.
  call atm_initialize_deformation_weights(mesh,1,.false.,1.)
  do i=1,6
   write(*,'(2(I4,1X),2(Z8.8,1X))')q,i,transfer(gradient_x(i,1),0),transfer(gradient_y(i,1),0)
  enddo
 enddo
end program
